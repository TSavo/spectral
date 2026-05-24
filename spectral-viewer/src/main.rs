//! Live spectral viewer: a glass prism dispersing light on the GPU.
//! SPACE toggles RGB-transport vs spectral-transport (rainbow blinks in/out).
//! Drag (or arrow keys) to orbit. ESC to quit.

use std::sync::Arc;

use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::geom::ConvexSolid;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_gpu::{GpuContext, GpuTracer};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// ---------------------------------------------------------------------------
// Blit shader: reads accum storage, converts XYZ->sRGB, tonemaps.
// Fullscreen triangle — no vertex buffer needed.
// ---------------------------------------------------------------------------
const BLIT_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    var x: f32;
    var y: f32;
    switch vi {
        case 0u: { x = -1.0; y = -1.0; }
        case 1u: { x =  3.0; y = -1.0; }
        default: { x = -1.0; y =  3.0; }
    }
    var out: VertexOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv  = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

struct BlitParams {
    width:  u32,
    height: u32,
    _pad0:  u32,
    _pad1:  u32,
};

@group(0) @binding(0) var<storage, read> accum:       array<vec4<f32>>;
@group(0) @binding(1) var<uniform>       blit_params: BlitParams;

// XYZ -> linear sRGB (IEC 61966-2-1, D65)
fn xyz_to_srgb(xyz: vec3<f32>) -> vec3<f32> {
    let r =  3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z;
    let g = -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z;
    let b =  0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z;
    return vec3<f32>(r, g, b);
}

// Reinhard tonemap then sRGB gamma (1/2.2)
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let t = c / (1.0 + c);
    return pow(max(t, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let px  = u32(in.uv.x * f32(blit_params.width));
    let py  = u32(in.uv.y * f32(blit_params.height));
    let idx = py * blit_params.width + px;
    let len = arrayLength(&accum);
    if idx >= len { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let v   = accum[idx];
    var xyz = vec3<f32>(0.0);
    if v.w > 0.0 { xyz = v.xyz / v.w; }
    return vec4<f32>(tonemap(xyz_to_srgb(xyz)), 1.0);
}
"#;

// ---------------------------------------------------------------------------
// BlitParams POD — must match WGSL BlitParams exactly.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlitParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// GPU state — everything tied to a live window+surface.
// 'static because GpuContext and wgpu::Instance are Box::leaked in App.
// ---------------------------------------------------------------------------
struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Borrowed from App's leaked GpuContext
    ctx: &'static GpuContext,

    tracer: GpuTracer<'static>,
    total_samples: u32,

    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    blit_params_buf: wgpu::Buffer,
    // Keep blit_bgl alive for potential future resize rebuild
    _blit_bgl: wgpu::BindGroupLayout,

    // Camera orbit state
    yaw: f32,
    pitch: f32,
    radius: f32,
    spectral_on: bool,

    // Mouse drag
    drag_active: bool,
    last_mouse: Option<(f64, f64)>,
}

impl State {
    fn build_camera(&self) -> Camera {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let eye = Vec3::new(
            self.radius * cy * cp,
            self.radius * sp,
            self.radius * sy * cp,
        );
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        let aspect = if h > 0.0 { w / h } else { 1.0 };
        Camera::look_at(eye, Vec3::ZERO, Vec3::Y, 40.0, aspect)
    }

    fn reset_accum(&mut self) {
        self.tracer.clear_accum();
        self.total_samples = 0;
    }

    /// `instance` must be the same &'static instance used to create the surface.
    fn new(
        ctx: &'static GpuContext,
        instance: &'static wgpu::Instance,
        window: Arc<Window>,
    ) -> Self {
        let device = &ctx.device;

        // Surface<'static> via Arc<Window>; instance is already 'static.
        let surface: wgpu::Surface<'static> = instance
            .create_surface(Arc::clone(&window))
            .expect("create_surface");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Query a surface-compatible adapter for capabilities
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            }),
        )
        .expect("no surface-compatible adapter");
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats[0];

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(device, &surface_config);

        // Prism scene
        let mut scene = Scene::new();
        scene.background = 3.0;
        scene.add_solid(ConvexSolid::wedge(30.0, 1.0, 4.0), Material::Dielectric { glass: Glass::Sf11 });

        let yaw: f32 = std::f32::consts::FRAC_PI_4;
        let pitch: f32 = 0.15;
        let radius: f32 = 6.0;
        let camera = build_camera_from(yaw, pitch, radius, width as f32 / height as f32);

        let tracer = GpuTracer::new(ctx, scene, camera, width as usize, height as usize, Illuminant::D65, 0xCAFE_u32);

        // Blit params uniform
        let blit_params = BlitParams { width, height, _pad0: 0, _pad1: 0 };
        let blit_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blit_params"),
            contents: bytemuck::bytes_of(&blit_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Blit shader module
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        // Blit bind group layout
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_module,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tracer.accum_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blit_params_buf.as_entire_binding(),
                },
            ],
        });

        State {
            window,
            surface,
            surface_config,
            ctx,
            tracer,
            total_samples: 0,
            blit_pipeline,
            blit_bind_group,
            blit_params_buf,
            _blit_bgl: blit_bgl,
            yaw,
            pitch,
            radius,
            spectral_on: true,
            drag_active: false,
            last_mouse: None,
        }
    }

    fn render(&mut self) {
        const SPP_PER_FRAME: u32 = 4;

        self.tracer.accumulate(SPP_PER_FRAME);
        self.total_samples += SPP_PER_FRAME;

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(e) => {
                eprintln!("surface error: {e:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit([enc.finish()]);
        frame.present();
    }
}

fn build_camera_from(yaw: f32, pitch: f32, radius: f32, aspect: f32) -> Camera {
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let eye = Vec3::new(radius * cy * cp, radius * sp, radius * sy * cp);
    Camera::look_at(eye, Vec3::ZERO, Vec3::Y, 40.0, aspect)
}

// ---------------------------------------------------------------------------
// App — winit 0.30 ApplicationHandler
// ---------------------------------------------------------------------------
struct App {
    // Leaked so we can hold 'static references in State/GpuTracer
    instance: &'static wgpu::Instance,
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        let instance: &'static wgpu::Instance =
            Box::leak(Box::new(wgpu::Instance::default()));
        App { instance, state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = winit::window::WindowAttributes::default()
            .with_title("spectral-viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(900u32, 600u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        // Build surface early so we can pick a compatible adapter for GpuContext.
        // Surface<'static> requires a 'static handle — Arc<Window> satisfies this.
        let surface: wgpu::Surface<'static> = self.instance
            .create_surface(Arc::clone(&window))
            .expect("create_surface");

        // GpuContext with an adapter that can present to this surface.
        let ctx: &'static GpuContext = Box::leak(Box::new(
            GpuContext::new_for_surface(self.instance, &surface),
        ));

        // Drop surface here; State::new will create a fresh one from the same instance.
        drop(surface);

        println!("spectral-viewer: drag to orbit, SPACE toggles RGB/spectral, ESC to quit.");

        let state = State::new(ctx, self.instance, window);
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event: key_event, .. }
                if key_event.state == ElementState::Pressed =>
            {
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        state.spectral_on = !state.spectral_on;
                        state.tracer.set_spectral(state.spectral_on);
                        state.reset_accum();
                        let mode = if state.spectral_on {
                            "SPECTRAL (dispersion on)"
                        } else {
                            "RGB (fixed n, no rainbow)"
                        };
                        println!("toggle -> {mode}");
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        state.yaw -= 0.05;
                        let cam = state.build_camera();
                        state.tracer.set_camera(&cam);
                        state.reset_accum();
                    }
                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                        state.yaw += 0.05;
                        let cam = state.build_camera();
                        state.tracer.set_camera(&cam);
                        state.reset_accum();
                    }
                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                        state.pitch = (state.pitch + 0.05).min(1.4);
                        let cam = state.build_camera();
                        state.tracer.set_camera(&cam);
                        state.reset_accum();
                    }
                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                        state.pitch = (state.pitch - 0.05).max(-1.4);
                        let cam = state.build_camera();
                        state.tracer.set_camera(&cam);
                        state.reset_accum();
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state: btn_state, button: MouseButton::Left, .. } => {
                state.drag_active = btn_state == ElementState::Pressed;
                if !state.drag_active {
                    state.last_mouse = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } if state.drag_active => {
                let pos = (position.x, position.y);
                if let Some((lx, ly)) = state.last_mouse {
                    let dx = (pos.0 - lx) as f32;
                    let dy = (pos.1 - ly) as f32;
                    state.yaw += dx * 0.005;
                    state.pitch = (state.pitch - dy * 0.005).clamp(-1.4, 1.4);
                    let cam = state.build_camera();
                    state.tracer.set_camera(&cam);
                    state.reset_accum();
                }
                state.last_mouse = Some(pos);
            }

            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state.surface.configure(&state.ctx.device, &state.surface_config);
                // Update blit params (accum stays original size; shader guards with arrayLength)
                let bp = BlitParams { width: size.width, height: size.height, _pad0: 0, _pad1: 0 };
                state.ctx.queue.write_buffer(&state.blit_params_buf, 0, bytemuck::bytes_of(&bp));
            }

            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }

            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("EventLoop::new");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run_app");
}

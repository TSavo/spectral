//! VOL-6: live forward + volumetric "Dark Side of the Moon" viewer.
//!
//! Renders the DSOTM scene live: photon batches accumulate each frame (the fan
//! sharpens over time as noise converges, NOT brightness), an orbit camera, and
//! SPACE toggles dispersion (n(λ) rainbow fan <-> n(550) collapsed white beam).
//!
//! Reuses the proven VOL pipeline:
//!   - spectral_gpu::vol_photons::GpuPhotonFilm — the forward photon kernel + the
//!     rim surface pass (VOL-2..5, diff-gate-proven).
//!   - A GPU-RESIDENT accumulator (u32 fixed-point, same SCALE as the film) drained
//!     each frame by a small resolve compute pass (atomicExchange: read + clear in
//!     one op). No per-frame CPU readback.
//!   - A blit fragment that composites accum (normalized by total photons, scaled
//!     to a fixed reference count so brightness is frame-count-independent) + the
//!     static rim, then exposure + XYZ->linear-sRGB + Reinhard tonemap.
//!
//! Controls: drag or arrow keys to orbit, SPACE toggles dispersion, ESC quits.
//!
//! Headless capture (no window) for verification:
//!   cargo run -p spectral-viewer --release -- --capture /tmp/x.png --mode spectral --frames 400
//!   cargo run -p spectral-viewer --release -- --capture /tmp/x.png --mode rgb      --frames 400

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_gpu::vol_photons::{dsotm_scene, GpuPhotonFilm, VolWeights};
use spectral_gpu::GpuContext;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const SIZE: u32 = 700;
/// Photons traced per frame. 500K keeps a 16" MBP's Metal under the watchdog and
/// off the thermal throttle while converging quickly.
// Point splatting is cheap, so a big batch per frame converges in a handful of
// frames; the screen-space neighbor kernel makes even the first frame read smooth.
const PHOTONS_PER_FRAME: u32 = 200_000;
/// Fixed reference photon count: the volumetric term is normalized by the actual
/// total photons emitted, then multiplied by TARGET_N so the fan reads at a stable
/// brightness regardless of how many frames have accumulated (converge in noise,
/// not brightness).
const TARGET_N: f32 = 4_000_000.0;
const MAX_DIST: f32 = 14.0;
const SCALE: f32 = 8388608.0; // 2^23, must match volume.wgsl / vol_photons.rs

// ---------------------------------------------------------------------------
// Resolve compute shader: drain the atomic film into the persistent u32 accum.
//   v = atomicExchange(&film[i], 0);  accum[i] += v;
// One thread per channel (3*w*h). atomicExchange reads AND clears in one op, so
// there is no race with the next frame's photon dispatch.
// ---------------------------------------------------------------------------
const RESOLVE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> film:  array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> accum: array<u32>;

// 2D dispatch: linear index = (gid.y * groups_x + workgroup-local) — but simplest
// is i = gid.y * (num_groups_x * 64) + gid.x. We instead pass the row stride via
// the dispatch shape: gid.x spans [0, groups_x*64), gid.y spans rows. The host
// keeps groups_x <= 65535. Linear i = gid.y * stride_x + gid.x, stride_x in a const
// is awkward, so we recompute from the global grid width via gid + dispatch dims.
@compute @workgroup_size(64)
fn resolve_main(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(num_workgroups) ng: vec3<u32>) {
    // Flatten the 2D dispatch: each row y contributes (ng.x * 64) lanes.
    let lanes_per_row = ng.x * 64u;
    let i = gid.y * lanes_per_row + gid.x;
    if i >= arrayLength(&accum) { return; }
    let v = atomicExchange(&film[i], 0u);
    accum[i] = accum[i] + v;
}
"#;

// ---------------------------------------------------------------------------
// Blit shader: composite (volumetric accum + static rim), expose, tonemap.
// Fullscreen triangle.
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

struct ViewerParams {
    width:         u32,
    height:        u32,
    total_photons: u32,
    _pad0:         u32,
    target_n:      f32,
    exposure:      f32,
    scale:         f32,
    kradius:       f32,   // screen-space reconstruction-kernel radius (px)
};

@group(0) @binding(0) var<storage, read> accum:   array<u32>;   // 3*w*h fixed-point
@group(0) @binding(1) var<storage, read> surface: array<f32>;   // 3*w*h rim XYZ
@group(0) @binding(2) var<uniform>       params:  ViewerParams;

// XYZ -> linear sRGB (IEC 61966-2-1, D65) — same matrix as spectral_core.
fn xyz_to_srgb(xyz: vec3<f32>) -> vec3<f32> {
    let r =  3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z;
    let g = -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z;
    let b =  0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z;
    return vec3<f32>(r, g, b);
}

// Reinhard tonemap then sRGB gamma (1/2.2).
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let t = c / (1.0 + c);
    return pow(max(t, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let w = params.width;
    let h = params.height;
    let px = min(u32(in.uv.x * f32(w)), w - 1u);
    let py = min(u32(in.uv.y * f32(h)), h - 1u);
    let idx = py * w + px;

    // Volumetric: fixed-point accum / SCALE / total_photons * TARGET_N, gathered
    // over a screen-space neighbor kernel (the "kernel for neighboring photons" —
    // a density estimate that makes sparse photon hits read as smooth volumetric
    // radiance and converges to the unfiltered result as photons accumulate).
    let inv_n = 1.0 / max(f32(params.total_photons), 1.0);
    let k = (1.0 / params.scale) * inv_n * params.target_n;
    let radius = i32(params.kradius);
    let sigma = max(params.kradius * 0.5, 0.5);
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sx = i32(px) + dx;
            let sy = i32(py) + dy;
            if sx >= 0 && sx < i32(w) && sy >= 0 && sy < i32(h) {
                let sidx = u32(sy) * w + u32(sx);
                let g = exp(-f32(dx * dx + dy * dy) * inv2s2);
                acc = acc + vec3<f32>(
                    f32(accum[3u * sidx + 0u]),
                    f32(accum[3u * sidx + 1u]),
                    f32(accum[3u * sidx + 2u]),
                ) * g;
                wsum = wsum + g;
            }
        }
    }
    // Normalized weighted mean (energy-preserving reconstruction, no brightening).
    let vol = acc * (k / max(wsum, 1e-6));
    let surf = vec3<f32>(surface[3u * idx + 0u], surface[3u * idx + 1u], surface[3u * idx + 2u]);

    let composite = (vol + surf) * params.exposure;
    return vec4<f32>(tonemap(xyz_to_srgb(composite)), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewerParams {
    width: u32,
    height: u32,
    total_photons: u32,
    _pad0: u32,
    target_n: f32,
    exposure: f32,
    scale: f32,
    kradius: f32,
}

// ---------------------------------------------------------------------------
// Renderer: the GPU-resident accum + resolve + blit, plus the photon film.
// Used by BOTH the live window and the headless capture path (same pipeline).
// ---------------------------------------------------------------------------
struct Renderer<'ctx> {
    ctx: &'ctx GpuContext,
    width: usize,
    height: usize,

    film: GpuPhotonFilm<'ctx>,
    accum_buf: wgpu::Buffer,   // 3*w*h u32 fixed-point, GPU-resident
    surface_buf: wgpu::Buffer, // 3*w*h f32 rim, uploaded from film.surface()

    resolve_pipeline: wgpu::ComputePipeline,
    resolve_bind_group: wgpu::BindGroup,

    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,

    params: ViewerParams,
    total_photons: u32,
    frame_seed: u32,
}

impl<'ctx> Renderer<'ctx> {
    fn new(ctx: &'ctx GpuContext, width: usize, height: usize, target_format: wgpu::TextureFormat) -> Self {
        let device = &ctx.device;
        let (scene, beam, cam) = dsotm_scene(width as f32 / height as f32);

        // The forward photon film (owns the kernel + film buffer + rim pass).
        let mut film = GpuPhotonFilm::new(
            ctx, scene, &cam, &beam, width, height,
            PHOTONS_PER_FRAME, 0, MAX_DIST, VolWeights::default(), None,
        );
        // Static rim, computed once for the initial camera.
        film.recompute_surface(&cam);

        let n_ch = (3 * width * height) as u64;

        // GPU-resident accumulator (u32 fixed-point), starts zeroed.
        let accum_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer_accum"),
            size: n_ch * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // GPU surface buffer, uploaded from the rim pass.
        let surface_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer_surface"),
            contents: bytemuck::cast_slice(film.surface()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Resolve pipeline.
        let resolve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resolve"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve_bgl"),
            entries: &[
                storage_entry(0, false), // film read_write (atomicExchange)
                storage_entry(1, false), // accum read_write
            ],
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resolve_layout"),
            bind_group_layouts: &[&resolve_bgl],
            push_constant_ranges: &[],
        });
        let resolve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resolve_pipeline"),
            layout: Some(&resolve_layout),
            module: &resolve_module,
            entry_point: "resolve_main",
            cache: None,
            compilation_options: Default::default(),
        });
        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resolve_bg"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: film.film_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buf.as_entire_binding() },
            ],
        });

        // Viewer params uniform.
        let params = ViewerParams {
            width: width as u32,
            height: height as u32,
            total_photons: 0,
            _pad0: 0,
            target_n: TARGET_N,
            exposure: 1.0,
            scale: SCALE,
            kradius: 2.0, // screen-space neighbor kernel: 5x5 gather
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewer_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Blit pipeline.
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                storage_entry_frag(0, true), // accum read
                storage_entry_frag(1, true), // surface read
                uniform_entry_frag(2),       // params
            ],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&blit_layout),
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
                    format: target_format,
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
                wgpu::BindGroupEntry { binding: 0, resource: accum_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: surface_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
            ],
        });

        Renderer {
            ctx,
            width,
            height,
            film,
            accum_buf,
            surface_buf,
            resolve_pipeline,
            resolve_bind_group,
            blit_pipeline,
            blit_bind_group,
            params_buf,
            params,
            total_photons: 0,
            frame_seed: 0,
        }
    }

    /// Zero the GPU accum and the photon counter (on camera move / SPACE toggle).
    fn reset_accum(&mut self) {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.accum_buf, 0, None);
        self.ctx.queue.submit([enc.finish()]);
        self.total_photons = 0;
        self.frame_seed = 0;
        self.params.total_photons = 0;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Recompute the static rim for `cam`, re-upload it to the GPU surface buffer,
    /// and reset the volumetric accum (camera changed).
    fn set_camera(&mut self, cam: &Camera) {
        self.film.recompute_surface(cam);
        self.ctx.queue.write_buffer(&self.surface_buf, 0, bytemuck::cast_slice(self.film.surface()));
        self.reset_accum();
    }

    fn set_spectral(&mut self, on: bool) {
        self.film.set_spectral(on);
        self.reset_accum();
    }

    /// One frame of accumulation: dispatch a photon batch into the film, resolve
    /// (drain film -> accum + clear), bump the photon counter + params.
    fn accumulate_frame(&mut self) {
        // A fresh batch with a frame-incremented seed: photon_base=0,
        // n_photons=PHOTONS_PER_FRAME runs indices 0..PHOTONS_PER_FRAME with this seed.
        let seed = 0xD50_u32.wrapping_add(self.frame_seed);
        self.film.dispatch_photons(0, PHOTONS_PER_FRAME, seed);

        // Resolve: film -> accum, clear film (atomicExchange).
        // 2D dispatch so neither dimension exceeds max_compute_workgroups_per_dimension
        // (65535). At a Retina-scaled window (e.g. 1400x1400) the channel count is
        // ~5.9M, whose 1D group count would overflow; tiling into rows fixes it.
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &self.resolve_bind_group, &[]);
            let (gx, gy) = self.resolve_grid();
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.ctx.queue.submit([enc.finish()]);

        self.total_photons = self.total_photons.saturating_add(PHOTONS_PER_FRAME);
        self.frame_seed = self.frame_seed.wrapping_add(1);
        self.params.total_photons = self.total_photons;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Blit the composited image into `view`.
    fn blit_into(&self, view: &wgpu::TextureView) {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
    }

    fn set_exposure(&mut self, exposure: f32) {
        self.params.exposure = exposure;
        self.ctx.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// 2D workgroup grid for the resolve pass, both dims <= 65535. The shader
    /// reconstructs the linear channel index as gid.y * (gx*64) + gid.x. gx is
    /// fixed small enough that gx*64 lanes/row covers width comfortably; gy tiles
    /// the rest. Slight over-dispatch is harmless (the shader bounds-checks).
    fn resolve_grid(&self) -> (u32, u32) {
        let n_ch = (3 * self.width * self.height) as u32;
        let gx: u32 = 1024; // 1024*64 = 65536 lanes per row
        let lanes_per_row = gx * 64;
        let gy = n_ch.div_ceil(lanes_per_row);
        (gx, gy)
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_entry_frag(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn uniform_entry_frag(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// Orbit camera: yaw/pitch/radius around the prism, matched to the dsotm framing.
// yaw=pitch=0 reproduces the dsotm face-on camera (eye on +Z looking at target).
// ---------------------------------------------------------------------------
const TARGET: Vec3 = Vec3::new(0.5, -0.5, 0.0);
const RADIUS0: f32 = 12.0; // |dsotm eye - target| = |(0,0,12)| = 12
const VFOV: f32 = 42.0;

fn orbit_camera(yaw: f32, pitch: f32, radius: f32, aspect: f32) -> Camera {
    // yaw=pitch=0 -> eye on +Z (face-on, matching the dsotm camera at (0.5,-0.5,12)).
    // Offset = R*(cos p sin y, sin p, cos p cos y).
    let cp = pitch.cos();
    let eye = TARGET + Vec3::new(radius * cp * yaw.sin(), radius * pitch.sin(), radius * cp * yaw.cos());
    Camera::look_at(eye, TARGET, Vec3::Y, VFOV, aspect)
}

// ---------------------------------------------------------------------------
// Headless capture: run the EXACT live pipeline for `frames` frames into an
// offscreen texture, then read back RGBA8 and write a PNG. No window/surface.
// ---------------------------------------------------------------------------
fn run_capture(path: &str, spectral: bool, frames: u32) {
    let ctx: &'static GpuContext = Box::leak(Box::new(
        GpuContext::new().expect("no GPU adapter for capture"),
    ));
    let (w, h) = (SIZE as usize, SIZE as usize);
    let format = wgpu::TextureFormat::Rgba8Unorm;

    let mut r = Renderer::new(ctx, w, h, format);
    // Default orbit camera (matches the window's initial framing) + dispersion mode.
    let cam = orbit_camera(0.0, 0.0, RADIUS0, w as f32 / h as f32);
    r.set_camera(&cam);
    r.set_spectral(spectral); // resets accum

    for _ in 0..frames {
        r.accumulate_frame();
    }
    ctx.device.poll(wgpu::Maintain::Wait);

    // Exposure: 99.7th-percentile composited luminance (matches prism_dsotm.rs).
    let exposure = compute_exposure_p997(&r);
    r.set_exposure(exposure);
    eprintln!(
        "capture: mode={} frames={frames} total_photons={} exposure={exposure:.4}",
        if spectral { "spectral" } else { "rgb" }, r.total_photons
    );

    // Offscreen target.
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture_target"),
        size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    r.blit_into(&view);

    // Copy texture -> buffer (256-byte row alignment).
    let bytes_per_pixel = 4u32;
    let unpadded = w as u32 * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture_readback"),
        size: (padded * h as u32) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &out_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h as u32),
            },
        },
        wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = out_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();

    // De-pad rows into a tight RGBA8 buffer.
    let mut rgba = vec![0u8; w * h * 4];
    for row in 0..h {
        let src = row * padded as usize;
        let dst = row * w * 4;
        rgba[dst..dst + w * 4].copy_from_slice(&data[src..src + w * 4]);
    }
    drop(data);
    out_buf.unmap();

    let img: image::RgbaImage = image::ImageBuffer::from_raw(w as u32, h as u32, rgba)
        .expect("rgba buffer size mismatch");
    img.save(path).unwrap_or_else(|e| panic!("save {path}: {e}"));
    let meta = std::fs::metadata(path).expect("PNG missing after save");
    println!("wrote {path} ({}x{}, {} bytes, {} photons)", w, h, meta.len(), r.total_photons);
}

/// Read back the accum and compute exposure = 1 / p99.7 of the composited
/// luminance (same exposure rule as prism_dsotm.rs).
fn compute_exposure_p997(r: &Renderer) -> f32 {
    let (w, h) = (r.width, r.height);
    let n_ch = 3 * w * h;

    let staging = r.ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("accum_readback"),
        size: (n_ch * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = r.ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&r.accum_buf, 0, &staging, 0, (n_ch * 4) as u64);
    r.ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    r.ctx.device.poll(wgpu::Maintain::Wait);
    let accum: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    let surface = r.film.surface();
    let inv_n = 1.0 / (r.total_photons.max(1) as f32);
    let k = (1.0 / SCALE) * inv_n * TARGET_N;
    let mut lums: Vec<f32> = (0..w * h)
        .map(|i| accum[3 * i + 1] as f32 * k + surface[3 * i + 1])
        .collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lums[(((w * h) as f32 * 0.997) as usize).min(w * h - 1)].max(1e-9);
    1.0 / p
}

// ---------------------------------------------------------------------------
// Live window App (winit 0.30 ApplicationHandler).
// ---------------------------------------------------------------------------
struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    ctx: &'static GpuContext,
    renderer: Renderer<'static>,

    yaw: f32,
    pitch: f32,
    radius: f32,
    drag_active: bool,
    last_mouse: Option<(f64, f64)>,
    frames: u32,
}

impl State {
    fn new(ctx: &'static GpuContext, instance: &'static wgpu::Instance, window: Arc<Window>) -> Self {
        let device = &ctx.device;
        let surface: wgpu::Surface<'static> =
            instance.create_surface(Arc::clone(&window)).expect("create_surface");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
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

        let mut renderer = Renderer::new(ctx, width as usize, height as usize, fmt);

        let (yaw, pitch, radius) = (0.0f32, 0.0f32, RADIUS0);
        let cam = orbit_camera(yaw, pitch, radius, width as f32 / height as f32);
        renderer.set_camera(&cam);
        renderer.set_exposure(2.5); // reasonable start; refined by rolling readback.

        State {
            window,
            surface,
            surface_config,
            ctx,
            renderer,
            yaw,
            pitch,
            radius,
            drag_active: false,
            last_mouse: None,
            frames: 0,
        }
    }

    fn rebuild_camera(&mut self) {
        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        let cam = orbit_camera(self.yaw, self.pitch, self.radius, aspect);
        self.renderer.set_camera(&cam);
    }

    fn render(&mut self) {
        // DIAGNOSTIC: dispatch the photon batch + resolve, then BLOCK on the GPU so
        // `accum_ms` is the real GPU frame time (not just CPU submission). A single
        // dispatch over ~5s is killed by the macOS GPU watchdog with no Rust-level
        // panic (process just vanishes), so the per-frame timing + the last logged
        // frame number are how we catch it.
        let t0 = std::time::Instant::now();
        self.renderer.accumulate_frame();
        self.ctx.device.poll(wgpu::Maintain::Wait);
        let accum_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.frames += 1;

        // Rolling exposure: refresh every 30 frames (cheap readback at this res).
        if self.frames.is_multiple_of(30) {
            let exposure = compute_exposure_p997(&self.renderer);
            self.renderer.set_exposure(exposure);
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[frame {}] surface error: {e:?} — reconfiguring", self.frames);
                self.surface.configure(&self.ctx.device, &self.surface_config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.renderer.blit_into(&view);
        frame.present();

        if self.frames <= 60 || self.frames.is_multiple_of(20) {
            eprintln!(
                "[frame {}] accumulate+resolve={:.1}ms  photons≈{}",
                self.frames,
                accum_ms,
                self.frames as u64 * PHOTONS_PER_FRAME as u64
            );
        }
        if accum_ms > 500.0 {
            eprintln!(
                "[frame {}] WARNING: GPU frame {:.0}ms — nearing the ~5s watchdog; a heavier \
                 frame (e.g. long beams after an orbit) may SIGKILL the process with no error",
                self.frames, accum_ms
            );
        }
    }
}

struct App {
    instance: &'static wgpu::Instance,
    state: Option<State>,
}

impl App {
    fn new() -> Self {
        let instance: &'static wgpu::Instance = Box::leak(Box::new(wgpu::Instance::default()));
        App { instance, state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = winit::window::WindowAttributes::default()
            .with_title("spectral-viewer — DSOTM (SPACE: dispersion, drag/arrows: orbit, ESC: quit)")
            .with_inner_size(winit::dpi::LogicalSize::new(SIZE, SIZE))
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        let surface: wgpu::Surface<'static> =
            self.instance.create_surface(Arc::clone(&window)).expect("create_surface");
        let ctx: &'static GpuContext =
            Box::leak(Box::new(GpuContext::new_for_surface(self.instance, &surface)));
        drop(surface);

        // DIAGNOSTIC: log GPU validation / OOM errors that wgpu would otherwise
        // swallow (these can precede a device loss / silent process death).
        ctx.device.on_uncaptured_error(Box::new(|e| {
            eprintln!("[wgpu uncaptured error] {e}");
        }));

        println!("spectral-viewer: live forward+volumetric DSOTM.");
        println!("  drag or arrow keys to orbit, SPACE toggles dispersion (rainbow <-> white), ESC to quit.");
        println!("  starting with dispersion ON (n(lambda)).");
        eprintln!(
            "[startup] {}x{} window, {} photons/frame (beam-splat); converges over ~100+ frames",
            SIZE as u32, SIZE as u32, PHOTONS_PER_FRAME
        );

        let state = State::new(ctx, self.instance, window);
        self.state = Some(state);

        // FIX for the old blank viewer: kick the first redraw AND poll continuously.
        event_loop.set_control_flow(ControlFlow::Poll);
        self.state.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => {
                eprintln!("[exit] window CloseRequested at frame {}", state.frames);
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        eprintln!("[exit] Escape at frame {}", state.frames);
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::Space) => {
                        let on = !state.renderer.film.spectral();
                        state.renderer.set_spectral(on);
                        println!(
                            "dispersion: {}",
                            if on { "ON  (n(lambda) — rainbow fan)" } else { "OFF (n(550) — collapsed white beam)" }
                        );
                    }
                    PhysicalKey::Code(KeyCode::ArrowLeft)  => { state.yaw -= 0.08; state.rebuild_camera(); }
                    PhysicalKey::Code(KeyCode::ArrowRight) => { state.yaw += 0.08; state.rebuild_camera(); }
                    PhysicalKey::Code(KeyCode::ArrowUp)    => { state.pitch = (state.pitch + 0.06).clamp(-1.4, 1.4); state.rebuild_camera(); }
                    PhysicalKey::Code(KeyCode::ArrowDown)  => { state.pitch = (state.pitch - 0.06).clamp(-1.4, 1.4); state.rebuild_camera(); }
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
                if let Some((lx, ly)) = state.last_mouse {
                    let dx = (position.x - lx) as f32;
                    let dy = (position.y - ly) as f32;
                    state.yaw += dx * 0.006;
                    state.pitch = (state.pitch - dy * 0.006).clamp(-1.4, 1.4);
                    state.rebuild_camera();
                }
                state.last_mouse = Some((position.x, position.y));
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
    let args: Vec<String> = std::env::args().collect();

    // Capture mode: --capture <path> --mode <spectral|rgb> --frames <N>
    if let Some(pos) = args.iter().position(|a| a == "--capture") {
        let path = args.get(pos + 1).cloned().unwrap_or_else(|| "/tmp/vol6_capture.png".into());
        let mode = arg_value(&args, "--mode").unwrap_or_else(|| "spectral".into());
        let frames: u32 = arg_value(&args, "--frames").and_then(|s| s.parse().ok()).unwrap_or(400);
        let spectral = match mode.as_str() {
            "spectral" => true,
            "rgb" => false,
            other => {
                eprintln!("unknown --mode '{other}' (use spectral|rgb)");
                std::process::exit(2);
            }
        };
        run_capture(&path, spectral, frames);
        return;
    }

    let event_loop = EventLoop::new().expect("EventLoop::new");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run_app");
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|p| args.get(p + 1).cloned())
}

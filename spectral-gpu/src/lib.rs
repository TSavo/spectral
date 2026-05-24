//! GPU mirror of the CPU spectral path tracer.

pub mod data;
pub mod film;
pub mod upload;
pub mod vol_photons;

use bytemuck::{Pod, Zeroable};
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::material::Material;
use spectral_core::scene::{Scene, Shape};
use spectral_core::sellmeier::Glass;
use spectral_core::Xyz;
use wgpu::util::DeviceExt;

use crate::data::sample_tables;
use crate::upload::{GpuMaterial, GpuPlane, GpuPrimitive};

/// A headless wgpu device + queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Request a headless adapter/device. Returns None if no GPU is available
    /// (so tests skip cleanly rather than fail on a headless CI box).
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .ok()?;
            Some(GpuContext { device, queue })
        })
    }

    /// Create a GpuContext using an existing instance and a surface for
    /// adapter compatibility (so the adapter supports presentation on that surface).
    /// Panics if no adapter or device is available.
    pub fn new_for_surface(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Self {
        pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(surface),
                    ..Default::default()
                })
                .await
                .expect("no wgpu adapter compatible with surface");
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("request_device failed");
            GpuContext { device, queue }
        })
    }
}

// ---------------------------------------------------------------------------
// Params uniform — must match the WGSL Params struct byte-for-byte.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cam_origin: [f32; 4],
    cam_lower_left: [f32; 4],
    cam_horizontal: [f32; 4],
    cam_vertical: [f32; 4],
    width: u32,
    height: u32,
    spp: u32,
    seed: u32,
    n_primitives: u32,
    illuminant: u32,
    background: f32,
    spectral: u32, // 1=spectral dispersion, 0=fixed n(550nm)
}

// ---------------------------------------------------------------------------
// Scene flatten
// ---------------------------------------------------------------------------

/// Flatten a Scene into GPU-friendly flat arrays.
/// Returns (primitives, planes, materials) with at least one element each
/// (wgpu storage buffers cannot be zero-sized).
fn flatten_scene(scene: &Scene) -> (Vec<GpuPrimitive>, Vec<GpuPlane>, Vec<GpuMaterial>) {
    let mut prims: Vec<GpuPrimitive> = Vec::new();
    let mut planes: Vec<GpuPlane> = Vec::new();
    let mut mats: Vec<GpuMaterial> = Vec::new();

    for p in &scene.primitives {
        let mat_idx = mats.len() as u32;
        match p.material {
            Material::Lambertian { reflectance } => {
                mats.push(GpuMaterial { reflectance, glass: 0, kind: 0, _pad: 0 });
            }
            Material::Dielectric { glass } => {
                let g = match glass {
                    Glass::Bk7 => 0u32,
                    Glass::Sf11 => 1,
                    Glass::Water => 2,
                };
                mats.push(GpuMaterial { reflectance: 0.0, glass: g, kind: 1, _pad: 0 });
            }
        }
        match &p.shape {
            Shape::Sphere(s) => prims.push(GpuPrimitive {
                center: s.center.into(),
                radius: s.radius,
                plane_start: 0,
                plane_count: 0,
                kind: 0,
                material: mat_idx,
            }),
            Shape::Solid(solid) => {
                let start = planes.len() as u32;
                for pl in &solid.planes {
                    planes.push(GpuPlane { normal: pl.normal.into(), d: pl.d });
                }
                prims.push(GpuPrimitive {
                    center: [0.0; 3],
                    radius: 0.0,
                    plane_start: start,
                    plane_count: solid.planes.len() as u32,
                    kind: 1,
                    material: mat_idx,
                });
            }
        }
    }

    // wgpu storage buffers can't be zero-sized; ensure at least one element each.
    if planes.is_empty() {
        planes.push(GpuPlane { normal: [0.0, 1.0, 0.0], d: 0.0 });
    }
    if mats.is_empty() {
        mats.push(GpuMaterial { reflectance: 0.0, glass: 0, kind: 0, _pad: 0 });
    }
    if prims.is_empty() {
        // kind=99 will never match 0 or 1 in scene_intersect so it's inert,
        // but n_primitives is set to 0 anyway so this element is never iterated.
        prims.push(GpuPrimitive {
            center: [0.0; 3],
            radius: 0.0,
            plane_start: 0,
            plane_count: 0,
            kind: 99,
            material: 0,
        });
    }

    (prims, planes, mats)
}

// ---------------------------------------------------------------------------
// GpuTracer
// ---------------------------------------------------------------------------

/// Holds the GPU pipeline and all scene/accumulation buffers.
pub struct GpuTracer<'ctx> {
    device: &'ctx wgpu::Device,
    queue: &'ctx wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    accum_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    params: Params,
    width: usize,
    height: usize,
}

impl<'ctx> GpuTracer<'ctx> {
    pub fn new(
        ctx: &'ctx GpuContext,
        scene: Scene,
        camera: Camera,
        width: usize,
        height: usize,
        illuminant: Illuminant,
        seed: u32,
    ) -> Self {
        let device = &ctx.device;
        let queue = &ctx.queue;

        // Build WGSL source: rng.wgsl + trace.wgsl concatenated.
        let shader_src =
            concat!(include_str!("shaders/rng.wgsl"), include_str!("shaders/trace.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Flatten scene.
        let n_real = scene.primitives.len() as u32;
        let (prims, plane_vec, mat_vec) = flatten_scene(&scene);

        // Upload immutable scene buffers.
        let prim_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("primitives"),
            contents: bytemuck::cast_slice(&prims),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let plane_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planes"),
            contents: bytemuck::cast_slice(&plane_vec),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: bytemuck::cast_slice(&mat_vec),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tables = sample_tables();
        let table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Accumulation buffer: vec4 per pixel, zero-initialized.
        let accum_size = (width * height * 16) as u64; // 4 f32 * 4 bytes each
        let accum_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("accum"),
            contents: &vec![0u8; accum_size as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Readback staging buffer.
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size: accum_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params uniform buffer (COPY_DST so we can write_buffer each frame).
        let (o, ll, h, v) = camera.view_basis();
        let ill_u32 = match illuminant {
            Illuminant::D65 => 0u32,
            Illuminant::A => 1,
        };
        let params = Params {
            cam_origin: [o.x, o.y, o.z, 0.0],
            cam_lower_left: [ll.x, ll.y, ll.z, 0.0],
            cam_horizontal: [h.x, h.y, h.z, 0.0],
            cam_vertical: [v.x, v.y, v.z, 0.0],
            width: width as u32,
            height: height as u32,
            spp: 0,
            seed,
            n_primitives: n_real,
            illuminant: ill_u32,
            background: scene.background,
            spectral: 1, // default: full spectral dispersion
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Build pipeline with auto layout.
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // Build bind group from the pipeline's auto-reflected layout.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prim_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: plane_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: accum_buf.as_entire_binding() },
            ],
        });

        GpuTracer {
            device,
            queue,
            pipeline,
            bind_group,
            params_buf,
            accum_buf,
            read_buf,
            params,
            width,
            height,
        }
    }

    /// Dispatch `spp` samples per pixel and accumulate into the internal buffer.
    pub fn accumulate(&mut self, spp: u32) {
        self.params.spp = spp;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let gx = (self.width as u32).div_ceil(8);
            let gy = (self.height as u32).div_ceil(8);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Copy accum buffer back to CPU and return per-pixel XYZ (divided by sample count w).
    pub fn read_xyz(&self) -> Vec<Xyz> {
        let accum_size = (self.width * self.height * 16) as u64;

        // Copy accum -> read staging.
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.accum_buf, 0, &self.read_buf, 0, accum_size);
        self.queue.submit([enc.finish()]);

        // Map and wait.
        let slice = self.read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let raw = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&raw);

        // Each pixel is 4 f32s: [x, y, z, count].
        let result = floats
            .chunks_exact(4)
            .map(|q| {
                let w = q[3];
                if w == 0.0 {
                    [0.0f32; 3]
                } else {
                    [q[0] / w, q[1] / w, q[2] / w]
                }
            })
            .collect();

        drop(raw);
        self.read_buf.unmap();

        result
    }

    /// Return a reference to the accumulation buffer (for blit pipelines).
    pub fn accum_buffer(&self) -> &wgpu::Buffer {
        &self.accum_buf
    }

    /// Return the render dimensions.
    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Toggle spectral dispersion on (1) or off (0).
    pub fn set_spectral(&mut self, on: bool) {
        self.params.spectral = if on { 1 } else { 0 };
    }

    /// Update the camera, rebuilding the params basis vectors.
    pub fn set_camera(&mut self, camera: &Camera) {
        let (o, ll, h, v) = camera.view_basis();
        self.params.cam_origin = [o.x, o.y, o.z, 0.0];
        self.params.cam_lower_left = [ll.x, ll.y, ll.z, 0.0];
        self.params.cam_horizontal = [h.x, h.y, h.z, 0.0];
        self.params.cam_vertical = [v.x, v.y, v.z, 0.0];
    }

    /// Zero the accumulation buffer and reset sample counts.
    pub fn clear_accum(&self) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.accum_buf, 0, None);
        self.queue.submit([enc.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_initializes_or_skips() {
        match GpuContext::new() {
            Some(_) => { /* GPU present (this Mac has Metal) */ }
            None => eprintln!("no GPU adapter; skipping GPU tests"),
        }
    }

    #[test]
    fn gpu_renders_nonzero_for_lit_scene() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping");
            return;
        };
        use glam::Vec3;
        use spectral_core::camera::Camera;
        use spectral_core::cie::Illuminant;
        use spectral_core::scene::Scene;
        let mut scene = Scene::new();
        scene.background = 1.0;
        let cam =
            Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let mut t = GpuTracer::new(&ctx, scene, cam, 16, 16, Illuminant::D65, 42);
        t.accumulate(4);
        let xyz = t.read_xyz();
        assert_eq!(xyz.len(), 256);
        let total: f32 = xyz.iter().map(|p| p[1]).sum();
        assert!(total > 0.0, "GPU render should have positive luminance, got {total}");
        eprintln!("total Y luminance: {total}");
    }
}

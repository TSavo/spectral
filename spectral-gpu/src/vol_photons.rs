// VOL-2: GPU forward photon kernel.
//
// Builds and dispatches the volume.wgsl compute kernel, which mirrors
// render_volumetric_scene() from spectral-core. One GPU thread per photon.
//
// Binding layout (must match volume.wgsl exactly):
//   @binding(0) uniform VolParamsGpu  -- photon params + camera + beam
//   @binding(1) storage  primitives
//   @binding(2) storage  planes
//   @binding(3) storage  materials
//   @binding(4) storage  tables (CMF + illuminants)
//   @binding(5) storage  film (atomic<u32>, 3 * width * height)

use bytemuck::{Pod, Zeroable, cast_slice};
use spectral_core::camera::Camera;
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::{Scene, Shape};
use spectral_core::sellmeier::Glass;
use wgpu::util::DeviceExt;

use crate::data::sample_tables;
use crate::upload::{GpuMaterial, GpuPlane, GpuPrimitive};
use crate::GpuContext;

/// Fixed-point scale — must match SCALE in volume.wgsl.
const SCALE: f32 = 4096.0;

// ---------------------------------------------------------------------------
// Uniform struct — must match VolParams in volume.wgsl byte-for-byte.
// All vec4 fields: padding to 16-byte alignment.
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VolParamsGpu {
    // scene + frame
    pub n_primitives: u32,
    pub n_photons: u32,
    pub seed: u32,
    pub max_dist: f32,
    // image
    pub width: u32,
    pub height: u32,
    pub debug_mode: u32,  // 0 = normal splat only; 1 = also record per-photon state
    pub debug_count: u32, // photons with idx < debug_count get their state recorded
    // beam (vec4 with xyz used, w=0)
    pub beam_corner: [f32; 4],
    pub beam_u: [f32; 4],
    pub beam_v: [f32; 4],
    pub beam_dir: [f32; 4],
    // camera projection precomputed fields
    pub cam_origin: [f32; 4],
    pub cam_u: [f32; 4], // xyz=horizontal.normalize(), w=horizontal.length()
    pub cam_v: [f32; 4], // xyz=vertical.normalize(),   w=vertical.length()
    pub cam_w: [f32; 4], // xyz=origin - horiz/2 - vert/2 - lower_left (the "back" vector)
}

/// Per-photon debug record — must match DebugPhoton in volume.wgsl byte-for-byte.
/// 304 bytes: header (16) + 18 vec4 (288). states are 9 (origin,dir) pairs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugPhotonGpu {
    pub num_states: u32, // = 1 + successful scatters (number of (origin,dir) pairs)
    pub lambda: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub states: [[f32; 4]; 18], // 9 pairs: [origin.xyz,_][dir.xyz,_]
}

/// Max recorded (origin, dir) pairs per debug photon (must match MAX_PAIRS in WGSL).
pub const DEBUG_MAX_PAIRS: usize = 9;

/// Build the camera projection parameters from `Camera::view_basis()`.
/// Mirrors Camera::project's precomputable fields.
pub fn camera_proj_params(cam: &Camera) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
    let (origin, lower_left, horizontal, vertical) = cam.view_basis();
    let u_hat = horizontal.normalize();
    let vw    = horizontal.length();
    let v_hat = vertical.normalize();
    let vh    = vertical.length();
    // w = origin - horizontal/2 - vertical/2 - lower_left  (camera "back" vector)
    // In Camera::project: let c = dir.dot(w); in front => c < 0
    let w_vec = origin - horizontal * 0.5 - vertical * 0.5 - lower_left;
    (
        [origin.x,    origin.y,    origin.z,    0.0],
        [u_hat.x,     u_hat.y,     u_hat.z,     vw ],
        [v_hat.x,     v_hat.y,     v_hat.z,     vh ],
        [w_vec.x,     w_vec.y,     w_vec.z,     0.0],
    )
}

/// Flatten Scene into GPU-friendly arrays (same logic as lib.rs flatten_scene).
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
                    Glass::Bk7   => 0u32,
                    Glass::Sf11  => 1,
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

    // wgpu storage buffers can't be zero-sized.
    if planes.is_empty() {
        planes.push(GpuPlane { normal: [0.0, 1.0, 0.0], d: 0.0 });
    }
    if mats.is_empty() {
        mats.push(GpuMaterial { reflectance: 0.0, glass: 0, kind: 0, _pad: 0 });
    }
    if prims.is_empty() {
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

/// GPU forward photon film.
///
/// Holds all GPU buffers for the VOL-2 forward photon kernel plus a CPU f32
/// accumulator for progressive multi-batch rendering.
pub struct GpuPhotonFilm<'ctx> {
    device:        &'ctx wgpu::Device,
    queue:         &'ctx wgpu::Queue,
    pipeline:      wgpu::ComputePipeline,
    bind_group:    wgpu::BindGroup,
    params:        VolParamsGpu,
    params_buf:    wgpu::Buffer,
    film_buf:      wgpu::Buffer,
    read_buf:      wgpu::Buffer,
    debug_buf:     wgpu::Buffer,
    debug_read:    wgpu::Buffer,
    debug_capacity: u32,
    accum:         Vec<f32>,
    pub width:     usize,
    pub height:    usize,
}

/// Fixed debug-buffer capacity (photons recorded). The binding is always present,
/// so it must be sized for the largest debug run we might do.
const DEBUG_CAPACITY: u32 = 2048;

impl<'ctx> GpuPhotonFilm<'ctx> {
    /// Build the kernel. `scene`, `camera`, `beam` define the scene.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'ctx GpuContext,
        scene: Scene,
        camera: &Camera,
        beam: &Beam,
        width: usize,
        height: usize,
        n_photons: u32,
        seed: u32,
        max_dist: f32,
    ) -> Self {
        let device = &ctx.device;
        let queue  = &ctx.queue;

        let n = width * height;
        let film_len  = (3 * n) as u64;
        let film_size = film_len * 4;

        // Shader: rng.wgsl + volume.wgsl
        let shader_src = concat!(
            include_str!("shaders/rng.wgsl"),
            include_str!("shaders/volume.wgsl"),
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("volume_photon"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Flatten scene.
        let n_real = scene.primitives.len() as u32;
        let (prims, plane_vec, mat_vec) = flatten_scene(&scene);

        // Camera projection params.
        let (cam_origin, cam_u, cam_v, cam_w) = camera_proj_params(camera);

        // Build params.
        let params = VolParamsGpu {
            n_primitives: n_real,
            n_photons,
            seed,
            max_dist,
            width:  width as u32,
            height: height as u32,
            debug_mode:  0,
            debug_count: 0,
            beam_corner: [beam.corner.x, beam.corner.y, beam.corner.z, 0.0],
            beam_u:      [beam.u.x,      beam.u.y,      beam.u.z,      0.0],
            beam_v:      [beam.v.x,      beam.v.y,      beam.v.z,      0.0],
            beam_dir:    [beam.dir.x,    beam.dir.y,    beam.dir.z,    0.0],
            cam_origin,
            cam_u,
            cam_v,
            cam_w,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let prim_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_primitives"),
            contents: cast_slice(&prims),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let plane_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_planes"),
            contents: cast_slice(&plane_vec),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_materials"),
            contents: cast_slice(&mat_vec),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let tables_data = sample_tables();
        let tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_tables"),
            contents: cast_slice(&tables_data),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        let film_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_film"),
            size:               film_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_film_read"),
            size:               film_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Debug per-photon state buffer (always bound; only written when debug_mode==1).
        let debug_size = (DEBUG_CAPACITY as u64) * std::mem::size_of::<DebugPhotonGpu>() as u64;
        let debug_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_debug"),
            size:               debug_size,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_read = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_debug_read"),
            size:               debug_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("vol_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("vol_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("vol_photon"),
            layout:      Some(&pipeline_layout),
            module:      &module,
            entry_point: "main",
            cache:       None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("vol_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: prim_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: plane_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tables_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: film_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: debug_buf.as_entire_binding() },
            ],
        });

        let accum = vec![0.0f32; 3 * n];

        GpuPhotonFilm {
            device,
            queue,
            pipeline,
            bind_group,
            params,
            params_buf,
            film_buf,
            read_buf,
            debug_buf,
            debug_read,
            debug_capacity: DEBUG_CAPACITY,
            accum,
            width,
            height,
        }
    }

    /// Dispatch the photon kernel (normal splat path), then resolve into the
    /// f32 accumulator. Debug recording is OFF.
    pub fn trace_and_resolve(&mut self, n_photons: u32, seed: u32) {
        self.params.n_photons   = n_photons;
        self.params.seed        = seed;
        self.params.debug_mode  = 0;
        self.params.debug_count = 0;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = n_photons.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);

        // Resolve: read back film, accumulate, clear.
        let raw = self.read_film_raw();
        for (i, v) in raw.iter().enumerate() {
            self.accum[i] += *v as f32 / SCALE;
        }
        self.clear_film();
    }

    /// Dispatch the photon kernel with debug recording ON for photons
    /// `0..debug_count`, then read back their per-bounce state.
    ///
    /// Returns one `DebugPhotonGpu` per recorded photon. The splat path still
    /// runs (the film also gets contributions), but we only care about the
    /// debug records here. `debug_count` is clamped to the buffer capacity.
    pub fn trace_debug(&mut self, n_photons: u32, seed: u32, debug_count: u32) -> Vec<DebugPhotonGpu> {
        let debug_count = debug_count.min(self.debug_capacity).min(n_photons);

        // Zero the debug buffer first so stale data can't leak in.
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.clear_buffer(&self.debug_buf, 0, None);
            self.queue.submit([enc.finish()]);
        }

        self.params.n_photons   = n_photons;
        self.params.seed        = seed;
        self.params.debug_mode  = 1;
        self.params.debug_count = debug_count;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = n_photons.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);

        // Read back the debug buffer.
        let want = debug_count as usize;
        let stride = std::mem::size_of::<DebugPhotonGpu>() as u64;
        let copy_size = (want as u64) * stride;

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.debug_buf, 0, &self.debug_read, 0, copy_size);
        self.queue.submit([enc.finish()]);

        let slice = self.debug_read.slice(0..copy_size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let raw = slice.get_mapped_range();
        let recs: Vec<DebugPhotonGpu> = cast_slice::<u8, DebugPhotonGpu>(&raw).to_vec();
        drop(raw);
        self.debug_read.unmap();

        // Reset to splat-only state and clear the film (so debug runs don't pollute accum).
        self.params.debug_mode  = 0;
        self.params.debug_count = 0;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        self.clear_film();

        recs
    }

    /// Read back the raw u32 film buffer.
    fn read_film_raw(&self) -> Vec<u32> {
        let film_size = (3 * self.width * self.height * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.film_buf, 0, &self.read_buf, 0, film_size);
        self.queue.submit([enc.finish()]);

        let slice = self.read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let raw = slice.get_mapped_range();
        let result: Vec<u32> = cast_slice::<u8, u32>(&raw).to_vec();
        drop(raw);
        self.read_buf.unmap();
        result
    }

    /// Zero the GPU film buffer.
    pub fn clear_film(&self) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.film_buf, 0, None);
        self.queue.submit([enc.finish()]);
    }

    /// Borrow the f32 accumulator.
    pub fn accum(&self) -> &[f32] {
        &self.accum
    }

    /// Auto-expose from Y channel.
    pub fn auto_expose(&self, scale: f32) -> f32 {
        let n = self.width * self.height;
        let max_y = (0..n).map(|i| self.accum[3 * i + 1]).fold(0.0f32, f32::max);
        (scale / max_y.max(1e-6)).max(1.0)
    }

    /// Tonemap the f32 accum to RGB8.
    pub fn to_rgb8(&self, exposure: f32) -> Vec<u8> {
        use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let xyz = [self.accum[3 * i], self.accum[3 * i + 1], self.accum[3 * i + 2]];
            let linear = xyz_to_linear_srgb(xyz);
            let rgb = tonemap_to_u8(linear, exposure);
            out.extend_from_slice(&rgb);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
pub mod tests {
    use super::*;
    use spectral_core::camera::Camera;
    use spectral_core::geom::{ConvexSolid, Plane, Ray};
    use spectral_core::lighttrace::Beam;
    use spectral_core::material::Material;
    use spectral_core::rng::PathRng;
    use spectral_core::scene::Scene;
    use spectral_core::sellmeier::Glass;
    use spectral_core::spectrum::{LAMBDA_MIN, LAMBDA_RANGE};
    use glam::Vec3;

    #[test]
    fn debug_photon_layout_is_304_bytes() {
        // Must match WGSL DebugPhoton: header(16) + array<vec4<f32>,18>(288) = 304.
        // A mismatch would corrupt the GPU debug readback silently.
        assert_eq!(std::mem::size_of::<DebugPhotonGpu>(), 304);
        assert_eq!(std::mem::size_of::<DebugPhotonGpu>() % 16, 0);
    }

    /// Build the exact prism_dsotm.rs scene + beam + camera.
    pub fn dsotm_scene() -> (Scene, Beam, Camera) {
        let k = 0.9_f32;
        let planes = vec![
            Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k },
            Plane { normal: Vec3::new( 0.866_025_4, 0.5, 0.0), d: -k },
            Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },
            Plane { normal: Vec3::Z,  d: -1.0 },
            Plane { normal: -Vec3::Z, d: -1.0 },
        ];
        let mut scene = Scene::new();
        scene.background = 0.0;
        scene.add_solid(ConvexSolid { planes }, Material::Dielectric { glass: Glass::Sf11 });

        let bdir  = Vec3::new(0.84, 0.54, 0.0).normalize();
        let bperp = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
        let beam = Beam {
            corner: Vec3::new(-2.6, -0.95, -0.6),
            u:      Vec3::new(0.0, 0.0, 1.2),
            v:      bperp,
            dir:    bdir,
        };

        let cam = Camera::look_at(
            Vec3::new(0.5, -0.5, 12.0),
            Vec3::new(0.5, -0.5,  0.0),
            Vec3::Y,
            42.0,
            1.0,
        );

        (scene, beam, cam)
    }

    /// CPU simulation of one photon through the DSOTM beam — convenience wrapper.
    pub fn cpu_simulate_photon(
        scene: &Scene,
        photon_idx: u32,
        seed: u32,
        max_dist: f32,
    ) -> (Vec<(Vec3, Vec3)>, f32) {
        let (_s, beam, _c) = dsotm_scene();
        let (states, lambda, _tir) = cpu_simulate_photon_beam(scene, &beam, photon_idx, seed, max_dist);
        (states, lambda)
    }

    /// CPU simulation of one photon through an arbitrary beam — mirrors
    /// volume.rs render_volumetric_scene loop (and the WGSL kernel) EXACTLY,
    /// including RNG consumption order.
    ///
    /// Returns (ray_states, lambda, tir_count) where ray_states is one (origin,dir)
    /// pair pushed before the loop plus one after each successful scatter, and
    /// tir_count is the number of bounces that hit the TIR fallback (refract
    /// returned None) — used to prove the parity set exercises the TIR branch.
    pub fn cpu_simulate_photon_beam(
        scene: &Scene,
        beam: &Beam,
        photon_idx: u32,
        seed: u32,
        max_dist: f32,
    ) -> (Vec<(Vec3, Vec3)>, f32, u32) {
        use spectral_core::optics::refract;

        let mut rng = PathRng::new(photon_idx, seed);

        let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
        let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32();

        let beam_dir = beam.dir;
        let mut ray_states: Vec<(Vec3, Vec3)> = vec![(origin, beam_dir)];
        let mut ro = origin;
        let mut rd = beam_dir;
        let mut tir_count = 0u32;

        for _ in 0..8 {
            let hit = scene.intersect(&Ray { origin: ro, dir: rd });
            let seg_len = hit.as_ref().map(|(h, _)| h.t).unwrap_or(max_dist);

            // Consume seg draws (RNG order critical)
            for _ in 0..4 {
                let _dist = rng.next_f32() * seg_len;
            }

            match hit {
                Some((h, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(lambda),
                        _ => 1.0,
                    };
                    // Detect whether this dielectric interaction WOULD TIR (refract None)
                    // BEFORE consuming the Fresnel roulette draw, so we don't perturb rng.
                    if let Material::Dielectric { .. } = mat {
                        let (n1, n2) = if h.front_face { (1.0, n_hero) } else { (n_hero, 1.0) };
                        if refract(rd, h.normal, n1, n2).is_none() {
                            tir_count += 1;
                        }
                    }
                    match mat.scatter(rd, &h, lambda, n_hero, &mut rng) {
                        Some(sc) => {
                            ro = h.point;
                            rd = sc.dir;
                            ray_states.push((ro, rd));
                        }
                        None => break,
                    }
                }
                None => break,
            }
        }

        (ray_states, lambda, tir_count)
    }

    // GPU simulation note: per-photon state extraction would require a debug readback
    // kernel. Instead, parity is verified by:
    //  - CPU self-consistency (forward_parity_cpu_deterministic)
    //  - TIR-boundary photon detection on CPU (forward_parity_tir_boundary_detected)
    //  - GPU film nonzero + deterministic across two runs (gpu tests below)

    /// Find a photon index that exhibits TIR at short wavelength (high n) but
    /// transmits at long wavelength for the DSOTM prism exit face.
    /// Returns (photon_idx, lambda_tir, lambda_transmit).
    pub fn find_tir_boundary_photon(
        scene: &Scene,
        seed: u32,
        max_dist: f32,
        n_search: u32,
    ) -> Option<(u32, f32, f32)> {
        let beam_corner = Vec3::new(-2.6, -0.95, -0.6);
        let beam_u      = Vec3::new(0.0, 0.0, 1.2);
        let bperp       = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
        let beam_dir    = Vec3::new(0.84, 0.54, 0.0).normalize();

        // SF11 exit-face critical angles:
        //   lambda=400nm: n~1.87 -> arcsin(1/1.87)~32.4deg
        //   lambda=700nm: n~1.76 -> arcsin(1/1.76)~34.7deg
        // Construct a beam aimed slightly off minimum-deviation to straddle this.
        // We'll check CPU sim for bounces that exit the prism: if exit bounce has
        // front_face=false (exiting), we can check whether it TIR'd or transmitted
        // by comparing ray_states length for short vs long wavelength.

        // We'll force-test by running each photon at two fixed wavelengths.
        let lambda_violet = 400.0_f32;
        let lambda_red    = 680.0_f32;

        for idx in 0..n_search {
            // Simulate at violet and red using the SAME rng origin/direction draws.
            let (states_v, _) = cpu_simulate_photon_fixed_lambda(
                scene, idx, seed, max_dist,
                beam_corner, beam_u, bperp, beam_dir, lambda_violet,
            );
            let (states_r, _) = cpu_simulate_photon_fixed_lambda(
                scene, idx, seed, max_dist,
                beam_corner, beam_u, bperp, beam_dir, lambda_red,
            );
            // TIR at violet means fewer bounces (violet ray stays inside or reflects back)
            // Transmit at red means more bounces (red exits prism and escapes).
            // Different bounce counts = different TIR behavior.
            if states_v.len() != states_r.len() {
                return Some((idx, lambda_violet, lambda_red));
            }
        }
        None
    }

    /// Like cpu_simulate_photon but with a fixed lambda (ignoring the rng lambda draw).
    /// Consumes the same 3 startup draws to keep origin the same, then uses fixed_lambda.
    #[allow(clippy::too_many_arguments)]
    fn cpu_simulate_photon_fixed_lambda(
        scene: &Scene,
        photon_idx: u32,
        seed: u32,
        max_dist: f32,
        beam_corner: Vec3,
        beam_u: Vec3,
        beam_v: Vec3,
        beam_dir: Vec3,
        fixed_lambda: f32,
    ) -> (Vec<(Vec3, Vec3)>, f32) {
        let mut rng = PathRng::new(photon_idx, seed);
        let origin = beam_corner + beam_u * rng.next_f32() + beam_v * rng.next_f32();
        let _lambda_draw = rng.next_f32(); // consume lambda draw to keep RNG state aligned

        let mut ray_states: Vec<(Vec3, Vec3)> = vec![(origin, beam_dir)];
        let mut ro = origin;
        let mut rd = beam_dir;

        for _ in 0..8 {
            let hit = scene.intersect(&Ray { origin: ro, dir: rd });
            let seg_len = hit.as_ref().map(|(h, _)| h.t).unwrap_or(max_dist);

            for _ in 0..4 {
                let _dist = rng.next_f32() * seg_len;
            }

            match hit {
                Some((h, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(fixed_lambda),
                        _ => 1.0,
                    };
                    match mat.scatter(rd, &h, fixed_lambda, n_hero, &mut rng) {
                        Some(sc) => {
                            ro = h.point;
                            rd = sc.dir;
                            ray_states.push((ro, rd));
                        }
                        None => break,
                    }
                }
                None => break,
            }
        }

        (ray_states, fixed_lambda)
    }

    // -----------------------------------------------------------------------
    // Supporting tests (CPU evidence + GPU smoke). The REAL GPU-vs-CPU per-photon
    // parity gate is forward_parity_gpu_matches_cpu (further below).
    //   - forward_parity_cpu_deterministic: CPU oracle is reproducible.
    //   - forward_parity_tir_boundary_detected: CPU evidence that the scene has
    //     photons that TIR at violet but transmit at red (the failure mode the
    //     real parity gate must catch).
    //   - forward_parity_gpu_film_nonzero: GPU kernel runs and produces a film.
    //   - forward_gpu_determinism: GPU atomicAdd is order-independent.
    // -----------------------------------------------------------------------

    #[test]
    fn forward_parity_cpu_deterministic() {
        let (scene, _beam, _cam) = dsotm_scene();
        let seed = 42u32;
        let max_dist = 14.0_f32;

        // Same photon, same seed -> identical ray states (CPU self-consistency).
        for idx in [0u32, 1, 5, 10, 100] {
            let (states_a, lambda_a) = cpu_simulate_photon(&scene, idx, seed, max_dist);
            let (states_b, lambda_b) = cpu_simulate_photon(&scene, idx, seed, max_dist);
            assert_eq!(lambda_a.to_bits(), lambda_b.to_bits(),
                "photon {idx}: lambda not deterministic");
            assert_eq!(states_a.len(), states_b.len(),
                "photon {idx}: bounce count not deterministic");
            for (i, ((oa, da), (ob, db))) in states_a.iter().zip(states_b.iter()).enumerate() {
                assert!((oa - ob).length() < 1e-5,
                    "photon {idx} bounce {i}: origin not deterministic: {oa:?} vs {ob:?}");
                assert!((da - db).length() < 1e-5,
                    "photon {idx} bounce {i}: dir not deterministic: {da:?} vs {db:?}");
            }
        }
    }

    #[test]
    fn forward_parity_tir_boundary_detected() {
        // CRITICAL parity gate: must find photons that TIR at violet but transmit at red.
        // If all photons in the search space cleanly transmit at ALL wavelengths,
        // this test fails and we need to investigate beam geometry.
        let (scene, _beam, _cam) = dsotm_scene();
        let seed = 42u32;
        let max_dist = 14.0_f32;

        let result = find_tir_boundary_photon(&scene, seed, max_dist, 5000);

        assert!(
            result.is_some(),
            "Could not find any photon that TIRs at violet but transmits at red in 5000 tries. \
             This means the beam is fully inside the transmission window or the TIR detection \
             logic is broken. Investigate: SF11 critical angle at 400nm ~32.4 deg, 680nm ~34.7 deg."
        );

        let (idx, lambda_v, lambda_r) = result.unwrap();
        eprintln!(
            "TIR-boundary photon: idx={idx}, violet={lambda_v}nm TIR-or-fewer-bounces, \
             red={lambda_r}nm transmit-or-more-bounces"
        );

        // Verify they do differ.
        let beam_corner = Vec3::new(-2.6, -0.95, -0.6);
        let beam_u      = Vec3::new(0.0, 0.0, 1.2);
        let bperp       = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
        let beam_dir    = Vec3::new(0.84, 0.54, 0.0).normalize();
        let (sv, _) = cpu_simulate_photon_fixed_lambda(
            &scene, idx, seed, max_dist, beam_corner, beam_u, bperp, beam_dir, lambda_v);
        let (sr, _) = cpu_simulate_photon_fixed_lambda(
            &scene, idx, seed, max_dist, beam_corner, beam_u, bperp, beam_dir, lambda_r);
        assert_ne!(sv.len(), sr.len(),
            "Expected different bounce counts for violet vs red at TIR-boundary photon {idx}");
    }

    #[test]
    fn forward_parity_gpu_film_nonzero() {
        // Integration test: GPU kernel runs and produces nonzero film.
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping forward_parity_gpu_film_nonzero");
            return;
        };

        let (scene, beam, cam) = dsotm_scene();
        let n_photons = 200_000u32;
        let seed = 42u32;
        let max_dist = 14.0_f32;
        let (w, h) = (256usize, 256usize);

        let mut film = GpuPhotonFilm::new(
            &ctx, scene, &cam, &beam, w, h, n_photons, seed, max_dist,
        );
        film.trace_and_resolve(n_photons, seed);

        let total_y: f32 = (0..w * h).map(|i| film.accum()[3 * i + 1]).sum();
        assert!(
            total_y > 0.0,
            "GPU photon film is all-zero after {n_photons} photons; dispersion kernel not working"
        );
        eprintln!("forward_parity_gpu_film_nonzero: total Y = {total_y:.4}");
    }

    #[test]
    fn forward_gpu_determinism() {
        // GPU determinism only: two fresh films with the same seed produce
        // bit-identical accumulators (atomicAdd is order-independent). This is NOT
        // a CPU-vs-GPU parity test — that is forward_parity_gpu_matches_cpu below.
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping forward_gpu_determinism");
            return;
        };

        let (scene_a, beam_a, cam_a) = dsotm_scene();
        let (scene_b, beam_b, cam_b) = dsotm_scene();
        let seed = 99u32;
        let max_dist = 14.0_f32;
        let n_photons = 100_000u32;
        let (w, h) = (128usize, 128usize);

        let mut film_a = GpuPhotonFilm::new(
            &ctx, scene_a, &cam_a, &beam_a, w, h, n_photons, seed, max_dist,
        );
        film_a.trace_and_resolve(n_photons, seed);

        let mut film_b = GpuPhotonFilm::new(
            &ctx, scene_b, &cam_b, &beam_b, w, h, n_photons, seed, max_dist,
        );
        film_b.trace_and_resolve(n_photons, seed);

        for i in 0..3 * w * h {
            let a = film_a.accum()[i];
            let b = film_b.accum()[i];
            assert!((a - b).abs() < 1e-6,
                "GPU film not deterministic at channel {i}: {a} vs {b}");
        }

        let total_a: f32 = (0..w * h).map(|i| film_a.accum()[3 * i + 1]).sum();
        assert!(total_a > 0.0, "GPU film A is all-zero");
        eprintln!("forward_gpu_determinism: two runs bit-identical, total Y = {total_a:.4}");
    }

    // -----------------------------------------------------------------------
    // THE REAL PARITY GATE: GPU photon paths must match the CPU oracle
    // per-bounce, including the violet TIR boundary.
    //
    // A debug readback path in volume.wgsl records, for photons 0..debug_count,
    // the per-bounce (origin, dir) pairs and the sampled lambda — mirroring
    // cpu_simulate_photon's ray_states exactly. We compare:
    //   - num_states (bounce count) — EXACT equality. This is what catches a
    //     kernel that TIRs the violet edge when the CPU transmits (or vice versa).
    //   - each bounce's origin and dir — within tolerance.
    // We also assert the sampled set provably contains a violet (<430nm) photon
    // whose path enters the prism (num_states > 1), so the gate exercises the
    // dispersive/TIR-prone end, not just the bright center.
    // -----------------------------------------------------------------------
    /// A steep beam engineered to make short-wavelength (violet, high-n) photons
    /// TIR at the SF11 prism's exit face while long-wavelength (red) ones still
    /// transmit. This forces the kernel's `refract_dir(...) -> w<=0.5` TIR fallback
    /// branch to execute on real photons, so the parity gate is not blind to it.
    ///
    /// The DSOTM beam is at minimum deviation (everything transmits); this one is
    /// aimed steeper so the internal ray hits the exit face past the critical angle
    /// at the violet end.
    fn tir_beam() -> Beam {
        // Found by scratch_find_tir_beam (ignored sweep): a shallow beam entering
        // the left face such that the internal ray hits the exit face at ~the
        // critical angle. At violet (400nm, n~1.87) it TIRs; at red (680nm, n~1.76)
        // it transmits. dir=(0.934,0.358,0), origins near (-3.0,-0.8,0).
        let bdir  = Vec3::new(0.934, 0.358, 0.0).normalize();
        // v spread along the beam's perpendicular in XY, small, centered so most
        // photons land in the TIR-straddling band around y=-0.8.
        let bperp = Vec3::new(-0.358, 0.934, 0.0).normalize() * 0.6;
        Beam {
            corner: Vec3::new(-3.0, -1.1, -0.6),
            u:      Vec3::new(0.0, 0.0, 1.2), // Z sheet width
            v:      bperp,
            dir:    bdir,
        }
    }

    /// Run the GPU-vs-CPU per-photon parity check for a specific beam.
    /// Returns (compared, total_bounces, max_origin_delta, max_dir_delta,
    ///          total_tir_bounces, violet_probe, tir_probe).
    #[allow(clippy::type_complexity)]
    fn run_parity_for_beam(
        ctx: &GpuContext,
        beam: &Beam,
        seed: u32,
        debug_count: u32,
        label: &str,
    ) -> (usize, usize, f32, f32, u32, Option<(u32, f32, usize)>, Option<(u32, f32, usize)>) {
        let max_dist = 14.0_f32;
        let (w, h) = (256usize, 256usize);
        let n_photons = debug_count;

        let (scene, _b, cam) = dsotm_scene();
        let mut film = GpuPhotonFilm::new(
            ctx, scene, &cam, beam, w, h, n_photons, seed, max_dist,
        );
        let gpu_recs = film.trace_debug(n_photons, seed, debug_count);
        assert_eq!(gpu_recs.len(), debug_count as usize, "[{label}] debug readback count mismatch");

        let (cpu_scene, _b2, _c2) = dsotm_scene();

        let tol = 1e-4_f32;
        let mut compared = 0usize;
        let mut max_origin_delta = 0.0f32;
        let mut max_dir_delta = 0.0f32;
        let mut total_bounces = 0usize;
        let mut total_tir = 0u32;
        let mut violet_probe: Option<(u32, f32, usize)> = None;
        let mut tir_probe: Option<(u32, f32, usize)> = None;

        for idx in 0..debug_count {
            let rec = &gpu_recs[idx as usize];
            let (cpu_states, cpu_lambda, tir_count) =
                cpu_simulate_photon_beam(&cpu_scene, beam, idx, seed, max_dist);

            assert!(
                (rec.lambda - cpu_lambda).abs() < 1e-3,
                "[{label}] photon {idx}: lambda mismatch GPU {} vs CPU {cpu_lambda}", rec.lambda
            );

            // BOUNCE COUNT EQUALITY — the TIR-vs-transmit discriminator.
            // Asserted BEFORE per-bounce float compare so divergence reports cleanly.
            assert_eq!(
                rec.num_states as usize, cpu_states.len(),
                "[{label}] photon {idx} (lambda={:.1}nm, cpu_tir={tir_count}): GPU bounce count {} \
                 != CPU {} — TIR-vs-transmit divergence at the dispersive edge",
                cpu_lambda, rec.num_states, cpu_states.len()
            );

            for (b, (cpu_o, cpu_d)) in cpu_states.iter().enumerate() {
                let gpu_o = glam::Vec3::new(
                    rec.states[2 * b][0], rec.states[2 * b][1], rec.states[2 * b][2]);
                let gpu_d = glam::Vec3::new(
                    rec.states[2 * b + 1][0], rec.states[2 * b + 1][1], rec.states[2 * b + 1][2]);
                let od = (gpu_o - *cpu_o).length();
                let dd = (gpu_d - *cpu_d).length();
                max_origin_delta = max_origin_delta.max(od);
                max_dir_delta = max_dir_delta.max(dd);
                assert!(od < tol,
                    "[{label}] photon {idx} (lambda={cpu_lambda:.1}nm) bounce {b}: origin delta {od} >= {tol} \
                     (GPU {gpu_o:?} vs CPU {cpu_o:?})");
                assert!(dd < tol,
                    "[{label}] photon {idx} (lambda={cpu_lambda:.1}nm) bounce {b}: dir delta {dd} >= {tol} \
                     (GPU {gpu_d:?} vs CPU {cpu_d:?})");
            }

            if cpu_lambda < 430.0 && cpu_states.len() > 1 && violet_probe.is_none() {
                violet_probe = Some((idx, cpu_lambda, cpu_states.len()));
            }
            if tir_count > 0 && tir_probe.is_none() {
                tir_probe = Some((idx, cpu_lambda, cpu_states.len()));
            }
            total_tir += tir_count;
            total_bounces += cpu_states.len();
            compared += 1;
        }

        (compared, total_bounces, max_origin_delta, max_dir_delta, total_tir, violet_probe, tir_probe)
    }

    #[test]
    fn forward_parity_gpu_matches_cpu() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping forward_parity_gpu_matches_cpu (no GPU adapter)");
            return;
        };
        let seed = 42u32;
        let debug_count = 512u32;

        // --- Case 1: DSOTM transmit beam (the production scene) ---
        let (_s, dsotm, _c) = dsotm_scene();
        let (c1, tb1, mo1, md1, tir1, violet1, _tirp1) =
            run_parity_for_beam(&ctx, &dsotm, seed, debug_count, "dsotm");

        assert!(
            violet1.is_some(),
            "[dsotm] No violet (<430nm) photon entered the prism in {debug_count} photons; \
             the parity set does not provably exercise the dispersive edge."
        );
        let (vi, vl, vb) = violet1.unwrap();
        eprintln!(
            "forward_parity_gpu_matches_cpu [dsotm]: {c1} photons, {tb1} bounces, {tir1} TIR bounces, \
             max origin Δ={mo1:.3e}, max dir Δ={md1:.3e}, tol=1e-4"
        );
        eprintln!("  [dsotm] violet probe: photon {vi} λ={vl:.1}nm, {vb} states, GPU bounce count == CPU");

        // --- Case 2: TIR beam (forces the violet TIR-fallback branch) ---
        let tirb = tir_beam();
        let (c2, tb2, mo2, md2, tir2, _violet2, tirp2) =
            run_parity_for_beam(&ctx, &tirb, seed, debug_count, "tir");

        // The whole point of this beam: it MUST drive real TIR events through the
        // kernel's else-branch, and the GPU must still agree with the CPU on the
        // (longer) bounce count. If this beam produced zero TIR, the engineered
        // geometry is wrong and the gate would be blind to a violet-drop bug.
        assert!(
            tir2 > 0,
            "[tir] The TIR beam produced ZERO total-internal-reflection bounces across {debug_count} \
             photons. The geometry no longer straddles the critical angle; the parity gate is not \
             exercising the TIR fallback branch. Re-aim tir_beam()."
        );
        assert!(
            tirp2.is_some(),
            "[tir] No individual photon recorded a TIR bounce despite total tir2={tir2}."
        );
        let (ti, tl, tbc) = tirp2.unwrap();
        eprintln!(
            "forward_parity_gpu_matches_cpu [tir]: {c2} photons, {tb2} bounces, {tir2} TIR bounces, \
             max origin Δ={mo2:.3e}, max dir Δ={md2:.3e}, tol=1e-4"
        );
        eprintln!(
            "  [tir] TIR probe: photon {ti} λ={tl:.1}nm took a TIR bounce ({tbc} states); \
             GPU bounce count == CPU through the TIR fallback"
        );
    }
}

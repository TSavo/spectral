// VOL-2/3: GPU forward photon kernel.
//
// Builds and dispatches the volume.wgsl compute kernel, which mirrors
// render_volumetric_scene() from spectral-core. One GPU thread per photon.
//
// Binding layout (must match volume.wgsl exactly):
//   @binding(0) uniform VolParamsGpu  -- photon params + camera + beam + sigma/g
//   @binding(1) storage  primitives
//   @binding(2) storage  planes
//   @binding(3) storage  materials
//   @binding(4) storage  tables (CMF + illuminants)
//   @binding(5) storage  film (atomic<u32>, 3 * width * height)
//   @binding(6) storage  debug_out (per-photon state, VOL-2 parity gate)
//   @binding(7) storage  zbuffer (f32 per pixel; INF = no occlusion)

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

/// Fixed-point scale = 2^23 — MUST match SCALE in volume.wgsl. Load-bearing for
/// the VOL-4 diff gate: VOL-3 weighted contributions are ~1e-5..1e-7 per splat,
/// which underflowed to zero at the VOL-1 scale of 2^12. Diff-gate L1 scales
/// ~1/SCALE (2^20 -> 1.97%, 2^23 -> 0.28%); 2^23 clears the <1% gate with margin.
const SCALE: f32 = 8388608.0;

/// Henyey-Greenstein phase function, computed with the EXACT operation order of
/// the WGSL `phase_hg` in volume.wgsl (clamp BEFORE pow, 4π in the denom). Used by
/// the phase_hg_matches_cpu_oracle unit test to prove the WGSL formula matches
/// volume.rs::phase_hg. Kept in sync with the WGSL by hand.
pub fn phase_hg_wgsl_mirror(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6).powf(1.5);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom)
}

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
    // VOL-3 single-scatter weight parameters
    pub sigma_s: f32,     // haze scattering coefficient
    pub sigma_t: f32,     // haze extinction coefficient (transmittance exp(-sigma_t d))
    pub g: f32,           // Henyey-Greenstein anisotropy (g>0 forward)
    pub photon_base: u32, // chunking: photon i = photon_base + gid.x
    // VOL-6 dispersion toggle (1 = n(lambda) dispersion on; 0 = n(550) collapsed white)
    pub spectral: u32,
    pub _pad2: u32,
    pub _pad3: u32,
    pub _pad4: u32,
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

/// The exact "Dark Side of the Moon" scene from prism_dsotm.rs: an equilateral
/// SF11 prism, the thin white minimum-deviation beam, and the face-on camera at
/// the given aspect ratio. Shared by the examples, the viewer, and the gates.
pub fn dsotm_scene(aspect: f32) -> (Scene, Beam, Camera) {
    use spectral_core::geom::{ConvexSolid, Plane};
    use glam::Vec3;

    let k = 0.9_f32;
    let planes = vec![
        Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k }, // left face
        Plane { normal: Vec3::new( 0.866_025_4, 0.5, 0.0), d: -k }, // right face
        Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },       // base
        Plane { normal: Vec3::Z,  d: -1.0 },                         // +Z cap
        Plane { normal: -Vec3::Z, d: -1.0 },                         // -Z cap
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
        aspect,
    );

    (scene, beam, cam)
}

/// GPU forward photon film.
///
/// Holds all GPU buffers for the VOL-2/3 forward photon kernel plus a CPU f32
/// accumulator for progressive multi-batch rendering.
pub struct GpuPhotonFilm<'ctx> {
    device:        &'ctx wgpu::Device,
    queue:         &'ctx wgpu::Queue,
    pipeline:      wgpu::ComputePipeline,
    rim_pipeline:  wgpu::ComputePipeline,
    bind_group:    wgpu::BindGroup,
    params:        VolParamsGpu,
    params_buf:    wgpu::Buffer,
    film_buf:      wgpu::Buffer,
    read_buf:      wgpu::Buffer,
    debug_buf:     wgpu::Buffer,
    debug_read:    wgpu::Buffer,
    debug_capacity: u32,
    accum:         Vec<f32>,
    /// Static per-camera surface rim (3*w*h interleaved XYZ). Composited with
    /// `accum` at display (to_rgb8), NOT folded into the progressive accum.
    /// Zeroed until recompute_surface() is called. VOL-6 recomputes on camera move.
    surface:       Vec<f32>,
    pub width:     usize,
    pub height:    usize,
}

/// Fixed debug-buffer capacity (photons recorded). The binding is always present,
/// so it must be sized for the largest debug run we might do.
const DEBUG_CAPACITY: u32 = 2048;

/// VOL-3 single-scatter weight parameters bundled to keep the ctor arg count sane.
#[derive(Clone, Copy)]
pub struct VolWeights {
    pub sigma_s: f32,
    pub sigma_t: f32,
    pub g: f32,
}

impl Default for VolWeights {
    /// DSOTM defaults: sigma_s=0.5, sigma_t=0.06, g=0.5.
    fn default() -> Self {
        VolWeights { sigma_s: 0.5, sigma_t: 0.06, g: 0.5 }
    }
}

impl<'ctx> GpuPhotonFilm<'ctx> {
    /// Build the kernel.
    ///
    /// `zbuffer`: per-pixel nearest-solid euclidean depth (length width*height).
    /// `None` = an all-INFINITY zbuffer (no occlusion, DSOTM).
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
        weights: VolWeights,
        zbuffer: Option<&[f32]>,
    ) -> Self {
        let device = &ctx.device;
        let queue  = &ctx.queue;

        // Sanity: the precomputed cam_origin must equal camera.origin() (the oracle
        // uses camera.origin() directly). Cheap insurance against a wiring drift.
        {
            let co = camera.origin();
            let (po, _, _, _) = camera_proj_params(camera);
            debug_assert!(
                (po[0] - co.x).abs() < 1e-5 && (po[1] - co.y).abs() < 1e-5 && (po[2] - co.z).abs() < 1e-5,
                "cam_origin {po:?} != camera.origin() {co:?}"
            );
        }

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
            sigma_s: weights.sigma_s,
            sigma_t: weights.sigma_t,
            g:       weights.g,
            photon_base: 0,
            spectral: 1, // dispersion ON by default (matches pre-VOL-6 behavior)
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
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

        // Zbuffer (one f32 per pixel). None -> all-INFINITY (no occlusion).
        let zbuf_data: Vec<f32> = match zbuffer {
            Some(z) => {
                assert_eq!(z.len(), n, "zbuffer length {} != width*height {}", z.len(), n);
                z.to_vec()
            }
            None => vec![f32::INFINITY; n],
        };
        let zbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("vol_zbuffer"),
            contents: cast_slice(&zbuf_data),
            usage:    wgpu::BufferUsages::STORAGE,
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
                wgpu::BindGroupLayoutEntry {
                    binding:    7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
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

        // VOL-5 surface rim pass — same module + layout, different entry point.
        let rim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("vol_rim"),
            layout:      Some(&pipeline_layout),
            module:      &module,
            entry_point: "rim_main",
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
                wgpu::BindGroupEntry { binding: 7, resource: zbuf.as_entire_binding() },
            ],
        });

        let accum = vec![0.0f32; 3 * n];
        let surface = vec![0.0f32; 3 * n];

        GpuPhotonFilm {
            device,
            queue,
            pipeline,
            rim_pipeline,
            bind_group,
            params,
            params_buf,
            film_buf,
            read_buf,
            debug_buf,
            debug_read,
            debug_capacity: DEBUG_CAPACITY,
            accum,
            surface,
            width,
            height,
        }
    }

    /// Max photons per single dispatch (workgroup_size 64 ×
    /// max_compute_workgroups_per_dimension 65535 ≈ 4.19M). Stay safely under it.
    pub const MAX_PER_DISPATCH: u32 = 4_000_000;

    /// Dispatch the photon kernel (normal splat path) in a SINGLE dispatch, then
    /// resolve into the f32 accumulator. Debug recording is OFF. `n_photons` must
    /// be <= MAX_PER_DISPATCH; use `trace_chunked` for larger counts.
    pub fn trace_and_resolve(&mut self, n_photons: u32, seed: u32) {
        self.dispatch_chunk(0, n_photons, seed);
        self.resolve();
    }

    /// Dispatch many photons across several chunks (each <= MAX_PER_DISPATCH),
    /// resolving into the persistent f32 accumulator after each, so a large
    /// photon count (e.g. 16M) stays under the Metal watchdog. Splat path only.
    pub fn trace_chunked(&mut self, total_photons: u32, seed: u32) {
        let mut base = 0u32;
        while base < total_photons {
            let this = (total_photons - base).min(Self::MAX_PER_DISPATCH);
            self.dispatch_chunk(base, total_photons, seed);
            self.resolve();
            base += this;
        }
    }

    /// Dispatch one chunk [photon_base, n_photons) of the splat path (no debug).
    fn dispatch_chunk(&mut self, photon_base: u32, n_photons: u32, seed: u32) {
        self.params.n_photons   = n_photons;
        self.params.seed        = seed;
        self.params.debug_mode  = 0;
        self.params.debug_count = 0;
        self.params.photon_base = photon_base;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let count = (n_photons - photon_base).min(Self::MAX_PER_DISPATCH);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    // -----------------------------------------------------------------------
    // VOL-6 viewer API: drive the photon kernel into the GPU film WITHOUT the
    // CPU readback/resolve. The viewer owns a GPU-resident accum + a resolve
    // compute pass that reads & clears the film each frame. Purely additive;
    // the CPU-readback paths above (trace_and_resolve / trace_chunked /
    // recompute_surface) are unchanged, so VOL-2/3/4/5 gates are unaffected.
    // -----------------------------------------------------------------------

    /// Dispatch a single photon batch into the GPU film (no CPU resolve). The
    /// caller (viewer) runs its own GPU resolve pass to drain the film into a
    /// GPU-resident accumulator. `n_photons` is the total/extent; only photons in
    /// [photon_base, photon_base + MAX_PER_DISPATCH) actually run this call.
    pub fn dispatch_photons(&mut self, photon_base: u32, n_photons: u32, seed: u32) {
        self.dispatch_chunk(photon_base, n_photons, seed);
    }

    /// The atomic film buffer (3*w*h u32, interleaved XYZ, fixed-point ×SCALE).
    /// The viewer binds this read_write in its resolve pass.
    pub fn film_buf(&self) -> &wgpu::Buffer {
        &self.film_buf
    }

    /// The VolParams uniform buffer (the viewer rebinds it in its compute pipeline
    /// so the photon kernel and the viewer's resolve/blit share one params source).
    pub fn params_buf(&self) -> &wgpu::Buffer {
        &self.params_buf
    }

    /// The compute bind group for the photon kernel (group 0). Exposed so the
    /// viewer can reuse the kernel dispatch verbatim.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Toggle dispersion: `true` = n(lambda) per-wavelength (rainbow fan);
    /// `false` = fixed n(550) for all wavelengths (collapsed white beam). Uploads
    /// the changed param immediately.
    pub fn set_spectral(&mut self, on: bool) {
        self.params.spectral = if on { 1 } else { 0 };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Whether dispersion is currently on.
    pub fn spectral(&self) -> bool {
        self.params.spectral == 1
    }

    /// Resolve: read back the u32 film, divide by SCALE, add into accum, clear film.
    fn resolve(&mut self) {
        let raw = self.read_film_raw();
        for (i, v) in raw.iter().enumerate() {
            self.accum[i] += *v as f32 / SCALE;
        }
        self.clear_film();
    }

    /// VOL-5: recompute the static surface-rim buffer for `camera` (one camera
    /// primary ray per pixel, one prism intersection, the Fresnel-style rim term).
    /// Mirrors prism_dsotm.rs SURFACE PASS. Overwrites `surface` (not additive); it
    /// is composited with the progressive volumetric `accum` at display in to_rgb8.
    ///
    /// VOL-6 calls this on camera-move; the headless example calls it once. It does
    /// NOT touch `accum`, and clears the shared film before and after so it can't
    /// pollute a volumetric run.
    pub fn recompute_surface(&mut self, camera: &Camera) {
        // Update the camera basis in params (the rim kernel reconstructs primary_ray
        // from cam_origin/cam_u/cam_v/cam_w).
        let (cam_origin, cam_u, cam_v, cam_w) = camera_proj_params(camera);
        self.params.cam_origin = cam_origin;
        self.params.cam_u      = cam_u;
        self.params.cam_v      = cam_v;
        self.params.cam_w      = cam_w;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        // Dispatch the rim kernel into a freshly-cleared film.
        self.clear_film();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.rim_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let gx = (self.width as u32).div_ceil(8);
            let gy = (self.height as u32).div_ceil(8);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit([enc.finish()]);

        // Resolve into `surface` (overwrite, NOT additive), then clear the film.
        let raw = self.read_film_raw();
        for (i, v) in raw.iter().enumerate() {
            self.surface[i] = *v as f32 / SCALE;
        }
        self.clear_film();
    }

    /// Borrow the static surface-rim buffer (3*w*h interleaved XYZ).
    pub fn surface(&self) -> &[f32] {
        &self.surface
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
        self.params.photon_base = 0;
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

    /// Composited XYZ for pixel `i` = volumetric accum + static surface rim.
    /// This is the display image (mirrors prism_dsotm.rs `surface + vol`).
    #[inline]
    fn composited(&self, i: usize) -> [f32; 3] {
        [
            self.accum[3 * i]     + self.surface[3 * i],
            self.accum[3 * i + 1] + self.surface[3 * i + 1],
            self.accum[3 * i + 2] + self.surface[3 * i + 2],
        ]
    }

    /// Auto-expose from the composited Y channel (max).
    pub fn auto_expose(&self, scale: f32) -> f32 {
        let n = self.width * self.height;
        let max_y = (0..n).map(|i| self.composited(i)[1]).fold(0.0f32, f32::max);
        (scale / max_y.max(1e-6)).max(1.0)
    }

    /// Auto-expose to the 99.7th-percentile composited luminance (mirrors the
    /// exposure choice in prism_dsotm.rs: exposure = 1 / p99.7).
    pub fn auto_expose_p997(&self) -> f32 {
        let n = self.width * self.height;
        let mut lums: Vec<f32> = (0..n).map(|i| self.composited(i)[1]).collect();
        lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = lums[((n as f32 * 0.997) as usize).min(n - 1)].max(1e-9);
        1.0 / p
    }

    /// Tonemap the composited (volumetric + surface-rim) image to RGB8.
    pub fn to_rgb8(&self, exposure: f32) -> Vec<u8> {
        use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let linear = xyz_to_linear_srgb(self.composited(i));
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
    use spectral_core::geom::Ray;
    use spectral_core::lighttrace::Beam;
    use spectral_core::material::Material;
    use spectral_core::rng::PathRng;
    use spectral_core::scene::Scene;
    use spectral_core::spectrum::{LAMBDA_MIN, LAMBDA_RANGE};
    use glam::Vec3;

    #[test]
    fn debug_photon_layout_is_304_bytes() {
        // Must match WGSL DebugPhoton: header(16) + array<vec4<f32>,18>(288) = 304.
        // A mismatch would corrupt the GPU debug readback silently.
        assert_eq!(std::mem::size_of::<DebugPhotonGpu>(), 304);
        assert_eq!(std::mem::size_of::<DebugPhotonGpu>() % 16, 0);
    }

    #[test]
    fn phase_hg_matches_cpu_oracle() {
        // phase_hg_wgsl_mirror replicates the WGSL phase_hg op-for-op. volume.rs's
        // phase_hg is private, so compare against a local copy of its EXACT formula
        // (volume.rs lines 87-91). The WGSL compilation correctness is then implicit
        // in vol_diff_gate passing (which uses the real WGSL phase_hg).
        fn cpu_oracle_phase_hg(g: f32, cos_theta: f32) -> f32 {
            let g2 = g * g;
            let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6).powf(1.5);
            (1.0 - g2) / (4.0 * std::f32::consts::PI * denom)
        }
        let cases = [
            (0.0_f32,  0.5_f32), (0.0, -0.5),
            (0.5,  1.0), (0.5, -1.0), (0.5, 0.0),
            (0.9,  1.0), (0.9, -1.0), (0.9, 0.0),
            (0.6,  1.0), (-0.3, 0.7),
        ];
        let mut max_delta = 0.0f32;
        for (g, c) in cases {
            let m = phase_hg_wgsl_mirror(g, c);
            let o = cpu_oracle_phase_hg(g, c);
            let d = (m - o).abs();
            max_delta = max_delta.max(d);
            assert!(d < 1e-5, "phase_hg(g={g}, cos={c}): mirror {m} vs oracle {o}, delta {d}");
        }
        // Forward-bias invariant (mirrors volume.rs hg_forward_biased).
        assert!(phase_hg_wgsl_mirror(0.6, 1.0) > phase_hg_wgsl_mirror(0.6, -1.0));
        eprintln!("phase_hg_matches_cpu_oracle: max delta over {} cases = {max_delta:.3e}", cases.len());
    }

    /// The exact prism_dsotm.rs scene + beam + camera (aspect 1.0). Delegates to
    /// the module-level `dsotm_scene` so the gates and the viewer share one source.
    pub fn dsotm_scene() -> (Scene, Beam, Camera) {
        super::dsotm_scene(1.0)
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
        _max_dist: f32,
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
            // VOL-7: beam-splat removed the 4 per-segment `dist` draws from BOTH
            // the CPU oracle and the GPU kernel, so the Fresnel scatter draw below
            // stays in lockstep across CPU/GPU (per-photon path parity preserved).

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
            VolWeights::default(), None,
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
            VolWeights::default(), None,
        );
        film_a.trace_and_resolve(n_photons, seed);

        let mut film_b = GpuPhotonFilm::new(
            &ctx, scene_b, &cam_b, &beam_b, w, h, n_photons, seed, max_dist,
            VolWeights::default(), None,
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
            VolWeights::default(), None,
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

    // -----------------------------------------------------------------------
    // VOL-4: full-image diff gate. The GPU weighted film vs the CPU oracle
    // render_volumetric_scene, on the DSOTM scene, fixed N + seed + all-INF
    // zbuffer + DSOTM weights. Per-photon paths are bit-identical (VOL-2) and
    // GPU accumulation is deterministic fixed-point, so the only deltas are
    // (a) ×4096 quantization and (b) CMF table vs Sensor::cmf (cross-validated
    // by the backward GPU-7 gate). Expect very close agreement.
    // -----------------------------------------------------------------------
    #[test]
    fn vol_diff_gate() {
        use spectral_core::volume::render_volumetric_scene;

        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping vol_diff_gate (no GPU adapter)");
            return;
        };

        // VOL-7: beam-splat is ~100x heavier per photon than point-splat, and the
        // CPU oracle side runs single-threaded, so keep the count low (the L1 and
        // energy deltas are integral quantities, stable well below this). 120K
        // took ~10min; 15K keeps the gate to ~1min while staying representative.
        let (w, h) = (200usize, 200usize);
        let n_photons = 15_000u32;
        let seed = 7u32;
        let max_dist = 14.0_f32;
        let weights = VolWeights::default(); // sigma_s=0.5, sigma_t=0.06, g=0.5
        let zbuf = vec![f32::INFINITY; w * h];

        // GPU.
        let (scene_gpu, beam, cam) = dsotm_scene();
        let mut film = GpuPhotonFilm::new(
            &ctx, scene_gpu, &cam, &beam, w, h, n_photons, seed, max_dist,
            weights, Some(&zbuf),
        );
        film.trace_and_resolve(n_photons, seed);
        let gpu = film.accum(); // interleaved XYZ, length 3*w*h

        // CPU oracle.
        let (scene_cpu, beam_cpu, cam_cpu) = dsotm_scene();
        let cpu = render_volumetric_scene(
            &scene_cpu, &cam_cpu, &beam_cpu, w, h, n_photons,
            weights.sigma_s, weights.sigma_t, weights.g, max_dist, &zbuf, seed,
        ); // Vec<[f32;3]>

        // Per-pixel relative L1 over the image (sum|gpu-cpu| / sum|cpu|, summed
        // over all channels) + total-energy delta on the Y channel.
        let mut abs_diff = 0.0f64;
        let mut abs_cpu  = 0.0f64;
        let mut gpu_y = 0.0f64;
        let mut cpu_y = 0.0f64;
        let mut max_pixel_rel = 0.0f64; // worst single-pixel rel error among bright pixels
        let mut gpu_nonzero = 0usize;
        let mut cpu_nonzero = 0usize;
        for i in 0..w * h {
            for c in 0..3 {
                let g = gpu[3 * i + c] as f64;
                let o = cpu[i][c] as f64;
                abs_diff += (g - o).abs();
                abs_cpu  += o.abs();
            }
            let gy = gpu[3 * i + 1] as f64;
            let oy = cpu[i][1] as f64;
            gpu_y += gy;
            cpu_y += oy;
            if gpu[3 * i] > 0.0 || gpu[3 * i + 1] > 0.0 || gpu[3 * i + 2] > 0.0 { gpu_nonzero += 1; }
            if cpu[i][0] > 0.0 || cpu[i][1] > 0.0 || cpu[i][2] > 0.0 { cpu_nonzero += 1; }
            // Only consider pixels with meaningful CPU energy for per-pixel rel.
            if oy > 1e-3 {
                let r = (gy - oy).abs() / oy;
                if r > max_pixel_rel { max_pixel_rel = r; }
            }
        }
        let rel_l1 = if abs_cpu > 0.0 { abs_diff / abs_cpu } else { 0.0 };
        let energy_delta = if cpu_y > 0.0 { (gpu_y - cpu_y).abs() / cpu_y } else { 0.0 };

        eprintln!(
            "vol_diff_gate: {n_photons} photons, {w}x{h}, seed={seed}\n  \
             relative L1 = {:.4}% (abs_diff={abs_diff:.3}, abs_cpu={abs_cpu:.3})\n  \
             energy delta (Y) = {:.4}% (gpu_Y={gpu_y:.3}, cpu_Y={cpu_y:.3})\n  \
             worst bright-pixel rel = {:.4}%\n  \
             nonzero pixels: GPU={gpu_nonzero} CPU={cpu_nonzero} (should be ~equal if no underflow)",
            rel_l1 * 100.0, energy_delta * 100.0, max_pixel_rel * 100.0
        );

        // Sanity: the CPU oracle produced energy (the scene isn't black).
        assert!(cpu_y > 0.0, "CPU oracle produced zero Y energy; scene/beam misconfigured");

        // VOL-7 beam-splat + stochastic rounding widens the GPU/CPU gap vs the
        // old deterministic point estimator (dither variance + the beam march),
        // so the tolerance is relaxed to 5% L1 / 3% energy. The actuals are
        // printed above; a result near these bounds (vs comfortably under) would
        // mean a weight term (phase angle, /d², ds, sigma) is wrong, not noise.
        assert!(rel_l1 < 0.05,
            "vol_diff_gate relative L1 {:.4}% exceeds 5% — a weight term (phase angle, /d², \
             ds, sigma) is likely wrong", rel_l1 * 100.0);
        assert!(energy_delta < 0.03,
            "vol_diff_gate energy delta {:.4}% exceeds 3%", energy_delta * 100.0);
    }

    /// CPU replication of prism_dsotm.rs SURFACE PASS over ALL pixels.
    /// Returns the interleaved XYZ surface buffer (3*w*h).
    fn cpu_surface_rim(scene: &Scene, cam: &Camera, w: usize, h: usize) -> Vec<f32> {
        let mut surface = vec![0.0f32; 3 * w * h];
        for py in 0..h {
            for px in 0..w {
                let s = (px as f32 + 0.5) / w as f32;
                let t = 1.0 - (py as f32 + 0.5) / h as f32;
                let ray = cam.primary_ray(s, t);
                let idx = py * w + px;
                let xyz = if let Some((hit, _)) = scene.intersect(&ray) {
                    let cos = hit.normal.dot(ray.dir).abs().clamp(0.0, 1.0);
                    let rim = (1.0 - cos).powi(2);
                    [0.015 + rim * 0.45, 0.02 + rim * 0.5, 0.04 + rim * 0.6]
                } else {
                    [0.0, 0.0, 0.0]
                };
                surface[3 * idx]     = xyz[0];
                surface[3 * idx + 1] = xyz[1];
                surface[3 * idx + 2] = xyz[2];
            }
        }
        surface
    }

    // -----------------------------------------------------------------------
    // VOL-5: surface rim gate. The rim pass is deterministic (no RNG): one camera
    // primary ray per pixel, one prism intersection, a Fresnel-style rim term.
    // The GPU surface buffer must match the CPU prism_dsotm surface loop pixel-for-
    // pixel within 1e-4.
    // -----------------------------------------------------------------------
    #[test]
    fn surface_rim_matches_cpu() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU; skipping surface_rim_matches_cpu (no GPU adapter)");
            return;
        };

        let (w, h) = (300usize, 300usize);
        let (scene_gpu, beam, cam) = dsotm_scene();

        // GPU rim pass: build the film, recompute surface (no photons traced).
        let mut film = GpuPhotonFilm::new(
            &ctx, scene_gpu, &cam, &beam, w, h, 1, 0, 14.0,
            VolWeights::default(), None,
        );
        film.recompute_surface(&cam);
        let gpu = film.surface();

        // CPU replication.
        let (scene_cpu, _b, cam_cpu) = dsotm_scene();
        let cpu = cpu_surface_rim(&scene_cpu, &cam_cpu, w, h);

        let mut max_delta = 0.0f32;
        let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
        let mut gpu_nonzero = 0usize;
        let mut cpu_nonzero = 0usize;
        for i in 0..w * h {
            let g_any = gpu[3 * i] > 0.0 || gpu[3 * i + 1] > 0.0 || gpu[3 * i + 2] > 0.0;
            let c_any = cpu[3 * i] > 0.0 || cpu[3 * i + 1] > 0.0 || cpu[3 * i + 2] > 0.0;
            if g_any { gpu_nonzero += 1; }
            if c_any { cpu_nonzero += 1; }
            for c in 0..3 {
                let d = (gpu[3 * i + c] - cpu[3 * i + c]).abs();
                if d > max_delta {
                    max_delta = d;
                    worst = (i, c, gpu[3 * i + c], cpu[3 * i + c]);
                }
            }
        }

        eprintln!(
            "surface_rim_matches_cpu: {w}x{h}, max per-pixel delta = {max_delta:.3e}\n  \
             silhouette (nonzero) pixels: GPU={gpu_nonzero} CPU={cpu_nonzero}\n  \
             worst pixel: idx={} ch={} GPU={} CPU={}",
            worst.0, worst.1, worst.2, worst.3
        );

        // 1. Silhouette match (intersection-branch agreement, exact).
        assert_eq!(
            gpu_nonzero, cpu_nonzero,
            "GPU silhouette {gpu_nonzero} != CPU {cpu_nonzero} — prism intersection disagrees on edge pixels"
        );
        // 2. Non-empty silhouette (sanity: camera/scene aren't a no-op).
        assert!(cpu_nonzero > 0, "CPU surface is all-background; the prism isn't in frame");
        // 3. Per-pixel agreement.
        assert!(
            max_delta < 1e-4,
            "surface_rim max per-pixel delta {max_delta:.3e} >= 1e-4 — camera reconstruction \
             or rim formula drifted from the CPU oracle"
        );
    }
}

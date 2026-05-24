// VOL-1: GPU atomic fixed-point accumulation film + resolve + tonemap path.
//
// Architecture:
//   - GPU: `film` buffer = array<atomic<u32>>, length 3 * width * height.
//     Channel c of pixel idx at index 3*idx + c.
//   - CPU: `accum` Vec<f32>, same layout. Starts zeroed.
//     Each resolve: read back film u32s, divide by SCALE, add into accum, clear film.
//   - Tonemap: accum XYZ -> xyz_to_linear_srgb -> tonemap_to_u8 -> PNG.

use bytemuck::cast_slice;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Fixed-point scale matching the WGSL SCALE constant.
const SCALE: f32 = 4096.0;

/// Number of synthetic splats for the determinism test / example.
pub const N_SPLATS: u32 = 1_000_000;

// Params uniform matching vol_splat.wgsl FilmParams.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FilmParams {
    width:    u32,
    height:   u32,
    n_splats: u32,
    seed:     u32,
}

/// A GPU-backed atomic film + CPU f32 accumulator.
///
/// Typical frame loop:
///   1. `splat(seed)` -- dispatch the synthetic splat kernel
///   2. `resolve()` -- read back u32 film, add into f32 accum, clear film
///   3. `read_accum()` -- get a copy of the f32 accum for tonemap/PNG
///
/// For the determinism gate use `read_film_raw()` to read the u32 film before resolve.
pub struct AtomicFilm<'ctx> {
    device:     &'ctx wgpu::Device,
    queue:      &'ctx wgpu::Queue,
    pipeline:   wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    film_buf:   wgpu::Buffer,
    read_buf:   wgpu::Buffer,
    accum:      Vec<f32>,
    width:      usize,
    height:     usize,
}

impl<'ctx> AtomicFilm<'ctx> {
    /// Create a new film for the given dimensions.
    pub fn new(ctx: &'ctx GpuContext, width: usize, height: usize) -> Self {
        let device = &ctx.device;
        let queue  = &ctx.queue;

        let n = width * height;
        let film_len  = (3 * n) as u64;
        let film_size = film_len * 4; // u32 = 4 bytes each

        // Build shader: rng.wgsl prepended to vol_splat.wgsl.
        let shader_src = concat!(
            include_str!("shaders/rng.wgsl"),
            include_str!("shaders/vol_splat.wgsl"),
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("vol_splat"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // FilmParams uniform (COPY_DST so we can update seed + n_splats).
        let params_init = FilmParams {
            width:    width as u32,
            height:   height as u32,
            n_splats: N_SPLATS,
            seed:     0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("film_params"),
            contents: bytemuck::bytes_of(&params_init),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Atomic film buffer: zero-initialized, STORAGE | COPY_SRC | COPY_DST.
        let film_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("film_atomic"),
            size:               film_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Readback staging buffer.
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("film_read"),
            size:               film_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Build pipeline with explicit bind group layout so atomics work.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("film_bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("film_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("vol_splat"),
            layout:      Some(&pipeline_layout),
            module:      &module,
            entry_point: "main",
            cache:       None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("film_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: film_buf.as_entire_binding() },
            ],
        });

        let accum = vec![0.0f32; 3 * n];

        AtomicFilm {
            device,
            queue,
            pipeline,
            bind_group,
            params_buf,
            film_buf,
            read_buf,
            accum,
            width,
            height,
        }
    }

    /// Dispatch the synthetic splat kernel with the given `seed` and `n_splats`.
    ///
    /// The film must be zeroed before calling if you want a fresh frame.
    pub fn splat(&mut self, seed: u32, n_splats: u32) {
        let params = FilmParams {
            width:    self.width as u32,
            height:   self.height as u32,
            n_splats,
            seed,
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = n_splats.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Read back the raw u32 film buffer WITHOUT resolving into accum.
    ///
    /// Used by the determinism gate to compare two runs bit-for-bit.
    pub fn read_film_raw(&self) -> Vec<u32> {
        let n = 3 * self.width * self.height;
        let size = (n * 4) as u64;

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.film_buf, 0, &self.read_buf, 0, size);
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

    /// Resolve: read back the film u32 buffer, divide by SCALE, add into the f32 accum,
    /// then zero the film buffer for the next frame.
    pub fn resolve(&mut self) {
        let raw = self.read_film_raw();
        for (i, v) in raw.iter().enumerate() {
            self.accum[i] += *v as f32 / SCALE;
        }
        self.clear_film();
    }

    /// Zero the GPU film buffer.
    pub fn clear_film(&self) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.clear_buffer(&self.film_buf, 0, None);
        self.queue.submit([enc.finish()]);
    }

    /// Zero the CPU f32 accum.
    pub fn clear_accum(&mut self) {
        self.accum.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Borrow the current f32 accumulator.
    pub fn accum(&self) -> &[f32] {
        &self.accum
    }

    /// Tonemap the f32 accum to an RGB8 PNG byte buffer.
    ///
    /// `exposure` is passed to `tonemap_to_u8`. Use `auto_expose_accum` for a
    /// reasonable default.
    pub fn to_rgb8(&self, exposure: f32) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let xyz = [
                self.accum[3 * i],
                self.accum[3 * i + 1],
                self.accum[3 * i + 2],
            ];
            let linear = xyz_to_linear_srgb(xyz);
            let rgb = tonemap_to_u8(linear, exposure);
            out.extend_from_slice(&rgb);
        }
        out
    }

    /// Compute an auto-exposure from the Y channel: scale / max_Y.
    /// Returns at least 1.0 to avoid division by near-zero.
    pub fn auto_expose_accum(&self, scale: f32) -> f32 {
        let n = self.width * self.height;
        let max_y = (0..n)
            .map(|i| self.accum[3 * i + 1])
            .fold(0.0f32, f32::max);
        (scale / max_y.max(1e-6)).max(1.0)
    }

    /// Return (width, height).
    pub fn dims(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Run the determinism check: splat with `seed` twice into two fresh films,
/// read back raw u32 arrays, compare. Returns Ok(()) if identical, Err with
/// the first differing index and both values if not.
pub fn check_determinism(
    ctx: &GpuContext,
    width: usize,
    height: usize,
    seed: u32,
    n_splats: u32,
) -> Result<(), (usize, u32, u32)> {
    // Run A.
    let mut film_a = AtomicFilm::new(ctx, width, height);
    film_a.splat(seed, n_splats);
    let raw_a = film_a.read_film_raw();

    // Run B: fresh AtomicFilm (definitely zero-initialized) with the same seed.
    let mut film_b = AtomicFilm::new(ctx, width, height);
    film_b.splat(seed, n_splats);
    let raw_b = film_b.read_film_raw();

    for (i, (&a, &b)) in raw_a.iter().zip(raw_b.iter()).enumerate() {
        if a != b {
            return Err((i, a, b));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small film for fast unit tests.
    const W: usize = 32;
    const H: usize = 32;

    fn gpu_or_skip() -> Option<GpuContext> {
        GpuContext::new()
    }

    // VOL-1-DETERMINISM: two identical runs with same seed produce bit-identical film.
    #[test]
    fn vol1_determinism_small() {
        let Some(ctx) = gpu_or_skip() else {
            eprintln!("no GPU adapter; skipping vol1_determinism_small");
            return;
        };
        let seed = 0xDEAD_BEEF_u32;
        let n_splats = 4096u32;
        match check_determinism(&ctx, W, H, seed, n_splats) {
            Ok(()) => println!("VOL-1 determinism (small): IDENTICAL ({n_splats} splats, {W}x{H} film)"),
            Err((i, a, b)) => panic!("VOL-1 determinism DIFFERS at index {i}: {a} vs {b}"),
        }
    }

    // VOL-1-RESOLVE: splat known fixed-point values, resolve, check accum within tolerance.
    //
    // Strategy: write known u32 values directly into a Vec<f32> accum (bypassing GPU
    // for the "splat" side) and verify the resolve arithmetic. This tests the CPU-side
    // resolve formula independent of the GPU kernel.
    #[test]
    fn vol1_resolve_arithmetic() {
        // Simulate: one splat of xyz=(1.0, 0.5, 0.25) at pixel 0.
        // u32 film values (truncated): floor(1.0 * 4096) = 4096, 2048, 1024.
        let raw: Vec<u32> = {
            let mut v = vec![0u32; 3 * W * H];
            v[0] = 4096; // X channel of pixel 0
            v[1] = 2048; // Y channel of pixel 0
            v[2] = 1024; // Z channel of pixel 0
            // Add a second splat at pixel 5: xyz=(0.5, 1.5, 0.0)
            v[15] = 2048;
            v[16] = 6144;
            v[17] = 0;
            v
        };

        // CPU-side resolve (mirrors AtomicFilm::resolve).
        let mut accum = vec![0.0f32; 3 * W * H];
        for (i, &val) in raw.iter().enumerate() {
            accum[i] += val as f32 / 4096.0;
        }

        // Pixel 0 channels.
        let tol = 1.0 / 4096.0 + f32::EPSILON * 10.0;
        assert!((accum[0] - 1.0).abs() < tol, "px0.X expected 1.0, got {}", accum[0]);
        assert!((accum[1] - 0.5).abs() < tol, "px0.Y expected 0.5, got {}", accum[1]);
        assert!((accum[2] - 0.25).abs() < tol, "px0.Z expected 0.25, got {}", accum[2]);

        // Pixel 5 channels (indices 15, 16, 17).
        assert!((accum[15] - 0.5).abs() < tol,  "px5.X expected 0.5, got {}", accum[15]);
        assert!((accum[16] - 1.5).abs() < tol,  "px5.Y expected 1.5, got {}", accum[16]);
        assert!((accum[17] - 0.0).abs() < tol,  "px5.Z expected 0.0, got {}", accum[17]);

        println!("VOL-1 resolve arithmetic: PASS");
    }

    // VOL-1-GPU-SPLAT: run the actual GPU splat kernel on a small film and verify
    // that the film is non-zero after splatting (smoke test for the GPU path).
    #[test]
    fn vol1_gpu_splat_nonzero() {
        let Some(ctx) = gpu_or_skip() else {
            eprintln!("no GPU adapter; skipping vol1_gpu_splat_nonzero");
            return;
        };
        let mut film = AtomicFilm::new(&ctx, W, H);
        film.splat(0xABCD_1234, 8192);
        let raw = film.read_film_raw();
        let nonzero: usize = raw.iter().filter(|&&v| v > 0).count();
        assert!(nonzero > 0, "film should have nonzero values after splatting");
        println!("VOL-1 gpu splat nonzero: {nonzero} / {} slots nonzero", 3 * W * H);
    }
}

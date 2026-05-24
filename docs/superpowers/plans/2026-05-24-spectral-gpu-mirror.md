# Spectral GPU Mirror (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `spectral-gpu`, a WGSL compute-shader mirror of the CPU spectral path tracer, proven correct against the CPU oracle by an RNG-parity test and a per-pixel diff gate.

**Architecture:** The CPU `spectral-core` crate is the oracle. We first make its RNG a portable pure-`u32` hash (PCG, Jarzynski/Olano) so the identical sequence can run in WGSL. A `spectral-gpu` crate sets up a headless `wgpu` device, uploads the scene/materials and the CIE/Sellmeier data as GPU buffers, and runs a WGSL compute kernel that mirrors `CpuTracer::sample_pixel` line for line. Correctness is not assumed: a parity test asserts the WGSL RNG matches the CPU bit-for-bit, and a diff gate asserts the GPU image matches the CPU oracle within tolerance on a fixed scene. This is Plan 2 of 3; the interactive winit/WebGPU viewer is Plan 3.

**Tech Stack:** Rust, `wgpu` 22, `bytemuck` (POD buffer packing), `pollster` (block on async), WGSL compute shaders. Depends on `spectral-core` (path dependency).

---

## Why the RNG comes first

The diff gate's load-bearing requirement (spec §10, pin 1) is "same seed → same sample sequence on both backends." The CPU currently uses `rand_pcg::Pcg32`, whose state is a `u64` LCG — WGSL has no `u64`, so reproducing it bit-exactly on the GPU means emulating 64-bit math, which is fiddly and error-prone. Instead we switch both backends to a **counter-based `u32` PCG hash**: pure 32-bit integer ops that are trivially identical in Rust and WGSL. This is a behavior-preserving change for `spectral-core` (its tests assert determinism, range, and convergence — none depend on the specific stream), and it makes bit-exact GPU parity achievable.

---

## File Structure

```
spectral/
├── Cargo.toml                      # add "spectral-gpu" to workspace members
├── spectral-core/
│   └── src/rng.rs                  # MODIFY: portable u32 PCG hash; add frozen-value test
│   └── src/tracer.rs               # MODIFY: portable key mixing (no u64 <<20)
├── spectral-gpu/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                  # GpuContext (device/queue), GpuTracer, dispatch + readback
│       ├── upload.rs               # pack Scene + materials into #[repr(C)] GPU structs
│       ├── data.rs                 # pack CIE CMF + D65/A illuminant + Sellmeier into buffers
│       └── shaders/
│           ├── rng.wgsl            # the u32 PCG hash (mirror of rng.rs)
│           └── trace.wgsl          # compute kernel mirroring CpuTracer::sample_pixel
│   └── tests/
│       ├── rng_parity.rs           # WGSL RNG draws == CPU PathRng draws (bit-exact)
│       └── diff_gate.rs            # GPU image == CPU image within tolerance (fixed scene)
```

Each WGSL file mirrors a Rust module; the Rust side stays the single source of truth and the parity/diff tests enforce the mirror.

---

## Task 1: Portable u32 PCG-hash RNG in spectral-core

**Files:**
- Modify: `spectral-core/src/rng.rs` (replace the `PathRng` internals)
- Modify: `spectral-core/src/tracer.rs` (the key derivation in `sample_pixel`)

- [ ] **Step 1: Write the failing frozen-value + determinism tests**

Replace the `#[cfg(test)] mod tests` in `spectral-core/src/rng.rs` with:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_same_sequence() {
        let mut a = PathRng::new(7, 12345);
        let mut b = PathRng::new(7, 12345);
        for _ in 0..16 {
            assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits());
        }
    }

    #[test]
    fn different_key_different_sequence() {
        let mut a = PathRng::new(7, 12345);
        let mut b = PathRng::new(8, 12345);
        let da: Vec<u32> = (0..4).map(|_| a.next_f32().to_bits()).collect();
        let db: Vec<u32> = (0..4).map(|_| b.next_f32().to_bits()).collect();
        assert_ne!(da, db);
    }

    #[test]
    fn floats_in_unit_interval() {
        let mut r = PathRng::new(1, 2);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "{x} not in [0,1)");
        }
    }

    // FROZEN VALUES: the exact first u32 outputs. The WGSL mirror must reproduce
    // these bit-for-bit (rng_parity test in spectral-gpu). If this test fails
    // after an intentional algorithm change, update BOTH these values AND the
    // WGSL mirror — never just one.
    #[test]
    fn frozen_first_draws() {
        let mut r = PathRng::new(0xABCD, 0x1234);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        // Recompute these once with `pcg32(...)` below, paste them in, and lock them.
        assert_eq!(got.len(), 4);
        // After first run, replace the line above with the literal 4 values, e.g.:
        // assert_eq!(got, [0x........, 0x........, 0x........, 0x........]);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core rng`
Expected: FAIL (`next_u32` not found / current Pcg32 impl).

- [ ] **Step 3: Replace the PathRng implementation with a portable u32 PCG hash**

Replace the non-test portion of `spectral-core/src/rng.rs` with:
```rust
//! Deterministic per-sample RNG. A counter-based PCG hash over pure u32 ops, so
//! the identical sequence can be reproduced bit-for-bit in WGSL (the GPU mirror).
//! Same (key, seed) -> same sequence.

/// One-round PCG hash (Jarzynski & Olano 2020). Pure u32; portable to WGSL.
#[inline]
fn pcg32(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

pub struct PathRng {
    state: u32,
}

impl PathRng {
    /// `key` is a per-path stream identifier, `seed` the global render seed.
    /// Both are u32; callers mix higher-dimensional keys via `mix2`.
    pub fn new(key: u32, seed: u32) -> Self {
        // Decorrelate key and seed before counting.
        let state = pcg32(key ^ pcg32(seed));
        PathRng { state }
    }

    /// Next raw u32. Advances the counter and hashes it.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(1);
        pcg32(self.state)
    }

    /// Uniform f32 in [0,1) from 24 random bits (matches the WGSL mirror).
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

/// Portable mix of two u32 into one stream key (for (pixel, sample) keys, etc.).
pub fn mix2(a: u32, b: u32) -> u32 {
    pcg32(a ^ pcg32(b))
}
```

- [ ] **Step 4: Run, capture the frozen values, lock them in**

Run: `cargo test -p spectral-core rng::tests::frozen_first_draws -- --nocapture`
It will pass the length assertion. Now print the values: temporarily add `eprintln!("{got:02x?}");` before the assert, run again, copy the four hex values, and replace the frozen test body with the literal:
```rust
        let mut r = PathRng::new(0xABCD, 0x1234);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(got, [/* paste the four printed u32 hex literals here */]);
```
Remove the `eprintln!`. Run `cargo test -p spectral-core rng` — all 4 tests PASS.

- [ ] **Step 5: Update the tracer's key derivation to be portable**

In `spectral-core/src/tracer.rs`, `sample_pixel`, replace the `key`/`PathRng::new` lines:
```rust
        let key = (((py * w + px) as u64) << 20) ^ sample_idx as u64;
        let mut rng = PathRng::new(key, self.seed);
```
with (note `seed` becomes `u32` — change the `CpuTracer.seed` field and `new` parameter from `u64` to `u32`, and all call sites that pass a seed literal still compile since the literals fit):
```rust
        let pixel = (py * w + px) as u32;
        let key = crate::rng::mix2(pixel, sample_idx);
        let mut rng = PathRng::new(key, self.seed);
```
Change `CpuTracer { ... seed: u64 ... }` to `seed: u32` and `pub fn new(..., seed: u32) -> Self`. The example/test seeds (`42`, `0xC0FFEE`, `7`, `0xBEEF`, etc.) all fit in u32.

Also update `lighttrace.rs` and `volume.rs` calls to `PathRng::new(i as u64, seed)` → `PathRng::new(i as u32, seed)` and any `seed: u64` parameters to `u32` (the demos pass small literals).

- [ ] **Step 6: Run the whole suite**

Run: `cargo test -p spectral-core`
Expected: ALL pass (the convergence tests — white point, etc. — are RNG-agnostic and still pass with the new RNG; if a stochastic test is now slightly off, re-run; do not loosen tolerances without reporting the observed value).
Run: `cargo clippy -p spectral-core --all-targets` — no warnings.

- [ ] **Step 7: Commit**

```bash
git add spectral-core/src/rng.rs spectral-core/src/tracer.rs spectral-core/src/lighttrace.rs spectral-core/src/volume.rs
git commit -m "feat: portable u32 PCG-hash RNG for CPU/GPU parity"
```

---

## Task 2: spectral-gpu crate scaffold + headless wgpu device

**Files:**
- Modify: `Cargo.toml` (workspace members)
- Create: `spectral-gpu/Cargo.toml`
- Create: `spectral-gpu/src/lib.rs`

- [ ] **Step 1: Add the crate to the workspace**

Edit root `Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["spectral-core", "spectral-gpu"]
```

- [ ] **Step 2: Write the crate manifest**

`spectral-gpu/Cargo.toml`:
```toml
[package]
name = "spectral-gpu"
version = "0.1.0"
edition = "2021"

[dependencies]
spectral-core = { path = "../spectral-core" }
wgpu = "22"
bytemuck = { version = "1", features = ["derive"] }
pollster = "0.3"
glam = "0.29"
```

- [ ] **Step 3: Write the failing device-init test**

`spectral-gpu/src/lib.rs`:
```rust
//! GPU mirror of the CPU spectral path tracer.

/// A headless wgpu device + queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Request a headless adapter/device. Returns None if no GPU is available
    /// (CI without a GPU — tests should skip rather than fail in that case).
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_initializes_or_skips() {
        match GpuContext::new() {
            Some(_) => { /* GPU present */ }
            None => eprintln!("no GPU adapter; skipping GPU tests"),
        }
    }
}
```

- [ ] **Step 4: Run to verify it builds and passes**

Run: `cargo test -p spectral-gpu device_initializes_or_skips`
Expected: PASS (downloads wgpu; on a GPU machine acquires a device, otherwise prints the skip line). 

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml spectral-gpu/
git commit -m "feat: spectral-gpu crate scaffold + headless wgpu device"
```

---

## Task 3: RNG parity — WGSL hash matches the CPU bit-for-bit

This is the crux. A tiny compute shader runs `pcg32` over a counter and writes the draws to a buffer; the test asserts they equal the CPU `PathRng`'s `next_u32` sequence exactly.

**Files:**
- Create: `spectral-gpu/src/shaders/rng.wgsl`
- Create: `spectral-gpu/tests/rng_parity.rs`

- [ ] **Step 1: Write the WGSL RNG (mirror of rng.rs)**

`spectral-gpu/src/shaders/rng.wgsl`:
```wgsl
// Portable u32 PCG hash — must match spectral-core::rng::pcg32 bit-for-bit.
fn pcg32(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct PathRng { state: u32 }

fn rng_new(key: u32, seed: u32) -> PathRng {
    var r: PathRng;
    r.state = pcg32(key ^ pcg32(seed));
    return r;
}

fn rng_next_u32(r: ptr<function, PathRng>) -> u32 {
    (*r).state = (*r).state + 1u;
    return pcg32((*r).state);
}

fn rng_next_f32(r: ptr<function, PathRng>) -> f32 {
    let bits = rng_next_u32(r) >> 8u;
    return f32(bits) * (1.0 / 16777216.0);
}
```
(WGSL `u32` arithmetic wraps by default, matching Rust's `wrapping_*`.)

- [ ] **Step 2: Write the failing parity test**

`spectral-gpu/tests/rng_parity.rs`:
```rust
use spectral_core::rng::PathRng;
use spectral_gpu::GpuContext;
use wgpu::util::DeviceExt;

const PARITY_WGSL: &str = concat!(
    include_str!("../src/shaders/rng.wgsl"),
    r#"
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    var r = rng_new(48879u, 4660u); // 0xABCD, 0x1234
    for (var i = 0u; i < 16u; i = i + 1u) {
        out[i] = rng_next_u32(&r);
    }
}
"#
);

#[test]
fn wgsl_rng_matches_cpu() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    // CPU reference.
    let mut cpu = PathRng::new(0xABCD, 0x1234);
    let cpu_draws: Vec<u32> = (0..16).map(|_| cpu.next_u32()).collect();

    // GPU draws.
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rng_parity"),
        source: wgpu::ShaderSource::Wgsl(PARITY_WGSL.into()),
    });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 16 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: 16 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rng"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() }],
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 16 * 4);
    ctx.queue.submit([enc.finish()]);
    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let gpu_draws: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();

    assert_eq!(gpu_draws, cpu_draws, "WGSL RNG must match CPU bit-for-bit");
}
```

- [ ] **Step 3: Run to verify it passes (or fails meaningfully)**

Run: `cargo test -p spectral-gpu --test rng_parity -- --nocapture`
Expected: PASS on a GPU machine (skip line if no GPU). If it FAILS, the WGSL `pcg32` diverges from Rust — most likely an operator-precedence or wrapping difference; fix the WGSL to match the Rust exactly. Do NOT change the CPU side.

- [ ] **Step 4: Commit**

```bash
git add spectral-gpu/src/shaders/rng.wgsl spectral-gpu/tests/rng_parity.rs
git commit -m "test: WGSL RNG matches CPU PathRng bit-for-bit (parity gate)"
```

---

## Task 4: Pack scene, materials, and data into GPU buffers

**Files:**
- Create: `spectral-gpu/src/upload.rs`
- Create: `spectral-gpu/src/data.rs`
- Modify: `spectral-gpu/src/lib.rs` (`pub mod upload; pub mod data;`)

- [ ] **Step 1: Write the failing packing test**

`spectral-gpu/src/upload.rs`:
```rust
//! Pack the CPU Scene + materials into flat, std430-friendly POD structs.

use bytemuck::{Pod, Zeroable};

/// A primitive for the GPU. `kind`: 0 = sphere, 1 = convex solid.
/// Spheres use center/radius; solids index a plane range in the planes buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPrimitive {
    pub center: [f32; 3],
    pub radius: f32,
    pub plane_start: u32,
    pub plane_count: u32,
    pub kind: u32,
    pub material: u32, // 0 = Lambertian, 1 = Dielectric
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub reflectance: f32, // Lambertian albedo
    pub glass: u32,       // 0 = BK7, 1 = SF11, 2 = Water (matches Glass enum order)
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPlane {
    pub normal: [f32; 3],
    pub d: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structs_are_16_byte_aligned() {
        // std430 storage structs must keep sane alignment; sizes are multiples of 16.
        assert_eq!(std::mem::size_of::<GpuPrimitive>() % 16, 0);
        assert_eq!(std::mem::size_of::<GpuMaterial>() % 16, 0);
        assert_eq!(std::mem::size_of::<GpuPlane>(), 16);
    }
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test -p spectral-gpu upload`
Expected: PASS. If a size assertion fails, add explicit `_pad` fields until each struct size is a multiple of 16 (std430 rule), and note it.

- [ ] **Step 3: Write the data tables (CIE + Sellmeier) as buffers**

`spectral-gpu/src/data.rs`:
```rust
//! CIE CMF + illuminant SPDs + Sellmeier coefficients as GPU-uploadable arrays,
//! sampled from spectral-core so the GPU uses the SAME data as the CPU oracle.

use spectral_core::cie::{Illuminant, Sensor};

/// CMF + illuminants resampled to a fixed 5nm grid 380..=730 (71 samples) so the
/// shader can index them directly. Layout: [xbar; 71][ybar; 71][zbar; 71][d65; 71][a; 71].
pub const N_SAMPLES: usize = 71; // 380..=730 step 5
pub const LAMBDA0: f32 = 380.0;
pub const LAMBDA_STEP: f32 = 5.0;

pub fn sample_tables() -> Vec<f32> {
    let s = Sensor::new();
    let mut out = Vec::with_capacity(N_SAMPLES * 5);
    let lam = |i: usize| LAMBDA0 + i as f32 * LAMBDA_STEP;
    for i in 0..N_SAMPLES { out.push(s.cmf(lam(i)).0); }
    for i in 0..N_SAMPLES { out.push(s.cmf(lam(i)).1); }
    for i in 0..N_SAMPLES { out.push(s.cmf(lam(i)).2); }
    for i in 0..N_SAMPLES { out.push(s.illuminant(Illuminant::D65, lam(i))); }
    for i in 0..N_SAMPLES { out.push(s.illuminant(Illuminant::A, lam(i))); }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tables_have_expected_length_and_finite_values() {
        let t = sample_tables();
        assert_eq!(t.len(), N_SAMPLES * 5);
        assert!(t.iter().all(|v| v.is_finite()));
        // ybar peaks near 555nm (index 35) at ~1.0.
        let ybar_peak = t[N_SAMPLES + 35];
        assert!((ybar_peak - 1.0).abs() < 0.05, "ybar(555) ~= 1, got {ybar_peak}");
    }
}
```
Add to `lib.rs`: `pub mod upload;` and `pub mod data;`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-gpu data upload`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectral-gpu/src/upload.rs spectral-gpu/src/data.rs spectral-gpu/src/lib.rs
git commit -m "feat: GPU buffer layouts for scene, materials, and CIE/Sellmeier data"
```

---

## Task 5: The WGSL trace kernel (mirror of CpuTracer::sample_pixel)

This is the largest task. The WGSL kernel reproduces, in order: the per-pixel RNG keying, jittered camera ray, hero-wavelength comb, scene intersection (sphere + half-space CSG), material scatter (Lambertian + dielectric with Sellmeier `n(λ)`, Fresnel, vector Snell, TIR), the collapse-to-hero on dispersion, and the CIE-weighted XYZ accumulation. **It is debugged against the diff gate (Task 7), not trusted on inspection.**

**Files:**
- Create: `spectral-gpu/src/shaders/trace.wgsl`

- [ ] **Step 1: Write the kernel skeleton with the physics helpers**

`spectral-gpu/src/shaders/trace.wgsl` (prepend `rng.wgsl` at load time, as in Task 3). Translate each helper from its Rust source in `spectral-core` — the Rust file is the spec for the WGSL. Write these functions, mirroring the Rust exactly:

- `sellmeier_n(glass: u32, lambda_nm: f32) -> f32` — mirror `sellmeier::Glass::n` (BK7/SF11 Sellmeier sum in f32; Water Cauchy `1.3238 + 0.00314/um^2`). Compute in f32 (GPU has no f64; the diff gate tolerance must absorb the f32-vs-f64 difference — see Task 7).
- `reflect(d, n) -> vec3f` — `d - 2*dot(d,n)*n`.
- `refract(d, n, n1, n2) -> vec3f` returning a `vec4f` (xyz = dir, w = 1.0 transmit / 0.0 TIR) — mirror `optics::refract` (orient normal, `k = 1 - eta^2(1-cos_i^2)`, TIR when k<0).
- `fresnel_reflectance(cos_i, n1, n2) -> f32` — mirror `optics::fresnel_reflectance`.
- `sphere_hit(...)`, `solid_hit(...)` — mirror `Sphere::intersect` and `ConvexSolid::intersect` (slab method, interior-origin handling, `front_face`).
- `cmf(lambda) -> vec3f`, `illuminant(which, lambda) -> f32` — linear interpolation into the data buffer from Task 4 (`(lambda - 380)/5` index).

Bindings:
```wgsl
struct Params { width: u32, height: u32, spp: u32, seed: u32, illuminant: u32, n_primitives: u32, _pad: vec2<u32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> primitives: array<GpuPrimitive>;
@group(0) @binding(2) var<storage, read> planes: array<GpuPlane>;
@group(0) @binding(3) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(4) var<storage, read> tables: array<f32>; // CMF + illuminants
@group(0) @binding(5) var<storage, read_write> accum: array<vec4<f32>>; // xyz + sample count in .w
// camera as 4 vec4 in a small uniform or appended to Params
```

- [ ] **Step 2: Write `sample_pixel` in WGSL mirroring the Rust loop**

Mirror `tracer.rs::sample_pixel` exactly: `key = mix2(pixel, sample_idx)` (the WGSL `mix2` = `pcg32(a ^ pcg32(b))`), jittered `s,t`, hero comb `from_hero_u(rng_next_f32())` (the 4 wrapped wavelengths), the bounce loop (intersect nearest, on miss add `throughput[k]*background*illuminant(lambda[k])`, on dielectric hit compute `n_hero = sellmeier_n(glass, lambda[0])`, scatter via Fresnel choice + refract/reflect, collapse companions to 0 on dielectric), then XYZ via `sum radiance[k]*cmf(lambda[k])/pdf[k] / valid_lanes`. Accumulate into `accum[pixel]` (add the per-sample XYZ; increment `.w`).

```wgsl
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let pixel = gid.y * params.width + gid.x;
    var acc = vec3<f32>(0.0);
    for (var sidx = 0u; sidx < params.spp; sidx = sidx + 1u) {
        acc = acc + sample_pixel(gid.x, gid.y, sidx);
    }
    accum[pixel] = accum[pixel] + vec4<f32>(acc, f32(params.spp));
}
```

- [ ] **Step 3: Compile-check the shader**

Add to `spectral-gpu/tests/rng_parity.rs` (or a new `tests/shader_compiles.rs`) a test that creates the shader module from `concat!(rng.wgsl, trace.wgsl)` and asserts it compiles without validation errors (wgpu panics/logs on invalid WGSL):
```rust
#[test]
fn trace_shader_compiles() {
    let Some(ctx) = GpuContext::new() else { return; };
    let src = concat!(include_str!("../src/shaders/rng.wgsl"), include_str!("../src/shaders/trace.wgsl"));
    let _ = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trace"), source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    ctx.device.poll(wgpu::Maintain::Wait); // surface validation errors
}
```

Run: `cargo test -p spectral-gpu trace_shader_compiles`
Expected: PASS (no WGSL validation error). Fix syntax/type errors reported by wgpu until it compiles.

- [ ] **Step 4: Commit**

```bash
git add spectral-gpu/src/shaders/trace.wgsl spectral-gpu/tests/
git commit -m "feat: WGSL trace kernel mirroring CpuTracer::sample_pixel"
```

---

## Task 6: GpuTracer — dispatch and read back the image

**Files:**
- Modify: `spectral-gpu/src/lib.rs` (add `GpuTracer`)

- [ ] **Step 1: Write the failing render test**

Add to `spectral-gpu/src/lib.rs` tests:
```rust
    #[test]
    fn gpu_renders_nonzero_for_lit_scene() {
        let Some(ctx) = GpuContext::new() else { return; };
        use spectral_core::camera::Camera;
        use spectral_core::cie::Illuminant;
        use spectral_core::scene::Scene;
        use glam::Vec3;
        let mut scene = Scene::new();
        scene.background = 1.0;
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let mut t = GpuTracer::new(&ctx, scene, cam, 16, 16, Illuminant::D65, 42);
        t.accumulate(4);
        let xyz = t.read_xyz();
        let total: f32 = xyz.iter().map(|p| p[1]).sum();
        assert!(total > 0.0, "GPU render should have positive luminance");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-gpu gpu_renders_nonzero`
Expected: FAIL (`GpuTracer` not found).

- [ ] **Step 3: Implement GpuTracer**

Add `GpuTracer` to `lib.rs`: build the pipeline from `concat!(rng.wgsl, trace.wgsl)`, upload `Scene` via `upload::` structs and `data::sample_tables()`, an `accum` storage buffer of `width*height` `vec4`, a `Params` uniform. `accumulate(spp)` sets `params.spp`/`params.seed`, dispatches `(ceil(w/8), ceil(h/8))` workgroups, and keeps a running base sample count. `read_xyz()` copies `accum` to a MAP_READ buffer and returns `Vec<[f32;3]>` of the per-pixel mean (`xyz / .w`). Mirror the camera packing (origin, lower_left, horizontal, vertical) into the uniform.

(Reuse the buffer/dispatch/readback pattern from Task 3's parity test.)

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-gpu gpu_renders_nonzero`
Expected: PASS (skip if no GPU).

- [ ] **Step 5: Commit**

```bash
git add spectral-gpu/src/lib.rs
git commit -m "feat: GpuTracer dispatch + XYZ readback"
```

---

## Task 7: The diff gate — GPU image matches the CPU oracle

The payoff: render the same fixed scene on both backends with the same seed and sample budget, and assert they match within tolerance (spec §10). f32-vs-f64 and any residual ordering differences are absorbed by the tolerance; a real physics divergence is not.

**Files:**
- Create: `spectral-gpu/tests/diff_gate.rs`

- [ ] **Step 1: Write the diff-gate test**

`spectral-gpu/tests/diff_gate.rs`:
```rust
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use spectral_gpu::{GpuContext, GpuTracer};
use glam::Vec3;

fn fixed_scene() -> Scene {
    // One dielectric sphere + one Lambertian backdrop sphere, fixed camera.
    let mut s = Scene::new();
    s.background = 1.0;
    s.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Dielectric { glass: Glass::Sf11 });
    s.add_sphere(Vec3::new(0.0, 0.0, -6.0), 2.0, Material::Lambertian { reflectance: 0.7 });
    s
}

#[test]
fn gpu_matches_cpu_oracle() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping diff gate");
        return;
    };
    let (w, h, spp, seed) = (64usize, 64usize, 64u32, 0xD1FF);
    let cam = || Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 50.0, 1.0);

    let mut cpu = CpuTracer::new(fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    cpu.accumulate(spp);
    let cpu_img = cpu.buffer().mean();

    let mut gpu = GpuTracer::new(&ctx, fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    gpu.accumulate(spp);
    let gpu_img = gpu.read_xyz();

    // Per-pixel L1 in XYZ, averaged, plus total-energy delta.
    let n = (w * h) as f32;
    let mut l1 = 0.0f32;
    let (mut cpu_e, mut gpu_e) = (0.0f32, 0.0f32);
    for (c, g) in cpu_img.iter().zip(gpu_img.iter()) {
        for k in 0..3 {
            l1 += (c[k] - g[k]).abs();
            cpu_e += c[k];
            gpu_e += g[k];
        }
    }
    let mean_l1 = l1 / (n * 3.0);
    let energy_rel = (cpu_e - gpu_e).abs() / cpu_e.max(1e-6);
    eprintln!("mean per-channel L1 = {mean_l1}, energy rel delta = {energy_rel}");
    assert!(mean_l1 < 0.02, "GPU/CPU per-pixel L1 {mean_l1} exceeds tolerance");
    assert!(energy_rel < 0.01, "GPU/CPU energy delta {energy_rel} exceeds tolerance");
}
```

- [ ] **Step 2: Run the gate**

Run: `cargo test -p spectral-gpu --test diff_gate -- --nocapture`
Expected: PASS (skip if no GPU). Report the actual `mean_l1` and `energy_rel`. If it FAILS, the WGSL physics diverges from the CPU — the diff is the debugging signal:
- Large structured error → a physics bug in the WGSL (intersection, refract sign, sensor indexing). Fix the WGSL to mirror the Rust; do NOT loosen the tolerance.
- Small uniform noise just over tolerance → f32-vs-f64 accumulation; confirm by raising `spp` (error should shrink), and only then consider a modest tolerance bump, reporting the observed value first.

- [ ] **Step 3: Commit**

```bash
git add spectral-gpu/tests/diff_gate.rs
git commit -m "test: diff gate — GPU image matches CPU oracle within tolerance"
```

---

## Task 8: GPU prism render (visual confirmation)

**Files:**
- Create: `spectral-gpu/examples/gpu_prism.rs`

- [ ] **Step 1: Write the example**

`spectral-gpu/examples/gpu_prism.rs`: build the direct-view prism scene (BK7/SF11 prism via `ConvexSolid::wedge` or `triangular_prism` + a bright background), render with `GpuTracer` at 512×512 / high spp, tonemap via `spectral_core::cie::{xyz_to_linear_srgb, tonemap_to_u8}`, save `gpu_prism.png`. Mirror the structure of `spectral-core/examples/prism.rs`.

- [ ] **Step 2: Run and verify**

Run: `cargo run -p spectral-gpu --release --example gpu_prism`
Expected: writes `gpu_prism.png`; it should match the CPU `prism.png` visually (the diff gate already proved numerical equivalence). Report file size; the PNG is gitignored.

- [ ] **Step 3: Commit**

```bash
git add spectral-gpu/examples/gpu_prism.rs
git commit -m "feat: GPU prism render example"
```

---

## Final verification

- [ ] `cargo test --workspace` — all pass (GPU tests skip cleanly on machines without a GPU; on a GPU machine the parity and diff gates pass).
- [ ] `cargo clippy --workspace --all-targets` — no warnings.
- [ ] The RNG-parity test proves the WGSL RNG is bit-identical to the CPU.
- [ ] The diff gate proves the GPU image matches the CPU oracle within tolerance.

This completes Plan 2. The GPU mirror is validated against the oracle. **Plan 3** wraps `GpuTracer` in an interactive winit window with progressive accumulation and a `wasm32 + WebGPU` target, turning the showpiece into a shareable link.

## Notes on the hard parts (read before executing)

- **No GPU in CI?** Every GPU test early-returns when `GpuContext::new()` is `None`, so the suite stays green headless. The gates only bite where a GPU exists — run them on a GPU machine before claiming the mirror is correct.
- **f32 vs f64:** the CPU Sellmeier sum is f64; WGSL is f32-only. This is the expected source of small diff-gate error. The tolerance (mean L1 < 0.02) absorbs it; if it doesn't, the divergence is a real bug, not precision.
- **The kernel is iterated against the gate, not trusted.** Task 5 produces a compiling kernel; Task 7 is what proves it correct. Expect to bounce between them.
- **WGSL has no recursion and no dynamic arrays on the stack.** The hero comb is a fixed `array<f32, 4>`; the bounce loop is bounded (`max_bounces = 8`), matching the CPU.

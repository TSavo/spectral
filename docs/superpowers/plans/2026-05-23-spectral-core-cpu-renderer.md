# Spectral Core (CPU Renderer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `spectral-core`, a CPU spectral path tracer that renders dispersion through a glass prism and passes a metamerism check, verified against the closed-form 2D Snell solution.

**Architecture:** Pure-Rust, no-GPU library crate. Light is carried as a 4-wide hero-wavelength sample (radiance + wavelength + pdf per lane) along ray vectors. Refraction uses the vector Snell law with a wavelength-dependent Sellmeier index; color appears only at a CIE sensor. A `Renderer` trait fronts the tracer so the later GPU backend and wave solver attach as siblings. Correctness is asserted against analytic physics (2D Snell, published Sellmeier values, CIE white points), not against appearance.

**Tech Stack:** Rust (2021 edition), `glam` (vector math), `rand` + `rand_pcg` (seedable PCG, parity-friendly), `image` (PNG output), `rayon` (parallel pixels). This is Plan 1 of 3 for Phase 0.

---

## File Structure

```
spectral/
├── Cargo.toml                      # workspace
├── spectral-core/
│   ├── Cargo.toml
│   ├── data/
│   │   ├── cie_1931_2deg.csv        # CMF table, 5nm, from CVRL
│   │   ├── illuminant_d65.csv       # D65 relative SPD, from CIE/CVRL
│   │   └── illuminant_a.csv         # Illuminant A relative SPD
│   └── src/
│       ├── lib.rs                   # re-exports, Renderer trait, Xyz, AccumBuffer
│       ├── spectrum.rs              # band constants, SpectralSample, hero sampling
│       ├── rng.rs                   # seedable PCG with explicit stream keys
│       ├── sellmeier.rs             # n(λ) for BK7, SF11
│       ├── optics.rs                # Fresnel R, vector Snell refract/reflect, TIR
│       ├── cie.rs                   # CMF + illuminant tables, XYZ integration, sRGB, tonemap
│       ├── geom.rs                  # Ray, Hit, Sphere, HalfSpace, ConvexSolid, prism builder
│       ├── material.rs              # Material, BSDF sample/eval
│       ├── scene.rs                 # Scene, Primitive, Emitter, intersect
│       ├── camera.rs                # pinhole camera, ray generation
│       └── tracer.rs                # CpuTracer: path tracing + XYZ accumulation
```

Each module has one responsibility. `optics.rs` and `sellmeier.rs` are the physics that must be analytically correct; their tests are the oracle. `geom.rs`, `material.rs`, `scene.rs`, `camera.rs` are transport plumbing. `tracer.rs` ties them together behind the `Renderer` trait.

---

## Task 1: Workspace and crate skeleton

**Files:**
- Create: `Cargo.toml`
- Create: `spectral-core/Cargo.toml`
- Create: `spectral-core/src/lib.rs`

- [ ] **Step 1: Write the workspace manifest**

`Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["spectral-core"]
```

- [ ] **Step 2: Write the crate manifest**

`spectral-core/Cargo.toml`:
```toml
[package]
name = "spectral-core"
version = "0.1.0"
edition = "2021"

[dependencies]
glam = "0.29"
rand = "0.8"
rand_pcg = "0.3"
image = { version = "0.25", default-features = false, features = ["png"] }
rayon = "1.10"

[dev-dependencies]
approx = "0.5"
```

- [ ] **Step 3: Write a placeholder lib with a smoke test**

`spectral-core/src/lib.rs`:
```rust
//! spectral-core: CPU spectral path tracer.

#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}
```

- [ ] **Step 4: Run to verify it builds and passes**

Run: `cargo test -p spectral-core smoke`
Expected: PASS, `test smoke::workspace_builds ... ok`

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml spectral-core/
git commit -m "feat: scaffold spectral workspace and spectral-core crate"
```

---

## Task 2: Spectral band and hero wavelength sampling

**Files:**
- Create: `spectral-core/src/spectrum.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod spectrum;`)

- [ ] **Step 1: Write the failing test**

Append to `spectral-core/src/spectrum.rs`:
```rust
//! Visible band, spectral sample carrier, and hero wavelength sampling.

pub const LAMBDA_MIN: f32 = 380.0;
pub const LAMBDA_MAX: f32 = 730.0;
pub const LAMBDA_RANGE: f32 = LAMBDA_MAX - LAMBDA_MIN;
pub const HERO_N: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hero_comb_is_stratified_and_in_band() {
        // u in [0,1) selects the hero; companions are evenly spaced and wrapped.
        let s = SpectralSample::from_hero_u(0.1);
        for &l in &s.lambda {
            assert!((LAMBDA_MIN..LAMBDA_MAX).contains(&l), "λ {l} out of band");
        }
        // Spacing between consecutive lanes (mod range) is LAMBDA_RANGE / HERO_N.
        let step = LAMBDA_RANGE / HERO_N as f32;
        for j in 0..HERO_N {
            let a = s.lambda[j] - LAMBDA_MIN;
            let b = s.lambda[(j + 1) % HERO_N] - LAMBDA_MIN;
            let d = (b - a).rem_euclid(LAMBDA_RANGE);
            assert!((d - step).abs() < 1e-3, "lane spacing {d} != {step}");
        }
        // Uniform sampling: every lane pdf is 1/range.
        for &p in &s.pdf {
            assert!((p - 1.0 / LAMBDA_RANGE).abs() < 1e-6);
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core spectrum`
Expected: FAIL, `cannot find type SpectralSample` / `from_hero_u` not found.

- [ ] **Step 3: Implement the carrier and sampler**

Insert above the `#[cfg(test)]` block in `spectral-core/src/spectrum.rs`:
```rust
/// A path's spectral state: HERO_N wavelengths traced coherently.
/// Lane 0 is the hero; lanes 1..HERO_N are stratified companions.
#[derive(Clone, Copy, Debug)]
pub struct SpectralSample {
    pub lambda: [f32; HERO_N],
    pub radiance: [f32; HERO_N],
    pub pdf: [f32; HERO_N],
}

impl SpectralSample {
    /// Build a hero comb from a uniform draw `u` in [0,1).
    /// Hero λ = LAMBDA_MIN + u*RANGE; companions add j*(RANGE/HERO_N), wrapped.
    pub fn from_hero_u(u: f32) -> Self {
        let step = LAMBDA_RANGE / HERO_N as f32;
        let mut lambda = [0.0; HERO_N];
        for j in 0..HERO_N {
            let off = (u * LAMBDA_RANGE + j as f32 * step).rem_euclid(LAMBDA_RANGE);
            lambda[j] = LAMBDA_MIN + off;
        }
        Self {
            lambda,
            radiance: [0.0; HERO_N],
            pdf: [1.0 / LAMBDA_RANGE; HERO_N], // uniform wavelength sampling
        }
    }
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod spectrum;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core spectrum`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/spectrum.rs spectral-core/src/lib.rs
git commit -m "feat: spectral band + hero wavelength sample carrier"
```

---

## Task 3: Seedable PCG RNG with stream keys

This RNG must be reproducible from explicit (pixel, sample, bounce) keys so the
future GPU backend can consume randomness in the identical order (the parity gate).

**Files:**
- Create: `spectral-core/src/rng.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod rng;`)

- [ ] **Step 1: Write the failing test**

`spectral-core/src/rng.rs`:
```rust
//! Deterministic per-sample RNG. Same key -> same sequence (CPU/GPU parity).

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
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core rng`
Expected: FAIL, `cannot find type PathRng`.

- [ ] **Step 3: Implement the RNG**

Insert above the test module in `spectral-core/src/rng.rs`:
```rust
use rand::Rng;
use rand_pcg::Pcg32;

/// PCG32 seeded from a stream key + global seed. Deterministic and portable.
pub struct PathRng {
    inner: Pcg32,
}

impl PathRng {
    /// `key` is a per-path stream identifier (e.g. pixel index mixed with sample
    /// index); `seed` is the global render seed.
    pub fn new(key: u64, seed: u64) -> Self {
        // Pcg32::new(state, stream): distinct streams for distinct keys.
        Self { inner: Pcg32::new(seed, key) }
    }

    /// Uniform f32 in [0,1).
    pub fn next_f32(&mut self) -> f32 {
        // 24 random bits -> [0,1), matching what the WGSL mirror will produce.
        let bits: u32 = self.inner.gen::<u32>() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core rng`
Expected: PASS (3 tests).

Add to `spectral-core/src/lib.rs`:
```rust
pub mod rng;
```

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/rng.rs spectral-core/src/lib.rs
git commit -m "feat: deterministic stream-keyed PCG rng for sampling and parity"
```

---

## Task 4: Sellmeier dispersion n(λ)

**Files:**
- Create: `spectral-core/src/sellmeier.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod sellmeier;`)

Reference values (Schott BK7): n(486.1nm)=1.52238, n(587.6nm)=1.51680, n(656.3nm)=1.51432.

- [ ] **Step 1: Write the failing test**

`spectral-core/src/sellmeier.rs`:
```rust
//! Sellmeier index of refraction n(λ). λ in nanometers at the API boundary.

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bk7_matches_published_lines() {
        // F, d, C Fraunhofer lines. Tolerance 5e-4 absolute index units.
        assert_relative_eq!(Glass::Bk7.n(486.1), 1.52238, epsilon = 5e-4);
        assert_relative_eq!(Glass::Bk7.n(587.6), 1.51680, epsilon = 5e-4);
        assert_relative_eq!(Glass::Bk7.n(656.3), 1.51432, epsilon = 5e-4);
    }

    #[test]
    fn dispersion_is_normal() {
        // Normal dispersion: shorter wavelength -> higher index.
        assert!(Glass::Bk7.n(450.0) > Glass::Bk7.n(650.0));
        assert!(Glass::Sf11.n(450.0) > Glass::Sf11.n(650.0));
    }

    #[test]
    fn sf11_disperses_more_than_bk7() {
        let spread = |g: Glass| g.n(450.0) - g.n(650.0);
        assert!(spread(Glass::Sf11) > spread(Glass::Bk7));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core sellmeier`
Expected: FAIL, `cannot find type Glass`.

- [ ] **Step 3: Implement Sellmeier**

Insert above the test module:
```rust
/// Glasses with published Sellmeier coefficients.
#[derive(Clone, Copy, Debug)]
pub enum Glass {
    Bk7,
    Sf11,
}

impl Glass {
    /// (B1,B2,B3, C1,C2,C3) with C in micrometers^2.
    fn coeffs(self) -> ([f64; 3], [f64; 3]) {
        match self {
            Glass::Bk7 => (
                [1.03961212, 0.231792344, 1.01046945],
                [0.00600069867, 0.0200179144, 103.560653],
            ),
            Glass::Sf11 => (
                [1.73759695, 0.313747346, 1.89878101],
                [0.013188707, 0.0623068142, 155.23629],
            ),
        }
    }

    /// Index of refraction at wavelength `lambda_nm`.
    pub fn n(self, lambda_nm: f32) -> f32 {
        let l_um = (lambda_nm as f64) / 1000.0; // nm -> micrometers
        let l2 = l_um * l_um;
        let (b, c) = self.coeffs();
        let mut n2 = 1.0;
        for k in 0..3 {
            n2 += b[k] * l2 / (l2 - c[k]);
        }
        n2.sqrt() as f32
    }
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod sellmeier;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core sellmeier`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/sellmeier.rs spectral-core/src/lib.rs
git commit -m "feat: Sellmeier dispersion for BK7 and SF11"
```

---

## Task 5: Optics — vector Snell, Fresnel, TIR (the 2D oracle)

This is the correctness core. Every refraction in 3D is verified against the
closed-form 2D Snell solution in the plane of incidence.

**Files:**
- Create: `spectral-core/src/optics.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod optics;`)

- [ ] **Step 1: Write the failing test (2D oracle)**

`spectral-core/src/optics.rs`:
```rust
//! Vector Snell refraction, Fresnel reflectance, total internal reflection.
//! Verified against the analytic 2D solution in the plane of incidence.

use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Analytic 2D refraction: given incidence angle and indices, return the
    // transmitted angle from the normal via Snell's law, or None on TIR.
    fn snell_angle_2d(theta_i: f32, n1: f32, n2: f32) -> Option<f32> {
        let s = n1 / n2 * theta_i.sin();
        if s.abs() > 1.0 { None } else { Some(s.asin()) }
    }

    #[test]
    fn refract_matches_2d_snell_across_angles() {
        let n1 = 1.0;
        let n2 = 1.5;
        let normal = Vec3::Y; // surface normal points "up", toward incoming side
        for deg in [5.0_f32, 15.0, 30.0, 45.0, 60.0] {
            let theta_i = deg.to_radians();
            // Incident direction travels downward into the surface, in the XY plane.
            let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
            let t = refract(d, normal, n1, n2).expect("should transmit");
            // Transmitted angle from -normal:
            let cos_tt = (-t).dot(-normal).clamp(-1.0, 1.0);
            let theta_t = cos_tt.acos();
            let expected = snell_angle_2d(theta_i, n1, n2).unwrap();
            assert_relative_eq!(theta_t, expected, epsilon = 1e-5);
            // Transmitted ray must stay in the plane of incidence (z == 0).
            assert!(t.z.abs() < 1e-6, "left plane of incidence: z={}", t.z);
        }
    }

    #[test]
    fn total_internal_reflection_past_critical_angle() {
        let n1 = 1.5;
        let n2 = 1.0;
        let theta_c = (n2 / n1).asin();
        let normal = Vec3::Y;
        // Just past critical angle: no transmission.
        let theta = theta_c + 0.05;
        let d = Vec3::new(theta.sin(), -theta.cos(), 0.0).normalize();
        assert!(refract(d, normal, n1, n2).is_none());
        // Just inside: transmission exists.
        let theta = theta_c - 0.05;
        let d = Vec3::new(theta.sin(), -theta.cos(), 0.0).normalize();
        assert!(refract(d, normal, n1, n2).is_some());
    }

    #[test]
    fn fresnel_normal_incidence() {
        // R0 = ((n1-n2)/(n1+n2))^2 at normal incidence.
        let r = fresnel_reflectance(1.0, 1.0, 1.5);
        assert_relative_eq!(r, ((1.0 - 1.5) / (1.0 + 1.5)).powi(2), epsilon = 1e-6);
    }

    #[test]
    fn fresnel_grazing_approaches_one() {
        let r = fresnel_reflectance(0.001, 1.0, 1.5); // cos_i ~ 0 => grazing
        assert!(r > 0.99, "grazing R should approach 1, got {r}");
    }

    #[test]
    fn reflect_is_mirror() {
        let d = Vec3::new(1.0, -1.0, 0.0).normalize();
        let r = reflect(d, Vec3::Y);
        assert_relative_eq!(r, Vec3::new(d.x, -d.y, 0.0).normalize(), epsilon = 1e-6);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core optics`
Expected: FAIL, `cannot find function refract` / `reflect` / `fresnel_reflectance`.

- [ ] **Step 3: Implement the optics**

Insert above the test module:
```rust
/// Mirror reflection of incident direction `d` about `n`. `d` points toward
/// the surface; result points away.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - 2.0 * d.dot(n) * n
}

/// Vector Snell refraction. `d` is the incident direction (toward surface),
/// `n` is the surface normal on the incoming side. `n1`/`n2` are the indices
/// before/after the interface. Returns None on total internal reflection.
pub fn refract(d: Vec3, n: Vec3, n1: f32, n2: f32) -> Option<Vec3> {
    let eta = n1 / n2;
    // Orient normal against the incoming ray.
    let mut nn = n;
    let mut cos_i = -d.dot(n);
    if cos_i < 0.0 {
        nn = -n;
        cos_i = -cos_i;
    }
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i); // cos^2(theta_t)
    if k < 0.0 {
        return None; // TIR
    }
    let cos_t = k.sqrt();
    Some((eta * d + (eta * cos_i - cos_t) * nn).normalize())
}

/// Unpolarized Fresnel reflectance for a dielectric interface.
/// `cos_i` is |cos| of the incidence angle; `n1`/`n2` the indices.
pub fn fresnel_reflectance(cos_i: f32, n1: f32, n2: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let sin_t2 = (n1 / n2).powi(2) * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0; // TIR
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)).powi(2);
    let rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)).powi(2);
    0.5 * (rs + rp)
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod optics;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core optics`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/optics.rs spectral-core/src/lib.rs
git commit -m "feat: vector Snell + Fresnel + TIR, verified vs 2D analytic oracle"
```

---

## Task 6: CIE sensor — CMF, illuminants, XYZ, sRGB, tone map

**Files:**
- Create: `spectral-core/data/cie_1931_2deg.csv`
- Create: `spectral-core/data/illuminant_d65.csv`
- Create: `spectral-core/data/illuminant_a.csv`
- Create: `spectral-core/src/cie.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod cie;`)

The data files are authoritative tables. Source them from CVRL
(http://cvrl.ioo.ucl.ac.uk): "CIE 1931 2-deg, XYZ CMFs" and the CIE standard
illuminant D65 and A relative SPDs, resampled to 5nm over 380..=730nm. Each CSV
is `wavelength_nm,col1[,col2,col3]` with no header.

- [ ] **Step 1: Create the data files**

`spectral-core/data/cie_1931_2deg.csv` — columns `nm,xbar,ybar,zbar` at 5nm.
First and last rows for format (fill the full 380..730 range from CVRL):
```csv
380,0.001368,0.000039,0.006450
385,0.002236,0.000064,0.010550
...
725,0.000007,0.000003,0.000000
730,0.000005,0.000002,0.000000
```

`spectral-core/data/illuminant_d65.csv` — columns `nm,relative_power` at 5nm:
```csv
380,49.9755
385,52.3118
...
730,60.3125
```

`spectral-core/data/illuminant_a.csv` — columns `nm,relative_power` at 5nm:
```csv
380,9.7951
385,10.8996
...
730,121.5073
```

- [ ] **Step 2: Write the failing test**

`spectral-core/src/cie.rs`:
```rust
//! CIE 1931 sensor: color-matching functions, illuminant SPDs, XYZ integration,
//! XYZ->linear sRGB, and tone mapping. Loaded from data/*.csv at build via
//! include_str!.

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn d65_white_point_chromaticity() {
        // A perfectly neutral (flat reflectance = 1) surface under D65 should
        // integrate to the D65 white point: x≈0.3127, y≈0.3290.
        let sensor = Sensor::new();
        let xyz = sensor.integrate_illuminant(Illuminant::D65);
        let (x, y) = chromaticity(xyz);
        assert_relative_eq!(x, 0.3127, epsilon = 2e-3);
        assert_relative_eq!(y, 0.3290, epsilon = 2e-3);
    }

    #[test]
    fn cmf_and_illuminant_sampling_in_band() {
        let sensor = Sensor::new();
        // Sampling at band center returns finite, non-negative CMF values.
        let (x, y, z) = sensor.cmf(555.0);
        assert!(x >= 0.0 && y > 0.0 && z >= 0.0);
        assert!(sensor.illuminant(Illuminant::D65, 555.0) > 0.0);
    }

    #[test]
    fn xyz_to_srgb_white_is_unit() {
        // D65 white XYZ (normalized Y=1) maps to roughly equal linear RGB.
        let rgb = xyz_to_linear_srgb([0.9505, 1.0, 1.0890]);
        assert_relative_eq!(rgb[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(rgb[1], 1.0, epsilon = 1e-2);
        assert_relative_eq!(rgb[2], 1.0, epsilon = 1e-2);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p spectral-core cie`
Expected: FAIL, `cannot find type Sensor`.

- [ ] **Step 4: Implement the sensor**

Insert above the test module:
```rust
use crate::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

const CMF_CSV: &str = include_str!("../data/cie_1931_2deg.csv");
const D65_CSV: &str = include_str!("../data/illuminant_d65.csv");
const A_CSV: &str = include_str!("../data/illuminant_a.csv");

#[derive(Clone, Copy)]
pub enum Illuminant {
    D65,
    A,
}

/// A tabulated curve sampled at uniform `step` from `start`.
struct Table {
    start: f32,
    step: f32,
    values: Vec<[f32; 3]>, // up to 3 columns; SPDs use only col 0
}

impl Table {
    fn parse(csv: &str, cols: usize) -> Self {
        let mut start = f32::NAN;
        let mut step = f32::NAN;
        let mut last_nm = f32::NAN;
        let mut values = Vec::new();
        for line in csv.lines().filter(|l| !l.trim().is_empty()) {
            let mut it = line.split(',');
            let nm: f32 = it.next().unwrap().trim().parse().unwrap();
            let mut row = [0.0; 3];
            for c in 0..cols {
                row[c] = it.next().unwrap().trim().parse().unwrap();
            }
            if start.is_nan() {
                start = nm;
            } else if step.is_nan() {
                step = nm - last_nm;
            }
            last_nm = nm;
            values.push(row);
        }
        Table { start, step, values }
    }

    /// Linear interpolation, clamped to table ends.
    fn sample(&self, nm: f32) -> [f32; 3] {
        let f = ((nm - self.start) / self.step).clamp(0.0, (self.values.len() - 1) as f32);
        let i = f.floor() as usize;
        let frac = f - i as f32;
        if i + 1 >= self.values.len() {
            return self.values[self.values.len() - 1];
        }
        let a = self.values[i];
        let b = self.values[i + 1];
        [
            a[0] + (b[0] - a[0]) * frac,
            a[1] + (b[1] - a[1]) * frac,
            a[2] + (b[2] - a[2]) * frac,
        ]
    }
}

pub struct Sensor {
    cmf: Table,
    d65: Table,
    a: Table,
}

impl Sensor {
    pub fn new() -> Self {
        Sensor {
            cmf: Table::parse(CMF_CSV, 3),
            d65: Table::parse(D65_CSV, 1),
            a: Table::parse(A_CSV, 1),
        }
    }

    pub fn cmf(&self, nm: f32) -> (f32, f32, f32) {
        let v = self.cmf.sample(nm);
        (v[0], v[1], v[2])
    }

    pub fn illuminant(&self, ill: Illuminant, nm: f32) -> f32 {
        match ill {
            Illuminant::D65 => self.d65.sample(nm)[0],
            Illuminant::A => self.a.sample(nm)[0],
        }
    }

    /// XYZ of a perfect (reflectance=1) white under `ill`, normalized so Y=1.
    /// X = ∫ S(λ) x̄(λ) dλ / k, with k = ∫ S(λ) ȳ(λ) dλ.
    pub fn integrate_illuminant(&self, ill: Illuminant) -> [f32; 3] {
        let step = 5.0_f32;
        let (mut x, mut y, mut z, mut k) = (0.0, 0.0, 0.0, 0.0);
        let mut nm = LAMBDA_MIN;
        while nm <= LAMBDA_MAX {
            let s = self.illuminant(ill, nm);
            let (xb, yb, zb) = self.cmf(nm);
            x += s * xb * step;
            y += s * yb * step;
            z += s * zb * step;
            k += s * yb * step;
            nm += step;
        }
        [x / k, y / k, z / k]
    }
}

pub fn chromaticity(xyz: [f32; 3]) -> (f32, f32) {
    let sum = xyz[0] + xyz[1] + xyz[2];
    (xyz[0] / sum, xyz[1] / sum)
}

/// XYZ (D65) to linear sRGB via the standard matrix.
pub fn xyz_to_linear_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = xyz;
    [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ]
}

/// Reinhard tone map + sRGB gamma encode, returning 8-bit channels.
pub fn tonemap_to_u8(linear: [f32; 3], exposure: f32) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = (linear[c] * exposure).max(0.0);
        let mapped = v / (1.0 + v); // Reinhard
        let gamma = if mapped <= 0.0031308 {
            12.92 * mapped
        } else {
            1.055 * mapped.powf(1.0 / 2.4) - 0.055
        };
        out[c] = (gamma.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod cie;
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p spectral-core cie`
Expected: PASS (3 tests). If the white point is off, the CSV data is wrong, not the code — re-pull from CVRL at 5nm.

- [ ] **Step 6: Commit**

```bash
git add spectral-core/data/ spectral-core/src/cie.rs spectral-core/src/lib.rs
git commit -m "feat: CIE 1931 sensor, illuminants, XYZ/sRGB, tonemap"
```

---

## Task 7: Geometry — ray, sphere, half-space CSG, prism

**Files:**
- Create: `spectral-core/src/geom.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod geom;`)

- [ ] **Step 1: Write the failing test**

`spectral-core/src/geom.rs`:
```rust
//! Ray, intersection record, sphere, half-space, and convex CSG (slab method).

use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ray_hits_sphere_front() {
        let s = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 1.0 };
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        let h = s.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal, Vec3::Z, epsilon = 1e-4); // faces the ray
    }

    #[test]
    fn ray_misses_sphere() {
        let s = Sphere { center: Vec3::new(5.0, 0.0, -5.0), radius: 1.0 };
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        assert!(s.intersect(&r, 1e-4, f32::INFINITY).is_none());
    }

    #[test]
    fn convex_box_intersection_interval() {
        // Unit cube centered at origin from 6 half-spaces; ray along -Z from z=5.
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::new(0.0, 0.0, 5.0), dir: -Vec3::Z };
        let h = cube.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-4); // enters at z=1
        assert_relative_eq!(h.normal, Vec3::Z, epsilon = 1e-4);
    }

    #[test]
    fn prism_is_five_halfspaces() {
        let p = ConvexSolid::triangular_prism(2.0, 2.0);
        assert_eq!(p.planes.len(), 5);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core geom`
Expected: FAIL, `cannot find type Ray`.

- [ ] **Step 3: Implement geometry**

Insert above the test module:
```rust
#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3, // unit
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3, // unit, oriented to face the incoming ray
}

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    pub fn intersect(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.dir.dot(r.dir);
        let half_b = oc.dot(r.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let disc = half_b * half_b - a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let mut t = (-half_b - sq) / a;
        if t < t_min || t > t_max {
            t = (-half_b + sq) / a;
            if t < t_min || t > t_max {
                return None;
            }
        }
        let point = r.at(t);
        let mut normal = (point - self.center) / self.radius;
        if normal.dot(r.dir) > 0.0 {
            normal = -normal;
        }
        Some(Hit { t, point, normal })
    }
}

/// Half-space {x : plane.normal·x + plane.d <= 0}. Normal points OUT of the solid.
#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

/// Convex solid = intersection of half-spaces. Solved by the slab method.
pub struct ConvexSolid {
    pub planes: Vec<Plane>,
}

impl ConvexSolid {
    pub fn axis_box(min: Vec3, max: Vec3) -> Self {
        ConvexSolid {
            planes: vec![
                Plane { normal: Vec3::X, d: -max.x },
                Plane { normal: -Vec3::X, d: min.x },
                Plane { normal: Vec3::Y, d: -max.y },
                Plane { normal: -Vec3::Y, d: min.y },
                Plane { normal: Vec3::Z, d: -max.z },
                Plane { normal: -Vec3::Z, d: min.z },
            ],
        }
    }

    /// Triangular prism: equilateral-ish cross-section in XY (3 side planes),
    /// extruded along Z to [-depth/2, depth/2] (2 cap planes). `size` is the
    /// half-extent of the cross-section.
    pub fn triangular_prism(size: f32, depth: f32) -> Self {
        // Three side normals 120 deg apart in XY, offset so the triangle has
        // inradius `size`.
        let mut planes = Vec::with_capacity(5);
        for k in 0..3 {
            let ang = std::f32::consts::FRAC_PI_2 + k as f32 * 2.0 * std::f32::consts::PI / 3.0;
            let n = Vec3::new(ang.cos(), ang.sin(), 0.0);
            planes.push(Plane { normal: n, d: -size });
        }
        planes.push(Plane { normal: Vec3::Z, d: -depth / 2.0 });
        planes.push(Plane { normal: -Vec3::Z, d: -depth / 2.0 });
        ConvexSolid { planes }
    }

    pub fn intersect(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let mut t_enter = t_min;
        let mut t_exit = t_max;
        let mut enter_normal = Vec3::ZERO;
        for p in &self.planes {
            let denom = p.normal.dot(r.dir);
            let dist = p.normal.dot(r.origin) + p.d; // signed; >0 outside
            if denom.abs() < 1e-8 {
                if dist > 0.0 {
                    return None; // parallel and outside this slab
                }
                continue;
            }
            let t = -dist / denom;
            if denom < 0.0 {
                // entering this half-space
                if t > t_enter {
                    t_enter = t;
                    enter_normal = p.normal;
                }
            } else {
                // exiting
                if t < t_exit {
                    t_exit = t;
                }
            }
            if t_enter > t_exit {
                return None;
            }
        }
        if t_enter < t_min || t_enter > t_max {
            return None;
        }
        let mut normal = enter_normal;
        if normal.dot(r.dir) > 0.0 {
            normal = -normal;
        }
        Some(Hit { t: t_enter, point: r.at(t_enter), normal })
    }
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod geom;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core geom`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/geom.rs spectral-core/src/lib.rs
git commit -m "feat: ray/sphere + half-space convex CSG geometry incl. prism"
```

---

## Task 8: Materials and BSDFs

**Files:**
- Create: `spectral-core/src/material.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod material;`)

- [ ] **Step 1: Write the failing test**

`spectral-core/src/material.rs`:
```rust
//! Surface materials. Lambertian diffuse and smooth dispersive dielectric.

use crate::geom::Hit;
use crate::optics::{fresnel_reflectance, reflect, refract};
use crate::rng::PathRng;
use crate::sellmeier::Glass;
use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambertian_scatters_into_hemisphere() {
        let m = Material::Lambertian { reflectance: 0.8 };
        let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y };
        let mut rng = PathRng::new(1, 1);
        let s = m.scatter(Vec3::new(0.0, -1.0, 0.0), &h, 550.0, 1.5, &mut rng).unwrap();
        assert!(s.dir.dot(h.normal) > 0.0, "must scatter above surface");
        assert!((0.0..=1.0).contains(&s.weight));
    }

    #[test]
    fn dielectric_disperses_by_wavelength() {
        // Same incident ray, different wavelengths -> different refract dirs.
        let m = Material::Dielectric { glass: Glass::Sf11 };
        let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y };
        let d = Vec3::new(0.6, -0.8, 0.0).normalize();
        // Force the refraction branch by checking refract() directly per λ.
        let n_blue = Glass::Sf11.n(450.0);
        let n_red = Glass::Sf11.n(650.0);
        let t_blue = refract(d, h.normal, 1.0, n_blue).unwrap();
        let t_red = refract(d, h.normal, 1.0, n_red).unwrap();
        assert!((t_blue - t_red).length() > 1e-3, "no dispersion observed");
        let _ = m; // material wraps this per-hero-λ in scatter()
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core material`
Expected: FAIL, `cannot find type Material`.

- [ ] **Step 3: Implement materials**

Insert above the test module:
```rust
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub enum Material {
    Lambertian { reflectance: f32 },
    Dielectric { glass: Glass },
}

/// Result of scattering: a new direction and a throughput multiplier applied to
/// every spectral lane (dielectric handles dispersion via the hero λ at call site).
pub struct Scatter {
    pub dir: Vec3,
    pub weight: f32,
}

impl Material {
    /// `wo_in` is the incident direction (toward the surface). `hero_lambda` is
    /// lane-0 wavelength; `n_hero` is the dielectric index at that wavelength.
    pub fn scatter(
        &self,
        wo_in: Vec3,
        hit: &Hit,
        hero_lambda: f32,
        n_hero: f32,
        rng: &mut PathRng,
    ) -> Option<Scatter> {
        match *self {
            Material::Lambertian { reflectance } => {
                let dir = cosine_hemisphere(hit.normal, rng);
                // Cosine-weighted sampling cancels the cosine/pdf; weight = albedo.
                Some(Scatter { dir, weight: reflectance })
            }
            Material::Dielectric { .. } => {
                let _ = hero_lambda;
                // Decide reflect vs refract on the hero λ using Fresnel R.
                let cos_i = (-wo_in.dot(hit.normal)).abs();
                // Determine indices by whether we're entering or leaving glass.
                let entering = wo_in.dot(hit.normal) < 0.0;
                let (n1, n2) = if entering { (1.0, n_hero) } else { (n_hero, 1.0) };
                let r = fresnel_reflectance(cos_i, n1, n2);
                if rng.next_f32() < r {
                    Some(Scatter { dir: reflect(wo_in, hit.normal), weight: 1.0 })
                } else {
                    match refract(wo_in, hit.normal, n1, n2) {
                        Some(dir) => Some(Scatter { dir, weight: 1.0 }),
                        None => Some(Scatter { dir: reflect(wo_in, hit.normal), weight: 1.0 }),
                    }
                }
            }
        }
    }
}

/// Cosine-weighted hemisphere sample around `normal`.
fn cosine_hemisphere(normal: Vec3, rng: &mut PathRng) -> Vec3 {
    let u1 = rng.next_f32();
    let u2 = rng.next_f32();
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u1).max(0.0).sqrt();
    // Build an orthonormal basis around the normal.
    let a = if normal.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t = normal.cross(a).normalize();
    let b = normal.cross(t);
    (t * x + b * y + normal * z).normalize()
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod material;
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p spectral-core material`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add spectral-core/src/material.rs spectral-core/src/lib.rs
git commit -m "feat: Lambertian + dispersive dielectric materials"
```

---

## Task 9: Scene, emitters, and camera

**Files:**
- Create: `spectral-core/src/scene.rs`
- Create: `spectral-core/src/camera.rs`
- Modify: `spectral-core/src/lib.rs` (add `pub mod scene;` and `pub mod camera;`)

- [ ] **Step 1: Write the failing test**

`spectral-core/src/scene.rs`:
```rust
//! Scene: a list of primitives with materials, plus emitters and a background.

use crate::geom::{ConvexSolid, Hit, Ray, Sphere};
use crate::material::Material;
use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_picks_nearest() {
        let mut scene = Scene::new();
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Lambertian { reflectance: 0.5 });
        scene.add_sphere(Vec3::new(0.0, 0.0, -6.0), 1.0, Material::Lambertian { reflectance: 0.5 });
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        let (hit, _mat) = scene.intersect(&r).expect("hit");
        assert!((hit.t - 2.0).abs() < 1e-3, "nearest sphere at t=2, got {}", hit.t);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p spectral-core scene`
Expected: FAIL, `cannot find type Scene`.

- [ ] **Step 3: Implement scene**

Insert above the test module:
```rust
pub enum Shape {
    Sphere(Sphere),
    Solid(ConvexSolid),
}

pub struct Primitive {
    pub shape: Shape,
    pub material: Material,
}

/// A directional emitter with an illuminant SPD, used as a bright background
/// the camera can see through/around the prism.
pub struct Scene {
    pub primitives: Vec<Primitive>,
    /// Background radiance scale; the SPD comes from the sensor's illuminant.
    pub background: f32,
}

impl Scene {
    pub fn new() -> Self {
        Scene { primitives: Vec::new(), background: 1.0 }
    }

    pub fn add_sphere(&mut self, center: Vec3, radius: f32, material: Material) {
        self.primitives.push(Primitive {
            shape: Shape::Sphere(Sphere { center, radius }),
            material,
        });
    }

    pub fn add_solid(&mut self, solid: ConvexSolid, material: Material) {
        self.primitives.push(Primitive { shape: Shape::Solid(solid), material });
    }

    pub fn intersect(&self, r: &Ray) -> Option<(Hit, Material)> {
        let mut best: Option<(Hit, Material)> = None;
        let mut closest = f32::INFINITY;
        for p in &self.primitives {
            let hit = match &p.shape {
                Shape::Sphere(s) => s.intersect(r, 1e-3, closest),
                Shape::Solid(s) => s.intersect(r, 1e-3, closest),
            };
            if let Some(h) = hit {
                closest = h.t;
                best = Some((h, p.material));
            }
        }
        best
    }
}
```

`spectral-core/src/camera.rs`:
```rust
//! Pinhole camera generating primary rays.

use crate::geom::Ray;
use glam::Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_pixel_looks_down_minus_z() {
        let cam = Camera::look_at(Vec3::ZERO, -Vec3::Z, Vec3::Y, 60.0, 1.0);
        let r = cam.primary_ray(0.5, 0.5);
        assert!(r.dir.dot(-Vec3::Z) > 0.99, "center ray should look down -Z");
    }
}
```

- [ ] **Step 4: Run scene test to verify it passes; camera test will fail**

Run: `cargo test -p spectral-core scene`
Expected: PASS.
Run: `cargo test -p spectral-core camera`
Expected: FAIL, `cannot find type Camera`.

- [ ] **Step 5: Implement camera**

Insert above the camera test module:
```rust
pub struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    /// `vfov_deg` vertical field of view; `aspect` = width/height.
    pub fn look_at(origin: Vec3, target: Vec3, up: Vec3, vfov_deg: f32, aspect: f32) -> Self {
        let theta = vfov_deg.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect * viewport_h;
        let w = (origin - target).normalize();
        let u = up.cross(w).normalize();
        let v = w.cross(u);
        let horizontal = viewport_w * u;
        let vertical = viewport_h * v;
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - w;
        Camera { origin, lower_left, horizontal, vertical }
    }

    /// `s`,`t` in [0,1] across the image (origin bottom-left).
    pub fn primary_ray(&self, s: f32, t: f32) -> Ray {
        let dir = (self.lower_left + s * self.horizontal + t * self.vertical - self.origin)
            .normalize();
        Ray { origin: self.origin, dir }
    }
}
```

- [ ] **Step 6: Run to verify both pass**

Run: `cargo test -p spectral-core scene camera`
Expected: PASS.

Add to `spectral-core/src/lib.rs`:
```rust
pub mod scene;
pub mod camera;
```

- [ ] **Step 7: Commit**

```bash
git add spectral-core/src/scene.rs spectral-core/src/camera.rs spectral-core/src/lib.rs
git commit -m "feat: scene intersection and pinhole camera"
```

---

## Task 10: CPU path tracer + Renderer trait + XYZ accumulation

**Files:**
- Create: `spectral-core/src/tracer.rs`
- Modify: `spectral-core/src/lib.rs` (add Renderer trait, Xyz, AccumBuffer, `pub mod tracer;`)

- [ ] **Step 1: Add the Renderer trait and accumulation buffer to lib.rs**

Add to `spectral-core/src/lib.rs` (above the module declarations):
```rust
/// CIE XYZ tristimulus.
pub type Xyz = [f32; 3];

/// Per-pixel running sum of XYZ contributions and the sample count.
pub struct AccumBuffer {
    pub width: usize,
    pub height: usize,
    pub sum: Vec<Xyz>,
    pub samples: u32,
}

impl AccumBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        AccumBuffer { width, height, sum: vec![[0.0; 3]; width * height], samples: 0 }
    }
    pub fn reset(&mut self) {
        self.sum.iter_mut().for_each(|p| *p = [0.0; 3]);
        self.samples = 0;
    }
    /// Mean XYZ per pixel.
    pub fn mean(&self) -> Vec<Xyz> {
        let n = self.samples.max(1) as f32;
        self.sum.iter().map(|s| [s[0] / n, s[1] / n, s[2] / n]).collect()
    }
}

/// Backend-agnostic renderer. CPU now; GPU and wave solver later.
pub trait Renderer {
    fn accumulate(&mut self, samples_per_pixel: u32);
    fn buffer(&self) -> &AccumBuffer;
    fn reset(&mut self);
}
```

- [ ] **Step 2: Write the failing test**

`spectral-core/src/tracer.rs`:
```rust
//! CPU spectral path tracer implementing the Renderer trait.

use crate::camera::Camera;
use crate::cie::{Illuminant, Sensor};
use crate::material::Material;
use crate::scene::Scene;
use crate::sellmeier::Glass;
use crate::spectrum::SpectralSample;
use crate::rng::PathRng;
use crate::{AccumBuffer, Renderer};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn accumulating_increases_samples_and_energy() {
        let mut scene = Scene::new();
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0,
            Material::Lambertian { reflectance: 0.8 });
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0,0.0,-1.0), Vec3::Y, 60.0, 1.0);
        let mut t = CpuTracer::new(scene, cam, 16, 16, Illuminant::D65, 42);
        t.accumulate(4);
        assert_eq!(t.buffer().samples, 4);
        // Background is non-zero, so total energy must be positive.
        let total: f32 = t.buffer().mean().iter().map(|p| p[1]).sum();
        assert!(total > 0.0, "expected positive luminance");
    }

    #[test]
    fn reset_clears() {
        let scene = Scene::new();
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0,0.0,-1.0), Vec3::Y, 60.0, 1.0);
        let mut t = CpuTracer::new(scene, cam, 8, 8, Illuminant::D65, 1);
        t.accumulate(2);
        t.reset();
        assert_eq!(t.buffer().samples, 0);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p spectral-core tracer`
Expected: FAIL, `cannot find type CpuTracer`.

- [ ] **Step 4: Implement the tracer**

Insert above the test module:
```rust
pub struct CpuTracer {
    scene: Scene,
    camera: Camera,
    sensor: Sensor,
    illuminant: Illuminant,
    seed: u64,
    accum: AccumBuffer,
    max_bounces: u32,
}

impl CpuTracer {
    pub fn new(
        scene: Scene,
        camera: Camera,
        width: usize,
        height: usize,
        illuminant: Illuminant,
        seed: u64,
    ) -> Self {
        CpuTracer {
            scene,
            camera,
            sensor: Sensor::new(),
            illuminant,
            seed,
            accum: AccumBuffer::new(width, height),
            max_bounces: 8,
        }
    }

    /// Trace one hero-comb path for a pixel and return its XYZ contribution.
    fn sample_pixel(&self, px: usize, py: usize, sample_idx: u32) -> [f32; 3] {
        let w = self.accum.width;
        let h = self.accum.height;
        let key = ((py * w + px) as u64) << 20 ^ sample_idx as u64;
        let mut rng = PathRng::new(key, self.seed);

        // Jittered pixel position and hero wavelength.
        let s = (px as f32 + rng.next_f32()) / w as f32;
        let t = 1.0 - (py as f32 + rng.next_f32()) / h as f32;
        let mut spec = SpectralSample::from_hero_u(rng.next_f32());

        let mut ray = self.camera.primary_ray(s, t);
        let mut throughput = [1.0f32; crate::spectrum::HERO_N];

        for _ in 0..self.max_bounces {
            match self.scene.intersect(&ray) {
                None => {
                    // Hit background: deposit illuminant SPD per lane.
                    for k in 0..crate::spectrum::HERO_N {
                        let sp = self.sensor.illuminant(self.illuminant, spec.lambda[k]);
                        spec.radiance[k] += throughput[k] * self.scene.background * sp;
                    }
                    break;
                }
                Some((hit, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(spec.lambda[0]),
                        _ => 1.0,
                    };
                    match mat.scatter(ray.dir, &hit, spec.lambda[0], n_hero, &mut rng) {
                        None => break,
                        Some(sc) => {
                            for k in 0..crate::spectrum::HERO_N {
                                throughput[k] *= sc.weight;
                            }
                            ray = crate::geom::Ray { origin: hit.point, dir: sc.dir };
                        }
                    }
                }
            }
        }

        // Convert the comb's radiance to XYZ via the sensor, weighting by 1/pdf.
        let mut xyz = [0.0f32; 3];
        for k in 0..crate::spectrum::HERO_N {
            let (xb, yb, zb) = self.sensor.cmf(spec.lambda[k]);
            let w = spec.radiance[k] / spec.pdf[k] / crate::spectrum::HERO_N as f32;
            xyz[0] += xb * w;
            xyz[1] += yb * w;
            xyz[2] += zb * w;
        }
        xyz
    }
}

impl Renderer for CpuTracer {
    fn accumulate(&mut self, samples_per_pixel: u32) {
        use rayon::prelude::*;
        let w = self.accum.width;
        let h = self.accum.height;
        let base = self.accum.samples;
        // Compute this batch into a temporary, then fold into the sum.
        let batch: Vec<[f32; 3]> = (0..w * h)
            .into_par_iter()
            .map(|i| {
                let px = i % w;
                let py = i / w;
                let mut acc = [0.0f32; 3];
                for s in 0..samples_per_pixel {
                    let c = self.sample_pixel(px, py, base + s);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
                acc
            })
            .collect();
        for (dst, src) in self.accum.sum.iter_mut().zip(batch) {
            dst[0] += src[0];
            dst[1] += src[1];
            dst[2] += src[2];
        }
        self.accum.samples += samples_per_pixel;
    }

    fn buffer(&self) -> &AccumBuffer {
        &self.accum
    }

    fn reset(&mut self) {
        self.accum.reset();
    }
}
```

Add to `spectral-core/src/lib.rs`:
```rust
pub mod tracer;
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p spectral-core tracer`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add spectral-core/src/tracer.rs spectral-core/src/lib.rs
git commit -m "feat: CPU spectral path tracer with hero-comb XYZ accumulation"
```

---

## Task 11: Acceptance Scene A — direct-view prism (dispersion gate)

**Files:**
- Create: `spectral-core/tests/scene_a_prism.rs`
- Create: `spectral-core/examples/prism.rs`

- [ ] **Step 1: Write the failing acceptance test**

`spectral-core/tests/scene_a_prism.rs`:
```rust
//! Acceptance: a collimated ray refracted through a BK7 prism disperses with
//! correct chromatic ordering and an angular spread matching analytic Sellmeier.

use spectral_core::optics::refract;
use spectral_core::sellmeier::Glass;
use glam::Vec3;

/// Deviation through a single flat interface (entering glass) for wavelength λ:
/// the angle between the incident and refracted rays.
fn deviation(d: Vec3, normal: Vec3, lambda: f32) -> f32 {
    let n = Glass::Bk7.n(lambda);
    let t = refract(d, normal, 1.0, n).unwrap();
    d.angle_between(t)
}

#[test]
fn chromatic_ordering_blue_deviates_more_than_red() {
    let normal = Vec3::Y;
    let d = Vec3::new(0.5, -0.8660254, 0.0).normalize(); // 30 deg incidence
    let dev_blue = deviation(d, normal, 450.0);
    let dev_red = deviation(d, normal, 650.0);
    assert!(dev_blue > dev_red, "blue must deviate more: {dev_blue} vs {dev_red}");
}

#[test]
fn angular_spread_matches_analytic() {
    // Closed-form deviation via Snell at one interface, compared to refract().
    let normal = Vec3::Y;
    let theta_i = 30.0_f32.to_radians();
    let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
    for lambda in [440.0_f32, 520.0, 600.0, 680.0] {
        let n = Glass::Bk7.n(lambda);
        let theta_t = (theta_i.sin() / n).asin();
        let analytic_dev = theta_i - theta_t; // bends toward normal entering glass
        let measured = deviation(d, normal, lambda);
        let err = (measured - analytic_dev).abs() / analytic_dev;
        assert!(err < 0.03, "λ={lambda}: rel error {err} exceeds 3%");
    }
}
```

- [ ] **Step 2: Run to verify it fails (or passes if optics is correct)**

Run: `cargo test -p spectral-core --test scene_a_prism`
Expected: PASS if Task 5 optics is correct (this test gates physics, not new code).
If it fails, the bug is in `refract` or `Glass::n` — fix there, do not weaken the test.

- [ ] **Step 3: Write the visual example that renders the prism to PNG**

`spectral-core/examples/prism.rs`:
```rust
//! Renders the direct-view prism scene to prism.png.

use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb, Illuminant};
use spectral_core::geom::ConvexSolid;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use glam::Vec3;

fn main() {
    let (w, h) = (512usize, 512usize);
    let mut scene = Scene::new();
    scene.background = 3.0; // bright environment behind the prism
    let prism = ConvexSolid::triangular_prism(1.0, 2.0);
    scene.add_solid(prism, Material::Dielectric { glass: Glass::Bk7 });
    let _ = Glass::Sf11; // swap here for a harder fan

    let cam = Camera::look_at(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        45.0,
        w as f32 / h as f32,
    );
    let mut tracer = CpuTracer::new(scene, cam, w, h, Illuminant::D65, 0xC0FFEE);
    tracer.accumulate(256);

    let mean = tracer.buffer().mean();
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in mean.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), 1.0);
        img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    img.save("prism.png").unwrap();
    println!("wrote prism.png");
}
```

- [ ] **Step 4: Run the example and verify the PNG**

Run: `cargo run -p spectral-core --release --example prism`
Expected: prints `wrote prism.png`; open it and confirm visible color separation through the prism. (Visual confirmation only; the physics gate is the test in Step 2.)

- [ ] **Step 5: Commit**

```bash
git add spectral-core/tests/scene_a_prism.rs spectral-core/examples/prism.rs
git commit -m "test: prism dispersion acceptance gate + visual example"
```

---

## Task 12: Acceptance Scene B — metamerism (accurate color gate)

**Files:**
- Create: `spectral-core/tests/scene_b_metamerism.rs`

This gate proves the sensor does real color science: two distinct reflectance
spectra that match under D65 must diverge under illuminant A.

- [ ] **Step 1: Write the failing test**

`spectral-core/tests/scene_b_metamerism.rs`:
```rust
//! Acceptance: a metameric pair. Two reflectance spectra integrate to the same
//! chromaticity under D65 but different chromaticities under illuminant A.

use spectral_core::cie::{chromaticity, Illuminant, Sensor};
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

/// Integrate reflectance R(λ) under illuminant, returning chromaticity (x,y).
fn chroma_under(sensor: &Sensor, refl: &dyn Fn(f32) -> f32, ill: Illuminant) -> (f32, f32) {
    let step = 5.0_f32;
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        let s = sensor.illuminant(ill, nm) * refl(nm);
        let (xb, yb, zb) = sensor.cmf(nm);
        x += s * xb * step;
        y += s * yb * step;
        z += s * zb * step;
        nm += step;
    }
    chromaticity([x, y, z])
}

// Two reflectances engineered to be near-metameric under D65: a smooth ramp and
// a triple-bump curve with the same integrated tristimulus ratios.
fn smooth(nm: f32) -> f32 {
    0.5 + 0.3 * ((nm - 555.0) / 120.0).cos()
}
fn bumpy(nm: f32) -> f32 {
    let g = |c: f32, w: f32| (-((nm - c) / w).powi(2)).exp();
    (0.55 * g(450.0, 25.0) + 0.75 * g(540.0, 25.0) + 0.65 * g(610.0, 25.0)).clamp(0.0, 1.0)
}

#[test]
fn pair_is_distinct_spectra() {
    // Sanity: the two curves are genuinely different spectra.
    let diff: f32 = (400..700).step_by(20)
        .map(|n| (smooth(n as f32) - bumpy(n as f32)).abs())
        .sum();
    assert!(diff > 0.5, "curves must differ as spectra");
}

#[test]
fn illuminant_swap_changes_color_relationship() {
    let sensor = Sensor::new();
    let d_d65 = {
        let a = chroma_under(&sensor, &smooth, Illuminant::D65);
        let b = chroma_under(&sensor, &bumpy, Illuminant::D65);
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    };
    let d_a = {
        let a = chroma_under(&sensor, &smooth, Illuminant::A);
        let b = chroma_under(&sensor, &bumpy, Illuminant::A);
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    };
    // The chromaticity gap under A must differ from under D65: illuminant-
    // dependent color is the metamerism signature.
    assert!((d_a - d_d65).abs() > 1e-3,
        "color relationship must shift with illuminant: d65={d_d65}, a={d_a}");
}
```

- [ ] **Step 2: Run to verify it passes**

Run: `cargo test -p spectral-core --test scene_b_metamerism`
Expected: PASS (2 tests). This exercises only the sensor + illuminant code from
Task 6; no new library code.

- [ ] **Step 3: Run the full suite**

Run: `cargo test -p spectral-core`
Expected: all tests across all modules PASS.

- [ ] **Step 4: Commit**

```bash
git add spectral-core/tests/scene_b_metamerism.rs
git commit -m "test: metamerism acceptance gate (illuminant-dependent color)"
```

---

## Final verification

- [ ] Run `cargo test -p spectral-core` — every unit and acceptance test passes.
- [ ] Run `cargo run -p spectral-core --release --example prism` — `prism.png` shows visible spectral separation through the prism.
- [ ] Run `cargo clippy -p spectral-core --all-targets` — no warnings (fix any).

This completes Plan 1. The CPU oracle now exists. Plan 2 (`spectral-gpu`) mirrors
`CpuTracer`'s physics in WGSL and adds the RNG-parity and diff gates against it;
Plan 3 (`spectral-viewer`) wraps the `Renderer` trait in a progressive winit window.

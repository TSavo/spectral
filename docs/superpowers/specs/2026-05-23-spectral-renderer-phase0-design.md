# Spectral Renderer — Phase 0: Core Spine

**Date:** 2026-05-23
**Status:** Design approved, pending spec review
**Scope:** Phase 0 of a phased spectral rendering engine

## 1. Context

We are building a wavelength-resolved light-transport renderer as a visual
showpiece: the kind of image RGB renderers physically cannot produce. Light is
carried as spectral power along ray vectors, refracted by a wavelength-dependent
index of refraction (Snell's law with `n = n(λ)`), and converted to color only at
the sensor via the CIE color-matching functions.

The full engine is too large for one spec, so it is decomposed into phases. Each
phase is its own spec → plan → build cycle.

| Phase | Deliverable | What's new |
|-------|-------------|-----------|
| **0 — Core spine (this spec)** | Prism disperses white light into a rainbow, interactively; metamerism payoff | HWSS spectral transport, dispersive Snell, spectral sensor, progressive viewer |
| 1 — Thin-film | Soap-bubble / oil-slick iridescence | Airy-summation BSDF |
| 2 — Grating | Rainbow off a CD | Grating-equation BSDF |
| 3 — Fluorescence | Tonic water under UV | Reradiation matrix + wavelength reassignment |
| 4 — Wave solver | True edge diffraction, interference fringes | Sibling engine (see §12) |

Phase 0 alone is a complete, shippable demo.

## 2. Goals and Non-Goals

### Goals
- A spectral path tracer that renders dispersion through a glass prism, verified
  against the closed-form 2D Snell solution.
- Progressive viewer: grainy interactive preview that converges to a clean image
  when the camera stops moving; accumulation resets on camera move.
- Spectral sensor with swappable illuminants, producing the metamerism demo for
  free (swap illuminant, watch chromaticities diverge).
- A CPU reference renderer that is the correctness oracle, with a CI gate that
  diffs the GPU renderer against it on a fixed scene under a tolerance.

### Non-Goals (deferred to later phases or explicitly out)
- Triangle meshes, BVH, scene-file loaders. Geometry is half-space CSG only.
- Caustics / projected-rainbow-on-a-wall. That is a forward-transport phenomenon
  requiring light tracing or bidirectional path tracing (see §11).
- Thin-film, gratings, fluorescence. Phases 1–3.
- True wave propagation. Phase 4, a sibling engine.
- Textures, normal maps, RGB-to-spectral upsampling of image inputs.

## 3. Architecture

A three-crate Rust workspace. The fault line keeps the CPU oracle structural
rather than bolted on.

```
spectral/
├── spectral-core/     # no GPU. Types + physics + CPU reference path tracer.
├── spectral-gpu/      # wgpu compute kernels mirroring spectral-core's physics.
└── spectral-viewer/   # winit window, progressive accumulation, tone map, input.
```

- **`spectral-core`** owns every physical quantity and law: spectral sample
  vectors, Sellmeier dispersion, Fresnel, vector Snell, the CIE sensor pipeline,
  half-space geometry, the scene description, and a single-threaded-or-rayon CPU
  path tracer. It depends on no GPU code. It is the ground truth.
- **`spectral-gpu`** ports the hot path-tracing loop to WGSL compute, consuming
  the *same* scene description and sensor from `spectral-core`. Its only job is to
  produce, fast, what `spectral-core` produces slowly.
- **`spectral-viewer`** owns the OS window, the accumulation buffer, camera input,
  per-frame tone mapping, and renderer selection behind a `Renderer` trait. Both
  the CPU and GPU tracers implement `Renderer`. This trait is the seam where the
  Phase 4 wave solver will later attach as a third implementation (§12).

```rust
trait Renderer {
    // Add `samples_per_pixel` new spectral samples to the accumulation buffer,
    // for the given camera and scene. Backend-agnostic.
    fn accumulate(&mut self, scene: &Scene, camera: &Camera, samples_per_pixel: u32);
    fn read_xyz(&self) -> &[Xyz];   // running mean of accumulated XYZ, per pixel
    fn reset(&mut self);            // called on camera move
}
```

## 4. Spectral Representation and Monte Carlo

### The pixel is an integral
A pixel's color is three integrals of arriving radiance against the eye's
response curves over the visible band (380–730 nm):

```
X = ∫ L(λ) x̄(λ) dλ,   Y = ∫ L(λ) ȳ(λ) dλ,   Z = ∫ L(λ) z̄(λ) dλ
```

`L(λ)` is whatever the scene did to the light, so the integral has no closed
form. We estimate it by Monte Carlo. With wavelength samples `λᵢ ~ p(λ)`:

```
X ≈ (1/N) Σ  L(λᵢ) · x̄(λᵢ) / p(λᵢ)
```

This is the coins-on-a-circular-table-in-a-square estimator: `p(λ)` is the
distribution we throw into, the weight `L·x̄/p` is how much each coin counts, and
the running average converges to the integral at rate `1/√N`. The "spread of a
single emission across the spectrum" is **temporal** — it accrues across
accumulated samples, not within one ray. In the viewer, the convergence the user
watches *is* this estimator settling.

### Hero Wavelength Sampling (HWSS)
One wavelength per expensive path is wasteful and leaves chromatic noise. We use
hero wavelength sampling (Wilkie et al. 2014) to lower the variance constant:

- Pick a hero `λ_h ~ p`. Derive 3 companions as a comb wrapped across the band:
  `λ_j = 380 + mod(λ_h − 380 + j · (350/4), 350)` for `j = 1..3`.
- A path therefore carries a **4-wide wavelength vector and 4 radiance values**.
- All four share the traced path until a dispersive interface separates them, so
  four stratified spectral samples cost ≈ one path.
- Recombine the four with **MIS (balance heuristic)** weights, because once
  dispersion bends them differently their path pdfs diverge.

In coin terms: instead of one random coin we drop a jittered row of four spanning
the table — same unbiased estimator, far lower variance.

### Importance sampling
- Sample emitter wavelength ∝ its SPD (do not throw coins where the lamp emits
  nothing); correct by `1/p`.
- We deliberately do **not** importance-sample by sensor luminous response: it
  biases toward green and starves the red/blue tails of the rainbow. For the prism
  scene we keep wavelength sampling near-uniform-with-stratification.

### Concrete data
```rust
const LAMBDA_MIN: f32 = 380.0;
const LAMBDA_MAX: f32 = 730.0;
const HERO_N:   usize = 4;

struct SpectralSample {
    lambda: [f32; HERO_N],   // hero + 3 stratified companions, nm
    radiance: [f32; HERO_N], // power surviving along the vector, per λ
    pdf: [f32; HERO_N],      // wavelength pdf, for MIS recombination
}
```

## 5. Geometry — Half-Space CSG

No meshes. Primitives are defined by intersections of half-spaces, which is all
the prism showpiece needs and avoids BVH/instancing tangents.

- **Sphere**: analytic ray–sphere intersection.
- **Half-space**: a plane `n·x + d ≤ 0`. Ray–half-space gives an entry/exit
  parameter interval.
- **Convex CSG**: intersection of half-spaces, solved by the slab/interval method
  (intersect the per-half-space `[t_enter, t_exit]` intervals; non-empty result is
  a hit, with the bounding half-space supplying the surface normal).

A 2D triangular prism is 3 half-planes; the 3D triangular prism is that triangular
column **capped at both ends**, i.e. the intersection of **5 half-spaces**. The
infinite ground plane is a single half-space.

## 6. Materials and BSDFs

Phase 0 ships exactly two materials.

### Lambertian
Wavelength-independent diffuse albedo (optionally a flat or simple-parametric
reflectance spectrum). Cosine-weighted hemisphere sampling. Used for the screen,
ground, and the colored swatches in the metamerism demo.

### Smooth dielectric (the dispersive one)
- Index of refraction from the **Sellmeier equation**, default coefficients for
  **BK7 glass**:
  ```
  n²(λ) = 1 + Σ_k  B_k λ² / (λ² − C_k),   λ in micrometers
  ```
- **Vector Snell's law** for refraction, with `η = n₁/n₂(λ)` and
  `cosθᵢ = −d·n̂`:
  ```
  t(λ) = η d + (η cosθᵢ − cosθₜ) n̂,   cosθₜ = √(1 − η²(1 − cos²θᵢ))
  ```
  A negative radicand is **total internal reflection** — wavelength-dependent, so
  some companions transmit while others reflect.
- **Fresnel equations** (dielectric, unpolarized average of s/p) give reflectance
  `R(λ,θ)`; transmittance is `1 − R`. This is "the power that survives the
  vector." At each interface we make a single **stochastic choice** of reflect vs
  refract (Russian roulette weighted by `R`), not a deterministic energy split, so
  the path stays one ray and the RNG-parity gate (§10) remains meaningful. Per
  hero/companion wavelength the `R` differs, so the branch is decided on the hero λ
  and companions follow, with MIS correcting the resulting pdf differences.
- Because `n` depends on λ, the hero and companion wavelengths refract by
  different amounts — a single incident path fans into a spectrum. This is the
  dispersion.

## 7. The Plane-of-Incidence Invariant (the 2D oracle)

The vector Snell formula combines only `d` and `n̂`, so the refracted direction
never leaves `span{d, n̂}` — the plane of incidence. As λ sweeps, `t(λ)` traces a
curve **within that single plane**. Therefore:

- **2D**: the dispersion fan opens in the world plane.
- **3D**: each ray's fan opens in *its own* plane of incidence; the macroscopic
  rainbow is the envelope of all per-ray fans across the prism faces.

This is the basis of the correctness oracle. The 3D dielectric BSDF is verified by
projecting an incident ray and normal into their plane of incidence and comparing
the refracted directions and Fresnel terms against the **closed-form 2D Snell
solution** for a sweep of wavelengths and incidence angles. The 2D analytic
solution is the ground truth; the 3D code must match it to floating-point
tolerance. It also confirms HWSS stays geometrically coherent through dispersion:
all four companions bend within the same plane.

## 8. Sensor and Illuminant Pipeline

- **Color-matching functions**: tabulated CIE 1931 2° `x̄, ȳ, z̄` at 5 nm spacing,
  linearly interpolated. Each accumulated spectral sample contributes
  `radiance · cmf(λ) / pdf(λ)` to the pixel's running XYZ mean.
- **Illuminants**: tabulated SPDs for **D65, A, and E**, selectable at runtime.
  The emitter's spectrum is one of these.
- **XYZ → sRGB**: standard matrix under the chosen white point, followed by tone
  mapping (start with simple exposure + Reinhard or a filmic curve; gamma encode).
- **Metamerism, for free**: render colored Lambertian swatches chosen to be a
  metameric pair under D65, then switch the illuminant to A. Under D65 the
  swatches match; under A their chromaticities diverge. This is the second Phase 0
  acceptance demo and requires no new code beyond illuminant selection.

## 9. Progressive Viewer

- Maintain an **unbounded XYZ accumulation buffer** (sum + sample count per pixel).
  Each frame asks the active `Renderer` to add `samples_per_pixel` more samples.
- Display = tone-mapped running mean, recomputed each frame.
- **Reset on camera move**: any camera change zeroes the accumulator so stale
  samples do not ghost. Grainy first frames, converging to clean when the camera
  is still — this is the watched MC convergence.
- Camera: orbit/pan/zoom via winit input. Pinhole camera with adjustable exposure.

## 10. Build Strategy — CPU Oracle, GPU Mirror, Diff Gate

Approach: build the CPU reference tracer in `spectral-core` first (debuggable,
analytically verifiable), then mirror its physics in `spectral-gpu`. The CPU
tracer is permanent regression infrastructure, not a throwaway prototype.

The diff gate is a real CI invariant only if all four of these are pinned:

1. **Deterministic seedable RNG**: the same seed produces the same sample
   sequence on both backends (e.g. PCG/`xoshiro` with explicit per-pixel,
   per-sample stream keys). CPU and GPU must consume randomness in the same order.
2. **Fixed tiny reference scene**: one dielectric sphere + one Lambertian plane +
   one illuminant, fixed camera, fixed resolution (e.g. 64×64).
3. **Tolerance metric**: per-pixel L1 distance in spectral radiance (pre-sensor)
   **and** total-energy delta across the frame, each under an explicit threshold.
4. **Fixed sample budget**: a set number of samples per pixel so the comparison is
   between converged-enough images, not noise.

CI fails if the GPU output drifts from the CPU oracle beyond tolerance. The 2D
Snell oracle (§7) is a separate, finer-grained unit-test gate on the BSDF itself.

## 11. Acceptance Scenes (Phase 0 complete = both pass)

### Scene A — Direct-view prism (dispersion)
- BK7 triangular prism (5 half-spaces), illuminated by a D65 source positioned so
  the **camera looks at/through the prism** with the light or a bright environment
  behind it. This is a direct view of dispersion, **not** a projected caustic, so
  it resolves cleanly under unidirectional path tracing.
- **Gate**: the refracted spectrum's angular spread and chromatic ordering (red
  deviated least, blue most) match the analytic Sellmeier prism-deviation
  prediction within a stated tolerance (target: angular error < a few percent).

The projected-rainbow-on-a-wall caustic is explicitly deferred: it is the forward
projection of all per-ray fans converging on a screen, which unidirectional
backward path tracing handles poorly. It returns once a later phase adds light
tracing or bidirectional path tracing.

### Scene B — Metamerism (accurate color)
- Two Lambertian swatches forming a metameric pair under D65.
- **Gate**: under D65 their rendered chromaticities agree within tolerance; after
  switching the illuminant to A they diverge by a stated minimum ΔE. Demonstrates
  the spectral sensor is doing real color science, not RGB tinting.

## 12. Phase Boundaries and Forward Seams

- **Fluorescence (Phase 3) fights HWSS — known sharp edge.** A reradiation event
  re-emits at a *different* wavelength, breaking the companion comb. Phase 3 will
  choose one of: (a) drop to single-wavelength transport after the first
  reradiation event, or (b) carry fluorescent paths bispectrally (Mojzík/Fichet).
  Flagged now, decided in Phase 3's spec. Phase 0 leaves the `SpectralSample`
  carrier and MIS structure able to accommodate either.
- **Wave solver (Phase 4) is a sibling, not a plugin.** A wave solver and a path
  tracer are different formulations of transport and do not compose at the BSDF
  layer. The honest shared seam is the `Renderer` trait (§3): same `Scene`, same
  sensor, swap the engine behind the trait. Phase 4 is a third `Renderer`
  implementation, not an addition to the path graph.

## 13. Risks and Open Questions

- **GPU/CPU RNG ordering parity** is the highest-risk item: divergent random
  consumption silently breaks the diff gate. Mitigation: explicit stream keys and
  a unit test that dumps the first N draws from each backend and asserts equality.
- **Tone-mapping choice** affects perceived correctness but not physical
  correctness; the gates operate pre-tone-map on radiance/chromaticity, so this
  stays a cosmetic decision.
- **Sellmeier domain**: coefficients assume λ in micrometers; unit slips here are
  a classic dispersion bug. Covered by the 2D oracle test.
- **wgpu version / WGSL feature set**: pin a known-good wgpu version early.

## 14. Testing Strategy

- **Unit (spectral-core)**: 2D Snell oracle (refraction angles, TIR onset, Fresnel
  R) across a wavelength × angle sweep; Sellmeier `n(λ)` against published BK7
  values; half-space CSG interval intersection; CIE integration of a D65 spectrum
  yielding near `(0.3127, 0.3290)` chromaticity for a neutral sensor; energy
  conservation at interfaces.
- **Integration (diff gate)**: GPU-vs-CPU on the fixed 64×64 scene (§10).
- **Acceptance**: Scenes A and B with their quantitative gates (§11).
- **RNG parity**: first-N-draws equality between CPU and GPU streams.

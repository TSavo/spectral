# GPU Forward + Volumetric DSOTM

Port the working CPU DSOTM renderer (`spectral-core::volume::render_volumetric_scene`,
the one that produced `prism_dsotm.png`: white beam → SF11 prism → rainbow fan glowing
in haze on black) to the GPU (wgpu compute), so it runs **live** in the viewer with a
SPACE toggle for dispersion. Built on the diff-gate-validated backward tracer (GPU-1..8,
commit `64cc077`, branch `gpu-mirror`).

## Why a new engine
The validated GPU tracer is **backward** (camera→scene) and **surface-only**. DSOTM is
**forward** (photons from a beam) + **single-scatter volumetric** (haze makes the beam
and fan visible). Different transport, new kernel — but it reuses the validated WGSL
Sellmeier `n(λ)`, Fresnel/TIR `scatter`, PCG `PathRng`, and CMF from `trace.wgsl`.

## Oracle (source of truth — mirror EXACTLY, do not "improve")
`render_volumetric_scene` in `spectral-core/src/volume.rs`. Per photon `i`:
- `rng = PathRng::new(i, seed)` — RNG consumption order is **load-bearing**:
  1. `next_f32()` → beam.u coefficient
  2. `next_f32()` → beam.v coefficient
  3. `next_f32()` → `lambda = LAMBDA_MIN + LAMBDA_RANGE * r`
  4. up to 8 bounces: intersect; `seg_len = hit.t || max_dist`;
     then `SEG_SAMPLES = 4` scatter samples each consuming `next_f32()` → `dist`;
     then if hit, `mat.scatter(...)` consumes its own draws (Fresnel roulette).
- Splat per scatter sample (point `p = origin + dir*dist`), if `camera.project(p)` is on
  screen and `d = |camO − p| < zbuffer[px]`:
  - `cos_theta = ray.dir · ((camO − p)/d)`  (angle to the **scatter→camera** vector, NOT cam forward)
  - `phase = phase_hg(g, cos_theta)`;  `trans = exp(−sigma_t·d)`
  - `contrib = power · sigma_s · phase · trans / d² · seg_len / SEG_SAMPLES`
  - `img[idx] += cmf(lambda) · contrib`   (XYZ)
- DSOTM uses `zbuffer = all INFINITY` (no occlusion); `g = 0.5` forward; `sigma_s = 0.5`,
  `sigma_t = 0.06`, `max_dist = 14`. Beam/prism/camera per `examples/prism_dsotm.rs`.

## Locked decisions
- **Film: 3× `atomic<u32>` per pixel** (X,Y,Z separate). No CAS loops, portable (keeps the
  WASM/WebGPU door open), contention-tolerant on hot pixels (beam axis, violet edge).
- **Fixed-point scale = 2¹²** (4096). Splat = `atomicAdd(round(channel · contrib · 2¹²))`.
- **Per-frame resolve→float→clear**: the u32 film is **transient per frame**. Each frame:
  splat a bounded photon batch → resolve u32→f32 (÷2¹²) → add into a persistent f32
  accumulator → zero the u32 film. Keeps per-frame integer sums well under `u32::MAX`
  (no overflow), and the f32 accumulator carries progressive convergence across frames.
- **Determinism**: integer atomicAdd is associative → order-independent → the kernel is
  reproducible regardless of GPU photon scheduling. This is what makes it diff-gateable.
- **Per-photon-index parity**: GPU photon `i` consumes the identical PCG sequence in the
  identical order as CPU photon `i` → the diff gate is near-exact, not merely statistical.

## Phases (each ends with a headless PNG the controller eyeballs before the next starts)
- **VOL-1 — atomic film + resolve + blit.** 3× atomic<u32> film, fixed-point splat helper,
  resolve→f32 accumulator, tonemap blit. No physics. PNG gate: a synthetic hash-based splat
  pattern rendered **twice with the same seed**, diffed → bit-identical (proves
  order-independent determinism). Headless render to PNG.
- **VOL-2 — forward photon kernel.** Beam sampling (corner + u·r + v·r, dir), forward
  propagation reusing validated WGSL Sellmeier/Fresnel/TIR `scatter`, mirrored RNG order.
  No haze yet — splat each photon's **exit/scatter points as dots**. PNG gate: the rainbow
  fan appears **as a constellation of points**. Parity test vs `lighttrace.rs` photon exits
  **must include photons that straddle the TIR/critical-angle boundary** (violet end, where
  n is highest) — not only safe transmits, or the gate green-lights a kernel that silently
  drops the violet edge.
- **VOL-3 — single-scatter camera connection + splat.** Full weight (phase·trans/d²·seg_len/4),
  scatter→camera angle, SEG_SAMPLES=4 matched to oracle. PNG gate: full DSOTM fan, noisy/low-spp.
- **VOL-4 — diff gate vs `render_volumetric_scene`.** Fixed scene + matched per-photon
  sampling + seeded RNG; assert GPU film ≈ CPU oracle (per-pixel L1 + energy, same tolerance
  discipline as GPU-7). PNG gate: converged fan visibly matches `prism_dsotm.png`.
- **VOL-5 — surface composite.** Backward rim-lit dark-glass prism pass (its own kernel pass,
  thin material) composited over the volumetric film — two passes into one film. Not a
  one-liner; wire both into the render graph. PNG gate: prism glass body + glowing fan.
- **VOL-6 — live viewer.** Replace the (blank, abandoned) backward viewer: progressive photon
  batches/frame into the persistent accumulator, wgpu blit, orbit camera (reset on move),
  SPACE toggles `n(λ)` vs `n(550)` (beam stays white, fan collapses). Fix the winit redraw
  kick (`request_redraw()` in `resumed`, `ControlFlow::Poll`). Controller verifies live.

## Rules
- Don't bundle phases. Each phase: build → headless PNG → controller eyeballs → next.
- Mirror the oracle's sampling/weights exactly; the diff gate is only a gate if both sides
  integrate the same estimator.
- Keep the existing backward diff gate (GPU-7) green throughout — the new kernel is additive.

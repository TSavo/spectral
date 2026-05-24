//! Three witnesses that spectral transport (carry the spectrum, project at the
//! sensor) dominates RGB transport (project to RGB first, then multiply).
//!
//! Witness 1 — Reduction: flat reflectance → spectral == RGB (agreement).
//! Witness 2 — Divergence by metamerism: RGB conflates a metameric pair that
//!             spectral separates under illuminant A.
//! Witness 3 — Divergence by bias: narrowband light on a wider notch surface →
//!             spectral ≈ black; RGB stays bright because the notch barely dents
//!             the broadband albedo.

use spectral_core::cie::{chromaticity, Illuminant, Sensor};
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

const STEP: f32 = 5.0;

fn nms() -> Vec<f32> {
    let mut v = Vec::new();
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        v.push(nm);
        nm += STEP;
    }
    v
}

// Ground truth: ∫ L(λ) ρ(λ) CMF(λ) dλ  ("transport then project").
fn spectral_shade(sensor: &Sensor, nms: &[f32], light: &[f32], rho: &[f32]) -> [f32; 3] {
    let mut xyz = [0.0f32; 3];
    for (i, &nm) in nms.iter().enumerate() {
        let s = light[i] * rho[i];
        let (x, y, z) = sensor.cmf(nm);
        xyz[0] += s * x * STEP;
        xyz[1] += s * y * STEP;
        xyz[2] += s * z * STEP;
    }
    xyz
}

// Per-channel sensor albedo: ∫ρ·c̄ / ∫c̄, so a white reflectance maps to (1,1,1).
fn albedo_xyz(sensor: &Sensor, nms: &[f32], rho: &[f32]) -> [f32; 3] {
    let (mut num, mut den) = ([0.0f32; 3], [0.0f32; 3]);
    for (i, &nm) in nms.iter().enumerate() {
        let (x, y, z) = sensor.cmf(nm);
        num[0] += rho[i] * x;
        num[1] += rho[i] * y;
        num[2] += rho[i] * z;
        den[0] += x;
        den[1] += y;
        den[2] += z;
    }
    [num[0] / den[0], num[1] / den[1], num[2] / den[2]]
}

// RGB transport: project light to XYZ, multiply by per-channel albedo ("project then transport").
fn rgb_transport_shade(sensor: &Sensor, nms: &[f32], light: &[f32], rho: &[f32]) -> [f32; 3] {
    let mut light_xyz = [0.0f32; 3];
    for (i, &nm) in nms.iter().enumerate() {
        let (x, y, z) = sensor.cmf(nm);
        light_xyz[0] += light[i] * x * STEP;
        light_xyz[1] += light[i] * y * STEP;
        light_xyz[2] += light[i] * z * STEP;
    }
    let a = albedo_xyz(sensor, nms, rho);
    [light_xyz[0] * a[0], light_xyz[1] * a[1], light_xyz[2] * a[2]]
}

fn d65(sensor: &Sensor, nms: &[f32]) -> Vec<f32> {
    nms.iter()
        .map(|&n| sensor.illuminant(Illuminant::D65, n))
        .collect()
}
fn illum_a(sensor: &Sensor, nms: &[f32]) -> Vec<f32> {
    nms.iter()
        .map(|&n| sensor.illuminant(Illuminant::A, n))
        .collect()
}

// ─── Witness 1 ───────────────────────────────────────────────────────────────

/// Reduction (≥): an achromatic surface makes spectral and RGB agree.
///
/// When ρ is flat, albedo_xyz = (ρ, ρ, ρ) exactly, so both sides reduce to
/// ρ · ∫ L·CMF dλ — the projection order doesn't matter.
#[test]
fn achromatic_surface_matches_rgb() {
    let s = Sensor::new();
    let nms = nms();
    let light = d65(&s, &nms);
    let rho = vec![0.6f32; nms.len()]; // flat gray
    let g = spectral_shade(&s, &nms, &light, &rho);
    let r = rgb_transport_shade(&s, &nms, &light, &rho);
    // With no sub-primary structure, project-then-multiply == multiply-then-project.
    for c in 0..3 {
        let rel = (g[c] - r[c]).abs() / g[c].max(1e-6);
        assert!(
            rel < 1e-3,
            "channel {c}: spectral {} vs rgb {} (rel {rel})",
            g[c],
            r[c]
        );
    }
}

// ─── Witness 2 ───────────────────────────────────────────────────────────────

/// Build a metameric black Δ(λ) orthogonal to {x̄,ȳ,z̄} under the sampled
/// inner product (Gram-Schmidt), so adding it leaves the sensor albedo unchanged.
fn metameric_black(sensor: &Sensor, nms: &[f32]) -> Vec<f32> {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let xb: Vec<f32> = nms.iter().map(|&n| sensor.cmf(n).0).collect();
    let yb: Vec<f32> = nms.iter().map(|&n| sensor.cmf(n).1).collect();
    let zb: Vec<f32> = nms.iter().map(|&n| sensor.cmf(n).2).collect();
    // Orthonormal basis of span{xb,yb,zb}.
    let mut basis: Vec<Vec<f32>> = Vec::new();
    for v in [xb, yb, zb] {
        let mut u = v.clone();
        for e in &basis {
            let d = dot(&u, e);
            for i in 0..u.len() {
                u[i] -= d * e[i];
            }
        }
        let norm = dot(&u, &u).sqrt();
        if norm > 1e-6 {
            for x in u.iter_mut() {
                *x /= norm;
            }
            basis.push(u);
        }
    }
    // A wiggle, then remove its components along the CMF span -> metameric black.
    // 3.0 cycles across the visible band; try 2.0 or 4.0 if dE under A is weak.
    let tau = std::f32::consts::TAU;
    let mut w: Vec<f32> = nms
        .iter()
        .map(|&n| ((n - 380.0) / 350.0 * 3.0 * tau).sin())
        .collect();
    for e in &basis {
        let d = dot(&w, e);
        for i in 0..w.len() {
            w[i] -= d * e[i];
        }
    }
    w
}

/// Divergence by metamerism (strict >): RGB conflates a pair spectral separates.
///
/// ρ2 = ρ1 + α·Δ where Δ is orthogonal to all three CMFs, so albedo_xyz(ρ1)
/// == albedo_xyz(ρ2). Under illuminant A (redder than equal-energy), Δ's
/// spectral shape produces a perceptible chromaticity shift that RGB cannot
/// represent.
#[test]
fn metameric_pair_conflated_by_rgb_separated_by_spectral() {
    let s = Sensor::new();
    let nms = nms();
    let delta = metameric_black(&s, &nms);
    let maxd = delta
        .iter()
        .fold(0.0f32, |m, &v| m.max(v.abs()))
        .max(1e-6);
    let alpha = 0.35 / maxd; // keep 0.5 + alpha*delta within [0.15, 0.85]
    let rho1 = vec![0.5f32; nms.len()];
    let rho2: Vec<f32> = (0..nms.len()).map(|i| 0.5 + alpha * delta[i]).collect();
    assert!(
        rho2.iter().all(|&r| (0.0..=1.0).contains(&r)),
        "rho2 left [0,1]; lower alpha"
    );

    // RGB transport conflates them: identical sensor albedo => identical under ANY light.
    let a1 = albedo_xyz(&s, &nms, &rho1);
    let a2 = albedo_xyz(&s, &nms, &rho2);
    for c in 0..3 {
        assert!(
            (a1[c] - a2[c]).abs() < 1e-4,
            "albedo channel {c} differs: {} vs {}",
            a1[c],
            a2[c]
        );
    }

    // Spectral separates them under illuminant A.
    let a_light = illum_a(&s, &nms);
    let g1 = spectral_shade(&s, &nms, &a_light, &rho1);
    let g2 = spectral_shade(&s, &nms, &a_light, &rho2);
    let (x1, y1) = chromaticity(g1);
    let (x2, y2) = chromaticity(g2);
    let de = ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt();
    assert!(
        de > 5e-3,
        "spectral should separate the metamers under A, got dE {de}"
    );
}

// ─── Witness 3 ───────────────────────────────────────────────────────────────

/// Gaussian bell centered at `center`, half-width `width`.
fn gaussian(nms: &[f32], center: f32, width: f32) -> Vec<f32> {
    nms.iter()
        .map(|&n| (-((n - center) / width).powi(2)).exp())
        .collect()
}

/// Divergence by bias: narrowband light on a notch surface.
///
/// Light: narrow Gaussian at 589 nm (σ=8 nm).
/// Surface: broadband white with a wide absorbing notch at 589 nm (σ=40 nm).
///   The notch is wider than the light so it swallows essentially all the
///   incident power. The notch width must exceed the light width for full
///   cancellation; equal widths (σ=8) only kill ~29% not ~100%.
///
/// Spectral: L·(1-notch) integrand ≈ 0 → nearly black.
/// RGB: the notch dents the broadband albedo only ~4%, times a yellow light →
///   stays clearly nonzero.
#[test]
fn narrowband_light_on_notch_surface_diverges() {
    let s = Sensor::new();
    let nms = nms();
    // Narrow yellow light (σ=8 nm), wide absorbing notch (σ=40 nm).
    // σ_notch >> σ_light ensures the notch fully swallows the light.
    let light = gaussian(&nms, 589.0, 8.0);
    let notch = gaussian(&nms, 589.0, 40.0);
    let rho: Vec<f32> = notch.iter().map(|g| 1.0 - g).collect(); // white with a wide notch at 589
    let g = spectral_shade(&s, &nms, &light, &rho);
    let r = rgb_transport_shade(&s, &nms, &light, &rho);
    // For reference, the same light on a perfect white surface (no notch).
    let white = vec![1.0f32; nms.len()];
    let g_white = spectral_shade(&s, &nms, &light, &white);
    // Spectral: nearly black (light absorbed). RGB: a large fraction of the white result.
    assert!(
        g[1] < 0.05 * g_white[1],
        "spectral should be ~black: {} vs white {}",
        g[1],
        g_white[1]
    );
    assert!(
        r[1] > 0.5 * g_white[1],
        "rgb transport should stay bright (can't see the notch): {} vs white {}",
        r[1],
        g_white[1]
    );
}

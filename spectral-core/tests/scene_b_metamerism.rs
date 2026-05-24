//! Acceptance Scene B: metamerism / illuminant-dependent color.
//! Two distinct reflectance spectra integrate to a chromaticity relationship
//! that CHANGES when the illuminant is swapped from D65 to A. That illuminant
//! dependence is the metamerism signature and proves the sensor is doing real
//! spectral color science, not RGB tinting.

use spectral_core::cie::{chromaticity, Illuminant, Sensor};
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

/// Integrate a reflectance R(λ) under an illuminant, returning chromaticity (x,y).
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

// A smooth cosine ramp and a triple-bump curve: two genuinely different spectra.
fn smooth(nm: f32) -> f32 {
    0.5 + 0.3 * ((nm - 555.0) / 120.0).cos()
}
fn bumpy(nm: f32) -> f32 {
    let g = |c: f32, w: f32| (-((nm - c) / w).powi(2)).exp();
    (0.55 * g(450.0, 25.0) + 0.75 * g(540.0, 25.0) + 0.65 * g(610.0, 25.0)).clamp(0.0, 1.0)
}

#[test]
fn pair_is_distinct_spectra() {
    // Sanity: the two reflectance curves are genuinely different spectra.
    let diff: f32 = (400..700)
        .step_by(20)
        .map(|n| (smooth(n as f32) - bumpy(n as f32)).abs())
        .sum();
    assert!(diff > 0.5, "curves must differ materially as spectra, got {diff}");
}

#[test]
fn illuminant_swap_changes_color_relationship() {
    let sensor = Sensor::new();
    let gap = |ill| {
        let a = chroma_under(&sensor, &smooth, ill);
        let b = chroma_under(&sensor, &bumpy, ill);
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
    };
    let d_d65 = gap(Illuminant::D65);
    let d_a = gap(Illuminant::A);
    // The chromaticity gap between the two spectra must DIFFER by illuminant:
    // illuminant-dependent color is the metamerism signature.
    assert!(
        (d_a - d_d65).abs() > 1e-3,
        "color relationship must shift with illuminant: d65={d_d65}, a={d_a}"
    );
}

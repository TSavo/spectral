//! The renderer predicts the color of the sky. White sunlight (D65), weighted by
//! Rayleigh's 1/λ⁴ scattering and integrated through the CIE sensor, comes out
//! BLUE (the scattered sky); after a long atmospheric path it comes out RED (the
//! sunset). Nothing about "blue" or "red" is coded — it falls out of 1/λ⁴.

use spectral_core::cie::{chromaticity, xyz_to_linear_srgb, Illuminant, Sensor};
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};

/// ∫ D65(λ) · weight(λ) · cmf(λ) dλ over the visible band (5nm steps).
fn integrate(sensor: &Sensor, weight: &dyn Fn(f32) -> f32) -> [f32; 3] {
    let step = 5.0_f32;
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        let s = sensor.illuminant(Illuminant::D65, nm) * weight(nm);
        let (xb, yb, zb) = sensor.cmf(nm);
        x += s * xb * step;
        y += s * yb * step;
        z += s * zb * step;
        nm += step;
    }
    [x, y, z]
}

fn rayleigh(nm: f32) -> f32 {
    (550.0 / nm).powi(4)
}

#[test]
fn scattered_sky_is_blue() {
    let sensor = Sensor::new();
    let xyz = integrate(&sensor, &rayleigh);
    let (x, y) = chromaticity(xyz);
    let rgb = xyz_to_linear_srgb(xyz);
    // Bluer than the D65 white point (0.3127, 0.3290): chromaticity shifts toward
    // blue, and the blue sRGB channel dominates red.
    assert!(x < 0.28 && y < 0.33, "scattered sky chromaticity ({x},{y}) not blue");
    assert!(rgb[2] > rgb[0], "sky: blue channel must exceed red, got {rgb:?}");
}

#[test]
fn sunset_transmitted_is_red() {
    let sensor = Sensor::new();
    // Long horizon path: heavy 1/λ⁴ extinction. Tune k so blue is mostly removed.
    let k = 1.5_f32;
    let xyz = integrate(&sensor, &|nm| (-k * rayleigh(nm)).exp());
    let (x, _y) = chromaticity(xyz);
    let rgb = xyz_to_linear_srgb(xyz);
    assert!(x > 0.40, "sunset chromaticity x={x} not warm/red enough");
    assert!(rgb[0] > rgb[2], "sunset: red channel must exceed blue, got {rgb:?}");
}

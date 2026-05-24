//! The sky, rendered. Rayleigh single-scatter (σ ∝ 1/λ⁴) gives the blue zenith
//! brightening toward the horizon, and the sun reddens as its light takes a longer
//! path through the air. All colors via the verified CIE sensor. Writes sky.png.

use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb, Illuminant, Sensor};
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};
use spectral_core::Xyz;

/// Rayleigh optical depth at the zenith (m=1), scaled by 1/λ⁴ from 550nm.
fn tau(lambda: f32) -> f32 {
    0.12 * (550.0 / lambda).powi(4)
}

/// Integrate D65 sunlight times a spectral weight through the CIE sensor.
fn integrate(sensor: &Sensor, weight: &dyn Fn(f32) -> f32) -> Xyz {
    let step = 5.0f32;
    let mut xyz = [0.0f32; 3];
    let mut nm = LAMBDA_MIN;
    while nm <= LAMBDA_MAX {
        let s = sensor.illuminant(Illuminant::D65, nm) * weight(nm);
        let (xb, yb, zb) = sensor.cmf(nm);
        xyz[0] += s * xb * step;
        xyz[1] += s * yb * step;
        xyz[2] += s * zb * step;
        nm += step;
    }
    xyz
}

fn main() {
    let (w, h) = (800usize, 520usize);
    let sensor = Sensor::new();

    // Sun: low in the sky (long air mass) so it reddens. Placed near the horizon.
    let sun_cx = 0.30f32;
    let sun_cy = 0.86f32; // fraction down the image (near the bottom/horizon)
    let sun_r = 0.045f32;
    let sun_airmass = 11.0f32; // long slant path -> sunset reddening
    let sun_xyz = integrate(&sensor, &|l| (-tau(l) * sun_airmass).exp());

    let mut img = image::RgbImage::new(w as u32, h as u32);
    // Common exposure so sky and sun share a scale.
    let exposure = 0.6 / sun_xyz[1].max(1e-6);

    for y in 0..h {
        // Altitude angle: ~88° at the top (zenith), ~2° at the bottom (horizon).
        let frac = y as f32 / h as f32;
        let altitude_deg = 88.0 - frac * 86.0;
        let sin_alt = altitude_deg.to_radians().sin().max(0.03);
        let airmass = 1.0 / sin_alt; // grows toward the horizon

        // Scattered sky color: fraction of each wavelength scattered into the eye.
        let sky_xyz = integrate(&sensor, &|l| 1.0 - (-tau(l) * airmass).exp());

        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            // Aspect-correct distance to the sun center.
            let dx = (fx - sun_cx) * (w as f32 / h as f32);
            let dy = fy - sun_cy;
            let xyz = if (dx * dx + dy * dy).sqrt() < sun_r {
                sun_xyz
            } else {
                sky_xyz
            };
            let rgb = tonemap_to_u8(xyz_to_linear_srgb(xyz), exposure);
            img.put_pixel(x as u32, y as u32, image::Rgb(rgb));
        }
    }
    img.save("sky.png").unwrap();
    println!("wrote sky.png ({w}x{h})");
}

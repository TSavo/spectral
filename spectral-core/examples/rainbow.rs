//! The rainbow, rendered. Sweep impact parameters and wavelengths through a water
//! sphere with the renderer's own refract/reflect/Fresnel; bin each exiting ray by
//! the angle it is seen at (180 - scattering angle) and weight by Fresnel
//! throughput. The caustic concentration paints the bands: primary ~42°, the dark
//! Alexander band, secondary ~51° (reversed). Writes rainbow.png.

use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb, Illuminant, Sensor};
use spectral_core::geom::{Ray, Sphere};
use spectral_core::optics::{fresnel_reflectance, reflect, refract};
use spectral_core::sellmeier::Glass;
use spectral_core::spectrum::{LAMBDA_MAX, LAMBDA_MIN};
use glam::Vec3;

/// Trace a +X ray at impact parameter `b` through a unit water sphere with
/// `bounces` internal reflections. Returns (rainbow angle in degrees, Fresnel
/// throughput) or None on TIR/miss.
fn trace_drop(b: f32, lambda: f32, bounces: u32) -> Option<(f32, f32)> {
    let n = Glass::Water.n(lambda);
    let sphere = Sphere { center: Vec3::ZERO, radius: 1.0 };
    let d0 = Vec3::X;
    let mut ray = Ray { origin: Vec3::new(-3.0, b, 0.0), dir: d0 };

    // Enter (air -> water): transmit fraction.
    let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
    let cos_i = (-ray.dir.dot(h.normal)).abs();
    let mut tput = 1.0 - fresnel_reflectance(cos_i, 1.0, n);
    let mut dir = refract(ray.dir, h.normal, 1.0, n)?;
    ray = Ray { origin: h.point, dir };

    // Internal reflections: each costs the internal reflectance.
    for _ in 0..bounces {
        let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
        let cos = ray.dir.dot(h.normal).abs();
        tput *= fresnel_reflectance(cos, n, 1.0);
        dir = reflect(ray.dir, h.normal);
        ray = Ray { origin: h.point, dir };
    }

    // Exit (water -> air): transmit fraction.
    let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
    let cos_o = ray.dir.dot(h.normal).abs();
    tput *= 1.0 - fresnel_reflectance(cos_o, n, 1.0);
    let out = refract(ray.dir, h.normal, n, 1.0)?;

    // Angle the exiting ray is seen at, measured from the incoming direction.
    // acos folds the secondary's >180° deviation back so both bands land correctly.
    let scatter = d0.dot(out).clamp(-1.0, 1.0).acos().to_degrees();
    Some((180.0 - scatter, tput))
}

fn main() {
    let (w, h) = (900usize, 240usize);
    let (ang_lo, ang_hi) = (30.0f32, 56.0f32); // covers primary, Alexander band, secondary
    let sensor = Sensor::new();

    // Accumulate XYZ per angle column.
    let mut cols = vec![[0.0f32; 3]; w];
    let b_steps = 6000;
    let lam_step = 2.0f32;
    for bounces in [1u32, 2] {
        for i in 1..b_steps {
            let b = i as f32 / b_steps as f32 * 0.9995;
            let mut lam = LAMBDA_MIN;
            while lam <= LAMBDA_MAX {
                if let Some((angle, tput)) = trace_drop(b, lam, bounces) {
                    if (ang_lo..ang_hi).contains(&angle) {
                        let col = (((angle - ang_lo) / (ang_hi - ang_lo)) * w as f32) as usize;
                        let col = col.min(w - 1);
                        let (xb, yb, zb) = sensor.cmf(lam);
                        // weight: sunlight (D65) * fresnel throughput * dλ
                        let sun = sensor.illuminant(Illuminant::D65, lam);
                        let g = tput * sun * lam_step;
                        cols[col][0] += xb * g;
                        cols[col][1] += yb * g;
                        cols[col][2] += zb * g;
                    }
                }
                lam += lam_step;
            }
        }
    }

    // Auto-expose to the 99th-percentile column luminance.
    let mut lums: Vec<f32> = cols.iter().map(|c| c[1]).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = lums[(w as f32 * 0.99) as usize].max(1e-9);
    let exposure = 1.4 / p99;

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (x, col) in cols.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*col), exposure);
        for y in 0..h {
            img.put_pixel(x as u32, y as u32, image::Rgb(rgb));
        }
    }
    img.save("rainbow.png").unwrap();
    println!("wrote rainbow.png ({w}x{h}, angle {ang_lo}..{ang_hi} deg)");
}

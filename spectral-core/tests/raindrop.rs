//! The renderer predicts the rainbow: trace sunlight through a water sphere with
//! the same refract/reflect/Fresnel optics used everywhere else, and recover the
//! primary (42°) and secondary (51°) rainbow angles from where the deviation is
//! stationary. Nothing about "42 degrees" is coded — it emerges from n(λ).

use spectral_core::geom::{Ray, Sphere};
use spectral_core::optics::{reflect, refract};
use spectral_core::sellmeier::Glass;
use glam::Vec3;

const R: f32 = 1.0;

/// Total cumulative deviation (degrees) of a +X ray at impact parameter `b`
/// through a water sphere, with `bounces` internal reflections.
///
/// Rather than using `acos(d0 · out)` — which is unsigned and collapses
/// angles beyond 180° — we track cumulative signed angular deflection in the
/// XY plane using `atan2`. Each segment contributes the signed angle between
/// consecutive direction vectors; we sum the absolute values to get the total
/// geometric deviation D. For the primary (one bounce) D ∈ [138°, 180°]; for
/// the secondary (two bounces) D ∈ [231°, 360°].
///
/// Returns None if the path TIRs or misses at any stage.
fn deviation_deg(b: f32, lambda: f32, bounces: u32) -> Option<f32> {
    let n = Glass::Water.n(lambda);
    let sphere = Sphere { center: Vec3::ZERO, radius: R };
    let d0 = Vec3::X;
    let mut ray = Ray { origin: Vec3::new(-3.0, b, 0.0), dir: d0 };

    // Collect direction vectors at each interface crossing so we can sum
    // the turn angles.
    let mut dirs: Vec<Vec3> = vec![d0];

    // Enter the sphere (air -> water).
    let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
    let dir = refract(ray.dir, h.normal, 1.0, n)?;
    ray = Ray { origin: h.point, dir };
    dirs.push(dir);

    // `bounces` internal reflections.
    for _ in 0..bounces {
        let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
        let dir = reflect(ray.dir, h.normal);
        ray = Ray { origin: h.point, dir };
        dirs.push(dir);
    }

    // Exit the sphere (water -> air).
    let h = sphere.intersect(&ray, 1e-4, f32::INFINITY)?;
    let out = refract(ray.dir, h.normal, n, 1.0)?; // None on TIR
    dirs.push(out);

    // Sum the turn angles between consecutive direction segments.
    // Each turn angle = angle between consecutive unit direction vectors.
    // We use the signed 2D angle in the XY plane (the geometry is planar).
    let mut total_deg = 0.0_f32;
    for w in dirs.windows(2) {
        let a = w[0];
        let b = w[1];
        // signed angle from a to b in XY plane: atan2(a x b, a . b)
        let cross_z = a.x * b.y - a.y * b.x; // z-component of a × b
        let dot = a.x * b.x + a.y * b.y;
        let angle = cross_z.atan2(dot).to_degrees().abs();
        total_deg += angle;
    }

    Some(total_deg)
}

/// Scan impact parameters and return the minimum cumulative deviation (the
/// rainbow caustic) for `bounces` internal reflections at wavelength `lambda`.
fn min_deviation(lambda: f32, bounces: u32) -> f32 {
    let mut best = f32::INFINITY;
    for i in 1..1000 {
        let b = i as f32 / 1000.0 * 0.999 * R;
        if let Some(dev) = deviation_deg(b, lambda, bounces) {
            if dev < best {
                best = dev;
            }
        }
    }
    best
}

#[test]
fn primary_rainbow_is_42_degrees() {
    // Primary: one internal reflection. Rainbow angle = 180 - min deviation.
    // (min deviation ≈ 138° → rainbow angle ≈ 42°)
    let green_dev = min_deviation(550.0, 1);
    let green = 180.0 - green_dev;
    println!("primary min_deviation(550nm, 1 bounce) = {green_dev:.4}°  → rainbow angle = {green:.4}°");
    assert!((green - 42.0).abs() < 2.0, "primary rainbow angle {green:.4}°, expected ~42°");
}

#[test]
fn primary_red_is_outside_violet() {
    // Red bends less (lower n) -> larger rainbow angle -> outer arc.
    let red_dev = min_deviation(650.0, 1);
    let violet_dev = min_deviation(450.0, 1);
    let red = 180.0 - red_dev;
    let violet = 180.0 - violet_dev;
    println!("primary: red={red:.4}°  violet={violet:.4}°  (red must be outside i.e. larger angle)");
    assert!(red > violet, "primary: red {red:.4}° should be outside violet {violet:.4}°");
}

#[test]
fn secondary_rainbow_is_51_degrees_and_reversed() {
    // Secondary: two internal reflections.
    // Cumulative deviation D_min ≈ 231° → secondary rainbow angle = D_min - 180° ≈ 51°.
    // Colors are reversed: violet outermost (larger rainbow angle).
    let green_dev = min_deviation(550.0, 2);
    let red_dev   = min_deviation(650.0, 2);
    let violet_dev = min_deviation(450.0, 2);
    let green  = green_dev  - 180.0;
    let red    = red_dev    - 180.0;
    let violet = violet_dev - 180.0;
    println!("secondary raw deviations: red={red_dev:.4}°  green={green_dev:.4}°  violet={violet_dev:.4}°");
    println!("secondary rainbow angles:  red={red:.4}°  green={green:.4}°  violet={violet:.4}°");
    println!("secondary convention: min_deviation - 180° (two-bounce D_min > 180°)");
    assert!((green - 51.0).abs() < 3.0, "secondary rainbow angle {green:.4}°, expected ~51°");
    assert!(violet > red, "secondary must be color-reversed: violet {violet:.4}° outside red {red:.4}°");
}

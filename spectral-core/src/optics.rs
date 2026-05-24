//! Vector Snell refraction, Fresnel reflectance, total internal reflection.
//! Verified against the analytic 2D solution in the plane of incidence.

use glam::Vec3;

/// Mirror reflection of incident direction `d` about `n`. `d` points toward
/// the surface; result points away.
pub fn reflect(d: Vec3, n: Vec3) -> Vec3 {
    d - 2.0 * d.dot(n) * n
}

/// Vector Snell refraction. `d` is the incident direction (toward the surface),
/// `n` is the surface normal on the incoming side. `n1`/`n2` are the indices
/// before/after the interface.
/// If `d` and `n` lie on the same side (`d.dot(n) > 0`), `n` is flipped
/// internally, so the result is independent of which way the caller orients
/// the normal (works for both entering and exiting a medium).
/// Returns the transmitted direction propagating
/// INTO the second medium (forward across the interface), or None on total
/// internal reflection.
pub fn refract(d: Vec3, n: Vec3, n1: f32, n2: f32) -> Option<Vec3> {
    let eta = n1 / n2;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // POSITIVE: refracted angle matches Snell, measured from the TRANSMISSION-side
    // normal (-normal here), and the ray stays in the plane of incidence.
    #[test]
    fn refract_matches_snell_angle() {
        let (n1, n2) = (1.0_f32, 1.5_f32);
        let normal = Vec3::Y;
        for deg in [5.0_f32, 15.0, 30.0, 45.0, 60.0] {
            let theta_i = deg.to_radians();
            let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
            let t = refract(d, normal, n1, n2).expect("should transmit");
            let theta_t = t.dot(-normal).clamp(-1.0, 1.0).acos();
            let expected = (n1 / n2 * theta_i.sin()).asin();
            assert_relative_eq!(theta_t, expected, epsilon = 1e-5);
            assert!(t.z.abs() < 1e-6, "left plane of incidence: z={}", t.z);
        }
    }

    // DISCRIMINATION: the transmitted ray must travel FORWARD into medium 2.
    // The angle-only oracle also admits the backward (sign-flipped) solution;
    // these assertions reject it, and we record that the rejected solution is
    // exactly the negated vector.
    #[test]
    fn transmitted_ray_propagates_forward_not_backward() {
        let (n1, n2) = (1.0_f32, 1.5_f32);
        let normal = Vec3::Y;
        let theta_i = 40.0_f32.to_radians();
        let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
        let t = refract(d, normal, n1, n2).unwrap();
        assert!(t.dot(d) > 0.0, "refracted ray reversed direction: t.d = {}", t.dot(d));
        assert!(t.dot(normal) < 0.0, "refracted ray did not cross into medium 2");
        let bogus = -t; // the wrong solution with the same angle
        assert!(bogus.dot(d) < 0.0 && bogus.dot(normal) > 0.0,
            "the rejected solution is the backward one");
    }

    // DISCRIMINATION: Snell in vector form. The tangential component scales by
    // n1/n2 (n1 sinθi = n2 sinθt) AND keeps the same sense (the ray bends, it
    // never flips to the other side of the normal).
    #[test]
    fn tangential_component_obeys_snell_and_keeps_sense() {
        let (n1, n2) = (1.0_f32, 1.5_f32);
        let normal = Vec3::Y;
        let theta_i = 50.0_f32.to_radians();
        let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
        let t = refract(d, normal, n1, n2).unwrap();
        let dt = d - d.dot(normal) * normal;
        let tt = t - t.dot(normal) * normal;
        assert!(dt.dot(tt) > 0.0, "tangential sense flipped");
        assert_relative_eq!(n1 * dt.length(), n2 * tt.length(), epsilon = 1e-5);
    }

    // STRUCTURAL: refraction is reversible. Refract in, then refract the result
    // back across the same interface with indices swapped, and recover d.
    #[test]
    fn refraction_is_reversible() {
        let (n1, n2) = (1.0_f32, 1.5_f32);
        let normal = Vec3::Y;
        let theta_i = 35.0_f32.to_radians();
        let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
        let t = refract(d, normal, n1, n2).unwrap();
        let back = refract(t, normal, n2, n1).expect("should transmit back");
        assert_relative_eq!(back, d, epsilon = 1e-5);
    }

    #[test]
    fn total_internal_reflection_past_critical_angle() {
        let (n1, n2) = (1.5_f32, 1.0_f32);
        let theta_c = (n2 / n1).asin();
        let normal = Vec3::Y;
        let past = theta_c + 0.05;
        let d = Vec3::new(past.sin(), -past.cos(), 0.0).normalize();
        assert!(refract(d, normal, n1, n2).is_none(), "must be TIR past critical angle");
        let inside = theta_c - 0.05;
        let d = Vec3::new(inside.sin(), -inside.cos(), 0.0).normalize();
        assert!(refract(d, normal, n1, n2).is_some(), "must transmit below critical angle");
    }

    #[test]
    fn fresnel_normal_incidence() {
        let r = fresnel_reflectance(1.0, 1.0, 1.5);
        assert_relative_eq!(r, ((1.0_f32 - 1.5) / (1.0 + 1.5)).powi(2), epsilon = 1e-6);
    }

    #[test]
    fn fresnel_grazing_approaches_one() {
        // cos_i = 1e-5 is ~89.999deg; analytic R here is ~0.99994, so a stub
        // returning a flat ~0.99x would fail this.
        let r = fresnel_reflectance(1e-5, 1.0, 1.5);
        assert!(r > 0.999, "near-grazing R should approach 1, got {r}");
    }

    // DISCRIMINATION: R rises monotonically from normal toward grazing and stays in [0,1].
    #[test]
    fn fresnel_monotonic_and_bounded() {
        let mut prev = fresnel_reflectance(1.0, 1.0, 1.5);
        for step in 1..=20 {
            let cos_i = 1.0 - step as f32 / 20.0 * 0.999;
            let r = fresnel_reflectance(cos_i, 1.0, 1.5);
            assert!((0.0..=1.0).contains(&r), "R out of [0,1]: {r}");
            assert!(r >= prev - 1e-6, "R must not decrease toward grazing: {r} < {prev}");
            prev = r;
        }
    }

    // DISCRIMINATION: the two TIR predicates must agree. fresnel_reflectance
    // returns exactly 1.0 iff refract returns None, for the same geometry.
    // Critical angle for 1.5->1.0 is asin(1/1.5) ~= 41.81deg; 41.0 transmits,
    // 42.0 is TIR. A bug in either function alone would break this.
    #[test]
    fn tir_predicates_agree() {
        let (n1, n2) = (1.5_f32, 1.0_f32);
        let normal = Vec3::Y;
        for deg in [10.0_f32, 30.0, 41.0, 42.0, 50.0, 70.0] {
            let theta = deg.to_radians();
            let d = Vec3::new(theta.sin(), -theta.cos(), 0.0).normalize();
            let cos_i = (-d.dot(normal)).abs();
            let r = fresnel_reflectance(cos_i, n1, n2);
            let transmits = refract(d, normal, n1, n2).is_some();
            assert_eq!(transmits, r < 1.0,
                "TIR disagreement at {deg}deg: transmits={transmits}, R={r}");
        }
    }

    #[test]
    fn reflect_is_mirror() {
        let d = Vec3::new(1.0, -1.0, 0.0).normalize();
        let r = reflect(d, Vec3::Y);
        assert_relative_eq!(r, Vec3::new(d.x, -d.y, 0.0).normalize(), epsilon = 1e-6);
        assert!(r.y > 0.0 && d.y < 0.0, "reflection must flip the normal component");
        // STRUCTURAL: reflection flips ONLY the normal component; both tangential
        // components (including z) are preserved.
        let d3 = Vec3::new(0.4, -0.7, 0.3).normalize();
        let r3 = reflect(d3, Vec3::Y);
        assert_relative_eq!(r3, Vec3::new(d3.x, -d3.y, d3.z), epsilon = 1e-6);
    }
}

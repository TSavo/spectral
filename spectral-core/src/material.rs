//! Surface materials. Lambertian diffuse and smooth dispersive dielectric.

use crate::geom::Hit;
use crate::optics::{fresnel_reflectance, reflect, refract};
use crate::rng::PathRng;
use crate::sellmeier::Glass;
use glam::Vec3;
use std::f32::consts::PI;

#[derive(Clone, Copy)]
pub enum Material {
    Lambertian { reflectance: f32 },
    /// Smooth dispersive dielectric. The tracer reads `glass` to compute the
    /// per-wavelength index `n_hero = glass.n(lambda)` and passes it into `scatter`.
    Dielectric { glass: Glass },
}

/// Result of scattering: a new ray direction and a throughput multiplier.
pub struct Scatter {
    pub dir: Vec3,
    pub weight: f32,
}

impl Material {
    /// `wo_in` is the incident direction (toward the surface). `n_hero` is the
    /// dielectric index at the hero wavelength; the dielectric branch decides
    /// reflect-vs-refract on the hero wavelength. `_hero_lambda` is reserved for
    /// future per-lane bookkeeping; the dielectric currently decides on the hero
    /// wavelength via the pre-computed `n_hero`.
    pub fn scatter(
        &self,
        wo_in: Vec3,
        hit: &Hit,
        _hero_lambda: f32,
        n_hero: f32,
        rng: &mut PathRng,
    ) -> Option<Scatter> {
        match *self {
            Material::Lambertian { reflectance } => {
                // Cosine-weighted sampling cancels the cosine/pdf, so weight = albedo.
                let dir = cosine_hemisphere(hit.normal, rng);
                Some(Scatter { dir, weight: reflectance })
            }
            Material::Dielectric { .. } => {
                let cos_i = (-wo_in.dot(hit.normal)).abs();
                // front_face distinguishes entering glass (1 -> n) from exiting (n -> 1).
                let (n1, n2) = if hit.front_face {
                    (1.0, n_hero)
                } else {
                    (n_hero, 1.0)
                };
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
    let a = if normal.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t = normal.cross(a).normalize();
    let b = normal.cross(t);
    (t * x + b * y + normal * z).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Hit;
    use crate::optics::reflect;
    use crate::rng::PathRng;
    use crate::sellmeier::Glass;
    use approx::assert_relative_eq;
    use glam::Vec3;

    #[test]
    fn lambertian_scatters_into_hemisphere() {
        let m = Material::Lambertian { reflectance: 0.8 };
        let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y, front_face: true };
        let wo = Vec3::new(0.0, -1.0, 0.0);
        for s in 0..200 {
            let mut rng = PathRng::new(s, 7);
            let sc = m.scatter(wo, &h, 550.0, 1.5, &mut rng).unwrap();
            assert!(sc.dir.dot(h.normal) > 0.0, "sample {s} scattered below the surface");
            assert_relative_eq!(sc.dir.length(), 1.0, epsilon = 1e-5);
            assert!((0.0..=1.0).contains(&sc.weight));
        }
    }

    #[test]
    fn dielectric_disperses_by_wavelength() {
        // Same incident ray, different wavelengths -> different refracted dirs.
        use crate::optics::refract;
        let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y, front_face: true };
        let d = Vec3::new(0.6, -0.8, 0.0).normalize();
        let t_blue = refract(d, h.normal, 1.0, Glass::Sf11.n(450.0)).unwrap();
        let t_red = refract(d, h.normal, 1.0, Glass::Sf11.n(650.0)).unwrap();
        assert!((t_blue - t_red).length() > 1e-3, "no dispersion observed");
    }

    // DISCRIMINATION: scatter MUST depend on front_face. Entering (1->n) and
    // exiting (n->1) at the same incident ray produce different outcomes; a bug
    // that ignored front_face would make the two sequences identical.
    #[test]
    fn dielectric_distinguishes_entering_from_exiting() {
        let m = Material::Dielectric { glass: Glass::Bk7 };
        let theta = 30.0_f32.to_radians();
        let wo = Vec3::new(theta.sin(), -theta.cos(), 0.0).normalize();
        let nh = Glass::Bk7.n(550.0);
        let entering: Vec<Vec3> = (0..32)
            .map(|s| {
                let mut rng = PathRng::new(s, 99);
                let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y, front_face: true };
                m.scatter(wo, &h, 550.0, nh, &mut rng).unwrap().dir
            })
            .collect();
        let exiting: Vec<Vec3> = (0..32)
            .map(|s| {
                let mut rng = PathRng::new(s, 99);
                let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y, front_face: false };
                m.scatter(wo, &h, 550.0, nh, &mut rng).unwrap().dir
            })
            .collect();
        assert!(entering != exiting, "scatter must depend on front_face");
    }

    // DISCRIMINATION: exiting glass past the critical angle is total internal
    // reflection (Fresnel R = 1), so scatter always reflects, for any rng draw.
    #[test]
    fn dielectric_tir_on_exit_always_reflects() {
        let m = Material::Dielectric { glass: Glass::Bk7 };
        let theta = 60.0_f32.to_radians(); // > critical (~41deg) for n~1.52 -> 1.0
        let wo = Vec3::new(theta.sin(), -theta.cos(), 0.0).normalize();
        let h = Hit { t: 1.0, point: Vec3::ZERO, normal: Vec3::Y, front_face: false };
        let expected = reflect(wo, Vec3::Y);
        for s in 0..8 {
            let mut rng = PathRng::new(s, 5);
            let sc = m.scatter(wo, &h, 550.0, Glass::Bk7.n(550.0), &mut rng).unwrap();
            assert_relative_eq!(sc.dir, expected, epsilon = 1e-5);
        }
    }
}

//! Volumetric forward light tracing. Photons scatter off a uniform haze; the
//! single-scattered radiance is splatted toward the camera, making the beam and
//! its dispersed fan visible in the air. The estimator randomly samples scatter
//! points along each path segment.

use crate::camera::Camera;
use crate::cie::Sensor;
use crate::geom::Ray;
use crate::lighttrace::Beam;
use crate::material::Material;
use crate::rng::PathRng;
use crate::scene::Scene;
use crate::spectrum::{LAMBDA_MIN, LAMBDA_RANGE};
use crate::Xyz;

/// Render the haze-scattered light from `n_photons` emitted by `beam` through
/// `scene`, as seen by `camera`. `sigma_s` is the haze scattering coefficient;
/// `max_dist` caps the length of a segment that escapes without hitting a surface.
/// Returns the accumulated XYZ image (width*height, row-major).
#[allow(clippy::too_many_arguments)]
pub fn render_volumetric(
    scene: &Scene,
    camera: &Camera,
    beam: &Beam,
    width: usize,
    height: usize,
    n_photons: u32,
    sigma_s: f32,
    max_dist: f32,
    seed: u64,
) -> Vec<Xyz> {
    let sensor = Sensor::new();
    let mut img = vec![[0.0f32; 3]; width * height];
    const SEG_SAMPLES: u32 = 4;
    const INV_4PI: f32 = 1.0 / (4.0 * std::f32::consts::PI);

    for i in 0..n_photons {
        let mut rng = PathRng::new(i as u64, seed);
        let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
        let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32();
        let (xb, yb, zb) = sensor.cmf(lambda);
        let mut ray = Ray { origin, dir: beam.dir };
        let mut power = 1.0f32;

        for _ in 0..8 {
            let hit = scene.intersect(&ray);
            let seg_len = hit.as_ref().map(|(h, _)| h.t).unwrap_or(max_dist);

            // Single-scatter splat: random points along this segment in the haze.
            for _ in 0..SEG_SAMPLES {
                let dist = rng.next_f32() * seg_len;
                let p = ray.origin + ray.dir * dist;
                if let Some((s, t, _depth)) = camera.project(p) {
                    let px = ((s * width as f32) as usize).min(width - 1);
                    let py = (((1.0 - t) * height as f32) as usize).min(height - 1);
                    // MC in-scatter estimate: power * sigma_s * (seg_len / N) * phase
                    let contrib = power * sigma_s * seg_len / SEG_SAMPLES as f32 * INV_4PI;
                    let idx = py * width + px;
                    img[idx][0] += xb * contrib;
                    img[idx][1] += yb * contrib;
                    img[idx][2] += zb * contrib;
                }
            }

            match hit {
                Some((h, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(lambda),
                        _ => 1.0,
                    };
                    match mat.scatter(ray.dir, &h, lambda, n_hero, &mut rng) {
                        Some(sc) => {
                            power *= sc.weight;
                            ray = Ray { origin: h.point, dir: sc.dir };
                        }
                        None => break,
                    }
                }
                None => break, // escaped (already splatted this last segment)
            }
        }
    }
    img
}

/// Henyey-Greenstein phase function. g in (-1,1): g>0 forward-scattering.
fn phase_hg(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6).powf(1.5);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom)
}

/// Single-scatter volumetric forward tracing with the full camera-connection
/// weight: phase(theta) * transmittance(exp(-sigma_t d)) / d^2, occluded by a
/// precomputed z-buffer of nearest-solid euclidean depths (INF where none). The
/// scatter points are sampled randomly along each path segment.
#[allow(clippy::too_many_arguments)]
pub fn render_volumetric_scene(
    scene: &Scene,
    camera: &Camera,
    beam: &Beam,
    width: usize,
    height: usize,
    n_photons: u32,
    sigma_s: f32,
    sigma_t: f32,
    g: f32,
    max_dist: f32,
    zbuffer: &[f32],
    seed: u64,
) -> Vec<Xyz> {
    let sensor = Sensor::new();
    let cam_o = camera.origin();
    let mut img = vec![[0.0f32; 3]; width * height];
    const SEG_SAMPLES: u32 = 4;

    for i in 0..n_photons {
        let mut rng = PathRng::new(i as u64, seed);
        let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
        let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32();
        let (xb, yb, zb) = sensor.cmf(lambda);
        let mut ray = Ray { origin, dir: beam.dir };
        let mut power = 1.0f32;

        for _ in 0..8 {
            let hit = scene.intersect(&ray);
            let seg_len = hit.as_ref().map(|(h, _)| h.t).unwrap_or(max_dist);

            for _ in 0..SEG_SAMPLES {
                let dist = rng.next_f32() * seg_len;
                let p = ray.origin + ray.dir * dist;
                if let Some((s, t, _)) = camera.project(p) {
                    let px = ((s * width as f32) as usize).min(width - 1);
                    let py = (((1.0 - t) * height as f32) as usize).min(height - 1);
                    let to = cam_o - p;
                    let d = to.length().max(1e-3);
                    if d < zbuffer[py * width + px] {
                        let cos_theta = ray.dir.dot(to / d);
                        let phase = phase_hg(g, cos_theta);
                        let trans = (-sigma_t * d).exp();
                        let contrib = power * sigma_s * phase * trans / (d * d)
                            * seg_len / SEG_SAMPLES as f32;
                        let idx = py * width + px;
                        img[idx][0] += xb * contrib;
                        img[idx][1] += yb * contrib;
                        img[idx][2] += zb * contrib;
                    }
                }
            }

            match hit {
                Some((h, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(lambda),
                        _ => 1.0,
                    };
                    match mat.scatter(ray.dir, &h, lambda, n_hero, &mut rng) {
                        Some(sc) => { power *= sc.weight; ray = Ray { origin: h.point, dir: sc.dir }; }
                        None => break,
                    }
                }
                None => break,
            }
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hg_forward_biased() {
        // g>0 scatters more forward (cos=1) than backward (cos=-1).
        assert!(phase_hg(0.6, 1.0) > phase_hg(0.6, -1.0));
        // g=0 is isotropic.
        assert!((phase_hg(0.0, 0.5) - phase_hg(0.0, -0.5)).abs() < 1e-6);
    }
}

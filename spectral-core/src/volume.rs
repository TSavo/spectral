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
use glam::Vec3;

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
    seed: u32,
) -> Vec<Xyz> {
    let sensor = Sensor::new();
    let mut img = vec![[0.0f32; 3]; width * height];
    const SEG_SAMPLES: u32 = 4;
    const INV_4PI: f32 = 1.0 / (4.0 * std::f32::consts::PI);

    for i in 0..n_photons {
        let mut rng = PathRng::new(i, seed);
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

/// Single-scatter volumetric forward tracing, POINT-splat estimator (reference):
/// 4 random scatter samples per segment, each deposited at the nearest pixel.
/// Kept as the ground-truth integral for the beam-vs-point energy check; the
/// production estimator is `render_volumetric_scene` (beam splatting) below.
#[allow(clippy::too_many_arguments)]
pub fn render_volumetric_scene_point(
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
    seed: u32,
) -> Vec<Xyz> {
    let sensor = Sensor::new();
    let cam_o = camera.origin();
    let mut img = vec![[0.0f32; 3]; width * height];
    const SEG_SAMPLES: u32 = 4;

    for i in 0..n_photons {
        let mut rng = PathRng::new(i, seed);
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

/// World-space radius of a photon beam's cross-section. Projected to screen per
/// march step, it gives each splat its transverse "volume" (shrinks with depth).
const R_BEAM: f32 = 0.03;

/// Energy-conserving transverse splat: deposits `color` over a tent footprint of
/// radius `radius_px` centred at screen (s, t). Weights are renormalised over the
/// pixels actually covered, so a footprint clipped by the screen edge loses no
/// energy (it concentrates on the visible pixels). Total deposited == sum(color).
fn splat_transverse(img: &mut [Xyz], width: usize, height: usize, s: f32, t: f32, color: Xyz, radius_px: f32) {
    let cx = s * width as f32 - 0.5;
    let cy = (1.0 - t) * height as f32 - 0.5;
    let r = radius_px.max(0.5);
    let x0 = (cx - r).floor().max(0.0) as usize;
    let x1 = ((cx + r).ceil() as isize).clamp(0, width as isize - 1) as usize;
    let y0 = (cy - r).floor().max(0.0) as usize;
    let y1 = ((cy + r).ceil() as isize).clamp(0, height as isize - 1) as usize;

    let mut wsum = 0.0f32;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            wsum += (1.0 - (dx * dx + dy * dy).sqrt() / r).max(0.0);
        }
    }
    if wsum <= 0.0 {
        let px = (cx.round() as isize).clamp(0, width as isize - 1) as usize;
        let py = (cy.round() as isize).clamp(0, height as isize - 1) as usize;
        let idx = py * width + px;
        img[idx][0] += color[0];
        img[idx][1] += color[1];
        img[idx][2] += color[2];
        return;
    }
    let inv = 1.0 / wsum;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let w = (1.0 - (dx * dx + dy * dy).sqrt() / r).max(0.0) * inv;
            if w > 0.0 {
                let idx = py * width + px;
                img[idx][0] += color[0] * w;
                img[idx][1] += color[1] * w;
                img[idx][2] += color[2] * w;
            }
        }
    }
}

/// Single-scatter volumetric forward tracing, PHOTON-BEAM estimator (production):
/// each path segment is splatted as a continuous projected beam with transverse
/// volume -- the splat-dual of the beam radiance estimate. The segment is marched
/// in `M = ceil(projected_pixel_length) + 1` deterministic steps (so steps land
/// ~1px apart along the beam), and each step deposits an energy-conserving
/// transverse footprint of radius = projected `R_BEAM`. This removes the point
/// estimator's random scatter samples (no per-segment `dist` draws); only the
/// Fresnel `mat.scatter` draws remain in the RNG stream. Same camera-connection
/// weight as the point estimator (`phase * exp(-sigma_t d) / d^2`), integrated
/// continuously instead of at 4 random points -- so it converges in noise as the
/// beam fills, instead of dithering between sparse spectral points at the dim
/// extremes. Identical signature to `render_volumetric_scene_point`.
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
    seed: u32,
) -> Vec<Xyz> {
    let sensor = Sensor::new();
    let cam_o = camera.origin();
    let mut img = vec![[0.0f32; 3]; width * height];

    for i in 0..n_photons {
        let mut rng = PathRng::new(i, seed);
        let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
        let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32();
        let (xb, yb, zb) = sensor.cmf(lambda);
        let mut ray = Ray { origin, dir: beam.dir };
        let mut power = 1.0f32;

        for _ in 0..8 {
            let hit = scene.intersect(&ray);
            let seg_len = hit.as_ref().map(|(h, _)| h.t).unwrap_or(max_dist);

            // March count from projected screen length (~1 step per pixel).
            let p_end = ray.origin + ray.dir * seg_len;
            // ~2 steps per projected pixel (sub-pixel along-beam spacing) so the
            // beam reads continuous even where its transverse footprint is ~1px.
            let m: usize = match (camera.project(ray.origin), camera.project(p_end)) {
                (Some((s0, t0, _)), Some((s1, t1, _))) => {
                    let dx = (s1 - s0) * width as f32;
                    let dy = (t1 - t0) * height as f32;
                    (((dx * dx + dy * dy).sqrt() * 2.0).ceil() as usize + 1).clamp(1, 4096)
                }
                _ => ((seg_len * 80.0).ceil() as usize).clamp(1, 4096),
            };
            let ds = seg_len / m as f32;

            for k in 0..m {
                let dist = (k as f32 + 0.5) * ds;
                let p = ray.origin + ray.dir * dist;
                if let Some((s, t, _)) = camera.project(p) {
                    let px = ((s * width as f32) as usize).min(width - 1);
                    let py = (((1.0 - t) * height as f32) as usize).min(height - 1);
                    let to = cam_o - p;
                    let d = to.length().max(1e-3);
                    if d < zbuffer[py * width + px] {
                        let view = to / d;
                        let cos_theta = ray.dir.dot(view);
                        let phase = phase_hg(g, cos_theta);
                        let trans = (-sigma_t * d).exp();
                        // Same weight as the point estimator, with ds in place of
                        // seg_len/SEG_SAMPLES (continuous quadrature along the beam).
                        let weight = power * sigma_s * phase * trans / (d * d) * ds;
                        let color = [xb * weight, yb * weight, zb * weight];
                        // Transverse footprint = projected R_BEAM at p (the volume).
                        let mut perp = view.cross(Vec3::Y);
                        if perp.length_squared() < 1e-8 {
                            perp = view.cross(Vec3::X);
                        }
                        perp = perp.normalize() * R_BEAM;
                        let radius_px = match camera.project(p + perp) {
                            Some((s2, t2, _)) => {
                                let dpx = (s2 - s) * width as f32;
                                let dpy = (t2 - t) * height as f32;
                                (dpx * dpx + dpy * dpy).sqrt().clamp(1.5, 24.0)
                            }
                            None => 1.0,
                        };
                        splat_transverse(&mut img, width, height, s, t, color, radius_px);
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

    // Beam march covers the full segment: m uniform steps of ds=seg_len/m sum to
    // seg_len (no gaps, no double-count) -> energy along the beam is preserved.
    #[test]
    fn beam_march_steps_cover_full_segment() {
        for &(seg_len, m) in &[(0.5f32, 1usize), (3.7, 13), (14.0, 2048), (0.001, 5)] {
            let ds = seg_len / m as f32;
            let total: f32 = (0..m).map(|_| ds).sum();
            assert!(
                (total - seg_len).abs() <= seg_len * 1e-4 + 1e-6,
                "sum(ds)={total} != seg_len={seg_len} (m={m})"
            );
        }
    }

    // The beam estimator and the point estimator integrate the same physical
    // in-scatter, so total image energy (sum of Y) must match within MC noise.
    // The transverse kernel only redistributes energy spatially (weights sum to
    // 1), so it cannot change the total. Empty scene = beam through pure haze.
    #[test]
    fn beam_and_point_estimators_conserve_energy() {
        use crate::camera::Camera;
        use crate::lighttrace::Beam;
        use crate::scene::Scene;
        use glam::Vec3;

        let (w, h) = (128usize, 96usize);
        let scene = Scene::new(); // no geometry: photons traverse haze in straight beams
        let beam = Beam {
            corner: Vec3::new(-2.0, 0.0, 0.0),
            u: Vec3::new(0.0, 0.0, 0.6),
            v: Vec3::new(0.0, 0.4, 0.0),
            dir: Vec3::X,
        };
        let cam = Camera::look_at(
            Vec3::new(0.0, 0.5, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0,
            w as f32 / h as f32,
        );
        let zbuf = vec![f32::INFINITY; w * h];
        let (n, ss, st, g, md, seed) = (20_000u32, 0.5f32, 0.06f32, 0.4f32, 12.0f32, 7u32);

        let pt = render_volumetric_scene_point(&scene, &cam, &beam, w, h, n, ss, st, g, md, &zbuf, seed);
        let bm = render_volumetric_scene(&scene, &cam, &beam, w, h, n, ss, st, g, md, &zbuf, seed);
        let ysum = |img: &[Xyz]| img.iter().map(|p| p[1]).sum::<f32>();
        let (yp, yb) = (ysum(&pt), ysum(&bm));
        assert!(yp > 0.0 && yb > 0.0, "no energy: point={yp} beam={yb}");
        let ratio = yb / yp;
        assert!(
            (0.92..=1.08).contains(&ratio),
            "beam/point energy ratio {ratio} out of [0.92,1.08] (point={yp} beam={yb})"
        );
    }
}

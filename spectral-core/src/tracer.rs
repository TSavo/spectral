//! CPU spectral path tracer implementing the Renderer trait.

use crate::camera::Camera;
use crate::cie::{Illuminant, Sensor};
use crate::geom::Ray;
use crate::material::Material;
use crate::rng::PathRng;
use crate::scene::Scene;
use crate::spectrum::{SpectralSample, HERO_N};
use crate::{AccumBuffer, Renderer};

pub struct CpuTracer {
    scene: Scene,
    camera: Camera,
    sensor: Sensor,
    illuminant: Illuminant,
    seed: u32,
    accum: AccumBuffer,
    max_bounces: u32,
}

impl CpuTracer {
    /// The path depth is fixed at 8 bounces.
    pub fn new(
        scene: Scene,
        camera: Camera,
        width: usize,
        height: usize,
        illuminant: Illuminant,
        seed: u32,
    ) -> Self {
        CpuTracer {
            scene,
            camera,
            sensor: Sensor::new(),
            illuminant,
            seed,
            accum: AccumBuffer::new(width, height),
            max_bounces: 8,
        }
    }

    /// Trace one hero-comb path for a pixel and return its XYZ contribution.
    fn sample_pixel(&self, px: usize, py: usize, sample_idx: u32) -> [f32; 3] {
        let w = self.accum.width;
        let h = self.accum.height;
        // Stream key: mix pixel index and sample index via mix2, which uses two
        // rounds of the PCG hash for collision resistance. Pure u32; mirrors WGSL.
        let pixel = (py * w + px) as u32;
        let key = crate::rng::mix2(pixel, sample_idx);
        let mut rng = PathRng::new(key, self.seed);

        let s = (px as f32 + rng.next_f32()) / w as f32;
        let t = 1.0 - (py as f32 + rng.next_f32()) / h as f32;
        let mut spec = SpectralSample::from_hero_u(rng.next_f32());

        let mut ray = self.camera.primary_ray(s, t);
        let mut throughput = [1.0f32; HERO_N];
        let mut valid_lanes = HERO_N;

        for _ in 0..self.max_bounces {
            match self.scene.intersect(&ray) {
                None => {
                    let bg = self.scene.background_radiance(ray.dir);
                    for (rad, (&lam, &tk)) in spec
                        .radiance
                        .iter_mut()
                        .zip(spec.lambda.iter().zip(throughput.iter()))
                    {
                        let sp = self.sensor.illuminant(self.illuminant, lam);
                        *rad += tk * bg * sp;
                    }
                    break;
                }
                Some((hit, mat)) => {
                    let n_hero = match mat {
                        Material::Dielectric { glass } => glass.n(spec.lambda[0]),
                        _ => 1.0,
                    };
                    match mat.scatter(ray.dir, &hit, spec.lambda[0], n_hero, &mut rng) {
                        None => break,
                        Some(sc) => {
                            for tk in throughput.iter_mut() {
                                *tk *= sc.weight;
                            }
                            // Dispersion collapse: companion wavelengths cannot follow
                            // the hero's refracted path, so keep only the hero lane.
                            if matches!(mat, Material::Dielectric { .. }) && valid_lanes > 1 {
                                throughput[1..].fill(0.0);
                                valid_lanes = 1;
                            }
                            ray = Ray { origin: hit.point, dir: sc.dir };
                        }
                    }
                }
            }
        }

        // Convert the comb's radiance to XYZ via the sensor. Divide by the number
        // of VALID lanes (HERO_N normally, 1 after a dispersion collapse) so the
        // Monte Carlo estimate stays unbiased in both cases.
        let mut xyz = [0.0f32; 3];
        for k in 0..HERO_N {
            let (xb, yb, zb) = self.sensor.cmf(spec.lambda[k]);
            let wk = spec.radiance[k] / spec.pdf[k] / valid_lanes as f32;
            xyz[0] += xb * wk;
            xyz[1] += yb * wk;
            xyz[2] += zb * wk;
        }
        xyz
    }
}

impl Renderer for CpuTracer {
    fn accumulate(&mut self, samples_per_pixel: u32) {
        use rayon::prelude::*;
        let w = self.accum.width;
        let h = self.accum.height;
        let base = self.accum.samples;
        let batch: Vec<[f32; 3]> = (0..w * h)
            .into_par_iter()
            .map(|i| {
                let px = i % w;
                let py = i / w;
                let mut acc = [0.0f32; 3];
                for sidx in 0..samples_per_pixel {
                    let c = self.sample_pixel(px, py, base + sidx);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
                acc
            })
            .collect();
        for (dst, src) in self.accum.sum.iter_mut().zip(batch) {
            dst[0] += src[0];
            dst[1] += src[1];
            dst[2] += src[2];
        }
        self.accum.samples += samples_per_pixel;
    }

    fn buffer(&self) -> &AccumBuffer {
        &self.accum
    }

    fn reset(&mut self) {
        self.accum.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cie::chromaticity;
    use crate::geom::ConvexSolid;
    use crate::sellmeier::Glass;
    use glam::Vec3;

    fn mean_luminance(t: &CpuTracer) -> f32 {
        let m = t.buffer().mean();
        m.iter().map(|p| p[1]).sum::<f32>() / m.len() as f32
    }

    #[test]
    fn accumulating_increases_samples_and_energy() {
        let mut scene = Scene::new();
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Lambertian { reflectance: 0.8 });
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let mut t = CpuTracer::new(scene, cam, 16, 16, Illuminant::D65, 42);
        t.accumulate(4);
        assert_eq!(t.buffer().samples, 4);
        let total: f32 = t.buffer().mean().iter().map(|p| p[1]).sum();
        assert!(total > 0.0, "expected positive luminance");
    }

    #[test]
    fn reset_clears() {
        let scene = Scene::new();
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let mut t = CpuTracer::new(scene, cam, 8, 8, Illuminant::D65, 1);
        t.accumulate(2);
        t.reset();
        assert_eq!(t.buffer().samples, 0);
        assert!(t.buffer().sum.iter().all(|p| *p == [0.0, 0.0, 0.0]), "reset must zero the sum buffer");
    }

    // END-TO-END: an empty scene shows only the D65 background. The Monte Carlo
    // wavelength estimator must integrate to the D65 white point chromaticity,
    // tying the tracer's accumulation to the verified sensor.
    #[test]
    fn background_only_integrates_to_d65_white_point() {
        let mut scene = Scene::new();
        scene.background = 1.0;
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let mut t = CpuTracer::new(scene, cam, 32, 32, Illuminant::D65, 7);
        t.accumulate(256);
        let m = t.buffer().mean();
        let mut sum = [0.0f32; 3];
        for px in &m {
            sum[0] += px[0]; sum[1] += px[1]; sum[2] += px[2];
        }
        let (x, y) = chromaticity(sum);
        assert!((x - 0.3127).abs() < 6e-3, "x={x} not near D65 white point");
        assert!((y - 0.3290).abs() < 6e-3, "y={y} not near D65 white point");
    }

    // DISCRIMINATION: a glass slab hit near-normal collapses each path to the hero
    // lane (valid_lanes=1). Its luminance must stay close to the no-slab background
    // (minus a few % Fresnel loss), NOT 1/HERO_N of it. A wrong "always divide by
    // HERO_N" normalization would make this ~4x too dark and fail.
    #[test]
    fn dispersion_collapse_preserves_brightness() {
        let look = || Camera::look_at(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y, 18.0, 1.0);
        let mut empty = Scene::new();
        empty.background = 1.0;
        let mut t0 = CpuTracer::new(empty, look(), 8, 8, Illuminant::D65, 3);
        t0.accumulate(256);
        let y_empty = mean_luminance(&t0);

        let mut slab = Scene::new();
        slab.background = 1.0;
        slab.add_solid(
            ConvexSolid::axis_box(Vec3::new(-5.0, -5.0, -0.1), Vec3::new(5.0, 5.0, 0.1)),
            Material::Dielectric { glass: Glass::Bk7 },
        );
        let mut t1 = CpuTracer::new(slab, look(), 8, 8, Illuminant::D65, 3);
        t1.accumulate(256);
        let y_slab = mean_luminance(&t1);

        assert!(y_empty > 0.0);
        assert!(y_slab > 0.7 * y_empty, "collapse too dark: slab={y_slab} empty={y_empty} (ratio {})", y_slab / y_empty);
        assert!(y_slab < 1.05 * y_empty, "slab brighter than background: {y_slab} vs {y_empty}");
    }
}

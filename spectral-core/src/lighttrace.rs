//! Forward light tracing: emit photons from a collimated source, refract them
//! through the scene, and splat survivors onto a screen. This produces caustics
//! (e.g. a prism's dispersed spectrum) that backward path tracing cannot.

use crate::cie::Sensor;
use crate::geom::Ray;
use crate::material::Material;
use crate::rng::PathRng;
use crate::scene::Scene;
use crate::spectrum::{LAMBDA_MIN, LAMBDA_RANGE};
use crate::Xyz;
use glam::Vec3;

/// A rectangular screen that catches photons and accumulates their color.
pub struct Screen {
    corner: Vec3,
    u: Vec3, // full-width edge vector
    v: Vec3, // full-height edge vector
    normal: Vec3,
    width: usize,
    height: usize,
    pub sum: Vec<Xyz>,
    sensor: Sensor,
}

impl Screen {
    pub fn new(corner: Vec3, u: Vec3, v: Vec3, width: usize, height: usize) -> Self {
        let normal = u.cross(v).normalize();
        Screen {
            corner, u, v, normal, width, height,
            sum: vec![[0.0; 3]; width * height],
            sensor: Sensor::new(),
        }
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }

    /// Ray-plane intersection. Returns (s, r, t) where s,r are the in-rectangle
    /// coordinates in [0,1) (s along u, r along v) and t is the ray parameter,
    /// or None if the ray misses the rectangle (or hits behind the origin).
    pub fn screen_coords(&self, ray: &Ray) -> Option<(f32, f32, f32)> {
        let denom = ray.dir.dot(self.normal);
        if denom.abs() < 1e-8 {
            return None;
        }
        let t = (self.corner - ray.origin).dot(self.normal) / denom;
        if t <= 1e-4 {
            return None;
        }
        let p = ray.origin + ray.dir * t;
        let rel = p - self.corner;
        let s = rel.dot(self.u) / self.u.dot(self.u);
        let r = rel.dot(self.v) / self.v.dot(self.v);
        if (0.0..1.0).contains(&s) && (0.0..1.0).contains(&r) {
            Some((s, r, t))
        } else {
            None
        }
    }

    /// Texel coordinates for fractional screen coords: returns the lower-left
    /// texel (x0,y0), its upper-right neighbor (x1,y1), and the bilinear weights
    /// (dx,dy). r runs along v (upward); row 0 is the top.
    fn texel_bilinear(&self, s: f32, r: f32) -> (usize, usize, usize, usize, f32, f32) {
        let fx = (s * self.width as f32 - 0.5).clamp(0.0, self.width as f32 - 1.0);
        let fy = ((1.0 - r) * self.height as f32 - 0.5).clamp(0.0, self.height as f32 - 1.0);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        (x0, y0, x1, y1, fx - x0 as f32, fy - y0 as f32)
    }

    fn splat(&mut self, s: f32, r: f32, lambda: f32, power: f32) {
        let (x, y, z) = self.sensor.cmf(lambda);
        let (x0, y0, x1, y1, dx, dy) = self.texel_bilinear(s, r);
        let w = self.width;
        // Bilinear deposit into the four surrounding texels for a smooth caustic.
        for (xi, yi, wgt) in [
            (x0, y0, (1.0 - dx) * (1.0 - dy)),
            (x1, y0, dx * (1.0 - dy)),
            (x0, y1, (1.0 - dx) * dy),
            (x1, y1, dx * dy),
        ] {
            let i = yi * w + xi;
            self.sum[i][0] += x * power * wgt;
            self.sum[i][1] += y * power * wgt;
            self.sum[i][2] += z * power * wgt;
        }
    }

    /// The accumulated XYZ buffer scaled by `scale` (e.g. 1/num_photons * gain).
    pub fn scaled(&self, scale: f32) -> Vec<Xyz> {
        self.sum.iter().map(|p| [p[0] * scale, p[1] * scale, p[2] * scale]).collect()
    }

    /// Deposited XYZ at fractional screen coordinates s,r in [0,1), bilinearly
    /// interpolated from the four surrounding texels.
    pub fn sample(&self, s: f32, r: f32) -> Xyz {
        let (x0, y0, x1, y1, dx, dy) = self.texel_bilinear(s, r);
        let w = self.width;
        let mut out = [0.0f32; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = self.sum[y0 * w + x0][c] * (1.0 - dx) * (1.0 - dy)
                + self.sum[y0 * w + x1][c] * dx * (1.0 - dy)
                + self.sum[y1 * w + x0][c] * (1.0 - dx) * dy
                + self.sum[y1 * w + x1][c] * dx * dy;
        }
        out
    }
}

/// A collimated monochromatic-per-photon source. Photon origins are sampled
/// uniformly over the aperture rectangle (corner + s*u + t*v); all travel `dir`.
pub struct Beam {
    pub corner: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub dir: Vec3, // unit
}

/// Trace one photon of wavelength `lambda` forward through the scene. Returns the
/// (s, r, power) where it lands on the screen, or None if it escapes/absorbs.
pub fn trace_photon(
    scene: &Scene,
    screen: &Screen,
    mut ray: Ray,
    lambda: f32,
    rng: &mut PathRng,
) -> Option<(f32, f32, f32)> {
    let mut power = 1.0f32;
    for _ in 0..8 {
        let scene_hit = scene.intersect(&ray);
        let screen_hit = screen.screen_coords(&ray);
        let scene_t = scene_hit.as_ref().map(|(h, _)| h.t).unwrap_or(f32::INFINITY);
        let screen_t = screen_hit.map(|(_, _, t)| t).unwrap_or(f32::INFINITY);

        if screen_t < scene_t {
            let (s, r, _) = screen_hit.unwrap();
            return Some((s, r, power));
        }
        let (hit, mat) = scene_hit?;
        let n_hero = match mat {
            Material::Dielectric { glass } => glass.n(lambda),
            _ => 1.0,
        };
        let sc = mat.scatter(ray.dir, &hit, lambda, n_hero, rng)?;
        power *= sc.weight;
        ray = Ray { origin: hit.point, dir: sc.dir };
    }
    None
}

/// Emit `n` photons from the beam (wavelengths uniform over the visible band) and
/// splat survivors onto the screen.
pub fn trace(scene: &Scene, screen: &mut Screen, beam: &Beam, n: u32, seed: u32) {
    for i in 0..n {
        let mut rng = PathRng::new(i, seed);
        let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
        let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32();
        if let Some((s, r, power)) = trace_photon(scene, screen, Ray { origin, dir: beam.dir }, lambda, &mut rng) {
            screen.splat(s, r, lambda, power);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::ConvexSolid;
    use crate::material::Material;
    use crate::sellmeier::Glass;

    // Mean landing (s,r) on the screen for a single wavelength, over many photons.
    fn mean_landing(scene: &Scene, screen: &Screen, beam: &Beam, lambda: f32, n: u32) -> Option<(f32, f32)> {
        let (mut sx, mut sr, mut cnt) = (0.0f32, 0.0f32, 0u32);
        for i in 0..n {
            let mut rng = PathRng::new(i, 1);
            let origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32();
            if let Some((s, r, _)) = trace_photon(scene, screen, Ray { origin, dir: beam.dir }, lambda, &mut rng) {
                sx += s; sr += r; cnt += 1;
            }
        }
        if cnt == 0 { None } else { Some((sx / cnt as f32, sr / cnt as f32)) }
    }

    fn make_screen() -> Screen {
        // Large screen at x=4 facing -X, spanning Z (u) and Y (v, from -3 to 5),
        // sized to catch the upward-deviated dispersed beam.
        Screen::new(
            Vec3::new(4.0, -3.0, -2.0),
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 8.0, 0.0),
            64,
            64,
        )
    }
    fn make_beam() -> Beam {
        // Thin +X beam at y~0, entering the wedge's vertical face at normal incidence.
        Beam {
            corner: Vec3::new(-3.0, -0.05, -0.05),
            u: Vec3::new(0.0, 0.0, 0.1),
            v: Vec3::new(0.0, 0.1, 0.0),
            dir: Vec3::X,
        }
    }

    // PHYSICS GATE + DISCRIMINATION: a prism separates blue from red spatially on
    // the screen; with no prism, all wavelengths land at the same place.
    #[test]
    fn prism_disperses_beam_on_screen() {
        let screen = make_screen();
        let beam = make_beam();

        let mut prism = Scene::new();
        prism.add_solid(ConvexSolid::wedge(30.0, 1.0, 2.0), Material::Dielectric { glass: Glass::Sf11 });
        let blue = mean_landing(&prism, &screen, &beam, 450.0, 256).expect("blue must reach the screen");
        let red = mean_landing(&prism, &screen, &beam, 650.0, 256).expect("red must reach the screen");
        let sep = ((blue.0 - red.0).powi(2) + (blue.1 - red.1).powi(2)).sqrt();

        let empty = Scene::new();
        let b0 = mean_landing(&empty, &screen, &beam, 450.0, 256).expect("blue control lands");
        let r0 = mean_landing(&empty, &screen, &beam, 650.0, 256).expect("red control lands");
        let sep0 = ((b0.0 - r0.0).powi(2) + (b0.1 - r0.1).powi(2)).sqrt();

        assert!(sep0 < 1e-4, "without a prism there is no dispersion, got sep0={sep0}");
        assert!(sep > 0.02, "prism must separate blue and red on the screen, got sep={sep}");
    }

    #[test]
    fn screen_sample_reads_deposited_energy() {
        // After tracing the dispersing beam, some texel on the screen is lit.
        let mut screen = make_screen();
        let beam = make_beam();
        let mut prism = Scene::new();
        prism.add_solid(crate::geom::ConvexSolid::wedge(30.0, 1.0, 2.0),
            crate::material::Material::Dielectric { glass: crate::sellmeier::Glass::Sf11 });
        trace(&prism, &mut screen, &beam, 100_000, 1);
        let mut max = 0.0f32;
        for r_i in 0..64 { for s_i in 0..64 {
            let c = screen.sample((s_i as f32 + 0.5)/64.0, (r_i as f32 + 0.5)/64.0);
            max = max.max(c[1]);
        }}
        assert!(max > 0.0, "the wall must have received some photon energy");
    }
}

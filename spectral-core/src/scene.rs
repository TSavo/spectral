//! Scene: a list of primitives with materials, plus a background.

use crate::geom::{ConvexSolid, Hit, Ray, Sphere};
use crate::material::Material;
use glam::Vec3;

pub enum Shape {
    Sphere(Sphere),
    Solid(ConvexSolid),
}

pub struct Primitive {
    pub shape: Shape,
    pub material: Material,
}

/// A scene: primitives plus a uniform background radiance scale. The background
/// SPD itself comes from the sensor's chosen illuminant in the tracer.
pub struct Scene {
    pub primitives: Vec<Primitive>,
    pub background: f32,
    /// Optional extra radiance for rays pointing above the horizon (y > 0).
    /// When `Some(above)`, rays with `dir.y > 0` receive `background + above`.
    /// `None` gives a fully uniform background.
    pub horizon: Option<f32>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Scene { primitives: Vec::new(), background: 1.0, horizon: None }
    }

    /// Background radiance for a ray that escaped the scene, as a function of
    /// its direction. Uniform `background` everywhere, plus, if `horizon` is set,
    /// an extra bright contribution for rays pointing above the horizon (y > 0).
    /// The spectral shape comes from the illuminant in the tracer; this is a
    /// scalar radiance scale only.
    pub fn background_radiance(&self, dir: Vec3) -> f32 {
        let sky = match self.horizon {
            Some(above) if dir.y > 0.0 => above,
            _ => 0.0,
        };
        self.background + sky
    }

    pub fn add_sphere(&mut self, center: Vec3, radius: f32, material: Material) {
        self.primitives.push(Primitive {
            shape: Shape::Sphere(Sphere { center, radius }),
            material,
        });
    }

    pub fn add_solid(&mut self, solid: ConvexSolid, material: Material) {
        self.primitives.push(Primitive { shape: Shape::Solid(solid), material });
    }

    /// Nearest intersection with any primitive, with its material.
    pub fn intersect(&self, r: &Ray) -> Option<(Hit, Material)> {
        let mut best: Option<(Hit, Material)> = None;
        let mut closest = f32::INFINITY;
        for p in &self.primitives {
            let hit = match &p.shape {
                Shape::Sphere(s) => s.intersect(r, 1e-3, closest),
                Shape::Solid(s) => s.intersect(r, 1e-3, closest),
            };
            if let Some(h) = hit {
                closest = h.t;
                best = Some((h, p.material));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_picks_nearest() {
        let mut scene = Scene::new();
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Lambertian { reflectance: 0.5 });
        scene.add_sphere(Vec3::new(0.0, 0.0, -6.0), 1.0, Material::Lambertian { reflectance: 0.5 });
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        let (hit, _mat) = scene.intersect(&r).expect("hit");
        assert!((hit.t - 2.0).abs() < 1e-3, "nearest sphere at t=2, got {}", hit.t);
    }

    #[test]
    fn intersect_returns_none_on_miss() {
        let mut scene = Scene::new();
        scene.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Lambertian { reflectance: 0.5 });
        // Ray pointing away from the only primitive.
        let r = Ray { origin: Vec3::ZERO, dir: Vec3::Z };
        assert!(scene.intersect(&r).is_none());
    }

    #[test]
    fn empty_scene_intersects_nothing() {
        let scene = Scene::new();
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        assert!(scene.intersect(&r).is_none());
    }

    #[test]
    fn background_radiance_uniform_then_horizon() {
        let mut scene = Scene::new();
        scene.background = 0.5;
        assert_eq!(scene.background_radiance(Vec3::Y), 0.5); // no horizon -> uniform
        assert_eq!(scene.background_radiance(-Vec3::Y), 0.5);
        scene.horizon = Some(2.0);
        assert_eq!(scene.background_radiance(Vec3::Y), 2.5);  // above horizon
        assert_eq!(scene.background_radiance(-Vec3::Y), 0.5); // below unchanged
    }
}

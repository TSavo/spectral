//! Ray, intersection record, sphere, half-space, and convex CSG (slab method).

use glam::Vec3;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3, // unit
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3, // unit, oriented to face the incoming ray
}

#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    #[must_use]
    pub fn intersect(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.dir.dot(r.dir);
        let half_b = oc.dot(r.dir);
        let c = oc.dot(oc) - self.radius * self.radius;
        let disc = half_b * half_b - a * c;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        let mut t = (-half_b - sq) / a;
        if t < t_min || t > t_max {
            t = (-half_b + sq) / a;
            if t < t_min || t > t_max {
                return None;
            }
        }
        let point = r.at(t);
        let mut normal = (point - self.center) / self.radius;
        if normal.dot(r.dir) > 0.0 {
            normal = -normal;
        }
        Some(Hit { t, point, normal })
    }
}

/// Half-space {x : plane.normal·x + plane.d <= 0}. Normal points OUT of the solid.
#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

/// Convex solid = intersection of half-spaces. Solved by the slab method.
pub struct ConvexSolid {
    pub planes: Vec<Plane>,
}

impl ConvexSolid {
    pub fn axis_box(min: Vec3, max: Vec3) -> Self {
        ConvexSolid {
            planes: vec![
                Plane { normal: Vec3::X, d: -max.x },
                Plane { normal: -Vec3::X, d: min.x },
                Plane { normal: Vec3::Y, d: -max.y },
                Plane { normal: -Vec3::Y, d: min.y },
                Plane { normal: Vec3::Z, d: -max.z },
                Plane { normal: -Vec3::Z, d: min.z },
            ],
        }
    }

    /// Triangular prism: triangular cross-section in XY (3 side planes), extruded
    /// along Z to [-depth/2, depth/2] (2 cap planes). `size` is the inradius of
    /// the cross-section.
    pub fn triangular_prism(size: f32, depth: f32) -> Self {
        let planes: Vec<Plane> = (0..3)
            .map(|k| {
                let ang =
                    std::f32::consts::FRAC_PI_2 + k as f32 * 2.0 * std::f32::consts::PI / 3.0;
                let n = Vec3::new(ang.cos(), ang.sin(), 0.0);
                Plane { normal: n, d: -size }
            })
            .chain([
                Plane { normal: Vec3::Z, d: -depth / 2.0 },
                Plane { normal: -Vec3::Z, d: -depth / 2.0 },
            ])
            .collect();
        ConvexSolid { planes }
    }

    /// Intersect the ray with the convex solid (slab method). Returns the first
    /// surface crossing strictly ahead of `t_min`. If the ray origin is INSIDE
    /// the solid, this is the exit face (essential for rays refracting through
    /// glass). The normal is flipped to face the incoming ray.
    #[must_use]
    pub fn intersect(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let mut t_enter = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;
        let mut enter_normal = Vec3::ZERO;
        let mut exit_normal = Vec3::ZERO;
        for p in &self.planes {
            let denom = p.normal.dot(r.dir);
            let dist = p.normal.dot(r.origin) + p.d; // signed; > 0 means outside
            if denom.abs() < 1e-8 {
                if dist > 0.0 {
                    return None; // parallel to this slab and outside it
                }
                continue;
            }
            let t = -dist / denom;
            if denom < 0.0 {
                // crossing into this half-space
                if t > t_enter {
                    t_enter = t;
                    enter_normal = p.normal;
                }
            } else {
                // crossing out of this half-space
                if t < t_exit {
                    t_exit = t;
                    exit_normal = p.normal;
                }
            }
        }
        if t_enter > t_exit {
            return None; // the ray misses the solid
        }
        // Pick the first crossing strictly ahead of t_min. If t_enter is behind
        // t_min, the origin is inside (or past the entry): use the exit face.
        let (t_hit, hit_normal) = if t_enter > t_min {
            (t_enter, enter_normal)
        } else if t_exit > t_min {
            (t_exit, exit_normal)
        } else {
            return None; // the whole solid is behind the ray
        };
        if t_hit > t_max {
            return None;
        }
        let mut normal = hit_normal;
        if normal.dot(r.dir) > 0.0 {
            normal = -normal;
        }
        Some(Hit { t: t_hit, point: r.at(t_hit), normal })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn ray_hits_sphere_front() {
        let s = Sphere { center: Vec3::new(0.0, 0.0, -5.0), radius: 1.0 };
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        let h = s.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal, Vec3::Z, epsilon = 1e-4); // faces the ray
    }

    #[test]
    fn ray_misses_sphere() {
        let s = Sphere { center: Vec3::new(5.0, 0.0, -5.0), radius: 1.0 };
        let r = Ray { origin: Vec3::ZERO, dir: -Vec3::Z };
        assert!(s.intersect(&r, 1e-4, f32::INFINITY).is_none());
    }

    #[test]
    fn convex_box_intersection_interval() {
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::new(0.0, 0.0, 5.0), dir: -Vec3::Z };
        let h = cube.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-4); // enters at z=1
        assert_relative_eq!(h.normal, Vec3::Z, epsilon = 1e-4);
    }

    #[test]
    fn prism_is_five_halfspaces() {
        let p = ConvexSolid::triangular_prism(2.0, 2.0);
        assert_eq!(p.planes.len(), 5);
    }

    #[test]
    fn convex_box_miss_lateral() {
        // Ray along +Z but offset to x=2, outside the [-1,1] x-extent: misses.
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::new(2.0, 0.0, -5.0), dir: Vec3::Z };
        assert!(cube.intersect(&r, 1e-4, f32::INFINITY).is_none());
    }

    #[test]
    fn convex_box_miss_parallel_outside() {
        // Ray along +X at y=2: parallel to the Y slab and outside it.
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::new(-5.0, 2.0, 0.0), dir: Vec3::X };
        assert!(cube.intersect(&r, 1e-4, f32::INFINITY).is_none());
    }

    #[test]
    fn convex_box_side_entry_normal() {
        // Enter from the +X side; entry normal is +X (faces the incoming -X ray).
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::new(5.0, 0.0, 0.0), dir: -Vec3::X };
        let h = cube.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal, Vec3::X, epsilon = 1e-4);
    }

    #[test]
    fn convex_box_origin_inside_returns_exit_face() {
        // The refraction case: a ray traveling INSIDE the solid must find the
        // exit face, with a valid non-zero normal. Origin at center going +Z
        // exits the +Z face at z=1 (t=1); normal flipped to face the ray is -Z.
        let cube = ConvexSolid::axis_box(Vec3::splat(-1.0), Vec3::splat(1.0));
        let r = Ray { origin: Vec3::ZERO, dir: Vec3::Z };
        let h = cube.intersect(&r, 1e-4, f32::INFINITY).expect("interior ray must hit exit");
        assert_relative_eq!(h.t, 1.0, epsilon = 1e-4);
        assert_relative_eq!(h.normal, -Vec3::Z, epsilon = 1e-4);
        assert!(h.normal.length() > 0.9, "exit normal must be non-zero");
    }

    #[test]
    fn prism_intersects_axial_ray() {
        // Inradius 1.0: the +Y face sits at y=1. Ray from (0,5,0) going -Y enters
        // at y=1 (t=4) with outward normal +Y.
        let prism = ConvexSolid::triangular_prism(1.0, 2.0);
        let r = Ray { origin: Vec3::new(0.0, 5.0, 0.0), dir: -Vec3::Y };
        let h = prism.intersect(&r, 1e-4, f32::INFINITY).expect("hit");
        assert_relative_eq!(h.t, 4.0, epsilon = 1e-3);
        assert_relative_eq!(h.normal, Vec3::Y, epsilon = 1e-3);
    }

    #[test]
    fn prism_misses_ray_outside_triangle() {
        // Ray along -Y offset to x=5, well outside the triangular cross-section.
        let prism = ConvexSolid::triangular_prism(1.0, 2.0);
        let r = Ray { origin: Vec3::new(5.0, 5.0, 0.0), dir: -Vec3::Y };
        assert!(prism.intersect(&r, 1e-4, f32::INFINITY).is_none());
    }
}

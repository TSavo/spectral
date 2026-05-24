//! Pinhole camera generating primary rays.

use crate::geom::Ray;
use glam::Vec3;

pub struct Camera {
    origin: Vec3,
    lower_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    /// `vfov_deg` vertical field of view; `aspect` = width/height.
    pub fn look_at(origin: Vec3, target: Vec3, up: Vec3, vfov_deg: f32, aspect: f32) -> Self {
        let theta = vfov_deg.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect * viewport_h;
        let w = (origin - target).normalize();
        let u = up.cross(w).normalize();
        let v = w.cross(u);
        let horizontal = viewport_w * u;
        let vertical = viewport_h * v;
        let lower_left = origin - horizontal / 2.0 - vertical / 2.0 - w;
        Camera { origin, lower_left, horizontal, vertical }
    }

    /// `s`,`t` in [0,1] across the image (origin bottom-left).
    pub fn primary_ray(&self, s: f32, t: f32) -> Ray {
        let dir = (self.lower_left + s * self.horizontal + t * self.vertical - self.origin)
            .normalize();
        Ray { origin: self.origin, dir }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_pixel_looks_at_target() {
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let r = cam.primary_ray(0.5, 0.5);
        assert!(r.dir.dot(-Vec3::Z) > 0.99, "center ray should look toward -Z");
    }

    #[test]
    fn rays_are_unit_length() {
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.5);
        for (s, t) in [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7)] {
            assert!((cam.primary_ray(s, t).dir.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn corner_rays_differ_from_center() {
        let cam = Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 60.0, 1.0);
        let center = cam.primary_ray(0.5, 0.5).dir;
        let corner = cam.primary_ray(0.0, 0.0).dir;
        assert!((center - corner).length() > 1e-2, "corner ray must differ from center");
    }
}

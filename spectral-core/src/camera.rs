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

    /// The camera's eye position (same for every pixel of a pinhole camera).
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// `s`,`t` in [0,1] across the image (origin bottom-left).
    pub fn primary_ray(&self, s: f32, t: f32) -> Ray {
        let dir = (self.lower_left + s * self.horizontal + t * self.vertical - self.origin)
            .normalize();
        Ray { origin: self.origin, dir }
    }

    /// Project a world point to screen coordinates (s, t) in [0,1] plus depth in
    /// front of the camera. Returns None if the point is behind the camera or
    /// outside the frame. Inverse of `primary_ray`.
    pub fn project(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let u = self.horizontal.normalize();
        let v = self.vertical.normalize();
        let vw = self.horizontal.length();
        let vh = self.vertical.length();
        // w points from target toward origin; the view direction is -w.
        let w = self.origin - self.horizontal * 0.5 - self.vertical * 0.5 - self.lower_left;
        let dir = p - self.origin;
        let c = dir.dot(w); // in front of camera => c < 0
        if c >= -1e-6 {
            return None;
        }
        let s = 0.5 + dir.dot(u) / (-c * vw);
        let t = 0.5 + dir.dot(v) / (-c * vh);
        if (0.0..1.0).contains(&s) && (0.0..1.0).contains(&t) {
            Some((s, t, -c))
        } else {
            None
        }
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

    #[test]
    fn project_inverts_primary_ray() {
        let cam = Camera::look_at(Vec3::new(0.0, 1.0, 5.0), Vec3::ZERO, Vec3::Y, 50.0, 1.5);
        for (s, t) in [(0.5, 0.5), (0.2, 0.7), (0.8, 0.35)] {
            let ray = cam.primary_ray(s, t);
            let p = ray.origin + ray.dir * 4.0; // a point along that pixel's ray
            let (ps, pt, depth) = cam.project(p).expect("point should be in frame");
            assert!((ps - s).abs() < 1e-3, "s: {ps} vs {s}");
            assert!((pt - t).abs() < 1e-3, "t: {pt} vs {t}");
            assert!(depth > 0.0, "depth must be positive in front of camera");
        }
    }
}

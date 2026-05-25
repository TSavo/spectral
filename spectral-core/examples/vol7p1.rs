//! VOL-7.1 eyeball: the DSOTM volumetric beam + fan rendered with the photon-BEAM
//! estimator (continuous beam splatting with transverse volume), at the default
//! face-on framing and at an extreme angle, to check the dim fan edges are smooth
//! (no blue/teal point dithering). Volumetric pass only (no surface rim).
//! Writes /tmp/vol7p1_dsotm.png and /tmp/vol7p1_extreme.png.

use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::{ConvexSolid, Plane};
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::volume::render_volumetric_scene;
use spectral_core::Xyz;

fn dsotm_scene() -> Scene {
    let k = 0.9_f32;
    let planes = vec![
        Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k },
        Plane { normal: Vec3::new(0.866_025_4, 0.5, 0.0), d: -k },
        Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },
        Plane { normal: Vec3::Z, d: -1.0 },
        Plane { normal: -Vec3::Z, d: -1.0 },
    ];
    let mut scene = Scene::new();
    scene.add_solid(ConvexSolid { planes }, Material::Dielectric { glass: Glass::Sf11 });
    scene
}

fn dsotm_beam() -> Beam {
    let bdir = Vec3::new(0.84, 0.54, 0.0).normalize();
    let bperp = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
    Beam {
        corner: Vec3::new(-2.6, -0.95, -0.6),
        u: Vec3::new(0.0, 0.0, 1.2),
        v: bperp,
        dir: bdir,
    }
}

fn render(path: &str, cam: &Camera, w: usize, h: usize, n: u32) {
    let scene = dsotm_scene();
    let beam = dsotm_beam();
    let zbuf = vec![f32::INFINITY; w * h];
    let img: Vec<Xyz> =
        render_volumetric_scene(&scene, cam, &beam, w, h, n, 0.5, 0.06, 0.5, 14.0, &zbuf, 0xDED);

    // Auto-expose to the 99.7th-percentile luminance (matches prism_dsotm.rs).
    let mut lums: Vec<f32> = img.iter().map(|p| p[1]).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lums[((lums.len() as f32 * 0.997) as usize).min(lums.len() - 1)].max(1e-9);
    let exposure = 1.0 / p;

    let mut out = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in img.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), exposure);
        out.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    out.save(path).unwrap();
    println!("wrote {path} ({w}x{h}, {n} photons)");
}

fn main() {
    let (w, h) = (400usize, 400usize);
    let n = 150_000u32;

    // Default DSOTM framing (face-on, looking down -Z).
    let cam_default = Camera::look_at(
        Vec3::new(0.5, -0.5, 12.0),
        Vec3::new(0.5, -0.5, 0.0),
        Vec3::Y,
        42.0,
        w as f32 / h as f32,
    );
    render("/tmp/vol7p1_dsotm.png", &cam_default, w, h, n);

    // Extreme angle: high + off-axis, putting the dim fan edges across the frame.
    let cam_extreme = Camera::look_at(
        Vec3::new(6.0, 4.0, 9.0),
        Vec3::new(0.8, -0.3, 0.0),
        Vec3::Y,
        38.0,
        w as f32 / h as f32,
    );
    render("/tmp/vol7p1_extreme.png", &cam_extreme, w, h, n);
}

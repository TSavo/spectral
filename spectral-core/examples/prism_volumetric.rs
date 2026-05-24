//! Volumetric demo: a white beam in haze passes through an SF11 prism and fans
//! into a rainbow. Writes prism_volumetric.png.

use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::ConvexSolid;
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::volume::render_volumetric;

fn main() {
    let (w, h) = (640usize, 480usize);
    let mut scene = Scene::new();
    scene.add_solid(ConvexSolid::wedge(30.0, 1.0, 1.0), Material::Dielectric { glass: Glass::Sf11 });

    // Pencil beam: thin in Y and Z, traveling +X into the wedge.
    let beam = Beam {
        corner: Vec3::new(-3.0, -0.03, -0.03),
        u: Vec3::new(0.0, 0.0, 0.06),
        v: Vec3::new(0.0, 0.06, 0.0),
        dir: Vec3::X,
    };

    // Camera looks down -Z at the XY plane where the fan spreads.
    let cam = Camera::look_at(
        Vec3::new(2.0, -2.0, 14.0),
        Vec3::new(2.0, -2.0, 0.0),
        Vec3::Y,
        50.0,
        w as f32 / h as f32,
    );

    let img = render_volumetric(&scene, &cam, &beam, w, h, 30_000_000, 0.6, 14.0, 0xF00D);

    // Auto-expose to the 99.5th percentile so the fan is bright but not blown.
    let mut lums: Vec<f32> = img.iter().map(|p| p[1]).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lums[((lums.len() as f32 * 0.995) as usize).min(lums.len() - 1)].max(1e-9);
    let exposure = 1.2 / p;

    let mut out = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in img.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), exposure);
        out.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    out.save("prism_volumetric.png").unwrap();
    println!("wrote prism_volumetric.png ({w}x{h})");
}

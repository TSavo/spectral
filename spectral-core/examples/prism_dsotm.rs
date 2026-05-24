//! "Dark Side of the Moon" composition: a white beam enters an equilateral glass
//! prism and a spectrum fans out the far side, glowing in haze on a black field.
//! Viewed face-on (down the prism's axis) so the dispersion fan spreads across
//! the frame. Writes prism_dsotm.png.

use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::{ConvexSolid, Plane};
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::volume::render_volumetric_scene;
use spectral_core::Xyz;
use glam::Vec3;

fn main() {
    let (w, h) = (800usize, 800usize);

    // Equilateral SF11 prism, apex up, triangular cross-section in XY, extruded
    // in Z. The two 120-degrees-apart upper-face normals make it equilateral.
    let k = 0.9_f32;
    let planes = vec![
        Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k }, // left face
        Plane { normal: Vec3::new(0.866_025_4, 0.5, 0.0), d: -k },  // right face
        Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },       // base
        Plane { normal: Vec3::Z, d: -1.0 },                         // +Z cap
        Plane { normal: -Vec3::Z, d: -1.0 },                        // -Z cap
    ];
    let mut scene = Scene::new();
    scene.add_solid(ConvexSolid { planes }, Material::Dielectric { glass: Glass::Sf11 });

    // White beam at the minimum-deviation angle for the 60-degree prism, so it
    // transmits (exit angle below the critical angle) instead of TIR-ing. It
    // enters the left face from the lower-left and refracts ~horizontally inside.
    let bdir = Vec3::new(0.84, 0.54, 0.0).normalize();
    let bperp = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05; // thin pencil
    let beam = Beam {
        corner: Vec3::new(-2.6, -0.95, -0.6),
        u: Vec3::new(0.0, 0.0, 1.2), // Z width (so the beam/fan are a sheet)
        v: bperp,
        dir: bdir,
    };

    // Camera face-on, looking down -Z at the XY plane where the fan spreads.
    let cam = Camera::look_at(
        Vec3::new(0.5, -0.5, 12.0),
        Vec3::new(0.5, -0.5, 0.0),
        Vec3::Y,
        42.0,
        w as f32 / h as f32,
    );

    let background: Xyz = [0.0, 0.0, 0.0];

    // SURFACE PASS: the prism as a dark glass triangle with luminous (Fresnel)
    // edges, on black. (No z-buffer occlusion of the haze — the beam and rainbow
    // are light and should glow through/over the glass.)
    let mut surface = vec![[0.0f32; 3]; w * h];
    for py in 0..h {
        for px in 0..w {
            let s = (px as f32 + 0.5) / w as f32;
            let t = 1.0 - (py as f32 + 0.5) / h as f32;
            let ray = cam.primary_ray(s, t);
            let idx = py * w + px;
            surface[idx] = if let Some((hit, _)) = scene.intersect(&ray) {
                let cos = hit.normal.dot(ray.dir).abs().clamp(0.0, 1.0);
                let rim = (1.0 - cos).powi(2);
                [0.015 + rim * 0.45, 0.02 + rim * 0.5, 0.04 + rim * 0.6]
            } else {
                background
            };
        }
    }

    // VOLUMETRIC PASS: the white beam and the dispersed rainbow fan, glowing in
    // haze. No occlusion (all-infinite z-buffer) so the light reads through the glass.
    let zbuf = vec![f32::INFINITY; w * h];
    let vol = render_volumetric_scene(
        &scene, &cam, &beam, w, h, 50_000_000,
        0.5,  // sigma_s
        0.06, // sigma_t
        0.5,  // g (forward)
        14.0, // max_dist
        &zbuf, 0xDED,
    );

    let mut img: Vec<Xyz> = surface
        .iter()
        .zip(vol.iter())
        .map(|(s, v)| [s[0] + v[0], s[1] + v[1], s[2] + v[2]])
        .collect();

    // Auto-expose to the 99.7th percentile luminance.
    let mut lums: Vec<f32> = img.iter().map(|p| p[1]).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lums[((lums.len() as f32 * 0.997) as usize).min(lums.len() - 1)].max(1e-9);
    let exposure = 1.0 / p;

    let mut out = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in img.iter_mut().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), exposure);
        out.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    out.save("prism_dsotm.png").unwrap();
    println!("wrote prism_dsotm.png ({w}x{h})");
}

//! Two-pass composed demo: a white beam through a glass prism casts a rainbow on
//! a wall, photographed by a camera. Writes prism_scene.png.

use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::{ConvexSolid, Ray};
use spectral_core::lighttrace::{trace, Beam, Screen};
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::Xyz;
use glam::Vec3;

fn shade_no_prism(
    ray: &Ray,
    beam_bar: &ConvexSolid,
    wall: &Screen,
    wall_gain: f32,
    background: Xyz,
    beam_glow: Xyz,
) -> Xyz {
    let beam_hit = beam_bar.intersect(ray, 1e-3, f32::INFINITY);
    let wall_hit = wall.screen_coords(ray);
    let t_beam = beam_hit.map(|hi| hi.t).unwrap_or(f32::INFINITY);
    let t_wall = wall_hit.map(|(_, _, t)| t).unwrap_or(f32::INFINITY);
    if t_beam < t_wall {
        beam_glow
    } else if let Some((s, r, _)) = wall_hit {
        let c = wall.sample(s, r);
        [
            background[0] + c[0] * wall_gain,
            background[1] + c[1] * wall_gain,
            background[2] + c[2] * wall_gain,
        ]
    } else {
        background
    }
}

fn main() {
    let (w, h) = (640usize, 480usize);

    // --- Geometry shared by both passes ---
    // The prism: a sub-critical SF11 wedge at the origin (transmits + disperses).
    let mut prism_scene = Scene::new();
    prism_scene.add_solid(ConvexSolid::wedge(30.0, 1.0, 6.0), Material::Dielectric { glass: Glass::Sf11 });

    // The wall: a plane at x = 8 facing -X (its caustic grid). u spans Z (width),
    // v spans Y (the downward region where the deviated beam lands, ~y=-4.5).
    let wall_corner = Vec3::new(8.0, -7.0, -3.5);
    let wall_u = Vec3::new(0.0, 0.0, 7.0); // Z width
    let wall_v = Vec3::new(0.0, 5.0, 0.0); // Y height (-7 .. -2)
    let mut wall = Screen::new(wall_corner, wall_u, wall_v, 512, 512);

    // The beam: collimated white light, thin in Y, wide in Z, traveling +X into
    // the prism. Also drawn as an emissive bar in pass 2 (the visible shaft).
    let beam = Beam {
        corner: Vec3::new(-3.0, -0.04, -2.5),
        u: Vec3::new(0.0, 0.0, 5.0),
        v: Vec3::new(0.0, 0.08, 0.0),
        dir: Vec3::X,
    };
    // Emissive bar geometry approximating the beam shaft from source to prism.
    let beam_bar = ConvexSolid::axis_box(Vec3::new(-3.0, -0.06, -2.6), Vec3::new(-0.2, 0.06, 2.6));

    // --- PASS 1: photons paint the rainbow on the wall ---
    trace(&prism_scene, &mut wall, &beam, 12_000_000, 0xCA05);

    // Normalize the wall caustic so its brightest texel is ~1.
    let wall_max = wall.sum.iter().map(|p| p[1]).fold(0.0f32, f32::max).max(1e-9);
    let wall_gain = 1.0 / wall_max;

    // --- PASS 2: camera renders the room ---
    let cam = Camera::look_at(
        Vec3::new(-1.5, 3.5, 9.0), // up and to the side
        Vec3::new(5.0, -3.0, 0.0), // look toward prism + wall region
        Vec3::Y,
        55.0,
        w as f32 / h as f32,
    );

    let background: Xyz = [0.01, 0.01, 0.012];
    let beam_glow: Xyz = [1.2, 1.2, 1.15];
    let glass_tint: Xyz = [0.10, 0.13, 0.16];

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for py in 0..h {
        for px in 0..w {
            let s = (px as f32 + 0.5) / w as f32;
            let t = 1.0 - (py as f32 + 0.5) / h as f32;
            let ray = cam.primary_ray(s, t);

            let prism_hit = prism_scene.intersect(&ray);
            let beam_hit = beam_bar.intersect(&ray, 1e-3, f32::INFINITY);
            let wall_hit = wall.screen_coords(&ray);
            let t_prism = prism_hit.as_ref().map(|(hi, _)| hi.t).unwrap_or(f32::INFINITY);
            let t_beam = beam_hit.map(|hi| hi.t).unwrap_or(f32::INFINITY);
            let t_wall = wall_hit.map(|(_, _, tt)| tt).unwrap_or(f32::INFINITY);

            let color: Xyz = if t_prism < t_beam && t_prism < t_wall {
                // Glassy prism: see the wall/beam through it (straight-through
                // approximation), tinted, with a bright Fresnel rim at grazing.
                let (hit, _) = prism_hit.unwrap();
                let behind = shade_no_prism(&ray, &beam_bar, &wall, wall_gain, background, beam_glow);
                let cos = hit.normal.dot(ray.dir).abs().clamp(0.0, 1.0);
                let rim = (1.0 - cos).powi(3); // grazing -> bright edge
                [
                    behind[0] * 0.75 + glass_tint[0] + rim * 0.8,
                    behind[1] * 0.75 + glass_tint[1] + rim * 0.8,
                    behind[2] * 0.75 + glass_tint[2] + rim * 0.9,
                ]
            } else {
                shade_no_prism(&ray, &beam_bar, &wall, wall_gain, background, beam_glow)
            };

            let rgb = tonemap_to_u8(xyz_to_linear_srgb(color), 1.0);
            img.put_pixel(px as u32, py as u32, image::Rgb(rgb));
        }
    }
    img.save("prism_scene.png").unwrap();
    println!("wrote prism_scene.png ({w}x{h})");
}

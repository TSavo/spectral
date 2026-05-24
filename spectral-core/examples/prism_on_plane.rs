//! Capstone: a triangular glass prism resting on a floor, a white beam entering,
//! the dispersed rainbow fanning through haze and spreading onto the floor. The
//! beam is shown by the haze (light is not a solid object). Writes prism_on_plane.png.

use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::{ConvexSolid, Ray};
use spectral_core::lighttrace::{trace, Beam, Screen};
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::volume::render_volumetric_scene;
use spectral_core::Xyz;
use glam::Vec3;

// Shade the floor (its caustic + ambient) ignoring the prism. Returns color + depth.
fn shade_floor(
    ray: &Ray,
    floor: &Screen,
    floor_gain: f32,
    floor_ambient: Xyz,
    background: Xyz,
) -> (Xyz, f32) {
    match floor.screen_coords(ray) {
        Some((s, r, t)) => {
            let c = floor.sample(s, r);
            (
                [
                    floor_ambient[0] + c[0] * floor_gain,
                    floor_ambient[1] + c[1] * floor_gain,
                    floor_ambient[2] + c[2] * floor_gain,
                ],
                t,
            )
        }
        None => (background, f32::INFINITY),
    }
}

/// Separable box blur over an XYZ buffer — a cheap reconstruction kernel that
/// smooths the photon caustic's Monte-Carlo speckle.
fn box_blur(buf: &[Xyz], w: usize, h: usize, radius: i32) -> Vec<Xyz> {
    let idx = |x: i32, y: i32| {
        (y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize
    };
    let norm = 1.0 / (2 * radius + 1) as f32;
    let mut tmp = vec![[0.0f32; 3]; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0.0f32; 3];
            for d in -radius..=radius {
                let p = buf[idx(x + d, y)];
                acc[0] += p[0];
                acc[1] += p[1];
                acc[2] += p[2];
            }
            tmp[idx(x, y)] = [acc[0] * norm, acc[1] * norm, acc[2] * norm];
        }
    }
    let mut out = vec![[0.0f32; 3]; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0.0f32; 3];
            for d in -radius..=radius {
                let p = tmp[idx(x, y + d)];
                acc[0] += p[0];
                acc[1] += p[1];
                acc[2] += p[2];
            }
            out[idx(x, y)] = [acc[0] * norm, acc[1] * norm, acc[2] * norm];
        }
    }
    out
}

fn main() {
    let (w, h) = (800usize, 600usize);

    // Floor at y = -1.5, normal +Y (u along Z, v along X so u x v = +Y). Large so
    // the fan, which exits high and descends, lands well out from the prism.
    let mut floor = Screen::new(
        Vec3::new(-5.0, -1.0, -5.0),
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(14.0, 0.0, 0.0),
        512,
        512,
    );

    // Prism: an SF11 triangular wedge resting on the floor (base at y = -1),
    // tall enough that the beam exits high above the plane.
    let prism = ConvexSolid::wedge(30.0, 1.0, 2.0);
    let mut scene = Scene::new();
    scene.add_solid(prism, Material::Dielectric { glass: Glass::Sf11 });

    // White beam: a collimated shaft entering +X high up the prism (y ~ 0.6), so
    // the dispersed fan has room to spread on its way down to the floor. Shown by
    // the haze in the volumetric pass (light is not a solid object).
    let beam = Beam {
        corner: Vec3::new(-3.0, 0.55, -0.75),
        u: Vec3::new(0.0, 0.0, 1.5),
        v: Vec3::new(0.0, 0.12, 0.0),
        dir: Vec3::X,
    };

    // PASS 1: photons paint the rainbow caustic onto the floor, then a small
    // reconstruction blur removes the Monte-Carlo speckle (density estimation).
    trace(&scene, &mut floor, &beam, 40_000_000, 0x0F1);
    floor.sum = box_blur(&floor.sum, 512, 512, 2);
    let floor_max = floor.sum.iter().map(|p| p[1]).fold(0.0f32, f32::max).max(1e-9);
    let floor_gain = 6.0 / floor_max;

    // Oblique camera, orbited toward the fan so more of the rainbow is visible.
    let cam = Camera::look_at(
        Vec3::new(0.0, 4.5, 9.0),
        Vec3::new(4.0, -1.2, 0.0),
        Vec3::Y,
        50.0,
        w as f32 / h as f32,
    );

    let background: Xyz = [0.005, 0.005, 0.008];
    let floor_ambient: Xyz = [0.03, 0.03, 0.035];
    let glass_tint: Xyz = [0.05, 0.07, 0.09];

    // SURFACE PASS + Z-BUFFER (euclidean nearest-solid depth per pixel).
    let mut surface = vec![[0.0f32; 3]; w * h];
    let mut zbuf = vec![f32::INFINITY; w * h];
    for py in 0..h {
        for px in 0..w {
            let s = (px as f32 + 0.5) / w as f32;
            let t = 1.0 - (py as f32 + 0.5) / h as f32;
            let ray = cam.primary_ray(s, t);
            let prism_hit = scene.intersect(&ray);
            let t_prism = prism_hit.as_ref().map(|(hi, _)| hi.t).unwrap_or(f32::INFINITY);
            let (floor_color, floor_t) = shade_floor(&ray, &floor, floor_gain, floor_ambient, background);

            let idx = py * w + px;
            if t_prism < floor_t {
                let (hit, _) = prism_hit.unwrap();
                let (behind, _) = shade_floor(&ray, &floor, floor_gain, floor_ambient, background);
                let cos = hit.normal.dot(ray.dir).abs().clamp(0.0, 1.0);
                let rim = (1.0 - cos).powi(3);
                surface[idx] = [
                    behind[0] * 0.7 + glass_tint[0] + rim * 0.6,
                    behind[1] * 0.7 + glass_tint[1] + rim * 0.6,
                    behind[2] * 0.7 + glass_tint[2] + rim * 0.7,
                ];
                zbuf[idx] = t_prism;
            } else {
                surface[idx] = floor_color;
                zbuf[idx] = floor_t;
            }
        }
    }

    // VOLUMETRIC PASS (haze single-scatter), z-buffer occluded, composited on top.
    let vol = render_volumetric_scene(
        &scene, &cam, &beam, w, h, 35_000_000,
        0.4,  // sigma_s
        0.12, // sigma_t
        0.6,  // g (forward)
        22.0, // max_dist
        &zbuf, 0xBEE5,
    );

    let mut img: Vec<Xyz> = surface
        .iter()
        .zip(vol.iter())
        .map(|(s, v)| [s[0] + v[0], s[1] + v[1], s[2] + v[2]])
        .collect();

    // Auto-expose to the 99.5th percentile luminance.
    let mut lums: Vec<f32> = img.iter().map(|p| p[1]).collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = lums[((lums.len() as f32 * 0.995) as usize).min(lums.len() - 1)].max(1e-9);
    let exposure = 1.1 / p;

    let mut out = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in img.iter_mut().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), exposure);
        out.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    out.save("prism_on_plane.png").unwrap();
    println!("wrote prism_on_plane.png ({w}x{h})");
}

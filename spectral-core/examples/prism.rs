//! Renders the direct-view prism over a horizon to prism.png (visual showpiece).

use spectral_core::camera::Camera;
use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb, Illuminant};
use spectral_core::geom::ConvexSolid;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use glam::Vec3;

fn main() {
    let (w, h) = (512usize, 512usize);
    let mut scene = Scene::new();
    scene.background = 0.3;
    scene.horizon = Some(4.0);
    scene.add_solid(
        ConvexSolid::triangular_prism(1.0, 2.0),
        Material::Dielectric { glass: Glass::Sf11 },
    );
    let cam = Camera::look_at(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        50.0,
        w as f32 / h as f32,
    );
    let mut tracer = CpuTracer::new(scene, cam, w, h, Illuminant::D65, 0xC0FFEE);
    tracer.accumulate(512);

    let mean = tracer.buffer().mean();
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in mean.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), 1.0);
        img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    img.save("prism.png").unwrap();
    println!("wrote prism.png ({w}x{h}, SF11 prism over a horizon edge)");
}

//! Forward light tracing: a white beam through an SF11 prism casts a dispersed
//! spectrum onto a screen. Writes prism_caustic.png.

use spectral_core::cie::{tonemap_to_u8, xyz_to_linear_srgb};
use spectral_core::geom::ConvexSolid;
use spectral_core::lighttrace::{trace, Beam, Screen};
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use glam::Vec3;

fn main() {
    let (w, h) = (512usize, 512usize);
    // A distant screen, positioned to catch the downward-deviated beam, so the
    // small angular dispersion spreads into a wide spectrum band.
    let mut screen = Screen::new(
        Vec3::new(12.0, -10.5, -2.0),
        Vec3::new(0.0, 0.0, 4.0), // u: Z (width)
        Vec3::new(0.0, 6.0, 0.0), // v: Y (height)
        w,
        h,
    );
    // A collimated white beam: thin in Y (clean wavelength separation), wide in Z
    // (the spectrum becomes a broad horizontal band).
    let beam = Beam {
        corner: Vec3::new(-3.0, -0.02, -1.5),
        u: Vec3::new(0.0, 0.0, 3.0),
        v: Vec3::new(0.0, 0.04, 0.0),
        dir: Vec3::X,
    };
    let mut scene = Scene::new();
    scene.add_solid(ConvexSolid::wedge(30.0, 1.0, 4.0), Material::Dielectric { glass: Glass::Sf11 });

    let n: u32 = 8_000_000;
    trace(&scene, &mut screen, &beam, n, 0xBEEF);

    // Scale so the brightest pixel is well-exposed, then tonemap.
    let raw = screen.scaled(1.0);
    let max_y = raw.iter().map(|p| p[1]).fold(0.0f32, f32::max).max(1e-9);
    let exposure = 1.5 / max_y;
    let mut img = image::RgbImage::new(w as u32, h as u32);
    for (i, px) in raw.iter().enumerate() {
        let rgb = tonemap_to_u8(xyz_to_linear_srgb(*px), exposure);
        img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(rgb));
    }
    img.save("prism_caustic.png").unwrap();
    println!("wrote prism_caustic.png ({n} photons, SF11 prism caustic)");
}

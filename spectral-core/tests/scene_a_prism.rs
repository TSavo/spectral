//! Acceptance Scene A: dispersion through a prism.
//! 1) Analytic gate: refraction through a BK7 prism interface disperses with the
//!    correct chromatic ordering and an angular spread matching Sellmeier.
//! 2) Render gate: a dispersive glass prism colors a horizon edge far more than a
//!    non-dispersive diffuse prism of the same shape.

use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::geom::ConvexSolid;
use spectral_core::material::Material;
use spectral_core::optics::refract;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use glam::Vec3;

/// Deviation through a single flat interface entering BK7, for wavelength λ.
fn deviation(d: Vec3, normal: Vec3, lambda: f32) -> f32 {
    let n = Glass::Bk7.n(lambda);
    let t = refract(d, normal, 1.0, n).unwrap();
    d.angle_between(t)
}

#[test]
fn chromatic_ordering_blue_deviates_more_than_red() {
    let normal = Vec3::Y;
    let d = Vec3::new(0.5, -0.8660254, 0.0).normalize(); // 30 deg incidence
    let dev_blue = deviation(d, normal, 450.0);
    let dev_red = deviation(d, normal, 650.0);
    assert!(dev_blue > dev_red, "blue must deviate more than red: {dev_blue} vs {dev_red}");
}

#[test]
fn angular_spread_matches_analytic() {
    let normal = Vec3::Y;
    let theta_i = 30.0_f32.to_radians();
    let d = Vec3::new(theta_i.sin(), -theta_i.cos(), 0.0).normalize();
    for lambda in [440.0_f32, 520.0, 600.0, 680.0] {
        let n = Glass::Bk7.n(lambda);
        let theta_t = (theta_i.sin() / n).asin();
        let analytic_dev = theta_i - theta_t; // bends toward normal entering glass
        let measured = deviation(d, normal, lambda);
        let err = (measured - analytic_dev).abs() / analytic_dev;
        assert!(err < 0.03, "λ={lambda}: relative error {err} exceeds 3%");
    }
}

/// Max chromaticity distance from the D65 white point across all pixels — a
/// scalar measure of how strongly colored the rendered image is.
fn chroma_spread(t: &CpuTracer) -> f32 {
    t.buffer()
        .mean()
        .iter()
        .filter_map(|px| {
            let s = px[0] + px[1] + px[2];
            if s <= 1e-5 {
                return None;
            }
            let (x, y) = (px[0] / s, px[1] / s);
            Some(((x - 0.3127f32).powi(2) + (y - 0.3290f32).powi(2)).sqrt())
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn dispersive_prism_colors_horizon_more_than_diffuse() {
    let setup = |mat: Material| {
        let mut scene = Scene::new();
        scene.background = 0.3; // dim ambient
        scene.horizon = Some(4.0); // bright sky above the horizon edge
        scene.add_solid(ConvexSolid::triangular_prism(1.0, 2.0), mat);
        let cam = Camera::look_at(
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            50.0,
            1.0,
        );
        let mut t = CpuTracer::new(scene, cam, 128, 128, Illuminant::D65, 11);
        t.accumulate(256);
        t
    };
    let glass = setup(Material::Dielectric { glass: Glass::Sf11 });
    let diffuse = setup(Material::Lambertian { reflectance: 0.8 });
    let cg = chroma_spread(&glass);
    let cd = chroma_spread(&diffuse);
    assert!(cg > 0.05, "dispersive prism should color the horizon edge strongly, got {cg}");
    assert!(cg > 2.0 * cd, "glass dispersion ({cg}) must clearly exceed the diffuse control ({cd})");
}

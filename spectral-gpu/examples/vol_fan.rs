//! VOL-3: converged DSOTM volumetric fan (full single-scatter weight).
//!
//! Emits a high photon count (chunked to dodge the Metal watchdog) from the DSOTM
//! beam through the SF11 prism, applying the full phase·transmittance/d²·seg_len/4
//! single-scatter camera-connection weight (mirrors render_volumetric_scene), then
//! auto-exposes and writes /tmp/vol3_fan.png.
//!
//! Expect: the glowing haze beam entering the prism + the rainbow fan spreading on
//! the exit side — the VOLUMETRIC pass of prism_dsotm.png, WITHOUT the surface rim
//! (that is VOL-5).
//!
//! Run:
//!   cargo run -p spectral-gpu --example vol_fan --release

use image::{ImageBuffer, Rgb};
use spectral_core::camera::Camera;
use spectral_core::geom::{ConvexSolid, Plane};
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_gpu::vol_photons::{GpuPhotonFilm, VolWeights};
use spectral_gpu::GpuContext;
use glam::Vec3;

const WIDTH:     usize = 800;
const HEIGHT:    usize = 800;
const N_PHOTONS: u32   = 16_000_000; // chunked under MAX_PER_DISPATCH
const SEED:      u32   = 0xF00D_BEEF;
const MAX_DIST:  f32   = 14.0;

fn main() {
    let ctx = GpuContext::new().expect("no GPU adapter available");

    // Exact prism_dsotm.rs scene.
    let k = 0.9_f32;
    let planes = vec![
        Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k },
        Plane { normal: Vec3::new( 0.866_025_4, 0.5, 0.0), d: -k },
        Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },
        Plane { normal: Vec3::Z,  d: -1.0 },
        Plane { normal: -Vec3::Z, d: -1.0 },
    ];
    let mut scene = Scene::new();
    scene.background = 0.0;
    scene.add_solid(ConvexSolid { planes }, Material::Dielectric { glass: Glass::Sf11 });

    let bdir  = Vec3::new(0.84, 0.54, 0.0).normalize();
    let bperp = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
    let beam = Beam {
        corner: Vec3::new(-2.6, -0.95, -0.6),
        u:      Vec3::new(0.0, 0.0, 1.2),
        v:      bperp,
        dir:    bdir,
    };

    let cam = Camera::look_at(
        Vec3::new(0.5, -0.5, 12.0),
        Vec3::new(0.5, -0.5,  0.0),
        Vec3::Y,
        42.0,
        WIDTH as f32 / HEIGHT as f32,
    );

    // DSOTM weights: sigma_s=0.5, sigma_t=0.06, g=0.5; all-INF zbuffer (no occlusion).
    let weights = VolWeights::default();

    println!(
        "VOL-3: {}x{} film, {} photons (chunked), seed=0x{:08X}",
        WIDTH, HEIGHT, N_PHOTONS, SEED
    );
    println!("  weights: sigma_s={}, sigma_t={}, g={}, all-INF zbuffer",
        weights.sigma_s, weights.sigma_t, weights.g);

    let mut film = GpuPhotonFilm::new(
        &ctx, scene, &cam, &beam,
        WIDTH, HEIGHT, N_PHOTONS, SEED, MAX_DIST,
        weights, None,
    );

    // Chunked dispatch: each chunk <= MAX_PER_DISPATCH, resolved into the accum.
    film.trace_chunked(N_PHOTONS, SEED);

    let exposure = film.auto_expose(1.0);
    println!("  auto-exposure = {exposure:.3}");

    let rgb8 = film.to_rgb8(exposure);

    let path = "/tmp/vol3_fan.png";
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, rgb8)
            .expect("buffer size mismatch");
    img.save(path).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));

    let meta = std::fs::metadata(path).expect("PNG missing after save");
    println!("  saved {path}  ({} bytes)", meta.len());
    println!("  image: {}x{}", WIDTH, HEIGHT);
    println!("  photon count: {N_PHOTONS}");

    let total_y: f32 = (0..WIDTH * HEIGHT).map(|i| film.accum()[3 * i + 1]).sum();
    if total_y < 1e-6 {
        eprintln!("WARNING: film Y is essentially zero ({total_y:.6}) — the weighted \
                   single-scatter pass produced no light. Check sigma/phase/d² terms.");
        std::process::exit(1);
    }
    println!("  total Y luminance = {total_y:.4}  (weighted single-scatter fan)");
}

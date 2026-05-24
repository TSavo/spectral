//! VOL-2: Forward photon kernel + constellation-of-dots PNG.
//!
//! Emits N photons from the DSOTM beam through the SF11 prism, splats each
//! photon's scatter points into the atomic film, resolves, tonemaps, writes
//! /tmp/vol2_photons.png.
//!
//! PNG gate: the dispersed rainbow fan must appear as a constellation of
//! colored points — white-ish beam entering the prism, fanning into
//! red->violet dots on the exit side. No haze glow (that's VOL-3).
//!
//! Run:
//!   cargo run -p spectral-gpu --example vol_photons --release

use image::{ImageBuffer, Rgb};
use spectral_core::camera::Camera;
use spectral_core::geom::{ConvexSolid, Plane};
use spectral_core::lighttrace::Beam;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_gpu::vol_photons::GpuPhotonFilm;
use spectral_gpu::GpuContext;
use glam::Vec3;

const WIDTH:     usize = 800;
const HEIGHT:    usize = 800;
const N_PHOTONS: u32   = 4_000_000;
const SEED:      u32   = 0xDEAD_CAFE;
const MAX_DIST:  f32   = 14.0;

fn main() {
    let ctx = GpuContext::new().expect("no GPU adapter available");

    // Exact prism_dsotm.rs scene -----------------------------------------------
    let k = 0.9_f32;
    let planes = vec![
        Plane { normal: Vec3::new(-0.866_025_4, 0.5, 0.0), d: -k }, // left face
        Plane { normal: Vec3::new( 0.866_025_4, 0.5, 0.0), d: -k }, // right face
        Plane { normal: Vec3::new(0.0, -1.0, 0.0), d: -1.0 },       // base
        Plane { normal: Vec3::Z,  d: -1.0 },                         // +Z cap
        Plane { normal: -Vec3::Z, d: -1.0 },                         // -Z cap
    ];
    let mut scene = Scene::new();
    scene.background = 0.0;
    scene.add_solid(ConvexSolid { planes }, Material::Dielectric { glass: Glass::Sf11 });

    // White beam at the minimum-deviation angle for the 60-degree SF11 prism.
    let bdir  = Vec3::new(0.84, 0.54, 0.0).normalize();
    let bperp = Vec3::new(-0.54, 0.84, 0.0).normalize() * 0.05;
    let beam = Beam {
        corner: Vec3::new(-2.6, -0.95, -0.6),
        u:      Vec3::new(0.0, 0.0, 1.2), // Z-width -> sheet beam
        v:      bperp,
        dir:    bdir,
    };

    // Camera face-on, looking down -Z at the XY plane where the fan spreads.
    let cam = Camera::look_at(
        Vec3::new(0.5, -0.5, 12.0),
        Vec3::new(0.5, -0.5,  0.0),
        Vec3::Y,
        42.0,
        WIDTH as f32 / HEIGHT as f32,
    );

    println!(
        "VOL-2: {}x{} film, {} photons, seed=0x{:08X}",
        WIDTH, HEIGHT, N_PHOTONS, SEED
    );

    // NOTE: since VOL-3 the kernel applies the full single-scatter weight, so this
    // now renders the weighted fan (not the VOL-2 unit-weight constellation). The
    // dedicated VOL-3 eyeball example is vol_fan.rs.
    let mut film = GpuPhotonFilm::new(
        &ctx, scene, &cam, &beam,
        WIDTH, HEIGHT, N_PHOTONS, SEED, MAX_DIST,
        spectral_gpu::vol_photons::VolWeights::default(), None,
    );

    film.trace_and_resolve(N_PHOTONS, SEED);

    // Auto-expose: scale so the brightest pixel is near white.
    let exposure = film.auto_expose(1.0);
    println!("  auto-exposure = {exposure:.3}");

    let rgb8 = film.to_rgb8(exposure);

    let path = "/tmp/vol2_photons.png";
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, rgb8)
            .expect("buffer size mismatch");
    img.save(path).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));

    let meta = std::fs::metadata(path).expect("PNG missing after save");
    println!("  saved {path}  ({} bytes)", meta.len());
    println!("  image: {}x{}", WIDTH, HEIGHT);
    println!("  photon count: {N_PHOTONS}");

    // Sanity: at least some pixels should be nonzero.
    let total_y: f32 = (0..WIDTH * HEIGHT)
        .map(|i| film.accum()[3 * i + 1])
        .sum();
    if total_y < 1e-6 {
        eprintln!("WARNING: film Y is essentially zero ({total_y:.6}) — dispersion or \
                   camera projection may be wrong. Check:
  1. Does sellmeier_n vary with lambda? (It should for SF11)
  2. Is the beam at minimum-deviation so it transmits? (bdir = (0.84,0.54,0) normalized)
  3. Is camera_project correct? (In-front check: dot(dir, w_vec) must be < -1e-6)
  4. Are photon scatter points actually projected on-screen?");
        std::process::exit(1);
    }
    println!("  total Y luminance = {total_y:.4}  (nonzero: dispersion visible)");
}

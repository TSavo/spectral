//! VOL-5: full DSOTM composite — glass body surface rim + glowing volumetric fan.
//!
//! Computes the static surface-rim pass (one camera ray per pixel, Fresnel-style
//! rim) ONCE, traces the volumetric single-scatter fan at a high photon count
//! (chunked under the Metal watchdog), composites surface + fan, auto-exposes to
//! the 99.7th-percentile luminance (mirroring prism_dsotm.rs), and writes
//! /tmp/vol5_dsotm.png.
//!
//! Should visually match spectral-core's prism_dsotm.png: a dark blue glass prism
//! body with luminous edges, the white beam entering, and the rainbow fan glowing
//! out the exit side.
//!
//! Run:
//!   cargo run -p spectral-gpu --example vol_dsotm --release

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
const SEED:      u32   = 0xDED;
const MAX_DIST:  f32   = 14.0;

fn main() {
    let ctx = GpuContext::new().expect("no GPU adapter available");

    // Exact prism_dsotm.rs scene + beam + camera.
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

    let weights = VolWeights::default(); // sigma_s=0.5, sigma_t=0.06, g=0.5

    println!(
        "VOL-5: {}x{} full composite, {} photons (chunked), seed=0x{:X}",
        WIDTH, HEIGHT, N_PHOTONS, SEED
    );

    let mut film = GpuPhotonFilm::new(
        &ctx, scene, &cam, &beam,
        WIDTH, HEIGHT, N_PHOTONS, SEED, MAX_DIST,
        weights, None,
    );

    // 1. Surface rim pass (static, computed once per camera).
    film.recompute_surface(&cam);
    let surf_y: f32 = (0..WIDTH * HEIGHT).map(|i| film.surface()[3 * i + 1]).sum();
    println!("  surface rim: total Y = {surf_y:.4}");

    // 2. Volumetric fan (chunked).
    film.trace_chunked(N_PHOTONS, SEED);
    let vol_y: f32 = (0..WIDTH * HEIGHT).map(|i| film.accum()[3 * i + 1]).sum();
    println!("  volumetric fan: total Y = {vol_y:.4}");

    // 3. Composite (surface + fan) and auto-expose to the 99.7th percentile.
    let exposure = film.auto_expose_p997();
    println!("  auto-exposure (1/p99.7) = {exposure:.4}");

    let rgb8 = film.to_rgb8(exposure);

    let path = "/tmp/vol5_dsotm.png";
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, rgb8)
            .expect("buffer size mismatch");
    img.save(path).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));

    let meta = std::fs::metadata(path).expect("PNG missing after save");
    println!("  saved {path}  ({} bytes)", meta.len());
    println!("  image: {}x{}", WIDTH, HEIGHT);
    println!("  photon count: {N_PHOTONS}");

    if surf_y <= 0.0 || vol_y <= 0.0 {
        eprintln!("WARNING: surface ({surf_y}) or volumetric ({vol_y}) pass produced no light.");
        std::process::exit(1);
    }
}

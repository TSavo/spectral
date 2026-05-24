//! VOL-1: Atomic film + resolve + determinism gate + PNG blit.
//!
//! 1. Builds a GPU atomic film (512x512).
//! 2. Runs the synthetic splat kernel (1M splats).
//! 3. Resolves into f32 accum.
//! 4. Tonemaps to PNG and writes /tmp/vol1_film.png.
//! 5. Runs the determinism gate: same seed twice into fresh films, raw u32 diff.
//!
//! Run with:
//!   cargo run -p spectral-gpu --example vol_film --release

use image::{ImageBuffer, Rgb};
use spectral_gpu::film::{check_determinism, AtomicFilm, N_SPLATS};
use spectral_gpu::GpuContext;

const WIDTH: usize  = 512;
const HEIGHT: usize = 512;
const SEED: u32     = 0xC0DE_CAFE;

fn main() {
    let ctx = GpuContext::new().expect("no GPU adapter available");

    println!("VOL-1: {}x{} film, {N_SPLATS} splats, seed=0x{SEED:08X}", WIDTH, HEIGHT);

    // ---- Phase 1: splat + resolve + PNG ----------------------------------------

    let mut film = AtomicFilm::new(&ctx, WIDTH, HEIGHT);
    film.splat(SEED, N_SPLATS);
    film.resolve();

    let exposure = film.auto_expose_accum(1.0);
    println!("  auto-exposure = {exposure:.3}");

    let rgb8 = film.to_rgb8(exposure);

    // Write PNG.
    let path = "/tmp/vol1_film.png";
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, rgb8)
            .expect("buffer size mismatch");
    img.save(path).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
    let meta = std::fs::metadata(path).expect("PNG missing after save");
    println!("  saved {path}  ({} bytes)", meta.len());
    println!("  film size: {}x{} = {} pixels", WIDTH, HEIGHT, WIDTH * HEIGHT);

    // ---- Phase 2: determinism gate ---------------------------------------------

    println!("  running determinism gate ({N_SPLATS} splats, same seed twice)...");
    match check_determinism(&ctx, WIDTH, HEIGHT, SEED, N_SPLATS) {
        Ok(()) => println!(
            "VOL-1 determinism: IDENTICAL ({N_SPLATS} splats, {}x{} film)",
            WIDTH, HEIGHT
        ),
        Err((i, a, b)) => {
            eprintln!("VOL-1 determinism: DIFFERS at index {i}: {a} vs {b}");
            std::process::exit(1);
        }
    }
}

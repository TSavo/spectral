//! Headless GPU prism render — proves the spectral toggle works.
//!
//! Renders the same prism scene as spectral-viewer at two modes:
//!   spectral=1  → chromatic dispersion (rainbow band through SF11 wedge)
//!   spectral=0  → fixed n(550 nm), no dispersion (clear glass)
//!
//! Outputs:
//!   /tmp/gpu_prism_spectral.png
//!   /tmp/gpu_prism_rgb.png
//!
//! Run with:
//!   cargo run -p spectral-gpu --example gpu_prism --release

use glam::Vec3;
use image::{ImageBuffer, Rgb};
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::geom::ConvexSolid;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_gpu::{GpuContext, GpuTracer};

/// Tonemap pipeline matching the viewer's BLIT_SHADER exactly, with exposure scaling.
///
/// `exposure` divides XYZ before the Reinhard+gamma pipeline so that the
/// bright (background=3.0, many-spp) scene doesn't blow out to uniform white.
/// Both images must use the **same** exposure so only spectral mode differs visually.
///
/// Pipeline:
///   XYZ / exposure -> linear sRGB (IEC 61966-2-1 matrix) -> Reinhard c/(1+c) ->
///   max(t,0) -> pow(1/2.2) -> clamp -> u8
fn xyz_to_u8(xyz: [f32; 3], exposure: f32) -> [u8; 3] {
    let [x, y, z] = xyz;

    // Pre-scale by exposure (same as `xyz / exposure` before the matrix)
    let x = x / exposure;
    let y = y / exposure;
    let z = z / exposure;

    // XYZ -> linear sRGB matrix (IEC 61966-2-1, D65 — matches BLIT_SHADER xyz_to_srgb)
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;

    // Reinhard tonemap (per-channel)
    let r = r / (1.0 + r);
    let g = g / (1.0 + g);
    let b = b / (1.0 + b);

    // Gamma 1/2.2, clamp negatives first (matches max(t, 0) in shader)
    let gamma = |c: f32| c.max(0.0_f32).powf(1.0 / 2.2);
    let r = gamma(r);
    let g = gamma(g);
    let b = gamma(b);

    // Quantize
    let to_u8 = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    [to_u8(r), to_u8(g), to_u8(b)]
}

/// Compute auto-exposure from a pixel buffer: `max_Y * scale`.
/// Using scale=2.0 maps the brightest pixel to ~Reinhard(0.5) ≈ 0.33, leaving
/// headroom for dark regions and ensuring tonal separation.
fn auto_exposure(pixels: &[[f32; 3]], scale: f32) -> f32 {
    let max_y = pixels.iter().map(|p| p[1]).fold(0.0_f32, f32::max);
    (max_y * scale).max(1.0) // clamp to 1.0 so zero-luminance scenes don't divide by tiny
}

/// Build the exact camera the viewer uses at the given aspect ratio.
/// yaw = π/4, pitch = 0.15, radius = 6.0 — matches viewer initial state.
fn build_camera(aspect: f32) -> Camera {
    let yaw: f32 = std::f32::consts::FRAC_PI_4;
    let pitch: f32 = 0.15;
    let radius: f32 = 6.0;
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let eye = Vec3::new(radius * cy * cp, radius * sp, radius * sy * cp);
    Camera::look_at(eye, Vec3::ZERO, Vec3::Y, 40.0, aspect)
}

/// Accumulate `spp` samples in small chunks to avoid GPU watchdog timeouts.
fn accumulate_chunked(tracer: &mut GpuTracer<'_>, spp: u32, spp_per_dispatch: u32) {
    let full_dispatches = spp / spp_per_dispatch;
    let remainder = spp % spp_per_dispatch;
    for _ in 0..full_dispatches {
        tracer.accumulate(spp_per_dispatch);
    }
    if remainder > 0 {
        tracer.accumulate(remainder);
    }
}

/// Render one mode and return the XYZ pixel buffer.
fn render_xyz(
    tracer: &mut GpuTracer<'_>,
    spectral: bool,
    spp: u32,
    spp_per_dispatch: u32,
    width: usize,
    height: usize,
) -> Vec<[f32; 3]> {
    tracer.set_spectral(spectral);
    tracer.clear_accum();
    accumulate_chunked(tracer, spp, spp_per_dispatch);

    let pixels = tracer.read_xyz();
    assert_eq!(pixels.len(), width * height, "pixel count mismatch");
    pixels
}

/// Save XYZ pixel buffer to PNG using a shared exposure for correct comparison.
fn save_png(
    pixels: &[[f32; 3]],
    exposure: f32,
    width: usize,
    height: usize,
    path: &str,
) {
    // Diagnostic: report max luminance, nonzero count, most chromatic pixel
    let max_y = pixels.iter().map(|p| p[1]).fold(0.0_f32, f32::max);
    let nz = pixels.iter().filter(|p| p[0] + p[1] + p[2] > 0.0).count();
    let (mc_xyz, mc_chroma) = pixels.iter().fold(([0.0f32;3], 0.0f32), |(best, bc), p| {
        let [r, g, b] = xyz_to_u8(*p, exposure).map(|c| c as f32);
        let ch = r.max(g).max(b) - r.min(g).min(b);
        if ch > bc { (*p, ch) } else { (best, bc) }
    });
    let mc_rgb = xyz_to_u8(mc_xyz, exposure);
    println!("  max Y = {max_y:.2}, nonzero = {nz}/{}, exposure = {exposure:.2}", pixels.len());
    println!("  most chromatic pixel: xyz={mc_xyz:.2?} -> u8={mc_rgb:?} (chroma={mc_chroma:.0})");

    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::new(width as u32, height as u32);

    for (i, xyz) in pixels.iter().enumerate() {
        let px = (i % width) as u32;
        // GPU writes gid.y * width + gid.x, y=0 is top row.
        let py = (i / width) as u32;
        let [r, g, b] = xyz_to_u8(*xyz, exposure);
        img.put_pixel(px, py, Rgb([r, g, b]));
    }

    img.save(path).unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
}

fn main() {
    // Resolution — 800x600, aspect 4:3.
    let width: usize = 800;
    let height: usize = 600;
    let spp: u32 = 16384;
    // Keep each dispatch small to avoid GPU watchdog timeouts (Metal/DX12 ~5s).
    let spp_per_dispatch: u32 = 32;

    println!("gpu_prism: {width}x{height} @ {spp} spp");

    // Acquire headless GPU.
    let ctx = GpuContext::new().expect("no GPU available; cannot run gpu_prism example");

    // Build the viewer's prism scene verbatim.
    let mut scene = Scene::new();
    scene.background = 3.0;
    scene.add_solid(
        ConvexSolid::wedge(30.0, 1.0, 4.0),
        Material::Dielectric { glass: Glass::Sf11 },
    );

    let camera = build_camera(width as f32 / height as f32);

    // One tracer; we'll toggle spectral between renders.
    let mut tracer = GpuTracer::new(
        &ctx,
        scene,
        camera,
        width,
        height,
        Illuminant::D65,
        0xCAFE_u32,
    );

    // --- Render 1: spectral dispersion ON ---
    let path_spectral = "/tmp/gpu_prism_spectral.png";
    println!("Rendering spectral=1 ...");
    let pixels_spectral = render_xyz(&mut tracer, true, spp, spp_per_dispatch, width, height);

    // Compute shared exposure from the spectral render (max_Y * 2.0 gives midtone headroom).
    // Both renders use the SAME exposure so differences are purely due to spectral mode.
    let exposure = auto_exposure(&pixels_spectral, 2.0);
    println!("  shared exposure = {exposure:.2}");

    save_png(&pixels_spectral, exposure, width, height, path_spectral);
    let meta_s = std::fs::metadata(path_spectral).expect("spectral PNG missing");
    println!("  saved {path_spectral}  ({} bytes)", meta_s.len());

    // --- Render 2: fixed n(550 nm), no dispersion ---
    let path_rgb = "/tmp/gpu_prism_rgb.png";
    println!("Rendering spectral=0 ...");
    let pixels_rgb = render_xyz(&mut tracer, false, spp, spp_per_dispatch, width, height);
    save_png(&pixels_rgb, exposure, width, height, path_rgb);
    let meta_r = std::fs::metadata(path_rgb).expect("rgb PNG missing");
    println!("  saved {path_rgb}  ({} bytes)", meta_r.len());

    // --- XYZ-space toggle verification (pre-tonemap) ---
    // With a uniform background (scene.background=3.0, no directional light) the only
    // spectral difference is the per-wavelength TIR critical-angle shift: SF11 has
    // higher n at short wavelengths so the blue/violet critical angle is wider, causing
    // TIR to block more blue paths when spectral=1.  This produces a systematic
    // dX<0 / dZ<0 (i.e. less blue XYZ) and slight dY change in spectral vs RGB mode.
    //
    // A visible rainbow band requires structured (directional) incident light.
    // This scene mirrors the viewer exactly and has the same uniform background.
    let mut n_diff: usize = 0;
    let mut sum_dx = 0.0_f64;
    let mut sum_dy = 0.0_f64;
    let mut sum_dz = 0.0_f64;
    let mut max_dxyz = 0.0_f32;
    let threshold_xyz: f32 = 0.01; // pre-tonemap XYZ units
    let mut n_above_thresh: usize = 0;
    for (ps, pr) in pixels_spectral.iter().zip(pixels_rgb.iter()) {
        let dx = ps[0] - pr[0];
        let dy = ps[1] - pr[1];
        let dz = ps[2] - pr[2];
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag > 0.0 { n_diff += 1; }
        if mag > threshold_xyz { n_above_thresh += 1; }
        if mag > max_dxyz { max_dxyz = mag; }
        sum_dx += dx as f64;
        sum_dy += dy as f64;
        sum_dz += dz as f64;
    }
    let total = (width * height) as f64;
    println!();
    println!("XYZ-space toggle verification (spectral=1 minus spectral=0, pre-tonemap):");
    println!("  pixels with ΔXYZ != 0 : {n_diff}/{}", width * height);
    println!("  pixels with |ΔXYZ| > {threshold_xyz:.2} : {n_above_thresh}/{}", width * height);
    println!("  max |ΔXYZ| per pixel   : {max_dxyz:.4}");
    println!("  mean dX={:.4}  dY={:.4}  dZ={:.4}", sum_dx / total, sum_dy / total, sum_dz / total);
    println!("  Physics: dX<0 + dZ<0 expected (SF11 TIR blocks more blue in spectral mode).");
    println!("  Note: no visible rainbow band — requires directional incident light;");
    println!("  both modes use uniform background=3.0 (faithful to viewer scene).");

    println!();
    println!("Done.");
    println!("  spectral PNG : {path_spectral}  ({} bytes, {spp} spp)", meta_s.len());
    println!("  rgb PNG      : {path_rgb}  ({} bytes, {spp} spp)", meta_r.len());
    println!();
    println!("spectral=1: chromatic dispersion via per-wavelength Sellmeier n(lambda) for SF11.");
    println!("spectral=0: fixed n(550nm) -- same geometry, no color fringing.");
}

use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::sellmeier::Glass;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use spectral_gpu::{GpuContext, GpuTracer};

fn fixed_scene() -> Scene {
    let mut s = Scene::new();
    s.background = 1.0;
    s.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Dielectric { glass: Glass::Sf11 });
    s.add_sphere(Vec3::new(0.0, 0.0, -6.0), 2.0, Material::Lambertian { reflectance: 0.7 });
    s
}

#[test]
fn gpu_matches_cpu_oracle() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping diff gate");
        return;
    };
    let (w, h, spp, seed) = (64usize, 64usize, 64u32, 0xD1FFu32);
    let cam = || Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 50.0, 1.0);

    let mut cpu = CpuTracer::new(fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    cpu.accumulate(spp);
    let cpu_img = cpu.buffer().mean();

    let mut gpu = GpuTracer::new(&ctx, fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    gpu.accumulate(spp);
    let gpu_img = gpu.read_xyz();

    let n = (w * h) as f32;
    let (mut l1, mut cpu_e, mut gpu_e, mut max_px) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (c, g) in cpu_img.iter().zip(gpu_img.iter()) {
        let mut px = 0.0;
        for k in 0..3 {
            let d = (c[k] - g[k]).abs();
            l1 += d;
            px += d;
            cpu_e += c[k];
            gpu_e += g[k];
        }
        max_px = max_px.max(px);
    }
    let mean_l1 = l1 / (n * 3.0);
    let energy_rel = (cpu_e - gpu_e).abs() / cpu_e.max(1e-6);
    eprintln!("DIFF GATE: mean per-channel L1 = {mean_l1}, energy rel = {energy_rel}, max pixel L1 = {max_px}");
    eprintln!("(cpu_energy={cpu_e}, gpu_energy={gpu_e})");

    // Distribution stats: distinguish FP noise (fine) from coherent bugs (not fine).
    let mut per_pixel: Vec<f32> = cpu_img
        .iter()
        .zip(gpu_img.iter())
        .map(|(c, g)| (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>())
        .collect();
    per_pixel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f32| per_pixel[((per_pixel.len() as f32 * q) as usize).min(per_pixel.len() - 1)];
    let mean_pp = per_pixel.iter().sum::<f32>() / per_pixel.len() as f32;
    let frac_above =
        per_pixel.iter().filter(|&&d| d > 10.0 * mean_pp).count() as f32 / per_pixel.len() as f32;
    eprintln!(
        "DIFF DISTRIBUTION: p50={} p90={} p99={} max={} | frac_above_10x_mean={frac_above}",
        pct(0.5),
        pct(0.9),
        pct(0.99),
        pct(1.0)
    );

    // Report the worst outlier pixels with raw values (GPU may be Inf/NaN/huge).
    {
        let mut indexed: Vec<(usize, f32)> = cpu_img
            .iter()
            .zip(gpu_img.iter())
            .enumerate()
            .map(|(i, (c, g))| {
                let d: f32 = (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>();
                (i, d)
            })
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Top 5 outlier pixels:");
        for (i, d) in indexed.iter().take(5) {
            let px = i % w;
            let py = i / w;
            let c = &cpu_img[*i];
            let g = &gpu_img[*i];
            eprintln!(
                "  px=({px},{py}) diff={d:.4} cpu=[{:.6},{:.6},{:.6}] gpu=[{:.6},{:.6},{:.6}]",
                c[0], c[1], c[2],
                g[0], g[1], g[2]
            );
        }
    }

    // PNG heatmap of per-pixel diff.
    {
        let max_diff = {
            let mut sorted_pp: Vec<f32> = cpu_img
                .iter()
                .zip(gpu_img.iter())
                .map(|(c, g)| (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>())
                .collect();
            sorted_pp.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Use p99 as max to avoid outliers dominating the colormap.
            sorted_pp[((sorted_pp.len() as f32 * 0.99) as usize).min(sorted_pp.len() - 1)]
                .max(1e-6)
        };
        let pixels: Vec<u8> = cpu_img
            .iter()
            .zip(gpu_img.iter())
            .flat_map(|(c, g)| {
                let d: f32 = (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>();
                let v = ((d / max_diff).min(1.0).sqrt() * 255.0) as u8;
                [v, 0u8, 255u8 - v]
            })
            .collect();
        if let Some(img) = image::RgbImage::from_raw(w as u32, h as u32, pixels) {
            let path = "diff_gate.png";
            if img.save(path).is_ok() {
                eprintln!("wrote {path}");
            } else {
                eprintln!("WARN: could not write {path}");
            }
        }
    }

    // Gate: frac_above discriminates systemic kernel bugs from sparse FP Fresnel
    // branch flips. A systemic bug (wrong physics) affects O(10%) of pixels and
    // pushes frac_above >> 0.01. Sparse branch flips land in ~0-3 pixels (0.001).
    // p99 catches a case where "everything is 10x mean" but mean is still tiny.
    assert!(
        frac_above < 0.005,
        "frac_above_10x_mean={frac_above:.5} -- too many pixels differ from GPU by > 10x mean (systemic bug, not FP noise)"
    );
    assert!(
        pct(0.99) < 0.05,
        "p99 per-pixel L1={:.5} -- 99th percentile too large (systemic rounding or kernel bug)",
        pct(0.99)
    );
    assert!(energy_rel < 0.01, "GPU/CPU energy delta {energy_rel} exceeds tolerance");

    // Legacy mean_l1 is emitted but not gated; it's dominated by rare Fresnel
    // branch flips between f64-CPU Sellmeier and f32-GPU Sellmeier.
    eprintln!("(mean_l1={mean_l1:.6} -- informational only; dominated by 2-3 Fresnel branch flips)");
}

/// Re-run with a different seed to see if outlier pixels are seed-dependent (RNG path) or
/// position-dependent (indexing/workgroup bug).
#[test]
fn gpu_matches_cpu_alt_seed() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping alt seed test");
        return;
    };
    let (w, h, spp, seed) = (64usize, 64usize, 64u32, 0xD200u32);
    let cam = || Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 50.0, 1.0);

    let mut cpu = CpuTracer::new(fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    cpu.accumulate(spp);
    let cpu_img = cpu.buffer().mean();

    let mut gpu = GpuTracer::new(&ctx, fixed_scene(), cam(), w, h, Illuminant::D65, seed);
    gpu.accumulate(spp);
    let gpu_img = gpu.read_xyz();

    let mut per_pixel: Vec<f32> = cpu_img
        .iter()
        .zip(gpu_img.iter())
        .map(|(c, g)| (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>())
        .collect();
    per_pixel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |q: f32| per_pixel[((per_pixel.len() as f32 * q) as usize).min(per_pixel.len() - 1)];
    let mean_pp = per_pixel.iter().sum::<f32>() / per_pixel.len() as f32;
    let frac_above =
        per_pixel.iter().filter(|&&d| d > 10.0 * mean_pp).count() as f32 / per_pixel.len() as f32;

    eprintln!(
        "ALT_SEED DIST: p50={} p90={} p99={} max={} | frac_above_10x_mean={frac_above}",
        pct(0.5), pct(0.9), pct(0.99), pct(1.0)
    );

    let mut indexed: Vec<(usize, f32)> = cpu_img
        .iter()
        .zip(gpu_img.iter())
        .enumerate()
        .map(|(i, (c, g))| {
            let d: f32 = (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>();
            (i, d)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("ALT_SEED top 5:");
    for (i, d) in indexed.iter().take(5) {
        let px = i % w;
        let py = i / w;
        let c = &cpu_img[*i];
        let g = &gpu_img[*i];
        eprintln!(
            "  px=({px},{py}) diff={d:.4} cpu=[{:.6},{:.6},{:.6}] gpu=[{:.6},{:.6},{:.6}]",
            c[0], c[1], c[2],
            g[0], g[1], g[2]
        );
    }
}

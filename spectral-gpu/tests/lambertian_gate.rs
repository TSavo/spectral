/// Discriminating test: all-Lambertian scene vs GPU.
///
/// If the outliers from diff_gate (2 pixels, diff=614, 141) are caused by
/// dielectric-branch arithmetic (Fresnel/refract/dispersion-collapse),
/// they should VANISH with an all-Lambertian scene where the GPU and CPU
/// take the exact same code paths.
///
/// If outliers persist in an all-Lambertian scene, the divergence is in
/// shared code (primary ray, sphere intersection, hemisphere sampling,
/// table lookup) and constitutes a real bug.
use glam::Vec3;
use spectral_core::camera::Camera;
use spectral_core::cie::Illuminant;
use spectral_core::material::Material;
use spectral_core::scene::Scene;
use spectral_core::tracer::CpuTracer;
use spectral_core::Renderer;
use spectral_gpu::{GpuContext, GpuTracer};

fn lambertian_scene() -> Scene {
    let mut s = Scene::new();
    s.background = 1.0;
    s.add_sphere(Vec3::new(0.0, 0.0, -3.0), 1.0, Material::Lambertian { reflectance: 0.5 });
    s.add_sphere(Vec3::new(0.0, 0.0, -6.0), 2.0, Material::Lambertian { reflectance: 0.7 });
    s
}

#[test]
fn lambertian_only_gpu_matches_cpu() {
    let Some(ctx) = GpuContext::new() else {
        eprintln!("no GPU; skipping lambertian gate");
        return;
    };
    let (w, h, spp, seed) = (64usize, 64usize, 64u32, 0xD1FFu32);
    let cam = || Camera::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y, 50.0, 1.0);

    let mut cpu = CpuTracer::new(lambertian_scene(), cam(), w, h, Illuminant::D65, seed);
    cpu.accumulate(spp);
    let cpu_img = cpu.buffer().mean();

    let mut gpu = GpuTracer::new(&ctx, lambertian_scene(), cam(), w, h, Illuminant::D65, seed);
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
        "LAMBERTIAN GATE: p50={} p90={} p99={} max={} | frac_above_10x_mean={frac_above} | mean={mean_pp}",
        pct(0.5), pct(0.9), pct(0.99), pct(1.0)
    );

    // Top outliers.
    let mut indexed: Vec<(usize, f32)> = cpu_img
        .iter()
        .zip(gpu_img.iter())
        .enumerate()
        .map(|(i, (c, g))| (i, (0..3).map(|k| (c[k] - g[k]).abs()).sum::<f32>()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top 5 outliers:");
    for (i, d) in indexed.iter().take(5) {
        let px = i % w;
        let py = i / w;
        let c = &cpu_img[*i];
        let g = &gpu_img[*i];
        eprintln!(
            "  px=({px},{py}) diff={d:.4} cpu=[{:.6},{:.6},{:.6}] gpu=[{:.6},{:.6},{:.6}]",
            c[0], c[1], c[2], g[0], g[1], g[2]
        );
    }

    // If outliers vanish: bug is specifically in the dielectric code path.
    // If outliers persist: bug is in shared code (ray-sphere, hemisphere, table).
    assert!(
        frac_above < 0.002,
        "Lambertian-only frac_above={frac_above:.5} -- divergence in SHARED code path (not dielectric-only)"
    );
    assert!(
        pct(0.99) < 0.02,
        "Lambertian-only p99={:.5} -- shared-code divergence (too large for FP noise)",
        pct(0.99)
    );
}

use spectral_core::sellmeier::Glass;

// The same Sellmeier sum, forced through f32 (mirroring the WGSL), to bound the
// precision the GPU loses vs the f64-internal CPU reference.
fn n_f32(glass: Glass, lambda_nm: f32) -> f32 {
    let l_um = lambda_nm / 1000.0;
    let l2 = l_um * l_um;
    let (b, c): ([f32; 3], [f32; 3]) = match glass {
        Glass::Bk7 => (
            [1.039_612_2, 0.231_792_35, 1.010_469_4],
            [0.006_000_698_5, 0.020_017_914, 103.560_65],
        ),
        Glass::Sf11 => (
            [1.737_597, 0.313_747_35, 1.898_781_1],
            [0.013188707, 0.062_306_814, 155.236_3],
        ),
        Glass::Water => return 1.3238 + 0.00314 / l2, // Cauchy
    };
    let mut n2 = 1.0f32;
    for k in 0..3 {
        n2 += b[k] * l2 / (l2 - c[k]);
    }
    n2.sqrt()
}

#[test]
fn f32_sellmeier_error_is_bounded() {
    let mut max_abs = 0.0f32;
    for glass in [Glass::Bk7, Glass::Sf11, Glass::Water] {
        let mut nm = 380.0;
        while nm <= 730.0 {
            let d = (n_f32(glass, nm) - glass.n(nm)).abs();
            max_abs = max_abs.max(d);
            nm += 5.0;
        }
    }
    eprintln!("F32 BUDGET: max |n_f32 - n_f64| across BK7/SF11/Water over the band = {max_abs}");
    // The Sellmeier sum is well-conditioned (no cancellation), so f32 costs ~1e-5 in n.
    assert!(max_abs < 1e-4, "f32 Sellmeier error {max_abs} larger than expected; check conditioning");
}

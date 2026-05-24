//! CIE CMF + illuminant SPDs as a GPU-uploadable flat array, sampled from
//! spectral-core so the GPU uses the SAME data as the CPU oracle.

use spectral_core::cie::{Illuminant, Sensor};

/// 5nm grid, 380..=730 (71 samples). Layout (flat f32):
/// [xbar;71][ybar;71][zbar;71][d65;71][a;71].
pub const N_SAMPLES: usize = 71;
pub const LAMBDA0: f32 = 380.0;
pub const LAMBDA_STEP: f32 = 5.0;

pub fn sample_tables() -> Vec<f32> {
    let s = Sensor::new();
    let lam = |i: usize| LAMBDA0 + i as f32 * LAMBDA_STEP;
    let mut out = Vec::with_capacity(N_SAMPLES * 5);
    out.extend((0..N_SAMPLES).map(|i| s.cmf(lam(i)).0));
    out.extend((0..N_SAMPLES).map(|i| s.cmf(lam(i)).1));
    out.extend((0..N_SAMPLES).map(|i| s.cmf(lam(i)).2));
    out.extend((0..N_SAMPLES).map(|i| s.illuminant(Illuminant::D65, lam(i))));
    out.extend((0..N_SAMPLES).map(|i| s.illuminant(Illuminant::A, lam(i))));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tables_have_expected_length_and_finite_values() {
        let t = sample_tables();
        assert_eq!(t.len(), N_SAMPLES * 5);
        assert!(t.iter().all(|v| v.is_finite()));
        // ybar peaks near 555nm (index 35) at ~1.0.
        let ybar_peak = t[N_SAMPLES + 35];
        assert!((ybar_peak - 1.0).abs() < 0.05, "ybar(555) ~= 1, got {ybar_peak}");
    }
}

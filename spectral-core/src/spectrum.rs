//! Visible band, spectral sample carrier, and hero wavelength sampling.

pub const LAMBDA_MIN: f32 = 380.0;
pub const LAMBDA_MAX: f32 = 730.0;
pub const LAMBDA_RANGE: f32 = LAMBDA_MAX - LAMBDA_MIN;
pub const HERO_N: usize = 4;

/// A path's spectral state: HERO_N wavelengths traced coherently.
/// Lane 0 is the hero; lanes 1..HERO_N are stratified companions.
#[derive(Clone, Copy, Debug)]
pub struct SpectralSample {
    pub lambda: [f32; HERO_N],
    pub radiance: [f32; HERO_N],
    pub pdf: [f32; HERO_N],
}

impl SpectralSample {
    /// Build a hero comb from a uniform draw `u` in [0,1).
    /// Hero λ = LAMBDA_MIN + u*RANGE; companions add j*(RANGE/HERO_N), wrapped.
    pub fn from_hero_u(u: f32) -> Self {
        let step = LAMBDA_RANGE / HERO_N as f32;
        let mut lambda = [0.0; HERO_N];
        for (j, lambda_slot) in lambda.iter_mut().enumerate() {
            let off = (u * LAMBDA_RANGE + j as f32 * step).rem_euclid(LAMBDA_RANGE);
            *lambda_slot = LAMBDA_MIN + off;
        }
        Self {
            lambda,
            radiance: [0.0; HERO_N],
            pdf: [1.0 / LAMBDA_RANGE; HERO_N],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hero_comb_is_stratified_and_in_band() {
        let s = SpectralSample::from_hero_u(0.1);
        for &l in &s.lambda {
            assert!((LAMBDA_MIN..LAMBDA_MAX).contains(&l), "λ {l} out of band");
        }
        let step = LAMBDA_RANGE / HERO_N as f32;
        for j in 0..HERO_N {
            let a = s.lambda[j] - LAMBDA_MIN;
            let b = s.lambda[(j + 1) % HERO_N] - LAMBDA_MIN;
            let d = (b - a).rem_euclid(LAMBDA_RANGE);
            assert!((d - step).abs() < 1e-3, "lane spacing {d} != {step}");
        }
        for &p in &s.pdf {
            assert!((p - 1.0 / LAMBDA_RANGE).abs() < 1e-6);
        }
    }
}

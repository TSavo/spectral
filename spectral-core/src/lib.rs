//! spectral-core: CPU spectral path tracer.

pub mod optics;
pub mod rng;
pub mod sellmeier;
pub mod spectrum;

#[cfg(test)]
mod smoke {
    #[test]
    fn workspace_builds() {
        assert_eq!(2 + 2, 4);
    }
}

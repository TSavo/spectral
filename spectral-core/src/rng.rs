//! Deterministic per-sample RNG. Same key -> same sequence (CPU/GPU parity).

use rand::Rng;
use rand_pcg::Pcg32;

/// PCG32 seeded from a stream key + global seed. Deterministic and portable.
pub struct PathRng {
    inner: Pcg32,
}

impl PathRng {
    /// `key` is a per-path stream identifier (e.g. pixel index mixed with sample
    /// index); `seed` is the global render seed.
    pub fn new(key: u64, seed: u64) -> Self {
        Self { inner: Pcg32::new(seed, key) }
    }

    /// Uniform f32 in [0,1).
    pub fn next_f32(&mut self) -> f32 {
        let bits: u32 = self.inner.gen::<u32>() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_same_sequence() {
        let mut a = PathRng::new(7, 12345);
        let mut b = PathRng::new(7, 12345);
        for _ in 0..16 {
            assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits());
        }
    }

    #[test]
    fn different_key_different_sequence() {
        let mut a = PathRng::new(7, 12345);
        let mut b = PathRng::new(8, 12345);
        let da: Vec<u32> = (0..4).map(|_| a.next_f32().to_bits()).collect();
        let db: Vec<u32> = (0..4).map(|_| b.next_f32().to_bits()).collect();
        assert_ne!(da, db);
    }

    #[test]
    fn floats_in_unit_interval() {
        let mut r = PathRng::new(1, 2);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x), "{x} not in [0,1)");
        }
    }
}

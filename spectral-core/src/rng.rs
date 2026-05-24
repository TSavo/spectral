//! Deterministic per-sample RNG. A counter-based PCG hash over pure u32 ops, so
//! the identical sequence can be reproduced bit-for-bit in WGSL (the GPU mirror).
//! Same (key, seed) -> same sequence.

/// One-round PCG hash (Jarzynski & Olano 2020). Pure u32; portable to WGSL.
#[inline]
fn pcg32(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

pub struct PathRng {
    state: u32,
}

impl PathRng {
    /// `key` is a per-path stream identifier, `seed` the global render seed.
    pub fn new(key: u32, seed: u32) -> Self {
        let state = pcg32(key ^ pcg32(seed));
        PathRng { state }
    }

    /// Next raw u32. Advances the counter and hashes it.
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(1);
        pcg32(self.state)
    }

    /// Uniform f32 in [0,1) from 24 random bits (matches the WGSL mirror).
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }
}

/// Portable mix of two u32 into one stream key (for (pixel, sample) keys, etc.).
pub fn mix2(a: u32, b: u32) -> u32 {
    pcg32(a ^ pcg32(b))
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

    // FROZEN VALUES: the exact first u32 outputs. The WGSL mirror must reproduce
    // these bit-for-bit. Fill in the literals after running once (Step 3).
    #[test]
    fn frozen_first_draws() {
        let mut r = PathRng::new(0xABCD, 0x1234);
        let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
        assert_eq!(got, FROZEN_DRAWS);
    }

    const FROZEN_DRAWS: [u32; 4] = [0xa1dd5847, 0x13248a2d, 0x37278def, 0xde1d5b62];
}

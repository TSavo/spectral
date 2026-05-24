// Portable u32 PCG hash — must match spectral-core::rng::pcg32 bit-for-bit.
fn pcg32(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct PathRng { state: u32 }

fn rng_new(key: u32, seed: u32) -> PathRng {
    var r: PathRng;
    r.state = pcg32(key ^ pcg32(seed));
    return r;
}

fn rng_next_u32(r: ptr<function, PathRng>) -> u32 {
    (*r).state = (*r).state + 1u;
    return pcg32((*r).state);
}

fn rng_next_f32(r: ptr<function, PathRng>) -> f32 {
    let bits = rng_next_u32(r) >> 8u;
    return f32(bits) * (1.0 / 16777216.0);
}

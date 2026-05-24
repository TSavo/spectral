// VOL-1: Atomic fixed-point film + synthetic splat kernel.
//
// film: 3 * width * height atomic<u32> slots (interleaved XYZ).
//   channel c of pixel idx lives at film[3*idx + c].
// SCALE = 4096 (2^12). Radiance is non-negative; guard negatives before cast.
//
// This file is prepended with rng.wgsl (which defines pcg32 / PathRng / rng_new /
// rng_next_u32 / rng_next_f32) when compiled.

struct FilmParams {
    width:    u32,
    height:   u32,
    n_splats: u32,
    seed:     u32,
}

@group(0) @binding(0) var<uniform>            params: FilmParams;
@group(0) @binding(1) var<storage, read_write> film:   array<atomic<u32>>;

const SCALE: f32 = 4096.0;

// Splat a non-negative XYZ contribution into pixel index `idx`.
fn splat(idx: u32, xyz: vec3<f32>) {
    let x = max(xyz, vec3<f32>(0.0, 0.0, 0.0)) * SCALE;
    atomicAdd(&film[3u * idx + 0u], u32(x.x));
    atomicAdd(&film[3u * idx + 1u], u32(x.y));
    atomicAdd(&film[3u * idx + 2u], u32(x.z));
}

// Synthetic splat kernel.
// Each thread t: derive pixel and xyz purely from (t, seed) via PCG so the
// result is deterministic regardless of GPU scheduling order.
// MANY threads intentionally map to the SAME pixel to exercise atomic contention.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.n_splats {
        return;
    }

    // Initialize RNG from (thread index, global seed).
    var rng = rng_new(t, params.seed);

    // Derive pixel index from hash: uniform distribution over all pixels.
    let n_pixels = params.width * params.height;
    let px_raw = rng_next_u32(&rng);
    let idx    = px_raw % n_pixels;

    // Derive xyz in [0, 2) from three further draws.
    let rx = rng_next_f32(&rng) * 2.0;
    let ry = rng_next_f32(&rng) * 2.0;
    let rz = rng_next_f32(&rng) * 2.0;

    splat(idx, vec3<f32>(rx, ry, rz));
}

// VOL-2/3: Forward photon kernel — forward light tracing with dispersion and
// full single-scatter camera-connection weighting.
//
// One GPU thread per photon. Mirrors render_volumetric_scene() from volume.rs
// EXACTLY, including RNG consumption order (load-bearing for the VOL-4 diff gate).
//
// RNG order per photon i:
//   rng = rng_new(i, seed)
//   1. next_f32() -> beam.u coefficient
//   2. next_f32() -> beam.v coefficient
//   3. next_f32() -> lambda
//   4. per bounce: 4x next_f32() for seg sample distances (SEG_SAMPLES=4)
//   5. per hit:    scatter() consumes 1 Fresnel roulette draw (dielectric)
//
// VOL-3 splat weight (mirrors volume.rs render_volumetric_scene lines 129-149):
//   to = camO - p;  d = max(length(to), 1e-3)
//   if d < zbuffer[py*w+px] {           // DSOTM = all-INF zbuffer -> always pass
//     cos_theta = dot(ray.dir, to/d)    // scatter->camera vector, NOT cam forward
//     phase = phase_hg(g, cos_theta)
//     trans = exp(-sigma_t * d)
//     contrib = power*sigma_s*phase*trans/(d*d) * seg_len/SEG_SAMPLES
//     splat (xb,yb,zb)*contrib
//   }
//
// Prepended at build time with rng.wgsl (pcg32/PathRng/rng_new/rng_next_u32/rng_next_f32).

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PI:           f32 = 3.14159265358979;
const LAMBDA_MIN:   f32 = 380.0;
const LAMBDA_RANGE: f32 = 350.0;
const N_SAMPLES:    u32 = 71u;
const LAMBDA0:      f32 = 380.0;
const LAMBDA_STEP:  f32 = 5.0;
// Fixed-point accumulation scale = 2^23. VOL-3 weighted contributions are tiny
// (CMF · phase · trans / d² ~ 1e-5..1e-7 per splat); 2^12 (VOL-1) underflowed them
// to zero (diff gate L1 92%). Empirically L1 scales ~1/SCALE: 2^20 -> 1.97%,
// 2^23 -> 0.28%. 2^23 keeps the diff gate <1% with headroom while peak pixel u32
// (~a few units × 2^23 ≈ 2-4e7, ~4e8 at 16M photons) stays well below u32::MAX
// (4.29e9). MUST match SCALE in vol_photons.rs (load-bearing for the diff gate).
const SCALE:        f32 = 8388608.0;
const SEG_SAMPLES:  f32 = 4.0;

// ---------------------------------------------------------------------------
// GPU structs — identical byte layout to upload.rs / trace.wgsl
// ---------------------------------------------------------------------------

struct GpuPrimitive {
    center:      vec3<f32>,
    radius:      f32,
    plane_start: u32,
    plane_count: u32,
    kind:        u32,
    material:    u32,
}

struct GpuMaterial {
    reflectance: f32,
    glass:       u32,
    kind:        u32,
    _pad:        u32,
}

struct GpuPlane {
    normal: vec3<f32>,
    d:      f32,
}

// VolParams: must match VolParamsGpu in vol_photons.rs byte-for-byte.
struct VolParams {
    // scene
    n_primitives: u32,
    n_photons:    u32,
    seed:         u32,
    max_dist:     f32,
    // image
    width:        u32,
    height:       u32,
    debug_mode:   u32,  // 0 = normal splat only; 1 = also record per-photon state
    debug_count:  u32,  // photons with idx < debug_count get their state recorded
    // beam: corner, u, v, dir as vec4 (xyz+pad)
    beam_corner:  vec4<f32>,
    beam_u:       vec4<f32>,
    beam_v:       vec4<f32>,
    beam_dir:     vec4<f32>,
    // camera projection: precomputed from Camera::project
    cam_origin:   vec4<f32>,
    cam_u:        vec4<f32>,  // xyz = horizontal.normalize(), w = horizontal.length()
    cam_v:        vec4<f32>,  // xyz = vertical.normalize(),   w = vertical.length()
    cam_w:        vec4<f32>,  // xyz = origin - horiz/2 - vert/2 - lower_left (the "w" vector)
    // VOL-3 single-scatter weight parameters
    sigma_s:      f32,
    sigma_t:      f32,
    g:            f32,
    photon_base:  u32,        // chunking: photon i = photon_base + gid.x
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<uniform>             params:     VolParams;
@group(0) @binding(1) var<storage, read>       primitives: array<GpuPrimitive>;
@group(0) @binding(2) var<storage, read>       planes:     array<GpuPlane>;
@group(0) @binding(3) var<storage, read>       materials:  array<GpuMaterial>;
// tables layout: [xbar;71][ybar;71][zbar;71][d65;71][a;71]
@group(0) @binding(4) var<storage, read>       tables:     array<f32>;
// film: 3 * width * height atomic<u32> (XYZ interleaved, channel c at 3*idx+c)
@group(0) @binding(5) var<storage, read_write> film:       array<atomic<u32>>;

// DEBUG: per-photon recorded state for the GPU-vs-CPU parity gate.
// Mirrors cpu_simulate_photon's ray_states: one (origin,dir) pair pushed before
// the bounce loop, then one more after each SUCCESSFUL scatter (max 9 pairs).
// Layout per photon (must match DebugPhotonGpu in vol_photons.rs, 304 bytes):
//   num_states: u32   (= 1 + successful scatters)
//   lambda:     f32
//   _pad:       vec2<u32>
//   states:     array<vec4<f32>, 18>  // 9 pairs: (origin.xyz,_)(dir.xyz,_)
const MAX_PAIRS: u32 = 9u;
struct DebugPhoton {
    num_states: u32,
    lambda:     f32,
    _pad0:      u32,
    _pad1:      u32,
    states:     array<vec4<f32>, 18>,
}
@group(0) @binding(6) var<storage, read_write> debug_out: array<DebugPhoton>;

// zbuffer: per-pixel nearest-solid euclidean depth; a scatter point splats only
// if its camera distance d < zbuffer[py*w+px]. DSOTM uses an all-INF zbuffer
// (no occlusion -> always pass).
@group(0) @binding(7) var<storage, read>       zbuffer:   array<f32>;

// Record the (origin, dir) pair at slot `pair_idx` for debug photon `idx`.
fn debug_record(idx: u32, pair_idx: u32, origin: vec3<f32>, dir: vec3<f32>) {
    if pair_idx >= MAX_PAIRS { return; }
    debug_out[idx].states[2u * pair_idx + 0u] = vec4<f32>(origin, 0.0);
    debug_out[idx].states[2u * pair_idx + 1u] = vec4<f32>(dir,    0.0);
    debug_out[idx].num_states = pair_idx + 1u; // running count of pairs written
}

// ---------------------------------------------------------------------------
// Shared helpers — VALIDATED copies from trace.wgsl (do not rewrite)
// ---------------------------------------------------------------------------

fn cmf(nm: f32) -> vec3<f32> {
    let f = clamp((nm - LAMBDA0) / LAMBDA_STEP, 0.0, f32(N_SAMPLES - 1u));
    let i = u32(f);
    let frac = f - f32(i);
    let i1 = min(i + 1u, N_SAMPLES - 1u);
    let x0 = tables[0u * N_SAMPLES + i];
    let x1 = tables[0u * N_SAMPLES + i1];
    let y0 = tables[1u * N_SAMPLES + i];
    let y1 = tables[1u * N_SAMPLES + i1];
    let z0 = tables[2u * N_SAMPLES + i];
    let z1 = tables[2u * N_SAMPLES + i1];
    return vec3<f32>(
        x0 + (x1 - x0) * frac,
        y0 + (y1 - y0) * frac,
        z0 + (z1 - z0) * frac,
    );
}

fn sellmeier_n(glass: u32, nm: f32) -> f32 {
    let l_um = nm / 1000.0;
    let l2   = l_um * l_um;
    if glass == 2u {
        return 1.3238 + 0.00314 / l2;
    }
    var b0: f32; var b1: f32; var b2: f32;
    var c0: f32; var c1: f32; var c2: f32;
    if glass == 0u {
        b0 = 1.03961212; b1 = 0.231792344; b2 = 1.01046945;
        c0 = 0.00600069867; c1 = 0.0200179144; c2 = 103.560653;
    } else {
        b0 = 1.73759695; b1 = 0.313747346; b2 = 1.89878101;
        c0 = 0.013188707;  c1 = 0.0623068142; c2 = 155.23629;
    }
    let n2 = 1.0
           + b0 * l2 / (l2 - c0)
           + b1 * l2 / (l2 - c1)
           + b2 * l2 / (l2 - c2);
    return sqrt(n2);
}

fn reflect_dir(d: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return d - 2.0 * dot(d, n) * n;
}

fn refract_dir(d: vec3<f32>, n: vec3<f32>, n1: f32, n2: f32) -> vec4<f32> {
    let eta = n1 / n2;
    var nn = n;
    var cos_i = -dot(d, n);
    if cos_i < 0.0 {
        nn    = -n;
        cos_i = -cos_i;
    }
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0); // TIR
    }
    let cos_t = sqrt(k);
    let dir   = normalize(eta * d + (eta * cos_i - cos_t) * nn);
    return vec4<f32>(dir, 1.0);
}

fn fresnel_reflectance(cos_i_in: f32, n1: f32, n2: f32) -> f32 {
    let cos_i  = clamp(cos_i_in, 0.0, 1.0);
    let eta    = n1 / n2;
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 {
        return 1.0; // TIR
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    return 0.5 * (rs * rs + rp * rp);
}

fn cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, PathRng>) -> vec3<f32> {
    let u1  = rng_next_f32(rng);
    let u2  = rng_next_f32(rng);
    let r   = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x   = r * cos(phi);
    let y   = r * sin(phi);
    let z   = sqrt(max(1.0 - u1, 0.0));
    var a: vec3<f32>;
    if abs(normal.x) > 0.9 {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(normal, a));
    let b = cross(normal, t);
    return normalize(t * x + b * y + normal * z);
}

struct Hit {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,
    front_face: bool,
    valid:      bool,
}

fn sphere_hit(prim: GpuPrimitive, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> Hit {
    let oc     = ro - prim.center;
    let a      = dot(rd, rd);
    let half_b = dot(oc, rd);
    let c      = dot(oc, oc) - prim.radius * prim.radius;
    let disc   = half_b * half_b - a * c;
    var h: Hit;
    h.valid = false;
    if disc < 0.0 {
        return h;
    }
    let sqrt_d = sqrt(disc);
    var t = (-half_b - sqrt_d) / a;
    if t < t_min || t > t_max {
        t = (-half_b + sqrt_d) / a;
        if t < t_min || t > t_max {
            return h;
        }
    }
    let point   = ro + rd * t;
    let outward = (point - prim.center) / prim.radius;
    let ff      = dot(outward, rd) < 0.0;
    h.t          = t;
    h.point      = point;
    h.normal     = select(-outward, outward, ff);
    h.front_face = ff;
    h.valid      = true;
    return h;
}

fn solid_hit(prim: GpuPrimitive, ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> Hit {
    var t_enter:      f32       = -1e30;
    var t_exit:       f32       =  1e30;
    var enter_normal: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var exit_normal:  vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var h: Hit;
    h.valid = false;

    for (var i = prim.plane_start; i < prim.plane_start + prim.plane_count; i = i + 1u) {
        let pl    = planes[i];
        let denom = dot(pl.normal, rd);
        let dist  = dot(pl.normal, ro) + pl.d;
        if abs(denom) < 1e-8 {
            if dist > 0.0 {
                return h;
            }
            continue;
        }
        let t = -dist / denom;
        if denom < 0.0 {
            if t > t_enter {
                t_enter      = t;
                enter_normal = pl.normal;
            }
        } else {
            if t < t_exit {
                t_exit      = t;
                exit_normal = pl.normal;
            }
        }
    }

    if t_enter > t_exit {
        return h;
    }

    var t_hit:    f32;
    var hit_norm: vec3<f32>;
    if t_enter > t_min {
        t_hit    = t_enter;
        hit_norm = enter_normal;
    } else if t_exit > t_min {
        t_hit    = t_exit;
        hit_norm = exit_normal;
    } else {
        return h;
    }
    if t_hit > t_max {
        return h;
    }

    let ff = dot(hit_norm, rd) < 0.0;
    h.t          = t_hit;
    h.point      = ro + rd * t_hit;
    h.normal     = select(-hit_norm, hit_norm, ff);
    h.front_face = ff;
    h.valid      = true;
    return h;
}

struct SceneHit {
    hit:     Hit,
    mat_idx: u32,
    any:     bool,
}

fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>) -> SceneHit {
    var sh: SceneHit;
    sh.any     = false;
    sh.mat_idx = 0u;
    var closest = 1e30;

    for (var i = 0u; i < params.n_primitives; i = i + 1u) {
        let prim = primitives[i];
        var h: Hit;
        if prim.kind == 0u {
            h = sphere_hit(prim, ro, rd, 1e-3, closest);
        } else {
            h = solid_hit(prim, ro, rd, 1e-3, closest);
        }
        if h.valid {
            closest    = h.t;
            sh.hit     = h;
            sh.mat_idx = prim.material;
            sh.any     = true;
        }
    }
    return sh;
}

struct ScatterResult {
    dir:    vec3<f32>,
    weight: f32,
    valid:  bool,
}

fn scatter(mat: GpuMaterial, wo_in: vec3<f32>, hit: Hit, n_hero: f32, rng: ptr<function, PathRng>) -> ScatterResult {
    var sc: ScatterResult;
    sc.valid = true;
    if mat.kind == 0u {
        sc.dir    = cosine_hemisphere(hit.normal, rng);
        sc.weight = mat.reflectance;
    } else {
        let cos_i = abs(-dot(wo_in, hit.normal));
        var n1: f32;
        var n2: f32;
        if hit.front_face {
            n1 = 1.0;
            n2 = n_hero;
        } else {
            n1 = n_hero;
            n2 = 1.0;
        }
        let r = fresnel_reflectance(cos_i, n1, n2);
        if rng_next_f32(rng) < r {
            sc.dir    = reflect_dir(wo_in, hit.normal);
            sc.weight = 1.0;
        } else {
            let refr = refract_dir(wo_in, hit.normal, n1, n2);
            if refr.w > 0.5 {
                sc.dir    = refr.xyz;
                sc.weight = 1.0;
            } else {
                // TIR fallback: reflect (mirrors CPU mat.scatter which never returns None)
                sc.dir    = reflect_dir(wo_in, hit.normal);
                sc.weight = 1.0;
            }
        }
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Henyey-Greenstein phase function — mirrors volume.rs::phase_hg EXACTLY.
//   denom = max(1 + g² - 2·g·cos, 1e-6)^1.5   (clamp BEFORE pow)
//   p     = (1 - g²) / (4π · denom)
// ---------------------------------------------------------------------------
fn phase_hg(g: f32, cos_theta: f32) -> f32 {
    let g2    = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-6), 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

// ---------------------------------------------------------------------------
// Film splat
// ---------------------------------------------------------------------------
fn film_splat(idx: u32, xyz: vec3<f32>) {
    // Clamp to [0, 4e9] before the u32 cast: negatives are nonsense and a huge
    // float -> u32 is implementation-defined. 4e9 < u32::MAX is wrap-safe.
    let x = clamp(xyz * SCALE, vec3<f32>(0.0), vec3<f32>(4.0e9));
    atomicAdd(&film[3u * idx + 0u], u32(x.x));
    atomicAdd(&film[3u * idx + 1u], u32(x.y));
    atomicAdd(&film[3u * idx + 2u], u32(x.z));
}

// ---------------------------------------------------------------------------
// Camera::project — mirrors camera.rs Camera::project exactly.
//
// Precomputed in VolParams:
//   cam_u.xyz  = horizontal.normalize(),  cam_u.w  = horizontal.length() (vw)
//   cam_v.xyz  = vertical.normalize(),    cam_v.w  = vertical.length()   (vh)
//   cam_w.xyz  = origin - horizontal/2 - vertical/2 - lower_left
//              (this is the camera "back" vector; in front of camera => dot < 0)
//   cam_origin.xyz = camera origin
//
// Mirrors Camera::project:
//   let w = origin - horizontal/2 - vertical/2 - lower_left
//   let dir = p - origin
//   let c = dir.dot(w)   // in front => c < 0
//   if c >= -1e-6 { return None }
//   let s = 0.5 + dir.dot(u) / (-c * vw)
//   let t = 0.5 + dir.dot(v) / (-c * vh)
//
// Returns vec3(s, t, depth=-c) if in-frame, or (-1,-1,-1) if not.
// ---------------------------------------------------------------------------
fn camera_project(p: vec3<f32>) -> vec3<f32> {
    let u_hat = params.cam_u.xyz;
    let vw    = params.cam_u.w;
    let v_hat = params.cam_v.xyz;
    let vh    = params.cam_v.w;
    let w_vec = params.cam_w.xyz;
    let org   = params.cam_origin.xyz;

    let dir = p - org;
    let c   = dot(dir, w_vec); // in front of camera => c < 0
    if c >= -1e-6 {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let neg_c = -c; // positive depth
    let s = 0.5 + dot(dir, u_hat) / (neg_c * vw);
    let t = 0.5 + dot(dir, v_hat) / (neg_c * vh);
    if s < 0.0 || s >= 1.0 || t < 0.0 || t >= 1.0 {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    return vec3<f32>(s, t, neg_c);
}

// ---------------------------------------------------------------------------
// Forward photon kernel — mirrors render_volumetric_scene() RNG order EXACTLY
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Chunking: global photon index = photon_base + local thread index.
    let photon_idx = params.photon_base + gid.x;
    if photon_idx >= params.n_photons {
        return;
    }

    // Mirror: rng = PathRng::new(i, seed)
    var rng = rng_new(photon_idx, params.seed);

    // Mirror: origin = beam.corner + beam.u * rng.next_f32() + beam.v * rng.next_f32()
    let ru     = rng_next_f32(&rng);
    let rv     = rng_next_f32(&rng);
    let origin = params.beam_corner.xyz
               + params.beam_u.xyz * ru
               + params.beam_v.xyz * rv;

    // Mirror: lambda = LAMBDA_MIN + LAMBDA_RANGE * rng.next_f32()
    let lambda = LAMBDA_MIN + LAMBDA_RANGE * rng_next_f32(&rng);

    // Mirror: (xb, yb, zb) = sensor.cmf(lambda)
    let xyz_cmf = cmf(lambda);

    var ro    = origin;
    var rd    = normalize(params.beam_dir.xyz);
    var power = 1.0;

    let w = params.width;
    let h = params.height;

    let do_debug = params.debug_mode == 1u && photon_idx < params.debug_count;

    // CPU mirror: ray_states starts with (origin, beam_dir) pushed BEFORE the loop.
    var n_pairs = 1u;
    if do_debug {
        debug_out[photon_idx].lambda = lambda;
        debug_record(photon_idx, 0u, origin, rd); // pair 0 = (origin, beam_dir)
    }

    // Mirror: for _ in 0..8 bounce loop
    for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
        let sh      = scene_intersect(ro, rd);
        let seg_len = select(params.max_dist, sh.hit.t, sh.any);

        // Mirror render_volumetric_scene lines 129-149: for _ in 0..SEG_SAMPLES.
        // CONSUME the rng draw regardless of on-screen / occlusion (RNG order is
        // load-bearing for the diff gate).
        for (var k = 0u; k < 4u; k = k + 1u) {
            let dist = rng_next_f32(&rng) * seg_len;
            let p    = ro + rd * dist;
            let proj = camera_project(p);
            if proj.x >= 0.0 {
                let px  = min(u32(proj.x * f32(w)), w - 1u);
                let py  = min(u32((1.0 - proj.y) * f32(h)), h - 1u);
                let pidx = py * w + px;

                // Full single-scatter camera-connection weight.
                let to = params.cam_origin.xyz - p;
                let d  = max(length(to), 1e-3);
                if d < zbuffer[pidx] {
                    let cos_theta = dot(rd, to / d);
                    let phase     = phase_hg(params.g, cos_theta);
                    let trans     = exp(-params.sigma_t * d);
                    let contrib   = power * params.sigma_s * phase * trans / (d * d)
                                    * seg_len / SEG_SAMPLES;
                    film_splat(pidx, xyz_cmf * contrib);
                }
            }
        }

        if !sh.any {
            break; // escaped
        }

        // Mirror: n_hero = glass.n(lambda) for dielectric, else 1.0
        let mat   = materials[sh.mat_idx];
        var n_hero: f32;
        if mat.kind == 1u {
            n_hero = sellmeier_n(mat.glass, lambda);
        } else {
            n_hero = 1.0;
        }

        // Mirror: sc = mat.scatter(..., rng) — consumes 1 Fresnel roulette draw
        let sc = scatter(mat, rd, sh.hit, n_hero, &rng);
        if !sc.valid {
            break;
        }
        power *= sc.weight;
        ro     = sh.hit.point;
        rd     = sc.dir;

        // CPU mirror: after a successful scatter, ray_states.push((h.point, sc.dir)).
        if do_debug {
            debug_record(photon_idx, n_pairs, ro, rd);
            n_pairs = n_pairs + 1u;
        }
    }
}

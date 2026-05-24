// WGSL spectral path tracer — mirrors CpuTracer::sample_pixel exactly (f32).
// Prepended at load time with rng.wgsl which provides pcg32/PathRng/rng_new/
// rng_next_u32/rng_next_f32.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const PI:          f32 = 3.14159265358979;
const LAMBDA_MIN:  f32 = 380.0;
const LAMBDA_MAX:  f32 = 730.0;
const LAMBDA_RANGE: f32 = 350.0;
const HERO_N:      u32 = 4u;
const N_SAMPLES:   u32 = 71u;   // table entries per curve (5nm, 380..=730)
const LAMBDA0:     f32 = 380.0;
const LAMBDA_STEP: f32 = 5.0;

// ---------------------------------------------------------------------------
// GPU structs (must match upload.rs byte layout exactly)
// ---------------------------------------------------------------------------

// GpuPrimitive: kind 0=sphere, 1=convex solid.
struct GpuPrimitive {
    center:      vec3<f32>,  // bytes 0-11
    radius:      f32,        // bytes 12-15
    plane_start: u32,        // bytes 16-19
    plane_count: u32,        // bytes 20-23
    kind:        u32,        // bytes 24-27
    material:    u32,        // bytes 28-31 (index into materials[])
}

// GpuMaterial: kind 0=Lambertian, 1=Dielectric
struct GpuMaterial {
    reflectance: f32,   // bytes 0-3
    glass:       u32,   // bytes 4-7  (0=BK7, 1=SF11, 2=Water)
    kind:        u32,   // bytes 8-11
    _pad:        u32,   // bytes 12-15
}

// GpuPlane: normal points OUT of the solid; half-space { x : n·x + d <= 0 }
struct GpuPlane {
    normal: vec3<f32>,  // bytes 0-11
    d:      f32,        // bytes 12-15
}

// Render parameters uniform
struct Params {
    cam_origin:     vec4<f32>,
    cam_lower_left: vec4<f32>,
    cam_horizontal: vec4<f32>,
    cam_vertical:   vec4<f32>,
    width:          u32,
    height:         u32,
    spp:            u32,
    seed:           u32,
    n_primitives:   u32,
    illuminant:     u32,   // 0=D65, 1=A
    background:     f32,
    _pad:           u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<uniform>           params:     Params;
@group(0) @binding(1) var<storage, read>     primitives: array<GpuPrimitive>;
@group(0) @binding(2) var<storage, read>     planes:     array<GpuPlane>;
@group(0) @binding(3) var<storage, read>     materials:  array<GpuMaterial>;
// tables layout: [xbar;71][ybar;71][zbar;71][d65;71][a;71]
@group(0) @binding(4) var<storage, read>     tables:     array<f32>;
// accum: vec4 per pixel, xyz = summed XYZ, w = total sample count
@group(0) @binding(5) var<storage, read_write> accum:    array<vec4<f32>>;

// ---------------------------------------------------------------------------
// Table lookup helpers (linear interpolation, mirrors Sensor::sample)
// ---------------------------------------------------------------------------

// cmf(nm) -> xbar, ybar, zbar  (offsets 0, 71, 142 in tables[])
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

// illuminant_at(which=0→D65 offset 213, which=1→A offset 284) -> f32
fn illuminant_at(which: u32, nm: f32) -> f32 {
    let base = select(3u, 4u, which == 1u); // 3*71=213 for D65, 4*71=284 for A
    let f = clamp((nm - LAMBDA0) / LAMBDA_STEP, 0.0, f32(N_SAMPLES - 1u));
    let i = u32(f);
    let frac = f - f32(i);
    let i1 = min(i + 1u, N_SAMPLES - 1u);
    let v0 = tables[base * N_SAMPLES + i];
    let v1 = tables[base * N_SAMPLES + i1];
    return v0 + (v1 - v0) * frac;
}

// ---------------------------------------------------------------------------
// Sellmeier / Cauchy IOR  (mirrors Glass::n)
// glass: 0=BK7, 1=SF11, 2=Water
// ---------------------------------------------------------------------------
fn sellmeier_n(glass: u32, nm: f32) -> f32 {
    let l_um = nm / 1000.0;
    let l2   = l_um * l_um;
    if glass == 2u {
        // Water: Cauchy  n = 1.3238 + 0.00314 / l_um^2
        return 1.3238 + 0.00314 / l2;
    }
    // Sellmeier: n^2 = 1 + sum(Bk * l2 / (l2 - Ck))
    var b0: f32; var b1: f32; var b2: f32;
    var c0: f32; var c1: f32; var c2: f32;
    if glass == 0u {
        // BK7
        b0 = 1.03961212; b1 = 0.231792344; b2 = 1.01046945;
        c0 = 0.00600069867; c1 = 0.0200179144; c2 = 103.560653;
    } else {
        // SF11
        b0 = 1.73759695; b1 = 0.313747346; b2 = 1.89878101;
        c0 = 0.013188707;  c1 = 0.0623068142; c2 = 155.23629;
    }
    let n2 = 1.0
           + b0 * l2 / (l2 - c0)
           + b1 * l2 / (l2 - c1)
           + b2 * l2 / (l2 - c2);
    return sqrt(n2);
}

// ---------------------------------------------------------------------------
// Optics  (mirrors optics.rs)
// ---------------------------------------------------------------------------

// reflect(d, n) -> d - 2*(d.n)*n
fn reflect_dir(d: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return d - 2.0 * dot(d, n) * n;
}

// refract: returns vec4; xyz = direction, w = 1.0 if transmitted / 0.0 if TIR
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

// fresnel_reflectance(cos_i, n1, n2) — unpolarized Fresnel for dielectric
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

// ---------------------------------------------------------------------------
// Cosine-weighted hemisphere sampling (mirrors material.rs::cosine_hemisphere)
// ---------------------------------------------------------------------------
fn cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, PathRng>) -> vec3<f32> {
    let u1  = rng_next_f32(rng);
    let u2  = rng_next_f32(rng);
    let r   = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x   = r * cos(phi);
    let y   = r * sin(phi);
    let z   = sqrt(max(1.0 - u1, 0.0));
    // Build ONB around normal (same branch as Rust: abs(normal.x) > 0.9 -> Y else X)
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

// ---------------------------------------------------------------------------
// Intersection records
// ---------------------------------------------------------------------------
struct Hit {
    t:          f32,
    point:      vec3<f32>,
    normal:     vec3<f32>,
    front_face: bool,
    valid:      bool,
}

// sphere_hit: mirrors Sphere::intersect
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
    // Find the nearest root in (t_min, t_max) — strict, mirrors Sphere::intersect
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

// solid_hit: mirrors ConvexSolid::intersect (slab method)
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
        let dist  = dot(pl.normal, ro) + pl.d; // >0 means outside this half-space
        if abs(denom) < 1e-8 {
            // Parallel to this plane: if outside, ray misses entirely
            if dist > 0.0 {
                return h;
            }
            continue;
        }
        let t = -dist / denom;
        if denom < 0.0 {
            // Entering this half-space
            if t > t_enter {
                t_enter      = t;
                enter_normal = pl.normal;
            }
        } else {
            // Exiting this half-space
            if t < t_exit {
                t_exit      = t;
                exit_normal = pl.normal;
            }
        }
    }

    if t_enter > t_exit {
        return h; // ray misses the solid
    }

    // Pick first crossing strictly ahead of t_min; fall back to exit face
    // if origin is inside (handles interior refracted rays)
    var t_hit:    f32;
    var hit_norm: vec3<f32>;
    if t_enter > t_min {
        t_hit    = t_enter;
        hit_norm = enter_normal;
    } else if t_exit > t_min {
        t_hit    = t_exit;
        hit_norm = exit_normal;
    } else {
        return h; // whole solid is behind the ray
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

// ---------------------------------------------------------------------------
// Scene intersect — returns nearest hit + material index
// ---------------------------------------------------------------------------
struct SceneHit {
    hit:      Hit,
    mat_idx:  u32,
    any:      bool,
}

fn scene_intersect(ro: vec3<f32>, rd: vec3<f32>) -> SceneHit {
    var sh: SceneHit;
    sh.any      = false;
    sh.mat_idx  = 0u;
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

// ---------------------------------------------------------------------------
// Scatter result (mirrors Scatter { dir, weight, valid })
// ---------------------------------------------------------------------------
struct ScatterResult {
    dir:   vec3<f32>,
    weight: f32,
    valid: bool,  // false means absorb/terminate path (None in Rust)
}

// scatter: mirrors Material::scatter + cosine_hemisphere
fn scatter(mat: GpuMaterial, wo_in: vec3<f32>, hit: Hit, n_hero: f32, rng: ptr<function, PathRng>) -> ScatterResult {
    var sc: ScatterResult;
    sc.valid = true;
    if mat.kind == 0u {
        // Lambertian: cosine-weighted hemisphere; weight = reflectance
        sc.dir    = cosine_hemisphere(hit.normal, rng);
        sc.weight = mat.reflectance;
    } else {
        // Dielectric
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
            // Fresnel reflect
            sc.dir    = reflect_dir(wo_in, hit.normal);
            sc.weight = 1.0;
        } else {
            // Try to refract
            let refr = refract_dir(wo_in, hit.normal, n1, n2);
            if refr.w > 0.5 {
                // Transmitted
                sc.dir    = refr.xyz;
                sc.weight = 1.0;
            } else {
                // TIR fallback: reflect
                sc.dir    = reflect_dir(wo_in, hit.normal);
                sc.weight = 1.0;
            }
        }
    }
    return sc;
}

// ---------------------------------------------------------------------------
// Hero comb: mirrors SpectralSample::from_hero_u
// ---------------------------------------------------------------------------
fn hero_comb_lambda(u: f32, lane: u32) -> f32 {
    let step = LAMBDA_RANGE / f32(HERO_N);
    let off  = (u * LAMBDA_RANGE + f32(lane) * step) % LAMBDA_RANGE;
    return LAMBDA_MIN + off;
}

// pdf for all lanes = 1 / LAMBDA_RANGE (constant)
fn hero_pdf() -> f32 {
    return 1.0 / LAMBDA_RANGE;
}

// ---------------------------------------------------------------------------
// mix2: mirrors rng::mix2(a,b) = pcg32(a ^ pcg32(b))
// ---------------------------------------------------------------------------
fn mix2(a: u32, b: u32) -> u32 {
    return pcg32(a ^ pcg32(b));
}

// ---------------------------------------------------------------------------
// sample_pixel: mirrors CpuTracer::sample_pixel
// ---------------------------------------------------------------------------
fn sample_pixel(px: u32, py: u32, sample_idx: u32) -> vec3<f32> {
    let w = params.width;
    let h = params.height;

    // Stream key (mirrors mix2(pixel, sample_idx))
    let pixel = py * w + px;
    let key   = mix2(pixel, sample_idx);
    var rng   = rng_new(key, params.seed);

    // Jittered UV sample
    let s = (f32(px) + rng_next_f32(&rng)) / f32(w);
    let t = 1.0 - (f32(py) + rng_next_f32(&rng)) / f32(h);

    // Hero wavelength comb
    let u_hero = rng_next_f32(&rng);
    var lambda:     array<f32, 4>;
    var radiance:   array<f32, 4>;
    var throughput: array<f32, 4>;
    let pdf = hero_pdf();
    for (var k = 0u; k < HERO_N; k = k + 1u) {
        lambda[k]     = hero_comb_lambda(u_hero, k);
        radiance[k]   = 0.0;
        throughput[k] = 1.0;
    }

    // Primary ray: mirrors camera.primary_ray(s, t)
    var ro = params.cam_origin.xyz;
    var rd = normalize(
        params.cam_lower_left.xyz
        + s * params.cam_horizontal.xyz
        + t * params.cam_vertical.xyz
        - params.cam_origin.xyz
    );

    var valid_lanes = HERO_N;
    var bounce      = 0u;

    loop {
        if bounce >= 8u { break; }
        bounce = bounce + 1u;

        let sh = scene_intersect(ro, rd);

        if !sh.any {
            // Miss: accumulate background illuminant into radiance for each lane
            // mirrors: rad += throughput[k] * bg * illuminant(lam)
            let bg = params.background; // background_radiance(dir) — uniform, no horizon
            for (var k = 0u; k < HERO_N; k = k + 1u) {
                let sp = illuminant_at(params.illuminant, lambda[k]);
                radiance[k] = radiance[k] + throughput[k] * bg * sp;
            }
            break;
        }

        let hit = sh.hit;
        let mat = materials[sh.mat_idx];

        // n_hero: ior at hero wavelength (lane 0) for dielectrics, else 1.0
        var n_hero: f32;
        if mat.kind == 1u {
            n_hero = sellmeier_n(mat.glass, lambda[0]);
        } else {
            n_hero = 1.0;
        }

        // Scatter
        let sc = scatter(mat, rd, hit, n_hero, &rng);
        if !sc.valid {
            break;
        }

        // Apply throughput weight to all lanes
        for (var k = 0u; k < HERO_N; k = k + 1u) {
            throughput[k] = throughput[k] * sc.weight;
        }

        // Dispersion collapse: companions can't follow hero's refracted path.
        // Zero out companion lanes; keep only hero (lane 0). Mirrors Rust:
        //   if matches!(mat, Material::Dielectric{..}) && valid_lanes > 1 {
        //       throughput[1..].fill(0.0); valid_lanes = 1; }
        if mat.kind == 1u && valid_lanes > 1u {
            throughput[1] = 0.0;
            throughput[2] = 0.0;
            throughput[3] = 0.0;
            valid_lanes   = 1u;
        }

        // Advance ray
        ro = hit.point;
        rd = sc.dir;
    }

    // Convert radiance comb to XYZ.
    // wk = radiance[k] / pdf / valid_lanes  (mirrors tracer.rs exactly)
    var xyz = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < HERO_N; k = k + 1u) {
        let wk = radiance[k] / pdf / f32(valid_lanes);
        let c  = cmf(lambda[k]);
        xyz = xyz + c * wk;
    }
    return xyz;
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    var acc   = vec3<f32>(0.0, 0.0, 0.0);
    for (var s = 0u; s < params.spp; s = s + 1u) {
        acc = acc + sample_pixel(gid.x, gid.y, s);
    }
    accum[pixel] = accum[pixel] + vec4<f32>(acc, f32(params.spp));
}

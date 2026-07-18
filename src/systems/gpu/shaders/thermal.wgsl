// Day-night/ambient thermal diffusion — GPU port of ThermalDiffusion
// (src/energy/thermodynamics/diffusion.rs). Same real PDE: Fourier's law
// ∂T/∂t = α·∇²T, plus Newton cooling dT/dt = −k_c·(T−ambient). Dense
// grid_res² dispatch every substep, no active-block optimization (matches
// CPU's own unconditional-dense-grid behavior — real, bounded scope).
//
// 4 passes, mirroring CPU's ThermalDiffusion::apply stages exactly, each a
// separate dispatch because the Laplacian pass needs every cell's NORMALIZED
// temperature to be settled before it reads any neighbor — a genuine global
// barrier, not something a single fused pass can satisfy:
//   1. thermal_clear_main      — zero thermal_mass + thermal_work
//   2. thermal_p2g_main        — one thread per particle, scatter mass-
//                                 weighted temperature (fixed-point atomics,
//                                 same convention as p2g.wgsl)
//   3. thermal_normalize_laplacian_main — one thread per cell: normalize
//                                 (thermal_temp_old = work/mass, or ambient
//                                 if empty), 5-point Laplacian FD into
//                                 thermal_work, Newton cooling folded in
//   4. thermal_g2p_main        — one thread per particle, gather Δparticle
//                                 temperature = (T_new − T_old) at this
//                                 particle's position

struct Particle {
    x:                    vec2<f32>,
    v:                    vec2<f32>,
    velocity_gradient:    mat2x2<f32>,
    deformation_gradient: mat2x2<f32>,
    mass:                 f32,
    initial_volume:       f32,
    volume:               f32,
    density:              f32,
    material_id:          u32,
    plastic_volume_ratio: f32,
    hardening_scale:      f32,
    friction_hardening:   f32,
    log_volume_strain:    f32,
    temperature:          f32,
    user_tag:             u32,
    activation:           f32,
    activation_dir:       vec2<f32>,
    muscle_group_id:      u32,
    contact_group:        u32,
    sleeping:             u32,
    pinned:               u32,
    scalar_field:         f32,
    _pad:                 u32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}

struct ThermalParams {
    alpha:        f32,
    ambient:      f32,
    cooling_rate: f32,
    enabled:      u32,
}

const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
// Same fixed-point atomic convention as p2g.wgsl's MASS_ATOMIC_SCALE/MOM_ATOMIC_SCALE
// (duplicated per-shader-file, WGSL has no cross-file includes — same precedent).
const THERMAL_ATOMIC_SCALE: f32 = 100000.0;

@group(0) @binding(0) var<storage, read_write> particles:   array<Particle>;
@group(0) @binding(3) var<uniform>             step_params: StepParams;

@group(2) @binding(20) var<uniform>             thermal_params:    ThermalParams;
@group(2) @binding(21) var<storage, read_write> thermal_mass:      array<atomic<i32>>;
@group(2) @binding(22) var<storage, read_write> thermal_temp_old:  array<f32>;
@group(2) @binding(23) var<storage, read_write> thermal_work:      array<atomic<i32>>;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

fn thermal_atomic_addf(buf_is_mass: bool, idx: u32, val: f32) {
    let scaled = i32(round(val * THERMAL_ATOMIC_SCALE));
    if buf_is_mass {
        atomicAdd(&thermal_mass[idx], scaled);
    } else {
        atomicAdd(&thermal_work[idx], scaled);
    }
}

@compute @workgroup_size(64, 1, 1)
fn thermal_clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = step_params.grid_res;
    let n = res * res;
    let i = gid.x;
    if i >= n { return; }
    atomicStore(&thermal_mass[i], 0);
    atomicStore(&thermal_work[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn thermal_p2g_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if thermal_params.enabled == 0u { return; }
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    let p = particles[i];
    if !(dot(p.x, p.x) >= 0.0) { return; }

    let res = step_params.grid_res;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));
    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    for (var k: i32 = 0; k <= 2; k++) {
        let di = k - 1;
        wx[k] = bspline_w(f32(base.x + di) + CELL_CENTER_OFFSET - p.x.x);
        wy[k] = bspline_w(f32(base.y + di) + CELL_CENTER_OFFSET - p.x.y);
    }
    for (var ki: i32 = 0; ki <= 2; ki++) {
        let cx = base.x + ki - 1;
        if cx < 0 || cx >= i32(res) { continue; }
        for (var kj: i32 = 0; kj <= 2; kj++) {
            let cy = base.y + kj - 1;
            if cy < 0 || cy >= i32(res) { continue; }
            let w = wx[ki] * wy[kj];
            let idx = u32(cy) * res + u32(cx);
            let mw = w * p.mass;
            thermal_atomic_addf(true, idx, mw);
            thermal_atomic_addf(false, idx, mw * p.temperature);
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn thermal_normalize_laplacian_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if thermal_params.enabled == 0u { return; }
    let res = step_params.grid_res;
    let n = res * res;
    let i = gid.x;
    if i >= n { return; }

    // Normalize: thermal_temp_old[i] = T_old (mass-weighted average, or ambient if
    // this cell has no particle mass). thermal_work still holds the raw P2G scatter
    // at this point -- read via atomicLoad since it's declared atomic<i32>, then
    // immediately overwritten below with the post-Laplacian T_new.
    let mass_i = f32(atomicLoad(&thermal_mass[i])) / THERMAL_ATOMIC_SCALE;
    let raw_i = f32(atomicLoad(&thermal_work[i])) / THERMAL_ATOMIC_SCALE;
    let ambient = thermal_params.ambient;
    let t_old_i = select(ambient, raw_i / mass_i, mass_i > 1e-10);
    thermal_temp_old[i] = t_old_i;

    storageBarrier();

    // 5-point Laplacian FD (column-major idx = cy*res+cx, matching thermal_p2g_main's
    // own scatter convention above). Off-grid neighbors treated as ambient (Dirichlet).
    let cx = i32(i % res);
    let cy = i32(i / res);
    let t_xm = select(ambient, thermal_temp_old[u32(cy) * res + u32(cx - 1)], cx > 0);
    let t_xp = select(ambient, thermal_temp_old[u32(cy) * res + u32(cx + 1)], cx + 1 < i32(res));
    let t_ym = select(ambient, thermal_temp_old[u32(cy - 1) * res + u32(cx)], cy > 0);
    let t_yp = select(ambient, thermal_temp_old[u32(cy + 1) * res + u32(cx)], cy + 1 < i32(res));
    let laplacian = t_xm + t_xp + t_ym + t_yp - 4.0 * t_old_i;
    var t_new = t_old_i + thermal_params.alpha * step_params.dt * laplacian;

    // Newton cooling: T_new += -k_c*dt*(T-ambient), folded into the same pass.
    let decay = thermal_params.cooling_rate * step_params.dt;
    t_new += decay * (ambient - t_new);

    atomicStore(&thermal_work[i], i32(round(t_new * THERMAL_ATOMIC_SCALE)));
}

@compute @workgroup_size(64, 1, 1)
fn thermal_g2p_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if thermal_params.enabled == 0u { return; }
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    var p = particles[i];
    if !(dot(p.x, p.x) >= 0.0) { return; }

    let res = step_params.grid_res;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));
    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    for (var k: i32 = 0; k <= 2; k++) {
        let di = k - 1;
        wx[k] = bspline_w(f32(base.x + di) + CELL_CENTER_OFFSET - p.x.x);
        wy[k] = bspline_w(f32(base.y + di) + CELL_CENTER_OFFSET - p.x.y);
    }
    var delta = 0.0;
    var w_sum = 0.0;
    for (var ki: i32 = 0; ki <= 2; ki++) {
        let cx = base.x + ki - 1;
        if cx < 0 || cx >= i32(res) { continue; }
        for (var kj: i32 = 0; kj <= 2; kj++) {
            let cy = base.y + kj - 1;
            if cy < 0 || cy >= i32(res) { continue; }
            let w = wx[ki] * wy[kj];
            let idx = u32(cy) * res + u32(cx);
            let t_new = f32(atomicLoad(&thermal_work[idx])) / THERMAL_ATOMIC_SCALE;
            delta += w * (t_new - thermal_temp_old[idx]);
            w_sum += w;
        }
    }
    if w_sum > 1e-10 {
        p.temperature += delta / w_sum;
        particles[i] = p;
    }
}

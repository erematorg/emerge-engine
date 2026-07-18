// Resource regrowth — GPU port of ScalarDiffusionField's real logistic-growth source
// (src/energy/thermodynamics/scalar_field.rs). Same real PDE shape as thermal.wgsl
// (scatter -> normalize -> Laplacian+reaction -> gather), but the reaction term is
// logistic growth (Verhulst 1838, dφ/dt = r·φ·(1−φ/K)) instead of Newton cooling.
// Own separate buffers/group from thermal -- carries state in particle.scalar_field,
// NOT particle.temperature (real fix, 2026-07-17: both fields used to hijack
// temperature as their carrier, meaning two already-shipped GPU features literally
// could not run in the same scene together -- see Particle::scalar_field's own doc).
//
// 4 passes, same reasoning as thermal.wgsl for why they're separate dispatches (the
// Laplacian pass needs every cell's normalized φ settled first, a genuine global
// barrier):
//   1. resource_clear_main               — zero resource_mass + resource_work
//   2. resource_p2g_main                 — scatter mass-weighted φ (particle.scalar_field)
//   3. resource_normalize_laplacian_main — normalize, 5-point Laplacian, logistic growth
//   4. resource_g2p_main                 — gather Δφ back to particles

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

struct ResourceParams {
    diffusivity: f32,
    ambient:     f32,
    resource_r:  f32,
    resource_k:  f32,
    enabled:     u32,
}

const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const RESOURCE_ATOMIC_SCALE: f32 = 100000.0;

@group(0) @binding(0) var<storage, read_write> particles:   array<Particle>;
@group(0) @binding(3) var<uniform>             step_params: StepParams;

@group(3) @binding(24) var<uniform>             resource_params:  ResourceParams;
@group(3) @binding(25) var<storage, read_write> resource_mass:    array<atomic<i32>>;
@group(3) @binding(26) var<storage, read_write> resource_phi_old: array<f32>;
@group(3) @binding(27) var<storage, read_write> resource_work:    array<atomic<i32>>;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

fn resource_atomic_addf(buf_is_mass: bool, idx: u32, val: f32) {
    let scaled = i32(round(val * RESOURCE_ATOMIC_SCALE));
    if buf_is_mass {
        atomicAdd(&resource_mass[idx], scaled);
    } else {
        atomicAdd(&resource_work[idx], scaled);
    }
}

@compute @workgroup_size(64, 1, 1)
fn resource_clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = step_params.grid_res;
    let n = res * res;
    let i = gid.x;
    if i >= n { return; }
    atomicStore(&resource_mass[i], 0);
    atomicStore(&resource_work[i], 0);
}

@compute @workgroup_size(64, 1, 1)
fn resource_p2g_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if resource_params.enabled == 0u { return; }
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
            resource_atomic_addf(true, idx, mw);
            resource_atomic_addf(false, idx, mw * p.scalar_field);
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn resource_normalize_laplacian_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if resource_params.enabled == 0u { return; }
    let res = step_params.grid_res;
    let n = res * res;
    let i = gid.x;
    if i >= n { return; }

    let mass_i = f32(atomicLoad(&resource_mass[i])) / RESOURCE_ATOMIC_SCALE;
    let raw_i = f32(atomicLoad(&resource_work[i])) / RESOURCE_ATOMIC_SCALE;
    let ambient = resource_params.ambient;
    let phi_old_i = select(ambient, raw_i / mass_i, mass_i > 1e-10);
    resource_phi_old[i] = phi_old_i;

    storageBarrier();

    let cx = i32(i % res);
    let cy = i32(i / res);
    let p_xm = select(ambient, resource_phi_old[u32(cy) * res + u32(cx - 1)], cx > 0);
    let p_xp = select(ambient, resource_phi_old[u32(cy) * res + u32(cx + 1)], cx + 1 < i32(res));
    let p_ym = select(ambient, resource_phi_old[u32(cy - 1) * res + u32(cx)], cy > 0);
    let p_yp = select(ambient, resource_phi_old[u32(cy + 1) * res + u32(cx)], cy + 1 < i32(res));
    let laplacian = p_xm + p_xp + p_ym + p_yp - 4.0 * phi_old_i;
    var phi_new = phi_old_i + resource_params.diffusivity * step_params.dt * laplacian;

    // Real logistic growth: dφ/dt = r·φ·(1−φ/K) (Verhulst 1838).
    let k = max(resource_params.resource_k, 1e-6);
    let growth = resource_params.resource_r * phi_new * (1.0 - phi_new / k);
    phi_new += growth * step_params.dt;

    atomicStore(&resource_work[i], i32(round(phi_new * RESOURCE_ATOMIC_SCALE)));
}

@compute @workgroup_size(64, 1, 1)
fn resource_g2p_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if resource_params.enabled == 0u { return; }
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
            let phi_new = f32(atomicLoad(&resource_work[idx])) / RESOURCE_ATOMIC_SCALE;
            delta += w * (phi_new - resource_phi_old[idx]);
            w_sum += w;
        }
    }
    if w_sum > 1e-10 {
        p.scalar_field += delta / w_sum;
        particles[i] = p;
    }
}

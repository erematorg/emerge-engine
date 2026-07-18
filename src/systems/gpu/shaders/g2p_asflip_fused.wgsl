// ASFLIP (GPU port, Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the
// Material Point Method: A Scheme for Easier Separation and Less Dissipation") --
// fused G2P + particles_update, dispatched INSTEAD OF the ordinary g2p+particles_update
// pair for a substep, ONLY when SimConfig::asflip_blend > 0.0 (see SubstepGates::
// asflip_active in encode_substep.rs). Ordinary scenes (asflip disabled, the default)
// never touch this file at all -- zero cost, zero behavior change.
//
// WHY FUSED (not two separate ASFLIP-aware passes mirroring g2p/particles_update's own
// split): ASFLIP's position correction is adaptive -- it applies while two bodies are
// SEPARATING (gamma=1) but not while COMPRESSING (gamma=0), see CPU's
// `gather_grid_to_particles` in transfer.rs for the reference formula this ports. That
// means the STORED velocity (v_store, always gets the ASFLIP kick) and the POSITION
// velocity (v_position, only gets the kick while separating) can genuinely differ. CPU
// computes both in one function and uses them locally. GPU's existing architecture
// splits velocity-gather (g2p.wgsl) from F-update/position (particles_update.wgsl) into
// two separate dispatches for a real, different reason (Gao et al. 2018 sorted-access
// cache locality) -- meaning v_position would need to survive from the first dispatch
// to the second. `Particle` has exactly one spare u32 (4 bytes) left; a second stored
// Vec2 needs 8. Rather than grow the 128-byte struct (real risk: this exact struct had
// a confirmed, still-not-fully-understood GPU corruption bug from a MUCH smaller
// mid-struct field insertion earlier this project -- see Particle::scalar_field's own
// doc), this fused kernel keeps v_store/v_position as pure local variables that never
// need to leave one thread's registers, at the cost of one new pipeline variant.
//
// REAL, DISCLOSED MAINTENANCE COST: WGSL has no module/include system, so this file
// duplicates particles_update.wgsl's plasticity math (svd2 and all 6 material-model
// return-mapping functions) verbatim rather than sharing it. If that file's plasticity
// logic ever changes, THIS file needs the identical change applied by hand -- there is
// no compiler enforcement keeping them in sync. Flagged honestly, not hidden; a real
// WGSL-side module system would remove this risk but doesn't exist in this toolchain.

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

struct Cell {
    momentum: vec2<f32>, // after grid_update this holds velocity, not momentum
    mass:     f32,
    _pad:     f32,
}

struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32, // Snow: xi; VonMises: yield_stress (union layout)
    compression_limit:       f32, // Snow: theta_c; DP: dilatancy psi; Bingham: yield_stress
    stretch_limit:           f32,
    rest_density:            f32,
    eos_stiffness:           f32,
    eos_power:               f32,
    dynamic_viscosity:       f32,
    volume_ratio_min:        f32,
    volume_ratio_max:        f32,
    dp_h0:                   f32,
    dp_h1:                   f32,
    dp_h2:                   f32,
    dp_h3:                   f32,
    active_stress_coeff:     f32,
    hardening_modulus:       f32,
    thermal_viscosity_coeff: f32,
    thermal_expansion:       f32,
    pressure_floor:          f32,
    bulk_viscosity:          f32,
    surface_tension_coeff:   f32,
    cohesion_coeff:              f32,
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
    contact_active:     u32,
}

// ASFLIP (GPU port) -- see GpuAsflipParams' own Rust doc.
struct AsflipParams {
    blend:   f32,
    enabled: u32,
    _pad0:   u32,
    _pad1:   u32,
}

const MAX_MATERIALS:    u32 = {{MAX_MATERIALS}}u;
const NUM_FLOOR:        f32 = 1e-6;
const NUM_FLOOR_TIGHT:  f32 = 1e-10;

const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;

@group(0) @binding(0) var<storage, read_write> particles:            array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid:                 array<Cell>;
@group(0) @binding(2) var<uniform>              materials:            array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>              step_params:          StepParams;
@group(0) @binding(5) var<storage, read_write> sorted_particle_ids:  array<u32>;
// Multi-field contact (GPU port) -- resolved velocities from resolve_contact_main, same
// fallback-to-total-velocity convention g2p.wgsl already relies on.
@group(1) @binding(17) var<storage, read_write> resolved_grip_v: array<vec2<f32>>;
@group(1) @binding(18) var<storage, read_write> resolved_rest_v: array<vec2<f32>>;
// ASFLIP -- shares group 3 with resource regrowth, see pipeline.rs's module doc comment.
@group(3) @binding(28) var<uniform>             asflip_params:    AsflipParams;
@group(3) @binding(29) var<storage, read_write> asflip_snapshot:  array<vec2<f32>>;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

// ── 2D SVD (verbatim copy of particles_update.wgsl's own -- see this file's top doc
// comment for why duplication, not sharing, is unavoidable here) ──────────────────────

struct Svd2 { u: mat2x2<f32>, s: vec2<f32>, v: mat2x2<f32> }

fn svd2(f: mat2x2<f32>) -> Svd2 {
    let ftf    = transpose(f) * f;
    let a      = ftf[0][0];
    let b      = ftf[1][0];
    let d      = ftf[1][1];
    let half_tr = 0.5 * (a + d);
    let disc   = sqrt(max(0.25 * (a - d) * (a - d) + b * b, 0.0));
    let lam1   = half_tr + disc;
    let lam2   = max(half_tr - disc, 0.0);
    let sig1   = sqrt(lam1);
    let sig2   = sqrt(lam2);

    var v0: vec2<f32>;
    var v1: vec2<f32>;
    if abs(b) < NUM_FLOOR {
        if a >= d { v0 = vec2<f32>(1.0, 0.0); } else { v0 = vec2<f32>(0.0, 1.0); }
        v1 = vec2<f32>(-v0.y, v0.x);
    } else {
        let ex = lam1 - d;
        let n  = sqrt(ex * ex + b * b);
        v0 = vec2<f32>(ex / n, b / n);
        v1 = vec2<f32>(-v0.y, v0.x);
    }
    let v_mat = mat2x2<f32>(v0, v1);

    let fv0 = f * v0;
    let fv1 = f * v1;
    var u0: vec2<f32> = select(vec2<f32>(1.0, 0.0),    fv0 / sig1, sig1 > NUM_FLOOR);
    var u1: vec2<f32> = select(vec2<f32>(-u0.y, u0.x), fv1 / sig2, sig2 > NUM_FLOOR);

    let det_f = f[0][0] * f[1][1] - f[1][0] * f[0][1];
    var s = vec2<f32>(sig1, sig2);
    if det_f < 0.0 { s.y = -s.y; u1 = -u1; }

    return Svd2(mat2x2<f32>(u0, u1), s, v_mat);
}

struct SnowReturn { f_e: mat2x2<f32>, jp: f32, h: f32 }

fn snow_plasticity(f_trial: mat2x2<f32>, jp_in: f32, mat: MaterialParams) -> SnowReturn {
    let svd = svd2(f_trial);
    let lo  = 1.0 - mat.compression_limit;
    let hi  = 1.0 + mat.stretch_limit;
    let sc  = clamp(svd.s, vec2<f32>(lo), vec2<f32>(hi));
    let jp_new = clamp(
        jp_in * (svd.s.x * svd.s.y) / max(sc.x * sc.y, NUM_FLOOR_TIGHT),
        mat.volume_ratio_min, mat.volume_ratio_max,
    );
    let h_new = clamp(exp(mat.hardening_exponent * (1.0 - jp_new)), 0.1, 7.0);
    let diag  = mat2x2<f32>(vec2<f32>(sc.x, 0.0), vec2<f32>(0.0, sc.y));
    return SnowReturn(svd.u * diag * transpose(svd.v), jp_new, h_new);
}

fn dp_alpha(q: f32, mat: MaterialParams) -> f32 {
    let phi = mat.dp_h0 + (mat.dp_h1 * q - mat.dp_h3) * exp(-mat.dp_h2 * q);
    let s   = sin(phi);
    return sqrt(2.0 / 3.0) * (2.0 * s) / max(3.0 - s, NUM_FLOOR_TIGHT);
}

struct DpReturn { sigma: vec2<f32>, dq: f32, log_vol_delta: f32 }

fn dp_plasticity(sigma_in: vec2<f32>, log_volume_strain: f32, q: f32, mat: MaterialParams) -> DpReturn {
    let sigma = max(sigma_in, vec2<f32>(NUM_FLOOR_TIGHT));
    let eps   = log(sigma) + vec2<f32>(log_volume_strain * 0.5);
    let tr    = eps.x + eps.y;
    let dev   = eps - vec2<f32>(tr * 0.5);
    let dn    = length(dev);

    if dn < NUM_FLOOR_TIGHT || tr > 0.0 {
        let prev_det = sigma.x * sigma.y;
        return DpReturn(vec2<f32>(1.0), dn, log(max(prev_det, NUM_FLOOR_TIGHT * NUM_FLOOR_TIGHT)));
    }

    let ratio = (mat.lambda + mat.mu) / max(mat.mu, NUM_FLOOR_TIGHT);
    let alpha = dp_alpha(q, mat);
    let cohesion_term = mat.stretch_limit / (2.0 * max(mat.mu, NUM_FLOOR_TIGHT));
    let gamma = dn + ratio * tr * alpha - cohesion_term;

    if gamma <= 0.0 {
        return DpReturn(sigma, 0.0, 0.0);
    }

    let h_eps     = eps - gamma * (dev / dn);
    let sigma_new = vec2<f32>(exp(h_eps.x), exp(h_eps.y));
    let prev_det  = sigma.x * sigma.y;
    let new_det   = sigma_new.x * sigma_new.y;
    var lvg_delta = log(max(prev_det, NUM_FLOOR_TIGHT * NUM_FLOOR_TIGHT))
                  - log(max(new_det,  NUM_FLOOR_TIGHT * NUM_FLOOR_TIGHT));

    if mat.compression_limit > 0.0 {
        lvg_delta += sin(mat.compression_limit) * gamma;
    }

    return DpReturn(sigma_new, gamma, lvg_delta);
}

fn hencky_from_stress(tau: vec2<f32>, lambda: f32, mu: f32) -> vec2<f32> {
    let a  = 2.0 * mu + lambda;
    let det = a * a - lambda * lambda;
    let x  = (a * tau.x - lambda * tau.y) / det;
    let y  = (a * tau.y - lambda * tau.x) / det;
    return vec2<f32>(x, y);
}

struct RankineReturn { f_e: mat2x2<f32>, damage_delta: f32 }

fn rankine_plasticity(f_trial: mat2x2<f32>, damage: f32, mat: MaterialParams) -> RankineReturn {
    let svd    = svd2(f_trial);
    let sigma  = max(svd.s, vec2<f32>(NUM_FLOOR_TIGHT));
    let eps    = log(sigma);
    let a      = 2.0 * mat.mu + mat.lambda;
    let tau    = vec2<f32>(a * eps.x + mat.lambda * eps.y, mat.lambda * eps.x + a * eps.y);
    let t_eff  = mat.hardening_exponent * exp(-mat.hardening_modulus * damage);

    let t1 = tau.x > t_eff;
    let t2 = tau.y > t_eff;
    if !t1 && !t2 {
        return RankineReturn(f_trial, 0.0);
    }

    let tau_proj = vec2<f32>(select(tau.x, t_eff, t1), select(tau.y, t_eff, t2));
    let eps_proj = hencky_from_stress(tau_proj, mat.lambda, mat.mu);
    let eps_prev = eps;
    let ddmg     = length(eps_prev - eps_proj);
    let sigma_new = exp(eps_proj);
    let diag = mat2x2<f32>(vec2<f32>(sigma_new.x, 0.0), vec2<f32>(0.0, sigma_new.y));
    return RankineReturn(svd.u * diag * transpose(svd.v), ddmg);
}

struct MuIReturn { f_e: mat2x2<f32>, mu_i: f32 }

fn sand_mui_plasticity(f_trial: mat2x2<f32>, mu_i_in: f32, mat: MaterialParams, dt: f32) -> MuIReturn {
    let svd   = svd2(f_trial);
    let sigma = max(svd.s, vec2<f32>(NUM_FLOOR_TIGHT));
    let eps   = log(sigma);
    let tr    = eps.x + eps.y;
    let k_2d  = mat.lambda + mat.mu;
    let p_tri = -k_2d * tr;

    if p_tri <= 0.0 {
        let diag = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        return MuIReturn(svd.u * diag * transpose(svd.v), mat.dp_h0);
    }

    let dev    = eps - vec2<f32>(tr * 0.5);
    let dn     = length(dev);
    let SQRT2: f32 = 1.41421356;
    let q_tri  = SQRT2 * mat.mu * dn;
    let q_yld  = mat.dp_h0 * p_tri;

    if q_tri <= q_yld || dn < NUM_FLOOR_TIGHT {
        let diag = mat2x2<f32>(vec2<f32>(sigma.x, 0.0), vec2<f32>(0.0, sigma.y));
        return MuIReturn(svd.u * diag * transpose(svd.v), mat.dp_h0);
    }

    let delta_q   = q_tri - q_yld;
    let sqrt_p    = sqrt(p_tri);
    let a_coef    = mat.mu * dt;
    let b_coef    = p_tri * (mat.dp_h1 - mat.dp_h0) + a_coef * mat.dp_h2 * sqrt_p - delta_q;
    let c_coef    = -delta_q * mat.dp_h2 * sqrt_p;
    let disc      = b_coef * b_coef - 4.0 * a_coef * c_coef;
    let gd        = max((-b_coef + sqrt(max(disc, 0.0))) / (2.0 * a_coef), 0.0);

    let mu_i = select(mat.dp_h0,
        mat.dp_h0 + (mat.dp_h1 - mat.dp_h0) / (mat.dp_h2 * sqrt_p / gd + 1.0),
        gd > NUM_FLOOR_TIGHT);

    let delta_gamma = gd * dt;
    let n_hat       = dev / dn;
    let eps_new     = eps - n_hat * (delta_gamma / SQRT2);
    let sigma_new   = exp(eps_new);
    let diag = mat2x2<f32>(vec2<f32>(sigma_new.x, 0.0), vec2<f32>(0.0, sigma_new.y));
    return MuIReturn(svd.u * diag * transpose(svd.v), mu_i);
}

struct VmReturn { f_e: mat2x2<f32>, dkappa: f32 }

fn vm_plasticity(f_trial: mat2x2<f32>, kappa: f32, mat: MaterialParams) -> VmReturn {
    let svd     = svd2(f_trial);
    let sigma   = max(svd.s, vec2<f32>(NUM_FLOOR_TIGHT));
    let eps     = log(sigma);
    let tr      = eps.x + eps.y;
    let dev     = eps - vec2<f32>(tr * 0.5);
    let dn      = length(dev);
    let yield_s = mat.hardening_exponent + mat.hardening_modulus * kappa;
    let elastic_dev = 2.0 * mat.mu * dn;

    if elastic_dev <= yield_s || dn < NUM_FLOOR_TIGHT {
        return VmReturn(f_trial, 0.0);
    }

    let denom     = 2.0 * mat.mu + mat.hardening_modulus;
    let gamma     = select((elastic_dev - yield_s) / denom, 0.0, denom < NUM_FLOOR_TIGHT);
    let eps_proj  = dev * (yield_s / elastic_dev) + vec2<f32>(tr * 0.5);
    let sigma_new = exp(eps_proj);
    let diag      = mat2x2<f32>(vec2<f32>(sigma_new.x, 0.0), vec2<f32>(0.0, sigma_new.y));
    return VmReturn(svd.u * diag * transpose(svd.v), gamma);
}

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// Workgroup size MUST match WG_PARTICLES (= 64) in src/gpu/mod.rs, same as g2p/particles_update.
@compute @workgroup_size(64, 1, 1)
fn g2p_asflip_fused_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= step_params.particle_count { return; }
    // Sorted access -- matches particles_update.wgsl's own convention (cache-coherent
    // for this shader's own particle-memory access pattern); no correctness dependence
    // on iteration order since every particle is still processed exactly once, by
    // exactly one thread, writing only to its own particles[p_idx] slot.
    let p_idx = sorted_particle_ids[gid.x];

    var p = particles[p_idx];
    let res = step_params.grid_res;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));

    // ── Sleeping / wake (verbatim g2p_main logic) ─────────────────────────────────
    if p.sleeping != 0u {
        var should_wake = false;
        for (var di: i32 = -1; di <= 1; di++) {
            for (var dj: i32 = -1; dj <= 1; dj++) {
                let cx = base.x + di;
                let cy = base.y + dj;
                if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) { continue; }
                let cell = grid[u32(cy) * res + u32(cx)];
                if cell.mass > NUM_FLOOR && length(cell.momentum) > step_params.sleep_threshold {
                    should_wake = true;
                }
            }
        }
        if !should_wake { return; }
        p.sleeping = 0u;
    }

    var new_v       = vec2<f32>(0.0);
    var b_col0      = vec2<f32>(0.0);
    var b_col1      = vec2<f32>(0.0);
    var new_density = 0.0;

    if p.pinned != 0u {
        // Dirichlet/kinematic anchor -- matches g2p_main's pinned branch exactly
        // (v=0, velocity_gradient=0), but density/volume are deliberately left
        // untouched (g2p_main never writes them for a pinned particle either).
        new_v = vec2<f32>(0.0);
        // b_col0/b_col1 stay zero -> C below is zero, matching velocity_gradient=0.
    } else {
        let is_grip = p.contact_group != 0u;
        let contact_active = step_params.contact_active != 0u;

        for (var di: i32 = -1; di <= 1; di++) {
            for (var dj: i32 = -1; dj <= 1; dj++) {
                let cx = base.x + di;
                let cy = base.y + dj;
                if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) { continue; }

                let cell_dist = vec2<f32>(f32(cx), f32(cy)) + vec2<f32>(CELL_CENTER_OFFSET) - p.x;
                let w = bspline_w(cell_dist.x) * bspline_w(cell_dist.y);

                let node_idx = u32(cy) * res + u32(cx);
                let cell   = grid[node_idx];
                let cell_v = select(
                    cell.momentum,
                    select(resolved_rest_v[node_idx], resolved_grip_v[node_idx], is_grip),
                    contact_active,
                );

                new_v       += w * cell_v;
                b_col0      += w * cell_v * cell_dist.x;
                b_col1      += w * cell_v * cell_dist.y;
                new_density += w * cell.mass;
            }
        }
    }

    // ── ASFLIP correction (Fei et al. 2021) ───────────────────────────────────────
    // v_old = this particle's OWN velocity before this substep's G2P -- captured before
    // any overwrite. gamma: 0 while locally compressing (tr(b)<0), 1 while separating --
    // the real "easier separation, less dissipation" adaptivity, computed from the RAW
    // affine matrix b (not the scaled C), matching CPU's own gather_grid_to_particles
    // exactly (transfer.rs: `trace_b = b.x_axis.x + b.y_axis.y`).
    let v_old = p.v;
    var v_store = new_v;
    var v_position = new_v;
    if asflip_params.enabled != 0u && p.pinned == 0u {
        var old_v = vec2<f32>(0.0);
        for (var di: i32 = -1; di <= 1; di++) {
            for (var dj: i32 = -1; dj <= 1; dj++) {
                let cx = base.x + di;
                let cy = base.y + dj;
                if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) { continue; }
                let cell_dist = vec2<f32>(f32(cx), f32(cy)) + vec2<f32>(CELL_CENTER_OFFSET) - p.x;
                let w = bspline_w(cell_dist.x) * bspline_w(cell_dist.y);
                old_v += w * asflip_snapshot[u32(cy) * res + u32(cx)];
            }
        }
        let diff_vel = v_old - old_v;
        let trace_b  = b_col0.x + b_col1.y;
        let gamma    = select(1.0, 0.0, trace_b < 0.0);
        v_store    = new_v + asflip_params.blend * diff_vel;
        v_position = new_v + gamma * asflip_params.blend * diff_vel;
    }

    // Velocity clamp -- mirrors CPU's own G2P clamp (applied to the FINAL v_store, with
    // v_position scaled by the same ratio to stay mutually consistent; identical to
    // v_store when ASFLIP disabled, since v_store==v_position==new_v then). NaN-safe
    // select, same defensive pattern g2p_main's own (pre-ASFLIP) clamp already used.
    let spd = length(v_store);
    if !(spd <= step_params.vel_limit) {
        let inv = step_params.vel_limit / spd;
        let scale = select(inv, 0.0, !(inv > 0.0));
        v_store *= scale;
        v_position *= scale;
    }

    let c = mat2x2<f32>(b_col0, b_col1) * step_params.kernel_d_inverse;

    if p.pinned != 0u {
        p.v = vec2<f32>(0.0);
        p.velocity_gradient = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        particles[p_idx] = p;
        return;
    }

    let density = max(new_density, NUM_FLOOR);
    let volume  = p.mass / density;
    p.v = v_store;
    p.velocity_gradient = c;
    p.density = density;
    p.volume = volume;

    // ── particles_update_main's body, verbatim except reading local `p`/`c` instead of
    // re-reading the buffer (g2p already wrote what particles_update would have re-read)
    // and using `v_position` (not `p.v`) for the position line. ─────────────────────────

    let mat = materials[p.material_id];
    let dt  = step_params.dt;
    let bt  = f32(step_params.boundary_thickness);
    let identity = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));

    // GPU state projection -- mirrors project_particle_state_to_admissible in solver/mod.rs.
    let fres = f32(res);
    let half = fres * 0.5;
    if !(p.x.x >= 0.0 && p.x.x < fres) { p.x.x = half; }
    if !(p.x.y >= 0.0 && p.x.y < fres) { p.x.y = half; }
    if !(dot(p.v, p.v) >= 0.0) { p.v = vec2<f32>(0.0); }
    let cg = dot(p.velocity_gradient[0], p.velocity_gradient[0])
           + dot(p.velocity_gradient[1], p.velocity_gradient[1]);
    if !(cg >= 0.0) { p.velocity_gradient = mat2x2<f32>(); }
    if !(det2(p.deformation_gradient) > 0.0) { p.deformation_gradient = identity; }
    if !(p.plastic_volume_ratio > 0.0)         { p.plastic_volume_ratio = 1.0; }
    if !(p.hardening_scale > 0.0)              { p.hardening_scale = 1.0; }
    if !(abs(p.friction_hardening) < 3.4e+38)  { p.friction_hardening = 0.0; }
    if !(abs(p.log_volume_strain)  < 3.4e+38)  { p.log_volume_strain  = 0.0; }

    var new_F = (identity + dt * p.velocity_gradient) * p.deformation_gradient;

    if mat.model == 4u && mat.compression_limit > 0.0 {
        let sr = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    } else if mat.model == 5u {
        let svd    = svd2(new_F);
        let dp_res = dp_plasticity(svd.s, p.log_volume_strain, p.friction_hardening, mat);
        var dp_sigma = abs(dp_res.sigma);
        let dp_j = dp_sigma.x * dp_sigma.y;
        if dp_j < mat.volume_ratio_min {
            dp_sigma *= sqrt(mat.volume_ratio_min / max(dp_j, NUM_FLOOR_TIGHT));
        }
        let diag = mat2x2<f32>(vec2<f32>(dp_sigma.x, 0.0), vec2<f32>(0.0, dp_sigma.y));
        new_F = svd.u * diag * transpose(svd.v);
        let q_max = 5.0 / max(mat.dp_h2, NUM_FLOOR_TIGHT);
        p.friction_hardening = min(p.friction_hardening + dp_res.dq, q_max);
        p.log_volume_strain  += dp_res.log_vol_delta;
    } else if mat.model == 6u {
        let vm_res = vm_plasticity(new_F, p.friction_hardening, mat);
        new_F = vm_res.f_e;
        p.friction_hardening += vm_res.dkappa;
    } else if mat.model == 7u {
        let rk_res = rankine_plasticity(new_F, p.friction_hardening, mat);
        new_F = rk_res.f_e;
        p.friction_hardening += rk_res.damage_delta;
    } else if mat.model == 8u {
        let mi_res = sand_mui_plasticity(new_F, p.friction_hardening, mat, dt);
        new_F = mi_res.f_e;
        p.friction_hardening = mi_res.mu_i;
    } else if mat.model == 11u && mat.compression_limit > 0.0 {
        let sr = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    }

    const FLUID_J_MIN: f32 = 0.5;
    if mat.model == 1u {
        let fluid_j_max = select(2.0, mat.volume_ratio_max, mat.volume_ratio_max > 1.0);
        var J_fluid = det2(new_F);
        if !(J_fluid > 0.0) { J_fluid = 1.0; }
        J_fluid = clamp(J_fluid, FLUID_J_MIN, fluid_j_max);
        let sqrtJ = sqrt(J_fluid);
        new_F = mat2x2<f32>(vec2<f32>(sqrtJ, 0.0), vec2<f32>(0.0, sqrtJ));

        if mat.dp_h0 > 0.0 {
            let damp = 1.0 - clamp(mat.dp_h0 * dt, 0.0, 0.5);
            p.v *= damp;
            v_position *= damp;
        }
    }

    let J_trial = det2(new_F);
    if !(J_trial > 0.0) {
        if mat.model == 1u {
            new_F = identity;
        } else {
            let svd_r = svd2(new_F);
            let sc    = vec2<f32>(svd_r.s.x, abs(svd_r.s.y) + NUM_FLOOR);
            let diag  = mat2x2<f32>(vec2<f32>(sc.x, 0.0), vec2<f32>(0.0, sc.y));
            new_F     = svd_r.u * diag * transpose(svd_r.v);
        }
    }

    let J_elastic = det2(new_F);
    if J_elastic > 0.0 && mat.model == 9u {
        let j_lo = max(mat.volume_ratio_min, 0.01);
        let j_hi = 2.5;
        if J_elastic < j_lo {
            new_F = new_F * sqrt(j_lo / J_elastic);
        } else if J_elastic > j_hi {
            new_F = new_F * sqrt(j_hi / J_elastic);
        }
    }

    // Light damping for plasticity models -- applied to BOTH v_store (p.v, persists for
    // next g2p gather) and v_position (this substep's own position advance), same
    // damping ratio, keeping ASFLIP's two velocities mutually consistent with how the
    // single-velocity (non-ASFLIP) path already behaves under this damping.
    if mat.model != 0u && mat.model != 1u && mat.model != 2u && mat.model != 3u && mat.model != 9u {
        p.v *= 0.999;
        v_position *= 0.999;
    }

    // Position: x = x + v_position * dt -- v_position, NOT p.v, is ASFLIP's real point:
    // while separating (gamma=1) this equals p.v exactly; while compressing (gamma=0)
    // this omits the ASFLIP kick, avoiding injecting extra positional noise into two
    // bodies already pressing together (Fei et al. 2021's own "easier separation"
    // framing). Identical to p.v*dt when ASFLIP disabled or gamma=1.
    var new_x = p.x + v_position * dt;

    let lo = max(0.0, bt - 1.0);
    let hi = f32(res) - bt;
    new_x  = clamp(new_x, vec2<f32>(lo), vec2<f32>(hi));

    particles[p_idx].x                    = new_x;
    particles[p_idx].v                    = p.v;
    particles[p_idx].velocity_gradient    = c;
    particles[p_idx].deformation_gradient = new_F;
    particles[p_idx].density              = p.density;
    particles[p_idx].volume               = p.volume;
    particles[p_idx].plastic_volume_ratio = p.plastic_volume_ratio;
    particles[p_idx].hardening_scale      = p.hardening_scale;
    particles[p_idx].friction_hardening   = p.friction_hardening;
    particles[p_idx].log_volume_strain    = p.log_volume_strain;
    particles[p_idx].sleeping             = p.sleeping;
}

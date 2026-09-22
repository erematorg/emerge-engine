// g2p_update -- fused per-particle substep tail: G2P gather, then F update,
// plasticity, volume/density, position, boundary, then force fields and
// sleep/wake scoring. MLS-MPM, Hu et al. 2018 SIGGRAPH §4.
//
// One thread per particle (sorted access via sorted_particle_ids). The gather
// (`g2p_gather.inc.wgsl`) and force-field (`force_fields_apply.inc.wgsl`) code is
// appended to this source at pipeline creation. These used to be three separate
// dispatches (g2p -> particles_update -> force_fields), each loading and storing
// the full 128-byte particle; on a small integrated GPU the fixed cost per
// dispatch alone was ~15-20us, a large share of a ~0.35ms substep.
//
// Update steps (in `update_particle`):
//   1. F = (I + dt·C) · F_old          (C = velocity_gradient from the gather)
//   2. Snow plasticity (model 4): 2D SVD → clamp σ → update Jp/h → reconstruct F_e
//   3. DP plasticity  (model 5): 2D SVD → log-strain return mapping → update q/log_volume_strain
//   4. Von Mises      (model 6): 2D SVD → J2 yield check → deviatoric return mapping
//   5. J = det(F), volume = initial_volume × J, density = mass / volume
//   6. Position: x = x + v · dt
//   7. Boundary clamp: slip -- clamp x within [bt, grid_res−bt)

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
    internal_pressure:    f32,  // total 128 bytes
}

struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32, // Snow: ξ; VonMises: yield_stress (union layout)
    compression_limit:       f32, // Snow: θ_c; DP: dilatancy ψ; Bingham: yield_stress
    stretch_limit:           f32,
    rest_density:            f32,
    eos_stiffness:           f32,
    eos_power:               f32,
    dynamic_viscosity:       f32,
    volume_ratio_min:        f32, // Snow/DP: Jp lower bound
    volume_ratio_max:        f32, // Snow/DP: Jp upper bound; Fluid: J_MAX for free surface
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
    // GPU/CPU parity fix (2026-08-15) -- see Rust MaterialParams's own doc.
    // 1u = this material derives density/volume analytically from its own
    // clamped F (matches CPU's `owns_deformation_volume_state()`), 0u =
    // unused by this material.
    owns_deformation_volume_state: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    contact_friction:   f32,
    grid_cell_size:     f32,
    contact_active:     u32,
    cfl_coefficient:    f32,
    material_cfl_coefficient: f32,
    min_dt:             f32,
    dt_cap:             f32,
}

const MAX_MATERIALS:    u32 = {{MAX_MATERIALS}}u;
const NUM_FLOOR:        f32 = 1e-6;
const NUM_FLOOR_TIGHT:  f32 = 1e-10;

// Bit m set = material model m (`MaterialParams::model`) is in this scene's registry
// (`MaterialRegistry::model_mask`), fixed at pipeline creation and re-specialized when the
// registry gains a new model. Branches for absent models compile away: measured on the
// fluid-only dam break, the never-taken plasticity/SVD code cost ~20us of a ~70us
// g2p_update dispatch (its register footprint lowers occupancy even when unused).
override MODELS_PRESENT: u32 = 0xFFFFFFFFu;
fn has_model(model: u32, m: u32) -> bool {
    return (MODELS_PRESENT & (1u << m)) != 0u && model == m;
}

@group(0) @binding(0) var<storage, read_write> particles:            array<Particle>;
@group(0) @binding(2) var<uniform>             materials:            array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>             step_params:          StepParams;

// This substep's timestep and velocity cap, decided on the GPU at the end of the previous
// substep -- see `adaptive_cfl.wgsl`. `substep_dt()`/`vel_limit` are the CPU's
// frame-start values and are NOT authoritative any more (the GPU may only tighten them).
@group(2) @binding(37) var<storage, read_write> adaptive_dt: array<atomic<u32>, 5>;

// Cached per invocation: this is an atomic storage load, and reading it at every use
// site cost ~40% of a substep (measured: 0.29 -> 0.41ms per substep on the dam break).
var<private> substep_dt_cache: f32 = -1.0;

fn substep_dt() -> f32 {
    if substep_dt_cache < 0.0 {
        substep_dt_cache = bitcast<f32>(atomicLoad(&adaptive_dt[0]));
    }
    return substep_dt_cache;
}

fn substep_vel_limit(dt: f32) -> f32 {
    return step_params.grid_cell_size / max(dt, 1.0e-12);
}

@group(0) @binding(5) var<storage, read_write> sorted_particle_ids:  array<u32>;

// ── 2D SVD ────────────────────────────────────────────────────────────────────
// Analytical thin SVD F = U · diag(s) · Vᵀ for a 2×2 matrix, s.x ≥ |s.y|.
// Sign convention: det(U) = +1 (proper rotation). Matches sparkl/Stomakhin.

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

// ── Snow plasticity ───────────────────────────────────────────────────────────
// Clamp singular values to [1−θ_c, 1+θ_s], accumulate Jp and hardening h.

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
    // h clamped [0.1, 7.0]. At E=5000: h=7 → c_P≈99 cells/s → sub_dt≈0.005 → ~20 substeps.
    // Upper bound is CFL-driven (sparkl uses 50 substeps, no clamp; we cap at 20).
    let h_new = clamp(exp(mat.hardening_exponent * (1.0 - jp_new)), 0.1, 7.0);
    let diag  = mat2x2<f32>(vec2<f32>(sc.x, 0.0), vec2<f32>(0.0, sc.y));
    return SnowReturn(svd.u * diag * transpose(svd.v), jp_new, h_new);
}

// ── Drucker-Prager plasticity ─────────────────────────────────────────────────
// Log-strain (Hencky) return mapping. Klar et al. 2016.
// Hardening formula:  φ(q) = h0 + (h1·q − h3)·exp(−h2·q)
//                     α(q) = √(2/3) · 2·sin(φ) / (3 − sin(φ))
// Yield function:     γ = |dev_ε| + (λ+2µ)/(2µ) · tr_ε · α
// Return mapping:     ε_proj = ε − γ · dev_ε/|dev_ε|,  σ_proj = exp(ε_proj)
// Volume correction:  log_volume_strain += ln(det_old) − ln(det_new)
// Reynolds dilatancy: log_volume_strain += sin(ψ)·γ  [mat.compression_limit = ψ]

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
        // dq = dn only (not length(eps)) -- log_volume_strain offset must not contribute.
        // length(eps) causes unbounded q growth in settled sand. Mirrors sand.rs:130.
        let prev_det = sigma.x * sigma.y;
        return DpReturn(vec2<f32>(1.0), dn, log(max(prev_det, NUM_FLOOR_TIGHT * NUM_FLOOR_TIGHT)));
    }

    // Single-pass: alpha evaluated once from the pre-step q, matching
    // wgsparkl::models::drucker_prager::project_deformation_gradient exactly (the
    // reference GPU implementation of Klar et al. 2016 -- no self-consistency corrector).
    // stretch_limit repurposed for DP: cohesion floor, see sand.rs's `cohesion` doc
    // comment -- NOT real "sand cohesion" (dry sand is ~0), a continuum-MPM-resolution
    // regularization for thin flowing layers, calibrated against the Lajeunesse 2004
    // runout benchmark.
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

    // Reynolds dilatancy: mat.compression_limit repurposed as dilatancy angle ψ for DP.
    if mat.compression_limit > 0.0 {
        lvg_delta += sin(mat.compression_limit) * gamma;
    }

    return DpReturn(sigma_new, gamma, lvg_delta);
}

// ── Rankine plasticity ────────────────────────────────────────────────────────
// Tensile cutoff with exponential damage softening (Wolper et al. 2019).
// Uses Hencky strains, same corotated-elastic basis as VonMises and DP.
// tensile_strength → mat.hardening_exponent, softening_rate → mat.hardening_modulus.
// Damage accumulates in friction_hardening.

// Convert Kirchhoff principal stresses τ to Hencky strain eigenvectors ε.
// Inverse of: τ = (2µ+λ)·εᵢ + λ·εⱼ.
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

// ── SandMuI plasticity ────────────────────────────────────────────────────────
// µ(I)-rheology Drucker-Prager (Blatny 2022). Rate-dependent friction.
// dp_h0=mu_static, dp_h1=mu_dynamic, dp_h2=inertial_q.
// mu_i stored in friction_hardening.

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

// ── Von Mises plasticity ──────────────────────────────────────────────────────
// J2 plasticity with linear isotropic hardening in Hencky strain space.
// yield_stress stored in mat.hardening_exponent (union layout).

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
    // Real, disclosed regression fix (2026-09-02, external review, same
    // bug as the CPU path's own von_mises.rs -- see that file's own doc
    // for the worked counterexample): must project onto the yield surface
    // AFTER this step's own hardening increment, not the pre-hardening
    // trial-state limit `yield_s`.
    let new_yield_s = yield_s + mat.hardening_modulus * gamma;
    let eps_proj  = dev * (new_yield_s / elastic_dev) + vec2<f32>(tr * 0.5);
    let sigma_new = exp(eps_proj);
    let diag      = mat2x2<f32>(vec2<f32>(sigma_new.x, 0.0), vec2<f32>(0.0, sigma_new.y));
    return VmReturn(svd.u * diag * transpose(svd.v), gamma);
}

// ─────────────────────────────────────────────────────────────────────────────

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// Trace of a 2x2 matrix, taking the matrix BY VALUE -- deliberately the same
// shape as `det2` above. Measured on this project's AMD Vulkan target
// (driver 25.10.2): indexing an element straight out of a mat2x2 member of a
// function-local struct copy (`p.velocity_gradient[1][1]`, and equally
// `[1].y`) returned the element of COLUMN 0 (`[0][1]`) -- the column index was
// lost. Passing the whole matrix into a function first, as `det2` always
// has, reads correctly. Confirmed by writing the shader's own computed value
// back to the particle and comparing it against the read-back matrix.
fn trace2(m: mat2x2<f32>) -> f32 {
    return m[0][0] + m[1][1];
}

// Squared Frobenius norm, matrix taken BY VALUE -- same reason as `trace2`
// (particles_update.wgsl): indexing a column straight out of a mat2x2 member
// of a function-local struct copy lost the column index on the AMD Vulkan
// target, so `p.velocity_gradient[1]` silently re-read column 0 and this
// NaN guard never saw column 1.
fn frob2_sq(m: mat2x2<f32>) -> f32 {
    return dot(m[0], m[0]) + dot(m[1], m[1]);
}

// Exact 2D exp(A), matching CPU `deformation_increment_exp`. Restricted at
// the call site to models explicitly migrated from forward Euler so plastic
// projections can be validated one family at a time.
fn deformation_increment_exp(a: mat2x2<f32>) -> mat2x2<f32> {
    let identity = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let half_trace = 0.5 * (a[0][0] + a[1][1]);
    let half_difference = 0.5 * (a[0][0] - a[1][1]);
    let delta_sq = half_difference * half_difference + a[1][0] * a[0][1];
    var even_factor = 0.0;
    var odd_factor = 0.0;
    if abs(delta_sq) < 1e-8 {
        let x2 = delta_sq * delta_sq;
        even_factor = 1.0 + 0.5 * delta_sq + x2 / 24.0;
        odd_factor = 1.0 + delta_sq / 6.0 + x2 / 120.0;
    } else if delta_sq > 0.0 {
        let delta = sqrt(delta_sq);
        even_factor = cosh(delta);
        odd_factor = sinh(delta) / delta;
    } else {
        let omega = sqrt(-delta_sq);
        even_factor = cos(omega);
        odd_factor = sin(omega) / omega;
    }
    let traceless = a - half_trace * identity;
    return exp(half_trace) * (even_factor * identity + odd_factor * traceless);
}

// Workgroup size MUST match WG_PARTICLES (= 64) in src/gpu/mod.rs.
@compute @workgroup_size(64, 1, 1)
fn g2p_update_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // Per-workgroup minimum of the next substep's CFL bound, flushed to the global one
    // by a single thread: 2912 particles all doing `atomicMin` on the same global slot
    // serialise on it (measured: +36us per substep, over half of this pass's own cost).
    if lid == 0u {
        atomicStore(&wg_cfl_min, F32_MAX_BITS);
    }
    workgroupBarrier();
    // No early return before the closing barrier below; spare substeps (dt == 0, see
    // adaptive_cfl.wgsl) and out-of-range invocations just do nothing.
    let runs = gid.x < step_params.particle_count && substep_dt() > 0.0;
    if runs {
        g2p_update_particle(sorted_particle_ids[gid.x]);
    }
    workgroupBarrier();
    if lid == 0u {
        let m = atomicLoad(&wg_cfl_min);
        if m != F32_MAX_BITS {
            atomicMin(&adaptive_dt[2], m);
        }
    }
}

const F32_MAX_BITS: u32 = 0x7F7FFFFFu;
var<workgroup> wg_cfl_min: atomic<u32>;

fn g2p_update_particle(p_idx: u32) {
    var p = particles[p_idx];
    g2p_gather(p_idx, &p);
    // Still-sleeping particles (didn't wake in the gather) are frozen -- skip state
    // projection, F update, every plasticity branch, and position integration
    // entirely. Particles that woke have sleeping=0u by this point and get the full
    // update, same as CPU (a newly-woken particle gets a real update the same substep
    // it wakes).
    if p.sleeping == 0u {
        update_particle(p_idx, &p);
    }
    // Force fields see the advanced position and the updated velocity, exactly as
    // the former standalone pass did when it ran after this one. Only `v` and
    // `sleeping` can change there, so only those are written back.
    if apply_force_fields(&p) {
        particles[p_idx].v = p.v;
        particles[p_idx].sleeping = p.sleeping;
    }
    if p.sleeping == 0u {
        accumulate_cfl_bound(p, materials[p.material_id]);
    }
}

// This particle's own CFL bound for the NEXT substep, folded into the shared minimum
// (`adaptive_cfl.wgsl` turns it into the next dt). Mirrors the terms of CPU's
// `choose_substep_dt` that actually CHANGE within a frame -- particle speed, the
// deformation-gradient ODE bound, and, for a material that owns its volume state (a
// strict fluid), its compression-dependent acoustic bound, the shock-viscosity
// correction and the Sun/Shinar/Schroeder 2020 single-particle bound. Terms not ported
// here simply leave the CPU's frame-start cap in charge, which is what the GPU used for
// all of them before; nothing here can raise dt above that cap.
// Exponentiation by squaring for a whole-numbered exponent -- same reason (and same
// shape) as p2g.wgsl's `fast_pow`: WGSL's `pow` always goes through exp2/log2, which is
// both slower and less precise near 1.0. Falls back for non-integer exponents.
fn cfl_fast_pow(x: f32, e: f32) -> f32 {
    if abs(fract(e)) > 1.0e-6 || abs(e) >= 32.0 {
        return pow(x, e);
    }
    var exp_i = i32(round(abs(e)));
    var base = x;
    var result = 1.0;
    while exp_i > 0 {
        if (exp_i & 1) == 1 {
            result = result * base;
        }
        base = base * base;
        exp_i = exp_i >> 1;
    }
    if e < 0.0 {
        return 1.0 / result;
    }
    return result;
}

fn accumulate_cfl_bound(p: Particle, mat: MaterialParams) {
    let dx = step_params.grid_cell_size;
    var bound = 3.4e38;

    let speed = length(p.v);
    if speed > NUM_FLOOR {
        bound = min(bound, step_params.cfl_coefficient * dx / speed);
    }
    let grad_norm = sqrt(frob2_sq(p.velocity_gradient));
    if grad_norm > NUM_FLOOR {
        bound = min(bound, min(step_params.cfl_coefficient, 0.5) / grad_norm);
    }

    if mat.owns_deformation_volume_state == 1u && mat.eos_stiffness > 0.0 {
        let rho0 = max(mat.rest_density, NUM_FLOOR);
        let j = max(det2(p.deformation_gradient), NUM_FLOOR);
        let c2_rest = mat.eos_stiffness * mat.eos_power / rho0;
        // Acoustic bound at THIS particle's own compression (c grows as it compresses).
        let c2 = c2_rest * cfl_fast_pow(1.0 / j, mat.eos_power - 1.0);
        if c2 > NUM_FLOOR_TIGHT {
            bound = min(bound, step_params.material_cfl_coefficient * dx / sqrt(c2));
        }
        // Von Neumann-Richtmyer shock viscosity (Bate et al. 1995 combined c_eff).
        if grad_norm > NUM_FLOOR && c2_rest > NUM_FLOOR_TIGHT {
            let c0_quadratic = (mat.eos_power + 1.0) * 0.25;
            let c_eff = sqrt(c2_rest) + 2.0 * c0_quadratic * dx * grad_norm;
            if c_eff > NUM_FLOOR_TIGHT {
                bound = min(bound, step_params.material_cfl_coefficient * dx / c_eff);
            }
        }
        // Single-particle instability (Sun, Shinar & Schroeder 2020), quadratic spline.
        if c2_rest > NUM_FLOOR_TIGHT {
            let kd_lambda = 6.0 * 2.0 * rho0 * c2_rest;
            var single = dx * sqrt(rho0 * (j + 1.0) / (j * j * j * kd_lambda));
            if j <= 1.0 {
                single = (dx / (2.0 - j)) * sqrt(2.0 * rho0 / kd_lambda);
            }
            if single > 0.0 {
                bound = min(bound, single);
            }
        }
    }

    if bound > 0.0 && bound < 3.4e38 {
        atomicMin(&wg_cfl_min, bitcast<u32>(bound));
    }
}

// F update, plasticity, state projection and position advance for one awake
// particle whose gathered v/C are already in `*pp`. Writes its results to
// `particles[p_idx]` and mirrors the advanced position and velocity into `*pp`.
fn update_particle(p_idx: u32, pp: ptr<function, Particle>) {
    var p = *pp;
    let mat = materials[p.material_id];
    let dt  = substep_dt();
    let res = step_params.grid_res;
    let bt  = f32(step_params.boundary_thickness);

    // Identity matrix -- used both in state projection and F update below.
    let I = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));

    // ── GPU state projection ──────────────────────────────────────────────────
    // Mirrors project_particle_state_to_admissible in solver/mod.rs.
    // The CPU runs this every substep; the GPU was missing it entirely.
    // !(x >= 0 && x < res) catches NaN (NaN >= 0 = false → !false = true) and out-of-bounds.
    // !(dot >= 0) catches NaN/Inf in vector fields (NaN/Inf squared = NaN, NaN >= 0 = false).
    let fres = f32(res);
    let half = fres * 0.5;
    if !(p.x.x >= 0.0 && p.x.x < fres) { p.x.x = half; }
    if !(p.x.y >= 0.0 && p.x.y < fres) { p.x.y = half; }
    // Velocity: NaN v makes new_x = NaN → position never recovers (clamp(NaN) = NaN on AMD).
    if !(dot(p.v, p.v) >= 0.0) { p.v = vec2<f32>(0.0); }
    // velocity_gradient: NaN C makes new_F = NaN → F never recovers.
    let cg = frob2_sq(p.velocity_gradient);
    if !(cg >= 0.0) { p.velocity_gradient = mat2x2<f32>(); }
    // deformation_gradient: NaN or det ≤ 0 → identity (J-projection below also covers post-update).
    if !(det2(p.deformation_gradient) > 0.0) { p.deformation_gradient = I; }
    // Plastic state -- NaN can cascade from bad F or extreme stress over long GPU sims.
    // !(x > 0) catches NaN+negative; !(abs(x) < BIG) catches NaN+Inf for signed fields.
    // Mirrors project_particle_state_to_admissible in solver/mod.rs lines 864–875.
    if !(p.plastic_volume_ratio > 0.0)           { p.plastic_volume_ratio = 1.0; }
    if !(p.hardening_scale > 0.0)                { p.hardening_scale = 1.0; }
    if !(abs(p.friction_hardening) < 3.4e+38)   { p.friction_hardening = 0.0; }
    if !(abs(p.log_volume_strain)  < 3.4e+38)   { p.log_volume_strain  = 0.0; }
    // ─────────────────────────────────────────────────────────────────────────

    // F = (I + dt·C) · F_old  (C = velocity_gradient written by g2p pass)
    // NeoHookean (2)/Corotated (3)/Snow (4)/Drucker-Prager (5)/Von Mises (6)/Rankine (7)/
    // Viscoelastic (9)/GranularFluid (11) are independently verified with the exact kinematic
    // increment. Plastic models were migrated one family at a time with
    // marginal-yield and CPU/GPU checks; see `deformation_increment_exp`.
    // Remaining tensor-F model SandMuI (8) stays on the original path until
    // their own plastic projections receive the same audit.
    var f_increment = I + dt * p.velocity_gradient;
    if has_model(mat.model, 2u) || has_model(mat.model, 3u) || has_model(mat.model, 4u)
        || has_model(mat.model, 5u) || has_model(mat.model, 6u) || has_model(mat.model, 7u)
        || has_model(mat.model, 9u) || has_model(mat.model, 11u) {
        f_increment = deformation_increment_exp(dt * p.velocity_gradient);
    }
    var new_F = f_increment * p.deformation_gradient;

    // Plasticity -- all three models via 2D analytical SVD.
    if has_model(mat.model, 4u) && mat.compression_limit > 0.0 {
        // Snow: clamp singular values to elastic range; accumulate Jp and hardening h.
        let sr          = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F           = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    } else if has_model(mat.model, 5u) {
        // Drucker-Prager (sand): log-strain return mapping + friction-angle hardening.
        let svd    = svd2(new_F);
        let dp_res = dp_plasticity(svd.s, p.log_volume_strain, p.friction_hardening, mat);
        // Volumetric floor mirrors sand.rs's CPU-side fix (see `min_volume_jacobian`'s
        // doc): the DP cone only ever trims shear, never caps pure hydrostatic
        // compression, so a near-vertical impact can crush a particle past sand's real
        // packing limit. Uniform rescale preserves the shear-yield projection's chosen
        // deviatoric shape, only corrects overall volume.
        //
        // Take magnitudes FIRST: this engine's svd2 does NOT guarantee non-negative
        // singular values -- it keeps u a proper rotation by encoding a reflection as a
        // NEGATIVE s.y instead (see this file's own svd2: `if det_f < 0.0 { s.y = -s.y;
        // ... }`). An inverted particle (sigma.y < 0) is exactly the "exceeded packing
        // limit" case this floor exists for, just approached from the other side.
        var dp_sigma = abs(dp_res.sigma);
        // Floor each axis individually before the product-based rescale below --
        // same real bug (and same fix) as CPU `DruckerPragerMaterial`'s own
        // `MIN_AXIS` guard, and the duplicate of this code in
        // `g2p_asflip_fused.wgsl` (see either doc): under a hard enough impact
        // one singular value can collapse to exactly (or within float noise of)
        // zero on its own axis, and a rescale that multiplies BOTH axes by the
        // same scalar can never recover an axis already at zero (0 * any finite
        // scalar is still 0).
        dp_sigma = max(dp_sigma, vec2<f32>(1e-3));
        let dp_j = dp_sigma.x * dp_sigma.y;
        if dp_j < mat.volume_ratio_min {
            dp_sigma *= sqrt(mat.volume_ratio_min / max(dp_j, NUM_FLOOR_TIGHT));
        }
        let diag   = mat2x2<f32>(vec2<f32>(dp_sigma.x, 0.0), vec2<f32>(0.0, dp_sigma.y));
        new_F                = svd.u * diag * transpose(svd.v);
        // q cap: mirrors sand.rs `q_max = 5.0 / hardening_decay`. Prevents unbounded accumulation.
        let q_max = 5.0 / max(mat.dp_h2, NUM_FLOOR_TIGHT);
        p.friction_hardening = min(p.friction_hardening + dp_res.dq, q_max);
        p.log_volume_strain  += dp_res.log_vol_delta;
    } else if has_model(mat.model, 6u) {
        // Von Mises: J2 plasticity with optional linear isotropic hardening.
        let vm_res           = vm_plasticity(new_F, p.friction_hardening, mat);
        new_F                = vm_res.f_e;
        p.friction_hardening += vm_res.dkappa;
    } else if has_model(mat.model, 7u) {
        // Rankine: tensile cutoff with exponential damage softening.
        let rk_res           = rankine_plasticity(new_F, p.friction_hardening, mat);
        new_F                = rk_res.f_e;
        p.friction_hardening += rk_res.damage_delta;
    } else if has_model(mat.model, 8u) {
        // SandMuI: µ(I)-rheology rate-dependent Drucker-Prager.
        let mi_res           = sand_mui_plasticity(new_F, p.friction_hardening, mat, dt);
        new_F                = mi_res.f_e;
        p.friction_hardening = mi_res.mu_i;
    } else if has_model(mat.model, 11u) && mat.compression_limit > 0.0 {
        // GranularFluid: snow-style SVD plasticity -- clamp singular values, accumulate Jp and h.
        let sr               = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F                = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    }

    // Fluid F reset: extract J = det(F), reset to isotropic F = sqrt(J)·I.
    //
    // Rotation and shear in F are physically meaningless for fluids -- the EOS uses only
    // J = det(F) (volume ratio). Accumulated shear/rotation can cause individual F elements
    // to drift toward ±∞ even when det(F) stays bounded → Inf−Inf=NaN. Reset preserves J.
    //
    // Fluid F reset: extract J = det(F), reset to isotropic F = sqrt(J)·I.
    // Rotation and shear in F are physically meaningless for fluids -- only J = det(F) matters.
    //
    // J bounds come from MaterialParams (set in NewtonianFluidMaterial::params()):
    //   J_MIN = 0.1: prevents sqrt(negative) and log(0) in stress.
    //   J_MAX = volume_ratio_max (default 2.0): caps free-surface expansion. Without this,
    //   divergent flow compounds J multiplicatively since EOS provides no restoring force above J≈1.
    //   Fallback to 2.0 if volume_ratio_max not set (e.g. Bingham fluid with default params).
    const FLUID_J_MIN: f32 = 0.5; // below this, EOS pressure overwhelms timestep → clamp to prevent crushing
    if mat.model == 1u {
        let fluid_j_max = select(2.0, mat.volume_ratio_max, mat.volume_ratio_max > 1.0);
        // Real, disclosed regression fixed 2026-08-30 -- same fix, same
        // root cause, as CPU's NewtonianFluidMaterial::update_particle (see
        // that function's own doc for the full writeup): `new_F`'s own
        // determinant (built above via `(I+dt*C)*F_old`) is NOT rotation-
        // invariant -- a pure rigid rotation (div(v)=0) should leave J
        // exactly unchanged, but that formula gives a strictly positive
        // O(dt^2) expansion every substep, baked in permanently by this
        // branch's own isotropic reset just below. Fixed with the
        // continuity equation's own exact exponential solution,
        // `J_new = J_old * exp(dt*div(v))`, computed from the OLD
        // (pre-substep) `p.deformation_gradient`/`p.velocity_gradient`
        // directly instead of trusting `new_F`'s determinant.
        // TESTED (2026-09-16): a GPU-native per-substep rate clamp on
        // `dt*div_v` (reusing `SimConfig::fluid_step_retry_threshold`'s real,
        // already-disclosed 0.5 bound, since that CPU-only rollback-and-retry
        // mechanism has zero GPU implementation) was tried here and found
        // INERT, not merely ineffective: byte-identical results to the
        // unclamped baseline (v=10.963 at frame 40, exact match). Real
        // reason, confirmed by arithmetic: at this scene's ~650-700
        // substeps/frame (needed by the Eleventh pass's `max_substeps_per_
        // step=1000` fix), each sub_dt is ~1.5e-4, so hitting the 0.5 bound
        // would need `div_v~=3250` -- implausible given max observed speed is
        // only 10-18 over ~1 grid-unit spacing. The finer substeps that fixed
        // the earlier CFL-truncation explosion also make a per-substep RATE
        // bound structurally unable to bind here. Not a lever for this
        // scene's disintegration bug; see HANDOFF_fluid_gpu_thin_layer_bug.md
        // Twelfth pass for the other 3 real fixes already ruled out.
        let old_J = det2(p.deformation_gradient);
        // `trace2`, not `p.velocity_gradient[0].x + p.velocity_gradient[1].y`:
        // see `trace2`'s own doc -- the direct-index form silently computed
        // C[0][0] + C[0][1] on the AMD Vulkan target, the real root cause of
        // the GPU fluid impact explosion (and of the per-substep shear damping
        // once added to mask it).
        let div_v = trace2(p.velocity_gradient);
        var J_fluid = old_J * exp(dt * div_v);
        if !(J_fluid > 0.0) { J_fluid = 1.0; }
        J_fluid = clamp(J_fluid, FLUID_J_MIN, fluid_j_max);
        let sqrtJ = sqrt(J_fluid);
        new_F = mat2x2<f32>(vec2<f32>(sqrtJ, 0.0), vec2<f32>(0.0, sqrtJ));

        // Settling damping: v *= (1 − k·dt). Damps gravity-wave sloshing and slow creep.
        // k = dp_h0 (repurposed -- dp_h0..dp_h3 are DP-only, unused for fluid model 1).
        if mat.dp_h0 > 0.0 {
            p.v *= 1.0 - clamp(mat.dp_h0 * dt, 0.0, 0.5);
        }

        // GPU/CPU parity fix (2026-08-15): derive density/volume ANALYTICALLY
        // from this already-clamped J, matching CPU's fluid.rs::update_particle
        // exactly (`density = (rest_density/j).max(min_density).min(2*rest_density)`,
        // NUM_FLOOR here is the same 1e-6 CPU's `min_density` default uses).
        // Only takes effect for materials with owns_deformation_volume_state=1u
        // -- g2p.wgsl already skipped its own kernel-mass write for exactly
        // these particles, so this is the ONLY place their density/volume get
        // set, every substep, same as CPU's own single source of truth.
        if mat.owns_deformation_volume_state == 1u {
            let density = clamp(mat.rest_density / J_fluid, NUM_FLOOR, mat.rest_density * 2.0);
            particles[p_idx].density = density;
            particles[p_idx].volume  = p.mass / density;
        }
    }

    // J-projection for elastic/plastic models: near-boundary APIC C can flip det(F) negative.
    // Uses !(J > 0) instead of J <= 0 to also catch NaN -- mirrors CPU project_invalid_state.
    // (NaN > 0 = false, so !(NaN > 0) = true → reset triggered. NaN <= 0 = false → missed.)
    let J_trial = det2(new_F);
    if !(J_trial > 0.0) {
        if mat.model == 1u {
            // Should not reach here after the fluid reset above, but guard defensively.
            new_F = I;
        } else if (MODELS_PRESENT & ~(1u << 1u)) != 0u {
            // (Unreachable, and compiled away, when the scene holds fluids only.)
            // Flip sign of smallest singular value to restore det > 0.
            let svd_r = svd2(new_F);
            let sc    = vec2<f32>(svd_r.s.x, abs(svd_r.s.y) + NUM_FLOOR);
            let diag  = mat2x2<f32>(vec2<f32>(sc.x, 0.0), vec2<f32>(0.0, sc.y));
            new_F     = svd_r.u * diag * transpose(svd_r.v);
        }
    }

    // Elastic F/J clamping -- only Viscoelastic (9) needs explicit bounds on F.
    //
    // NeoHookean (2) and Corotated (3): NO floor applied here.
    //   p2g kirchhoff() already does J=max(det2(F), NUM_FLOOR) in stress → no explosion.
    //   Modifying F would corrupt stored elastic energy and kill bounce (energy dissipated
    //   each clamp event because particle positions are inconsistent with the rescaled F).
    //   wgsparkl ref: corotated has no J clamp; NeoHookean clamps J in stress only.
    //
    // Plasticity models (4=Snow, 5=DP, 6=VM) clamp their own singular values via
    //   return mapping above, so they never reach here.
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

    // density and volume are written by g2p (grid-mass gather: Σ w_i·m_i).
    // This mirrors CPU estimate_density_and_volume_impl (density.rs) exactly.
    // p.density and p.volume already hold the correct values -- nothing to recompute here.

    // No velocity damping for elastic/viscoelastic models (0, 2, 3, 9) -- APIC is
    // energy-conserving and extra damping causes over-settling that leads to floor-compression
    // instability. Plastic flow (snow, sand, VM, etc.) provides its own dissipation.
    // For plasticity models we apply a very light damping as a boundary-edge safety margin.
    // Model 1u (fluid) excluded: explicit viscosity already dissipates; extra damping slows flow.
    // Light damping for plasticity models -- their explicit dissipation (yield, flow) is enough,
    // but a small margin prevents edge-particle instability near boundaries.
    // Elastic (2, 3) and fluid (0, 1) excluded -- APIC is energy-conserving; damping fights that.
    // Viscoelastic (9) excluded: viscosity stress handles dissipation during deformation.
    // Velocity damping would bleed into free-fall and make vis fall slower than other materials.
    if mat.model != 0u && mat.model != 1u && mat.model != 2u && mat.model != 3u && mat.model != 9u {
        p.v *= 0.999;
    }

    // Position update: x += v · dt  (v written by g2p pass)
    var new_x = p.x + p.v * dt;

    // Boundary clamp (slip boundary -- mirrors clamp_position_inside_grid in boundary.rs).
    // CPU: min = thickness.saturating_sub(1) = bt-1, max = grid_res - bt.
    let lo = max(0.0, bt - 1.0);
    let hi = f32(res) - bt;
    new_x  = clamp(new_x, vec2<f32>(lo), vec2<f32>(hi));

    // Write updated fields back.
    particles[p_idx].x                    = new_x;
    particles[p_idx].v                    = p.v;  // damped velocity must persist for next g2p gather
    particles[p_idx].deformation_gradient = new_F;
    // Plastic fields (only modified for the matching material model above).
    particles[p_idx].plastic_volume_ratio = p.plastic_volume_ratio;
    particles[p_idx].hardening_scale      = p.hardening_scale;
    particles[p_idx].friction_hardening   = p.friction_hardening;
    particles[p_idx].log_volume_strain    = p.log_volume_strain;
    (*pp).x = new_x;
    (*pp).v = p.v;
}

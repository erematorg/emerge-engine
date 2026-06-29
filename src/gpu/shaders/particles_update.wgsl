// particles_update — per-particle F update, plasticity, volume/density, position, boundary.
// MLS-MPM, Hu et al. 2018 SIGGRAPH §4.
//
// One thread per particle (sorted access via sorted_particle_ids).
// Reads v and velocity_gradient (C matrix) written by the preceding g2p pass,
// then runs all remaining per-particle state updates.
//
// Steps (mirrors the second half of the old fused g2p pass):
//   1. F = (I + dt·C) · F_old          (C = velocity_gradient from g2p)
//   2. Snow plasticity (model 4): 2D SVD → clamp σ → update Jp/h → reconstruct F_e
//   3. DP plasticity  (model 5): 2D SVD → log-strain return mapping → update q/log_volume_strain
//   4. Von Mises      (model 6): 2D SVD → J2 yield check → deviatoric return mapping
//   5. J = det(F), volume = initial_volume × J, density = mass / volume
//   6. Position: x = x + v · dt
//   7. Boundary clamp: slip — clamp x within [bt, grid_res−bt)
//
// Sorted particle access: reads particles[sorted_particle_ids[gid.x]] for
// cache-coherent scatter in p2g (same permutation used there).
// CPU sort in step_frame() provides per-frame spatial ordering; particle_sort
// seeds the identity permutation at the start of each frame.

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
    sleeping:             u32,       // total 112 bytes
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

const MAX_MATERIALS:    u32 = {{MAX_MATERIALS}}u;
const NUM_FLOOR:        f32 = 1e-6;
const NUM_FLOOR_TIGHT:  f32 = 1e-10;

@group(0) @binding(0) var<storage, read_write> particles:            array<Particle>;
@group(0) @binding(2) var<uniform>             materials:            array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>             step_params:          StepParams;
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
        // dq = dn only (not length(eps)) — log_volume_strain offset must not contribute.
        // length(eps) causes unbounded q growth in settled sand. Mirrors sand.rs:130.
        let prev_det = sigma.x * sigma.y;
        return DpReturn(vec2<f32>(1.0), dn, log(max(prev_det, NUM_FLOOR_TIGHT * NUM_FLOOR_TIGHT)));
    }

    // Single-pass: alpha evaluated once from the pre-step q, matching
    // wgsparkl::models::drucker_prager::project_deformation_gradient exactly (the
    // reference GPU implementation of Klar et al. 2016 — no self-consistency corrector).
    // stretch_limit repurposed for DP: cohesion floor, see sand.rs's `cohesion` doc
    // comment — NOT real "sand cohesion" (dry sand is ~0), a continuum-MPM-resolution
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
    let eps_proj  = dev * (yield_s / elastic_dev) + vec2<f32>(tr * 0.5);
    let sigma_new = exp(eps_proj);
    let diag      = mat2x2<f32>(vec2<f32>(sigma_new.x, 0.0), vec2<f32>(0.0, sigma_new.y));
    return VmReturn(svd.u * diag * transpose(svd.v), gamma);
}

// ─────────────────────────────────────────────────────────────────────────────

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// Workgroup size MUST match WG_PARTICLES (= 64) in src/gpu/mod.rs.
@compute @workgroup_size(64, 1, 1)
fn particles_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= step_params.particle_count { return; }
    let p_idx = sorted_particle_ids[gid.x]; // sorted for cache-coherent p2g scatter

    var p   = particles[p_idx];

    // Still-sleeping particles (didn't wake in g2p this substep) are frozen — skip
    // state projection, F update, every plasticity branch, and position integration
    // entirely. Particles that woke in g2p have sleeping=0u by this point and get the
    // full update below, same as CPU (a newly-woken particle gets a real update the
    // same substep it wakes).
    if p.sleeping != 0u { return; }

    let mat = materials[p.material_id];
    let dt  = step_params.dt;
    let res = step_params.grid_res;
    let bt  = f32(step_params.boundary_thickness);

    // Identity matrix — used both in state projection and F update below.
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
    let cg = dot(p.velocity_gradient[0], p.velocity_gradient[0])
           + dot(p.velocity_gradient[1], p.velocity_gradient[1]);
    if !(cg >= 0.0) { p.velocity_gradient = mat2x2<f32>(); }
    // deformation_gradient: NaN or det ≤ 0 → identity (J-projection below also covers post-update).
    if !(det2(p.deformation_gradient) > 0.0) { p.deformation_gradient = I; }
    // Plastic state — NaN can cascade from bad F or extreme stress over long GPU sims.
    // !(x > 0) catches NaN+negative; !(abs(x) < BIG) catches NaN+Inf for signed fields.
    // Mirrors project_particle_state_to_admissible in solver/mod.rs lines 864–875.
    if !(p.plastic_volume_ratio > 0.0)           { p.plastic_volume_ratio = 1.0; }
    if !(p.hardening_scale > 0.0)                { p.hardening_scale = 1.0; }
    if !(abs(p.friction_hardening) < 3.4e+38)   { p.friction_hardening = 0.0; }
    if !(abs(p.log_volume_strain)  < 3.4e+38)   { p.log_volume_strain  = 0.0; }
    // ─────────────────────────────────────────────────────────────────────────

    // F = (I + dt·C) · F_old  (C = velocity_gradient written by g2p pass)
    var new_F = (I + dt * p.velocity_gradient) * p.deformation_gradient;

    // Plasticity — all three models via 2D analytical SVD.
    if mat.model == 4u && mat.compression_limit > 0.0 {
        // Snow: clamp singular values to elastic range; accumulate Jp and hardening h.
        let sr          = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F           = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    } else if mat.model == 5u {
        // Drucker-Prager (sand): log-strain return mapping + friction-angle hardening.
        let svd    = svd2(new_F);
        let dp_res = dp_plasticity(svd.s, p.log_volume_strain, p.friction_hardening, mat);
        let diag   = mat2x2<f32>(vec2<f32>(dp_res.sigma.x, 0.0), vec2<f32>(0.0, dp_res.sigma.y));
        new_F                = svd.u * diag * transpose(svd.v);
        // q cap: mirrors sand.rs `q_max = 5.0 / hardening_decay`. Prevents unbounded accumulation.
        let q_max = 5.0 / max(mat.dp_h2, NUM_FLOOR_TIGHT);
        p.friction_hardening = min(p.friction_hardening + dp_res.dq, q_max);
        p.log_volume_strain  += dp_res.log_vol_delta;
    } else if mat.model == 6u {
        // Von Mises: J2 plasticity with optional linear isotropic hardening.
        let vm_res           = vm_plasticity(new_F, p.friction_hardening, mat);
        new_F                = vm_res.f_e;
        p.friction_hardening += vm_res.dkappa;
    } else if mat.model == 7u {
        // Rankine: tensile cutoff with exponential damage softening.
        let rk_res           = rankine_plasticity(new_F, p.friction_hardening, mat);
        new_F                = rk_res.f_e;
        p.friction_hardening += rk_res.damage_delta;
    } else if mat.model == 8u {
        // SandMuI: µ(I)-rheology rate-dependent Drucker-Prager.
        let mi_res           = sand_mui_plasticity(new_F, p.friction_hardening, mat, dt);
        new_F                = mi_res.f_e;
        p.friction_hardening = mi_res.mu_i;
    } else if mat.model == 11u && mat.compression_limit > 0.0 {
        // GranularFluid: snow-style SVD plasticity — clamp singular values, accumulate Jp and h.
        let sr               = snow_plasticity(new_F, p.plastic_volume_ratio, mat);
        new_F                = sr.f_e;
        p.plastic_volume_ratio = sr.jp;
        p.hardening_scale      = sr.h;
    }

    // Fluid F reset: extract J = det(F), reset to isotropic F = sqrt(J)·I.
    //
    // Rotation and shear in F are physically meaningless for fluids — the EOS uses only
    // J = det(F) (volume ratio). Accumulated shear/rotation can cause individual F elements
    // to drift toward ±∞ even when det(F) stays bounded → Inf−Inf=NaN. Reset preserves J.
    //
    // Fluid F reset: extract J = det(F), reset to isotropic F = sqrt(J)·I.
    // Rotation and shear in F are physically meaningless for fluids — only J = det(F) matters.
    //
    // J bounds come from MaterialParams (set in NewtonianFluidMaterial::params()):
    //   J_MIN = 0.1: prevents sqrt(negative) and log(0) in stress.
    //   J_MAX = volume_ratio_max (default 2.0): caps free-surface expansion. Without this,
    //   divergent flow compounds J multiplicatively since EOS provides no restoring force above J≈1.
    //   Fallback to 2.0 if volume_ratio_max not set (e.g. Bingham fluid with default params).
    const FLUID_J_MIN: f32 = 0.5; // below this, EOS pressure overwhelms timestep → clamp to prevent crushing
    if mat.model == 1u {
        let fluid_j_max = select(2.0, mat.volume_ratio_max, mat.volume_ratio_max > 1.0);
        var J_fluid = det2(new_F);
        if !(J_fluid > 0.0) { J_fluid = 1.0; }
        J_fluid = clamp(J_fluid, FLUID_J_MIN, fluid_j_max);
        let sqrtJ = sqrt(J_fluid);
        new_F = mat2x2<f32>(vec2<f32>(sqrtJ, 0.0), vec2<f32>(0.0, sqrtJ));

        // Settling damping: v *= (1 − k·dt). Damps gravity-wave sloshing and slow creep.
        // k = dp_h0 (repurposed — dp_h0..dp_h3 are DP-only, unused for fluid model 1).
        if mat.dp_h0 > 0.0 {
            p.v *= 1.0 - clamp(mat.dp_h0 * dt, 0.0, 0.5);
        }
    }

    // J-projection for elastic/plastic models: near-boundary APIC C can flip det(F) negative.
    // Uses !(J > 0) instead of J <= 0 to also catch NaN — mirrors CPU project_invalid_state.
    // (NaN > 0 = false, so !(NaN > 0) = true → reset triggered. NaN <= 0 = false → missed.)
    let J_trial = det2(new_F);
    if !(J_trial > 0.0) {
        if mat.model == 1u {
            // Should not reach here after the fluid reset above, but guard defensively.
            new_F = I;
        } else {
            // Flip sign of smallest singular value to restore det > 0.
            let svd_r = svd2(new_F);
            let sc    = vec2<f32>(svd_r.s.x, abs(svd_r.s.y) + NUM_FLOOR);
            let diag  = mat2x2<f32>(vec2<f32>(sc.x, 0.0), vec2<f32>(0.0, sc.y));
            new_F     = svd_r.u * diag * transpose(svd_r.v);
        }
    }

    // Elastic F/J clamping — only Viscoelastic (9) needs explicit bounds on F.
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
    // p.density and p.volume already hold the correct values — nothing to recompute here.

    // No velocity damping for elastic/viscoelastic models (0, 2, 3, 9) — APIC is
    // energy-conserving and extra damping causes over-settling that leads to floor-compression
    // instability. Plastic flow (snow, sand, VM, etc.) provides its own dissipation.
    // For plasticity models we apply a very light damping as a boundary-edge safety margin.
    // Model 1u (fluid) excluded: explicit viscosity already dissipates; extra damping slows flow.
    // Light damping for plasticity models — their explicit dissipation (yield, flow) is enough,
    // but a small margin prevents edge-particle instability near boundaries.
    // Elastic (2, 3) and fluid (0, 1) excluded — APIC is energy-conserving; damping fights that.
    // Viscoelastic (9) excluded: viscosity stress handles dissipation during deformation.
    // Velocity damping would bleed into free-fall and make vis fall slower than other materials.
    if mat.model != 0u && mat.model != 1u && mat.model != 2u && mat.model != 3u && mat.model != 9u {
        p.v *= 0.999;
    }

    // Position update: x += v · dt  (v written by g2p pass)
    var new_x = p.x + p.v * dt;

    // Boundary clamp (slip boundary — mirrors clamp_position_inside_grid in boundary.rs).
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
}

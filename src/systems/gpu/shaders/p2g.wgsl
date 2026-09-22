// P2G -- scatter particle mass, momentum and stress to the 3×3 grid neighbourhood.
// One thread per particle. Uses fixed-point atomicAdd (WebGPU has no atomic<f32>).

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
    internal_pressure:    f32,
}

struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32, // Snow: ξ;  VonMises: yield_stress
    compression_limit:       f32, // Snow: θ_c;  DP: dilatancy ψ;  Bingham: yield_stress
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
    cohesion_coeff:          f32,
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
    contact_friction:              f32,
    grid_cell_size:              f32,
    contact_active:              u32,
}

const MAX_MATERIALS:        u32 = {{MAX_MATERIALS}}u;
const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const NUM_FLOOR:            f32 = 1e-6;

// Real, restored 2026-09-15 (see the fluid-branch shock-viscosity comment
// below for the full story): a real, simple magnitude-based finiteness
// check -- `abs(NaN) <= X` is false under IEEE754 comparison rules, so this
// correctly rejects NaN too, not just +-inf.
fn finite_scalar(value: f32) -> bool {
    return abs(value) <= 3.4e38;
}

// Mirrors `utils::fast_pow` (Rust, CPU) exactly: for a WHOLE-numbered
// exponent, exponentiation by squaring (repeated multiplication) instead of
// WGSL's built-in `pow()`, which always evaluates as `exp2(e*log2(x))` even
// for integer exponents -- measurably LESS precise than direct
// multiplication for x near 1.0, a real, confirmed (not theoretical) GPU
// numerics gap found 2026-09-15 by a real CPU/GPU parity test: at
// eos_power=4.0 (shock-viscosity term needs eos_power-1.0=3.0), plain
// `pow()` gave a velocity mismatch of 0.042 against a 2e-3 tolerance, an
// order of magnitude over -- no prior test had ever checked tight CPU/GPU
// numeric agreement under real compression before this one (the one that
// would have, `gpu_and_cpu_strict_fluid_match_one_substep`, was already
// `#[ignore]`d for an unrelated reason). Falls back to `pow()` for
// non-integer or very large exponents, matching the CPU function's own
// real fallback rule exactly.
fn fast_pow(x: f32, e: f32) -> f32 {
    if abs(fract(e)) > 1.0e-6 || abs(e) >= 32.0 {
        return pow(x, e);
    }
    var exp_i = i32(round(e));
    let neg = exp_i < 0;
    if neg {
        exp_i = -exp_i;
    }
    var base = x;
    var result = 1.0;
    var n = exp_i;
    while n > 0 {
        if (n & 1) == 1 {
            result = result * base;
        }
        base = base * base;
        n = n >> 1;
    }
    if neg {
        return 1.0 / result;
    }
    return result;
}
// Fixed-point scales: mass and momentum use different scales to avoid i32 overflow.
// With 9 particles per cell: mass × 1e6 ≤ 9e6 (safe). Momentum at vel_limit=1000: 9×1000×1e5=9e8 (safe).
// MOM_ATOMIC_SCALE=1e5 gives 1e-5 precision -- 100× better than 1e3, avoids overflow at min_dt=0.001.
const MASS_ATOMIC_SCALE:    f32 = 1000000.0;
const MOM_ATOMIC_SCALE:     f32 = 100000.0;
// Matches render::step_params::MAX_RENDER_MATERIAL_SLOTS exactly (Rust-side source of
// truth) -- render::OpticalTable's own real 16-slot cap, not MAX_MATERIALS' larger
// 64-material solver cap. material_id >= 16 collides into slot material_id % 16, same
// convention Renderer::set_optical_params already uses.
const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;
// Multi-field contact (GPU port, first slice) -- must match
// `step_params::MAX_CONTACT_POINTS_PER_BLOCK` (Rust-side source of truth, sizes the
// `contact_points` buffer at construction) exactly, same duplicated-constant
// convention already used for MASS_ATOMIC_SCALE/MOM_ATOMIC_SCALE above. Bucketed per
// a DEDICATED finer contact-block partition, not per exact node -- see that constant's
// own doc in step_params.rs for why (a first per-node version OOM'd at high grid_res;
// a 2026-07-18 re-partition then split this off from the coarser P2G-sort partition to
// fix a real scan-to-keep mismatch, see MAX_CONTACT_POINTS_PER_BLOCK's doc).
const MAX_POINTS_PER_BLOCK: u32 = 256u;
// override, not a hardcoded literal -- must match resolve_contact.wgsl's
// NUM_CONTACT_BLOCKS_PER_DIM exactly, single Rust-side source of truth
// (src/gpu/step_params.rs). Needed here so gather_contact_points_main computes the SAME
// block index resolve_contact's gather_local_points reads. DEDICATED to contact-point
// bucketing -- deliberately NOT the same override as particle_sort.wgsl's
// NUM_BLOCKS_PER_DIM (an unrelated partition, sort-permutation/active-block occupancy).
override NUM_CONTACT_BLOCKS_PER_DIM: u32;

@group(0) @binding(0) var<storage, read_write> particles:           array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid_atomic:         array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             materials:           array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>             step_params:         StepParams;

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

@group(0) @binding(5) var<storage, read_write> sorted_particle_ids: array<u32>;
// Multi-field contact (GPU port, first slice) -- see buffers.rs doc. binding 12 is the
// SAME underlying buffer as grid_clear.wgsl's `grip_grid: array<Cell>` binding, viewed
// here as raw atomics for scatter (same dual-view convention already used for `grid`
// itself, bound as `array<Cell>` in grid_clear.wgsl and `array<atomic<i32>>` here).
@group(1) @binding(12) var<storage, read_write> grip_grid_atomic:     array<atomic<i32>>;
@group(1) @binding(13) var<storage, read_write> contact_points:       array<vec4<f32>>;
@group(1) @binding(14) var<storage, read_write> contact_point_counts: array<atomic<u32>>;

struct MaterialMassParams {
    enabled: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
}
// `ColorMode::GridVolume`'s opt-in per-cell per-material mass accumulator -- see
// buffers.rs's `material_mass` doc. Shares group 1 purely for bind-group economy
// (same reason ASFLIP shares group 3), nothing to do with contact thematically.
@group(1) @binding(30) var<storage, read_write> material_mass_atomic: array<atomic<i32>>;
@group(1) @binding(31) var<uniform>              material_mass_params: MaterialMassParams;

// Exact copy of resolve_contact.wgsl's block_index_of -- WGSL has no cross-file
// includes, so this is duplicated the same way MASS_ATOMIC_SCALE etc. already are
// across shader files. Must stay byte-for-byte identical: gather_contact_points_main
// needs the SAME contact-block a given cell belongs to as resolve_contact's
// gather_local_points scans by this same index.
fn contact_block_index(pos: vec2<f32>, grid_res: u32) -> u32 {
    let max_cell = grid_res - 1u;
    let cell_x = u32(clamp(pos.x, 0.0, f32(max_cell)));
    let cell_y = u32(clamp(pos.y, 0.0, f32(max_cell)));
    let block_size = (grid_res + NUM_CONTACT_BLOCKS_PER_DIM - 1u) / NUM_CONTACT_BLOCKS_PER_DIM;
    let block_x = min(cell_x / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    let block_y = min(cell_y / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    return block_y * NUM_CONTACT_BLOCKS_PER_DIM + block_x;
}

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// 2D polar decomposition R -- analytical, mirrors corotated.rs.
fn polar_r(f: mat2x2<f32>) -> mat2x2<f32> {
    let x = f[0][0] + f[1][1];
    let y = f[0][1] - f[1][0];
    let n = sqrt(x * x + y * y);
    if n < NUM_FLOOR { return mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0)); }
    let inv = 1.0 / n;
    return mat2x2<f32>(vec2<f32>(x, y) * inv, vec2<f32>(-y, x) * inv);
}

// Kirchhoff stress τ for all supported material models.
fn kirchhoff(p: Particle, mat: MaterialParams) -> mat2x2<f32> {
    let F = p.deformation_gradient;
    let J = max(det2(F), NUM_FLOOR);
    let h = p.hardening_scale;
    let I = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));

    var tau: mat2x2<f32>;
    switch mat.model {
        case 1u: { // Fluid -- Tait EOS + Newtonian or Bingham deviatoric viscosity
            // Use J = det(F) for EOS density: ρ = ρ₀/J (sparkl canonical, no grid-lag).
            // F is reset to sqrt(J)·I in particles_update, so det(F) = J always for fluid.
            // This eliminates the one-step lag from grid-mass gather (p.density) and keeps
            // EOS consistent with the F-tracked volume ratio.
            let rho   = clamp(mat.rest_density / max(J, NUM_FLOOR), NUM_FLOOR, mat.rest_density * 2.0);
            let ratio = rho / max(mat.rest_density, NUM_FLOOR);
            let press = max(mat.eos_stiffness * (fast_pow(ratio, mat.eos_power) - 1.0), mat.pressure_floor);
            var t     = -press * I;

            let sym  = p.velocity_gradient + transpose(p.velocity_gradient);
            let tr_s = sym[0][0] + sym[1][1];
            let dev  = sym - (tr_s * 0.5) * I;

            // Arrhenius thermal thinning: µ_eff = µ₀·exp(−k·T)
            //
            // TESTED AND REVERTED (2026-09-16): a Smagorinsky (1963) sub-grid
            // eddy-viscosity addition here (real citation, Lilly 1967's
            // Cs~0.17-0.2) was tried as a fix for the free-fall velocity-
            // gradient runaway traced on idx=1994. At the real Cs=0.17 it had
            // negligible effect (byte-near-identical trajectory). Escalated
            // diagnostically to Cs=2.0 (real damping effect, still visually
            // fragmented) then Cs=8.0 -- which caused a SECOND, much worse,
            // unrelated catastrophic explosion (|v| in the tens of thousands)
            // even after adding a matching viscous-CFL bound
            // (`smagorinsky_viscous_dt_bound`, since removed along with this
            // term -- both were real, correctly-derived code, just didn't
            // fix the problem): GPU's
            // CFL scan runs once per FRAME batch (hundreds of substeps share
            // one dt), so a periodic re-check cannot react fast enough if the
            // underlying growth compounds within a single batch -- the same
            // structural latency this whole investigation's growth pattern
            // exhibits. Real lesson: any fix here needs to act every
            // substep directly, not depend on a periodic CFL re-scan. See
            // HANDOFF_fluid_gpu_thin_layer_bug.md's Twelfth pass.
            let eff_visc = select(mat.dynamic_viscosity,
                mat.dynamic_viscosity * exp(-mat.thermal_viscosity_coeff * p.temperature),
                mat.thermal_viscosity_coeff > 0.0);

            let yield_s = mat.compression_limit; // Bingham τ₀; 0 for Newtonian
            if yield_s > 0.0 {
                // Bingham: apparent viscosity = τ₀/γ̇ + µ. Skip deviatoric below plug threshold.
                // γ̇ uses the deviatoric strain rate only -- a yield criterion must not respond
                // to pure volumetric expansion/compression, which isn't shear.
                let dx = dev[0][0]; let dy = dev[1][1]; let dxy = dev[0][1];
                let shear_rate = sqrt(max(0.5 * (dx*dx + dy*dy + 2.0*dxy*dxy), 0.0));
                if shear_rate > 1e-4 {
                    // Real, disclosed regression fix (external review): `dev`
                    // here is `sym = C+C^T = 2*D`, i.e. already 2*D_dev, same
                    // as the Newtonian branch below uses directly (`eff_visc *
                    // dev`) -- the real tensorial Bingham law is
                    // `2*eta_app*D_dev = eta_app*dev`, not `eta_app*dev*0.5`.
                    // Pre-fix, this branch gave exactly HALF the real stress,
                    // and (since the Newtonian branch was always correct)
                    // produced a real discontinuity right at yield_s -> 0
                    // instead of converging to it.
                    let eta_app = yield_s / shear_rate + eff_visc;
                    t = t + dev * eta_app;
                }
            } else {
                t = t + eff_visc * dev;
            }

            // Bulk viscosity damps compression waves.
            if mat.bulk_viscosity > 0.0 {
                t = t + mat.bulk_viscosity * (tr_s * 0.5) * I;
            }

            // Real, sourced (von Neumann & Richtmyer 1950 + Landshoff)
            // shock-capturing viscosity -- RESTORED 2026-09-15. This branch
            // had it once (weak-shock form, see `finite_scalar`'s own doc
            // above), but it was silently deleted in commit `7c7991f`
            // ("consolidate GPU/render backlog") alongside ~480 unrelated
            // line changes to this same file, never disclosed there or
            // restored since -- every GPU-run fluid scene had zero shock-
            // capturing viscosity despite `fluid.rs::kirchhoff_stress`'s own
            // doc still claiming CPU/GPU parity. Ported from the CURRENT,
            // authoritative CPU formula (`fluid.rs::artificial_bulk_
            // viscosity` / `utils::von_neumann_richtmyer_q`), NOT the stale
            // pre-deletion GPU snapshot -- that snapshot still used the
            // strong-shock Kurapatenko coefficient `(gamma+1)/2`, which the
            // CPU side has since moved away from after finding it made a
            // real crash worse (see `fluid.rs`'s own doc: the quadratic
            // term's contribution must feed into the CFL bound before a
            // stronger coefficient is safe, real disclosed future work).
            // Gated to compression (div_v<0) only -- real shocks only form
            // under compression. Uses its OWN `rho = rest_density/J`
            // (floor-clamped only via `J`'s own definition above), NOT the
            // ceiling-clamped `rho` this branch's pressure term uses --
            // matches `von_neumann_richtmyer_q`'s own separate computation
            // exactly, not an approximation.
            let div_v_true = tr_s * 0.5; // sym = C+Cᵀ = 2D, so div(v) = tr(sym)/2
            if div_v_true < 0.0 {
                let density_ratio = 1.0 / J;
                let c2 = mat.eos_stiffness * mat.eos_power
                    * fast_pow(max(density_ratio, 1.0e-8), mat.eos_power - 1.0)
                    / max(mat.rest_density, NUM_FLOOR);
                let c_sound = sqrt(max(c2, 0.0));
                let c0_quadratic = (mat.eos_power + 1.0) * 0.25; // Kurapatenko 1967, weak-shock
                let h = 1.0; // grid_cell_size -- same disclosed scope limit as the CPU side
                let quadratic = c0_quadratic * h * h * div_v_true * div_v_true;
                let linear = h * c_sound * div_v_true; // Landshoff, c1=1.0
                let rho_shock = mat.rest_density / J;
                let q = rho_shock * (quadratic - linear);
                if finite_scalar(q) {
                    t = t - q * I;
                }
            }

            // Surface tension: τ += γ·J·I
            if mat.surface_tension_coeff != 0.0 {
                t = t + mat.surface_tension_coeff * J * I;
            }

            return t;
        }
        case 2u: { // NeoHookean -- Simo-Pister vol-dev split
            // REAL FIX (2026-07-30, root-caused via a headless reproduction --
            // see elastic.rs's `j_min` doc for the full writeup): this used to
            // hard-zero stress below `NUM_FLOOR` (1e-6), matching CPU's OLD
            // behavior -- but that defeated the log-barrier's own documented
            // purpose (diverging restoring stress as J->0) at exactly the
            // moment it's needed most, letting a body under real (not the
            // demos' disclosed-weak) gravity compress past that floor and
            // then NEVER get pushed back out (stress stays permanently zero).
            // Confirmed: a real NeoHookean body under SimConfig::earth's true
            // 981-unit gravity collapsed to J=0.000000 and kept compressing
            // for 190+ more simulated seconds, never recovering -- the long-
            // standing "gravité trop forte" bug. Fix: clamp to `mat.
            // volume_ratio_min` (real, per-material, same GPU param slot
            // `ViscoelasticMaterial` already uses for its own `j_min`, default
            // 0.01 -- NOT the razor-thin NUM_FLOOR) and ALWAYS compute real
            // stress, never zero -- large-but-finite at the floor, not
            // exploding (a naive clamp-to-NUM_FLOOR would blow `mu_e/J` up to
            // `mu_e*1e6`, which is why zero was chosen originally; 0.01 avoids
            // both failure modes).
            let j_floor = max(mat.volume_ratio_min, NUM_FLOOR);
            let J2 = max(det2(F), j_floor);
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            // Damage softening: mu_eff = mu*exp(-rate*damage), same exponential form
            // RankineMaterial uses for tensile strength (continuum damage mechanics).
            // cohesion_coeff repurposed for damage_softening_rate (see elastic.rs
            // params() -- documented reusable padding, zero for other materials).
            let damage_scale = exp(-mat.cohesion_coeff * p.friction_hardening);
            let mu_e  = mat.mu * t_scale * damage_scale;
            let lam_e = mat.lambda * t_scale * damage_scale;
            let B     = F * transpose(F);
            let tr_B  = B[0][0] + B[1][1];
            let dev_B = B - (tr_B * 0.5) * I;
            // 2D plane-strain bulk modulus (k = lam_e + mu_e, not the 3D
            // relation this used to mirror -- see elastic.rs's Rust-side fix
            // for the full derivation; CPU and GPU must match exactly here).
            // Volumetric term: k*ln(J), NOT k/2*(J^2-1) (changed 2026-07-11,
            // mirroring elastic.rs's real fix -- the bounded (J^2-1) form has
            // only a finite compression ceiling and let a sustained driven
            // load ratchet a creature body into unrecoverable compaction; see
            // elastic.rs's kirchhoff_stress doc for the full derivation).
            let k     = lam_e + mu_e;
            // Kelvin-Voigt viscous term, same as case 9u's -- opt-in via
            // dynamic_viscosity (0.0 default, matches elastic.rs's `viscosity`
            // field added 2026-07-11; see that file's timestep_bound for the
            // matching CFL bound this term needs).
            let sym   = p.velocity_gradient + transpose(p.velocity_gradient);
            let d     = sym * 0.5;
            let tr_d  = d[0][0] + d[1][1];
            let d_dev = d - (tr_d * 0.5) * I;
            tau = (mu_e / J2) * dev_B + (k * log(J2)) * I + mat.dynamic_viscosity * d_dev;
        }
        case 3u: { // Corotated, used standalone (no upstream plastic clamp)
            // Split out from the shared 3u-8u group (2026-07-30, same real
            // bug/fix as case 2u's NeoHookean -- see CorotatedMaterial::
            // j_min's own doc). Snow/DP/VonMises/Rankine/SandMuI (still
            // sharing the block below) all have their OWN upstream plastic
            // clamp on F's singular values (in particles_update.wgsl) before
            // this stress function ever sees it -- Corotated used standalone
            // (e.g. `basic_jellies_gpu`) has no such clamp, so its OWN
            // volumetric term's floor is the only thing standing between it
            // and unbounded compression. Uses ITS OWN `volume_ratio_min` (not
            // shared with Snow/Sand's real, different reuse of that same
            // slot for their plastic clamp range -- Corotated doesn't
            // populate it for anything else, so this is safe).
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            let R     = polar_r(F);
            let mu_e  = mat.mu * h * t_scale;
            let lam_e = mat.lambda * h * t_scale;
            let j_floor3 = max(mat.volume_ratio_min, NUM_FLOOR);
            let J3 = max(det2(F), j_floor3);
            tau = 2.0 * mu_e * (F - R) * transpose(F) + lam_e * (J3 - 1.0) * J3 * I;
        }
        case 4u, 5u, 6u, 7u, 8u: { // Snow / DP / VonMises / Rankine / SandMuI
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            let R     = polar_r(F);
            let mu_e  = mat.mu * h * t_scale;
            let lam_e = mat.lambda * h * t_scale;
            tau = 2.0 * mu_e * (F - R) * transpose(F) + lam_e * (J - 1.0) * J * I;
        }
        case 11u: { // GranularFluid -- Tait EOS pressure + corotated elastic deviatoric + SVD plasticity
            // EOS pressure: −k·((ρ/ρ₀)^γ − 1)·I
            let rho   = clamp(mat.rest_density / max(J, NUM_FLOOR), NUM_FLOOR, mat.rest_density * 4.0);
            let ratio = rho / max(mat.rest_density, NUM_FLOOR);
            let press = max(mat.eos_stiffness * (fast_pow(ratio, mat.eos_power) - 1.0), mat.pressure_floor);
            // Corotated elastic deviatoric: 2µ·h·dev[(F−R)·Fᵀ]
            let h      = p.hardening_scale;
            let R      = polar_r(F);
            let mu_eff = mat.mu * h;
            let coro   = 2.0 * mu_eff * (F - R) * transpose(F);
            let tr_c   = coro[0][0] + coro[1][1];
            let dev_c  = coro - (tr_c * 0.5) * I;
            // Small elastic volumetric term from λ -- prevents total collapse under EOS alone
            let lam_e  = mat.lambda * h;
            let lam_vol = lam_e * (J - 1.0) * J * I;
            tau = -press * I + dev_c + lam_vol;

            // Real viscous dissipation -- RESTORED 2026-09-15. Identical
            // form to `NewtonianFluidMaterial`'s own (τ += η·dev(D) +
            // ζ·(∇·v)·I), matching `GranularFluidMaterial::kirchhoff_
            // stress`'s current CPU formula exactly (real, disclosed
            // 2026-08-06 addition -- see `dynamic_viscosity`'s own Rust doc:
            // a real fix for a "superball bounce" hard-impact failure mode).
            // Silently deleted from this branch in the same `7c7991f`
            // consolidation that removed the fluid branch's shock viscosity
            // above -- every GPU-run granular-fluid (mud) scene has been
            // missing exactly the damping mechanism added to prevent hard-
            // impact bouncing, with no disclosure anywhere that it was gone.
            if mat.dynamic_viscosity > 0.0 || mat.bulk_viscosity > 0.0 {
                let sym_v = p.velocity_gradient + transpose(p.velocity_gradient);
                let tr_v  = sym_v[0][0] + sym_v[1][1];
                if mat.dynamic_viscosity > 0.0 {
                    let dev_v = sym_v - (tr_v * 0.5) * I;
                    tau = tau + mat.dynamic_viscosity * dev_v;
                }
                if mat.bulk_viscosity > 0.0 {
                    tau = tau + mat.bulk_viscosity * (tr_v * 0.5) * I;
                }
            }
        }
        case 9u: { // Viscoelastic (Kelvin-Voigt) -- elastic NeoHookean + viscous dashpot
            let j_min   = max(mat.volume_ratio_min, NUM_FLOOR);
            let J_vis   = clamp(J, j_min, 1.0 / j_min);
            let B       = F * transpose(F);
            let lnJ     = log(J_vis);
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            let mu_e    = mat.mu * t_scale;
            let lam_e   = mat.lambda * t_scale;
            let elastic = mu_e * (B - I) + (lam_e * lnJ) * I;
            let sym     = p.velocity_gradient + transpose(p.velocity_gradient);
            let d       = sym * 0.5;
            let tr_d    = d[0][0] + d[1][1];
            let d_dev   = d - (tr_d * 0.5) * I;
            tau = elastic + mat.dynamic_viscosity * d_dev;
        }
        default: { return mat2x2<f32>(); }
    }

    // Snow cohesion: compacted snow resists re-expansion. Only fires when Jp < 1 and J > 1.
    if mat.model == 4u && mat.cohesion_coeff > 0.0 && p.plastic_volume_ratio < 1.0 && J > 1.0 {
        tau = tau + mat.cohesion_coeff * p.plastic_volume_ratio * (J - 1.0) * J * I;
    }

    // Active stress. Viscoelastic (9) uses isotropic form (matches CPU viscoelastic.rs).
    // All other elastic models use directional F·(n₀⊗n₀)·Fᵀ (follows fiber deformation).
    if mat.active_stress_coeff > 0.0 && p.activation > 0.0 {
        if mat.model == 9u {
            tau = tau + (p.activation * mat.active_stress_coeff) * I;
        } else {
            let n  = p.activation_dir;
            let ls = dot(n, n);
            if ls > NUM_FLOOR {
                let n0      = n / sqrt(ls);
                let n_outer = mat2x2<f32>(n0 * n0.x, n0 * n0.y);
                tau = tau + F * ((p.activation * mat.active_stress_coeff) * n_outer) * transpose(F);
            } else {
                tau = tau + (p.activation * mat.active_stress_coeff) * I;
            }
        }
    }

    // Internal pre-stress (turgor-pressure-style, generic -- see Particle::internal_pressure
    // doc). Isotropic -P*I, gated on the same 3 models that override pressure_scale() on the
    // CPU side (NeoHookean=2, Corotated=3, Viscoelastic=9) -- pressure_scale() is a fixed
    // per-model constant (1.0 or 0.0), not a per-instance tunable, so no new MaterialParams
    // field is needed, same style as the mat.model==4u snow-cohesion check above.
    if p.internal_pressure != 0.0 && (mat.model == 2u || mat.model == 3u || mat.model == 9u) {
        tau = tau - p.internal_pressure * I;
    }
    return tau;
}

// stress_volume: fluids use initial_volume * J (= current volume, J from det(F)).
// J-based volume is consistent with the J-based EOS density above.
// Elastic models use initial (reference) volume -- J accounted for in Kirchhoff stress.
fn sv(p: Particle, mat: MaterialParams) -> f32 {
    switch mat.model {
        case 1u: {
            // J = det(F); F is reset to sqrt(J)·I in particles_update.
            let J = max(det2(p.deformation_gradient), NUM_FLOOR);
            return max(p.initial_volume * J, NUM_FLOOR);
        }
        case 11u: {
            // GranularFluid: EOS is density-based -- use current volume (tracks J each substep).
            return max(p.volume, NUM_FLOOR);
        }
        default: { return p.initial_volume; }
    }
}

// Exact f32 atomic add on the main grid (mass + momentum), via a
// compare-and-swap loop on the value's bit pattern -- WGSL has no float
// atomics. Replaces a fixed-point encoding (`round(val * SCALE)` into i32)
// that silently deleted every contribution smaller than half a quantum:
// measured, a GPU fluid left alone at dt=1e-4 could not even fall under
// gravity (per-substep momentum ~4e-6 < the 1e-5 quantum, rounded to 0),
// while the CPU twin, summing real floats, fell and stayed stable through
// impact. No single fixed scale can work: an i32 holds ~9 significant
// digits, but resolving a small-dt increment (~1e-8) while holding a heavy
// fast node's momentum (~300, e.g. mud at impact) needs ~4e10. Summing real
// floats gives the GPU the same arithmetic as the CPU path.
// `grid_clear` zeroes the buffer; integer 0 is also the bit pattern of 0.0.
fn atomic_add_f32_grid(idx: u32, val: f32) {
    var old = atomicLoad(&grid_atomic[idx]);
    loop {
        let swapped = atomicCompareExchangeWeak(
            &grid_atomic[idx],
            old,
            bitcast<i32>(bitcast<f32>(old) + val),
        );
        if swapped.exchanged { break; }
        old = swapped.old_value;
    }
}
fn grip_atomic_addf_mass(idx: u32, val: f32) {
    atomicAdd(&grip_grid_atomic[idx], i32(round(val * MASS_ATOMIC_SCALE)));
}
fn material_mass_atomic_addf(idx: u32, val: f32) {
    atomicAdd(&material_mass_atomic[idx], i32(round(val * MASS_ATOMIC_SCALE)));
}
fn grip_atomic_addf_mom(idx: u32, val: f32) {
    atomicAdd(&grip_grid_atomic[idx], i32(round(val * MOM_ATOMIC_SCALE)));
}

// Workgroup-local accumulation tile for the main-grid scatter (Gao et al.
// 2018, "GPU Optimization of Material Point Methods", SIGGRAPH Asia: sorted
// particles, shared-memory reduction before global atomics; `tmp/pbmpm`'s
// g2p2g does the same). Particles arrive block-sorted, so one workgroup's 64
// particles touch a small patch of nodes: summing there first (float CAS on
// shared memory, cheap) and flushing each touched node to the global grid
// ONCE per workgroup replaces ~27 contended global CAS loops per particle.
// Measured: the per-particle global CAS scatter took p2g from 111us
// (fixed-point atomicAdd) to 336us on the demo's 2912-particle dam break.
// Exactness is unchanged -- still real f32 sums, only the summation order
// differs (as with any atomic scatter). Nodes outside the tile (a workgroup
// straddling two distant blocks) fall back to the direct global add.
const TILE_DIM: i32 = 16;
const TILE_NODES: u32 = 256u; // TILE_DIM * TILE_DIM
// Measured and NOT kept (2026-09-19): splitting this into two banks by lane parity, to
// halve how many lanes CAS the same slot at once, changed nothing (168-184us vs
// 171-188us). The cost is the CAS round trips themselves, not the contention.
var<workgroup> tile_acc: array<atomic<i32>, 768>; // TILE_NODES * (mom.x, mom.y, mass)
var<workgroup> tile_origin: array<atomic<i32>, 2>;

fn tile_add(slot: u32, val: f32) {
    var old = atomicLoad(&tile_acc[slot]);
    loop {
        let swapped = atomicCompareExchangeWeak(
            &tile_acc[slot],
            old,
            bitcast<i32>(bitcast<f32>(old) + val),
        );
        if swapped.exchanged { break; }
        old = swapped.old_value;
    }
}

@compute @workgroup_size(64, 1, 1)
fn p2g_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    // No early return before the barriers below (they must be reached by every
    // invocation): out-of-range / NaN-position threads just skip the scatter.
    if lid == 0u {
        atomicStore(&tile_origin[0], 2147483647);
        atomicStore(&tile_origin[1], 2147483647);
    }
    for (var k = lid; k < TILE_NODES * 3u; k += 64u) {
        atomicStore(&tile_acc[k], 0);
    }
    let in_range = gid.x < step_params.particle_count;
    let p_idx = sorted_particle_ids[min(gid.x, step_params.particle_count - 1u)];
    let p = particles[p_idx];
    // NaN position would corrupt the grid sums -- skip silently.
    // Spare encoded substeps (the frame's time is already advanced) skip the scatter.
    // Not an early return: the barriers below must be reached by every invocation.
    let valid = in_range && dot(p.x, p.x) >= 0.0 && substep_dt() > 0.0;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));
    workgroupBarrier();
    if valid {
        atomicMin(&tile_origin[0], base.x - 1);
        atomicMin(&tile_origin[1], base.y - 1);
    }
    workgroupBarrier();
    let origin = vec2<i32>(atomicLoad(&tile_origin[0]), atomicLoad(&tile_origin[1]));
    if valid {
        scatter_particle(p, base, origin);
    }
    workgroupBarrier();
    let res = step_params.grid_res;
    if substep_dt() <= 0.0 {
        return;
    }
    for (var k = lid; k < TILE_NODES; k += 64u) {
        let mass_bits = atomicLoad(&tile_acc[k * 3u + 2u]);
        if mass_bits == 0 { continue; }
        let cx = origin.x + i32(k % u32(TILE_DIM));
        let cy = origin.y + i32(k / u32(TILE_DIM));
        let base4 = (u32(cy) * res + u32(cx)) * 4u;
        atomic_add_f32_grid(base4 + 0u, bitcast<f32>(atomicLoad(&tile_acc[k * 3u + 0u])));
        atomic_add_f32_grid(base4 + 1u, bitcast<f32>(atomicLoad(&tile_acc[k * 3u + 1u])));
        atomic_add_f32_grid(base4 + 2u, bitcast<f32>(mass_bits));
    }
}

fn scatter_particle(p: Particle, base: vec2<i32>, origin: vec2<i32>) {
    let res = step_params.grid_res;
    let dt  = substep_dt();
    let mat = materials[p.material_id];

    // Sleeping particles still scatter normally -- their mass+stress is exactly what
    // provides support to anything resting on top of them. Skipping P2G for sleeping
    // particles (an earlier version of this code did) makes them invisible to the grid:
    // an awake neighbor stacked on a sleeping one would suddenly find no support beneath
    // it, generating permanent unresolvable jitter at every awake/asleep boundary -- the
    // pile could never fully settle. Frozen (x, v, F) means the SAME scatter contribution
    // every substep, so this is deterministic, not wasted-but-harmless extra work: it's
    // the actual support mechanism. The real savings are in g2p/particles_update/
    // force_fields, which skip recomputing things that provably don't change for a
    // particle whose state is frozen -- not in skipping the scatter itself.

    let tau   = kirchhoff(p, mat);
    let vol   = sv(p, mat);
    let scale = -vol * step_params.kernel_d_inverse * dt;

    // Separable quadratic B-spline: only 3 distinct x-offsets and 3 distinct y-offsets
    // occur across the 9-cell neighborhood (di, dj each range over {-1,0,1}), so the
    // 1D weight only needs computing 3+3=6 times, not fresh for all 9 combinations
    // (18 calls) as a naive nested loop does. Matches the reference algorithm's own
    // technique -- Hu et al.'s mls-mpm88 (SIGGRAPH 2018) precomputes separable per-axis
    // weights the same way, cross-multiplying them per cell instead of recomputing the
    // 2D weight from scratch every iteration.
    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    var dx: array<f32, 3>;
    var dy: array<f32, 3>;
    for (var k: i32 = 0; k <= 2; k++) {
        let di = k - 1;
        dx[k] = f32(base.x + di) + CELL_CENTER_OFFSET - p.x.x;
        dy[k] = f32(base.y + di) + CELL_CENTER_OFFSET - p.x.y;
        wx[k] = bspline_w(dx[k]);
        wy[k] = bspline_w(dy[k]);
    }

    for (var ki: i32 = 0; ki <= 2; ki++) {
        let cx = base.x + ki - 1;
        if cx < 0 || cx >= i32(res) { continue; }
        for (var kj: i32 = 0; kj <= 2; kj++) {
            let cy = base.y + kj - 1;
            if cy < 0 || cy >= i32(res) { continue; }

            let cell_dist = vec2<f32>(dx[ki], dy[kj]);
            let w = wx[ki] * wy[kj];

            let apic_v    = p.v + p.velocity_gradient * cell_dist;
            let mass_w    = w * p.mass;
            let apic_mom  = mass_w * apic_v;
            let stress_mom = (scale * w) * (tau * cell_dist);

            let base4 = (u32(cy) * res + u32(cx)) * 4u;
            let local = vec2<i32>(cx, cy) - origin;
            if all(local >= vec2<i32>(0)) && all(local < vec2<i32>(TILE_DIM)) {
                let slot = u32(local.y * TILE_DIM + local.x) * 3u;
                tile_add(slot + 0u, apic_mom.x + stress_mom.x);
                tile_add(slot + 1u, apic_mom.y + stress_mom.y);
                tile_add(slot + 2u, mass_w);
            } else {
                atomic_add_f32_grid(base4 + 0u, apic_mom.x + stress_mom.x);
                atomic_add_f32_grid(base4 + 1u, apic_mom.y + stress_mom.y);
                atomic_add_f32_grid(base4 + 2u, mass_w);
            }

            // Multi-field contact (GPU port, first slice): additive second scatter for
            // the "grip" field (contact_group != 0), exactly mirroring the total-field
            // scatter above -- same weights, same stress/APIC contributions -- into the
            // separate grip_grid accumulator. No-op (branch not taken) for every
            // particle with contact_group == 0, matching CPU's zero-cost-when-unused
            // property (`scatter_particles_to_grid`'s own doc: "a no-op call for every
            // particle with contact_group == 0").
            if p.contact_group != 0u {
                grip_atomic_addf_mom(base4 + 0u, apic_mom.x + stress_mom.x);
                grip_atomic_addf_mom(base4 + 1u, apic_mom.y + stress_mom.y);
                grip_atomic_addf_mass(base4 + 2u, mass_w);
            }

            // `ColorMode::GridVolume`'s opt-in per-cell per-material mass scatter --
            // real, gated cost: skipped entirely (branch not taken) when disabled,
            // matching every other opt-in GPU subsystem's zero-cost-when-unused gate.
            if material_mass_params.enabled != 0u {
                let cell_idx = u32(cy) * res + u32(cx);
                let slot = p.material_id % MAX_RENDER_MATERIAL_SLOTS;
                material_mass_atomic_addf(cell_idx * MAX_RENDER_MATERIAL_SLOTS + slot, mass_w);
            }
        }
    }
}

// Multi-field contact (GPU port, first slice) -- mirrors CPU's `gather_contact_point_cloud`
// (transfer.rs): a SECOND per-particle pass, run AFTER p2g_main has fully scattered grip
// mass (wgpu inserts the necessary barrier between separate compute dispatches
// automatically, same guarantee particle_sort's own multi-pass sequence already relies
// on). For each of a particle's 9 stencil nodes, if that node's grip mass (just written
// by p2g_main) is nonzero, atomically claims a slot in that node's point-cloud bucket and
// records this particle's (position, label). Labeling and the "only where grip already
// registered" gating exactly match CPU's `add_contact_point`/`gather_contact_point_cloud`
// semantics -- see those functions' doc comments in `transfer.rs`/`grid/mod.rs` for the
// full rationale (this is what lets the LR normal fit ignore particles far from any real
// contact interface, not just cheaply skip the whole pass).
@compute @workgroup_size(64, 1, 1)
fn gather_contact_points_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= step_params.particle_count { return; }
    let p_idx = sorted_particle_ids[gid.x];
    let p = particles[p_idx];
    if !(dot(p.x, p.x) >= 0.0) { return; }

    let res = step_params.grid_res;
    let label = select(-1.0, 1.0, p.contact_group != 0u);

    // Real gate, not an optimization shortcut: only a particle whose OWN home cell
    // already has nonzero grip mass this substep is near a genuine contact interface
    // (matches CPU's `add_contact_point`, which only ever appends to an ALREADY-
    // existing `contact_cells` entry -- the CPU equivalent of "grip mass already
    // registered here"). Checking one representative cell (not all 9 stencil cells)
    // is deliberate: bucketing is per dedicated contact-block now (see
    // MAX_POINTS_PER_BLOCK's doc), and resolve_contact's gather_local_points scans a
    // node's own block PLUS its neighbors, so a particle recorded once in its own
    // block is already visible to every node that could plausibly need it -- recording
    // once per particle avoids redundantly writing the same particle into the same
    // block bucket up to 9 times. Note this partition is now finer than P2G's own
    // block_size (that's the whole point of the 2026-07-18 re-partition), so a
    // particle's 3×3 cell stencil CAN span multiple contact blocks -- still correct:
    // gather_local_points's own 3×3 NEIGHBOR-BLOCK scan is exactly what covers that.
    let home_x = clamp(u32(p.x.x), 0u, res - 1u);
    let home_y = clamp(u32(p.x.y), 0u, res - 1u);
    let home_idx = home_y * res + home_x;
    let grip_mass_bits = atomicLoad(&grip_grid_atomic[home_idx * 4u + 2u]);
    if grip_mass_bits <= 0 { return; }

    let block = contact_block_index(p.x, res);
    // NOTE for any future reader of `contact_point_counts`: this counter keeps
    // incrementing past MAX_POINTS_PER_BLOCK even though writes beyond it are dropped
    // below (a real, honest overflow signal, not silently capped) -- any consumer must
    // clamp its own iteration to `min(count, MAX_POINTS_PER_BLOCK)`, never trust the
    // raw count as the number of VALID slots in `contact_points`.
    let slot_in_block = atomicAdd(&contact_point_counts[block], 1u);
    if slot_in_block >= MAX_POINTS_PER_BLOCK { return; }
    let slot = block * MAX_POINTS_PER_BLOCK + slot_in_block;
    contact_points[slot] = vec4<f32>(p.x.x, p.x.y, label, 0.0);
}

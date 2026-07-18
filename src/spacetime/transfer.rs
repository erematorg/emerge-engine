use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{ConstitutiveModel, MaterialModel};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::{
    grid::Grid,
    grid::kernel::{axis_weights_derivative, quadratic_weights},
    particle::Particles,
};

/// Elastic/plastic Kirchhoff stress plus the active-stress (muscle contraction) term, if any.
///
/// KNOWN OPEN BUG (found 2026-07-11, still not fixed despite real, repeated effort): a driven
/// creature body settles into a real, unbounded compaction ratchet over long horizons — net
/// drift collapses to ~0 while min(J) keeps falling and never recovers. FIVE distinct real
/// fixes were tried and empirically falsified (each via a real 16,000-20,000-step headless
/// sweep on `basic_creature`'s exact Simulation/RatchetFrictionBoundary/NeoHookeanMaterial
/// setup, not guessed):
///   1. Higher material stiffness — only delays onset (6500 -> 13000 steps), same collapse.
///   2. Lower `apic_blend` (numerical PIC damping) — same, only delays onset.
///   3. Signed [-1,1] activation, naive `2*sigmoid-1` remap — WORSE: real instability
///      (min(J) toward the numerical floor, max(J) past 3.0), because it also doubled the
///      drive amplitude, not a clean test of signedness alone.
///   4. `NeoHookeanMaterial`'s volumetric Kirchhoff term was ALSO a real, separate,
///      independently-worth-fixing bug: it used a bounded `k/2*(J²-1)` (finite ceiling on
///      compression resistance) where Simo & Pister's actual 1984 formulation (which the
///      old doc already cited but didn't implement) uses the log-barrier `k*(ln J)²`
///      potential (τ_vol = k·ln(J), diverges as J→0, genuinely unbounded resistance). Fixed
///      in `kirchhoff_stress`/`kirchhoff_stress_vjp` below. Real, legitimate, kept -- but
///      verified NOT sufficient alone: the same 20,000-step creature sweep still stalls,
///      just with a somewhat different J-trajectory. `MIN_J` (1e-6) was checked and ruled
///      out as an interfering clamp -- min(J) in these runs never gets within three orders
///      of magnitude of it.
///   5. Signed activation retried with amplitude MATCHED to the unsigned case (span 0.9
///      either way, not doubled) -- still stalls (drift ~0 by step ~2500), though without
///      the earlier catastrophic collapse; max(J) still drifts upward over time (up to 3+).
///
/// Separately confirmed via a passive (zero-activation) body: min(J) settles to a FIXED
/// value and velocity decays cleanly to exactly 0 -- the core P2G/G2P/F-update solver is NOT
/// numerically drifting on its own. This is specific to the muscle-driven cyclic-loading +
/// directional-friction interaction, not a general integration artifact. Root cause remains
/// genuinely unsolved; a real fix likely needs rethinking the friction/actuation mechanism
/// itself (e.g. a redesigned contact model, or a controller that never enters the failure
/// regime) rather than another parameter or activation-scheme tweak.
///
/// Single source of truth for "what stress does this particle contribute to P2G" — shared by
/// `scatter_particles_to_grid` and tests, so the two can never drift apart. Mirrors the GPU
/// shader's post-switch active-stress block in `p2g.wgsl` exactly: Viscoelastic uses an
/// isotropic contractile term (matches its own Kelvin-Voigt formulation), every other elastic
/// model uses the directional F·(n₀⊗n₀)·Fᵀ fiber form (follows material deformation).
pub(crate) fn combined_kirchhoff_stress(
    material: &dyn MaterialModel,
    particles: &Particles,
    i: usize,
) -> Mat2 {
    let tau = material.kirchhoff_stress(particles, i);
    let coeff = material.activation_scale();
    if particles.activation[i] <= 0.0 || coeff <= 0.0 {
        return tau;
    }
    let isotropic = material.constitutive_model() == ConstitutiveModel::Viscoelastic;
    let tau_active = if isotropic {
        Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
    } else {
        let n = particles.activation_dir[i];
        let len_sq = n.dot(n);
        if len_sq > f32::EPSILON {
            let n0 = n / len_sq.sqrt();
            let n_outer = Mat2::from_cols(n0 * n0.x, n0 * n0.y);
            let a_mat = n_outer * (particles.activation[i] * coeff);
            let f = particles.deformation_gradient[i];
            f * a_mat * f.transpose()
        } else {
            Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
        }
    };
    tau + tau_active
}

/// Analytic adjoint of the DIRECTIONAL active-stress term
/// `combined_kirchhoff_stress` adds on top of a material's passive
/// `kirchhoff_stress` -- `tau_active = activation*coeff*F*A*Fᵀ`, where
/// `A = n0⊗n0` (fiber direction outer product, symmetric, fixed for a given
/// particle). Needed to train `activation` itself via gradient descent (the
/// actual trainable control signal for muscle-driven locomotion), not just
/// F -- `kirchhoff_stress_vjp` only covers the passive term.
///
/// SCOPED to the directional case only (every material except Viscoelastic's
/// isotropic branch) -- matches what a real trained creature body actually
/// uses (fiber-directed contraction); the isotropic branch is a much simpler
/// constant-diagonal term not needed here.
///
/// Since `A` is symmetric and `tau_active` is linear in `activation`, this
/// is the exact same `Y=F*A*Fᵀ` shape as `kirchhoff_stress_vjp`'s own `B=F*Fᵀ`
/// term (that's the `A=I` special case) -- so its adjoint follows the same
/// derivation: `dL/dF = (Ḡ+Ḡᵀ)*F*A` (using `A=Aᵀ` to combine the two `Y=F*A*Fᵀ`
/// product-rule terms into one). `dL/d(activation)` is just the Frobenius
/// inner product against `tau_active/activation` (linear in the scalar, so
/// its own derivative is that same fixed matrix): `dL/d(activation) = coeff *
/// (Ḡ : F*A*Fᵀ)`.
///
/// The SAME `d_loss_d_tau` gradient that feeds `kirchhoff_stress_vjp` feeds
/// this too -- `tau = tau_passive + tau_active` is a plain sum, whose adjoint
/// passes the incoming gradient through to BOTH summands unchanged, so a real
/// trainer calls both functions with the same `g` and adds their F-gradients.
///
/// Verified against central-difference numerical gradients in this module's
/// own tests.
pub fn active_stress_vjp(
    f: Mat2,
    activation: f32,
    coeff: f32,
    fiber_dir: Vec2,
    d_loss_d_tau: Mat2,
) -> (Mat2, f32) {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON || activation <= 0.0 || coeff <= 0.0 {
        return (Mat2::ZERO, 0.0);
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y);

    let g = d_loss_d_tau;
    let k_mat = f * a_mat * f.transpose(); // tau_active / (activation*coeff)
    let d_loss_d_activation = coeff
        * (g.x_axis.x * k_mat.x_axis.x
            + g.x_axis.y * k_mat.x_axis.y
            + g.y_axis.x * k_mat.y_axis.x
            + g.y_axis.y * k_mat.y_axis.y);

    let g_sym = g + g.transpose();
    let d_loss_d_f = (activation * coeff) * (g_sym * f * a_mat);

    (d_loss_d_f, d_loss_d_activation)
}

/// P2G: scatter particle mass, momentum, and stress forces onto the grid (MLS-MPM, Hu 2018 §4).
///
/// Stress is pre-integrated as a momentum impulse so the grid needs one accumulation pass.
/// The APIC affine term conserves angular momentum without a correction step.
///
/// NOT parallelized (unlike G2P below): multiple particles write to the same grid cell (3×3
/// B-spline stencils overlap), so summing their contributions requires either a shared mutable
/// map (unsound across threads — `HashMap::entry()` can trigger a resize) or a thread-local
/// fold/reduce merge. The latter was attempted and reverted 2026-06-20: it's safe and compiles
/// clean, but changes floating-point summation order across particles sharing a cell, and that
/// shifted results enough to break `fluid_spreads_more_than_elastic_under_gravity` (a 600-step
/// chaotic simulation) — confirmed by isolated A/B, not assumed. Reverted rather than accepted
/// the correctness risk for an unmeasured gain.
pub fn scatter_particles_to_grid(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
) {
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];
        let contact_group = particles.contact_group[i];
        let mixture_phase = material.mixture_phase();

        let stress = combined_kirchhoff_stress(material, particles, i);
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight
                    * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                grid.add_mass_momentum(cell_pos, weight * mass_i, momentum);
                // Additive second scatter for multi-field contact (Bardenhagen 2001) —
                // see `Particle::contact_group` doc. A no-op call for every particle
                // with contact_group == 0 (the default, i.e. every scene that doesn't
                // use this feature): `Grid::add_grip_mass_momentum` just never gets
                // called, so there's no extra work, not even an empty branch, for the
                // common case.
                if contact_group != 0 {
                    grid.add_grip_mass_momentum(cell_pos, weight * mass_i, momentum);
                }
                // Additive second scatter for two-phase mixture coupling (Tampubolon
                // et al. 2017) — see `WithMixturePhase`/`MixturePhase` doc. A no-op
                // for every particle whose material never opts in (the default),
                // same zero-cost-when-unused property as the contact scatter above.
                if let Some(phase) = mixture_phase {
                    grid.add_mixture_mass_momentum(cell_pos, phase, weight * mass_i, momentum);
                }
            }
        }
    }
}

/// Gathers the labeled particle point cloud (`+1.0` grip / `-1.0` rest) that
/// `Grid::resolve_contact`'s logistic-regression normal fit (`fit_contact_normal_lr`)
/// needs, at every node `scatter_particles_to_grid` already marked contact-active.
///
/// Deliberately a SECOND pass over particles, not merged into `scatter_particles_to_grid`
/// above: which nodes are contact-active isn't fully known until that first pass has
/// scattered every grip particle's mass, and `Grid::add_contact_point` only appends to a
/// node that already exists in `contact_cells` (never creates one) — so running this
/// before the first pass completes would silently miss point-cloud data for nodes whose
/// grip contribution hadn't been seen yet. Gated on `grid.has_contact_activity()`: a full
/// no-op, not even a loop iteration, for every scene that never sets
/// `Particle::contact_group` — the same zero-cost-when-unused property as the rest of
/// this feature.
pub fn gather_contact_point_cloud(particles: &Particles, grid: &mut Grid, active_count: usize) {
    if !grid.has_contact_activity() {
        return;
    }
    for i in 0..active_count {
        let x = particles.x[i];
        let label = if particles.contact_group[i] != 0 {
            1.0
        } else {
            -1.0
        };
        let weights = quadratic_weights(x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx - 1, gy - 1);
                grid.add_contact_point(cell_pos, x, label);
            }
        }
    }
}

/// Analytic adjoint of P2G's stress→force scatter contribution w.r.t. the
/// particle's own Kirchhoff stress tensor -- the second real piece of
/// differentiable stepping, after `NeoHookeanMaterial::kirchhoff_stress_vjp`.
///
/// SCOPED, not a full P2G adjoint: differentiates only the elastic-force term
/// `weight * stress_coeff * (stress * cell_dist)` inside `scatter_particles_to_grid`,
/// treating the particle's position `x` (and therefore the kernel weights and
/// `cell_dist`) as FIXED. The mass/velocity/affine-C term is untouched here --
/// a separate, much simpler linear adjoint, not yet implemented. Differentiating
/// through the kernel weights' own dependence on `x` (how MOVING the particle
/// changes which cells it deposits to, and by how much) is the real remaining
/// gap in a fully general P2G adjoint -- deliberately deferred, not silently
/// dropped: this covers the actual control-relevant path (muscle activation →
/// stress → grid force) needed to train a controller, without yet handling
/// the harder position-dependence.
///
/// Real derivation: for one particle, cell `c`'s momentum contribution from
/// stress is `y_c = (weight_c * stress_coeff) * (stress * cell_dist_c)` --
/// linear in `stress`, a matrix-vector product `y = M*v` scaled by a fixed
/// scalar. Given the gradient flowing back from each cell's grid momentum,
/// `d_loss_d_momentum[c]` (a Vec2), the standard VJP for `y=Mv` is
/// `dL/dM = outer(dL/dy, v)`, i.e. `dL/dM_kl = dL/dy_k * v_l`. Summed over
/// all 9 stencil cells:
///
///   d_loss_d_stress = sum_c (weight_c * stress_coeff) * outer(d_loss_d_momentum[c], cell_dist_c)
///
/// Returns d_loss_d_stress, ready to feed into e.g.
/// `NeoHookeanMaterial::kirchhoff_stress_vjp` to continue the chain back to F.
/// Verified against central-difference numerical gradients in this module's
/// own tests, same non-negotiable discipline as the stress adjoint itself.
pub fn p2g_stress_vjp(x: Vec2, stress_coeff: f32, d_loss_d_momentum: &[[Vec2; 3]; 3]) -> Mat2 {
    let weights = quadratic_weights(x);
    let mut d_loss_d_stress = Mat2::ZERO;
    for (gx, (wx, momentum_row)) in weights.wx.iter().zip(d_loss_d_momentum.iter()).enumerate() {
        for (gy, (wy, &g)) in weights.wy.iter().zip(momentum_row.iter()).enumerate() {
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let scalar = weight * stress_coeff;
            // outer(g, cell_dist): column 0 = cell_dist.x * g, column 1 = cell_dist.y * g
            // (matches glam's column-major Mat2, verified against the matrix-vector
            // VJP already proven correct in kirchhoff_stress_vjp).
            d_loss_d_stress += scalar * Mat2::from_cols(cell_dist.x * g, cell_dist.y * g);
        }
    }
    d_loss_d_stress
}

/// Analytic adjoint of P2G's FULL forward pass (`scatter_particles_to_grid`)
/// w.r.t. the particle's own position `x` -- the last confirmed-real gap,
/// now closed for P2G. Combines `axis_weights_derivative` (the kernel's own
/// position-sensitivity) with the product rule across the complete momentum
/// AND mass scatter (not just the stress term `p2g_stress_vjp` covers).
///
/// Forward, restated from `scatter_particles_to_grid`: per cell `c`,
///   mass_contrib_c     = weight_c * mass
///   momentum_contrib_c = weight_c * A_c,  A_c = mass*v + M*cell_dist_c
///   M = mass*C + stress_coeff*stress   (constant across cells, for fixed particle state)
///
/// BOTH `weight_c(x)` and `cell_dist_c(x) = cell_pos_c - x + 0.5` depend on
/// `x` (`d(cell_dist)/dx = -I`), so differentiating the product `weight * A`
/// needs the product rule on both factors. Per cell, given the gradients
/// flowing back from that cell's momentum and mass, `d_loss_d_momentum[c]`
/// (Vec2) and `d_loss_d_mass[c]` (f32):
///
///   d_loss_d_x += d(weight_c)/dx * (d_loss_d_momentum[c].A_c + d_loss_d_mass[c]*mass)
///               - weight_c * (Mᵀ * d_loss_d_momentum[c])
///
/// where `d(weight_c)/dx = (dwx[gx]/dx.x * wy[gy], wx[gx] * dwy[gy]/dx.y)`
/// via `axis_weights_derivative`, and the `-weight_c * Mᵀ*d_loss_d_momentum`
/// term comes from `d(A_c)/dx = M * d(cell_dist_c)/dx = -M`.
///
/// Verified against central-difference numerical gradients taken through a
/// forward function reconstructing `scatter_particles_to_grid`'s exact
/// per-cell formula, in this module's own tests.
///
/// Bundles the particle state P2G itself reads (`mass`, `v`, `C`, `stress`,
/// `stress_coeff`) into one struct rather than five separate parameters --
/// this function differentiates the FULL forward pass, so it genuinely needs
/// all of it, but five-plus-position-plus-two-gradient-array parameters
/// crossed the project's own no-`#[allow]` line for argument count.
pub struct P2GParticleState {
    pub mass: f32,
    pub v: Vec2,
    pub c: Mat2,
    pub stress: Mat2,
    pub stress_coeff: f32,
}

pub fn p2g_position_vjp(
    x: Vec2,
    state: &P2GParticleState,
    d_loss_d_momentum: &[[Vec2; 3]; 3],
    d_loss_d_mass: &[[f32; 3]; 3],
) -> Vec2 {
    let weights = quadratic_weights(x);
    let diff = x - weights.base_cell.as_vec2() - Vec2::splat(0.5);
    let dwx = axis_weights_derivative(diff.x);
    let dwy = axis_weights_derivative(diff.y);
    let m = state.mass * state.c + state.stress_coeff * state.stress;

    let mut d_loss_d_x = Vec2::ZERO;
    for gx in 0..3 {
        for gy in 0..3 {
            let wx = weights.wx[gx];
            let wy = weights.wy[gy];
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let a = state.mass * state.v + m * cell_dist;

            let d_weight_dx = Vec2::new(dwx[gx] * wy, wx * dwy[gy]);
            let g_momentum = d_loss_d_momentum[gx][gy];
            let g_mass = d_loss_d_mass[gx][gy];

            d_loss_d_x += d_weight_dx * (g_momentum.dot(a) + g_mass * state.mass);
            d_loss_d_x -= weight * (m.transpose() * g_momentum);
        }
    }
    d_loss_d_x
}

pub struct G2PParams<'a> {
    pub vel_limit: f32,
    pub apic_blend: f32,
    pub active_count: usize,
    /// ASFLIP blend factor (`SimConfig::asflip_blend`, Fei et al. 2021). 0.0 = disabled,
    /// the exact original G2P formula below (see `pre_force_snapshot`'s doc for the gate).
    pub asflip_blend: f32,
    /// The grid's pre-force velocity snapshot (see `Grid::snapshot_velocities`), or `None`
    /// when ASFLIP is disabled. This, not `asflip_blend` alone, is the real gate: the ASFLIP
    /// correction below only runs when `Some`, so a caller that never opts in (passes `None`)
    /// gets the byte-identical original code path regardless of what `asflip_blend` holds.
    pub pre_force_snapshot: Option<&'a crate::grid::VelocitySnapshot>,
}

/// Analytic adjoint of G2P's velocity gather (`new_v = sum_c weight_c *
/// grid.velocity_at(cell_c)`, see `gather_grid_to_particles`'s Phase 1) w.r.t.
/// the 9 grid velocities in the particle's stencil -- fifth real piece of
/// differentiable stepping, and the mathematical transpose of
/// `p2g_stress_vjp`: same quadratic kernel weights, same 3x3 stencil, but
/// scattering a gradient back out to the grid instead of gathering a value in
/// from it (the well-known P2G/G2P transpose relationship in MPM literature,
/// e.g. Jiang et al. 2016 "The Material Point Method for Simulating
/// Continuum Materials", carries over directly to differentiation).
///
/// SCOPED, matching the P2G adjoint's own scoping: treats particle position
/// `x` (and therefore the kernel weights) as FIXED. Also covers only the new
/// velocity `new_v`, not the APIC affine matrix `b`/`velocity_gradient` G2P
/// computes alongside it (`b = sum_c weight_c * outer(v_grid_c, dist_c)`) --
/// a related, still-open piece: same per-cell structure, needs its own
/// derivation and verification, not silently folded in here. Also doesn't
/// cover the velocity clamp or position boundary-clamp applied after this in
/// the real G2P (piecewise/conditional, same deferred-with-a-name status as
/// grid update's boundary/clamp gap).
///
/// Given the gradient flowing back from the particle's new velocity,
/// `d_loss_d_new_v` (a Vec2), the adjoint of a weighted sum distributes it
/// back to each grid cell by the SAME weight it was gathered with:
///
///   d_loss_d_v_grid[c] = weight_c * d_loss_d_new_v
///
/// Returns the per-cell gradient in the same `[[Vec2; 3]; 3]` shape
/// `p2g_stress_vjp` consumes, so a real trainer can pass this straight
/// through to the P2G side once both meet at the same grid cells. Verified
/// against central-difference numerical gradients in this module's own
/// tests.
pub fn g2p_velocity_vjp(x: Vec2, d_loss_d_new_v: Vec2) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (row, wx) in out.iter_mut().zip(weights.wx.iter()) {
        for (cell, wy) in row.iter_mut().zip(weights.wy.iter()) {
            *cell = (wx * wy) * d_loss_d_new_v;
        }
    }
    out
}

/// Analytic adjoint of G2P's APIC affine matrix (`velocity_gradient`)
/// computation w.r.t. the 9 grid velocities -- the piece `g2p_velocity_vjp`
/// deliberately left open, now closed. Real, externally cross-checked: this
/// exact term appears in ChainQueen's own hand-written CUDA backward pass
/// (`backward.cu`, `P2G_backward`'s "(C)" comment) as
/// `invD * N * grad_C_next[alpha][beta] * dpos[beta]` -- confirms both that
/// this term is genuinely needed (not paranoia) and, since it algebraically
/// matches the independently-derived formula below once ChainQueen's `invD`
/// is read as this codebase's `KERNEL_D_INVERSE`, that the derivation is
/// right. `apic_blend` is an emerge-specific extra factor ChainQueen's own
/// formula doesn't have (see `gather_grid_to_particles`'s `vg = b *
/// KERNEL_D_INVERSE * apic_blend`), included here since it's part of
/// emerge's own forward formula.
///
/// Forward (see `gather_grid_to_particles`'s Phase 1): `new_c = scale *
/// sum_c weight_c * outer(v_grid_c, dist_c)`, where `scale =
/// KERNEL_D_INVERSE * apic_blend` and `outer(v,d)` has column 0 = `d.x*v`,
/// column 1 = `d.y*v` (same convention as `p2g_stress_vjp`'s own outer
/// product). Linear in each `v_grid_c`; given the gradient flowing back from
/// the affine matrix, `d_loss_d_new_c` (a Mat2), the VJP of `outer(v,d)`
/// w.r.t. `v` is `M*d` (matrix-vector product, standard result for an outer
/// product's adjoint):
///
///   d_loss_d_v_grid[c] = weight_c * scale * (d_loss_d_new_c * dist_c)
///
/// Callers combine this additively with `g2p_velocity_vjp`'s output (both
/// scatter to the SAME 9 grid cells, since `new_v` and `new_c` are computed
/// from the same stencil in the same G2P pass) to get the true total
/// per-cell gradient. Verified against central-difference numerical
/// gradients in this module's own tests, independently and composed with
/// `g2p_velocity_vjp`.
pub fn g2p_affine_vjp(
    x: Vec2,
    kernel_d_inverse: f32,
    apic_blend: f32,
    d_loss_d_new_c: Mat2,
) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let scale = kernel_d_inverse * apic_blend;
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (gx, (row, wx)) in out.iter_mut().zip(weights.wx.iter()).enumerate() {
        for (gy, (cell, wy)) in row.iter_mut().zip(weights.wy.iter()).enumerate() {
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            *cell = (wx * wy * scale) * (d_loss_d_new_c * dist);
        }
    }
    out
}

/// Analytic adjoint of the deformation-gradient update `F_new = (I + dt*C) *
/// F_old` w.r.t. both `C` (the APIC affine matrix / velocity_gradient G2P
/// produces) and `F_old` -- sixth real piece of differentiable stepping, and
/// the one that actually CLOSES the loop: `C` comes from G2P, `F_old` is the
/// previous substep's deformation gradient, and this update's own output
/// (`F_new`) is exactly what `kirchhoff_stress_vjp` needs as input for the
/// NEXT substep. Chaining this repeatedly is what backprop-through-multiple-
/// substeps actually means.
///
/// This exact formula is universal MPM kinematics, not any one material's own
/// logic -- confirmed by grep: every material in `matter::materials`
/// (NeoHookean, Corotated, Viscoelastic, and every plastic model's F_trial
/// before its own return-mapping) computes `F_new`/`F_trial` this identical
/// way. Lives here in `spacetime::transfer`, not any material file, for that
/// reason.
///
/// Derivation: let `A = I + dt*C`, so `F_new = A * F_old` -- a plain matrix
/// product. Standard VJP for `Y = A*B`: `dL/dA = Ḡ*Bᵀ`, `dL/dB = Aᵀ*Ḡ`. Since
/// `A` is linear in `C` (`dA/dC = dt` component-wise), `dL/dC = dt * dL/dA`:
///
///   d_loss_d_C     = dt * (d_loss_d_F_new * F_oldᵀ)
///   d_loss_d_F_old = (I + dt*C)ᵀ * d_loss_d_F_new
///
/// Verified against central-difference numerical gradients in this module's
/// own tests, on both outputs independently.
pub fn f_update_vjp(c: Mat2, f_old: Mat2, dt: f32, d_loss_d_f_new: Mat2) -> (Mat2, Mat2) {
    let a = Mat2::IDENTITY + dt * c;
    let d_loss_d_c = dt * (d_loss_d_f_new * f_old.transpose());
    let d_loss_d_f_old = a.transpose() * d_loss_d_f_new;
    (d_loss_d_c, d_loss_d_f_old)
}

/// G2P: read grid velocities back into particles, advance state, apply boundaries.
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    params: G2PParams,
) -> usize {
    let G2PParams {
        vel_limit,
        apic_blend,
        active_count,
        asflip_blend,
        pre_force_snapshot,
    } = params;
    let grid_res = grid.resolution();

    // Phase 1 (parallel): grid gather -> v, velocity_gradient, position advance + boundary
    // position clamp. Pure math over read-only grid/boundary state, writing only the calling
    // particle's own x/v/velocity_gradient — no cross-particle data dependency, so disjoint
    // per-field slices can be processed concurrently (gather passes are race-free by
    // construction; see Gao et al. 2018, "GPU Optimization of Material Point Methods").
    let xs = &mut particles.x[..active_count];
    let vs = &mut particles.v[..active_count];
    let vgs = &mut particles.velocity_gradient[..active_count];
    let contact_groups = &particles.contact_group[..active_count];
    let pinned_flags = &particles.pinned[..active_count];
    let material_ids = &particles.material_id[..active_count];
    // Gate once, not per particle: when no grip particle ever touched the grid this
    // substep (every scene that doesn't use `Particle::contact_group`), this is false
    // and the loop below takes the exact same path it always has — a plain
    // `grid.velocity_at` lookup, no extra branching cost worth measuring.
    let contact_active = grid.has_contact_activity();
    // Same gate for two-phase mixture coupling (Tampubolon et al. 2017) — see
    // `WithMixturePhase` doc. False (the default) for every scene that never
    // wraps a material this way, same zero-cost property as contact above.
    let mixture_active = grid.has_mixture_activity();

    let clamp_count: usize = xs
        .par_iter_mut()
        .zip(vs.par_iter_mut())
        .zip(vgs.par_iter_mut())
        .zip(contact_groups.par_iter())
        .zip(pinned_flags.par_iter())
        .zip(material_ids.par_iter())
        .map(
            |(((((x, v), vg), &contact_group), &pinned), &material_id)| {
                let mixture_phase = if mixture_active {
                    materials.get(material_id).mixture_phase()
                } else {
                    None
                };
                let v_old = *v;
                let weights = quadratic_weights(*x);
                let mut new_v = Vec2::ZERO;
                let mut b = Mat2::ZERO;

                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights.wx[gx] * weights.wy[gy];
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        let dist = cell_pos.as_vec2() - *x + Vec2::splat(0.5);
                        // Multi-field contact routing (Bardenhagen 2001): a grip particle
                        // reads the resolved grip field, a non-grip particle reads the
                        // resolved rest field, at nodes where contact was ever registered
                        // this substep. Both helpers fall back to the ordinary total
                        // velocity where no contact exists at that node, so this is exact
                        // everywhere, not just near contact.
                        let node_v = if contact_active {
                            if contact_group != 0 {
                                grid.grip_velocity_at(cell_pos)
                            } else {
                                grid.rest_velocity_at(cell_pos)
                            }
                        } else if let Some(phase) = mixture_phase {
                            // Two-phase mixture coupling routing (Tampubolon et al. 2017):
                            // a solid-phase particle reads the resolved solid field, a
                            // fluid-phase particle reads the resolved fluid field — both
                            // fall back to the ordinary total velocity where no coupling
                            // was registered at that node, same convention as contact.
                            use crate::materials::MixturePhase;
                            match phase {
                                MixturePhase::Solid => grid.resolved_solid_velocity_at(cell_pos),
                                MixturePhase::Fluid => grid.resolved_fluid_velocity_at(cell_pos),
                            }
                        } else {
                            grid.velocity_at(cell_pos)
                        };
                        let weighted_velocity = node_v * weight;
                        let term =
                            Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                        b += term;
                        new_v += weighted_velocity;
                    }
                }

                // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
                // velocity_gradient=0 instead of gathering from the grid, so a pinned
                // particle never moves and never accumulates local strain from being
                // dragged — while its own mass/stress still scattered into P2G normally,
                // so it acts as a real, immovable anchor other bodies push against (the
                // standard technique for static/bedrock geometry in deformable-body sims).
                // Checked before the speed cap/position advance so a pinned particle takes
                // neither — position is deliberately left completely untouched, not just
                // re-clamped to itself, avoiding any float drift from a v=0*dt add-then-
                // reclamp round trip.
                if pinned != 0 {
                    *v = Vec2::ZERO;
                    *vg = Mat2::ZERO;
                    return 0;
                }

                // ASFLIP (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the
                // Material Point Method" -- see `SimConfig::asflip_blend` doc). Reintroduces
                // the classic FLIP residual (`v_p_old - old_v`) on top of the PIC/APIC gather
                // above -- `old_v` is a PIC-style gather against the grid's PRE-FORCE velocity
                // (`pre_force_snapshot`, taken right after P2G's own momentum normalization,
                // before this substep's gravity/boundary/contact modified it), using the SAME
                // stencil weights as `new_v` above. `pre_force_snapshot` being `None` (the
                // default, `asflip_blend=0.0`) is the real gate: `v_store`/`v_position` both
                // stay exactly `new_v`, reproducing the original formula below bit-for-bit.
                //
                // `gamma` (position-correction strength) is 0 while the local velocity
                // gradient indicates compression (`trace(b) < 0` -- two bodies pressing
                // together, e.g. a creature pushing into terrain via multi-field contact, or
                // material pressing against a boundary, since boundary conditions are already
                // baked into `new_v`/`b` by the time G2P reads the grid) and 1 while
                // separating -- exactly the paper's own "easier separation" adaptivity,
                // avoiding injecting extra positional noise while two bodies are in contact.
                let (mut v_store, mut v_position) = (new_v, new_v);
                if let Some(snapshot) = pre_force_snapshot {
                    let mut old_v = Vec2::ZERO;
                    for gx in 0..3 {
                        for gy in 0..3 {
                            let weight = weights.wx[gx] * weights.wy[gy];
                            let cell_pos =
                                weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                            old_v += grid.pre_force_velocity_at(snapshot, cell_pos) * weight;
                        }
                    }
                    let diff_vel = v_old - old_v;
                    let trace_b = b.x_axis.x + b.y_axis.y;
                    let gamma = if trace_b < 0.0 { 0.0 } else { 1.0 };
                    v_store = new_v + asflip_blend * diff_vel;
                    v_position = new_v + gamma * asflip_blend * diff_vel;
                }

                // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
                // This fires only when CFL is violated despite the timestep limiter (e.g. first
                // substep of a high-energy spawn). Magnitude clamp preserves direction; no
                // anisotropic bias unlike per-component clamping. Clamps both `v_store` and
                // `v_position` by the SAME safety ratio (derived from the stored velocity's own
                // magnitude) so they stay mutually consistent -- when ASFLIP is disabled the two
                // are identical (`v_store == v_position == new_v`), so this is byte-identical to
                // the original single-velocity clamp.
                let spd = v_store.length();
                let clamped = if spd > vel_limit {
                    let scale = vel_limit / spd;
                    v_store *= scale;
                    v_position *= scale;
                    1
                } else {
                    0
                };

                // Apply all boundaries' position clamp (pure function, no particle-struct access).
                let mut new_pos = *x + v_position * dt;
                for boundary in boundaries.iter() {
                    new_pos = boundary.clamp_particle_position(new_pos, grid_res);
                }

                *v = v_store;
                *vg = b * KERNEL_D_INVERSE * apic_blend;
                *x = new_pos;
                clamped
            },
        )
        .sum();

    // Phase 2 (sequential): plasticity update + boundary post-hooks need whole-`Particles`
    // mutable access (deformation_gradient, hardening_scale, etc. per material) — not
    // split-borrow-friendly without a larger `MaterialModel` trait redesign, so kept sequential.
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        material.update_particle(particles, i, dt);
        for boundary in boundaries.iter() {
            boundary.post_g2p_particle(particles, i, grid_res, dt);
        }
    }

    clamp_count
}

pub fn scatter_particle_mass(particles: &Particles, grid: &mut Grid, active_count: usize) {
    for i in 0..active_count {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * mass, Vec2::ZERO);
            }
        }
    }
}

// Test suite split into its own file -- was ~1280 of this file's ~1950 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`. Pure mechanical
// line-range extraction, see that file's own doc comment.
#[cfg(test)]
mod transfer_tests;

use glam::{Mat2, Vec2};

use crate::materials::{ConstitutiveModel, MaterialModel};
use crate::particle::Particles;
// Only needed for the test submodules' own use via `super::*` (matches
// these items' role in the original single-file layout) -- p2g.rs/g2p.rs
// each import what their own production code needs directly.
#[cfg(test)]
use crate::grid::Grid;
#[cfg(test)]
use crate::grid::kernel::quadratic_weights;
#[cfg(test)]
use crate::solver::config::KERNEL_D_INVERSE;
#[cfg(test)]
use glam::IVec2;

mod g2p;
mod p2g;

pub use g2p::{
    G2PParams, f_update_vjp, g2p_affine_vjp, g2p_velocity_vjp, gather_grid_to_particles,
};
pub use p2g::{
    P2GParticleState, gather_contact_point_cloud, p2g_position_vjp, p2g_stress_vjp,
    scatter_particle_mass, scatter_particles_to_grid,
};

// The two test modules' `use super::*;` see every item re-exported above
// (P2G/G2P functions + structs) plus `combined_kirchhoff_stress`/
// `active_stress_vjp` defined directly below, exactly as they did in the
// single-file layout -- no separate re-import needed.
#[cfg(test)]
mod g2p_tests;
#[cfg(test)]
mod p2g_tests;

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

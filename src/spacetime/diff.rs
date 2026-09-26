//! Differentiable mini-solver for offline gait training.
//!
//! A self-contained, differentiable MLS-MPM forward simulation plus its
//! hand-derived reverse pass -- the trainer the whole adjoint chain in
//! `spacetime::transfer` was built toward. Structured to match the canonical
//! open-loop DiffTaichi `diffmpm.py` walker demo (verified against the real
//! cloned source, not from memory), because that is the simplest published
//! setup proven to produce visible trained locomotion:
//!
//! - **Time-varying actuation from a sinusoid basis controller** -- the
//!   trainable parameters are per-muscle-group weights over `n_waves` phase-
//!   shifted sinusoids plus a bias, squashed with tanh. Constant per-particle
//!   activation (the first prototype here) can only learn a static squeeze;
//!   a time-varying signal learns a *gait*.
//! - **Signed actuation** (`tanh` in (-1,1)): muscles both contract and
//!   extend, exactly DiffTaichi's convention (`A = [[0,0],[0,1]] * act`, both
//!   signs). NOTE: the engine's runtime muscle model
//!   (`transfer::combined_kirchhoff_stress`) is contract-only `[0,1]` -- a
//!   trained gait transfers to the runtime by remapping, this module does not
//!   change engine semantics.
//! - **Gravity + a sticky floor** as the locomotion symmetry-breaker.
//!   Verified detail from the real `diffmpm.py` source: its friction-cone
//!   code runs on an already-zeroed velocity, so the canonical walker
//!   actually trains against a *sticky* floor (grid cells at floor level
//!   moving downward get zeroed) -- which is exactly what this module
//!   implements, with the branch decision (stick vs. not) recorded forward
//!   and replayed as a fixed linear map backward -- a genuinely universal
//!   non-differentiability (a hard `if`), the same treatment any
//!   differentiable contact simulator has to make somewhere.
//! - **Actuator groups**: particles share muscle groups (legs), not one
//!   trainable scalar per particle.
//!
//! Every backward formula is either one of the individually finite-difference-
//! verified adjoints from `spacetime::transfer`/`grid`, or is derived and
//! FD-verified in this module's own tests. One deliberate scope limit, UNIQUE
//! to this module (not shared with the real runtime chain): `spacetime::
//! transfer::p2g_position_vjp` differentiates the full kernel-weight-value
//! dependence on position (`axis_weights_derivative`, matches finite
//! difference exactly -- verified by reading that function directly), but
//! THIS module's own simpler backward re-derivation does not -- it evaluates
//! each step's weights at the recorded forward position and does not backprop
//! through them. Real, disclosed limitation, not a shared community
//! convention: a prior version of this doc claimed ChainQueen/DiffTaichi made
//! the same simplification; direct inspection of both (`tmp/ChainQueen/src/
//! backward.cu`'s `G2P_backward`, which explicitly computes and sums `dw()`
//! terms into the position gradient, and `tmp/difftaichi/examples/
//! diffmpm.py`'s autodiff-generated backward, which has no `stop_grad` on
//! `p2g`/`g2p` and therefore differentiates through the weight automatically)
//! showed that claim was false -- both canonical implementations DO
//! differentiate through the weight. This module's omission is a bespoke
//! shortcut for a small, short-horizon offline training tool (see the
//! `controller_gradient_matches_finite_difference_smooth_regime` test's
//! measured ~5-7% gap this causes), not precedent-backed.
//!
//! Scale/units note: this is a *training tool*, not the runtime solver. It
//! runs a small body (tens of particles) for a short horizon (~100 substeps)
//! thousands of times; the trained controller parameters are the output.
//!
//! Split into submodules 2026-07-19 (was 1632 lines in one file) by pipeline
//! phase -- `body_plan`/`config`/`stress` are shared building blocks,
//! `forward`/`backward` mirror the real solver's own P2G/G2P phase split,
//! `metrics`/`train` are the outer training-loop layer. Every item that was
//! `pub` at this module's top level before the split is still re-exported
//! here at the exact same path, so nothing outside `diff` observes a
//! difference. `backward.rs` deliberately stayed ONE file rather than
//! splitting sinusoid/feedback backprop apart: `IncomingGrad`/`OutgoingGrad`/
//! `SubstepCtx`/`GradSeed` are shared infrastructure both paths lean on, and
//! separating them would mean duplicating that plumbing for no real gain.

use glam::Mat2;

use crate::materials::{MaterialModel, NeoHookeanMaterial};
use crate::particle::Particles;

mod backward;
mod body_plan;
mod config;
mod forward;
mod metrics;
mod stress;
mod train;

pub use backward::{
    controller_gradient, controller_gradient_seeded, feedback_controller_gradient,
    feedback_controller_gradient_seeded,
};
pub use body_plan::BodyPlan;
pub use config::{DiffConfig, DiffState, FeedbackController, SinusoidController, StepRecord};
pub use forward::{forward_substep, rollout, rollout_feedback};
pub use metrics::{GaitMetrics, drift, drift_feedback, gait_metrics, gait_metrics_feedback};
pub use stress::StressEval;
pub use train::{train, train_feedback};

// Private re-imports so `#[cfg(test)] mod tests`'s `use super::*;` keeps
// seeing these (they were plain private items in the single-file layout,
// visible to a child `tests` module automatically; now they live one level
// deeper in a sibling submodule, so they need pulling back into this
// module's own namespace -- privacy unchanged, still invisible outside
// `diff`, just re-routed through here for the same reason a private `use`
// in a parent module is visible to its child modules).
#[cfg(test)]
use backward::backprop_through_time;
#[cfg(test)]
use glam::Vec2;
#[cfg(test)]
use stress::{signed_active_stress, signed_active_stress_vjp};

// ── Differentiable materials ──────────────────────────────────────────────────

/// A material whose passive Kirchhoff stress has a known analytic adjoint --
/// what makes it usable inside this trainer. Everything in this module was
/// hardcoded to `NeoHookeanMaterial` specifically until this generalization
/// (requested explicitly: emerge's whole design is one solver for all
/// matter, and the trainer shouldn't be the one place that's tied to a
/// single constitutive model). `NeoHookeanMaterial` is the only
/// implementation today; `CorotatedMaterial` is the concrete next target --
/// ChainQueen's real `Times_Rotated_dP_dF_FixedCorotated` (its own hand-
/// written CUDA backward pass, `linalg.h`) gives the reference formula for
/// its polar-decomposition-based stress, but deriving emerge's actual
/// `kirchhoff_stress = P*F^T` adjoint from it needs an extra product-rule
/// step (P depends on F, AND there's an explicit trailing F^T) that hasn't
/// been carefully derived+FD-verified yet -- real remaining work, not
/// silently skipped.
pub trait DifferentiableMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2;
    fn kirchhoff_stress_vjp(&self, particles: &Particles, i: usize, d_loss_d_tau: Mat2) -> Mat2;
}

impl DifferentiableMaterial for NeoHookeanMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        MaterialModel::kirchhoff_stress(self, particles, i)
    }
    fn kirchhoff_stress_vjp(&self, particles: &Particles, i: usize, d_loss_d_tau: Mat2) -> Mat2 {
        NeoHookeanMaterial::kirchhoff_stress_vjp(self, particles, i, d_loss_d_tau)
    }
}

// Test suite split into its own file -- was ~900 of this file's ~2530 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`. Pure mechanical
// line-range extraction, see that file's own doc comment.
#[cfg(test)]
mod tests;

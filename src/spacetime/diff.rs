//! Differentiable mini-solver for offline gait training.
//!
//! A self-contained, differentiable MLS-MPM forward simulation plus its
//! hand-derived reverse pass, built on the adjoint chain in
//! `spacetime::transfer`. Structured like DiffTaichi's open-loop
//! `diffmpm.py` walker demo, the simplest published setup shown to produce
//! visible trained locomotion:
//!
//! - **Time-varying actuation from a sinusoid basis controller** -- the
//!   trainable parameters are per-muscle-group weights over `n_waves` phase-
//!   shifted sinusoids plus a bias, squashed with tanh. Constant
//!   per-particle activation can only learn a static squeeze; a time-varying
//!   signal learns a *gait*.
//! - **Signed actuation** (`tanh` in (-1,1)): muscles both contract and
//!   extend, DiffTaichi's convention (`A = [[0,0],[0,1]] * act`, both
//!   signs). The engine's runtime muscle model
//!   (`transfer::combined_kirchhoff_stress`) is contract-only `[0,1]`: a
//!   trained gait transfers to the runtime by remapping; this module does
//!   not change engine semantics.
//! - **Gravity + a sticky floor** as the locomotion symmetry-breaker. In
//!   `diffmpm.py` the friction-cone code runs on an already-zeroed velocity,
//!   so the walker trains against a *sticky* floor (grid cells at floor
//!   level moving downward are zeroed). This module does the same, recording
//!   the stick/no-stick branch forward and replaying it as a fixed linear
//!   map backward: a hard `if`, the non-differentiability any differentiable
//!   contact simulator has to handle somewhere.
//! - **Actuator groups**: particles share muscle groups (legs), not one
//!   trainable scalar per particle.
//!
//! Every backward formula is either one of the finite-difference-verified
//! adjoints from `spacetime::transfer`/`grid`, or derived and FD-verified in
//! this module's tests. One scope limit specific to this module:
//! `spacetime::transfer::p2g_position_vjp` differentiates the kernel
//! weights' dependence on position (`axis_weights_derivative`, matches
//! finite difference), but this module's backward pass evaluates each step's
//! weights at the recorded forward position and does not backprop through
//! them. ChainQueen (`G2P_backward` in `tmp/ChainQueen/src/backward.cu`
//! sums `dw()` terms into the position gradient) and DiffTaichi
//! (`diffmpm.py`'s autodiff backward, no `stop_grad` on `p2g`/`g2p`) both
//! differentiate through the weights. The omission is a shortcut for a
//! small, short-horizon offline tool; it causes the ~5-7% gap measured in
//! `controller_gradient_matches_finite_difference_smooth_regime`.
//!
//! Scale/units note: this is a *training tool*, not the runtime solver. It
//! runs a small body (tens of particles) for a short horizon (~100 substeps)
//! thousands of times; the trained controller parameters are the output.
//!
//! Submodules follow the pipeline: `body_plan`/`config`/`stress` are shared
//! building blocks, `forward`/`backward` mirror the solver's P2G/G2P split,
//! `metrics`/`train` are the outer training loop. `backward.rs` stays one
//! file because sinusoid and feedback backprop share
//! `IncomingGrad`/`OutgoingGrad`/`SubstepCtx`/`GradSeed`.

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
// line-range extraction, see that file's doc comment.
#[cfg(test)]
mod tests;

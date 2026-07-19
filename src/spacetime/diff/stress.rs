use glam::{Mat2, Vec2};

use crate::particle::{Particle, Particles};

use super::DifferentiableMaterial;

// ── Signed active stress ──────────────────────────────────────────────────────

/// Signed directional active stress `act * strength * F * (n0 x n0) * F^T` --
/// DiffTaichi's exact actuation form (`new_F @ A @ new_F.transpose()` with
/// `A = diag(0, act)` for a Y fiber), allowing both contraction and extension.
/// Distinct from the engine's contract-only runtime form on purpose; see the
/// module docs.
pub(super) fn signed_active_stress(f: Mat2, act: f32, strength: f32, fiber_dir: Vec2) -> Mat2 {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON {
        return Mat2::ZERO;
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y) * (act * strength);
    f * a_mat * f.transpose()
}

/// VJP of `signed_active_stress` w.r.t. `F` and `act`. Same derivation as
/// `transfer::active_stress_vjp` (`Y = F*A*F^T` with symmetric `A`:
/// `dL/dF = (G + G^T) * F * A`; `dL/dact = strength * (G : F*(n0 x n0)*F^T)`),
/// minus the engine's `act <= 0` guard -- signed actuation must keep its
/// gradient on both sides of zero. FD-verified in this module's tests.
pub(super) fn signed_active_stress_vjp(
    f: Mat2,
    act: f32,
    strength: f32,
    fiber_dir: Vec2,
    g: Mat2,
) -> (Mat2, f32) {
    let len_sq = fiber_dir.dot(fiber_dir);
    if len_sq <= f32::EPSILON {
        return (Mat2::ZERO, 0.0);
    }
    let n0 = fiber_dir / len_sq.sqrt();
    let a_unit = Mat2::from_cols(n0 * n0.x, n0 * n0.y);

    let k_mat = f * a_unit * f.transpose();
    let d_loss_d_act = strength
        * (g.x_axis.x * k_mat.x_axis.x
            + g.x_axis.y * k_mat.x_axis.y
            + g.y_axis.x * k_mat.y_axis.x
            + g.y_axis.y * k_mat.y_axis.y);

    let g_sym = g + g.transpose();
    let d_loss_d_f = (act * strength) * (g_sym * f * a_unit);

    (d_loss_d_f, d_loss_d_act)
}

// ── Stress evaluation via the engine material ─────────────────────────────────

/// Reusable single-particle scratch so per-particle stress/VJP evaluation
/// doesn't allocate a fresh SoA per call (this runs particles x substeps x
/// training-iterations times).
///
/// Holds the material as `Box<dyn DifferentiableMaterial>` (dynamic
/// dispatch), not a generic type parameter -- this trainer's ~20 other
/// functions all take `&mut StressEval` without needing to know or
/// propagate which material is inside, and boxing keeps every one of those
/// signatures unchanged while still letting `new` accept ANY differentiable
/// material, not just `NeoHookeanMaterial`. The per-particle call overhead
/// of dynamic dispatch is irrelevant here (training-time tool, not the
/// real-time solver).
pub struct StressEval {
    scratch: Particles,
    mat: Box<dyn DifferentiableMaterial>,
}

impl StressEval {
    pub fn new(mat: impl DifferentiableMaterial + 'static) -> Self {
        let mut scratch = Particles::default();
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p.deformation_gradient = Mat2::IDENTITY;
        scratch.push(p);
        Self {
            scratch,
            mat: Box::new(mat),
        }
    }

    pub(super) fn passive(&mut self, f: Mat2) -> Mat2 {
        self.scratch.deformation_gradient[0] = f;
        self.mat.kirchhoff_stress(&self.scratch, 0)
    }

    pub(super) fn passive_vjp(&mut self, f: Mat2, g: Mat2) -> Mat2 {
        self.scratch.deformation_gradient[0] = f;
        self.mat.kirchhoff_stress_vjp(&self.scratch, 0, g)
    }
}

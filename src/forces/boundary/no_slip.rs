use glam::Vec2;

use super::{BoundaryCondition, apply_no_slip_wall_velocity, clamp_position_inside_grid};

/// The real Navier-Stokes no-slip wall condition: `v = 0` at the wall, both
/// normal AND tangential -- not `SlipBoundary`'s free-slip (normal only) or
/// `FrictionBoundary`'s partial Coulomb damping (tangential reduced
/// proportionally to normal impact speed, not forced to zero). This is the
/// STRONGEST of the three, matching a wall a real viscous fluid genuinely
/// sticks to (a boundary layer forms at the wall in real flow because of
/// exactly this condition).
///
/// Real, cited source, not hand-derived: `tmp/sparkl`'s own
/// `grid_update.rs`, `BoundaryHandling::Stick` variant -- `if proj.
/// is_inside: cell.velocity = 0`. sparkl (Dimforge, Apache-2.0) is already
/// trusted and cited elsewhere in this engine (its Monaghan-SPH Tait EOS
/// backs `NewtonianFluidMaterial`).
///
/// See `project_fluid_wall_noslip_friction_vision_2026-08-15` project
/// memory for the original motivation: fluid demos currently all use
/// `SlipBoundary`, which reads visually as fluid "sliding" along a wall
/// with zero drag -- a real, disclosed simplification (legitimate for
/// inviscid/high-Reynolds approximations), not a bug. This is the real
/// alternative for scenes that want the wall drag back.
#[derive(Debug, Clone, Copy)]
pub struct NoSlipBoundary {
    pub thickness: usize,
}

impl NoSlipBoundary {
    pub const fn new(thickness: usize) -> Self {
        Self { thickness }
    }
}

impl BoundaryCondition for NoSlipBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        apply_no_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }

    // Deliberately NOT overridden -- stays the conservative `false` default.
    // THREE real, sourced validation attempts tonight (2026-08-16), all
    // inconclusive for real, distinct, disclosed reasons -- not evidence
    // the mechanism is wrong, evidence real validation needs more care
    // than a quick check:
    //   1. `diag_noslip_fluid_probe.rs` v1: collapsing column against the
    //      wall -- near-wall speed measured HIGHER under no-slip (0.2315
    //      vs slip's 0.2083), but a real confound (34 vs 22 particles
    //      counted near the wall in the two runs, not the same set).
    //   2. Same file, v2: fixed the confound (136 SAME tracked particle
    //      indices in both runs) -- result held (0.6462 vs 0.5319, still
    //      higher). Working hypothesis: an ACTIVELY COLLAPSING column
    //      generates real wall-vorticity/shear under no-slip that a
    //      settled scene wouldn't show the same way -- not confirmed.
    //   3. `diag_noslip_poiseuille_validation.rs`: WebSearch-confirmed
    //      real MPM benchmark (arXiv 2402.11719 cites Couette/Poiseuille
    //      flow specifically for no-slip validation, <1% error vs. real
    //      experiment in a cited Taylor-Couette MPM study) -- built a
    //      horizontal channel, no-slip on all 4 walls (this boundary isn't
    //      direction-selective, a real disclosed simplification), driven
    //      by a horizontal body force, sampled far from the side walls.
    //      Result inconclusive: the analytical parabola prediction (using
    //      the material's own `viscosity` parameter literally) was off by
    //      4 orders of magnitude from measured speeds -- this engine's
    //      viscosity-to-SI mapping for this formula isn't understood well
    //      enough yet to trust that number. The measured profile itself
    //      was also not a clean parabola (noisy, even sign-changing in
    //      places) -- likely NOT yet at steady state: viscous diffusion
    //      time scales as H^2/nu, and at this material's real viscosity
    //      that could genuinely exceed the 800 steps tried.
    // Real next step, not done tonight: either run far longer (correctly
    // reaching steady state) or derive the real viscosity unit conversion
    // first (same rigor `NewtonianFluidMaterial::from_physical`'s own real
    // SI derivation already uses elsewhere in this engine) before trusting
    // any quantitative comparison. Do not flip this flag without a clean
    // result from one of those.
}

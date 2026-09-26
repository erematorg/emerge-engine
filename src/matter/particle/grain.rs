use glam::{Mat2, Vec2};

use crate::matter::materials::granular::grain_contact_law::GrainContactState;

/// One rigid circular grain -- the discrete-element analog of `Particle`:
/// pure per-point kinematic state, no dynamics of its own (moved out of
/// `spacetime::grains::population` 2026-08-05 to match that exact
/// precedent -- `Particle` lives here in `matter::particle`, its own P2G/G2P
/// dynamics live in `spacetime`; `Grain` now follows the same split instead
/// of bundling state and stepping logic together the way the much smaller,
/// standalone-first grains subsystem originally did). Real physical
/// quantities throughout (SI-scaled same as the rest of this engine, not
/// literal-grain-diameter -- see `DruckerPragerMaterial`'s own
/// `cosserat_length_scale_m` doc for the established real precedent of
/// using the SCENE's effective resolution, not literal micro-physics, when
/// literal grain scale is intractable).
#[derive(Clone, Copy, Debug)]
pub struct Grain {
    pub x: Vec2,
    pub v: Vec2,
    /// Planar angular velocity (2D: a scalar, the z-component of what would
    /// be a 3D angular-velocity vector -- this engine has no out-of-plane
    /// rotation to track).
    pub spin: f32,
    pub radius: f32,
    pub mass: f32,
    /// Real, accumulated planar orientation angle (radians) -- `integral(spin
    /// dt)`, the actual rotation a real rolling grain undergoes. Purely
    /// kinematic bookkeeping: no force/contact computation reads this (a
    /// circular grain's dynamics are rotationally symmetric, `spin` alone is
    /// what the physics needs), it exists so a RENDERER can show a grain
    /// actually rolling instead of just sliding -- a real, visible gap found
    /// 2026-08-03 building the first live grain demo (`sand_repose_angle_gui.rs`'s
    /// Grains mode): grains genuinely roll (that's the entire point of
    /// tonight's rolling-resistance fixes), but nothing tracked an angle to
    /// actually draw that rolling with.
    pub orientation: f32,
    /// Real APIC affine matrix (Jiang, Schroeder, Selle, Teran & Stomakhin
    /// 2015, "The Affine Particle-In-Cell Method") -- self-consistent grid
    /// transfer state, NOT a duplicate of `spin`: `c` is reconstructed FRESH
    /// each substep from the grid's own gathered local velocity field
    /// (`gather_grid_to_grains`) and consumed by the NEXT scatter
    /// (`scatter_grains_to_grid`), exactly the closed loop
    /// `Particle::velocity_gradient` already provides for ordinary MPM
    /// particles. `spin` stays the real, independent, `contact_law`-owned
    /// rigid-body angular velocity (the actual physics grains roll with);
    /// `c` is purely a transfer-layer device that lets a grain's momentum
    /// exchange with the shared grid conserve linear AND angular momentum
    /// without the grid-level dissipation pure PIC has (confirmed real,
    /// 2026-08-20: a grid-coupled grain pile stayed frozen near its initial
    /// shape under pure PIC, matching the literature's own documented "PIC
    /// causes sand to clump together" failure mode -- Jiang et al. 2015's
    /// own granular collision comparison).
    pub c: Mat2,
}

impl Grain {
    pub const fn new(x: Vec2, radius: f32, mass: f32) -> Self {
        Self {
            x,
            v: Vec2::ZERO,
            spin: 0.0,
            radius,
            mass,
            orientation: 0.0,
            c: Mat2::ZERO,
        }
    }

    /// Moment of inertia of a solid, uniform-density disc about its own
    /// center -- real, standard formula (elementary rigid-body mechanics),
    /// same convention `CosseratConfig::micro_inertia`'s own doc already
    /// used this session (`I = 0.5 * m * r^2` for a solid disc/sphere's
    /// mass-specific polar moment in 2D).
    pub fn moment_of_inertia(&self) -> f32 {
        0.5 * self.mass * self.radius * self.radius
    }

    pub(crate) const fn contact_state(&self) -> GrainContactState {
        GrainContactState {
            x: self.x,
            v: self.v,
            spin: self.spin,
            radius: self.radius,
            mass: self.mass,
        }
    }
}

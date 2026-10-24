use glam::{Mat2, Vec2};

use crate::matter::materials::granular::grain_contact_law::GrainContactState;

/// One rigid circular grain, the discrete-element analog of `Particle`:
/// per-point kinematic state only, with the dynamics in `spacetime`, as for
/// `Particle`. Physical quantities are SI-scaled like the rest of the
/// engine, not literal grain diameters (see `DruckerPragerMaterial`'s
/// `cosserat_length_scale_m` for using the scene's effective resolution when
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
    /// Accumulated planar orientation angle (radians), `integral(spin dt)`.
    /// Kinematic bookkeeping only: no force or contact computation reads it
    /// (a circular grain's dynamics are rotationally symmetric, `spin` is
    /// enough). It lets a renderer show a grain rolling rather than sliding.
    pub orientation: f32,
    /// APIC affine matrix (Jiang, Schroeder, Selle, Teran & Stomakhin 2015,
    /// "The Affine Particle-In-Cell Method"), grid transfer state, not a
    /// duplicate of `spin`: `c` is rebuilt each substep from the gathered
    /// local velocity field (`gather_grid_to_grains`) and consumed by the next
    /// scatter (`scatter_grains_to_grid`), the loop
    /// `Particle::velocity_gradient` provides for MPM particles. `spin` stays
    /// the `contact_law`-owned rigid-body angular velocity the grain rolls
    /// with; `c` lets the grain's exchange with the grid conserve linear and
    /// angular momentum without pure PIC's grid-level dissipation (under pure
    /// PIC a grid-coupled grain pile stayed frozen near its initial shape,
    /// the "PIC causes sand to clump together" failure of Jiang et al.
    /// 2015's granular collision comparison).
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

    /// A disc of `radius_m` metres and unit depth made of `elastic`, at
    /// `x` in cells: the 2D grain contract. Its mass per unit depth is
    /// `rho pi r^2` in the particles' own mass unit, `(rho / rho_ref) pi
    /// r_cells^2` as `SpawnRegion::mass_from` gives a particle
    /// `(rho / rho_ref) spacing^2`, so grains and particles exchange momentum
    /// in one unit. Its contact is `disc_contact`'s line contact.
    pub fn from_si(
        x: Vec2,
        radius_m: f32,
        elastic: &crate::Elastic,
        config: &crate::SimConfig,
    ) -> Self {
        let radius = radius_m / config.dx_meters;
        let mass = elastic.rho_kg_m3 / config.reference_density_kg_m3
            * std::f32::consts::PI
            * radius
            * radius;
        Self::new(x, radius, mass)
    }

    /// Moment of inertia of a solid, uniform-density disc about its centre,
    /// `I = 0.5 * m * r^2`.
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

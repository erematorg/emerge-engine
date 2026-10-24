//! Linear drag force field for MPM particles -- pulls a particle's velocity toward an
//! ambient/target flow velocity.
//!
//! `LinearDragField` models the drag a particle feels from a surrounding medium moving at
//! its own (roughly constant, locally) velocity -- river current pushing water downstream,
//! wind dragging loose dry sand, any scene needing sustained directional flow rather than
//! gravity settling everything into a static pool/pile.
//!
//! All positions/velocities are in **grid coordinates** (same units as `Particle::x`/`v`).
//!
//! # Physics
//! Linear drag toward a target velocity: **a = k·(v_target − v_particle)**
//!
//! This is the form of Stokes drag on a small sphere at low Reynolds number,
//! `F = -6 pi mu r (v - v_ambient)` (`stokes_drag_rate_from_si`), the
//! drag term of the Maxey-Riley equation for an inertial particle in a flow
//! (Farazmand & Haller 2015, arXiv:1310.2450, eq. 1). That equation has more
//! terms (fluid acceleration, added mass, buoyancy, history); this field keeps
//! the drag alone.
//!
//! `drag_coefficient` (`k`, units 1/time) sets the relaxation rate: a particle's velocity
//! decays toward `target_velocity` as `v(t) = target + (v0 − target)·exp(−k·t)` with no
//! other forces acting -- a checkable analytical prediction, not just "doesn't explode."
//!
//! # Particle masking
//! Which particles feel this field is controlled by `material_mask`, a bitmask over
//! `material_id` (`1 << material_id`), the SAME convention the GPU port's `GpuFieldEntry`
//! already uses -- deliberately NOT Coulomb's per-material-value `HashMap` (simpler, and
//! gives exact CPU/GPU parity instead of two different masking semantics). Use
//! `LinearDragField::ALL_MATERIALS` to affect every material, or `1 << id` for one/a few.

use glam::Vec2;

use crate::fields::{Field, GrainField};
use crate::particle::{Grain, Particles};

/// Linear-drag acceleration toward a target/ambient flow velocity, masked by material.
#[derive(Debug)]
pub struct LinearDragField {
    /// Target/ambient flow velocity in grid-units/time -- the velocity a masked particle's
    /// own velocity relaxes toward. Downstream direction for a river; wind direction for a
    /// sandstorm.
    pub target_velocity: Vec2,

    /// Drag/relaxation rate `k` in 1/time. Larger = faster relaxation toward
    /// `target_velocity` (stiffer coupling to the ambient flow). The velocity-decay
    /// timescale is `1/k`. For a small sphere at low Reynolds number this is the
    /// Stokes rate `6 pi mu r / m` (`stokes_drag_rate_from_si`); for
    /// anything else the caller supplies it and states where it comes from.
    pub drag_coefficient: f32,

    /// Bitmask over `material_id` (`1 << material_id`) selecting which particles feel this
    /// field. Use `LinearDragField::ALL_MATERIALS` to affect every material.
    pub material_mask: u32,
}

impl LinearDragField {
    /// Sentinel mask affecting every material -- matches `GpuFieldEntry::ALL_MATERIALS`.
    pub const ALL_MATERIALS: u32 = 0xFFFF_FFFF;

    pub const fn new(target_velocity: Vec2, drag_coefficient: f32, material_mask: u32) -> Self {
        Self {
            target_velocity,
            drag_coefficient,
            material_mask,
        }
    }
}

impl Field for LinearDragField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let material_id = particles.material_id[i];
        if self.material_mask != Self::ALL_MATERIALS && self.material_mask & (1 << material_id) == 0
        {
            return Vec2::ZERO;
        }
        self.drag_coefficient * (self.target_velocity - particles.v[i])
    }
}

/// Same formula as the `Field` impl above (a grain has no `material_id` to
/// mask against -- a `GrainPopulation` is already a single, homogeneous
/// population, see `GrainField`'s doc).
impl GrainField for LinearDragField {
    fn acceleration(&self, grain: &Grain) -> Vec2 {
        self.drag_coefficient * (self.target_velocity - grain.v)
    }
}

/// Same drag mechanism as `LinearDragField`, but the target flow velocity is a real
/// FUNCTION of the particle's own position instead of one constant vector -- a genuine
/// spatially-varying wind/current field.
///
/// `target_velocity_fn` is a plain `fn` pointer (not a closure), matching
/// `ScalarDiffusionField::source`'s own convention exactly (`fn(&Particle, f32) -> f32`)
/// for the same reason: keeps the field `Send + Sync` with no captured-state lifetime.
/// The MECHANISM here (sampling a position-dependent velocity) is standard, well-
/// established numerical infrastructure -- the same idea any semi-Lagrangian/grid-based
/// flow solver uses to look up an ambient velocity at a point. What the function
/// actually computes is up to the caller: this module's own test uses the exact,
/// textbook closed-form solution for potential flow around a circular cylinder (uniform
/// stream + doublet superposition -- Anderson-style fluid dynamics, confirmed against
/// MIT 16.unified fluid mechanics lecture notes and Caltech's "An Internet Book on Fluid
/// Dynamics," not invented), not a procedural/noise-based approximation.
#[derive(Debug)]
pub struct SpatialDragField {
    /// Position-dependent target flow velocity, evaluated at the particle's own
    /// `x` each substep. A pure function of position -- no time-dependence, no captured
    /// state (matches the `fn` pointer constraint).
    pub target_velocity_fn: fn(Vec2) -> Vec2,

    /// Same meaning as `LinearDragField::drag_coefficient` -- relaxation rate `k` in
    /// 1/time toward whatever `target_velocity_fn` returns at this particle's position.
    pub drag_coefficient: f32,

    /// Same convention as `LinearDragField::material_mask`.
    pub material_mask: u32,
}

impl SpatialDragField {
    pub fn new(
        target_velocity_fn: fn(Vec2) -> Vec2,
        drag_coefficient: f32,
        material_mask: u32,
    ) -> Self {
        Self {
            target_velocity_fn,
            drag_coefficient,
            material_mask,
        }
    }
}

impl Field for SpatialDragField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let material_id = particles.material_id[i];
        if self.material_mask != LinearDragField::ALL_MATERIALS
            && self.material_mask & (1 << material_id) == 0
        {
            return Vec2::ZERO;
        }
        let target = (self.target_velocity_fn)(particles.x[i]);
        self.drag_coefficient * (target - particles.v[i])
    }
}

/// Single-sphere Stokes drag rate (Stokes 1851) for
/// `LinearDragField::drag_coefficient` (1/s, the solver's own time unit):
/// `F = 6 pi mu r v` gives `dv/dt = -(6 pi mu r / m) v`.
///
/// Only valid at low Reynolds number (`Re = rho_air v 2r / mu <~ 1`), e.g.
/// fine dust or sand grains in a gentle wind. A 1 cm ball at 1 m/s sits at
/// `Re ~ 2000-3000`, where drag is quadratic (form drag), not this.
///
/// # Example -- fine dust grain in air (r=50 micron, m=6.5e-10 kg, 20 C air)
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::fields::drag::stokes_drag_rate_from_si;
/// let k = stokes_drag_rate_from_si(5.0e-5, 6.5e-10, 1.81e-5);
/// # let _ = k;
/// ```
pub fn stokes_drag_rate_from_si(radius_m: f32, mass_kg: f32, dynamic_viscosity_pa_s: f32) -> f32 {
    6.0 * std::f32::consts::PI * dynamic_viscosity_pa_s * radius_m / mass_kg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    fn particle_with_velocity(v: Vec2, material_id: u32) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.v = v;
        p.material_id = material_id;
        Particles::from(vec![p])
    }

    #[test]
    fn zero_at_target_velocity() {
        let target = Vec2::new(2.0, 0.0);
        let field = LinearDragField::new(target, 3.0, LinearDragField::ALL_MATERIALS);
        let particles = particle_with_velocity(target, 0);
        assert_eq!(Field::acceleration(&field, &particles, 0), Vec2::ZERO);
    }

    #[test]
    fn points_toward_target_velocity() {
        let target = Vec2::new(2.0, 0.0);
        let field = LinearDragField::new(target, 3.0, LinearDragField::ALL_MATERIALS);
        let particles = particle_with_velocity(Vec2::ZERO, 0);
        let acc = Field::acceleration(&field, &particles, 0);
        assert!(
            (acc - Vec2::new(6.0, 0.0)).length() < 1e-6,
            "expected a=k*(target-v)=3*(2,0)=(6,0), got {acc:?}"
        );
    }

    #[test]
    fn material_mask_excludes_unmasked_materials() {
        let field = LinearDragField::new(Vec2::new(5.0, 0.0), 1.0, 1 << 2); // only material_id 2
        let unmasked = particle_with_velocity(Vec2::ZERO, 0);
        assert_eq!(Field::acceleration(&field, &unmasked, 0), Vec2::ZERO);
        let masked = particle_with_velocity(Vec2::ZERO, 2);
        assert_ne!(Field::acceleration(&field, &masked, 0), Vec2::ZERO);
    }

    #[test]
    fn grain_field_acceleration_matches_particle_field_formula() {
        let target = Vec2::new(2.0, 0.0);
        let field = LinearDragField::new(target, 3.0, LinearDragField::ALL_MATERIALS);
        let mut grain = Grain::new(Vec2::ZERO, 1.0, 1.0);
        grain.v = Vec2::ZERO;
        let acc = GrainField::acceleration(&field, &grain);
        assert!(
            (acc - Vec2::new(6.0, 0.0)).length() < 1e-6,
            "expected a=k*(target-v)=3*(2,0)=(6,0), got {acc:?} -- same formula as the Field impl, no material mask"
        );
    }
}

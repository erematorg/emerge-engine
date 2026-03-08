pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod snow;

pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use snow::SnowMaterial;

use glam::Mat2;

use crate::state::particle::Particle;

pub trait MaterialModel: Send + Sync + core::fmt::Debug {
    // Returns the Kirchhoff-like stress used by the transfer kernel.
    // The kernel applies geometry/time factors (dt, d_inverse, cell_dist, weight).
    fn kirchhoff_stress(&self, _particle: &Particle) -> Mat2 {
        Mat2::ZERO
    }

    // Returns the particle volume used in the stress contribution.
    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.volume
    }

    fn timestep_bound(
        &self,
        _particle: &Particle,
        _cell_width: f32,
        _material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        f32::INFINITY
    }

    fn update_particle(&self, _particle: &mut Particle, _dt: f32) {}
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

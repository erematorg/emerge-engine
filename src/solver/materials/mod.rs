pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod params;
pub mod snow;

pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use params::MaterialParams;
pub use snow::SnowMaterial;

use glam::Mat2;

use crate::state::particle::Particle;

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles — do not change them.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstitutiveModel {
    Fallback   = 0,
    Fluid      = 1, // Weakly-compressible Newtonian fluid, Tait EOS
    NeoHookean = 2, // Neo-Hookean hyperelastic (jelly, soft solids)
    Corotated  = 3, // Corotated linear elastic (stiffer baseline)
    Snow       = 4, // Corotated + SVD plasticity (Stomakhin 2013)
}

pub trait MaterialModel: Send + Sync + core::fmt::Debug {
    /// Which constitutive law this material implements.
    /// Used by the GPU shader to select the correct stress branch per particle.
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fallback
    }
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

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

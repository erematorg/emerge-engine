use glam::Vec2;

use crate::fields::ForceField;
use crate::particle::Particle;

/// Archimedes buoyancy — lighter particles rise, heavier particles sink.
///
/// Applies `a = gravity × (fluid_density / particle.density − 1)` per particle.
/// At ρ_particle = ρ_fluid: zero net force (neutrally buoyant).
/// At ρ_particle < ρ_fluid: net upward force (floats).
/// At ρ_particle > ρ_fluid: net downward force (sinks faster).
///
/// `gravity` should match `SolverConfig::gravity` — the field computes the
/// density-differential force, not total gravity (the solver already applies base gravity).
/// Setting `gravity` equal to the solver gravity gives physically correct buoyancy;
/// setting it higher exaggerates the effect.
///
/// # IRL calibration
/// For water (ρ₀ = 1000 kg/m³):
/// - Wood (ρ ≈ 600 kg/m³): floats at ~60% submerged — buoyancy_ratio ≈ 0.67
/// - Steel (ρ ≈ 7800 kg/m³): sinks — buoyancy_ratio ≈ 0.13
/// - Ice (ρ ≈ 917 kg/m³): floats at ~8% above surface — buoyancy_ratio ≈ 1.09
///
/// In grid units, set `fluid_density` to match your fluid material's `rest_density`.
#[derive(Debug, Clone, Copy)]
pub struct BuoyancyField {
    /// Reference fluid density — typically the `rest_density` of the surrounding fluid material.
    pub fluid_density: f32,
    /// Gravitational direction and magnitude — should match `SolverConfig::gravity`.
    pub gravity: Vec2,
    /// Density floor to prevent division by zero for near-vacuum particles.
    pub min_density: f32,
}

impl BuoyancyField {
    pub fn new(fluid_density: f32, gravity: Vec2) -> Self {
        Self { fluid_density, gravity, min_density: 1.0e-4 }
    }

    /// For a fluid sim: `fluid_density` is your water/mud material's `rest_density`,
    /// `gravity` matches `SolverConfig::gravity`.
    pub fn for_fluid(fluid_density: f32, gravity: Vec2) -> Self {
        Self::new(fluid_density, gravity)
    }
}

impl ForceField for BuoyancyField {
    fn acceleration(&self, particle: &Particle) -> Vec2 {
        let rho = particle.density.max(self.min_density);
        // Buoyancy = −gravity × (ρ_fluid/ρ − 1).
        // Negative sign: gravity is downward (negative y), so buoyancy is upward for light particles.
        -self.gravity * (self.fluid_density / rho - 1.0)
    }
}

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;

/// Archimedes buoyancy — lighter particles rise, heavier particles sink.
///
/// Applies `Δv = −gravity · (fluid_density / particle.density) · dt` per particle.
/// The solver already applies gravity (−g) to all particles; this field adds +g·(ρ_fluid/ρ)
/// so the net acceleration on the particle is `g·(ρ_fluid/ρ − 1)`:
///   - ρ_particle < ρ_fluid: net upward (floats)
///   - ρ_particle = ρ_fluid: zero net (neutrally buoyant, hovers)
///   - ρ_particle > ρ_fluid: net downward but weaker than free-fall (sinks slower)
///
/// `gravity` should match `SimConfig::gravity` exactly.
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
    /// Gravitational direction and magnitude — should match `SimConfig::gravity`.
    pub gravity: Vec2,
    /// Density floor to prevent division by zero for near-vacuum particles.
    pub min_density: f32,
}

impl BuoyancyField {
    pub fn new(fluid_density: f32, gravity: Vec2) -> Self {
        Self {
            fluid_density,
            gravity,
            min_density: 1.0e-4,
        }
    }

    /// For a fluid sim: `fluid_density` is your water/mud material's `rest_density`,
    /// `gravity` matches `SimConfig::gravity`.
    pub fn for_fluid(fluid_density: f32, gravity: Vec2) -> Self {
        Self::new(fluid_density, gravity)
    }
}

impl Field for BuoyancyField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let rho = particles.density[i].max(self.min_density);
        -self.gravity * (self.fluid_density / rho)
    }
}

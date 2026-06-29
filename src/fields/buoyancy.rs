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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    fn particle_with_density(rho: f32) -> Particles {
        let mut p = Particle::zeroed();
        p.density = rho;
        Particles::from(vec![p])
    }

    /// Net acceleration (solver gravity + this field) for a particle of density `rho`,
    /// per the field's own doc comment: g·(ρ_fluid/ρ − 1).
    fn net_with_gravity(field: &BuoyancyField, gravity: Vec2, rho: f32) -> Vec2 {
        let soa = particle_with_density(rho);
        gravity + field.acceleration(&soa, 0)
    }

    #[test]
    fn lighter_than_fluid_floats_up() {
        let gravity = Vec2::new(0.0, -9.8);
        let field = BuoyancyField::new(1000.0, gravity); // water
        let net = net_with_gravity(&field, gravity, 600.0); // wood
        assert!(net.y > 0.0, "wood in water must net upward: {net:?}");
    }

    #[test]
    fn denser_than_fluid_sinks_slower_than_free_fall() {
        let gravity = Vec2::new(0.0, -9.8);
        let field = BuoyancyField::new(1000.0, gravity); // water
        let net = net_with_gravity(&field, gravity, 7800.0); // steel
        assert!(
            net.y < 0.0,
            "steel in water must still net downward: {net:?}"
        );
        assert!(
            net.y > gravity.y,
            "steel must sink slower than free fall: net={:.3} free_fall={:.3}",
            net.y,
            gravity.y
        );
    }

    #[test]
    fn matching_density_hovers() {
        let gravity = Vec2::new(0.0, -9.8);
        let field = BuoyancyField::new(1000.0, gravity);
        let net = net_with_gravity(&field, gravity, 1000.0);
        assert!(
            net.length() < 1e-4,
            "neutrally buoyant particle must have ~zero net acceleration: {net:?}"
        );
    }
}

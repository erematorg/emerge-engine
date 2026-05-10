//! Electromagnetic force fields for MPM particles.
//!
//! Bridges `electromagnetics::` field math into the `ForceField` substep hook.
//!
//! `UniformElectricField` applies a spatially-constant external E-field to charged particles.
//! This is distinct from `CoulombField` (which models point-source interactions).
//! Use it for capacitor plates, external confinement fields, or EM traps.

use std::collections::HashMap;

use glam::Vec2;

use crate::fields::ForceField;
use crate::particle::Particle;

/// Uniform external electric field E applied to charged particles.
///
/// Force on a particle: F = q × E  →  acceleration a = q × E / mass
///
/// # Charge encoding
/// Same convention as `CoulombField`: `material_charges` maps `material_id → charge_value`.
/// Particles whose material is not in the map are treated as neutral (charge = 0).
///
/// # Units
/// `field` is in simulation units (force-per-unit-charge). In SI: V/m.
/// `charge` values must be in the same unit system as your Coulomb constant.
pub struct UniformElectricField {
    /// The uniform electric field vector E (force per unit charge, in simulation units).
    pub field: Vec2,

    /// Per-material charge. `material_id → charge`.
    /// Materials not in this map are neutral and unaffected.
    pub material_charges: HashMap<u32, f32>,
}

impl UniformElectricField {
    pub fn new(field: Vec2, material_charges: HashMap<u32, f32>) -> Self {
        Self {
            field,
            material_charges,
        }
    }

    /// Convenience constructor: one material, one charge.
    pub fn for_material(field: Vec2, material_id: u32, charge: f32) -> Self {
        let mut map = HashMap::new();
        map.insert(material_id, charge);
        Self::new(field, map)
    }
}

impl ForceField for UniformElectricField {
    fn acceleration(&self, particle: &Particle) -> Vec2 {
        let q = match self.material_charges.get(&particle.material_id) {
            Some(&q) if q.abs() > f32::EPSILON => q,
            _ => return Vec2::ZERO,
        };
        let inv_mass = if particle.mass > f32::EPSILON {
            1.0 / particle.mass
        } else {
            0.0
        };
        // a = q · E / m
        self.field * (q * inv_mass)
    }
}

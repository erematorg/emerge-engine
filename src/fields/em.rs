//! Electromagnetic force fields for MPM particles.
//!
//! Bridges `electromagnetics::` field math into the `Field` substep hook.
//!
//! `UniformElectricField` applies a spatially-constant external E-field to charged particles.
//! This is distinct from `CoulombField` (which models point-source interactions).
//! Use it for capacitor plates, external confinement fields, or EM traps.

use std::collections::HashMap;

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;

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

impl Field for UniformElectricField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let q = match self.material_charges.get(&particles.material_id[i]) {
            Some(&q) if q.abs() > f32::EPSILON => q,
            _ => return Vec2::ZERO,
        };
        let inv_mass = if particles.mass[i] > f32::EPSILON {
            1.0 / particles.mass[i]
        } else {
            0.0
        };
        self.field * (q * inv_mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    /// `UniformElectricField` had ZERO test coverage of any kind before this
    /// (confirmed via a full test-file audit, 2026-07-07). Checks the real,
    /// analytic physics directly: F = qE, a = F/m -- the Lorentz force with no
    /// magnetic component, exactly what this field claims to model.
    fn particle_with(material_id: u32, mass: f32) -> Particles {
        let mut p = Particle::zeroed();
        p.material_id = material_id;
        p.mass = mass;
        Particles::from(vec![p])
    }

    #[test]
    fn neutral_material_feels_no_force() {
        let field = UniformElectricField::for_material(Vec2::new(10.0, 0.0), 1, 5.0);
        let particles = particle_with(0, 2.0); // material 0, not in the charge map
        assert_eq!(field.acceleration(&particles, 0), Vec2::ZERO);
    }

    #[test]
    fn zero_charge_entry_is_treated_as_neutral() {
        let mut map = HashMap::new();
        map.insert(1, 0.0); // explicitly zero charge
        let field = UniformElectricField::new(Vec2::new(10.0, 0.0), map);
        let particles = particle_with(1, 2.0);
        assert_eq!(field.acceleration(&particles, 0), Vec2::ZERO);
    }

    /// Real Lorentz force law: a = q*E / m, matched exactly (this is a pure
    /// analytic formula, not a simulated/approximated quantity).
    #[test]
    fn charged_particle_matches_qe_over_m_exactly() {
        let e_field = Vec2::new(10.0, -4.0);
        let charge = 3.0;
        let mass = 2.0;
        let field = UniformElectricField::for_material(e_field, 1, charge);
        let particles = particle_with(1, mass);

        let expected = e_field * (charge / mass);
        let actual = field.acceleration(&particles, 0);
        assert!(
            (actual - expected).length() < 1.0e-6,
            "expected a=qE/m={expected:?}, got {actual:?}"
        );
    }

    /// Doubling charge must exactly double acceleration -- F=qE is linear in q.
    #[test]
    fn acceleration_scales_linearly_with_charge() {
        let e_field = Vec2::new(5.0, 0.0);
        let mass = 1.0;
        let field_1x = UniformElectricField::for_material(e_field, 1, 2.0);
        let field_2x = UniformElectricField::for_material(e_field, 1, 4.0);
        let particles = particle_with(1, mass);

        let a1 = field_1x.acceleration(&particles, 0);
        let a2 = field_2x.acceleration(&particles, 0);
        assert!(
            (a2 - a1 * 2.0).length() < 1.0e-6,
            "doubling charge should exactly double acceleration: a1={a1:?} a2={a2:?}"
        );
    }

    /// Doubling mass must exactly halve acceleration -- real F=ma inverse scaling.
    #[test]
    fn acceleration_scales_inversely_with_mass() {
        let e_field = Vec2::new(5.0, 0.0);
        let field = UniformElectricField::for_material(e_field, 1, 3.0);
        let light = particle_with(1, 1.0);
        let heavy = particle_with(1, 2.0);

        let a_light = field.acceleration(&light, 0);
        let a_heavy = field.acceleration(&heavy, 0);
        assert!(
            (a_light - a_heavy * 2.0).length() < 1.0e-6,
            "doubling mass should exactly halve acceleration: a_light={a_light:?} \
             a_heavy={a_heavy:?}"
        );
    }

    /// Negative charge must accelerate OPPOSITE to the field direction -- real
    /// electrostatics (a positive test charge and a negative one in the same
    /// field feel opposite forces).
    #[test]
    fn negative_charge_accelerates_opposite_to_field() {
        let e_field = Vec2::new(8.0, 0.0);
        let field = UniformElectricField::for_material(e_field, 1, -2.0);
        let particles = particle_with(1, 1.0);
        let a = field.acceleration(&particles, 0);
        assert!(
            a.x < 0.0,
            "negative charge must accelerate against +X field, got {a:?}"
        );
    }

    /// Zero mass (degenerate/uninitialized particle) must not produce Inf/NaN --
    /// same defensive convention as the rest of this field's `inv_mass` guard.
    #[test]
    fn zero_mass_does_not_divide_by_zero() {
        let field = UniformElectricField::for_material(Vec2::new(10.0, 0.0), 1, 5.0);
        let particles = particle_with(1, 0.0);
        let a = field.acceleration(&particles, 0);
        assert_eq!(
            a,
            Vec2::ZERO,
            "zero mass must yield zero acceleration, not Inf/NaN"
        );
    }
}

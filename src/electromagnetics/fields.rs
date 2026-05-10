//! Electric and magnetic field math.
//!
//! Pure-Rust, no ECS. Ported from `crates/energy/src/electromagnetism/fields.rs`.
//!
//! # Reference
//! - Coulomb's law: E = k·q / r²
//! - Biot-Savart law: dB = (μ₀/4π)(I dl × r̂)/r²

use glam::Vec2;

/// Coulomb constant k = 1/(4πε₀) in SI units (N·m²/C²).
pub const COULOMB_CONSTANT: f32 = 8.99e9;
/// μ₀/(4π) in SI units (T·m/A).
pub const MAGNETIC_CONSTANT_DIV_4PI: f32 = 1e-7;

/// Electric field vector at a point in 2D space.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElectricField {
    /// Field vector (N/C or V/m).
    pub field: Vec2,
    /// Position where the field is evaluated.
    pub position: Vec2,
}

impl ElectricField {
    pub fn new(field: Vec2, position: Vec2) -> Self {
        Self { field, position }
    }

    /// |E|
    pub fn strength(&self) -> f32 {
        self.field.length()
    }

    /// E field at `field_position` due to a point charge at `charge_position`.
    ///
    /// E = k·q·r̂ / r²  (direction: away from positive charge)
    pub fn from_point_charge(charge: f32, charge_pos: Vec2, field_pos: Vec2) -> Self {
        let r = field_pos - charge_pos;
        let r2 = r.length_squared();
        if r2 < 1e-10 {
            return Self::new(Vec2::ZERO, field_pos);
        }
        Self::new(r.normalize() * (COULOMB_CONSTANT * charge / r2), field_pos)
    }

    /// Superpose two fields at the same position (linearity).
    pub fn superpose(&self, other: &ElectricField) -> Self {
        debug_assert!(
            (self.position - other.position).length() < 1e-6,
            "Cannot superpose fields at different positions"
        );
        Self::new(self.field + other.field, self.position)
    }
}

/// Magnetic field vector at a point in 2D space.
#[derive(Debug, Clone, Copy, Default)]
pub struct MagneticField {
    /// Field vector (Tesla, out-of-plane component represented in 2D).
    pub field: Vec2,
    pub position: Vec2,
}

impl MagneticField {
    pub fn new(field: Vec2, position: Vec2) -> Self {
        Self { field, position }
    }
    pub fn strength(&self) -> f32 {
        self.field.length()
    }

    /// Biot-Savart: dB at `field_pos` from a current element at `current_pos`.
    ///
    /// dB = (μ₀/4π)(I dl × r̂) / r²
    pub fn from_current_element(
        current: f32,
        current_dir: Vec2,
        current_pos: Vec2,
        field_pos: Vec2,
    ) -> Self {
        let r = field_pos - current_pos;
        let dist = r.length();
        if dist < 1e-10 {
            return Self::new(Vec2::ZERO, field_pos);
        }
        let r_unit = r / dist;
        let mag = MAGNETIC_CONSTANT_DIV_4PI * current / (dist * dist);
        let field = Vec2::new(-r_unit.y, r_unit.x)
            * (current_dir.x * r_unit.y - current_dir.y * r_unit.x)
            * mag;
        Self::new(field, field_pos)
    }

    pub fn superpose(&self, other: &MagneticField) -> Self {
        debug_assert!(
            (self.position - other.position).length() < 1e-6,
            "Cannot superpose fields at different positions"
        );
        Self::new(self.field + other.field, self.position)
    }
}

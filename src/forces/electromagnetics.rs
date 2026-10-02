//! Electric and magnetic field-query math -- the Forces half of `electromagnetics::`.
//!
//! Pure Rust, no ECS. Point-charge/current force math (Forces domain); wave
//! propagation and optical material properties are `energy::electromagnetics`.
//!
//! Dormant/experimental: not wired into the solver's `Field` trait (see
//! `forces::fields::em` for the wired-in `UniformElectricField`, which
//! implements F=qE on its own rather than calling into this module).
//!
//! # Reference
//! - Coulomb's law: E = k·q / r²
//! - Biot-Savart law: dB = (μ₀/4π)(I dl × r̂)/r²

use glam::Vec2;

/// Coulomb constant k = 1/(4πε₀) in SI units (N·m²/C²), from
/// ε₀ = 8.8541878188e-12 F/m (CODATA 2022, NIST). Was 8.99e9, 0.03 percent
/// high.
pub const COULOMB_CONSTANT: f32 = 8.987_552e9;
/// μ₀/(4π) in SI units (T·m/A). μ₀ = 1.256 637 061 27(20) e-6 N/A² (CODATA
/// 2022, NIST, physics.nist.gov/cgi-bin/cuu/Value?mu0), so μ₀/(4π) =
/// 0.999 999 999 87 e-7: no longer exactly 1e-7 since the 2019 SI
/// redefinition, but the difference is below f32 resolution.
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

/// Magnetic field at a point in 2D space: a (pseudo)scalar, not a vector.
///
/// In 2+1-dimensional electromagnetism the magnetic field is the out-of-plane
/// (z) pseudoscalar component; there is no in-plane B (Kirk T. McDonald,
/// "Electrodynamics in 1 and 2 Spatial Dimensions", Princeton,
/// http://kirkmcd.princeton.edu/examples/2dem.pdf). For a current element
/// along +x and a field point along +y from it, Biot-Savart gives B purely
/// along +z.
#[derive(Debug, Clone, Copy, Default)]
pub struct MagneticField {
    /// Out-of-plane (z) component, Tesla. Positive = out of the page (+z,
    /// right-hand rule), matching the sign convention `from_current_element`'s
    /// own `dl × r̂` cross product already used internally.
    pub field: f32,
    pub position: Vec2,
}

impl MagneticField {
    pub fn new(field: f32, position: Vec2) -> Self {
        Self { field, position }
    }
    pub fn strength(&self) -> f32 {
        self.field.abs()
    }

    /// Biot-Savart: dB at `field_pos` from a current element at `current_pos`.
    ///
    /// dB = (μ₀/4π)(I dl × r̂) / r² -- in 2D, `dl × r̂` (both in-plane vectors)
    /// is exactly the scalar 2D cross product `dl.x*r̂.y - dl.y*r̂.x`, the real
    /// z-component of the 3D cross product when both operands lie in the
    /// xy-plane. No projection back into the plane needed or correct.
    pub fn from_current_element(
        current: f32,
        current_dir: Vec2,
        current_pos: Vec2,
        field_pos: Vec2,
    ) -> Self {
        let r = field_pos - current_pos;
        let dist = r.length();
        if dist < 1e-10 {
            return Self::new(0.0, field_pos);
        }
        let r_unit = r / dist;
        let mag = MAGNETIC_CONSTANT_DIV_4PI * current / (dist * dist);
        let cross_z = current_dir.x * r_unit.y - current_dir.y * r_unit.x;
        Self::new(mag * cross_z, field_pos)
    }

    pub fn superpose(&self, other: &MagneticField) -> Self {
        debug_assert!(
            (self.position - other.position).length() < 1e-6,
            "Cannot superpose fields at different positions"
        );
        Self::new(self.field + other.field, self.position)
    }
}

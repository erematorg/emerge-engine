//! Electromagnetic plane waves and optical material properties.
//!
//! Pure Rust, no ECS. Wave propagation and optical `MaterialProperties` are
//! radiative energy transfer (Energy domain); point-charge/current force
//! math is `forces::electromagnetics`.

use crate::forces::electromagnetics::{ElectricField, MagneticField};
use glam::Vec2;

/// Speed of light in vacuum (m/s), `radiation::SPEED_OF_LIGHT` in `f32`.
pub const C: f32 = crate::energy::radiation::SPEED_OF_LIGHT as f32;

/// A plane electromagnetic wave propagating in 2D.
///
/// E and B are transverse to the propagation direction.
/// B_amplitude = E_amplitude / c (vacuum relation).
pub struct ElectromagneticWave {
    pub frequency: f32,
    pub direction: Vec2,
    pub electric_amplitude: f32,
    pub magnetic_amplitude: f32,
    pub phase: f32,
    pub wave_number: f32,
}

impl ElectromagneticWave {
    /// Construct from frequency, propagation direction, electric amplitude, and phase.
    pub fn new(frequency: f32, direction: Vec2, electric_amplitude: f32, phase: f32) -> Self {
        assert!(frequency > 0.0, "Wave frequency must be positive");
        let wavelength = C / frequency;
        let wave_number = 2.0 * std::f32::consts::PI / wavelength;
        Self {
            frequency,
            direction: direction.normalize(),
            electric_amplitude,
            magnetic_amplitude: electric_amplitude / C,
            phase,
            wave_number,
        }
    }

    /// E and B fields at `position` and `time`.
    ///
    /// In 2D, B is the out-of-plane scalar (see `MagneticField`), so E (in
    /// plane, perpendicular to propagation), B (out of plane) and the
    /// in-plane propagation direction are mutually perpendicular, a
    /// transverse wave. An in-plane B would have to lie along the
    /// propagation direction, making the wave longitudinal.
    pub fn get_fields_at(&self, position: Vec2, time: f32) -> (ElectricField, MagneticField) {
        let proj = self.direction.dot(position);
        let phi = self.wave_number * proj - 2.0 * std::f32::consts::PI * self.frequency * time
            + self.phase;
        let sin_p = phi.sin();
        let e_dir = Vec2::new(-self.direction.y, self.direction.x);
        (
            ElectricField::new(e_dir * (self.electric_amplitude * sin_p), position),
            MagneticField::new(self.magnetic_amplitude * sin_p, position),
        )
    }
}

/// Electromagnetic material properties (permittivity, permeability, conductivity).
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
    /// Electric permittivity ε (F/m).
    pub permittivity: f32,
    /// Magnetic permeability μ (H/m).
    pub permeability: f32,
    /// Electrical conductivity σ (S/m).
    pub conductivity: f32,
}

impl MaterialProperties {
    pub fn vacuum() -> Self {
        Self {
            permittivity: 8.854_188e-12,
            permeability: 1.256_637e-6,
            conductivity: 0.0,
        }
    }
    pub fn new(permittivity: f32, permeability: f32, conductivity: f32) -> Self {
        Self {
            permittivity,
            permeability,
            conductivity,
        }
    }
    /// Refractive index n = √(εᵣ·μᵣ).
    pub fn refractive_index(&self) -> f32 {
        let vac = Self::vacuum();
        ((self.permittivity / vac.permittivity) * (self.permeability / vac.permeability)).sqrt()
    }
    /// Speed of light in this material: v = c/n.
    pub fn light_speed(&self) -> f32 {
        C / self.refractive_index()
    }
}

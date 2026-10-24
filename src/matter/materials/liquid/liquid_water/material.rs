//! Liquid water as a material of the precise path: the barotropic liquid
//! (`liquid::barotropic`) with the record's coefficients at one
//! temperature, nothing defaulted.
//!
//! The modulus is the isentropic `K_s = w^2 / v` (`LiquidWater`): no heat
//! flows in the path's water (no conduction is coupled), so its
//! compressions are adiabatic and reversible. `K_s / K_T = c_p / c_v`,
//! 1.0065 at 20 C. Declared gaps: inviscid (the record's viscosity is not
//! applied yet), no surface tension, the modulus fixed at the material's
//! temperature (the particles' temperatures do not move it), and the
//! quadratic energy's validity (`QUADRATIC_VALID_ABOVE_P0_PA`). Only the
//! total Lagrangian path carries it: the explicit path and the GPU refuse
//! it.

use super::LiquidWater;
use crate::SimConfig;
use crate::materials::{MaterialModel, ParticleMass};

/// Liquid water at `temperature_k`, from IAPWS SR6-08 (`LiquidWater`).
#[derive(Clone, Copy, Debug)]
pub struct LiquidWaterMaterial {
    pub temperature_k: f64,
    pub record: LiquidWater,
    /// The record's density in the grid's units
    /// (`SimConfig::density_from_si`), for `MaterialModel::rest_density`.
    rest_density: f32,
}

impl LiquidWaterMaterial {
    /// The material at `temperature_k` for `config`'s units. `None`
    /// outside the record's range (`VALID_K`).
    pub fn new(temperature_k: f64, config: &SimConfig) -> Option<Self> {
        let record = LiquidWater::at(temperature_k)?;
        Some(Self {
            temperature_k,
            record,
            rest_density: config.density_from_si(record.density() as f32),
        })
    }

    /// The modulus the path's law takes, `K_s`, Pa.
    pub fn bulk_modulus(&self) -> f64 {
        self.record.isentropic_bulk()
    }
}

impl MaterialModel for LiquidWaterMaterial {
    fn rest_density(&self) -> Option<f32> {
        Some(self.rest_density)
    }

    fn specific_heat_j_kg_k(&self) -> f32 {
        self.record.cp as f32
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "LiquidWaterMaterial: the barotropic liquid is carried by the total Lagrangian path only",
        )
    }
}

/// A particle's mass at `spacing` cells, `rho (spacing dx)^2` with the
/// record's density, kg per metre of depth.
impl ParticleMass for LiquidWaterMaterial {
    fn particle_mass(&self, spacing: f32, config: &SimConfig) -> f32 {
        self.record.density() as f32 * (spacing * config.dx_meters).powi(2)
    }
}

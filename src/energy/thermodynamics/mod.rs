//! Thermodynamics — heat and generic scalar transport, MPM-coupled.
//!
//! - `diffusion.rs`    — Fourier heat diffusion ∂T/∂t = α∇²T + Newton cooling
//! - `scalar_field.rs` — generic ∂φ/∂t = D·∇²φ − λ·φ + S (pheromone, nutrients, morphogen)
//! - `stencil.rs`      — shared Laplacian FD step used by both of the above
//! - `transfer.rs`     — scalar IRL primitives: conduction, Stefan-Boltzmann radiation, entropy/2nd law
//! - `radiance.rs`     — real point-source irradiance (inverse-square law), e.g. a star's real flux at a planet's distance

pub mod cosserat_field;
pub mod diffusion;
pub mod granular_fluidity;
pub mod ideal_gas;
pub mod radiance;
pub mod scalar_field;
mod stencil;
pub mod transfer;

pub use cosserat_field::{CosseratConfig, CosseratField};
pub use diffusion::{ThermalConfig, ThermalDiffusion};
pub use granular_fluidity::{GranularFluidityConfig, GranularFluidityField};
pub use ideal_gas::{
    AIR_ADIABATIC_INDEX as GAS_AIR_ADIABATIC_INDEX, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
    ideal_gas_pressure, ideal_gas_sound_speed, ideal_gas_sound_speed_from_temperature,
};
pub use radiance::RadianceField;
pub use scalar_field::{ScalarDiffusionConfig, ScalarDiffusionField};
pub use transfer::{
    STEFAN_BOLTZMANN, entropy_change_heat_transfer, entropy_change_irreversible, heat_conduction,
    heat_radiation, irradiance_at_distance, saturating_uptake, second_law_holds,
    stellar_luminosity_w, thermal_diffusivity,
};

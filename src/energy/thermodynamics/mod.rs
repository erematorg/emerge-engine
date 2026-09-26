//! Thermodynamics -- heat and generic scalar transport, MPM-coupled.
//!
//! - `diffusion.rs`    -- Fourier heat diffusion ∂T/∂t = α∇²T + Newton cooling
//! - `enthalpy.rs` -- Voller-Cross/Voller-Swaminathan enthalpy method: real
//!   continuous mushy-zone phase change (partial melting), the numerical
//!   core behind GitHub issue #7
//! - `scalar_field.rs` -- generic ∂φ/∂t = D·∇²φ − λ·φ + S (pheromone, nutrients, morphogen)
//! - `stencil.rs`      -- shared Laplacian FD step used by both of the above
//! - `transfer.rs`     -- scalar IRL primitives: conduction, Stefan-Boltzmann radiation, entropy/2nd law

pub mod cosserat_field;
pub mod diffusion;
pub mod enthalpy;
pub mod frictional_heating;
pub mod granular_fluidity;
pub mod ideal_gas;
pub mod scalar_field;
mod stencil;
pub mod transfer;
pub mod water_saturation;

pub use cosserat_field::{CosseratConfig, CosseratField};
pub use diffusion::{ThermalConfig, ThermalDiffusion};
pub use enthalpy::{
    PhaseChainProperties, PhaseState, chained_enthalpy_from_temperature,
    chained_state_from_enthalpy, enthalpy_from_temperature,
    temperature_and_phase_fraction_from_enthalpy,
};
pub use frictional_heating::{specific_energy_grid_to_si, temperature_rise_from_dissipation};
pub use granular_fluidity::{GranularFluidityConfig, GranularFluidityField};
pub use scalar_field::{ScalarDiffusionConfig, ScalarDiffusionField};
pub use transfer::{
    STEFAN_BOLTZMANN, entropy_change_heat_transfer, entropy_change_irreversible, heat_conduction,
    heat_radiation, saturating_uptake, second_law_holds, thermal_diffusivity,
};

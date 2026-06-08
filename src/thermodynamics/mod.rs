//! Thermodynamics — heat and generic scalar transport, MPM-coupled.
//!
//! - `diffusion.rs`    — Fourier heat diffusion ∂T/∂t = α∇²T + Newton cooling
//! - `scalar_field.rs` — generic ∂φ/∂t = D·∇²φ − λ·φ + S (pheromone, nutrients, morphogen)
//! - `transfer.rs`     — scalar IRL primitives: conduction, Stefan-Boltzmann radiation, entropy/2nd law

pub mod diffusion;
pub mod scalar_field;
pub mod transfer;

pub use diffusion::{ThermalConfig, ThermalDiffusion};
pub use scalar_field::{ScalarDiffusionConfig, ScalarDiffusionField};
pub use transfer::{
    STEFAN_BOLTZMANN, entropy_change_heat_transfer, entropy_change_irreversible, heat_conduction,
    heat_radiation, second_law_holds, thermal_diffusivity,
};

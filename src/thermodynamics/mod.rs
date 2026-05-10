//! Thermodynamics — heat and generic scalar transport, MPM-coupled.
//!
//! - `diffusion.rs`    — Fourier heat diffusion ∂T/∂t = α∇²T + Newton cooling
//! - `scalar_field.rs` — generic ∂φ/∂t = D·∇²φ − λ·φ + S (pheromone, nutrients, morphogen)

pub mod diffusion;
pub mod scalar_field;

pub use diffusion::{ThermalConfig, ThermalDiffusion};
pub use scalar_field::{ScalarDiffusionConfig, ScalarDiffusionField};

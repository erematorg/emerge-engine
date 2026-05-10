//! Acoustics — wave propagation through continuous media.
//!
//! Implements the scalar 2D wave equation ∂²u/∂t² = c²∇²u via finite differences.
//! Pressure waves, sound, seismic — all are instances of the same PDE kernel.

pub mod wave_equation;
pub use wave_equation::WaveEquation2D;

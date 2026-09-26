//! Acoustics -- wave propagation through continuous media.
//!
//! Implements the scalar 2D wave equation ∂²u/∂t² = c²∇²u via finite differences.
//! Pressure waves, sound, seismic -- all are instances of the same PDE kernel.
//!
//! `modal` -- real vibrational frequencies/damping derived from a rod's own
//! material properties (Euler-Bernoulli cantilever bending modes), the
//! physics half of zero-asset procedural sound. No audio synthesis/buffer
//! generation here -- that's real, separate, not-yet-started work.

pub mod modal;
pub mod wave_equation;
pub use modal::{AcousticMode, cantilever_rod_modes};
pub use wave_equation::WaveEquation2D;

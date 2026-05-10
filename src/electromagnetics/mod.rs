//! Electromagnetism — field math and wave interactions.
//!
//! Pairwise Coulomb forces are in `fields::coulomb`. This module handles
//! field queries and EM wave physics independent of the particle solver.

pub mod fields;
pub mod interactions;

pub use fields::{COULOMB_CONSTANT, ElectricField, MAGNETIC_CONSTANT_DIV_4PI, MagneticField};
pub use interactions::{C, ElectromagneticWave, MaterialProperties};

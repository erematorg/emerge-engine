//! Classical field implementations — spatial sources that apply acceleration to particles.
//!
//! All positions are in grid coordinates (same units as `Particle::x`).
//! The `ForceField` trait is defined here — it is the substep hook for external body forces.
//! Dependency is one-way: fields → core particle, never reverse.

/// Fraction of `cutoff` at which the force-switch fade begins.
/// 0.85 matches the GROMACS/LAMMPS force-switch convention: 15% taper range avoids
/// energy discontinuities without wasting usable interaction radius.
pub(crate) const FADE_ONSET_RATIO: f32 = 0.85;

pub mod buoyancy;
pub mod chemotaxis;
pub mod confinement;
pub mod coulomb;
pub mod em;
mod force_field;
pub mod gravity;
pub mod n_body;

pub use buoyancy::BuoyancyField;
pub use chemotaxis::ChemotaxisField;
pub use confinement::{AabbConfinementField, RadialConfinementField};
pub use coulomb::CoulombField;
pub use em::UniformElectricField;
pub use force_field::ForceField;
pub use gravity::GravityWellField;
pub use n_body::NBodyGravityField;

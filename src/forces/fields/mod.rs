//! Classical field implementations -- spatial sources that apply acceleration to particles.
//!
//! All positions are in grid coordinates (same units as `Particle::x`).
//! The `Field` trait is defined here -- it is the substep hook for external body forces.
//! Dependency is one-way: fields → core particle, never reverse.

/// Fraction of `cutoff` at which the force-switch fade begins.
/// 0.85 is a practical engineering choice, not a fixed cross-tool standard --
/// real switching-function taper ratios vary by force field/tool (commonly
/// ~0.67-0.83 in MD packages), so this is honestly disclosed rather than
/// mis-attributed: a 15% taper range avoids an abrupt cutoff without giving
/// up much usable interaction radius.
pub(crate) const FADE_ONSET_RATIO: f32 = 0.85;

pub mod buoyancy;
pub mod chemotaxis;
pub mod confinement;
pub mod coulomb;
pub mod cutoff;
pub mod drag;
pub mod em;
mod force_field;
mod grain_field;
pub mod gravity;
pub mod n_body;

pub use buoyancy::BuoyancyField;
pub use chemotaxis::ChemotaxisField;
pub use confinement::{AabbConfinementField, RadialConfinementField};
pub use coulomb::CoulombField;
pub use drag::{LinearDragField, SpatialDragField};
pub use em::UniformElectricField;
pub use force_field::Field;
pub use grain_field::GrainField;
pub use gravity::GravityWellField;
pub use n_body::NBodyGravityField;

//! Classical field implementations -- spatial sources that apply acceleration to particles.
//!
//! All positions are in grid coordinates (same units as `Particle::x`).
//! The `Field` trait is defined here -- it is the substep hook for external body forces.
//! Dependency is one-way: fields → core particle, never reverse.

pub mod buoyancy;
pub mod coulomb;
pub mod drag;
pub mod em;
mod force_field;
mod grain_field;
pub mod gravity;
pub mod gyroscopic;
pub mod n_body;

pub use buoyancy::BuoyancyField;
pub use coulomb::CoulombField;
pub use drag::{LinearDragField, SpatialDragField};
pub use em::UniformElectricField;
pub use force_field::Field;
pub use grain_field::GrainField;
pub use gravity::{EARTH_GRAVITY_M_S2, GravityWellField};
pub use gyroscopic::{
    CoriolisField, EARTH_ROTATION_RATE_RAD_S, RotatingSlice, UniformMagneticField,
};
pub use n_body::NBodyGravityField;

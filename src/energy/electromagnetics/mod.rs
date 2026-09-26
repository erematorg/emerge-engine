//! Electromagnetics -- the Energy half of `electromagnetics::`.
//!
//! Split into a folder 2026-08-27 (was one 637-line file) so each real,
//! physically distinct piece gets its own home instead of growing inside a
//! single flat file as lightning/dielectric-breakdown work builds out:
//!
//! - `wave.rs` -- `ElectromagneticWave` (plane-wave E/B fields), optical
//!   `MaterialProperties` (permittivity, permeability, conductivity,
//!   refractive index)
//! - `potential_field.rs` -- `ElectricPotentialField`, a generic
//!   steady-state electric-potential solver (Laplace's equation, Jacobi
//!   relaxation)
//! - `leader.rs` -- `DielectricBreakdownLeader`, the NPW 1984
//!   dielectric-breakdown model (lightning-leader growth) built on top of
//!   `potential_field`
//!
//! Point-charge/current force-application math is `forces::electromagnetics`
//! instead -- see this crate's own domain-taxonomy doc (`energy::` owns
//! radiative/field energy transfer, `forces::` owns force application).

pub mod leader;
pub mod potential_field;
pub mod wave;

pub use leader::DielectricBreakdownLeader;
pub use potential_field::ElectricPotentialField;
pub use wave::{C, ElectromagneticWave, MaterialProperties};

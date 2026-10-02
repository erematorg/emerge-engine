//! Matter domain: what things are made of.
//!
//! `materials` -- constitutive models, `MaterialModel` trait, `MaterialRegistry`.
//! `particle` -- the `Particle` struct, the per-particle state every model reads/writes.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use matter::materials;` etc in `lib.rs`) so
//! every existing `crate::materials::`/`crate::particle::` path and every LP
//! `emerge::materials::`/`emerge::particle::` path keeps resolving unchanged --
//! this move only changes where the files physically live, not any public API.

pub mod materials;
pub mod particle;

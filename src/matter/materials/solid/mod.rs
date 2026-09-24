//! Solid-state materials — real elastic/plastic deformation, no equation-
//! of-state bulk-fluid pressure term.
//!
//! Real solid/liquid/gas/plasma/mixture taxonomy restructuring of
//! `matter/materials/`, 2026-08-18. See the design artifact for the full
//! plan and the real classification of every existing material:
//! <https://claude.ai/code/artifact/90290560-8992-4d7c-ae4b-11ede12a737f>
//!
//! Contents: `elastic.rs` (NeoHookean), `corotated.rs`, `viscoelastic.rs`,
//! `von_mises.rs`, `rankine.rs`, `no_compression.rs`, `snow.rs`, `nacc.rs`,
//! `granular/` (sand, sand_mui, cosserat, grain_contact_law,
//! scale_contract — real external call sites in `spacetime::grains`,
//! `spacetime::solver`, and `energy::thermodynamics::cosserat_field` fixed
//! to the new `solid::granular::` path when this moved).

pub mod corotated;
pub mod elastic;
pub mod granular;
pub mod nacc;
pub mod no_compression;
pub mod rankine;
pub mod snow;
pub mod viscoelastic;
pub mod von_mises;

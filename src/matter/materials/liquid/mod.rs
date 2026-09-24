//! Liquid-state materials — weakly-compressible, real Tait equation of
//! state (a real bulk stiffness plus a nonzero rest pressure at ρ=ρ₀).
//!
//! PLACEHOLDER (2026-08-17): folder scaffolding for the real
//! solid/liquid/gas/plasma/mixture taxonomy restructuring of
//! `matter/materials/` — not yet wired into `materials/mod.rs`, no files
//! moved here yet. See the design artifact for the full plan and the
//! real classification of every existing material:
//! <https://claude.ai/code/artifact/90290560-8992-4d7c-ae4b-11ede12a737f>
//!
//! Contents: `fluid.rs` (NewtonianFluid), `bingham.rs` (viscoplastic,
//! yield stress). Moved here 2026-08-18, first group of the real
//! restructuring.

pub mod bingham;
pub mod fluid;

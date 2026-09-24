//! Mixture materials — real substances genuinely combining two states in
//! one constitutive law, not forced into either `solid/` or `liquid/`.
//! Real materials-science precedent: a saturated soil is a genuine mixture
//! of two states, not a fifth state of matter.
//!
//! Real solid/liquid/gas/plasma/mixture taxonomy restructuring of
//! `matter/materials/`, 2026-08-18. This category's placement — beside
//! the four states, not nested — see the design artifact:
//! <https://claude.ai/code/artifact/90290560-8992-4d7c-ae4b-11ede12a737f>
//!
//! Contents: `granular_fluid.rs` (`GranularFluidMaterial`) — its own doc:
//! `τ = τ_EOS(liquid) + τ_corotated(solid)` (Dunatunga & Kamrin 2015). A
//! DIFFERENT mixture concept from the engine's existing `MixturePhase`/
//! `WithMixturePhase` (two separate materials/particle populations
//! coupled by Darcy drag) — this one blends both terms inside ONE
//! particle's own stress formula.

pub mod granular_fluid;

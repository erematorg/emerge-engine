//! Systems domain: pure orchestration with NO IRL counterpart of its own --
//! the deliberate exception to the IRL-grounding mandate that governs every
//! other domain here, carved out on purpose, not an oversight.
//!
//! `diagnostics` -- health monitoring, NDJSON logging, plugin-based stats
//! collection: engineering observability, not physics. `gpu`
//! [feature = "gpu"] -- `GpuSimulation` + WGSL compute shaders: backend
//! plumbing. `render` [feature = "render"] -- instanced particle debug draw:
//! pipeline setup, not the physics it visualizes.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use systems::diagnostics;` etc in
//! `lib.rs`) so every existing `crate::diagnostics::`/`crate::gpu::`/
//! `crate::render::` path and every LP equivalent keeps resolving unchanged
//! -- this move only changes where the files physically live, not any
//! public API.

pub mod diagnostics;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "render")]
pub mod render;

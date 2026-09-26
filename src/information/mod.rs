//! Information domain: what senses, decides, and remembers -- IN the
//! simulated world, not tooling that observes it from outside.
//!
//! `control` -- `Lnn` (Liquid Time-constant Network locomotion controller): a
//! genome/weights IS information, and this is a real in-world decision-making
//! system. `measures` [feature = "experimental"] -- O(N) entropy
//! (spatial/kinetic/phase), local mutual information, KL divergence:
//! literally information theory applied to real simulated quantities.
//!
//! `diagnostics` (NDJSON logging, health thresholds, stats collection) is
//! deliberately NOT here -- it has no IRL counterpart of its own (no physical
//! law describes a `FrameLogger`), it's pure engineering observability, same
//! category as GPU/render plumbing. See `systems::diagnostics` instead.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use information::control;` etc in
//! `lib.rs`) so every existing `crate::control::` path and every LP
//! `emerge::control::` path keeps resolving unchanged -- this move only
//! changes where the files physically live, not any public API.

pub mod control;
#[cfg(feature = "experimental")]
pub mod measures;

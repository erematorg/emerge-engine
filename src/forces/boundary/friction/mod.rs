//! Coulomb wall friction family -- three real variants sharing one base
//! mechanism (`apply_coulomb_wall`, defined in `boundary`'s own `mod.rs`,
//! one level up): plain friction, muscle-phase-gated grip, and directional
//! (anisotropic) ratchet friction. `grip_friction` builds directly on
//! `friction::FrictionBoundary`; `ratchet_friction` is an independent
//! variant of the same base mechanism, not a further specialization of
//! `grip_friction`.

mod base;
mod grip_friction;
mod ratchet_friction;

pub use base::FrictionBoundary;
pub use grip_friction::GripFrictionBoundary;
pub use ratchet_friction::RatchetFrictionBoundary;

//! Coulomb friction family, sharing one base mechanism (`apply_coulomb_wall`
//! and its anisotropic form `apply_anisotropic_coulomb_wall`, defined in
//! `boundary`'s own `mod.rs`, one level up): isotropic wall friction
//! (`FrictionBoundary`), directional friction on the floor
//! (`RatchetFrictionBoundary`), and the same directional law for body-to-body
//! contact (`DirectionalContactGrip`).

mod base;
mod directional_contact;
mod ratchet_friction;

pub use base::FrictionBoundary;
pub use directional_contact::DirectionalContactGrip;
pub use ratchet_friction::RatchetFrictionBoundary;

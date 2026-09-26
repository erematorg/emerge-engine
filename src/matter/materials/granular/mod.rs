//! Granular constitutive models and their supporting physics -- grouped by
//! active research thread (dry-sand repose-angle/rolling-resistance
//! investigation), not by rheological class alone. `sand` (Drucker-Prager)
//! and `sand_mui` (µ(I) rheology) are the two constitutive models proper;
//! `cosserat` (micropolar grain kinematics) and `grain_contact_law` (DEM
//! contact force law) are the two real candidate mechanisms built to close
//! sand's self-arrest gap; `scale_contract` is the REV grid-resolution
//! validity check written specifically for grain-diameter scenes.
//!
//! Real, disclosed axis conflict (see the taxonomy artifact this grouping
//! came from): `sand` classifies as an ordinary elastoplastic solid under a
//! pure rheological taxonomy, same bucket as `von_mises`/`rankine`/`nacc` one
//! level up in `materials/`. It lives here instead because it's actively
//! co-developed with `cosserat`/`grain_contact_law` every session, not
//! because rheology says so -- grouping by what you look things up WITH,
//! not just what a thing formally IS.

pub mod cosserat;
pub mod grain_contact_law;
pub mod sand;
pub mod sand_mui;
pub mod scale_contract;

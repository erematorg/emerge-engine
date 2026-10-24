//! Granular materials and their supporting physics. Granular materials
//! "have properties that are different from those commonly associated with
//! either solids, liquids, or gases" (Jaeger, Nagel & Behringer 1996,
//! "Granular solids, liquids, and gases", Rev. Mod. Phys. 68, 1259), so this
//! folder is a peer of `solid`, `liquid`, `gas` and `mixture`, not a kind of
//! solid.
//!
//! `sand` (Drucker-Prager) and `sand_mui` (µ(I) rheology) are the
//! constitutive models: continuum elastoplastic like the laws in `solid`,
//! but with frictional, pressure-dependent yield. `cosserat` (micropolar
//! grain kinematics) and `grain_contact_law` (DEM contact force law) are the
//! two candidate mechanisms for sand's self-arrest, developed together with
//! it; `disc_contact` is the elastic normal contact of 2D grains;
//! `scale_contract` is the REV grid-resolution check for grain-diameter
//! scenes.

pub mod cosserat;
pub mod disc_contact;
pub mod grain_contact_law;
pub mod sand;
pub mod sand_mui;
pub mod scale_contract;

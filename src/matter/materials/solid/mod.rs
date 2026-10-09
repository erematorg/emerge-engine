//! Solid constitutive laws: continuum models that compute stress from the
//! deformation gradient, with an elastic, elastic-plastic, hardening or
//! damage response. Hyperelastic (`elastic` NeoHookean, `corotated`),
//! viscoelastic (`viscoelastic`, Kelvin-Voigt), yield and fracture
//! (`von_mises`, `rankine`), tension-only (`no_compression`), cap plasticity
//! (`nacc`, Cam-Clay) and hardening-cohesion snow (`snow`, Stomakhin).
//!
//! Grouped by what the model computes, not by the everyday name of the
//! substance: snow and Cam-Clay soils are solids here because their laws are
//! continuum elastic-plastic ones, while sand's friction-dilatancy laws live
//! in `granular`.

pub mod corotated;
pub mod elastic;
pub mod nacc;
pub mod no_compression;
pub mod rankine;
pub mod snow;
pub mod viscoelastic;
pub mod von_mises;

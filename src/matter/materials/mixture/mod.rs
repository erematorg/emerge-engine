//! Mixture laws: multiphase materials, where two phases coexist in one
//! continuum and the law tracks both. `granular_fluid` is a granular solid
//! with its pore liquid (saturated soil, mud); `boiling_mixture` and
//! `cavitating_fluid` are liquid water with its vapour, sharing the
//! liquid-vapour equation of state of `cavitating_eos`.

pub mod boiling_mixture;
pub mod cavitating_eos;
pub mod cavitating_fluid;
pub mod granular_fluid;

//! Liquid constitutive laws: a weakly compressible pressure from the volume
//! change plus a viscous stress from the strain rate. `fluid` is the
//! Newtonian liquid (Tait equation of state, shear and bulk viscosity);
//! `bingham` adds a yield stress below which the material holds its shape
//! (viscoplastic fluids such as mud or paste).

pub mod bingham;
pub mod fluid;

//! Kinematics: how matter moves and deforms, whatever it is made of.
//!
//! These relations hold for every material. What a material does in response
//! to a deformation (its constitutive law) lives in `matter::materials`.
//!
//! Source: Gurtin, Fried & Anand, *The Mechanics and Thermodynamics of
//! Continua* (2010), part III (kinematics), kept apart from the balance laws
//! (part IV) and the constitutive theories (parts VII onward).

pub(crate) mod deformation_gradient;
pub(crate) mod hencky_strain;
pub mod polar_decomposition;
pub(crate) mod volume_ratio;

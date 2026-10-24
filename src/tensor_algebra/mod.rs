//! Vector and tensor algebra: the mathematics every physical domain is
//! written in. Nothing here is a physical law; it is the toolkit the laws use.
//!
//! Kept apart from the physics the way continuum mechanics texts open with it
//! as its own part before any kinematics or balance law (Gurtin, Fried &
//! Anand, *The Mechanics and Thermodynamics of Continua*, 2010, parts I-II).

pub(crate) mod dense;
pub(crate) mod frobenius;
pub mod matrix2;
pub(crate) mod matrix_exponential;
pub(crate) mod svd;
pub(crate) mod symmetric_eigen;

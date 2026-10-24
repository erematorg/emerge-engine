//! Hencky (logarithmic) strain: the logarithm of each principal stretch.
//!
//! Source: Gurtin, Fried & Anand (2010), ch. 7 (stretch, strain and rotation).

use glam::Vec2;

/// Floor applied to singular values before taking log -- prevents ln(0).
/// All material `update_particle` implementations clamp σᵢ above this value.
pub(crate) const LOG_CLAMP: f32 = 1e-10;

/// Compute Hencky (logarithmic) strains from SVD singular values.
///
/// ε_i = ln(|σ_i|), clamped above 1e-10 to avoid ln(0).
/// The absolute value preserves Hencky strain magnitudes when `svd2()` encodes
/// an inversion via a signed second singular value.
/// Used identically by VonMisesMaterial, DruckerPragerMaterial, RankineMaterial.
#[inline]
pub(crate) fn hencky_strains(sigma: Vec2) -> Vec2 {
    let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
    Vec2::new(sigma.x.ln(), sigma.y.ln())
}

//! Beer-Lambert attenuation: what a medium takes out of light passing through.
//!
//! The counterpart of `super::blackbody`, which says what a medium *adds*.
//! Both are energy transport, and both are stated in SI so a coefficient
//! measured in a laboratory can be used unchanged.
//!
//! The coefficients themselves belong to the substance, not to this module
//! and not to the renderer: see `matter::materials::optical` for measured
//! values and for how a material declares its own.

use std::{error::Error, fmt};

/// Per-material absorption/scattering coefficients in inverse metres.
///
/// Reduced scattering stays one visible-band scalar. The unit is explicit; a
/// later spectral transport upgrade can widen it without changing the
/// absorption contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalCoefficientsSi {
    pub absorption_m_inv: [f32; 3],
    pub reduced_scattering_m_inv: f32,
}

impl OpticalCoefficientsSi {
    pub fn new(
        absorption_m_inv: [f32; 3],
        reduced_scattering_m_inv: f32,
    ) -> Result<Self, OpticalCoefficientsError> {
        if absorption_m_inv
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(OpticalCoefficientsError::InvalidAbsorption);
        }
        if !reduced_scattering_m_inv.is_finite() || reduced_scattering_m_inv < 0.0 {
            return Err(OpticalCoefficientsError::InvalidReducedScattering);
        }
        Ok(Self {
            absorption_m_inv,
            reduced_scattering_m_inv,
        })
    }
}

/// Exact absorption-only solution for a homogeneous slab.
///
/// `relative_density` is `rho/rho_ref`, so the exponent is dimensionless:
/// `(m^-1) * 1 * m`. This is the reference used by CPU and GPU validation;
/// it does not include scattering, reflection, or emission.
pub fn beer_lambert_transmittance(
    absorption_m_inv: [f32; 3],
    relative_density: f32,
    path_length_meters: f32,
) -> [f32; 3] {
    let column_length = relative_density.max(0.0) * path_length_meters.max(0.0);
    absorption_m_inv.map(|sigma| (-sigma.max(0.0) * column_length).exp())
}

/// Radiance leaving a slab of absorbing, scattering medium, in whatever unit
/// `background` and `incident` are given in.
///
/// Three real terms, in the order light meets them:
///
/// - Fresnel reflection at the front face. What reflects never enters, so it
///   carries the incident radiance straight back and the two interior terms
///   are weighted by `1 - R`.
/// - Direct transmission of whatever is behind the slab, attenuated by the
///   full extinction `sigma_t = sigma_a + sigma_s`. Scattering removes light
///   from the direct beam just as absorption does, which is why the exponent
///   is not `sigma_a` alone.
/// - Single scattering. Of everything the slab removed from the beam,
///   `sigma_s / sigma_t` -- the single-scattering albedo -- was scattered
///   rather than absorbed. Lit by the incident radiance, so the glow takes
///   the colour of the actual light source rather than a chosen tint.
///   (Jacques, "Optical properties of biological tissues: a review", Phys.
///   Med. Biol. 58, 2013.)
///
/// `cos_view` is measured from the surface normal; pass 1.0 where no normal
/// exists, which reduces Fresnel to its normal-incidence value.
///
/// Disclosed simplification: single scattering with an isotropic phase
/// function and no multiple-scattering term, so a very dense, highly
/// scattering medium is under-lit. Fixing that means adding the diffusion
/// term, not a constant.
///
/// `systems/render/shaders/radiative_transfer.inc.wgsl` mirrors this
/// function for the GPU.
pub fn slab_radiance(
    background: [f32; 3],
    incident: [f32; 3],
    absorption_m_inv: [f32; 3],
    reduced_scattering_m_inv: f32,
    path_length_meters: f32,
    fresnel_r0: f32,
    cos_view: f32,
) -> [f32; 3] {
    let sigma_s = reduced_scattering_m_inv.max(0.0);
    let path = path_length_meters.max(0.0);
    let reflectance = super::fresnel::schlick_reflectance(fresnel_r0, cos_view);
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let sigma_t = absorption_m_inv[channel].max(0.0) + sigma_s;
        let transmitted = (-sigma_t * path).exp();
        let albedo = sigma_s / sigma_t.max(1.0e-9);
        let interior =
            background[channel] * transmitted + incident[channel] * albedo * (1.0 - transmitted);
        out[channel] = interior * (1.0 - reflectance) + incident[channel] * reflectance;
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalCoefficientsError {
    InvalidAbsorption,
    InvalidReducedScattering,
}

impl fmt::Display for OpticalCoefficientsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidAbsorption => "absorption coefficients must be finite and non-negative",
            Self::InvalidReducedScattering => "reduced scattering must be finite and non-negative",
        };
        f.write_str(message)
    }
}

impl Error for OpticalCoefficientsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beer_lambert_matches_known_homogeneous_slab() {
        let transmittance = beer_lambert_transmittance([2.0, 1.0, 0.5], 1.0, 0.25);
        let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
        for (got, want) in transmittance.into_iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-6, "got {got}, expected {want}");
        }
    }

    /// With no scattering and no reflection, the slab model must collapse
    /// exactly onto Beer-Lambert -- the general law has to contain the
    /// special case, or one of the two is wrong.
    #[test]
    fn slab_reduces_to_beer_lambert_without_scattering_or_reflection() {
        let sigma = [0.35, 0.033, 0.011];
        let background = [0.7, 0.8, 0.9];
        let plain = beer_lambert_transmittance(sigma, 1.0, 2.0);
        let slab = slab_radiance(background, [1.0; 3], sigma, 0.0, 2.0, 0.0, 1.0);
        for channel in 0..3 {
            let expected = background[channel] * plain[channel];
            assert!(
                (slab[channel] - expected).abs() < 1.0e-6,
                "channel {channel}: slab {} vs Beer-Lambert {expected}",
                slab[channel]
            );
        }
    }

    /// A slab that reflects everything shows only the light falling on it.
    #[test]
    fn total_reflection_hides_whatever_is_behind() {
        let incident = [0.2, 0.4, 0.6];
        let slab = slab_radiance([1.0; 3], incident, [0.5; 3], 1.0, 3.0, 1.0, 1.0);
        for channel in 0..3 {
            assert!((slab[channel] - incident[channel]).abs() < 1.0e-6);
        }
    }

    /// Scattering brightens a medium lit more strongly than its background,
    /// which is the whole reason milk is white and not grey.
    #[test]
    fn scattering_brightens_a_medium_lit_from_the_front() {
        let dark_background = [0.0; 3];
        let incident = [1.0; 3];
        let absorbing = slab_radiance(dark_background, incident, [0.5; 3], 0.0, 2.0, 0.0, 1.0);
        let scattering = slab_radiance(dark_background, incident, [0.5; 3], 5.0, 2.0, 0.0, 1.0);
        for channel in 0..3 {
            assert!(
                scattering[channel] > absorbing[channel] + 0.1,
                "channel {channel}: scattering {} should clearly exceed {}",
                scattering[channel],
                absorbing[channel]
            );
        }
    }

    #[test]
    fn beer_lambert_is_invariant_to_spatial_discretization() {
        let sigma = [0.35, 0.033, 0.011];
        let physical_length = 2.0;
        let analytic = beer_lambert_transmittance(sigma, 1.0, physical_length);

        for cells in [2usize, 20, 200, 2_000] {
            let dx = physical_length / cells as f32;
            let accumulated_tau = sigma.map(|s| (0..cells).map(|_| s * dx).sum::<f32>());
            let discrete = accumulated_tau.map(|tau| (-tau).exp());
            for (got, want) in discrete.into_iter().zip(analytic) {
                assert!(
                    (got - want).abs() < 5.0e-5,
                    "{cells} cells changed transmittance: got {got}, expected {want}"
                );
            }
        }
    }
}

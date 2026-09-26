//! Planck's law: the light matter emits because it is hot.
//!
//! Nothing here knows about pixels, displays, or human vision. Reducing a
//! spectrum to a colour needs the CIE standard observer, a measurement of
//! human photoreceptors rather than of the world; that is `super::spectrum`.
//!
//! Every constant is an SI-defining exact value (2019 SI redefinition).

use crate::energy::thermodynamics::STEFAN_BOLTZMANN;

/// Planck constant h -- J.s (exact by definition since 2019).
pub const PLANCK: f64 = 6.626_070_15e-34;
/// Speed of light in vacuum c -- m/s (exact by definition).
pub const SPEED_OF_LIGHT: f64 = 2.997_924_58e8;
/// Boltzmann constant k_B -- J/K (exact by definition since 2019).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Wien displacement-law constant b -- m.K (CODATA 2018).
pub const WIEN_DISPLACEMENT: f64 = 2.897_771_955e-3;

/// Planck's law: spectral radiance of a blackbody, W / (m^2 . sr . m).
///
/// `B(lambda, T) = (2hc^2 / lambda^5) / (exp(hc / (lambda k_B T)) - 1)`
///
/// Returns 0 for non-physical inputs and in the deep Wien tail, where the
/// exponent overflows `f64` long before the value is representable -- the
/// emitted power there is below `f64::MIN_POSITIVE` and genuinely negligible,
/// not clipped for convenience.
pub fn planck_spectral_radiance(wavelength_m: f64, temperature_k: f64) -> f64 {
    if !wavelength_m.is_finite()
        || wavelength_m <= 0.0
        || !temperature_k.is_finite()
        || temperature_k <= 0.0
    {
        return 0.0;
    }
    let exponent = PLANCK * SPEED_OF_LIGHT / (wavelength_m * BOLTZMANN * temperature_k);
    if exponent > 700.0 {
        return 0.0;
    }
    let numerator = 2.0 * PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    numerator / (wavelength_m.powi(5) * (exponent.exp() - 1.0))
}

/// Wavelength of peak spectral radiance -- Wien's displacement law, `b / T`.
pub fn wien_peak_wavelength_m(temperature_k: f64) -> f64 {
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return f64::INFINITY;
    }
    WIEN_DISPLACEMENT / temperature_k
}

/// Radiance of a Lambertian blackbody integrated over all wavelengths --
/// W / (m^2 . sr).
///
/// The Stefan-Boltzmann law gives the hemispherical exitance `M = sigma T^4`;
/// a Lambertian emitter spreads that over pi steradians, so `L = sigma T^4 / pi`.
/// This is the quantity a camera or an eye actually measures, and the one the
/// renderer divides by its display-white radiance to get an exposure.
pub fn blackbody_radiance_w_m2_sr(temperature_k: f32) -> f32 {
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return 0.0;
    }
    let t2 = temperature_k * temperature_k;
    STEFAN_BOLTZMANN * t2 * t2 / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Planck's law must reproduce Wien's displacement law -- two independent
    /// statements about the same spectrum, so agreement is a real check on the
    /// constants and the formula, not a tautology.
    #[test]
    fn planck_peak_matches_wien_displacement_law() {
        for temperature in [1000.0, 2500.0, 5772.0, 10_000.0] {
            let predicted = wien_peak_wavelength_m(temperature);
            let mut best = (0.0, 0.0f64);
            // Scan a decade either side of the prediction at 0.1 nm steps.
            let mut lambda = predicted * 0.1;
            while lambda < predicted * 10.0 {
                let radiance = planck_spectral_radiance(lambda, temperature);
                if radiance > best.1 {
                    best = (lambda, radiance);
                }
                lambda += 1.0e-10;
            }
            let relative_error = (best.0 - predicted).abs() / predicted;
            assert!(
                relative_error < 1.0e-3,
                "{temperature}K: scanned peak {:.4}nm vs Wien {:.4}nm",
                best.0 * 1.0e9,
                predicted * 1.0e9
            );
        }
    }

    /// Integrating Planck's law over all wavelengths must reproduce
    /// Stefan-Boltzmann -- the same cross-check in the other direction, and the
    /// one that validates `blackbody_radiance_w_m2_sr`'s `/pi`.
    #[test]
    fn planck_integral_matches_stefan_boltzmann() {
        let temperature = 3000.0;
        // Integrate on a log-spaced grid: the spectrum spans decades.
        let (mut integral, steps) = (0.0f64, 20_000);
        let (log_lo, log_hi) = ((1.0e-8f64).ln(), (1.0e-3f64).ln());
        let step = (log_hi - log_lo) / steps as f64;
        for i in 0..steps {
            let a = (log_lo + step * i as f64).exp();
            let b = (log_lo + step * (i + 1) as f64).exp();
            let mid = 0.5 * (a + b);
            integral += planck_spectral_radiance(mid, temperature) * (b - a);
        }
        let expected = f64::from(STEFAN_BOLTZMANN) * temperature.powi(4) / std::f64::consts::PI;
        let relative_error = (integral - expected).abs() / expected;
        assert!(
            relative_error < 1.0e-3,
            "integrated radiance {integral:.6e} vs sigma T^4/pi {expected:.6e}"
        );
    }

    /// The sun's photosphere (5772 K) peaks in the visible, near green-yellow.
    /// A real, checkable number rather than a self-consistency test.
    #[test]
    fn solar_peak_lands_in_the_visible_band() {
        let peak_nm = wien_peak_wavelength_m(5772.0) * 1.0e9;
        assert!(
            (495.0..510.0).contains(&peak_nm),
            "solar peak came out at {peak_nm:.1}nm"
        );
    }
}

//! Spectrum to colour: radiometry becoming photometry.
//!
//! `super::blackbody` says how much power a hot body emits at each wavelength
//! (Planck's law). This module says what that spectrum amounts to as a
//! colour, which needs one thing physics alone does not supply: the response
//! of human photoreceptors, i.e. the CIE 1931 standard observer.
//!
//! This is still the energy domain, not the render domain. A colour derived
//! from a spectrum is a property of the light, and `systems::render` consumes
//! it rather than deriving its own.
//!
//! Two paths compute the same colour:
//!
//! - [`blackbody_linear_srgb`] integrates real Planck radiance against the
//!   observer. It is the ground truth, and too expensive for a fragment
//!   shader (hundreds of `exp` calls per pixel).
//! - [`blackbody_linear_srgb_locus_fit`] evaluates a published closed-form
//!   fit to the Planckian locus. It is what `blackbody.inc.wgsl` implements,
//!   line for line, so the GPU and the CPU agree.
//!
//! `locus_fit_matches_planck_integration` is the test that keeps the second
//! honest about the first.
//!
//! References:
//! - Wyman, Sloan, Shirley, "Simple Analytic Approximations to the CIE XYZ
//!   Color Matching Functions", Journal of Computer Graphics Techniques 2(2),
//!   2013 -- the multi-lobe Gaussian fit used by [`cie_1931_observer`].
//! - Kim, Garcia-Hansen, Moon, Lee, "Design of Advanced Color Temperature
//!   Control System for HDTV Applications", Journal of the Korean Physical
//!   Society 41(6), 2002 -- the cubic-spline Planckian locus used by
//!   [`blackbody_linear_srgb_locus_fit`], valid over 1667 K to 25000 K.
//! - IEC 61966-2-1 (sRGB) -- the XYZ to linear-RGB matrix, D65 white point.

use super::blackbody::planck_spectral_radiance;

/// Lowest temperature the Kim 2002 locus fit is defined for. Below this the
/// fit is evaluated at its own lower bound: a blackbody there is a deep
/// orange-red whose chromaticity barely moves, while its radiance has already
/// fallen by orders of magnitude under `T^4`, so the visible error is
/// dominated by the brightness term, not the hue.
pub const LOCUS_FIT_MIN_K: f32 = 1667.0;
/// Highest temperature the Kim 2002 locus fit is defined for.
pub const LOCUS_FIT_MAX_K: f32 = 25_000.0;

/// CIE 1931 2-degree standard observer colour matching functions.
///
/// Wyman/Sloan/Shirley 2013's multi-lobe fit: each lobe is a Gaussian whose
/// width differs either side of its centre, which is how an asymmetric
/// photoreceptor response is captured without a 471-entry table.
pub fn cie_1931_observer(wavelength_nm: f64) -> [f64; 3] {
    fn lobe(x: f64, center: f64, inverse_width_lo: f64, inverse_width_hi: f64) -> f64 {
        let width = if x < center {
            inverse_width_lo
        } else {
            inverse_width_hi
        };
        let t = (x - center) * width;
        (-0.5 * t * t).exp()
    }
    let l = wavelength_nm;
    [
        0.362 * lobe(l, 442.0, 0.0624, 0.0374) + 1.056 * lobe(l, 599.8, 0.0264, 0.0323)
            - 0.065 * lobe(l, 501.1, 0.0490, 0.0382),
        0.821 * lobe(l, 568.8, 0.0213, 0.0247) + 0.286 * lobe(l, 530.9, 0.0613, 0.0322),
        1.217 * lobe(l, 437.0, 0.0845, 0.0278) + 0.681 * lobe(l, 459.0, 0.0385, 0.0725),
    ]
}

/// CIE xy chromaticity of a blackbody, by integrating Planck's law against the
/// standard observer over the visible band.
///
/// 1 nm steps over 360-830 nm: the observer's narrowest lobe is ~25 nm wide,
/// so this resolves it with room to spare, and the whole integral runs once
/// per lookup-table entry, never per pixel.
pub fn blackbody_chromaticity_xy(temperature_k: f64) -> [f64; 2] {
    let mut xyz = [0.0f64; 3];
    for nm in 360..=830 {
        let wavelength_nm = f64::from(nm);
        let radiance = planck_spectral_radiance(wavelength_nm * 1.0e-9, temperature_k);
        let observer = cie_1931_observer(wavelength_nm);
        for channel in 0..3 {
            xyz[channel] += radiance * observer[channel];
        }
    }
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum <= 0.0 {
        // Colder than anything that emits measurable visible light.
        return [0.6, 0.4];
    }
    [xyz[0] / sum, xyz[1] / sum]
}

/// Spectral sensitivity of each linear sRGB channel, at one wavelength.
///
/// The CIE observer passed through the sRGB primaries: `r(l)`, `g(l)`, `b(l)`
/// say how much light at wavelength `l` each channel reports. Real sRGB
/// matching functions have negative lobes, because the primaries cannot
/// reproduce every visible colour; a negative *weight* is meaningless when
/// averaging a material property, so those are clamped to zero. That is a
/// stated approximation of band averaging, not of the observer itself.
pub fn srgb_channel_sensitivity(wavelength_nm: f64) -> [f64; 3] {
    let [x, y, z] = cie_1931_observer(wavelength_nm);
    [
        (3.2406 * x - 1.5372 * y - 0.4986 * z).max(0.0),
        (-0.9689 * x + 1.8758 * y + 0.0415 * z).max(0.0),
        (0.0557 * x - 0.2040 * y + 1.0570 * z).max(0.0),
    ]
}

/// Collapses a measured spectrum into one value per linear sRGB channel.
///
/// `samples` are `(wavelength_nm, value)` pairs in increasing wavelength, in
/// whatever unit the quantity has -- this returns the same unit, since it is
/// a weighted mean, not an integral. Each channel is weighted by its own
/// spectral sensitivity, so a quantity that varies steeply across the visible
/// band (water's absorption rises ~50x from blue to red) lands in the right
/// channel instead of being smeared by a flat block average.
///
/// Trapezoidal in wavelength, so unevenly spaced measurements are handled
/// correctly. A channel with no sensitivity over the sampled range returns 0.
pub fn spectral_band_average(samples: &[(f32, f32)]) -> [f32; 3] {
    let mut weighted = [0.0f64; 3];
    let mut weight = [0.0f64; 3];
    for pair in samples.windows(2) {
        let (lo_nm, lo_value) = (f64::from(pair[0].0), f64::from(pair[0].1));
        let (hi_nm, hi_value) = (f64::from(pair[1].0), f64::from(pair[1].1));
        let width = hi_nm - lo_nm;
        if width <= 0.0 {
            continue;
        }
        let mid_nm = 0.5 * (lo_nm + hi_nm);
        let mid_value = 0.5 * (lo_value + hi_value);
        let sensitivity = srgb_channel_sensitivity(mid_nm);
        for channel in 0..3 {
            weighted[channel] += sensitivity[channel] * mid_value * width;
            weight[channel] += sensitivity[channel] * width;
        }
    }
    [0, 1, 2].map(|channel| {
        if weight[channel] <= 0.0 {
            0.0
        } else {
            (weighted[channel] / weight[channel]) as f32
        }
    })
}

/// CIE xy chromaticity to linear sRGB, exposure-normalized so the brightest
/// channel is 1.
///
/// Out-of-gamut chromaticities (which the Planckian locus reaches at both
/// ends) are brought into gamut by desaturating toward white -- adding the
/// most negative channel to all three. That is a real, visible approximation:
/// sRGB simply cannot show those colours, and this keeps the hue while
/// admitting the loss of saturation, rather than clipping one channel to zero
/// and shifting the hue silently.
///
/// Brightness is deliberately NOT part of this: it comes from
/// `energy::radiation::blackbody_radiance_w_m2_sr`, so the two physical
/// factors stay separable.
pub fn chromaticity_to_linear_srgb(chromaticity_xy: [f64; 2]) -> [f32; 3] {
    let [x, y] = chromaticity_xy;
    if y <= 1.0e-6 {
        return [1.0, 1.0, 1.0];
    }
    // Unit-luminance XYZ for this chromaticity.
    let (big_x, big_y, big_z) = (x / y, 1.0, (1.0 - x - y) / y);
    // IEC 61966-2-1, D65.
    let mut rgb = [
        3.2406 * big_x - 1.5372 * big_y - 0.4986 * big_z,
        -0.9689 * big_x + 1.8758 * big_y + 0.0415 * big_z,
        0.0557 * big_x - 0.2040 * big_y + 1.0570 * big_z,
    ];
    let lowest = rgb[0].min(rgb[1]).min(rgb[2]);
    if lowest < 0.0 {
        for channel in &mut rgb {
            *channel -= lowest;
        }
    }
    let highest = rgb[0].max(rgb[1]).max(rgb[2]);
    if highest <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [
        (rgb[0] / highest) as f32,
        (rgb[1] / highest) as f32,
        (rgb[2] / highest) as f32,
    ]
}

/// Exposure-normalized linear sRGB of a blackbody, from real Planck radiance
/// integrated against the CIE observer. Ground truth for the shader fit.
pub fn blackbody_linear_srgb(temperature_k: f32) -> [f32; 3] {
    chromaticity_to_linear_srgb(blackbody_chromaticity_xy(f64::from(temperature_k)))
}

/// Exposure-normalized linear sRGB of a blackbody, via Kim et al. 2002's
/// closed-form Planckian locus.
///
/// This is the exact CPU mirror of `blackbody.inc.wgsl`'s `blackbody_srgb`.
/// Keeping both and testing them against each other is what lets a fragment
/// shader claim real blackbody colour without integrating a spectrum per
/// pixel.
pub fn blackbody_linear_srgb_locus_fit(temperature_k: f32) -> [f32; 3] {
    let t = f64::from(temperature_k.clamp(LOCUS_FIT_MIN_K, LOCUS_FIT_MAX_K));
    let (inv, inv2) = (1.0 / t, 1.0 / (t * t));
    let inv3 = inv2 * inv;
    let x = if t <= 4000.0 {
        -0.266_123_9e9 * inv3 - 0.234_358_9e6 * inv2 + 0.877_695_6e3 * inv + 0.179_910
    } else {
        -3.025_846_9e9 * inv3 + 2.107_037_9e6 * inv2 + 0.222_634_7e3 * inv + 0.240_390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_20 * x2 + 2.185_558_32 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_93 * x2 + 2.091_370_15 * x - 0.167_488_67
    } else {
        3.081_758_0 * x3 - 5.873_386_70 * x2 + 3.751_129_97 * x - 0.370_014_83
    };
    chromaticity_to_linear_srgb([x, y])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CIE standard illuminant A is *defined* as a Planckian radiator at
    /// 2856 K with chromaticity x = 0.44758, y = 0.40745. Reproducing it
    /// validates Planck's law, the constants, the observer fit and the
    /// integration together, against a number this codebase did not choose.
    #[test]
    fn reproduces_cie_illuminant_a_chromaticity() {
        let [x, y] = blackbody_chromaticity_xy(2856.0);
        assert!(
            (x - 0.447_58).abs() < 0.006 && (y - 0.407_45).abs() < 0.006,
            "illuminant A came out at ({x:.5}, {y:.5}), expected (0.44758, 0.40745)"
        );
    }

    /// D65's correlated colour temperature is ~6504 K. A true Planckian
    /// radiator there sits slightly below the daylight locus, near
    /// (0.3135, 0.3237) -- close to, but deliberately not equal to, D65's own
    /// (0.3127, 0.3290).
    #[test]
    fn reproduces_planckian_locus_near_daylight() {
        let [x, y] = blackbody_chromaticity_xy(6504.0);
        assert!(
            (x - 0.3135).abs() < 0.006 && (y - 0.3237).abs() < 0.006,
            "6504K came out at ({x:.5}, {y:.5}), expected (0.3135, 0.3237)"
        );
    }

    /// The shader's closed-form fit must agree with the real spectral
    /// integration across the range the engine actually renders: embers,
    /// flame, lava, incandescence, sunlight.
    #[test]
    fn locus_fit_matches_planck_integration() {
        for temperature in [1667.0f32, 2000.0, 2856.0, 3500.0, 4000.0, 5772.0, 9000.0] {
            let exact = blackbody_linear_srgb(temperature);
            let fit = blackbody_linear_srgb_locus_fit(temperature);
            for channel in 0..3 {
                assert!(
                    (exact[channel] - fit[channel]).abs() < 0.03,
                    "{temperature}K channel {channel}: exact {exact:?} vs fit {fit:?}"
                );
            }
        }
    }

    /// The single physical fact the old `heat()` ramp got backwards: hotter
    /// blackbodies get bluer, not redder.
    ///
    /// Checked twice. Chromaticity `x` must fall monotonically -- that is the
    /// Planckian locus itself, unaffected by any display. Blue over red must
    /// then not fall -- the same statement once sRGB has had its say. The two
    /// are not redundant: below ~2000 K the locus is outside the sRGB gamut,
    /// so `b` sits at 0 for several steps while `x` keeps moving.
    #[test]
    fn hotter_blackbodies_are_bluer() {
        let temperatures = [1200.0f32, 1800.0, 2500.0, 3500.0, 5000.0, 7000.0, 10_000.0];
        let mut previous_x = f64::INFINITY;
        let mut previous_ratio = f32::NEG_INFINITY;
        for temperature in temperatures {
            let [x, _] = blackbody_chromaticity_xy(f64::from(temperature));
            assert!(
                x < previous_x,
                "{temperature}K chromaticity x {x:.4} did not fall below {previous_x:.4}"
            );
            previous_x = x;

            let [r, _, b] = blackbody_linear_srgb(temperature);
            let ratio = b / r.max(1.0e-6);
            assert!(
                ratio >= previous_ratio,
                "{temperature}K blue/red {ratio:.4} fell below {previous_ratio:.4}"
            );
            previous_ratio = ratio;
        }
        // And the endpoints must actually differ, or "monotonic" is vacuous.
        let [.., cold_b] = blackbody_linear_srgb(1200.0);
        let [.., hot_b] = blackbody_linear_srgb(10_000.0);
        assert!(
            hot_b > cold_b + 0.5,
            "{cold_b:.3} -> {hot_b:.3} is too flat"
        );
    }

    /// An ember is red, the sun is near-white. Absolute sanity, not a ratio.
    #[test]
    fn ember_is_red_and_sunlight_is_near_white() {
        let [r, g, b] = blackbody_linear_srgb(1200.0);
        assert!(
            r > 0.99 && g < 0.55 && b < 0.2,
            "1200K should be deep red, got [{r:.3}, {g:.3}, {b:.3}]"
        );
        let [r, g, b] = blackbody_linear_srgb(5772.0);
        assert!(
            r.min(g).min(b) > 0.75,
            "solar temperature should be near-white, got [{r:.3}, {g:.3}, {b:.3}]"
        );
    }
}

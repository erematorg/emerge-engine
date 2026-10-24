//! What each substance does to light: measured optical constants.
//!
//! The laws live in `energy::radiation` -- Beer-Lambert for attenuation,
//! Planck for emission. What belongs here is the part that is a property of
//! the matter itself: how strongly this particular substance absorbs, at
//! which wavelengths.
//!
//! Spectra are stored as measured, in `(nm, m^-1)` pairs, and collapsed to
//! the three sRGB channels by `energy::radiation::spectral_band_average` at
//! the point of use. Storing the spectrum rather than three pre-averaged
//! numbers costs a few hundred bytes and keeps the door open to real
//! spectral rendering; it also means the three numbers a renderer ends up
//! with are derived, and can be re-derived, rather than typed in.
//!
//! A word on what these values look like in practice: pure water absorbs
//! about 0.62 m^-1 at 700 nm and 0.0044 m^-1 at its 418 nm minimum, a ratio
//! of ~140. Band-averaged to sRGB that comes out as roughly
//! `[0.217, 0.056, 0.0088] m^-1` -- red absorbed ~25x more strongly than
//! blue, which is why water is blue. It is also why a glass of water is
//! colourless: over 10 cm even the red loses only ~2%. Scenes that want
//! visibly blue water state a path length through
//! `SimConfig::slice_thickness_m`; inflating the
//! coefficient instead would be painting, not measuring.

use crate::energy::radiation::{OpticalCoefficientsSi, spectral_band_average};

/// Absorption spectrum of pure water, `(wavelength nm, absorption m^-1)`.
///
/// R. M. Pope and E. S. Fry, "Absorption spectrum (380-700 nm) of pure
/// water. II. Integrating cavity measurements", Applied Optics 36(33),
/// 8710-8723 (1997). Published in `cm^-1`; converted to `m^-1` here, which
/// is the unit `OpticalCoefficientsSi` uses. Values above 700 nm are outside
/// the paper's title range but present in the distributed dataset, and are
/// kept because they are past the visible band's edge where the sRGB
/// sensitivities have already fallen to near zero.
// Four measurements per line: this is a data table, and one pair per line
// would make it 140 lines of noise instead of a readable spectrum.
#[rustfmt::skip]
pub const PURE_WATER_ABSORPTION: [(f32, f32); 140] = [
    (380.0, 0.01137), (382.5, 0.01044), (385.0, 0.00941), (387.5, 0.00917),
    (390.0, 0.00851), (392.5, 0.00829), (395.0, 0.00813), (397.5, 0.00775),
    (400.0, 0.00663), (402.5, 0.00579), (405.0, 0.00530), (407.5, 0.00503),
    (410.0, 0.00473), (412.5, 0.00452), (415.0, 0.00444), (417.5, 0.00442),
    (420.0, 0.00454), (422.5, 0.00474), (425.0, 0.00478), (427.5, 0.00482),
    (430.0, 0.00495), (432.5, 0.00504), (435.0, 0.00530), (437.5, 0.00580),
    (440.0, 0.00635), (442.5, 0.00696), (445.0, 0.00751), (447.5, 0.00830),
    (450.0, 0.00922), (452.5, 0.00969), (455.0, 0.00962), (457.5, 0.00957),
    (460.0, 0.00979), (462.5, 0.01005), (465.0, 0.01011), (467.5, 0.01020),
    (470.0, 0.01060), (472.5, 0.01090), (475.0, 0.01140), (477.5, 0.01210),
    (480.0, 0.01270), (482.5, 0.01310), (485.0, 0.01360), (487.5, 0.01440),
    (490.0, 0.01500), (492.5, 0.01620), (495.0, 0.01730), (497.5, 0.01910),
    (500.0, 0.02040), (502.5, 0.02280), (505.0, 0.02560), (507.5, 0.02800),
    (510.0, 0.03250), (512.5, 0.03720), (515.0, 0.03960), (517.5, 0.03990),
    (520.0, 0.04090), (522.5, 0.04160), (525.0, 0.04170), (527.5, 0.04280),
    (530.0, 0.04340), (532.5, 0.04470), (535.0, 0.04520), (537.5, 0.04660),
    (540.0, 0.04740), (542.5, 0.04890), (545.0, 0.05110), (547.5, 0.05370),
    (550.0, 0.05650), (552.5, 0.05930), (555.0, 0.05960), (557.5, 0.06060),
    (560.0, 0.06190), (562.5, 0.06400), (565.0, 0.06420), (567.5, 0.06720),
    (570.0, 0.06950), (572.5, 0.07330), (575.0, 0.07720), (577.5, 0.08360),
    (580.0, 0.08960), (582.5, 0.09890), (585.0, 0.11000), (587.5, 0.12200),
    (590.0, 0.13510), (592.5, 0.15160), (595.0, 0.16720), (597.5, 0.19250),
    (600.0, 0.22240), (602.5, 0.24700), (605.0, 0.25770), (607.5, 0.26290),
    (610.0, 0.26440), (612.5, 0.26650), (615.0, 0.26780), (617.5, 0.27070),
    (620.0, 0.27550), (622.5, 0.28100), (625.0, 0.28340), (627.5, 0.29040),
    (630.0, 0.29160), (632.5, 0.29950), (635.0, 0.30120), (637.5, 0.30770),
    (640.0, 0.31080), (642.5, 0.32200), (645.0, 0.32500), (647.5, 0.33500),
    (650.0, 0.34000), (652.5, 0.35800), (655.0, 0.37100), (657.5, 0.39300),
    (660.0, 0.41000), (662.5, 0.42400), (665.0, 0.42900), (667.5, 0.43600),
    (670.0, 0.43900), (672.5, 0.44800), (675.0, 0.44800), (677.5, 0.46100),
    (680.0, 0.46500), (682.5, 0.47800), (685.0, 0.48600), (687.5, 0.50200),
    (690.0, 0.51600), (692.5, 0.53800), (695.0, 0.55900), (697.5, 0.59200),
    (700.0, 0.62400), (702.5, 0.66300), (705.0, 0.70400), (707.5, 0.75600),
    (710.0, 0.82700), (712.5, 0.91400), (715.0, 1.00700), (717.5, 1.11900),
    (720.0, 1.23100), (722.5, 1.35600), (725.0, 1.48900), (727.5, 1.67800),
];

/// Molecular (Rayleigh) scattering of pure water at 500 nm, `m^-1`.
///
/// A. Morel, "Optical properties of pure water and pure sea water", in
/// Optical Aspects of Oceanography (1974). Pure water scatters about two
/// orders of magnitude less than it absorbs in the red, which is why water's
/// colour is an absorption effect; real water bodies look far more scattering
/// than this because of suspended particles, which are a property of the
/// mixture, not of water.
pub const PURE_WATER_SCATTERING_M_INV: f32 = 0.0029;

/// Pure water's optical coefficients, band-averaged to linear sRGB.
///
/// Derived from [`PURE_WATER_ABSORPTION`] every call rather than stored: the
/// derivation is cheap, and a constant that can be recomputed from its source
/// cannot silently drift away from it.
pub fn pure_water() -> OpticalCoefficientsSi {
    OpticalCoefficientsSi {
        absorption_m_inv: spectral_band_average(&PURE_WATER_ABSORPTION),
        reduced_scattering_m_inv: PURE_WATER_SCATTERING_M_INV,
    }
}

/// Imaginary part `k` of the complex refractive index of ice Ih across the
/// visible band, `(wavelength nm, k)`.
///
/// S. G. Warren and R. E. Brandt, "Optical constants of ice from the
/// ultraviolet to the microwave: A revised compilation", J. Geophys. Res.
/// 113, D14220 (2008), doi:10.1029/2007JD009744. Copied from the authors'
/// distributed table (`IOP_2008_ASCIItable.dat`), wavelength converted from
/// micrometres to nanometres, `k` verbatim. Stored as `k`, the quantity the
/// table gives, rather than as an absorption coefficient: [`pure_ice`]
/// derives that.
// Four measurements per line, like `PURE_WATER_ABSORPTION`.
#[rustfmt::skip]
pub const ICE_IMAGINARY_INDEX: [(f32, f32); 37] = [
    (390.0, 2.0e-11), (400.0, 2.365e-11), (410.0, 2.669e-11), (420.0, 3.135e-11),
    (430.0, 4.140e-11), (440.0, 6.268e-11), (450.0, 9.239e-11), (460.0, 1.325e-10),
    (470.0, 1.956e-10), (480.0, 2.861e-10), (490.0, 4.172e-10), (500.0, 5.889e-10),
    (510.0, 8.036e-10), (520.0, 1.076e-9), (530.0, 1.409e-9), (540.0, 1.813e-9),
    (550.0, 2.289e-9), (560.0, 2.839e-9), (570.0, 3.461e-9), (580.0, 4.159e-9),
    (590.0, 4.930e-9), (600.0, 5.730e-9), (610.0, 6.890e-9), (620.0, 8.580e-9),
    (630.0, 1.040e-8), (640.0, 1.220e-8), (650.0, 1.430e-8), (660.0, 1.660e-8),
    (670.0, 1.890e-8), (680.0, 2.090e-8), (690.0, 2.400e-8), (700.0, 2.900e-8),
    (710.0, 3.440e-8), (720.0, 4.030e-8), (730.0, 4.300e-8), (740.0, 4.920e-8),
    (750.0, 5.870e-8),
];

/// Absorption coefficient, `m^-1`, of a medium whose refractive index has
/// imaginary part `k` at `wavelength_nm`: `4 pi k / lambda`.
///
/// A plane wave in a medium of index `n + ik` carries the factor
/// `exp(i 2 pi (n + ik) z / lambda)`; its intensity, the squared magnitude,
/// decays as `exp(-4 pi k z / lambda)`, which is Beer-Lambert with that
/// coefficient.
pub fn absorption_from_imaginary_index(wavelength_nm: f32, k: f32) -> f32 {
    4.0 * std::f32::consts::PI * k / (wavelength_nm * 1.0e-9)
}

/// Clear, bubble-free ice's optical coefficients, band-averaged to linear
/// sRGB from [`ICE_IMAGINARY_INDEX`].
///
/// Ice absorbs about as weakly as water, and like water more in the red
/// than the blue, so on its own it looks like clear water: at 650 nm ice
/// absorbs 0.28 m^-1 against water's 0.34, and at 450 nm it is clearer
/// still. Scattering is left at zero: the milky look of lake or glacier ice
/// comes from the bubbles and cracks of a particular sample, not from ice,
/// just as murky water owes its look to what is suspended in it.
pub fn pure_ice() -> OpticalCoefficientsSi {
    let absorption =
        ICE_IMAGINARY_INDEX.map(|(nm, k)| (nm, absorption_from_imaginary_index(nm, k)));
    OpticalCoefficientsSi {
        absorption_m_inv: spectral_band_average(&absorption),
        reduced_scattering_m_inv: 0.0,
    }
}

/// Reduced scattering coefficient, `m^-1`, of a densely packed bed of
/// transparent grains of diameter `grain_diameter_m`: about `1/d`.
///
/// Light scatters at every grain boundary. In a densely packed medium of
/// strongly scattering particles the transport mean free path is on the
/// order of one scatterer, so the reduced scattering coefficient is
/// approximately `1/d` -- the standard transport argument, and the reason
/// finer powders look whiter than coarse ones. 0 for a non-positive
/// diameter.
fn packed_grain_reduced_scattering(grain_diameter_m: f32) -> f32 {
    if grain_diameter_m > 0.0 {
        1.0 / grain_diameter_m
    } else {
        0.0
    }
}

/// Intrinsic density of ice, `kg/m^3`: 916.5, the value C. Henley, J. L.
/// Hollmann, C. R. Meyer and R. Raskar use in "Measurement of Snowpack
/// Density, Grain Size, and Black Carbon Concentration Using Time-domain
/// Diffuse Optics", arXiv:2310.20068v2 (2024), submitted to the Journal of
/// Glaciology, p. 9. A dry snowpack's ice volume fraction is its bulk
/// density over this (their p. 5; air's share of the mass is negligible).
pub const ICE_DENSITY_KG_M3: f32 = 916.5;

/// Absorption enhancement factor `B` of snow: internal reflections lengthen
/// a photon's path inside the ice. Henley et al. 2024 (see
/// [`ICE_DENSITY_KG_M3`]), p. 6, use `B = 1.7`, around which they report,
/// citing Robledano and others (2023), that most real snow samples cluster.
pub const SNOW_ABSORPTION_ENHANCEMENT: f32 = 1.7;

/// Scattering asymmetry parameter `g` of snow, the mean cosine of the
/// scattering angle. Henley et al. 2024, p. 6, use `g = 0.825`, from the
/// same Robledano and others (2023) clustering as
/// [`SNOW_ABSORPTION_ENHANCEMENT`].
pub const SNOW_SCATTERING_ASYMMETRY: f32 = 0.825;

/// Optical coefficients of dry, clean snow, from its grain radius and ice
/// volume fraction.
///
/// Henley et al. 2024 (see [`ICE_DENSITY_KG_M3`]), p. 6, Eqs. 5 and 6, from
/// the geometric-optics snow model of Kokhanovsky and Zege (2004):
///
/// - absorption `sigma_a = B * v_i * alpha_ice`, with `alpha_ice` pure
///   ice's own ([`pure_ice`]): snow absorbs less than solid ice, by its ice
///   fraction, and a little more per unit ice, by `B`;
/// - reduced scattering `sigma_s' = 3/2 * (1 - g) * v_i / r_e`.
///
/// `grain_radius_m` is their optical grain radius `r_e`: the radius of the
/// ice sphere with the same surface-area-to-volume ratio as the snow's ice,
/// `3 <V> / <S>` over mean grain volume and area. `ice_volume_fraction` is
/// `v_i`, bulk density over [`ICE_DENSITY_KG_M3`] for dry snow. Scattering
/// outweighs absorption by thousands of times, which is why a clear
/// substance makes a white, opaque pack.
pub fn snow(grain_radius_m: f32, ice_volume_fraction: f32) -> OpticalCoefficientsSi {
    let v_i = ice_volume_fraction.clamp(0.0, 1.0);
    let reduced_scattering_m_inv = if grain_radius_m > 0.0 {
        1.5 * (1.0 - SNOW_SCATTERING_ASYMMETRY) * v_i / grain_radius_m
    } else {
        0.0
    };
    OpticalCoefficientsSi {
        absorption_m_inv: pure_ice()
            .absorption_m_inv
            .map(|alpha| SNOW_ABSORPTION_ENHANCEMENT * v_i * alpha),
        reduced_scattering_m_inv,
    }
}

/// Absorption of bulk quartz across the visible band, `m^-1`.
///
/// Quartz is the same substance as window glass and is essentially
/// non-absorbing in the visible -- that is why a pane of it is clear. The
/// value is flat across the three bands because there is no absorption
/// feature in the visible to give it a colour.
///
/// The value itself is not from a measurement: silica datasets tabulate an
/// extinction coefficient `k` of zero across the visible (below what they
/// resolve), so the true figure is smaller still. It stands as a small
/// placeholder until a measured bulk-quartz absorption is found; at any
/// value this small, sand's colour is set by its scattering, not by this.
///
/// What this deliberately does NOT model: the faint yellow of most beach
/// sand, which comes from iron-oxide coatings on the grains, not from the
/// quartz. Modelling that needs hematite/goethite absorption data this
/// project does not have, so clean quartz sand renders pale rather than
/// being given an invented tint.
pub const DRY_QUARTZ_ABSORPTION_M_INV: f32 = 0.01;

/// Optical coefficients of a dry granular quartz bed, from its grain size.
///
/// Sand looks pale and opaque despite being made of a transparent mineral,
/// because light scatters at every grain boundary
/// ([`packed_grain_reduced_scattering`]).
///
/// So the appearance is derived from a mechanical property the material
/// already carries, not chosen: state the grain diameter and the optics
/// follow.
pub fn dry_quartz_sand(grain_diameter_m: f32) -> OpticalCoefficientsSi {
    OpticalCoefficientsSi {
        absorption_m_inv: [DRY_QUARTZ_ABSORPTION_M_INV; 3],
        reduced_scattering_m_inv: packed_grain_reduced_scattering(grain_diameter_m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sand scatters overwhelmingly more than it absorbs, which is why a
    /// transparent mineral makes an opaque pale bed. Finer grains scatter
    /// more, matching the everyday observation that flour is whiter than
    /// gravel.
    #[test]
    fn sand_is_scattering_dominated_and_finer_grains_scatter_more() {
        let medium = dry_quartz_sand(1.0e-3);
        let mean_absorption = medium.absorption_m_inv.iter().sum::<f32>() / 3.0;
        assert!(
            medium.reduced_scattering_m_inv > 1000.0 * mean_absorption,
            "sand must be scattering dominated: sigma_s {} vs mean sigma_a {mean_absorption}",
            medium.reduced_scattering_m_inv
        );
        let fine = dry_quartz_sand(0.1e-3);
        assert!(
            fine.reduced_scattering_m_inv > medium.reduced_scattering_m_inv,
            "finer grains must scatter more"
        );
    }

    /// The conversion from `k` against a value worked by hand from the
    /// table: at 650 nm, `k = 1.430e-8`, so `4 pi k / lambda` is
    /// `4 * 3.14159 * 1.430e-8 / 650e-9 = 0.2765 m^-1`.
    #[test]
    fn ice_absorption_follows_from_the_tabulated_imaginary_index() {
        let (nm, k) = ICE_IMAGINARY_INDEX[26];
        assert_eq!(nm, 650.0, "row 26 must be the 650 nm sample");
        let alpha = absorption_from_imaginary_index(nm, k);
        assert!((alpha - 0.2765).abs() < 1.0e-3, "got {alpha} m^-1");
    }

    /// Ice, like water, absorbs red more than blue, and by a large factor;
    /// and it is not more absorbing than water in any band, which is why a
    /// block of clear ice looks like clear water.
    #[test]
    fn ice_absorbs_red_more_than_blue_and_no_more_than_water() {
        let ice = pure_ice().absorption_m_inv;
        let water = pure_water().absorption_m_inv;
        assert!(
            ice[0] > ice[1] && ice[1] > ice[2],
            "expected red > green > blue, got {ice:?}"
        );
        assert!(ice[0] / ice[2] > 20.0, "red/blue ratio too weak: {ice:?}");
        for channel in 0..3 {
            assert!(
                ice[channel] <= water[channel],
                "ice {ice:?} must not absorb more than water {water:?}"
            );
        }
    }

    /// Henley et al. 2024's Eqs. 5 and 6 on their own ground-truth sample
    /// (p. 21: `v_i = 0.465`, `r_e = 242.5` micrometres), worked by hand:
    /// `sigma_s' = 1.5 * 0.175 * 0.465 / 242.5e-6 = 503.3 m^-1`, and
    /// absorption `1.7 * 0.465 = 0.79` times pure ice's in every band.
    #[test]
    fn snow_follows_henley_eqs_5_and_6_on_their_measured_sample() {
        let ice = pure_ice();
        let snow = snow(242.5e-6, 0.465);
        assert!(
            (snow.reduced_scattering_m_inv - 503.3).abs() < 0.5,
            "got {} m^-1",
            snow.reduced_scattering_m_inv
        );
        for channel in 0..3 {
            let ratio = snow.absorption_m_inv[channel] / ice.absorption_m_inv[channel];
            assert!(
                (ratio - 0.7905).abs() < 1.0e-4,
                "band {channel}: ratio {ratio}"
            );
        }
    }

    /// Snow is ice that scatters: ice scatters nothing, snow scatters
    /// thousands of times more than it absorbs, so it is white where ice is
    /// clear.
    #[test]
    fn snow_is_ice_that_scatters() {
        let ice = pure_ice();
        let snow = snow(242.5e-6, 0.465);
        assert_eq!(ice.reduced_scattering_m_inv, 0.0);
        let strongest_absorption = snow.absorption_m_inv[0];
        assert!(
            snow.reduced_scattering_m_inv > 1000.0 * strongest_absorption,
            "snow must be scattering dominated: sigma_s {} vs red sigma_a {strongest_absorption}",
            snow.reduced_scattering_m_inv
        );
    }

    /// Water is blue because it absorbs red far more strongly than blue.
    /// After band averaging that ordering must survive, and by a large
    /// factor -- if it did not, the whole point of storing a spectrum instead
    /// of three hand-picked numbers would be lost.
    #[test]
    fn band_averaged_water_absorbs_red_far_more_than_blue() {
        let water = pure_water();
        let [red, green, blue] = water.absorption_m_inv;
        assert!(
            red > green && green > blue,
            "expected red > green > blue, got {:?}",
            water.absorption_m_inv
        );
        assert!(
            red / blue > 20.0,
            "red/blue absorption ratio {:.1} is too weak to make water blue ({:?})",
            red / blue,
            water.absorption_m_inv
        );
    }

    /// The band average must land inside the measured spectrum's own range in
    /// each channel's own band -- a weighted mean cannot exceed its inputs,
    /// and this catches a sensitivity curve wired to the wrong channel.
    #[test]
    fn band_averages_stay_within_the_measured_spectrum() {
        let [red, green, blue] = pure_water().absorption_m_inv;
        let lowest = PURE_WATER_ABSORPTION
            .iter()
            .fold(f32::INFINITY, |a, (_, v)| a.min(*v));
        let highest = PURE_WATER_ABSORPTION
            .iter()
            .fold(f32::NEG_INFINITY, |a, (_, v)| a.max(*v));
        for value in [red, green, blue] {
            assert!(
                (lowest..=highest).contains(&value),
                "{value} is outside the measured range [{lowest}, {highest}]"
            );
        }
        // Blue's band sits near the absorption minimum (418 nm, 0.0044 m^-1
        // per Pope and Fry), so it must stay small in absolute terms too.
        assert!(blue < 0.03, "blue band average {blue} is implausibly high");
    }

    /// A glass of water is colourless and the sea is blue: the same
    /// coefficients, a different path length. Beer-Lambert says so, and this
    /// is the check that stops anyone "fixing" pale water by inflating the
    /// coefficient instead of stating a depth.
    #[test]
    fn path_length_not_coefficient_is_what_makes_water_blue() {
        use crate::energy::radiation::beer_lambert_transmittance;
        let sigma = pure_water().absorption_m_inv;

        let glass = beer_lambert_transmittance(sigma, 1.0, 0.1);
        assert!(
            glass.iter().all(|t| *t > 0.94),
            "10 cm of water should be near-colourless, got {glass:?}"
        );

        // At 10 m the band-averaged coefficients give [0.115, 0.573, 0.916]:
        // most of the red is gone, nearly all of the blue survives. Stated as
        // fractions rather than tight numbers, so the claim is "red goes,
        // blue stays" and not a transcription of today's output.
        let sea = beer_lambert_transmittance(sigma, 1.0, 10.0);
        assert!(
            sea[0] < 0.2 && sea[2] > 0.8,
            "10 m of water should have lost its red and kept its blue, got {sea:?}"
        );
    }
}

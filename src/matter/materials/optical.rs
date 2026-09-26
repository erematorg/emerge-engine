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
//! visibly blue water state a real path length through
//! `PhysicalRenderContract::view_thickness_meters`; inflating the
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

/// Absorption of bulk quartz across the visible band, `m^-1`.
///
/// Quartz is the same substance as window glass and is essentially
/// non-absorbing in the visible -- that is why a pane of it is clear. The
/// value is flat across the three bands because there is no absorption
/// feature in the visible to give it a colour.
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
/// because light scatters at every grain boundary. In a densely packed
/// medium of strongly scattering particles the transport mean free path is
/// on the order of one scatterer, so the reduced scattering coefficient is
/// approximately `1/d` -- the standard transport argument, and the reason
/// finer powders look whiter than coarse ones.
///
/// So the appearance is derived from a mechanical property the material
/// already carries, not chosen: state the grain diameter and the optics
/// follow.
pub fn dry_quartz_sand(grain_diameter_m: f32) -> OpticalCoefficientsSi {
    let reduced_scattering_m_inv = if grain_diameter_m > 0.0 {
        1.0 / grain_diameter_m
    } else {
        0.0
    };
    OpticalCoefficientsSi {
        absorption_m_inv: [DRY_QUARTZ_ABSORPTION_M_INV; 3],
        reduced_scattering_m_inv,
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
    /// coefficient instead of stating a real depth.
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

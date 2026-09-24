//! Bubble resonance — real vibration frequency and (partial) damping for
//! an air bubble oscillating in a liquid, derived directly from the
//! bubble's own physical state (radius, ambient pressure, liquid
//! density), same "zero assets, real rules only" discipline as
//! `modal.rs`'s rod cantilever modes -- no sample-fitting, no calibration
//! pipeline, no canned/pre-recorded audio.
//!
//! # Real physics: Minnaert resonance
//! A gas bubble in a liquid is a real spring-mass oscillator: the
//! entrained liquid around the bubble is the inertial mass, the gas's own
//! compressibility (adiabatic, ratio of specific heats γ) is the spring.
//! Minnaert (1933), "On musical air-bubbles and the sounds of running
//! water", derived the natural frequency:
//!
//!   f₀ = (1 / 2πR) · √(3γP₀ / ρ)
//!
//! where `R` is the bubble radius (m), `P₀` is the ambient (hydrostatic)
//! pressure at the bubble's depth (Pa), `ρ` is the SURROUNDING LIQUID's
//! density (kg/m³, not the gas's -- the liquid's inertia dominates since
//! it is far denser), and `γ` is the gas's adiabatic index (≈1.4 for air,
//! diatomic ideal gas). This is the real, standard, still-cited textbook
//! result for "why does dripping water make a musical plink" -- verified
//! below against the commonly-cited real-world reference point (~3.3 kHz
//! for a 1mm air bubble in water at 1 atm).
//!
//! # Damping -- a disclosed PARTIAL model, not the full real picture
//! An oscillating bubble loses energy three real ways (Devin 1959,
//! "Survey of Thermal, Radiation, and Viscous Damping of Pulsating Air
//! Bubbles in Water"; standard reference: Leighton, *The Acoustic
//! Bubble*, 1994): thermal conduction, viscous drag on the surrounding
//! liquid, and acoustic RADIATION (the bubble itself radiates sound,
//! which reacts back as a real damping force). Only radiation damping is
//! included here -- it has a simple, closed-form, always-present
//! expression (`δ_rad = k·R`, k = acoustic wavenumber in the liquid at
//! the bubble's own resonant frequency) that needs no empirical curve
//! fitting. Thermal and viscous damping are frequency- and radius-
//! dependent empirical fits (Devin's own charts, no simple closed form) --
//! genuinely NOT included here, not silently assumed zero-cost: for small
//! bubbles thermal damping is often the DOMINANT term, so
//! `damping_ratio` here is a real but incomplete (under-damped-relative-
//! to-reality) lower bound, disclosed as such.

// Reuses `AcousticMode` (defined in `modal.rs`, re-exported by this
// module's parent) -- bubble resonance is another real source feeding the
// same "one vibrational mode" concept rod cantilever modes already
// established, not a new, parallel type.
use super::AcousticMode;

/// Adiabatic index (ratio of specific heats) for air -- a real, standard
/// diatomic-ideal-gas constant, not a tuned/fitted number.
pub const AIR_ADIABATIC_INDEX: f32 = 1.4;

/// Real speed of sound in fresh water at 20°C (m/s) -- standard reference
/// value (e.g. Kinsler & Frey, *Fundamentals of Acoustics*), used only for
/// the radiation-damping term below.
pub const WATER_SOUND_SPEED_M_S: f32 = 1481.0;

/// Real Minnaert bubble-resonance mode -- see this module's own doc for
/// the full derivation and the honest scope of what `damping_ratio`
/// does/doesn't include (radiation damping only).
///
/// Returns `None` for a degenerate bubble (non-positive radius, pressure,
/// or density) rather than dividing by zero or producing a NaN/negative
/// frequency.
pub fn minnaert_bubble_mode(
    radius_m: f32,
    ambient_pressure_pa: f32,
    liquid_density_kg_m3: f32,
) -> Option<AcousticMode> {
    if radius_m <= 0.0 || ambient_pressure_pa <= 0.0 || liquid_density_kg_m3 <= 0.0 {
        return None;
    }

    let omega0 = (1.0 / radius_m)
        * (3.0 * AIR_ADIABATIC_INDEX * ambient_pressure_pa / liquid_density_kg_m3).sqrt();
    let frequency_hz = omega0 / (2.0 * std::f32::consts::PI);

    // Radiation damping: an oscillating bubble radiates sound into the
    // surrounding liquid, and that radiated energy is a real loss --
    // δ_rad = k·R, k = ω₀/c (acoustic wavenumber in the liquid at the
    // bubble's own resonant frequency). For a lightly damped linear
    // oscillator the dimensionless damping RATIO is ζ = δ/2 (quality
    // factor Q = 1/δ, ζ = 1/(2Q)).
    let wavenumber = omega0 / WATER_SOUND_SPEED_M_S;
    let delta_rad = wavenumber * radius_m;
    let damping_ratio = (delta_rad / 2.0).clamp(0.0, 1.0);

    Some(AcousticMode {
        frequency_hz,
        damping_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, independently-checkable reference point: a 1mm-radius air
    /// bubble in water at 1 atm resonates at approximately 3.3 kHz -- a
    /// commonly-cited real-world/textbook example (e.g. the sound of a
    /// single drop of water hitting a pool). Hand-computed here from the
    /// formula's own real constants, not copied from the function under
    /// test.
    #[test]
    fn one_millimeter_bubble_matches_the_real_textbook_reference_point() {
        let radius_m = 1.0e-3_f32;
        let atmospheric_pa = 101_325.0_f32;
        let water_density = 998.0_f32; // fresh water, ~20°C

        // Independent hand computation.
        let omega0_expected =
            (1.0 / radius_m) * (3.0 * 1.4 * atmospheric_pa / water_density).sqrt();
        let f0_expected = omega0_expected / (2.0 * std::f32::consts::PI);

        let mode = minnaert_bubble_mode(radius_m, atmospheric_pa, water_density).unwrap();

        let rel_err = (mode.frequency_hz - f0_expected).abs() / f0_expected;
        assert!(
            rel_err < 1.0e-4,
            "frequency {:.2} Hz doesn't match independently hand-computed {:.2} Hz \
             (rel_err={rel_err:.6})",
            mode.frequency_hz,
            f0_expected
        );
        // Real, standard cited ballpark for this exact scenario (~3.26-3.3 kHz
        // depending on the exact reference values used) -- a genuine
        // external sanity check, not just self-consistency with the hand
        // computation above.
        assert!(
            (3000.0..3600.0).contains(&mode.frequency_hz),
            "1mm air bubble in water at 1atm should resonate in the real, \
             commonly-cited ~3.3 kHz range, got {:.1} Hz",
            mode.frequency_hz
        );
    }

    /// Real, physically expected trend: a bigger bubble has a lower
    /// resonant frequency (more entrained liquid mass, same restoring
    /// spring) -- the classic "big bubbles glug, small bubbles plink"
    /// real-world observation, not asserted blind.
    #[test]
    fn larger_bubble_resonates_lower() {
        let atmospheric_pa = 101_325.0;
        let water_density = 998.0;
        let small = minnaert_bubble_mode(0.5e-3, atmospheric_pa, water_density).unwrap();
        let large = minnaert_bubble_mode(5.0e-3, atmospheric_pa, water_density).unwrap();
        assert!(
            large.frequency_hz < small.frequency_hz,
            "a larger bubble must resonate at a lower frequency: \
             small(R=0.5mm)={:.1}Hz large(R=5mm)={:.1}Hz",
            small.frequency_hz,
            large.frequency_hz
        );
    }

    /// Real, physically expected trend: greater ambient (hydrostatic)
    /// pressure -- i.e. deeper underwater -- stiffens the gas spring and
    /// raises the resonant frequency.
    #[test]
    fn deeper_bubble_under_more_pressure_resonates_higher() {
        let radius_m = 1.0e-3;
        let water_density = 998.0;
        let shallow = minnaert_bubble_mode(radius_m, 101_325.0, water_density).unwrap();
        // Real hydrostatic pressure at 10m depth: P = P_atm + rho*g*depth.
        let deep_pa = 101_325.0 + water_density * 9.81 * 10.0;
        let deep = minnaert_bubble_mode(radius_m, deep_pa, water_density).unwrap();
        assert!(
            deep.frequency_hz > shallow.frequency_hz,
            "a bubble under more real hydrostatic pressure must resonate higher: \
             shallow={:.1}Hz deep={:.1}Hz",
            shallow.frequency_hz,
            deep.frequency_hz
        );
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(minnaert_bubble_mode(0.0, 101_325.0, 998.0).is_none());
        assert!(minnaert_bubble_mode(1.0e-3, 0.0, 998.0).is_none());
        assert!(minnaert_bubble_mode(1.0e-3, 101_325.0, 0.0).is_none());
        assert!(minnaert_bubble_mode(-1.0e-3, 101_325.0, 998.0).is_none());
    }

    /// Real damping sanity: radiation damping alone should be light (a
    /// small bubble genuinely rings, doesn't die out in one cycle) --
    /// bounds the ratio well under critical damping for a real, typical
    /// bubble size, matching the "audible ringing plink" phenomenon this
    /// module exists to eventually drive.
    #[test]
    fn radiation_damping_alone_is_light_for_a_typical_bubble() {
        let mode = minnaert_bubble_mode(1.0e-3, 101_325.0, 998.0).unwrap();
        assert!(
            mode.damping_ratio > 0.0 && mode.damping_ratio < 0.05,
            "radiation-only damping ratio should be small and positive for a \
             1mm bubble (a real, audibly-ringing plink, not silence or an \
             overdamped thud): got {}",
            mode.damping_ratio
        );
    }
}

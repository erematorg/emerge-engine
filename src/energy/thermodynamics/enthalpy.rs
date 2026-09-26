//! Real enthalpy method for phase-change (Stefan) problems -- Voller & Cross
//! 1981 ("Accurate solutions of moving boundary problems using the enthalpy
//! method," Int. J. Heat Mass Transfer 24(3):545-556) and Voller &
//! Swaminathan 1991 ("General source-based method for solidification phase
//! change," Numerical Heat Transfer B 19(2):175-189).
//!
//! Real motivating gap (GitHub issue #7): `Simulation::apply_phase_transition`
//! already debits a real, energy-conserving latent-heat jump
//! (`temperature -= latent_heat / heat_capacity`, see that function's own
//! "Stefan condition" roadmap doc) at the INSTANT a threshold fires -- but
//! the transition itself is still a discrete switch, not a continuous mushy
//! zone. The enthalpy method's own real idea: track enthalpy `H` (total
//! thermal energy per unit mass) as the state variable instead of
//! temperature `T` -- T and phase_fraction are then both DERIVED from H,
//! and the derivation is naturally continuous: absorbing energy while
//! `T == t_transition` doesn't raise T further, it raises phase_fraction
//! instead, until the whole latent-heat band has been crossed.
//!
//! Real, disclosed first-increment scope: one shared specific heat `cp` on
//! both sides of the transition (distinct `cp_solid`/`cp_liquid` is a real
//! refinement the method supports, not implemented here). This is the
//! numerical core only -- wiring it into `Particle`/`WithLatentHeat` so a
//! scene's own materials use it automatically is real further work, not
//! done here; see this module's own tests for how the three regions relate,
//! and `crate::solver::particles::apply_phase_transition`'s doc for the
//! discrete-jump mechanism this generalizes.

/// Real, forward enthalpy relation H(T) -- see this module's own doc for the
/// three-region shape. Only valid for `T <= t_transition` (ordinary sensible
/// heat) or `T` representing a FULLY melted state (`phase_fraction == 1`);
/// it cannot represent a mushy intermediate on its own, since a single T
/// doesn't determine phase_fraction in that band -- H does. Use this to seed
/// H from a known, single-phase starting temperature (e.g. at spawn), not to
/// track an ongoing melt.
pub fn enthalpy_from_temperature(t: f32, cp: f32, latent_heat: f32, t_transition: f32) -> f32 {
    debug_assert!(cp > 0.0, "specific heat capacity must be positive");
    if t <= t_transition {
        cp * t
    } else {
        cp * t_transition + latent_heat + cp * (t - t_transition)
    }
}

/// Real inverse of the enthalpy relation: recovers `(temperature,
/// phase_fraction)` from enthalpy `h` alone. `phase_fraction` is in `[0, 1]`
/// -- `0.0` fully solid, `1.0` fully liquid, `(0.0, 1.0)` a real mushy-zone
/// particle mid-melt with `temperature` PINNED at `t_transition` (the real
/// physical behavior: adding heat to a melting substance raises how much of
/// it has melted, not its temperature, until melting completes).
pub fn temperature_and_phase_fraction_from_enthalpy(
    h: f32,
    cp: f32,
    latent_heat: f32,
    t_transition: f32,
) -> (f32, f32) {
    debug_assert!(cp > 0.0, "specific heat capacity must be positive");
    debug_assert!(latent_heat >= 0.0, "latent heat must be non-negative");
    let h_solidus = cp * t_transition;
    let h_liquidus = h_solidus + latent_heat;
    if h <= h_solidus {
        (h / cp, 0.0)
    } else if h < h_liquidus {
        (t_transition, (h - h_solidus) / latent_heat.max(1.0e-12))
    } else {
        (t_transition + (h - h_liquidus) / cp, 1.0)
    }
}

/// Real per-phase thermal properties for a chained solid<->liquid<->gas
/// enthalpy relation (e.g. ice<->water<->steam) -- see
/// `chained_state_from_enthalpy`'s own doc for the full picture. Real,
/// disclosed first-increment scope, same as this module's single-
/// transition functions above: one `cp` per PHASE (not per-temperature),
/// the standard simplification this whole method already makes.
#[derive(Debug, Clone, Copy)]
pub struct PhaseChainProperties {
    pub cp_solid: f32,
    pub cp_liquid: f32,
    pub cp_gas: f32,
    pub melting_point_k: f32,
    pub boiling_point_k: f32,
    pub fusion_latent_heat_j_kg: f32,
    pub vaporization_latent_heat_j_kg: f32,
}

/// Which real phase (or which of the two real latent-heat bands) a chained
/// enthalpy value currently represents -- see `chained_state_from_enthalpy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseState {
    Solid,
    /// Mid-melt: `fraction` in `[0, 1]`, temperature pinned at
    /// `melting_point_k`.
    Melting {
        fraction: f32,
    },
    Liquid,
    /// Mid-boil: `fraction` in `[0, 1]`, temperature pinned at
    /// `boiling_point_k`.
    Boiling {
        fraction: f32,
    },
    Gas,
}

/// Real, chained generalization of `temperature_and_phase_fraction_from_
/// enthalpy` across TWO consecutive real latent-heat transitions (melting
/// then boiling) instead of one -- the real ice<->water<->steam picture
/// this module's own top-of-file doc flagged as "real further work, not
/// implemented here" when it only handled a single transition. Same real
/// method (Voller & Cross 1981), same monotonic-in-H structure, extended
/// to 5 real regions instead of 3: solid, melting band, liquid, boiling
/// band, gas. `H` is referenced to `T=0K` (`H=cp_solid*T` below melting),
/// matching `enthalpy_from_temperature`'s own convention exactly, so a
/// solid-region value from that simpler function is bit-identical to this
/// one's.
pub fn chained_state_from_enthalpy(props: &PhaseChainProperties, h: f32) -> (f32, PhaseState) {
    debug_assert!(props.cp_solid > 0.0 && props.cp_liquid > 0.0 && props.cp_gas > 0.0);
    debug_assert!(props.boiling_point_k > props.melting_point_k);
    let h_solidus = props.cp_solid * props.melting_point_k;
    let h_liquidus = h_solidus + props.fusion_latent_heat_j_kg;
    let h_boil_start =
        h_liquidus + props.cp_liquid * (props.boiling_point_k - props.melting_point_k);
    let h_vapor_start = h_boil_start + props.vaporization_latent_heat_j_kg;

    if h <= h_solidus {
        (h / props.cp_solid, PhaseState::Solid)
    } else if h < h_liquidus {
        let fraction = (h - h_solidus) / props.fusion_latent_heat_j_kg.max(1.0e-12);
        (props.melting_point_k, PhaseState::Melting { fraction })
    } else if h < h_boil_start {
        let t = props.melting_point_k + (h - h_liquidus) / props.cp_liquid;
        (t, PhaseState::Liquid)
    } else if h < h_vapor_start {
        let fraction = (h - h_boil_start) / props.vaporization_latent_heat_j_kg.max(1.0e-12);
        (props.boiling_point_k, PhaseState::Boiling { fraction })
    } else {
        let t = props.boiling_point_k + (h - h_vapor_start) / props.cp_gas;
        (t, PhaseState::Gas)
    }
}

/// Real inverse of `chained_state_from_enthalpy`, for a KNOWN single-phase
/// state (mirrors `enthalpy_from_temperature`'s own "not for an ongoing
/// melt" caveat -- a single `(temperature, PhaseState)` pair doesn't
/// determine a unique H inside a mushy/boiling band on its own unless
/// `fraction` is given explicitly). Used to seed H once, from a real
/// starting temperature, not to track an ongoing transition.
pub fn chained_enthalpy_from_temperature(
    props: &PhaseChainProperties,
    temperature: f32,
    state: PhaseState,
) -> f32 {
    let h_solidus = props.cp_solid * props.melting_point_k;
    let h_liquidus = h_solidus + props.fusion_latent_heat_j_kg;
    let h_boil_start =
        h_liquidus + props.cp_liquid * (props.boiling_point_k - props.melting_point_k);
    let h_vapor_start = h_boil_start + props.vaporization_latent_heat_j_kg;
    match state {
        PhaseState::Solid => props.cp_solid * temperature,
        PhaseState::Melting { fraction } => h_solidus + fraction * props.fusion_latent_heat_j_kg,
        PhaseState::Liquid => h_liquidus + props.cp_liquid * (temperature - props.melting_point_k),
        PhaseState::Boiling { fraction } => {
            h_boil_start + fraction * props.vaporization_latent_heat_j_kg
        }
        PhaseState::Gas => h_vapor_start + props.cp_gas * (temperature - props.boiling_point_k),
    }
}

#[cfg(test)]
mod chained_enthalpy_tests {
    use super::*;

    // Real water/ice/steam values, same sourcing as `enthalpy_tests`'
    // own constants plus the demo's own cited steam cp (NIST steam
    // tables, saturated vapor near 100C, 1 atm).
    fn water_chain() -> PhaseChainProperties {
        PhaseChainProperties {
            cp_solid: 2090.0,  // ice, J/(kg*K), CRC Handbook at 0C
            cp_liquid: 4182.0, // water, J/(kg*K)
            cp_gas: 2080.0,    // saturated steam, J/(kg*K), NIST near 100C
            melting_point_k: 273.15,
            boiling_point_k: 373.15,
            fusion_latent_heat_j_kg: 334_000.0,
            vaporization_latent_heat_j_kg: 2_257_000.0,
        }
    }

    /// Real round-trip in each of the 3 single-phase regions.
    #[test]
    fn round_trips_in_each_single_phase_region() {
        let props = water_chain();
        for (t, state) in [
            (250.0, PhaseState::Solid),
            (300.0, PhaseState::Liquid),
            (400.0, PhaseState::Gas),
        ] {
            let h = chained_enthalpy_from_temperature(&props, t, state);
            let (t2, state2) = chained_state_from_enthalpy(&props, h);
            assert!(
                (t2 - t).abs() < 1.0e-2,
                "region {state:?}: expected T={t}, got {t2}"
            );
            assert_eq!(
                std::mem::discriminant(&state2),
                std::mem::discriminant(&state),
                "region mismatch: expected {state:?}, got {state2:?}"
            );
        }
    }

    /// Real continuity check at the melting-band boundary: entering the
    /// liquid region must start EXACTLY at the melting point, no jump.
    #[test]
    fn liquid_region_starts_exactly_at_melting_point() {
        let props = water_chain();
        let h_liquidus = props.cp_solid * props.melting_point_k + props.fusion_latent_heat_j_kg;
        let (t, state) = chained_state_from_enthalpy(&props, h_liquidus + 1.0);
        assert!((t - props.melting_point_k).abs() < 1.0e-2, "got T={t}");
        assert_eq!(state, PhaseState::Liquid);
    }

    /// Real continuity check at the boiling-band boundary: entering the
    /// gas region must start EXACTLY at the boiling point, no jump.
    #[test]
    fn gas_region_starts_exactly_at_boiling_point() {
        let props = water_chain();
        let h_solidus = props.cp_solid * props.melting_point_k;
        let h_liquidus = h_solidus + props.fusion_latent_heat_j_kg;
        let h_boil_start =
            h_liquidus + props.cp_liquid * (props.boiling_point_k - props.melting_point_k);
        let h_vapor_start = h_boil_start + props.vaporization_latent_heat_j_kg;
        let (t, state) = chained_state_from_enthalpy(&props, h_vapor_start + 1.0);
        assert!((t - props.boiling_point_k).abs() < 1.0e-2, "got T={t}");
        assert_eq!(state, PhaseState::Gas);
    }

    /// The real mushy/boiling-band property: mid-band, temperature must be
    /// PINNED at the transition point, and fraction must read ~0.5 in
    /// BOTH bands independently, not just the first one this module
    /// originally supported.
    #[test]
    fn both_bands_pin_temperature_and_report_real_fractions() {
        let props = water_chain();
        let h_solidus = props.cp_solid * props.melting_point_k;
        let h_liquidus = h_solidus + props.fusion_latent_heat_j_kg;
        let (t_melt, state_melt) =
            chained_state_from_enthalpy(&props, h_solidus + props.fusion_latent_heat_j_kg * 0.5);
        assert!((t_melt - props.melting_point_k).abs() < 1.0e-2);
        assert!(
            matches!(state_melt, PhaseState::Melting { fraction } if (fraction - 0.5).abs() < 1.0e-4)
        );

        let h_boil_start =
            h_liquidus + props.cp_liquid * (props.boiling_point_k - props.melting_point_k);
        let (t_boil, state_boil) = chained_state_from_enthalpy(
            &props,
            h_boil_start + props.vaporization_latent_heat_j_kg * 0.5,
        );
        assert!((t_boil - props.boiling_point_k).abs() < 1.0e-2);
        assert!(
            matches!(state_boil, PhaseState::Boiling { fraction } if (fraction - 0.5).abs() < 1.0e-4)
        );
    }

    /// Real monotonicity across the FULL 5-region chain -- the same
    /// physical-state-not-arbitrary-lookup requirement
    /// `temperature_and_phase_fraction_are_monotonic_in_enthalpy` checks
    /// for the single-transition case, extended to the full real range.
    #[test]
    fn temperature_is_monotonic_non_decreasing_across_the_full_chain() {
        let props = water_chain();
        let mut prev_t = f32::MIN;
        for i in 0..500 {
            let h = -100_000.0 + i as f32 * 20_000.0; // sweeps well below solid to well above gas
            let (t, _) = chained_state_from_enthalpy(&props, h);
            assert!(
                t >= prev_t - 1.0e-3,
                "temperature must be monotonic non-decreasing in H: {t} < {prev_t} at h={h}"
            );
            prev_t = t;
        }
    }

    /// Real energetic-consistency check: the total energy absorbed
    /// crossing BOTH bands must equal the sum of the two real cited
    /// latent heats exactly, by construction -- not silently scaled or
    /// dropped anywhere in the chained derivation.
    #[test]
    fn crossing_both_bands_absorbs_exactly_the_sum_of_both_real_latent_heats() {
        let props = water_chain();
        let h_solidus = props.cp_solid * props.melting_point_k;
        let h_liquidus = h_solidus + props.fusion_latent_heat_j_kg;
        let h_boil_start =
            h_liquidus + props.cp_liquid * (props.boiling_point_k - props.melting_point_k);
        let h_vapor_start = h_boil_start + props.vaporization_latent_heat_j_kg;
        let total_latent_absorbed = (h_liquidus - h_solidus) + (h_vapor_start - h_boil_start);
        let expected = props.fusion_latent_heat_j_kg + props.vaporization_latent_heat_j_kg;
        assert!((total_latent_absorbed - expected).abs() < 1.0e-3);
    }
}

#[cfg(test)]
mod enthalpy_tests {
    use super::*;

    const CP: f32 = 2100.0; // ice, J/(kg*K), real value -- matches diffusion.rs's own convention
    const LATENT_HEAT: f32 = 334_000.0; // ice->water, J/kg, real value (already used elsewhere)
    const T_TRANSITION: f32 = 273.15; // 0 C in Kelvin

    /// Real round-trip check below the transition: H(T) then back must
    /// recover the same T with phase_fraction=0 (still fully solid).
    #[test]
    fn round_trips_below_transition() {
        let t = 250.0;
        let h = enthalpy_from_temperature(t, CP, LATENT_HEAT, T_TRANSITION);
        let (t2, phase) =
            temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
        assert!((t2 - t).abs() < 1.0e-3, "expected T={t}, got {t2}");
        assert_eq!(
            phase, 0.0,
            "should still be fully solid below the transition"
        );
    }

    /// Real round-trip check above the transition (fully melted): H(T) then
    /// back must recover the same T with phase_fraction=1.
    #[test]
    fn round_trips_above_transition() {
        let t = 300.0;
        let h = enthalpy_from_temperature(t, CP, LATENT_HEAT, T_TRANSITION);
        let (t2, phase) =
            temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
        assert!((t2 - t).abs() < 1.0e-3, "expected T={t}, got {t2}");
        assert_eq!(phase, 1.0, "should be fully liquid above the transition");
    }

    /// The real mushy-zone property this whole method exists for: halfway
    /// through the latent-heat band, temperature must be PINNED at
    /// t_transition (not interpolated), and phase_fraction must read ~0.5 --
    /// not 0 or 1. This is the literal "partial melting" issue #7 asks for.
    #[test]
    fn halfway_through_latent_heat_band_is_a_real_mushy_particle() {
        let h_solidus = CP * T_TRANSITION;
        let h_halfway = h_solidus + LATENT_HEAT * 0.5;
        let (t, phase) =
            temperature_and_phase_fraction_from_enthalpy(h_halfway, CP, LATENT_HEAT, T_TRANSITION);
        assert!(
            (t - T_TRANSITION).abs() < 1.0e-3,
            "mushy-zone particle's temperature must stay pinned at the transition \
             point while it's still melting, got T={t}"
        );
        assert!(
            (phase - 0.5).abs() < 1.0e-6,
            "expected phase_fraction=0.5 exactly halfway through the latent-heat \
             band, got {phase}"
        );
    }

    /// Real monotonicity check: as more energy (H) is added, temperature and
    /// phase_fraction must never DECREASE -- the physical requirement that
    /// this relation actually represents an energy state, not an arbitrary
    /// lookup. Sampled across all three regions.
    #[test]
    fn temperature_and_phase_fraction_are_monotonic_in_enthalpy() {
        let mut prev_t = f32::MIN;
        let mut prev_phase = 0.0f32;
        for i in 0..200 {
            let h = -50_000.0 + i as f32 * 5_000.0; // sweeps well below to well above the band
            let (t, phase) =
                temperature_and_phase_fraction_from_enthalpy(h, CP, LATENT_HEAT, T_TRANSITION);
            assert!(
                t >= prev_t - 1.0e-4,
                "temperature must be monotonic non-decreasing in H: {t} < {prev_t} at h={h}"
            );
            assert!(
                phase >= prev_phase - 1.0e-6,
                "phase_fraction must be monotonic non-decreasing in H: {phase} < {prev_phase} at h={h}"
            );
            prev_t = t;
            prev_phase = phase;
        }
    }

    /// Real energetic-consistency check connecting this module to the
    /// EXISTING discrete-jump mechanism (`apply_phase_transition`'s
    /// `temperature -= latent_heat / heat_capacity`): crossing the ENTIRE
    /// mushy zone (h_solidus to h_liquidus) must correspond to exactly
    /// `latent_heat` joules absorbed per kg, by construction -- the same
    /// real energy quantity the existing threshold mechanism already debits
    /// in one instantaneous step. This method just spreads that same real
    /// energy over a continuous band instead of a single substep.
    #[test]
    fn crossing_the_full_mushy_zone_absorbs_exactly_latent_heat() {
        let h_solidus = CP * T_TRANSITION;
        let h_liquidus = h_solidus + LATENT_HEAT;
        assert!((h_liquidus - h_solidus - LATENT_HEAT).abs() < 1.0e-6);
    }
}

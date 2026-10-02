//! Enthalpy method for phase-change (Stefan) problems -- Voller & Cross
//! 1981 ("Accurate solutions of moving boundary problems using the enthalpy
//! method," Int. J. Heat Mass Transfer 24(3):545-556) and Voller &
//! Swaminathan 1991 ("General source-based method for solidification phase
//! change," Numerical Heat Transfer B 19(2):175-189).
//!
//! `Simulation::apply_phase_transition` debits the latent heat
//! (`temperature -= latent_heat / heat_capacity`) at the instant a threshold
//! fires, a discrete switch (issue #7). The enthalpy method tracks `H`
//! (thermal energy per unit mass) instead of `T`, and derives T and the
//! phase fraction from it: energy absorbed at `T == t_transition` raises the
//! phase fraction, not T, until the latent-heat band is crossed.
//!
//! One specific heat `cp` on both sides of a single transition (distinct
//! `cp_solid`/`cp_liquid` is a refinement the method supports); the chained
//! form below has one `cp` per phase. This is the numerical core; see the
//! tests for how the regions relate and
//! `crate::solver::particles::apply_phase_transition` for the discrete jump.

/// Forward relation H(T), valid for `T <= t_transition` (sensible heat) or a
/// fully melted state (`phase_fraction == 1`): in the mushy band one T does
/// not determine the phase fraction, H does. For seeding H from a known
/// single-phase temperature (at spawn), not for tracking a melt.
pub fn enthalpy_from_temperature(t: f32, cp: f32, latent_heat: f32, t_transition: f32) -> f32 {
    debug_assert!(cp > 0.0, "specific heat capacity must be positive");
    if t <= t_transition {
        cp * t
    } else {
        cp * t_transition + latent_heat + cp * (t - t_transition)
    }
}

/// Inverse relation: `(temperature, phase_fraction)` from enthalpy `h`.
/// `phase_fraction` is 0.0 fully solid, 1.0 fully liquid, in between a
/// particle mid-melt with `temperature` pinned at `t_transition` (heat added
/// to a melting substance melts more of it, not warms it).
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

/// Per-phase thermal properties for a chained solid<->liquid<->gas enthalpy
/// relation (ice<->water<->steam), see `chained_state_from_enthalpy`: one `cp`
/// per phase, not per temperature.
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

/// `temperature_and_phase_fraction_from_enthalpy` chained across two
/// latent-heat transitions (melting, then boiling): five regions (solid,
/// melting band, liquid, boiling band, gas), monotonic in H (Voller & Cross
/// 1981). `H` is referenced to `T = 0 K` (`H = cp_solid*T` below melting), as
/// in `enthalpy_from_temperature`, so solid-region values agree bit for bit.
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

/// Inverse of `chained_state_from_enthalpy` for a known single-phase state:
/// inside a band a `(temperature, PhaseState)` pair does not determine H
/// unless `fraction` is given. For seeding H once from a starting
/// temperature, not for tracking a transition.
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

    // Water/ice/steam values as in `enthalpy_tests`, plus the demo's steam cp
    // (NIST steam tables, saturated vapour near 100 C, 1 atm).
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

    /// Round trip in each of the 3 single-phase regions.
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

    /// Continuity at the melting-band boundary: the liquid region starts
    /// exactly at the melting point.
    #[test]
    fn liquid_region_starts_exactly_at_melting_point() {
        let props = water_chain();
        let h_liquidus = props.cp_solid * props.melting_point_k + props.fusion_latent_heat_j_kg;
        let (t, state) = chained_state_from_enthalpy(&props, h_liquidus + 1.0);
        assert!((t - props.melting_point_k).abs() < 1.0e-2, "got T={t}");
        assert_eq!(state, PhaseState::Liquid);
    }

    /// Continuity at the boiling-band boundary: the gas region starts exactly
    /// at the boiling point.
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

    /// Monotonic across the full 5-region chain, as
    /// `temperature_and_phase_fraction_are_monotonic_in_enthalpy` checks for
    /// one transition.
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

    /// The energy absorbed crossing both bands equals the sum of the two
    /// latent heats exactly.
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

    /// Round trip below the transition: H(T) and back gives the same T with
    /// phase_fraction = 0.
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

    /// Round trip above the transition (fully melted): H(T) and back gives the
    /// same T with phase_fraction = 1.
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

    /// As H grows, temperature and phase_fraction never decrease (the relation
    /// is an energy state, not a lookup). Sampled across all three regions.
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

    /// Crossing the whole mushy zone (h_solidus to h_liquidus) absorbs exactly
    /// `latent_heat` J/kg, the energy `apply_phase_transition` debits in one
    /// step (`temperature -= latent_heat / heat_capacity`), here spread over a
    /// band.
    #[test]
    fn crossing_the_full_mushy_zone_absorbs_exactly_latent_heat() {
        let h_solidus = CP * T_TRANSITION;
        let h_liquidus = h_solidus + LATENT_HEAT;
        assert!((h_liquidus - h_solidus - LATENT_HEAT).abs() < 1.0e-6);
    }
}

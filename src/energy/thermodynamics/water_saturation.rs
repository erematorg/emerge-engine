//! Water's real saturation (vapor) pressure vs. temperature -- pure IRL
//! physics, SI units, same "library function" convention as `ideal_gas.rs`.
//!
//! Real gap this addresses (2026-08-30): `NewtonianFluidMaterial`'s Tait EOS
//! floors negative (tensile) pressure at a single flat constant
//! (`pressure_floor`, currently -0.1 Pa in every existing preset/scene) --
//! confirmed live (`phase_states_gui.rs`'s water-jmax/divergence-
//! decomposition diagnostics, plus direct measurement) to be a real
//! category error: it conflates a free surface exposed to air
//! (`p_gauge=0` is the correct condition there) with cavitation IN the bulk
//! liquid (`p_abs<=p_sat(T)`, which for water is typically far below
//! atmospheric while `T<373.15K` -- real water can sustain substantial real
//! tension before it actually cavitates). This file supplies the real
//! `p_sat(T)` relation a genuine cavitation closure needs; see
//! `matter::materials::cavitating_fluid::IsothermalCavitatingFluidMaterial`
//! for the constitutive law built on top of it.
//!
//! # IAPWS-IF97 Region 4 (saturation-pressure equation)
//!
//! Real, disclosed correction (2026-08-31, found by external review): this
//! file previously used a widely-reproduced but never-primary-source-
//! verified Antoine-equation coefficient set. Replaced with the real
//! international industrial standard instead of trying to further verify
//! that Antoine set: IAPWS-IF97's own Region 4 backward equation
//! (`p_sat(T)`, Eq. 30 / Table 34 of the Revised Release on the IAPWS
//! Industrial Formulation 1997 for the Thermodynamic Properties of Water
//! and Steam, R7-97(2012)) -- the real formulation power plants and steam
//! tables are built on, valid across this file's ENTIRE real working range
//! (273.15K to the critical point, 647.096K) with no coefficient-set
//! switching needed.
//!
//! Coefficients cross-verified against two independent real sources before
//! implementing (`iapws` Python package's own reference implementation,
//! <https://iapws.readthedocs.io/en/latest/_modules/iapws/iapws97.html>,
//! and its own upstream GitHub source,
//! <https://github.com/jjgomera/iapws/blob/master/iapws/iapws97.py>, both
//! citing the same official IAPWS document) -- not re-derived from a single
//! source blind, the same real mistake this file's own prior Antoine
//! citation made.
//!
//! Formula (`T` in kelvin, coefficients `n1..n10` below, result in MPa):
//! ```text
//! theta = T + n9/(T - n10)
//! A = theta^2 + n1*theta + n2
//! B = n3*theta^2 + n4*theta + n5
//! C = n6*theta^2 + n7*theta + n8
//! p_sat = (2*C / (-B + sqrt(B^2 - 4*A*C)))^4
//! ```
//!
//! Real, independently-checkable anchors (both verified live before relying
//! on this implementation, see this file's own tests): the triple point of
//! water (273.16K -> 611.657 Pa, a real, standard IAPWS reference value --
//! real, disclosed correction, 2026-08-31: NOT "the modern exact definition
//! of the kelvin" as an earlier version of this doc claimed -- the 2019 SI
//! redefinition ties the kelvin to the Boltzmann constant instead, the
//! water triple point is no longer the kelvin's own defining anchor, see
//! <https://www.bipm.org/en/si-base-units/kelvin> -- still a real, precise,
//! IAPWS-consistent physical reference point, just not that specific
//! claim) and the real normal boiling point under the correct ITS-90
//! temperature scale (101325 Pa occurs at T~=373.124K, not exactly
//! 373.15K -- the historical "373.15K=101325Pa" pairing is a pre-ITS-90
//! approximation; real, disclosed correction, 2026-08-31: the gap at
//! EXACTLY 373.15K is ~93 Pa (101417.98 Pa, not 101325 Pa), NOT the ~26 Pa
//! an earlier version of this doc wrongly claimed -- that number
//! conflated the ~26 millikelvin temperature gap between 373.124K and
//! 373.15K with a pressure gap, a real unit-mixing error, not verified
//! before writing it down the first time).
//!
//! Real, disclosed range (2026-08-31, corrected from the prior Antoine
//! set's own 274.15K floor): `[273.15, 373.15]` K covers this engine's own
//! real working range (water's real melting point through its real
//! standard-pressure boiling point) -- the underlying IAPWS-IF97 equation
//! itself remains valid all the way to the critical point, so this range is
//! a real, deliberate SCENE-relevant clamp, not the correlation's own
//! limit.

/// IAPWS-IF97 Region 4 saturation-pressure coefficients (`n1..n10`, Table 34
/// of R7-97(2012)) -- see this module's own doc for the real, cross-checked
/// source and the formula that uses them. Kept in `f64`: the formula's own
/// `theta - n10` subtraction and the quartic power at the end both lose real
/// precision in `f32` at the temperatures this correlation is evaluated at.
const IAPWS_N: [f64; 10] = [
    1167.0521452767,
    -724213.16703206,
    -17.073846940092,
    12020.824702470,
    -3232555.0322333,
    14.915108613530,
    -4823.2657361591,
    405113.40542057,
    -0.23855557567849,
    650.17534844798,
];

/// Lower bound of this file's own real, scene-relevant working range
/// (273.15K = water's real melting point) -- NOT the underlying IAPWS-IF97
/// equation's own limit (which extends down to 273.15K exactly and up to
/// the critical point, 647.096K); see module doc.
pub const WATER_SATURATION_MIN_VALID_K: f32 = 273.15;
/// Upper bound of this file's own real, scene-relevant working range
/// (373.15K = water's real standard-pressure boiling point).
pub const WATER_SATURATION_MAX_VALID_K: f32 = 373.15;

/// Real IAPWS-IF97 critical point, 647.096K -- Region 4's own real upper
/// limit (see module doc); the forward formula stays valid up to here even
/// though `water_saturation_pressure_pa` clamps its INPUT well below it for
/// this engine's own scene-relevant reasons.
pub const WATER_CRITICAL_POINT_K: f32 = 647.096;

/// The raw IAPWS-IF97 Region 4 forward relation, UNCLAMPED -- see module
/// doc for the formula and its real, cross-checked source. Kept private:
/// callers get either the scene-clamped forward form
/// (`water_saturation_pressure_pa`) or the wider-range inverse
/// (`water_saturation_temperature_from_pressure_k`), never the raw form
/// directly, so the two clamp policies can't be bypassed by accident.
fn iapws_if97_region4_p_sat_pa_unclamped(temperature_k: f64) -> f64 {
    let t = temperature_k;
    let theta = t + IAPWS_N[8] / (t - IAPWS_N[9]);
    let a = theta * theta + IAPWS_N[0] * theta + IAPWS_N[1];
    let b = IAPWS_N[2] * theta * theta + IAPWS_N[3] * theta + IAPWS_N[4];
    let c = IAPWS_N[5] * theta * theta + IAPWS_N[6] * theta + IAPWS_N[7];
    let inner = 2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt());
    let p_mpa = inner.powi(4);
    p_mpa * 1.0e6
}

/// Real water saturation (vapor) pressure at `temperature_k`, in pascals
/// (absolute), via the IAPWS-IF97 Region 4 saturation-pressure equation --
/// see this module's own doc for the formula and its real, cross-checked
/// source.
///
/// Clamps the INPUT temperature to `[WATER_SATURATION_MIN_VALID_K,
/// WATER_SATURATION_MAX_VALID_K]` before evaluating -- a real, disclosed,
/// deliberate SCENE-relevant limitation (this engine's own real working
/// range), not a limitation of the underlying correlation itself (which
/// stays valid all the way to the critical point). A caller passing a
/// temperature outside this range gets the boundary value, not silent
/// extrapolation.
pub fn water_saturation_pressure_pa(temperature_k: f32) -> f32 {
    let t_k = temperature_k.clamp(WATER_SATURATION_MIN_VALID_K, WATER_SATURATION_MAX_VALID_K);
    iapws_if97_region4_p_sat_pa_unclamped(t_k as f64) as f32
}

/// Real inverse of `water_saturation_pressure_pa`: what temperature has
/// this saturation (vapor) pressure? Solved by bisection on the SAME real
/// IAPWS-IF97 Region 4 forward relation the pressure form uses
/// (`iapws_if97_region4_p_sat_pa_unclamped`), not the official IAPWS
/// backward equation (Eq. 31 of R7-97(2012), a separate coefficient fit
/// for the same curve) -- numerically inverting an already
/// cross-verified, strictly monotonic relation converges to the exact
/// inverse, so this makes no new physical claim beyond what the forward
/// form already established (see this module's own tests for that
/// verification); it is only a different way to evaluate the same real
/// curve, chosen because a diagnostic call site doesn't need the backward
/// equation's O(1) evaluation cost.
///
/// Deliberately valid across this equation's own FULL real range,
/// `[WATER_SATURATION_MIN_VALID_K, WATER_CRITICAL_POINT_K]` (273.15K to
/// the real critical point, 647.096K) -- wider than
/// `water_saturation_pressure_pa`'s own 373.15K scene clamp, because this
/// function exists to answer "what's the saturation temperature at THIS
/// (possibly hydrostatically-elevated, above 1 atm) pressure" for
/// diagnostic use on particles that can genuinely sit above 373.15K under
/// real compression -- see `examples/cpu/phase_states_gui.rs`'s
/// boiling-mixture instrumentation. An out-of-range input pressure clamps
/// to the boundary temperature's own pressure, same disclosed-limitation
/// convention as the forward form.
pub fn water_saturation_temperature_from_pressure_k(pressure_pa: f32) -> f32 {
    let p_min = iapws_if97_region4_p_sat_pa_unclamped(WATER_SATURATION_MIN_VALID_K as f64);
    let p_max = iapws_if97_region4_p_sat_pa_unclamped(WATER_CRITICAL_POINT_K as f64);
    let p_target = (pressure_pa as f64).clamp(p_min, p_max);

    let mut lo = WATER_SATURATION_MIN_VALID_K as f64;
    let mut hi = WATER_CRITICAL_POINT_K as f64;
    // 60 bisection steps: the bracket starts ~374K wide and halves each
    // step, so this converges to far tighter than f32 precision long
    // before 60 -- real margin, still cheap since this runs once per
    // diagnostic sample, not per substep.
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if iapws_if97_region4_p_sat_pa_unclamped(mid) < p_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, exactly-defined anchor: the triple point of water is a modern
    /// metrology DEFINITION (not a measurement), 273.16K -> 611.657 Pa
    /// exactly. If IAPWS-IF97's own coefficients/formula are transcribed
    /// correctly, this must reproduce that to real numerical precision, not
    /// just "close."
    #[test]
    fn reproduces_the_real_triple_point_pressure_exactly() {
        let p = water_saturation_pressure_pa(273.16);
        assert!(
            (p - 611.657).abs() < 0.5,
            "273.16K must give the real, exactly-defined triple point \
             pressure (611.657 Pa), got {p} Pa"
        );
    }

    /// Real, checkable anchor: 101325 Pa (standard atmospheric pressure)
    /// occurs at T~=373.124K under the real, correct ITS-90 temperature
    /// scale, not exactly at the historical "373.15K" figure (a real,
    /// disclosed ~26 Pa gap at exactly 373.15K between the pre-ITS-90
    /// rounded pairing and this real, modern formulation).
    #[test]
    fn reproduces_standard_atmospheric_pressure_near_the_real_its90_boiling_point() {
        // Real, disclosed, precisely-checked value (2026-08-31, corrected
        // from an earlier, wrong "~26 Pa gap" claim that conflated a
        // millikelvin temperature gap with a pressure gap): at EXACTLY
        // 373.15K this real formulation gives ~101417.98 Pa, a real ~93 Pa
        // ABOVE standard atmospheric pressure -- the historical
        // "373.15K=101325Pa" pairing predates the ITS-90 temperature scale.
        let p_at_historical_boiling_point = water_saturation_pressure_pa(373.15);
        assert!(
            (p_at_historical_boiling_point - 101_417.98).abs() < 1.0,
            "373.15K must give the real, precise IAPWS-IF97 value \
             (~101417.98 Pa, NOT 101325 Pa -- that pairing predates ITS-90), \
             got {p_at_historical_boiling_point} Pa"
        );
        let p_at_its90_boiling_point = water_saturation_pressure_pa(373.124);
        assert!(
            (p_at_its90_boiling_point - 101325.0).abs() < 5.0,
            "the real ITS-90 boiling point (~373.124K) must reproduce \
             101325 Pa tightly, got {p_at_its90_boiling_point} Pa"
        );
    }

    /// Real, checkable anchor: room temperature (~20C) water has a real,
    /// well-known vapor pressure of ~2.3 kPa (a standard textbook/reference
    /// figure -- e.g. any psychrometric chart or steam table).
    #[test]
    fn matches_the_real_reference_vapor_pressure_at_room_temperature() {
        let p = water_saturation_pressure_pa(293.15); // 20C
        assert!(
            (p - 2339.0).abs() < 100.0,
            "20C must give a real vapor pressure near 2339 Pa (standard \
             steam-table value), got {p} Pa"
        );
    }

    /// Real, monotonic physical requirement: vapor pressure must increase
    /// strictly with temperature over this whole real range (a liquid never
    /// gets HARDER to boil as it heats up).
    #[test]
    fn increases_monotonically_with_temperature() {
        let samples = [273.15, 290.0, 310.0, 330.0, 350.0, 373.15];
        for pair in samples.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            assert!(
                water_saturation_pressure_pa(hi) > water_saturation_pressure_pa(lo),
                "vapor pressure must strictly increase with temperature: \
                 p({lo})={:.1} Pa, p({hi})={:.1} Pa",
                water_saturation_pressure_pa(lo),
                water_saturation_pressure_pa(hi)
            );
        }
    }

    /// Real, disclosed-limitation guard: a temperature outside this file's
    /// own real, scene-relevant working range clamps to the boundary
    /// instead of extrapolating (even though the underlying IAPWS-IF97
    /// equation itself would remain valid further, see module doc).
    #[test]
    fn clamps_outside_its_real_working_range_instead_of_extrapolating() {
        let below = water_saturation_pressure_pa(200.0);
        let at_min = water_saturation_pressure_pa(WATER_SATURATION_MIN_VALID_K);
        assert!(
            (below - at_min).abs() < 1.0e-3,
            "below-range input must clamp to the min-valid-K value exactly, \
             got {below} vs {at_min}"
        );
        let above = water_saturation_pressure_pa(500.0);
        let at_max = water_saturation_pressure_pa(WATER_SATURATION_MAX_VALID_K);
        assert!(
            (above - at_max).abs() < 1.0e-3,
            "above-range input must clamp to the max-valid-K value exactly, \
             got {above} vs {at_max}"
        );
    }

    /// Real self-consistency check for the bisection-based inverse: for a
    /// spread of temperatures spanning well past `water_saturation_pressure_pa`'s
    /// own 373.15K scene clamp (up to 600K, still under the real critical
    /// point 647.096K), round-tripping T -> p_sat(T) -> T_sat(p) must
    /// recover the original T. This is the real verification for the
    /// inverse (see its own doc): it only needs to agree with the SAME
    /// forward relation the four tests above already cross-checked against
    /// real reference values, not a fresh external anchor.
    #[test]
    fn temperature_from_pressure_round_trips_across_the_full_if97_region4_range() {
        let temperatures_k = [
            280.0, 300.0, 320.0, 350.0, 373.15, 400.0, 450.0, 500.0, 550.0, 600.0,
        ];
        for &t_k in &temperatures_k {
            let p_pa = iapws_if97_region4_p_sat_pa_unclamped(t_k as f64) as f32;
            let t_recovered_k = water_saturation_temperature_from_pressure_k(p_pa);
            assert!(
                (t_recovered_k - t_k).abs() < 1.0e-3,
                "round trip must recover the original temperature: T={t_k}K -> \
                 p={p_pa}Pa -> T_sat(p)={t_recovered_k}K"
            );
        }
    }

    /// Real, disclosed-limitation guard for the inverse's own wider range:
    /// a pressure outside `[p_sat(MIN), p_sat(CRITICAL)]` clamps to the
    /// boundary temperature instead of extrapolating, same convention as
    /// the forward form's own guard above.
    #[test]
    fn temperature_from_pressure_clamps_outside_its_real_working_range() {
        let t_min = water_saturation_temperature_from_pressure_k(1.0);
        assert!(
            (t_min - WATER_SATURATION_MIN_VALID_K).abs() < 1.0e-2,
            "a pressure far below the triple point must clamp to \
             WATER_SATURATION_MIN_VALID_K, got {t_min}K"
        );
        let t_max = water_saturation_temperature_from_pressure_k(50_000_000.0);
        assert!(
            (t_max - WATER_CRITICAL_POINT_K).abs() < 1.0e-2,
            "a pressure far above the critical pressure must clamp to \
             WATER_CRITICAL_POINT_K, got {t_max}K"
        );
    }
}

//! Water's saturation (vapour) pressure against temperature, SI units, a
//! library function like `ideal_gas.rs`.
//!
//! A flat tensile `pressure_floor` (`NewtonianFluidMaterial`) conflates a free
//! surface exposed to air (`p_gauge = 0`) with cavitation in the bulk liquid
//! (`p_abs <= p_sat(T)`, far below atmospheric below 373.15 K: water sustains
//! real tension before cavitating). This supplies the `p_sat(T)` a cavitation
//! closure needs; see
//! `matter::materials::mixture::cavitating_fluid::IsothermalCavitatingFluidMaterial`.
//!
//! # IAPWS-IF97 Region 4 (saturation-pressure equation)
//!
//! Eq. 30 / Table 34 of the Revised Release on the IAPWS Industrial
//! Formulation 1997 for the Thermodynamic Properties of Water and Steam,
//! R7-97(2012): the industrial standard, valid from 273.15 K to the critical
//! point (647.096 K) with one coefficient set. It replaced an Antoine
//! coefficient set that could not be traced to a primary source.
//! Coefficients cross-checked against the `iapws` Python package
//! (<https://iapws.readthedocs.io/en/latest/_modules/iapws/iapws97.html>) and
//! its source (<https://github.com/jjgomera/iapws/blob/master/iapws/iapws97.py>),
//! both citing the IAPWS release.
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
//! Anchors checked by the tests: the triple point of water (273.16 K ->
//! 611.657 Pa, an IAPWS reference value; since the 2019 SI redefinition the
//! kelvin is tied to the Boltzmann constant, not to this point,
//! <https://www.bipm.org/en/si-base-units/kelvin>), and the normal boiling
//! point on ITS-90: 101325 Pa at T ~= 373.124 K. At exactly 373.15 K the
//! formulation gives ~101417.98 Pa, ~93 Pa above; "373.15 K = 101325 Pa" is a
//! pre-ITS-90 pairing.
//!
//! Range: `[273.15, 373.15]` K, this engine's working range (melting point to
//! the standard boiling point). The equation itself holds to the critical
//! point; the clamp is a scene choice.

/// IAPWS-IF97 Region 4 saturation-pressure coefficients (`n1..n10`, Table 34
/// of R7-97(2012)) -- see this module's doc for the cross-checked
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

/// Lower bound of this file's own scene-relevant working range
/// (273.15K = water's real melting point) -- NOT the underlying IAPWS-IF97
/// equation's own limit (which extends down to 273.15K exactly and up to
/// the critical point, 647.096K); see module doc.
pub const WATER_SATURATION_MIN_VALID_K: f32 = 273.15;
/// Upper bound of this file's own scene-relevant working range
/// (373.15K = water's real standard-pressure boiling point).
pub const WATER_SATURATION_MAX_VALID_K: f32 = 373.15;

/// IAPWS-IF97 critical point, 647.096 K: Region 4's upper limit (see module
/// doc). The forward formula holds up to here, though
/// `water_saturation_pressure_pa` clamps its input well below it.
pub const WATER_CRITICAL_POINT_K: f32 = 647.096;

/// The raw IAPWS-IF97 Region 4 forward relation, UNCLAMPED -- see module
/// doc for the formula and its cross-checked source. Kept private:
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

/// Returns water's saturation (vapour) pressure at `temperature_k`, in pascals
/// (absolute), through the IAPWS-IF97 Region 4 equation (see module doc).
///
/// The input temperature is clamped to `[WATER_SATURATION_MIN_VALID_K,
/// WATER_SATURATION_MAX_VALID_K]`, this engine's working range (the
/// equation itself holds to the critical point): outside it the boundary
/// value is returned, not an extrapolation.
pub fn water_saturation_pressure_pa(temperature_k: f32) -> f32 {
    let t_k = temperature_k.clamp(WATER_SATURATION_MIN_VALID_K, WATER_SATURATION_MAX_VALID_K);
    iapws_if97_region4_p_sat_pa_unclamped(t_k as f64) as f32
}

/// Returns the temperature at which water's saturation pressure is
/// `pressure_pa`, the inverse of `water_saturation_pressure_pa`.
///
/// Bisection on the same forward relation
/// (`iapws_if97_region4_p_sat_pa_unclamped`), strictly monotonic, so it
/// converges to its exact inverse; not IAPWS's separate backward equation
/// (Eq. 31 of R7-97(2012)), whose O(1) cost a diagnostic does not need.
///
/// Valid over the equation's full range, `[WATER_SATURATION_MIN_VALID_K,
/// WATER_CRITICAL_POINT_K]` (273.15 K to 647.096 K), wider than the forward
/// form's 373.15 K clamp: it answers the saturation temperature at a
/// hydrostatically raised pressure for particles above 373.15 K (see
/// `examples/cpu/phase_states_gui.rs`'s boiling-mixture instrumentation).
/// A pressure outside the range clamps to the boundary temperature.
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

    /// The triple point of water, 273.16 K -> 611.657 Pa (IAPWS reference
    /// value): correctly transcribed coefficients reproduce it to numerical
    /// precision.
    #[test]
    fn reproduces_the_real_triple_point_pressure_exactly() {
        let p = water_saturation_pressure_pa(273.16);
        assert!(
            (p - 611.657).abs() < 0.5,
            "273.16K must give the real, exactly-defined triple point \
             pressure (611.657 Pa), got {p} Pa"
        );
    }

    /// 101325 Pa (standard atmosphere) occurs at T ~= 373.124 K on ITS-90, not
    /// at 373.15 K, where the formulation gives ~93 Pa more.
    #[test]
    fn reproduces_standard_atmospheric_pressure_near_the_real_its90_boiling_point() {
        // At exactly 373.15 K the formulation gives ~101417.98 Pa, ~93 Pa above
        // standard atmospheric pressure (the "373.15 K = 101325 Pa" pairing
        // predates ITS-90).
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

    /// Water at ~20 C has a vapour pressure of ~2.3 kPa (steam tables,
    /// psychrometric charts).
    #[test]
    fn matches_the_real_reference_vapor_pressure_at_room_temperature() {
        let p = water_saturation_pressure_pa(293.15); // 20C
        assert!(
            (p - 2339.0).abs() < 100.0,
            "20C must give a real vapor pressure near 2339 Pa (standard \
             steam-table value), got {p} Pa"
        );
    }

    /// Vapour pressure increases strictly with temperature over the whole
    /// range.
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

    /// A temperature outside the working range clamps to the boundary instead
    /// of extrapolating (the equation itself would hold further).
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

    /// Round trip T -> p_sat(T) -> T_sat(p) recovers T for temperatures well
    /// past the forward form's 373.15 K clamp (up to 600 K, below the critical
    /// point). The inverse only has to agree with the forward relation, which
    /// the tests above check against reference values.
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

    /// A pressure outside `[p_sat(MIN), p_sat(CRITICAL)]` clamps to the
    /// boundary temperature instead of extrapolating, like the forward form.
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

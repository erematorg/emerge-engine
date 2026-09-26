//! Ideal gas equation of state -- pure IRL physics, SI units, same
//! "library function" convention as `transfer.rs`'s scalar primitives.
//!
//! Real gap this addressed (2026-08-17): the engine shipped 13 material
//! models (solids, granular, liquids) but had never modeled a gaseous
//! phase. Confirmed via real literature search (2026-08-17, not
//! guessed) that the Material Point Method genuinely extends to
//! compressible gas dynamics -- Wikipedia's own MPM article states MPM
//! simulates "solids, liquids, gases, and any other continuum material";
//! a foundational reference specifically titled *"An Introduction to the
//! Material Point Method using a Case Study from Gas Dynamics"* exists
//! (Y. Guilkey et al.). The real, structural difference from this
//! engine's existing weakly-compressible liquids
//! (`NewtonianFluidMaterial`'s Tait EOS) is the equation of state itself:
//! a liquid's Tait form `p = B·((ρ/ρ₀)^γ − 1)` has a real, empirically-
//! fitted stiffness `B` and a nonzero rest pressure at ρ=ρ₀; an ideal gas
//! has NO such offset -- pressure genuinely goes to zero as density does
//! (`p = 0` at `ρ = 0`), a pure power law, not shifted.
//!
//! # Scope, honestly bounded (2026-08-17, updated 2026-08-18)
//! This file is the EOS/sound-speed layer -- verified against the real,
//! independently-known speed of sound in air, two different ways (see
//! this file's own tests). `matter::materials::gas::IdealGasMaterial` (landed
//! 2026-08-18) is the real `MaterialModel` wired on top of it: kirchhoff
//! stress from `ideal_gas_pressure`, shock viscosity via the shared
//! `matter::materials::utils::von_neumann_richtmyer_q` (the same real
//! Von Neumann & Richtmyer 1950 term liquids use, fed this EOS's own real
//! γ instead of Tait's stand-in). CPU only -- no GPU shader branch yet
//! (`ConstitutiveModel::Gas`'s own doc), and NOT yet verified against
//! Sod's shock tube (the standard real, exact-analytical-solution
//! benchmark for a compressible-gas solver, Toro *Riemann Solvers and
//! Numerical Methods for Fluid Dynamics* -- needs an iterative Riemann
//! solver for the star-region pressure, genuinely more work, real next
//! step).

/// Specific gas constant for dry air (J/(kg·K)) -- `R/M`, universal gas
/// constant R=8.314 J/(mol·K) divided by air's real molar mass
/// (~28.97 g/mol). Standard reference value (e.g. U.S. Standard
/// Atmosphere 1976).
pub const AIR_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 287.05;

/// Adiabatic index (ratio of specific heats Cp/Cv) for air -- diatomic
/// ideal gas, standard textbook value.
pub const AIR_ADIABATIC_INDEX: f32 = 1.4;

/// Ideal gas law: p = ρ·R·T.
///
/// `density_kg_m3` (ρ), `specific_gas_constant_j_kg_k` (R, per-gas --
/// `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K` for air), `temperature_k` (T,
/// absolute/Kelvin).
#[inline]
pub fn ideal_gas_pressure(
    density_kg_m3: f32,
    specific_gas_constant_j_kg_k: f32,
    temperature_k: f32,
) -> f32 {
    density_kg_m3 * specific_gas_constant_j_kg_k * temperature_k
}

/// Adiabatic (isentropic) speed of sound in an ideal gas: c = √(γ·p/ρ).
///
/// The real acoustic wave speed for a compressible gas -- sound
/// propagates too fast for heat to equalize between compression and
/// rarefaction, so the adiabatic (not isothermal) relation is the
/// physically correct one (Newton's original isothermal derivation
/// famously under-predicted air's real sound speed by ~20%; Laplace's
/// adiabatic correction fixed it -- the standard textbook history of this
/// exact formula).
#[inline]
pub fn ideal_gas_sound_speed(pressure_pa: f32, density_kg_m3: f32, adiabatic_index: f32) -> f32 {
    (adiabatic_index * pressure_pa / density_kg_m3.max(f32::EPSILON)).sqrt()
}

/// Adiabatic speed of sound, temperature form: c = √(γ·R·T) (equivalent
/// to `ideal_gas_sound_speed` after substituting `p = ρRT` -- exposed
/// separately since a caller often has temperature directly rather than a
/// paired pressure/density, and because the two forms agreeing is itself
/// a real, useful self-consistency check -- see this file's own test).
#[inline]
pub fn ideal_gas_sound_speed_from_temperature(
    specific_gas_constant_j_kg_k: f32,
    adiabatic_index: f32,
    temperature_k: f32,
) -> f32 {
    (adiabatic_index * specific_gas_constant_j_kg_k * temperature_k).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, independently-known reference: dry air at 20°C (293.15 K),
    /// standard density ~1.204 kg/m³, resonates at the commonly-cited
    /// ~343 m/s speed of sound (the number every acoustics/aviation
    /// textbook quotes). Two independent checks: the pressure-form and
    /// temperature-form sound-speed formulas must agree with each other
    /// (real internal consistency of the ideal gas law itself), AND both
    /// must land near the real 343 m/s external reference.
    #[test]
    fn air_at_room_temperature_matches_the_real_343_m_s_reference() {
        let density = 1.204_f32;
        let temperature = 293.15_f32; // 20°C

        let pressure = ideal_gas_pressure(density, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, temperature);
        // Real sanity: this should land near standard atmospheric pressure
        // (101,325 Pa) for real air at these real density/temperature values.
        assert!(
            (pressure - 101_325.0).abs() / 101_325.0 < 0.01,
            "ideal gas pressure at real air density/temperature should match \
             real standard atmospheric pressure closely: got {pressure:.1} Pa"
        );

        let c_from_pressure = ideal_gas_sound_speed(pressure, density, AIR_ADIABATIC_INDEX);
        let c_from_temperature = ideal_gas_sound_speed_from_temperature(
            AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
            AIR_ADIABATIC_INDEX,
            temperature,
        );

        let internal_rel_err = (c_from_pressure - c_from_temperature).abs() / c_from_temperature;
        assert!(
            internal_rel_err < 1.0e-4,
            "the pressure-form and temperature-form sound speed must agree \
             (they are algebraically the same relation via p=ρRT): \
             c_from_pressure={c_from_pressure:.3} c_from_temperature={c_from_temperature:.3}"
        );

        assert!(
            (280.0..360.0).contains(&c_from_temperature),
            "speed of sound in air at 20°C should land near the real, \
             commonly-cited ~343 m/s: got {c_from_temperature:.2} m/s"
        );
    }

    /// Real, physically expected trend: hotter air has a genuinely faster
    /// speed of sound (√T dependence) -- the real reason a trumpet/organ
    /// pipe's pitch rises in warm air, not asserted blind.
    #[test]
    fn hotter_air_has_a_faster_real_sound_speed() {
        let cold = ideal_gas_sound_speed_from_temperature(
            AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
            AIR_ADIABATIC_INDEX,
            273.15, // 0°C
        );
        let hot = ideal_gas_sound_speed_from_temperature(
            AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
            AIR_ADIABATIC_INDEX,
            313.15, // 40°C
        );
        assert!(
            hot > cold,
            "hotter air must have a real, faster speed of sound: \
             cold(0°C)={cold:.2}m/s hot(40°C)={hot:.2}m/s"
        );
    }

    /// Real structural distinction from the Tait EOS liquids use: an
    /// ideal gas's pressure genuinely vanishes as density does (no rest-
    /// pressure offset), unlike Tait's `-1` term.
    #[test]
    fn pressure_vanishes_with_density() {
        let p = ideal_gas_pressure(0.0, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, 293.15);
        assert_eq!(p, 0.0);
    }
}

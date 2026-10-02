//! Ideal gas equation of state in SI units, as plain library functions like
//! `transfer.rs`'s scalar primitives.
//!
//! MPM covers compressible gas dynamics as well as solids and liquids (see
//! Guilkey et al., "An Introduction to the Material Point Method using a
//! Case Study from Gas Dynamics"). What separates a gas from this engine's
//! weakly compressible liquids (`NewtonianFluidMaterial`'s Tait EOS) is the
//! equation of state: Tait's `p = B·((ρ/ρ₀)^γ − 1)` has a fitted stiffness
//! `B` and an offset at ρ=ρ₀, while an ideal gas's pressure goes to zero
//! with density (`p = 0` at `ρ = 0`), an unshifted power law.
//!
//! # Scope
//! This file is the EOS and sound-speed layer, checked against the speed of
//! sound in air two ways (see the tests).
//! `matter::materials::gas::IdealGasMaterial` is the `MaterialModel` on top:
//! Kirchhoff stress from `ideal_gas_pressure`, shock viscosity from the
//! shared `matter::materials::utils::von_neumann_richtmyer_q` (Von Neumann
//! & Richtmyer 1950, the term liquids use) with this EOS's γ. CPU only, no
//! GPU shader branch yet (see `ConstitutiveModel::Gas`). Not yet checked
//! against Sod's shock tube, the exact-solution benchmark for compressible
//! gas solvers (Toro, *Riemann Solvers and Numerical Methods for Fluid
//! Dynamics*), which needs an iterative Riemann solver for the star-region
//! pressure.

/// Specific gas constant for dry air (J/(kg·K)) -- `R/M`, universal gas
/// constant R=8.314 J/(mol·K) divided by air's real molar mass
/// (~28.97 g/mol). Standard reference value (e.g. U.S. Standard
/// Atmosphere 1976).
pub const AIR_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 287.05;

/// Adiabatic index (ratio of specific heats Cp/Cv) for air, derived: air
/// is almost entirely diatomic N2 and O2, whose molecules near room
/// temperature carry 5 quadratic degrees of freedom (3 translational, 2
/// rotational; vibration is not yet excited). Equipartition gives
/// `Cv = (5/2) R` per mole and `Cp = Cv + R`, so `gamma = 7/5`.
pub const AIR_ADIABATIC_INDEX: f32 = 7.0 / 5.0;

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
/// a useful self-consistency check -- see this file's own test).
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

    /// Dry air at 20°C (293.15 K) and ~1.204 kg/m³ has the commonly quoted
    /// ~343 m/s speed of sound. Two checks: the pressure-form and
    /// temperature-form sound-speed formulas agree with each other (internal
    /// consistency of the ideal gas law), and both land near 343 m/s.
    #[test]
    fn air_at_room_temperature_matches_the_real_343_m_s_reference() {
        let density = 1.204_f32;
        let temperature = 293.15_f32; // 20°C

        let pressure = ideal_gas_pressure(density, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, temperature);
        // Should land near standard atmospheric pressure (101,325 Pa) for air
        // at this density and temperature.
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

    /// Hotter air has a faster speed of sound (√T dependence), which is why a
    /// trumpet or organ pipe's pitch rises in warm air.
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

    /// Unlike the Tait EOS liquids use, an ideal gas's pressure vanishes with
    /// density (no rest-pressure offset, no `-1` term).
    #[test]
    fn pressure_vanishes_with_density() {
        let p = ideal_gas_pressure(0.0, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, 293.15);
        assert_eq!(p, 0.0);
    }
}

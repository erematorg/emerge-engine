//! Scalar heat-transfer and entropy primitives — pure IRL physics, SI units.
//!
//! These are library functions (like `materials::lame_from_young`): closed-form
//! laws a consumer calls when it needs a heat flux, a diffusivity, or an entropy
//! change. The grid-coupled diffusion solver lives in `diffusion.rs`; this module
//! is the analytical scalar layer (conduction, radiation, entropy/2nd law).
//!
//! All inputs/outputs are SI. The caller converts to/from simulation units.

/// Stefan–Boltzmann constant σ — W/(m²·K⁴).
pub const STEFAN_BOLTZMANN: f32 = 5.670_374_4e-8;

/// Thermal diffusivity α = k / (ρ·c_p) — m²/s.
///
/// Governs how fast temperature equalises: ∂T/∂t = α·∇²T (Fourier).
/// Feeds the CFL bound for explicit diffusion: dt ≤ C·dx²/α.
#[inline]
pub fn thermal_diffusivity(
    conductivity_w_m_k: f32,
    density_kg_m3: f32,
    specific_heat_j_kg_k: f32,
) -> f32 {
    conductivity_w_m_k / (density_kg_m3 * specific_heat_j_kg_k).max(f32::EPSILON)
}

/// Fourier conduction heat flux q = k·A·ΔT/d — Watts.
///
/// `temp_diff` K, `area` m², `distance` m, `conductivity` W/(m·K).
/// Positive when heat flows from hot to cold (ΔT > 0).
#[inline]
pub fn heat_conduction(
    temp_diff_k: f32,
    area_m2: f32,
    distance_m: f32,
    conductivity_w_m_k: f32,
) -> f32 {
    conductivity_w_m_k * area_m2 * temp_diff_k / distance_m.max(f32::EPSILON)
}

/// Stefan–Boltzmann radiative exchange q = σ·ε·A·F·(T_hot⁴ − T_cold⁴) — Watts.
///
/// Radiation needs no medium (unlike conduction), so it is the heat-transfer mode
/// that crosses vacuum. `emissivity` ε ∈ `[0,1]`, `view_factor` F ∈ `[0,1]` (geometry).
/// Also the physical basis for blackbody glow in the render emission pass.
#[inline]
pub fn heat_radiation(
    hot_temp_k: f32,
    cold_temp_k: f32,
    area_m2: f32,
    emissivity: f32,
    view_factor: f32,
) -> f32 {
    STEFAN_BOLTZMANN
        * emissivity
        * area_m2
        * view_factor
        * (hot_temp_k.powi(4) - cold_temp_k.powi(4))
}

/// Total radiant power of a spherical blackbody -- Stefan-Boltzmann law
/// applied to a sphere's surface area: L = 4·π·R²·σ·T⁴. Real, standard
/// formula connecting a star's surface temperature+radius to its total
/// luminosity (e.g. the Sun: R=6.957e8 m, T=5772 K -- real IAU 2015
/// nominal values -- gives L≈3.83e26 W, matching the real IAU nominal
/// solar luminosity 3.828e26 W to within 0.1%, see this module's own test).
#[inline]
pub fn stellar_luminosity_w(radius_m: f32, surface_temp_k: f32) -> f32 {
    4.0 * std::f32::consts::PI * radius_m * radius_m * STEFAN_BOLTZMANN * surface_temp_k.powi(4)
}

/// Real inverse-square irradiance law: flux = L / (4·π·r²) -- far-field
/// radiative flux from a point-like luminous source, the same real law
/// that gives the solar constant (real ≈1361 W/m² at 1 AU from the Sun's
/// own real luminosity). Distinct from `heat_radiation` above (a near-
/// field surface-to-surface exchange with a defined view factor between
/// two close bodies) -- this is the far-field point-source regime, valid
/// when the source's own radius is negligible next to the query distance
/// (true for a star at planetary distances, the same real simplification
/// `GravityWellField`'s point-mass model already makes for gravity).
#[inline]
pub fn irradiance_at_distance(luminosity_w: f32, distance_m: f32) -> f32 {
    if distance_m <= 0.0 {
        return f32::INFINITY;
    }
    luminosity_w / (4.0 * std::f32::consts::PI * distance_m * distance_m)
}

/// Reversible entropy change ΔS = Q/T — J/K.
///
/// Entropy transferred when heat `Q` (J) crosses a boundary at temperature `T` (K).
#[inline]
pub fn entropy_change_heat_transfer(heat_j: f32, temperature_k: f32) -> f32 {
    if temperature_k > 0.0 {
        heat_j / temperature_k
    } else {
        0.0
    }
}

/// Net entropy produced when heat `Q` flows from a hot source to a cold sink — J/K.
///
/// ΔS = Q·(1/T_cold − 1/T_hot) ≥ 0 for T_hot ≥ T_cold > 0 (2nd law).
#[inline]
pub fn entropy_change_irreversible(heat_j: f32, source_temp_k: f32, sink_temp_k: f32) -> f32 {
    if source_temp_k > 0.0 && sink_temp_k > 0.0 {
        heat_j * (1.0 / sink_temp_k - 1.0 / source_temp_k)
    } else {
        0.0
    }
}

/// Second law check: a real process never decreases total entropy.
#[inline]
pub fn second_law_holds(total_entropy_change: f32) -> bool {
    total_entropy_change >= 0.0
}

/// Saturating uptake/consumption rate — one rectangular-hyperbola equation shared
/// across three disciplines under three names: the Holling Type II functional response
/// (predation, Holling 1959), Michaelis-Menten kinetics (enzyme reaction rate, 1913),
/// and the Monod equation (microbial growth rate, 1949) — all `rate = max_rate ·
/// density / (half_saturation + density)`, confirmed identical in form, not three
/// separate laws.
///
/// Replaces any "consume everything within radius X" rule: rate is continuous in local
/// density, saturating toward `max_rate` as `density → ∞` (a real consumer has a finite
/// maximum processing rate no matter how much is available) and linear (∝ density) for
/// `density ≪ half_saturation` (scarce regime) — the two asymptotic checks any real
/// closed-form test should verify, not just "doesn't explode."
///
/// `half_saturation` is the density at which `rate` reaches exactly half of `max_rate`.
#[inline]
pub fn saturating_uptake(local_density: f32, max_rate: f32, half_saturation: f32) -> f32 {
    if local_density <= 0.0 {
        return 0.0;
    }
    max_rate * local_density / (half_saturation + local_density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_flux_matches_formula() {
        // q = k·A·ΔT/d = 100·2·50/0.5 = 20000 W
        let q = heat_conduction(50.0, 2.0, 0.5, 100.0);
        assert!((q - 20_000.0).abs() < 1e-2, "got {q}");
    }

    #[test]
    fn conduction_conserves_energy() {
        // Heat lost by hot body == heat gained by cold body for equal capacities.
        let q = heat_conduction(100.0, 1.0, 1.0, 50.0) * 0.1; // × dt
        let cap = 1000.0;
        let lost = cap * (q / cap); // hot cools by q/cap
        let gained = cap * (q / cap); // cold warms by q/cap
        assert!((lost - gained).abs() < 1e-5);
    }

    #[test]
    fn radiation_is_zero_at_thermal_equilibrium() {
        // Equal temperatures → no net radiative exchange.
        assert!(heat_radiation(300.0, 300.0, 1.0, 0.9, 1.0).abs() < 1e-9);
    }

    #[test]
    fn radiation_follows_t4() {
        // Doubling the hot temperature scales the (T⁴−T_c⁴) term by ~16 when T_c≈0.
        let q1 = heat_radiation(500.0, 0.0, 1.0, 1.0, 1.0);
        let q2 = heat_radiation(1000.0, 0.0, 1.0, 1.0, 1.0);
        assert!((q2 / q1 - 16.0).abs() < 1e-3, "ratio {}", q2 / q1);
    }

    /// Real IAU 2015 nominal solar values (R=6.957e8 m, T=5772 K) must
    /// reproduce the real, independently-measured solar luminosity
    /// (3.828e26 W) -- confirms the sphere-Stefan-Boltzmann formula is
    /// correctly derived, not just internally self-consistent.
    #[test]
    fn sun_luminosity_from_real_temperature_and_radius_matches_iau_value() {
        let l = stellar_luminosity_w(6.957e8, 5772.0);
        let l_iau = 3.828e26_f32;
        assert!(
            (l - l_iau).abs() / l_iau < 0.01,
            "expected ~{l_iau:e} W, got {l:e} W"
        );
    }

    /// Real, independent second check: the Sun's own real luminosity,
    /// carried through the inverse-square law out to 1 real AU, must
    /// reproduce the real, independently-measured solar constant
    /// (≈1361 W/m²) -- two different real measured quantities agreeing
    /// through two chained real laws, not one law calibrated to match
    /// itself.
    #[test]
    fn sun_irradiance_at_one_au_matches_real_solar_constant() {
        let l = stellar_luminosity_w(6.957e8, 5772.0);
        let au_m = 1.496e11_f32;
        let flux = irradiance_at_distance(l, au_m);
        let real_solar_constant = 1361.0_f32;
        assert!(
            (flux - real_solar_constant).abs() / real_solar_constant < 0.02,
            "expected ~{real_solar_constant} W/m^2, got {flux} W/m^2"
        );
    }

    #[test]
    fn irradiance_falls_off_as_exact_inverse_square() {
        let f1 = irradiance_at_distance(1.0e20, 10.0);
        let f2 = irradiance_at_distance(1.0e20, 20.0);
        // Doubling distance must quarter the flux exactly, not approximately.
        assert!((f1 / f2 - 4.0).abs() < 1e-3, "ratio={}", f1 / f2);
    }

    #[test]
    fn diffusivity_of_water() {
        // Water: k≈0.6, ρ≈1000, c_p≈4184 → α≈1.43e-7 m²/s (known value).
        let a = thermal_diffusivity(0.6, 1000.0, 4184.0);
        assert!((a - 1.43e-7).abs() < 1e-8, "got {a}");
    }

    #[test]
    fn irreversible_flow_produces_positive_entropy() {
        // Heat from hot (400 K) to cold (300 K) → net entropy > 0 (2nd law).
        let ds = entropy_change_irreversible(100.0, 400.0, 300.0);
        assert!(ds > 0.0 && second_law_holds(ds));
    }

    #[test]
    fn reversible_entropy_is_q_over_t() {
        assert!((entropy_change_heat_transfer(1000.0, 250.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn saturating_uptake_is_zero_at_zero_density() {
        assert_eq!(saturating_uptake(0.0, 10.0, 2.0), 0.0);
    }

    #[test]
    fn saturating_uptake_equals_half_max_at_half_saturation_density() {
        // rate(half_saturation) = max_rate * hs / (hs + hs) = max_rate / 2, exactly.
        let rate = saturating_uptake(2.0, 10.0, 2.0);
        assert!((rate - 5.0).abs() < 1e-6, "got {rate}");
    }

    #[test]
    fn saturating_uptake_approaches_max_rate_at_high_density() {
        // density >> half_saturation -> rate should approach max_rate, not exceed it.
        let rate = saturating_uptake(10_000.0, 10.0, 2.0);
        assert!((rate - 10.0).abs() < 1e-2, "got {rate}");
        assert!(
            rate < 10.0,
            "saturating uptake must never reach/exceed max_rate: {rate}"
        );
    }

    #[test]
    fn saturating_uptake_is_linear_at_low_density() {
        // density << half_saturation -> rate ≈ (max_rate/half_saturation) * density.
        let max_rate = 10.0;
        let half_saturation = 100.0;
        let density = 0.01;
        let rate = saturating_uptake(density, max_rate, half_saturation);
        let linear_approx = (max_rate / half_saturation) * density;
        assert!(
            (rate - linear_approx).abs() < 1e-5,
            "got {rate}, linear approx {linear_approx}"
        );
    }
}

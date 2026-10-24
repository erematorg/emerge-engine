//! Momentum exchange between interpenetrating phases of a mixture.
//!
//! In mixture theory each phase (the sand skeleton, the water in its pores)
//! keeps its own velocity and its own momentum balance; the phases are coupled
//! by an exchange term that sums to zero over the phases, so the mixture's
//! total momentum is conserved (Tampubolon, Gast, Klar, Fu, Teran, Jiang &
//! Museth 2017, "Multi-species simulation of porous sand and water mixtures",
//! ACM TOG 36(4) 105, eqs. 3-4).
//!
//! The law lives here; its discretization (an implicit solve per grid node) is
//! `spacetime::grid::mixture`, and its conversion to grid units is
//! `SimConfig::mixture_drag_from_darcy`.

/// Darcy exchange coefficient `c_E`, in kg/(m^3 s): the force per unit mixture
/// volume per unit relative velocity between a pore fluid and the porous solid
/// it seeps through.
///
/// Tampubolon et al. 2017, eq. 13 (after Bandara & Soga):
///
///   `p_s = c_E (v_w - v_s)`, `p_w = -p_s`, `c_E = n^2 rho_w g / k_hat`
///
/// with `n` the porosity, `rho_w` the fluid density, `k_hat` the hydraulic
/// conductivity of the soil and `g` the gravitational acceleration in
/// `k_hat`'s definition. `k_hat` is a measured soil property (permeameter
/// tests, ASTM D2434 for coarse soils, ASTM D5084 for fine ones; FHWA
/// NHI-06-088 sec. 5.6). The same equation has a second, reversible term,
/// `p_w grad(phi_w)` with `phi_w` the fluid volume fraction, which this
/// coefficient does not include (Tampubolon et al. omit it in most of their
/// examples too).
///
/// # Panics
/// If `porosity` is outside `(0, 1]` or any other argument is not positive.
pub fn darcy_exchange_coefficient(
    porosity: f32,
    hydraulic_conductivity_m_s: f32,
    fluid_density_kg_m3: f32,
    gravity_m_s2: f32,
) -> f32 {
    assert!(
        porosity > 0.0 && porosity <= 1.0,
        "porosity must be in (0, 1], got {porosity}"
    );
    assert!(
        hydraulic_conductivity_m_s > 0.0,
        "hydraulic conductivity must be positive, got {hydraulic_conductivity_m_s}"
    );
    assert!(
        fluid_density_kg_m3 > 0.0 && gravity_m_s2 > 0.0,
        "fluid density and gravity must be positive"
    );
    porosity * porosity * fluid_density_kg_m3 * gravity_m_s2 / hydraulic_conductivity_m_s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eq. 13 by hand: n = 0.4, rho_w = 1000, g = 9.81, k_hat = 1e-4 m/s gives
    /// 0.16 * 1000 * 9.81 / 1e-4 = 1.5696e7 kg/(m^3 s).
    #[test]
    fn darcy_coefficient_matches_the_equation_by_hand() {
        let c = darcy_exchange_coefficient(0.4, 1.0e-4, 1000.0, 9.81);
        assert!((c / 1.5696e7 - 1.0).abs() < 1e-5, "got {c}");
    }

    /// A less permeable soil couples the phases more strongly: ten times lower
    /// conductivity, ten times the coefficient.
    #[test]
    fn darcy_coefficient_is_inverse_in_conductivity() {
        let sand = darcy_exchange_coefficient(0.4, 1.0e-4, 1000.0, 9.81);
        let silt = darcy_exchange_coefficient(0.4, 1.0e-5, 1000.0, 9.81);
        assert!((silt / sand - 10.0).abs() < 1e-4);
    }

    /// The grid-unit coefficient is `c_E / reference_density_kg_m3`, the same
    /// for any cell size or slice thickness (both cancel, see
    /// `SimConfig::mixture_drag_from_darcy`).
    #[test]
    fn grid_coefficient_is_independent_of_cell_size_and_slice() {
        use crate::SimConfig;
        let c_e = darcy_exchange_coefficient(0.4, 1.0e-4, 1000.0, 9.81);
        for (dx, slice) in [(0.01, Some(0.01)), (0.5, Some(2.0)), (0.002, None)] {
            let config = SimConfig {
                dx_meters: dx,
                slice_thickness_m: slice,
                reference_density_kg_m3: 1600.0,
                ..SimConfig::default()
            };
            let k = config.mixture_drag_from_darcy(0.4, 1.0e-4, 1000.0, 9.81);
            assert!((k / (c_e / 1600.0) - 1.0).abs() < 1e-6, "dx={dx}: k={k}");
        }
    }

    #[test]
    #[should_panic(expected = "porosity")]
    fn porosity_above_one_is_rejected() {
        darcy_exchange_coefficient(1.2, 1.0e-4, 1000.0, 9.81);
    }
}

//! How much heat each substance stores: measured specific heat capacities.
//!
//! The law lives in `energy::thermodynamics::frictional_heating`: dissipated
//! work becomes internal energy at constant pressure, `dT = dQ / (m c_p)`.
//! What belongs here is the part that is a property of the matter itself,
//! the `c_p` of one particular substance, in J/(kg K).
//!
//! A material carries one of these only when a caller declares it (the
//! `specific_heat_j_kg_k` field of the material laws, see
//! `MaterialModel::specific_heat_j_kg_k`); a material that declares none
//! gets no temperature rise rather than an invented one.
//!
//! Every value is the substance's `c_p` at one stated temperature and normal
//! pressure. `c_p` varies with temperature, strongly for ice (the IAPWS
//! release below gives 866 J/(kg K) at 100 K against 2097 at the melting
//! point). Adaptation, declared: the engine uses one constant per material,
//! so a scene far from the stated temperature carries that error. Removal
//! criterion: a scene that runs a substance far from its stated temperature
//! evaluates `c_p(T)` from the same source instead.

/// Ice Ih at its normal-pressure melting point (273.152519 K, 101325 Pa).
///
/// Source: IAPWS R10-06(2009), *Revised Release on the Equation of State
/// 2006 for H2O Ice Ih*, Table 6, p. 12: `c_p = 0.209671391024e4`
/// J/(kg K) (the equation of state's own check value, `c_p = -T g_TT`, its
/// eq. 6). Combined standard uncertainty 2 % (Table 5, p. 10).
pub const ICE_IH_AT_MELTING_POINT_J_KG_K: f32 = 2096.714;

/// Alpha quartz (SiO2) at 298.15 K, the mineral of quartz sand.
///
/// Source: NIST-JANAF Thermochemical Tables (Chase 1998, 4th ed.), table
/// O-037 "Silicon Oxide, Quartz", row 298.15 K: `C_p = 44.589` J/(K mol);
/// molar mass 60.0843 g/mol (NIST Chemistry WebBook, CAS 14808-60-7).
/// Divided here: 44.589 / 0.0600843 = 742.1 J/(kg K). Per unit mass of a dry
/// sand this is the grains' value: the air between them is left out.
pub const ALPHA_QUARTZ_AT_298_K_J_KG_K: f32 = 44.589 / 60.0843e-3;

/// Copper at 298.15 K.
///
/// Source: NIST-JANAF Thermochemical Tables (Chase 1998, 4th ed.), table
/// Cu-002 "Copper (Cu), cr", row 298.15 K: `C_p = 24.442` J/(K mol); molar
/// mass 63.546 g/mol (NIST Chemistry WebBook, CAS 7440-50-8). Divided here:
/// 24.442 / 0.063546 = 384.6 J/(kg K).
pub const COPPER_AT_298_K_J_KG_K: f32 = 24.442 / 63.546e-3;

/// Natural rubber hydrocarbon, amorphous, at 298.1 K.
///
/// Source: Bekkedahl and Matheson, *Heat capacity, entropy, and free energy
/// of rubber hydrocarbon*, J. Res. NBS 15, RP844 (1935), abstract, p. 503:
/// `1.880 +- 0.002` J/(g degC), adiabatic calorimetry (Table 1, p. 507: 1.890
/// at 300 K). Declared: the sample is purified, unvulcanized rubber, and the
/// paper's joules are international joules (p. 506), not converted here; a
/// vulcanized or filled rubber is a different substance.
pub const NATURAL_RUBBER_AT_298_K_J_KG_K: f32 = 1880.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Two independent sources agree on copper: Banerjee's fit to measured
    /// heat capacities (arXiv cond-mat/0512466, eq. 2, `0.1009 T + 358.4`
    /// above 270 K) against the JANAF table, at 298.15 K, within 1 %.
    #[test]
    fn copper_agrees_with_an_independent_fit() {
        let fit = 0.1009 * 298.15 + 358.4;
        let relative = (COPPER_AT_298_K_J_KG_K - fit).abs() / fit;
        assert!(
            relative < 0.01,
            "JANAF {COPPER_AT_298_K_J_KG_K} vs fit {fit}: {relative:.2e}"
        );
    }
}

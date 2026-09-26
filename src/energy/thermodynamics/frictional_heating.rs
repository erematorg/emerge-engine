//! Dissipated work becoming temperature: the first law at a rubbing contact.
//!
//! A Coulomb contact that removes kinetic energy has to put it somewhere.
//! Until this existed the engine simply deleted it, which is a first-law
//! violation committed at every frictional wall, every substep. The energy
//! is now measured where it is lost (`forces::boundary::apply_coulomb_wall`
//! reports it, `Grid` records it per node) and converted here.
//!
//! The conversion is one line of thermodynamics and no modelling choices:
//! all the dissipated work goes into the internal energy of the matter that
//! did the rubbing, at constant pressure, so `dU = dQ` and `dT = dQ /
//! (m c_p)`. Per unit mass that is `dT = de / c_p`.
//!
//! What this does NOT claim: that the wall also heats up (it is a boundary
//! condition, not matter with a temperature), that the heat then conducts
//! away on its own (`ThermalDiffusion` does that, if a scene runs it), or
//! that the normal part of an impact is heat -- see `apply_coulomb_wall`'s
//! own doc for why only the tangential part is reported.

/// Temperature rise, in kelvin, from a specific dissipated energy.
///
/// `specific_energy_j_kg` is energy per unit mass, J/kg.
/// `specific_heat_j_kg_k` is the material's `c_p`, J/(kg*K).
///
/// Returns 0 for a material that has not declared a heat capacity: an
/// unknown `c_p` means the temperature rise is unknown, and inventing one
/// would be worse than reporting none. The energy stays recorded on the
/// grid either way, so nothing is silently lost.
pub fn temperature_rise_from_dissipation(
    specific_energy_j_kg: f32,
    specific_heat_j_kg_k: f32,
) -> f32 {
    if !specific_energy_j_kg.is_finite()
        || specific_energy_j_kg <= 0.0
        || !specific_heat_j_kg_k.is_finite()
        || specific_heat_j_kg_k <= 0.0
    {
        return 0.0;
    }
    specific_energy_j_kg / specific_heat_j_kg_k
}

/// Converts a specific energy expressed in the solver's grid units
/// (velocity-squared, where velocity is cells per second) into SI J/kg.
///
/// One cell is `dx_meters` across, so a grid velocity is `dx_meters` times
/// an SI velocity, and a velocity squared is `dx_meters^2` times an SI
/// specific energy. Stated as its own function because getting this factor
/// wrong is silent: the temperature would simply be off by a constant and
/// still look plausible.
pub fn specific_energy_grid_to_si(specific_energy_grid: f32, dx_meters: f32) -> f32 {
    specific_energy_grid * dx_meters * dx_meters
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textbook number, worked by hand: a block of iron
    /// (`c_p = 449 J/(kg*K)`) sliding at 10 m/s and brought to rest by
    /// friction dissipates `0.5 * 10^2 = 50 J/kg`, so it warms by
    /// `50 / 449 = 0.111 K`. Barely perceptible, which is itself the point:
    /// braking a real object heats it a little, not a lot.
    #[test]
    fn sliding_iron_block_warms_by_the_textbook_amount() {
        let specific_energy = 0.5 * 10.0f32 * 10.0;
        let rise = temperature_rise_from_dissipation(specific_energy, 449.0);
        assert!(
            (rise - 0.1114).abs() < 1.0e-3,
            "expected ~0.111 K, got {rise} K"
        );
    }

    /// Water's high heat capacity is why it barely warms: the same 50 J/kg
    /// raises it only 0.012 K, four times less than iron.
    #[test]
    fn water_warms_four_times_less_than_iron_for_the_same_work() {
        let specific_energy = 50.0;
        let iron = temperature_rise_from_dissipation(specific_energy, 449.0);
        let water = temperature_rise_from_dissipation(specific_energy, 4182.0);
        let ratio = iron / water;
        assert!(
            (ratio - 4182.0 / 449.0).abs() < 1.0e-3,
            "the ratio of rises must be the inverse ratio of heat capacities, got {ratio}"
        );
    }

    /// An undeclared heat capacity yields no temperature change rather than
    /// a guessed one.
    #[test]
    fn unknown_heat_capacity_produces_no_rise() {
        assert_eq!(temperature_rise_from_dissipation(50.0, 0.0), 0.0);
        assert_eq!(temperature_rise_from_dissipation(50.0, f32::NAN), 0.0);
    }

    /// The unit bridge: at 1 cm cells, a grid speed of 100 cells/s is 1 m/s,
    /// so a grid specific energy of `0.5 * 100^2 = 5000` must come out as
    /// `0.5 * 1^2 = 0.5 J/kg`.
    #[test]
    fn grid_units_convert_to_si_by_dx_squared() {
        let si = specific_energy_grid_to_si(0.5 * 100.0 * 100.0, 0.01);
        assert!((si - 0.5).abs() < 1.0e-6, "got {si} J/kg, expected 0.5");
    }
}

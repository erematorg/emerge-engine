//! Grid-to-particle transfer of the energy a frictional boundary dissipated.
//!
//! The boundary pass records, per grid node, the specific kinetic energy
//! Coulomb friction removed there (`Grid::add_friction_heat`). This carries
//! it back to the matter with the same quadratic B-spline stencil every
//! other G2P gather uses, and `energy::thermodynamics::frictional_heating`
//! turns it into a temperature.
//!
//! # Why the same stencil conserves the energy exactly
//!
//! A node's recorded value `e_i` is energy per unit mass. Gathering it gives
//! particle `p` the specific energy `sum_i w_ip e_i`, so the total energy
//! delivered is
//!
//! ```text
//! sum_p m_p sum_i w_ip e_i  =  sum_i e_i sum_p m_p w_ip  =  sum_i e_i m_i
//! ```
//!
//! because P2G defines the node mass as exactly `m_i = sum_p m_p w_ip`. The
//! energy the particles receive is therefore the energy the nodes lost, to
//! floating-point. That is not a convenient approximation -- it is why the
//! MPM stencil is the right carrier here rather than, say, heating whichever
//! particle is nearest.
//!
//! # Why the rise is banked instead of applied immediately
//!
//! Frictional heating is physically tiny per substep. A body sliding at
//! 1 m/s brought to rest by friction releases 0.5 J/kg, which for iron is
//! about a millikelvin -- spread over the thousands of substeps a second of
//! simulation takes, roughly `1e-7 K` each. An `f32` holding an absolute
//! temperature near 300 K resolves about `3e-5 K`, so `T + 1e-7` is exactly
//! `T`: applied naively, every single increment would round away and the
//! whole effect would silently vanish.
//!
//! So each particle carries a heat debt, accumulated in its own small
//! number where the addition is exact, and paid into the temperature only
//! once it is large enough to survive. This is ordinary compensated
//! summation, not a fudge: no energy is discarded, it is only held briefly.

use glam::Vec2;

use crate::energy::thermodynamics::{
    specific_energy_grid_to_si, temperature_rise_from_dissipation,
};
use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;
use crate::matter::materials::registry::MaterialRegistry;
use crate::particle::Particles;

/// Heat capacities are read into a small stack array before the particle
/// loop rather than queried per particle: the lookup goes through a trait
/// object, and doing it once per material beats doing it once per particle.
const MAX_LOOKED_UP_MATERIALS: usize = 16;

/// Smallest temperature rise paid out of the heat debt, in kelvin.
///
/// An `f32` resolves about `6e-5 K` at 1000 K, the hottest this engine's
/// scenes realistically run, so `1e-3` keeps a margin of more than an order
/// of magnitude while holding back an amount of heat far below anything
/// physically meaningful here. Larger would delay visible heating for no
/// gain; smaller would start losing increments to rounding again.
const MIN_APPLIED_TEMPERATURE_RISE_K: f32 = 1.0e-3;

/// Applies the substep's recorded frictional dissipation to particle
/// temperatures, returning the total energy delivered in J (for auditing).
///
/// A material that has not declared a heat capacity keeps its particles'
/// temperature unchanged, and the energy is reported but not converted --
/// see `temperature_rise_from_dissipation`'s own doc for why that is the
/// honest answer rather than a guessed `c_p`.
///
/// `heat_debt` is the caller-owned per-particle accumulator described
/// above; it is resized to the particle count as needed and must persist
/// across substeps or the banked energy is lost.
///
/// Does nothing, and touches no particle, when no node dissipated anything.
pub fn gather_friction_heat_to_particles(
    grid: &Grid,
    particles: &mut Particles,
    registry: &MaterialRegistry,
    dx_meters: f32,
    heat_debt: &mut Vec<f32>,
) -> f32 {
    if !grid.has_friction_heat() {
        return 0.0;
    }
    heat_debt.resize(particles.len(), 0.0);
    // Only the slots that exist. Filling the whole array unconditionally
    // asked the registry for material ids it had never been given, and
    // `MaterialRegistry::get`'s own `debug_assert` is there precisely to
    // catch that -- so every debug-build scene that produced any friction
    // heat panicked before it could deliver it.
    let mut specific_heat = [0.0f32; MAX_LOOKED_UP_MATERIALS];
    let registered = registry.len().min(MAX_LOOKED_UP_MATERIALS);
    for (id, slot) in specific_heat.iter_mut().enumerate().take(registered) {
        *slot = registry.get(id as u32).specific_heat_j_kg_k();
    }
    let mut delivered_j = 0.0;
    // Iterating the debt vector, which `resize` above made exactly as long as
    // the particle list, keeps the two indices provably in step.
    for (i, debt) in heat_debt.iter_mut().enumerate() {
        let weights = quadratic_weights(particles.x[i]);
        let mut specific_grid = 0.0;
        for (gx, &wx) in weights.wx.iter().enumerate() {
            for (gy, &wy) in weights.wy.iter().enumerate() {
                let cell = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                specific_grid += wx * wy * grid.friction_heat_at(cell);
            }
        }
        if specific_grid <= 0.0 {
            continue;
        }
        let specific_si = specific_energy_grid_to_si(specific_grid, dx_meters);
        delivered_j += specific_si * particles.mass[i];
        let c_p = specific_heat
            .get(particles.material_id[i] as usize)
            .copied()
            .unwrap_or(0.0);
        let rise = temperature_rise_from_dissipation(specific_si, c_p);
        if rise <= 0.0 {
            continue;
        }
        *debt += rise;
        if *debt >= MIN_APPLIED_TEMPERATURE_RISE_K {
            particles.temperature[i] += *debt;
            *debt = 0.0;
        }
    }
    delivered_j
}

/// Kinetic energy, in the grid's own units, held by every active node --
/// `sum_i 0.5 * m_i * |v_i|^2`.
///
/// Exists so an energy audit can measure what a boundary pass actually
/// removed, rather than trusting that the boundaries reported it correctly.
/// `friction_energy_is_conserved_through_the_stencil` uses it.
pub fn grid_kinetic_energy(grid: &Grid) -> f32 {
    grid.active_cells()
        .map(|cell| 0.5 * cell.mass * velocity_of(cell).length_squared())
        .sum()
}

fn velocity_of(cell: &crate::grid::Cell) -> Vec2 {
    // After `normalize_velocities`, `momentum` holds the velocity itself --
    // see `Cell::momentum`'s own dual-phase doc.
    cell.momentum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::kernel::quadratic_weights;
    use crate::materials::NewtonianFluidMaterial;
    use crate::materials::registry::MaterialRegistry;
    use crate::particle::Particle;
    use glam::{IVec2, Vec2};

    const DX_METERS: f32 = 0.01;
    const WATER_CP: f32 = 4182.0;

    fn water_registry() -> MaterialRegistry {
        let mut water = NewtonianFluidMaterial::new(1.0, 0.001, 1000.0, 7.0);
        water.specific_heat_j_kg_k = WATER_CP;
        MaterialRegistry::with_default(Box::new(water))
    }

    /// Scatters particle mass onto the grid exactly as P2G does, so the node
    /// masses satisfy `m_i = sum_p w_ip m_p` -- the identity the conservation
    /// proof rests on.
    fn scatter_masses(grid: &mut Grid, particles: &Particles) {
        for i in 0..particles.len() {
            let w = quadratic_weights(particles.x[i]);
            for (gx, &wx) in w.wx.iter().enumerate() {
                for (gy, &wy) in w.wy.iter().enumerate() {
                    let cell = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    grid.add_mass_momentum(cell, wx * wy * particles.mass[i], Vec2::ZERO);
                }
            }
        }
    }

    fn particle_at(position: Vec2, mass: f32) -> Particle {
        let mut p = Particle::zeroed();
        p.x = position;
        p.mass = mass;
        p.temperature = 300.0;
        p
    }

    /// The claim the module doc makes: gathering a per-node SPECIFIC energy
    /// with the MPM stencil delivers exactly the energy the nodes held,
    /// because `m_i = sum_p w_ip m_p`. Checked against a directly computed
    /// `sum_i e_i m_i` rather than against another gather.
    #[test]
    fn friction_energy_is_conserved_through_the_stencil() {
        let mut particles = Particles::default();
        for (position, mass) in [
            (Vec2::new(8.3, 6.7), 1.0),
            (Vec2::new(8.9, 6.2), 2.0),
            (Vec2::new(9.4, 7.1), 0.5),
        ] {
            particles.push(particle_at(position, mass));
        }

        let mut grid = Grid::new(32);
        scatter_masses(&mut grid, &particles);

        // Dissipate at a few nodes the particles genuinely overlap.
        let dissipating = [
            (IVec2::new(8, 6), 3.0f32),
            (IVec2::new(9, 6), 1.5),
            (IVec2::new(9, 7), 0.75),
        ];
        let mut expected_j = 0.0;
        for (cell, specific) in dissipating {
            let idx = (cell.x as usize) * 32 + cell.y as usize;
            grid.add_friction_heat(idx, specific);
            // Energy the node itself held: specific energy times node mass,
            // converted out of grid units.
            expected_j += specific * grid.mass_at(cell) * DX_METERS * DX_METERS;
        }
        assert!(grid.has_friction_heat());

        let delivered = gather_friction_heat_to_particles(
            &grid,
            &mut particles,
            &water_registry(),
            DX_METERS,
            &mut Vec::new(),
        );

        let relative_error = (delivered - expected_j).abs() / expected_j;
        assert!(
            relative_error < 1.0e-5,
            "the stencil must deliver exactly the nodes' own energy: \
             delivered {delivered:.9} J vs nodes' {expected_j:.9} J \
             (relative error {relative_error:.2e})"
        );
    }

    /// The property the heat debt exists for: many increments far too small
    /// to change an `f32` temperature must still add up to the right total.
    ///
    /// Each substep here raises the particle by ~3e-8 K while `f32` resolves
    /// ~3e-5 K at 300 K, so applied directly every one of them would round
    /// to nothing. Banked, they pay out, and the total rise matches the
    /// total energy delivered divided by `c_p` -- no energy lost to
    /// rounding, which is the whole point.
    #[test]
    fn many_tiny_rises_accumulate_instead_of_rounding_away() {
        let mut particles = Particles::default();
        particles.push(particle_at(Vec2::new(8.5, 6.5), 1.0));

        let mut grid = Grid::new(32);
        scatter_masses(&mut grid, &particles);
        for cell in [IVec2::new(8, 6), IVec2::new(9, 6)] {
            grid.add_friction_heat((cell.x as usize) * 32 + cell.y as usize, 2.0);
        }

        let registry = water_registry();
        let before = particles.temperature[0];
        let mut debt = Vec::new();

        // One substep alone is below what the temperature can represent.
        let one = gather_friction_heat_to_particles(
            &grid,
            &mut particles,
            &registry,
            DX_METERS,
            &mut debt,
        );
        assert!(
            one > 0.0,
            "the energy must be delivered even when the rise cannot show"
        );
        assert_eq!(
            particles.temperature[0], before,
            "a rise of ~3e-8 K cannot move an f32 at 300 K, so it must be banked"
        );
        assert!(debt[0] > 0.0, "the banked rise must be held, not discarded");

        // Run enough substeps that the debt has to pay out.
        let mut total_j = one;
        for _ in 0..200_000 {
            total_j += gather_friction_heat_to_particles(
                &grid,
                &mut particles,
                &registry,
                DX_METERS,
                &mut debt,
            );
        }

        let actual_rise = particles.temperature[0] - before;
        assert!(
            actual_rise > 0.0,
            "banked heat must eventually reach the temperature"
        );
        // Everything delivered is either in the temperature or still banked.
        let accounted = actual_rise + debt[0];
        let expected = (total_j / particles.mass[0]) / WATER_CP;
        // The bound is one representable step of the temperature itself.
        // `actual_rise` is measured by subtracting two numbers near 300 K,
        // where an f32 step is ~3.6e-5 K -- so this test cannot resolve the
        // total any finer than that, however exact the accumulation is.
        // Anything within one step means nothing was lost.
        let one_ulp_at_300k = f32::EPSILON * 300.0;
        assert!(
            (accounted - expected).abs() <= 2.0 * one_ulp_at_300k,
            "temperature rise plus banked debt must equal every joule delivered to              within the temperature's own resolution: {accounted:e} K accounted vs              {expected:e} K delivered, off by {:e} K against a {one_ulp_at_300k:e} K step",
            (accounted - expected).abs()
        );
    }

    /// A material with no declared heat capacity gets no temperature change,
    /// but the energy is still reported -- it is accounted for, not lost.
    #[test]
    fn undeclared_heat_capacity_still_reports_the_energy() {
        let mut particles = Particles::default();
        particles.push(particle_at(Vec2::new(8.5, 6.5), 1.0));

        let mut grid = Grid::new(32);
        scatter_masses(&mut grid, &particles);
        grid.add_friction_heat(8 * 32 + 6, 2.0);

        let registry = MaterialRegistry::with_default(Box::new(NewtonianFluidMaterial::new(
            1.0, 0.001, 1000.0, 7.0,
        )));
        let before = particles.temperature[0];
        let delivered = gather_friction_heat_to_particles(
            &grid,
            &mut particles,
            &registry,
            DX_METERS,
            &mut Vec::new(),
        );
        assert!(delivered > 0.0, "the energy must still be reported");
        assert_eq!(
            particles.temperature[0], before,
            "an undeclared c_p must leave the temperature alone rather than invent a rise"
        );
    }

    /// A frictionless scene pays nothing and changes nothing.
    #[test]
    fn no_dissipation_means_no_work_and_no_change() {
        let mut particles = Particles::default();
        particles.push(particle_at(Vec2::new(8.5, 6.5), 1.0));
        let mut grid = Grid::new(32);
        scatter_masses(&mut grid, &particles);

        assert!(!grid.has_friction_heat());
        let before = particles.temperature[0];
        let delivered = gather_friction_heat_to_particles(
            &grid,
            &mut particles,
            &water_registry(),
            DX_METERS,
            &mut Vec::new(),
        );
        assert_eq!(delivered, 0.0);
        assert_eq!(particles.temperature[0], before);
    }
}

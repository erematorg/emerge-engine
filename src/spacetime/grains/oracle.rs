//! Packing-fraction oracle signal from Hybrid Grains (Yue, Smith, Chen,
//! Chantharayukhonthorn, Kamrin & Grinspun, ACM TOG 2018): the criterion
//! their adaptive discrete/continuum coupling uses to decide where
//! grain-scale physics is needed. Their oracle computes the packing fraction
//! of an already-discrete grain population on a background grid; here the
//! same signal is read from a continuum region's grid mass, to find free
//! surfaces and thin flowing layers (the paper's triggers: "low pressure",
//! "thin flows", "finite-size effects"). That is where plain Drucker-Prager
//! fails: as pressure goes to 0 the Coulomb yield stress vanishes whatever
//! the friction angle.
//!
//! Signal only: it flags cells where discrete treatment would help.
//! Creating grains from continuum is `Simulation`'s enrichment path
//! (`solver/particles.rs`); homogenization back to continuum is not
//! implemented.

use glam::IVec2;

use crate::grid::Grid;

/// Reference mass for a fully packed MPM cell: ~4 particles per cell, the
/// usual minimum for MLS-MPM stability and quadrature accuracy (Jiang et al.
/// 2016).
const DENSE_PARTICLES_PER_CELL: f32 = 4.0;

/// Reference ("fully packed") mass for one grid cell, given the scene's own
/// per-particle mass. `packing_fraction_at` divides a cell's real
/// accumulated mass by this to get a dimensionless density ratio.
pub fn reference_mass_per_cell(particle_mass: f32) -> f32 {
    particle_mass * DENSE_PARTICLES_PER_CELL
}

/// Packing fraction (accumulated grid mass / dense-reference mass) at
/// `cell_pos`, clamped to `[0.0, 1.0]`: kernel overlap and jitter can push a
/// cell slightly past the nominal reference.
pub fn packing_fraction_at(grid: &Grid, cell_pos: IVec2, reference_mass_per_cell: f32) -> f32 {
    if reference_mass_per_cell <= 0.0 {
        return 0.0;
    }
    (grid.mass_at(cell_pos) / reference_mass_per_cell).clamp(0.0, 1.0)
}

/// The real oracle check: does `cell_pos` look like a free surface or thin
/// flowing layer (real material present, but below the dense-packing
/// threshold), where this project's own investigation already root-caused
/// plain continuum Drucker-Prager as structurally unable to hold a real
/// repose angle? `threshold` is the paper's own `phi_crit` -- a scene-
/// tunable cutoff, not hardcoded (their own oracle treats it as a user
/// parameter too).
///
/// Empty space (packing fraction exactly 0 -- no material at all) does NOT
/// count as "needs discrete treatment": that's just empty space, not a real
/// thin-flow surface. The signal only fires where there IS real material,
/// just not densely packed.
pub fn needs_discrete_treatment(
    grid: &Grid,
    cell_pos: IVec2,
    reference_mass_per_cell: f32,
    threshold: f32,
) -> bool {
    let phi = packing_fraction_at(grid, cell_pos, reference_mass_per_cell);
    phi > 0.0 && phi < threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cell_has_zero_packing_fraction_and_is_not_flagged() {
        let grid = Grid::new(16);
        let reference = reference_mass_per_cell(1.0);
        assert_eq!(packing_fraction_at(&grid, IVec2::new(5, 5), reference), 0.0);
        assert!(!needs_discrete_treatment(
            &grid,
            IVec2::new(5, 5),
            reference,
            0.7
        ));
    }

    #[test]
    fn fully_dense_cell_has_packing_fraction_near_one_and_is_not_flagged() {
        let mut grid = Grid::new(16);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);
        grid.add_mass_momentum(IVec2::new(5, 5), reference, glam::Vec2::ZERO);
        let phi = packing_fraction_at(&grid, IVec2::new(5, 5), reference);
        assert!((phi - 1.0).abs() < 1e-5, "phi={phi}");
        assert!(!needs_discrete_treatment(
            &grid,
            IVec2::new(5, 5),
            reference,
            0.7
        ));
    }

    #[test]
    fn partially_packed_cell_below_threshold_is_flagged() {
        let mut grid = Grid::new(16);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);
        // Half-density -- a thin-flow/free-surface signature.
        grid.add_mass_momentum(IVec2::new(5, 5), reference * 0.5, glam::Vec2::ZERO);
        let phi = packing_fraction_at(&grid, IVec2::new(5, 5), reference);
        assert!((phi - 0.5).abs() < 1e-5, "phi={phi}");
        assert!(needs_discrete_treatment(
            &grid,
            IVec2::new(5, 5),
            reference,
            0.7
        ));
    }

    #[test]
    fn overshoot_mass_clamps_to_one_not_an_unbounded_ratio() {
        let mut grid = Grid::new(16);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);
        grid.add_mass_momentum(IVec2::new(5, 5), reference * 3.0, glam::Vec2::ZERO);
        let phi = packing_fraction_at(&grid, IVec2::new(5, 5), reference);
        assert_eq!(phi, 1.0);
    }

    #[test]
    fn a_real_sloped_pile_flags_its_surface_but_not_its_dense_interior() {
        // A small triangular pile of per-particle mass deposits (mimicking
        // P2G accumulation): the oracle must tell the dense interior (fully
        // surrounded) from the sloped surface layer (material present, but
        // thin and partly packed), the region where the repose angle is set.
        let mut grid = Grid::new(32);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);

        // A triangular pile: row 0 (bottom) is widest and fully dense: rows
        // above narrow and are only partially filled at their outer edges,
        // mimicking a sloped free surface.
        for row in 0..6i32 {
            let half_width = 6 - row;
            for col in -half_width..=half_width {
                let x = 16 + col;
                let y = 5 + row;
                // Interior of each row: full density. Outermost column of
                // each row: half density, standing in for the partial
                // occupancy a sloped surface has.
                let is_edge = col == half_width || col == -half_width;
                let mass = if is_edge { reference * 0.4 } else { reference };
                grid.add_mass_momentum(IVec2::new(x, y), mass, glam::Vec2::ZERO);
            }
        }

        // Deep interior, bottom-center: fully packed, must NOT be flagged.
        assert!(
            !needs_discrete_treatment(&grid, IVec2::new(16, 5), reference, 0.7),
            "dense pile interior was incorrectly flagged as needing discrete treatment"
        );
        // Sloped edge cell of row 4 (y=9, half_width=2 -> edge at col=+-2,
        // x=16+2=18): must be flagged.
        assert!(
            needs_discrete_treatment(&grid, IVec2::new(18, 9), reference, 0.7),
            "sloped surface cell was NOT flagged, oracle missed the real free surface"
        );
    }
}

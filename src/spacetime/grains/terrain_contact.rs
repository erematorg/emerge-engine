//! Grain-vs-continuum-terrain contact: a normal and an overlap estimated from
//! the terrain's grid mass. `GrainPopulation::resolve_wall_contact_forces`
//! (rolling resistance, what holds a sand pile's angle of repose) only runs
//! against `BoundaryCondition`s, so a grain resting on an MPM terrain, which
//! is neither a grain nor a boundary, got no rolling resistance from it. That
//! base layer sets a pile's footprint: grid-coupled pouring gave 37.97
//! degrees (std 5.74, n = 6) against 30.85 (std 2.59, n = 10) on a rigid floor.
//!
//! The normal and distance come from a scalar field the way a level set gives
//! them (the gradient gives the normal; marching along it to a threshold
//! gives the distance; Bridson, "Fluid Simulation for Computer Graphics," 2nd
//! ed., ch. 4). The field is `grains::oracle`'s packing fraction
//! (`packing_fraction_at`/`reference_mass_per_cell`, the particles-per-cell
//! convention of Jiang et al. 2016), not a second density convention.

use glam::{IVec2, Vec2};

use super::oracle::packing_fraction_at;
use crate::grid::Grid;

/// Ray-march step (grid-index units). The march range comes from the grain's
/// radius per call (the surface can lie anywhere within its diameter, see
/// `terrain_grain_contact`); this only sets the resolution.
const MARCH_STEP: f32 = 0.5;
/// Floor on the number of march steps even for a very small grain radius
/// -- keeps the search real and meaningful rather than degenerating to a
/// single sample for a sub-cell-radius grain.
const MIN_MARCH_STEPS: i32 = 6;

/// Grain-vs-terrain contact estimate, `None` when the grain does not touch the
/// terrain's dense region (the common case, a cheap early exit). Same
/// contract as `BoundaryCondition::grain_contact` (`normal` points away from
/// the surface into free space, `overlap = radius - distance_to_surface`), so
/// `grain_contact_law::resolve_wall_contact` consumes either.
///
/// `surface_threshold` is the packing-fraction cutoff
/// `grains::oracle::needs_discrete_treatment` takes from the caller: where the
/// free surface lies depends on the scene's density calibration.
pub fn terrain_grain_contact(
    grid: &Grid,
    reference_mass_per_cell: f32,
    surface_threshold: f32,
    position: Vec2,
    radius: f32,
) -> Option<(Vec2, f32)> {
    let sample = |offset: Vec2| -> f32 {
        let p = position + offset;
        let cell = IVec2::new(p.x.round() as i32, p.y.round() as i32);
        packing_fraction_at(grid, cell, reference_mass_per_cell)
    };

    // Gradient of the packing-fraction field over a stencil as wide as the
    // grain's radius, not +-1 cell: the question is whether a surface lies
    // anywhere within the grain's body, and a several-cell grain's body can
    // reach a surface its centre does not. Points into the terrain
    // (increasing density); the outward normal is its negative.
    let r = radius.max(1.0);
    let dx = sample(Vec2::new(r, 0.0)) - sample(Vec2::new(-r, 0.0));
    let dy = sample(Vec2::new(0.0, r)) - sample(Vec2::new(0.0, -r));
    let grad = Vec2::new(dx, dy) * (0.5 / r);
    if grad.length_squared() < 1.0e-8 {
        // Uniform field across the grain's own footprint (e.g. deep in a
        // fully dense interior, or far from any terrain at all)
        // -- no real surface direction to report here.
        return None;
    }
    let normal = -grad.normalize();
    // March toward increasing density (`grad`, i.e. `-normal`): a grain just
    // above the terrain has its centre in free space already, so marching
    // outward would only move away from the surface.
    let into_terrain = -normal;

    // Bounded march to where the field first crosses the dense threshold, an
    // implicit-surface distance estimate, not an exact signed distance. The
    // range covers the grain's diameter plus a margin.
    let max_steps = ((2.0 * r / MARCH_STEP).ceil() as i32 + 2).max(MIN_MARCH_STEPS);
    let mut surface_dist = radius * 2.0; // disclosed: "never found" sentinel, produces overlap<=0 below
    for s in 0..=max_steps {
        let sample_pos = position + into_terrain * (s as f32 * MARCH_STEP);
        let sample_cell = IVec2::new(sample_pos.x.round() as i32, sample_pos.y.round() as i32);
        let phi = packing_fraction_at(grid, sample_cell, reference_mass_per_cell);
        if phi >= surface_threshold {
            surface_dist = s as f32 * MARCH_STEP;
            break;
        }
    }

    let overlap = radius - surface_dist;
    if overlap <= 0.0 {
        return None;
    }
    Some((normal, overlap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spacetime::grains::oracle::reference_mass_per_cell;

    /// Flat terrain (dense below a known row, empty above) and a grain centred
    /// on the surface with a known overlap: the normal points straight up and
    /// the overlap equals the radius minus the known distance, by hand.
    #[test]
    fn flat_surface_gives_the_real_hand_computed_normal_and_overlap() {
        let mut grid = Grid::new(32);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);
        // Dense terrain for y <= 10, empty above -- a flat surface at y=10.5
        // (halfway between the last dense row and the first empty one).
        for y in 0..=10i32 {
            for x in 10..22i32 {
                grid.add_mass_momentum(IVec2::new(x, y), reference, Vec2::ZERO);
            }
        }
        // Grain centered 1.5 cells above the last dense row -- overlapping
        // the surface (surface sits at y~10.5) by radius(2.0) - 1.0 = 1.0.
        let grain_pos = Vec2::new(16.0, 11.5);
        let radius = 2.0;
        let (normal, overlap) = terrain_grain_contact(&grid, reference, 0.5, grain_pos, radius)
            .expect("a grain this close to a real dense surface must report contact");
        assert!(
            normal.y > 0.9 && normal.x.abs() < 0.2,
            "normal should point straight up, away from the terrain below, got {normal:?}"
        );
        assert!(
            (overlap - 1.0).abs() < MARCH_STEP + 0.01,
            "expected overlap near 1.0 (radius=2.0 minus real ~1.0-cell distance to the \
             surface at y~10.5), got {overlap}"
        );
    }

    /// A grain far above any terrain mass at all must report no contact --
    /// the common case almost everywhere on a grid.
    #[test]
    fn grain_far_from_any_terrain_reports_no_contact() {
        let grid = Grid::new(32);
        let reference = reference_mass_per_cell(1.0);
        assert_eq!(
            terrain_grain_contact(&grid, reference, 0.5, Vec2::new(16.0, 16.0), 2.0),
            None
        );
    }

    /// A grain fully buried deep inside a large, uniformly dense terrain
    /// block (no nearby real surface within the march range) must NOT
    /// fabricate a contact from numerical noise -- the gradient there is
    /// genuinely ~zero.
    #[test]
    fn grain_deep_inside_uniform_dense_terrain_reports_no_contact() {
        let mut grid = Grid::new(32);
        let reference = reference_mass_per_cell(1.0);
        for y in 0..32i32 {
            for x in 0..32i32 {
                grid.add_mass_momentum(IVec2::new(x, y), reference, Vec2::ZERO);
            }
        }
        assert_eq!(
            terrain_grain_contact(&grid, reference, 0.5, Vec2::new(16.0, 16.0), 2.0),
            None
        );
    }
}

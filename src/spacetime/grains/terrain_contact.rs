//! Real grain-vs-CONTINUUM-terrain contact normal/overlap estimation --
//! closes a genuine, root-caused structural gap: `GrainPopulation::
//! resolve_wall_contact_forces` (real rolling resistance, the ONE
//! parameter this whole project's own sand investigation found necessary
//! to hold a genuine angle of repose at all) only ever runs against real
//! `BoundaryCondition` implementors (static geometry) -- a grain resting
//! on a real, DYNAMIC MPM terrain material (sharing the same grid via
//! ordinary P2G/G2P, not a `BoundaryCondition`) gets ZERO rolling
//! resistance from that contact, because the terrain is neither a grain
//! (`resolve_contact_forces`) nor a boundary (`resolve_wall_contact_
//! forces`). Root-caused 2026-09-14 while investigating why grid-coupled
//! pouring gives a real, measured, higher/less-reliable angle
//! (mean=37.97deg, std=5.74deg, n=6) than the standalone rigid-floor
//! result (30.85deg, std=2.59deg, n=10) -- the BASE layer of grains,
//! which directly sets the pile's own footprint, is exactly the layer
//! missing this mechanism.
//!
//! Real technique: estimate a local surface normal + penetration overlap
//! from the terrain's own real grid mass field, the same way an implicit
//! surface (a level set) yields a normal and distance from a scalar field
//! -- gradient of the field gives the normal direction, marching along it
//! until the field crosses a real threshold gives the distance (standard
//! numerical technique, e.g. Bridson, "Fluid Simulation for Computer
//! Graphics," 2nd ed., ch. 4 -- not invented here). Reuses `grains::
//! oracle`'s own real, already-cited packing-fraction field
//! (`packing_fraction_at`/`reference_mass_per_cell`, Jiang et al. 2016's
//! dense-particles-per-cell convention) as that scalar field, rather than
//! introducing a second, competing density convention.

use glam::{IVec2, Vec2};

use super::oracle::packing_fraction_at;
use crate::grid::Grid;

/// Real, disclosed bounded ray-march step size (grid-index units) -- the
/// real march RANGE is derived per-call from the grain's own radius (see
/// `terrain_grain_contact`'s own doc: the surface can sit anywhere within
/// the grain's own diameter), this only sets the resolution.
const MARCH_STEP: f32 = 0.5;
/// Floor on the number of march steps even for a very small grain radius
/// -- keeps the search real and meaningful rather than degenerating to a
/// single sample for a sub-cell-radius grain.
const MIN_MARCH_STEPS: i32 = 6;

/// Real, generic grain-vs-terrain contact estimate -- `None` when the
/// grain isn't touching the terrain's real dense region at all (the
/// common case almost everywhere on the grid; cheap early exit). Contract
/// matches `BoundaryCondition::grain_contact` exactly (`normal` points
/// AWAY from the surface into free space, `overlap = radius -
/// distance_to_surface`) so the SAME real contact-resolution code
/// (`grain_contact_law::resolve_wall_contact`) can consume either source
/// identically.
///
/// `surface_threshold` is the SAME real, scene-tunable packing-fraction
/// cutoff `grains::oracle::needs_discrete_treatment` already takes as a
/// caller-supplied parameter, not a hardcoded constant (that function's
/// own doc: "a real, scene-tunable cutoff, not hardcoded"). Deliberately
/// threaded through here the same way -- what counts as "the real free
/// surface" is a property of the scene's own density calibration, not a
/// universal number this function should assume on the caller's behalf.
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

    // Real gradient of the packing-fraction field, sampled at a stencil
    // width matching the GRAIN's OWN radius (not a fixed +-1 cell) -- a
    // grain's relevant question is "is there a real surface anywhere
    // within my own body," and a fixed narrow stencil would only ever see
    // a transition that happens to fall exactly one cell from the grain's
    // rounded center, missing the far more common case of a several-cell-
    // wide grain whose body reaches a surface its own center does not.
    // Points toward increasing density (into the terrain); the real
    // outward surface normal is the NEGATIVE of that.
    let r = radius.max(1.0);
    let dx = sample(Vec2::new(r, 0.0)) - sample(Vec2::new(-r, 0.0));
    let dy = sample(Vec2::new(0.0, r)) - sample(Vec2::new(0.0, -r));
    let grad = Vec2::new(dx, dy) * (0.5 / r);
    if grad.length_squared() < 1.0e-8 {
        // Uniform field across the grain's own footprint (e.g. deep in a
        // fully dense interior, or genuinely far from any terrain at all)
        // -- no real surface direction to report here.
        return None;
    }
    let normal = -grad.normalize();
    // Real march direction: TOWARD increasing density (i.e. `grad`'s own
    // direction, equivalently `-normal`) -- a grain floating just above
    // the terrain has its own CENTER in free space (phi < threshold
    // already), so marching outward (along `normal`) would only ever
    // move further away from the surface. Marching inward instead finds
    // the real distance from the grain's center to where the field first
    // crosses into genuinely dense territory.
    let into_terrain = -normal;

    // Real, bounded ray-march looking for where the field first crosses
    // INTO the real dense threshold -- a standard, disclosed implicit-
    // surface distance estimate, not an exact signed distance field.
    // Range covers the grain's own diameter plus margin, since the real
    // surface can sit anywhere within that span for a grain whose body
    // already reaches it.
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

    /// Real, hand-verified case: a flat terrain surface (dense below a
    /// known row, empty above), a grain centered exactly AT the surface
    /// with real, known overlap -- confirms both the normal direction
    /// (straight up, away from the dense terrain below) and the overlap
    /// magnitude (grain radius minus the real known distance to the
    /// surface) match a hand computation, not just "returns something."
    #[test]
    fn flat_surface_gives_the_real_hand_computed_normal_and_overlap() {
        let mut grid = Grid::new(32);
        let particle_mass = 1.0;
        let reference = reference_mass_per_cell(particle_mass);
        // Dense terrain for y <= 10, empty above -- a real flat surface at y=10.5
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
    /// the common case almost everywhere on a real grid.
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

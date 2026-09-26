use std::collections::HashMap;

use glam::IVec2;

use super::{FxU32BuildHasher, Grid, flat_index};

/// Material-Induced Boundary Friction (MIBF) accumulator -- see
/// `MaterialModel::current_friction_coefficient`'s own doc for the real
/// citation (Blatny & Gaume 2025, `tmp/ref_matter.md` sec.19). Mirrors
/// `contact::ContactCell`'s own real, already-proven shape: a sparse side
/// map keyed the same way as the main `cells`, only ever allocated at
/// nodes a friction-reporting particle actually touches, with its own
/// dirty list so `has_friction_activity()` gates every extra cost to zero
/// for a scene with no granular material.
///
/// `friction_mass` is tracked SEPARATELY from `Cell::mass` (not reused) --
/// a mixed-material node (e.g. sand sharing a node with an elastic solid)
/// must not have its average pulled toward a wrong value by mass that
/// never reported a coefficient at all.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FrictionCell {
    /// Dual-phase field, same convention as `Cell::momentum`: during P2G,
    /// the accumulated `weight*mass_i*coefficient` sum; after
    /// `normalize_friction()`, the true mass-weighted average coefficient.
    friction: f32,
    friction_mass: f32,
}

pub(super) type FrictionCellMap = HashMap<u32, FrictionCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate one particle's mass-weighted friction contribution during
    /// P2G, additively alongside the normal `add_mass_momentum` call for
    /// the SAME particle -- a second, separate accumulator, not a
    /// replacement. `weight_mass` is `weight*mass_i` (already computed by
    /// the caller for the ordinary mass/momentum scatter, not recomputed
    /// here). OOB silently ignored.
    pub fn add_friction_mass(&mut self, cell_pos: IVec2, weight_mass: f32, coefficient: f32) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        match self.friction_cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                cell.friction += weight_mass * coefficient;
                cell.friction_mass += weight_mass;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.friction_dirty.push(idx);
                e.insert(FrictionCell {
                    friction: weight_mass * coefficient,
                    friction_mass: weight_mass,
                });
            }
        }
    }

    /// Normalizes every touched friction cell's raw weighted sum into a true
    /// mass-weighted average -- same pattern as `normalize_velocities()`,
    /// iterating only `friction_dirty` (O(touched), not O(resolution^2)).
    pub fn normalize_friction(&mut self) {
        let (dirty, cells) = (&self.friction_dirty, &mut self.friction_cells);
        for &idx in dirty {
            if let Some(cell) = cells.get_mut(&idx)
                && cell.friction_mass > 0.0
            {
                cell.friction /= cell.friction_mass;
            }
        }
    }

    /// Real, normalized per-node friction coefficient at the flat index a
    /// `BoundaryCondition::apply_to_grid_velocity_with_node_friction` call
    /// already carries (same `x*resolution+y` convention `flat_index`
    /// uses) -- `None` when no friction-reporting particle ever touched
    /// this node this substep, letting the caller fall back to its own
    /// fixed coefficient (the same "caller decides the fallback"
    /// convention `grip_velocity_at` already establishes for the contact
    /// system, adapted to return `Option` since here the fallback is a
    /// BOUNDARY's own constant, not another grid quantity).
    pub fn node_friction_at_index(&self, idx: usize) -> Option<f32> {
        self.friction_cells
            .get(&(idx as u32))
            .filter(|c| c.friction_mass > 0.0)
            .map(|c| c.friction)
    }

    /// True if any friction-reporting particle touched the grid this
    /// substep. Gates the extra MIBF work in P2G/boundary application --
    /// when false (every scene with no granular material active), those
    /// paths run their original, unmodified logic.
    pub const fn has_friction_activity(&self) -> bool {
        !self.friction_dirty.is_empty()
    }

    /// Same as `active_cells_with_index_mut`, additionally yielding each
    /// node's real MIBF friction coefficient (`None` when no friction-
    /// reporting particle touched it). Exists because `apply_boundary_
    /// conditions_to_grid` (a DIFFERENT module) cannot call `&self`
    /// `node_friction_at_index` on `grid` while `active_cells_with_index_
    /// mut`'s returned iterator still holds `grid`'s own `&mut self`
    /// borrow from the caller's perspective -- Rust sees that as a whole-
    /// struct borrow from outside this module, even though the two fields
    /// (`cells`, `friction_cells`) are disjoint. THIS method lives inside
    /// `grid`'s own module tree, where splitting `self.cells`/`self.
    /// friction_cells`/`self.dirty` into independent local borrows before
    /// building the iterator (the exact same trick `active_cells_with_
    /// index_mut` already uses for `dirty`/`cells`) is legal, so the
    /// combined result can safely cross the module boundary as one
    /// opaque iterator.
    pub fn active_cells_with_index_and_friction_mut(
        &mut self,
    ) -> impl Iterator<Item = (usize, &mut super::Cell, Option<f32>)> {
        let friction_cells = &self.friction_cells;
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        let ptr = cells as *mut super::CellMap;
        dirty.iter().filter_map(move |&idx| {
            // SAFETY: same as `active_cells_with_index_mut` -- unique
            // indices in `dirty`, no concurrent inserts during this loop.
            let cell = unsafe { (*ptr).get_mut(&idx) }?;
            let friction = friction_cells
                .get(&idx)
                .filter(|c| c.friction_mass > 0.0)
                .map(|c| c.friction);
            Some((idx as usize, cell, friction))
        })
    }
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::super::Grid;

    /// A single particle's own contribution, normalized, must equal its
    /// own reported coefficient exactly (trivial mass-weighted average of
    /// one term).
    #[test]
    fn single_particle_normalizes_to_its_own_coefficient() {
        let mut grid = Grid::new(8);
        grid.add_friction_mass(IVec2::new(2, 2), 3.0, 0.7);
        grid.normalize_friction();
        let idx = 2 * 8 + 2;
        assert!((grid.node_friction_at_index(idx).unwrap() - 0.7).abs() < 1.0e-6);
    }

    /// Real, hand-computed mass-weighted average of two particles sharing
    /// one node -- not just "some value comes back."
    #[test]
    fn two_particles_sharing_a_node_average_by_mass() {
        let mut grid = Grid::new(8);
        // particle A: weight*mass=1.0, coefficient=0.2
        // particle B: weight*mass=3.0, coefficient=0.6
        // expected average = (1.0*0.2 + 3.0*0.6)/(1.0+3.0) = 2.0/4.0 = 0.5
        grid.add_friction_mass(IVec2::new(4, 4), 1.0, 0.2);
        grid.add_friction_mass(IVec2::new(4, 4), 3.0, 0.6);
        grid.normalize_friction();
        let idx = 4 * 8 + 4;
        let got = grid.node_friction_at_index(idx).unwrap();
        assert!(
            (got - 0.5).abs() < 1.0e-6,
            "expected mass-weighted average 0.5, got {got}"
        );
    }

    /// A node no friction-reporting particle ever touched must return
    /// `None`, not a silent zero -- the "caller decides the fallback"
    /// contract `node_friction_at_index`'s own doc promises.
    #[test]
    fn untouched_node_returns_none_not_zero() {
        let mut grid = Grid::new(8);
        grid.add_friction_mass(IVec2::new(1, 1), 1.0, 0.9);
        grid.normalize_friction();
        let untouched_idx = 5 * 8 + 5;
        assert_eq!(grid.node_friction_at_index(untouched_idx), None);
    }

    /// A mixed-material node -- some mass from a non-reporting particle
    /// (only ever added via `add_mass_momentum`, never `add_friction_
    /// mass`) alongside a real friction-reporting particle -- must NOT
    /// have its average diluted by the non-reporting mass. This is the
    /// real reason `friction_mass` is tracked separately from `Cell::mass`.
    #[test]
    fn non_reporting_particles_mass_does_not_dilute_the_average() {
        let mut grid = Grid::new(8);
        let pos = IVec2::new(3, 3);
        // A large non-granular particle shares this node -- only ordinary
        // mass/momentum, no friction report at all.
        grid.add_mass_momentum(pos, 100.0, glam::Vec2::ZERO);
        // One real granular particle reports its own coefficient.
        grid.add_friction_mass(pos, 1.0, 0.42);
        grid.normalize_friction();
        let idx = 3 * 8 + 3;
        let got = grid.node_friction_at_index(idx).unwrap();
        assert!(
            (got - 0.42).abs() < 1.0e-6,
            "non-reporting mass must not dilute the friction average, got {got}"
        );
    }
}

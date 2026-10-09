use glam::Vec2;

use super::BoundaryCondition;

/// A static, frictionless solid rectangle inside the domain: a ridge between
/// two basins, a sill in a channel, the wall of a beaker.
///
/// It is `SlipBoundary` turned inside out. Every grid node whose position
/// (`index + 0.5`, see `grid::kernel::quadratic_weights`) lies in
/// `[min, max]` is a wall node, and its velocity loses the component that
/// points into the solid along the normal of the nearest face; matter may
/// leave the face freely, as at an open container wall. The constraint acts
/// on grid velocity, so it is a no-penetration condition on the momentum
/// equation, not a correction of particles after the fact.
///
/// `clamp_particle_position` is the same last-resort backstop as
/// `SlipBoundary`'s: a particle deeper than one cell inside the solid is
/// moved back to that depth. A wall the grid condition holds never needs it.
///
/// Each axis must span at least two cells, so that both opposite faces own
/// their own column (or row) of wall nodes: a single column would carry the
/// normal of one face only and let matter through from the other side.
///
/// No `grain_contact`: like the outer box walls, a flat axis-aligned face
/// gives grains no rolling torque from the boundary.
#[derive(Debug, Clone, Copy)]
pub struct StaticBoxBoundary {
    /// Lower-left corner of the solid, in grid units.
    pub min: Vec2,
    /// Upper-right corner of the solid, in grid units.
    pub max: Vec2,
}

impl StaticBoxBoundary {
    /// Solid rectangle between `min` and `max` (grid units).
    ///
    /// # Panics
    /// If either side is shorter than two cells (see the type's doc).
    pub fn new(min: Vec2, max: Vec2) -> Self {
        let size = max - min;
        assert!(
            size.x >= 2.0 && size.y >= 2.0,
            "StaticBoxBoundary needs at least two cells per axis so each face owns its wall nodes (got {size})"
        );
        Self { min, max }
    }

    /// Outward normal of the face nearest to `p`, a point inside the box.
    /// Ties go to the first face in the order left, right, bottom, top.
    fn nearest_face_normal(&self, p: Vec2) -> Vec2 {
        let faces = [
            (p.x - self.min.x, Vec2::NEG_X),
            (self.max.x - p.x, Vec2::X),
            (p.y - self.min.y, Vec2::NEG_Y),
            (self.max.y - p.y, Vec2::Y),
        ];
        let mut nearest = faces[0];
        for face in &faces[1..] {
            if face.0 < nearest.0 {
                nearest = *face;
            }
        }
        nearest.1
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

impl BoundaryCondition for StaticBoxBoundary {
    /// Frictionless, so it dissipates nothing (see `SlipBoundary`).
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        let node = Vec2::new(
            (cell_index / grid_res) as f32 + 0.5,
            (cell_index % grid_res) as f32 + 0.5,
        );
        if self.contains(node) {
            let normal = self.nearest_face_normal(node);
            let v_n = velocity.dot(normal);
            if v_n < 0.0 {
                *velocity -= v_n * normal;
            }
        }
        0.0
    }

    fn clamp_particle_position(&self, position: Vec2, _grid_res: usize) -> Vec2 {
        // One cell inside each face, as `clamp_position_inside_grid` stops a
        // particle one cell inside the outer wall nodes.
        let inner = Self {
            min: self.min + Vec2::ONE,
            max: self.max - Vec2::ONE,
        };
        if !inner.contains(position) {
            return position;
        }
        let normal = inner.nearest_face_normal(position);
        let mut pos = position;
        match normal {
            n if n == Vec2::NEG_X => pos.x = inner.min.x,
            n if n == Vec2::X => pos.x = inner.max.x,
            n if n == Vec2::NEG_Y => pos.y = inner.min.y,
            _ => pos.y = inner.max.y,
        }
        pos
    }

    // `true`, on measurement (1 cm cells, real gravity, water at
    // K = 2.25e5 Pa, 5 s; probe kept out of tree):
    //   1. Two pools at different depths either side of a tall ridge: no
    //      particle crosses, and the backstop never fires.
    //   2. The ridge face acts as the domain wall: the left pool, between
    //      the domain's left wall and the ridge, matches the same pool
    //      between two domain walls (centre of mass 23.14 against 23.31,
    //      mean J 0.9949 against 0.9952, deepest particle 0.55 against
    //      0.51 cells past the wall plane).
    //   3. A ridge lower than the pool: the water spills over the top face
    //      and corners and the two levels meet (21.7 and 22.4 cells over a
    //      sill at 20). Deepest particle 0.82 cells against
    //      0.59 at the domain walls in the same run; backstop never fires.
    // `tests/solver.rs::static_box_holds_two_pools_of_strict_water_apart_
    // under_gravity` keeps 1 and 2; it fails with the grid condition off.
    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRID: usize = 32;

    fn index(x: usize, y: usize) -> usize {
        x * GRID + y
    }

    fn ridge() -> StaticBoxBoundary {
        StaticBoxBoundary::new(Vec2::new(14.0, 0.0), Vec2::new(18.0, 20.0))
    }

    /// Velocity into a face is removed, velocity along it passes unchanged.
    #[test]
    fn face_blocks_inward_velocity_and_keeps_tangential() {
        let mut v = Vec2::new(2.0, -0.7);
        // Node 14 sits at x = 14.5, nearest the left face.
        ridge().apply_to_grid_velocity(index(14, 5), GRID, &mut v);
        assert_eq!(v, Vec2::new(0.0, -0.7));

        let mut v = Vec2::new(-2.0, 0.3);
        // Node 17 sits at x = 17.5, nearest the right face.
        ridge().apply_to_grid_velocity(index(17, 5), GRID, &mut v);
        assert_eq!(v, Vec2::new(0.0, 0.3));
    }

    /// Matter may leave a face: outward velocity is untouched.
    #[test]
    fn face_does_not_hold_outward_velocity() {
        let mut v = Vec2::new(-2.0, 0.4);
        ridge().apply_to_grid_velocity(index(14, 5), GRID, &mut v);
        assert_eq!(v, Vec2::new(-2.0, 0.4));
    }

    /// The top face of the ridge is a floor for what lands on it.
    #[test]
    fn top_face_is_a_floor() {
        let mut v = Vec2::new(0.5, -3.0);
        // Node (15, 19) sits at (15.5, 19.5): 0.5 below the top face.
        ridge().apply_to_grid_velocity(index(15, 19), GRID, &mut v);
        assert_eq!(v, Vec2::new(0.5, 0.0));
    }

    /// Nodes outside the box are untouched.
    #[test]
    fn nodes_outside_are_untouched() {
        let mut v = Vec2::new(3.0, -3.0);
        ridge().apply_to_grid_velocity(index(13, 5), GRID, &mut v);
        ridge().apply_to_grid_velocity(index(15, 20), GRID, &mut v);
        assert_eq!(v, Vec2::new(3.0, -3.0));
    }

    /// The backstop moves only a particle deeper than one cell inside, and
    /// only to that depth.
    #[test]
    fn backstop_only_catches_deep_particles() {
        let shallow = Vec2::new(14.6, 5.0);
        assert_eq!(ridge().clamp_particle_position(shallow, GRID), shallow);
        let deep = Vec2::new(15.4, 5.0);
        assert_eq!(
            ridge().clamp_particle_position(deep, GRID),
            Vec2::new(15.0, 5.0)
        );
    }
}

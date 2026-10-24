use glam::Vec2;

use super::{BoundaryCondition, apply_coulomb_wall, position_clamp_bounds};

/// Heightmap terrain boundary -- arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[i]`, the surface height in grid units
/// at `x = i`, linearly interpolated in between. A grid node is solid terrain
/// when its own position (`i + 0.5`, see `grid::kernel::quadratic_weights`)
/// lies at or below the surface.
///
/// The terrain surface normal comes from the heightmap's slope (central
/// difference `dh/dx`), not a fixed +Y: with +Y everywhere a sloped
/// heightmap is a staircase of flat blocks and grains released on it never
/// roll. A flat floor (`flat_floor`) has the normal (0,1) at every column.
///
/// Coulomb friction acts on the tangential velocity in this local frame
/// (horizontal friction when the slope is zero).
///
/// Outer axis-aligned walls are always enforced (same as `SlipBoundary`), so the
/// heightmap sits inside the standard simulation domain.
///
/// # Coordinate convention
/// Y increases upward. `heights[0]` is the left column, `heights[grid_res-1]` is the right.
/// Heights beyond the array length clamp to the last value.
///
/// # Examples
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::HeightmapBoundary;
/// // Flat floor at y=3, with a hill at column 20–40 rising to y=10
/// let mut heights = vec![3.0f32; 64];
/// for x in 20..40 { heights[x] = 3.0 + (10.0 - 3.0) * (1.0 - ((x as f32 - 30.0) / 10.0).abs()); }
/// let boundary = HeightmapBoundary::new(heights, 0.4, 2);
/// ```
#[derive(Debug, Clone)]
pub struct HeightmapBoundary {
    /// Terrain surface height in grid cells for each x-column. Fractional values are supported.
    pub heights: Vec<f32>,
    /// Coulomb friction coefficient of the terrain surface (>= 0, 0 is
    /// frictionless; see `FrictionBoundary::friction_coefficient`).
    pub friction: f32,
    /// Thickness of outer box walls (standard MPM boundary padding).
    pub wall_thickness: usize,
}

impl HeightmapBoundary {
    pub const fn new(heights: Vec<f32>, friction: f32, wall_thickness: usize) -> Self {
        Self {
            heights,
            friction,
            wall_thickness,
        }
    }

    /// Flat floor at a constant height -- equivalent to a floor-only boundary.
    pub fn flat_floor(grid_res: usize, floor_height: f32, friction: f32) -> Self {
        Self::new(vec![floor_height; grid_res], friction, 2)
    }

    /// Returns the height at a fractional x, linearly interpolated between the
    /// two bracketing columns (`heights[i]` is the surface height at `x = i`).
    /// Without it a grain's contact response jumped at every integer column,
    /// and rolling looked stair-stepped.
    #[inline]
    fn height_at_f32(&self, x: f32) -> f32 {
        if self.heights.is_empty() {
            return 0.0;
        }
        let last = self.heights.len() - 1;
        let x0 = x.floor().max(0.0) as usize;
        let x0 = x0.min(last);
        let x1 = (x0 + 1).min(last);
        let t = (x - x0 as f32).clamp(0.0, 1.0);
        self.heights[x0] * (1.0 - t) + self.heights[x1] * t
    }

    /// Returns the normal at a fractional x, by central difference of
    /// `height_at_f32`, so it varies smoothly with `x`. The tangent along the
    /// surface is (1, slope); the outward normal is that rotated 90 degrees
    /// counterclockwise, (-slope, 1): a surface rising to the right tilts it
    /// up and to the left, away from the uphill mass.
    #[inline]
    fn normal_at_f32(&self, x: f32) -> Vec2 {
        if self.heights.len() < 2 {
            return Vec2::Y;
        }
        let slope = self.height_at_f32(x + 0.5) - self.height_at_f32(x - 0.5);
        Vec2::new(-slope, 1.0).normalize()
    }
}

impl BoundaryCondition for HeightmapBoundary {
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let t = self.wall_thickness;
        let hi = grid_res.saturating_sub(t + 1);

        // Outer box walls -- standard slip (no-penetration, free tangential).
        if x < t {
            velocity.x = velocity.x.max(0.0);
        }
        if x > hi {
            velocity.x = velocity.x.min(0.0);
        }
        if y > hi {
            velocity.y = velocity.y.min(0.0);
        }

        // Terrain: a grid node is solid when its own position, `i + 0.5`
        // (`grid::kernel::quadratic_weights`), lies at or below the surface
        // there. The same Coulomb wall as every other boundary, in the
        // surface's local normal/tangent frame.
        let node_x = x as f32 + 0.5;
        let node_y = y as f32 + 0.5;
        if node_y <= self.height_at_f32(node_x) {
            return apply_coulomb_wall(velocity, self.normal_at_f32(node_x), self.friction);
        }
        0.0
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        // Outer walls.
        let (wall_min, wall_max) = position_clamp_bounds(self.wall_thickness, grid_res);
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));

        // Terrain: the same last-resort rule as the outer walls
        // (`position_clamp_bounds`: one cell past the wall plane), measured
        // from the surface under the particle. The physics is in
        // `apply_to_grid_velocity`; a particle the grid condition holds never
        // reaches this depth. Grains use `clamp_grain_position` instead.
        let deepest = self.height_at_f32(pos.x) - 1.0;
        if pos.y < deepest {
            pos.y = deepest;
        }

        pos
    }

    /// Returns the local normal and overlap of a grain touching the surface
    /// (see `BoundaryCondition::grain_contact`: grain-vs-terrain rolling
    /// torque), from the continuous height and normal (`height_at_f32`/
    /// `normal_at_f32`), so the response varies smoothly as the grain moves.
    fn grain_contact(&self, position: Vec2, radius: f32, _grid_res: usize) -> Option<(Vec2, f32)> {
        if self.heights.is_empty() {
            return None;
        }
        let terrain_h = self.height_at_f32(position.x);
        let normal = self.normal_at_f32(position.x);
        let surface_point = Vec2::new(position.x, terrain_h);
        let dist = (position - surface_point).dot(normal);
        if dist < radius {
            Some((normal, radius - dist))
        } else {
            None
        }
    }

    /// Radius-aware grain backstop: the outer walls use the generic
    /// `clamp_particle_position` clamp (a box edge has no tilt), the terrain
    /// uses this boundary's continuous, tilted, radius-aware `grain_contact`
    /// (see `BoundaryCondition::clamp_grain_position`).
    ///
    /// No per-step position correction: soft-sphere DEM resolves contact
    /// through the penalty force alone (`resolve_wall_contact`'s
    /// `normal_stiffness * overlap`), with the timestep bounded by
    /// `critical_timestep` to keep penetration small; GeoTaichi's
    /// `dem/contact/ContactKernel.py` (`wall_contact_model_type1/2`) has no
    /// position correction either. Correcting overlap to zero starved the
    /// friction cap of normal force (a slipping grain kept its slip for
    /// 10,000+ steps), and a slop margin or a Baumgarte fraction each fixed
    /// one scene and broke another.
    ///
    /// So this only catches tunnelling (a fast grain passing the surface in
    /// one substep): it fires at a full radius of overlap, the centre at the
    /// surface, while resting and rolling contact overlaps a few percent of
    /// the radius.
    fn clamp_grain_position(&self, position: Vec2, radius: f32, grid_res: usize) -> Vec2 {
        let (wall_min, wall_max) = position_clamp_bounds(self.wall_thickness, grid_res);
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));
        if let Some((normal, overlap)) = self.grain_contact(pos, radius, grid_res)
            && overlap > radius
        {
            pos += normal * (overlap - radius);
        }
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat floor's normal is exactly (0,1) at every column
    /// (`tests/stress.rs::boundary_count_stress` relies on it).
    #[test]
    fn flat_floor_normal_is_exactly_up_everywhere() {
        let b = HeightmapBoundary::flat_floor(20, 3.0, 0.4);
        for x in 0..20 {
            let n = b.normal_at_f32(x as f32 + 0.5);
            assert!(
                (n - Vec2::Y).length() < 1e-6,
                "flat floor should have normal=(0,1) at every column, got {n:?} at x={x}"
            );
        }
    }

    /// A grid node is terrain when its own position `i + 0.5` lies at or
    /// below the surface: with the floor at y = 3, node row 2 (at 2.5) is
    /// solid and node row 3 (at 3.5) is open air.
    #[test]
    fn terrain_nodes_are_tested_at_their_physical_position() {
        let grid_res = 32usize;
        let b = HeightmapBoundary::flat_floor(grid_res, 3.0, 0.0);
        let falling = Vec2::new(0.0, -1.0);

        let mut below = falling;
        b.apply_to_grid_velocity(10 * grid_res + 2, grid_res, &mut below);
        assert_eq!(below, Vec2::ZERO, "node at y = 2.5 is inside the floor");

        let mut above = falling;
        b.apply_to_grid_velocity(10 * grid_res + 3, grid_res, &mut above);
        assert_eq!(above, falling, "node at y = 3.5 is above the floor");
    }

    /// A sloped heightmap gives a tilted normal, so the downhill component of
    /// gravity accelerates a resting body; a vertical normal cancels all of
    /// it.
    #[test]
    fn sloped_heightmap_normal_is_tilted_and_lets_gravity_drive_motion_downhill() {
        // A 45-degree rise: height increases by 1.0 per column.
        let heights: Vec<f32> = (0..40).map(|x| 2.0 + x as f32).collect();
        let b = HeightmapBoundary::new(heights, 0.2, 2);

        let n = b.normal_at_f32(20.5);
        assert!(
            (n.length() - 1.0).abs() < 1e-5,
            "normal must stay unit length, got {n:?}"
        );
        assert!(
            n.x < -0.1,
            "a surface rising to the right must tilt its normal to the \
             left (negative x), got {n:?}"
        );

        // A cell resting on this slope under straight-down gravity for one
        // substep keeps some velocity along the tangent instead of being
        // fully arrested.
        let gravity_dt = Vec2::new(0.0, -0.05); // one substep's worth of g*dt
        let mut v = gravity_dt;
        // cell_index = x*grid_res + y, as in `apply_to_grid_velocity`.
        let grid_res = 64usize;
        // Node (20, 22) sits at (20.5, 22.5), exactly on the surface there
        // (h(20.5) = 2 + 20.5).
        let cell_index = 20 * grid_res + 22;
        b.apply_to_grid_velocity(cell_index, grid_res, &mut v);
        assert!(
            v.length() > 1e-4,
            "gravity applied to a body resting on a real slope must leave \
             SOME surviving tangential velocity (real downhill motion), \
             not be fully arrested like the old fixed-+Y-normal bug -- got \
             v={v:?}"
        );
    }
}

use glam::Vec2;

use super::{BoundaryCondition, position_clamp_bounds};

/// Heightmap terrain boundary -- arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[x]` in grid units for each x-column.
/// All grid cells at (x, y) with `y ≤ heights[x]` are treated as solid terrain.
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
    /// Coulomb friction coefficient on the terrain surface. 0.0 = slip, 1.0 = full friction.
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

    /// Sample the terrain height at grid column x. Clamps to array bounds.
    #[inline]
    fn height_at(&self, x: usize) -> f32 {
        if self.heights.is_empty() {
            return 0.0;
        }
        self.heights[x.min(self.heights.len() - 1)]
    }

    /// Returns the surface normal at column `x`, from a central-difference
    /// slope (one-sided at the array edges). Unit length, pointing up
    /// (positive y component).
    #[inline]
    fn normal_at(&self, x: usize) -> Vec2 {
        if self.heights.len() < 2 {
            return Vec2::Y;
        }
        let last = self.heights.len() - 1;
        let slope = if x == 0 {
            self.heights[1] - self.heights[0]
        } else if x >= last {
            self.heights[last] - self.heights[last - 1]
        } else {
            (self.heights[x + 1] - self.heights[x - 1]) * 0.5
        };
        // Tangent along the surface is (1, slope); the outward normal is
        // that rotated 90 degrees CCW: (-slope, 1). For a surface rising to
        // the right (slope > 0), this correctly tilts up-and-left, away
        // from the uphill mass.
        Vec2::new(-slope, 1.0).normalize()
    }

    /// Returns the height at a fractional x, linearly interpolated between the
    /// two bracketing columns. Without it a grain's contact response jumped at
    /// every integer column, and rolling looked stair-stepped.
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
    /// `height_at_f32`, so it varies smoothly with `x`.
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

        // Heightmap terrain: cells at or below terrain surface.
        let terrain_h = self.height_at(x);
        if (y as f32) <= terrain_h {
            let normal = self.normal_at(x);
            let v_n = velocity.dot(normal);
            // Block velocity moving INTO the surface along its local
            // normal (v_n < 0), same real Coulomb-wall convention every
            // other boundary in this engine uses -- just projected onto the
            // correct tangent/normal frame instead of assuming horizontal/
            // vertical.
            if v_n < 0.0 {
                let tangent_v = *velocity - v_n * normal;
                *velocity = tangent_v;
                if self.friction > 0.0 {
                    let friction_impulse = self.friction * v_n.abs();
                    let v_t = velocity.length();
                    let v_t_after = (v_t - friction_impulse).max(0.0);
                    *velocity = if v_t > friction_impulse {
                        *velocity * (v_t_after / v_t)
                    } else {
                        Vec2::ZERO
                    };
                    // Same tangential-only accounting as `apply_coulomb_wall`
                    // -- see that function's doc for why the normal part of
                    // the correction is deliberately not reported as heat.
                    return 0.5 * (v_t * v_t - v_t_after * v_t_after);
                }
            }
        }
        0.0
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        // Outer walls.
        let (wall_min, wall_max) = position_clamp_bounds(self.wall_thickness, grid_res);
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));

        // Terrain: push particles above the surface. Keeps the per-column
        // lookup, not `height_at_f32`: the continuous one here froze a grain
        // that rolled down a slope. A last-resort domain clamp; the physics
        // is in `apply_to_grid_velocity` and `grain_contact`, which use the
        // continuous lookup.
        let x_col = (pos.x as usize).min(grid_res.saturating_sub(1));
        let terrain_h = self.height_at(x_col);
        if pos.y < terrain_h + 1.0 {
            pos.y = terrain_h + 1.0;
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
            let n = b.normal_at(x);
            assert!(
                (n - Vec2::Y).length() < 1e-6,
                "flat floor should have normal=(0,1) at every column, got {n:?} at x={x}"
            );
        }
    }

    /// A sloped heightmap gives a tilted normal, so the downhill component of
    /// gravity accelerates a resting body; a vertical normal cancels all of
    /// it.
    #[test]
    fn sloped_heightmap_normal_is_tilted_and_lets_gravity_drive_motion_downhill() {
        // A real 30-degree-ish rise: height increases by 1.0 per column.
        let heights: Vec<f32> = (0..40).map(|x| 2.0 + x as f32).collect();
        let b = HeightmapBoundary::new(heights, 0.2, 2);

        let n = b.normal_at(20);
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
        let x = 20usize;
        let y = b.height_at(x) as usize; // exactly on the surface
        let cell_index = x * grid_res + y;
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

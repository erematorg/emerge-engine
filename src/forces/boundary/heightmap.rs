use glam::Vec2;

use super::{BoundaryCondition, apply_coulomb_wall};

/// Heightmap terrain boundary — arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[x]` in grid units for each x-column.
/// All grid cells at (x, y) with `y ≤ heights[x]` are treated as solid terrain.
/// The real local surface normal (`normalize(-dh/dx, 1)`, a central
/// difference of `heights` -- reduces to exactly +Y wherever the terrain is
/// flat) is used for contact, not a fixed +Y -- real per-column slope
/// support, not a staircase of flat micro-floors. Coulomb friction is
/// applied on the real tangential component at the surface (`apply_coulomb_
/// wall`, the same primitive `KinematicCircleBoundary` uses).
///
/// Outer axis-aligned walls are always enforced (same as `SlipBoundary`), so the
/// heightmap sits inside the standard simulation domain.
///
/// # Coordinate convention
/// Y increases upward. `heights[0]` is the left column, `heights[grid_res-1]` is the right.
/// Heights beyond the array length clamp to the last value.
///
/// # Usage
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

    /// Flat floor at a constant height — equivalent to a floor-only boundary.
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
}

impl BoundaryCondition for HeightmapBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let t = self.wall_thickness;
        let hi = grid_res.saturating_sub(t + 1);

        // Outer box walls — standard slip (no-penetration, free tangential).
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
            // Real local terrain-slope normal (central difference), not a
            // fixed +Y. Found live 2026-08-16: the old fixed-+Y version
            // (kept in this struct's own doc history) blocks ALL downward
            // velocity every substep regardless of slope, deleting it
            // outright instead of redirecting the part of it that's real
            // tangential (downhill) motion -- fine for material with its
            // own internal pressure/stress pushing it downhill (ordinary
            // granular MPM particles, which is why existing snow/sand
            // scenes never surfaced this), but a rigid DEM grain has no
            // such internal stress of its own and simply never accelerates
            // at all on a real slope under the old model (confirmed via a
            // real isolated A/B: an app-level probe using this exact
            // `normalize(-dh/dx, 1)` formula reached speed=15.4 down this
            // same terrain; the grain under the old Y-only boundary decayed
            // to a dead stop instead). `HeightmapBoundary` has exactly one
            // real consumer today (`examples/rolling_snowball_demo.rs` and
            // its headless probe sibling) and both use a real, non-flat
            // slope, so this is a real fix, not a hypothetical one -- and
            // for a genuinely flat floor (`dh_dx=0`) this normal reduces to
            // exactly `(0,1)`, reproducing the old behavior bit-for-bit, so
            // any future flat-floor use is unaffected. Reuses
            // `apply_coulomb_wall`, the same real Coulomb-wall primitive
            // `KinematicCircleBoundary` already relies on -- not a new
            // friction formula.
            let h_minus = self.height_at(x.saturating_sub(1));
            let h_plus = self.height_at((x + 1).min(grid_res.saturating_sub(1)));
            let dh_dx = (h_plus - h_minus) * 0.5;
            let normal = Vec2::new(-dh_dx, 1.0).normalize();
            apply_coulomb_wall(velocity, normal, self.friction);
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        // Outer walls.
        let wall_min = self.wall_thickness.saturating_sub(1) as f32;
        let wall_max = grid_res.saturating_sub(self.wall_thickness) as f32;
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));

        // Terrain: push particles above the surface.
        let x_col = (pos.x as usize).min(grid_res.saturating_sub(1));
        let terrain_h = self.height_at(x_col);
        if pos.y < terrain_h + 1.0 {
            pos.y = terrain_h + 1.0;
        }

        pos
    }
}

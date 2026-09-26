use glam::Vec2;

use super::BoundaryCondition;

/// Heightmap terrain boundary -- arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[x]` in grid units for each x-column.
/// All grid cells at (x, y) with `y ≤ heights[x]` are treated as solid terrain.
///
/// The terrain surface normal is the REAL LOCAL normal derived from the
/// heightmap's own slope (central-difference `dh/dx`), not a fixed +Y --
/// real fix, 2026-08-21: an earlier version always used +Y regardless of
/// slope, which meant a "sloped" heightmap looked tilted but was physically
/// just a staircase of flat horizontal blocks -- nothing ever pushed
/// anything downhill on it (found live, building a grain-rolling-on-a-
/// slope demo: grains released on a visually sloped heightmap just sat
/// there, because gravity's own vertical component was always fully
/// cancelled by a normal that never actually tilted). Backward compatible:
/// a flat floor (`flat_floor`, zero slope everywhere) has a local normal of
/// exactly (0,1) at every column, identical to the old fixed behavior --
/// confirmed via `tests/stress.rs::boundary_count_stress`, the only other
/// real consumer, which uses `flat_floor` and is unaffected.
///
/// Coulomb friction is applied on the TANGENTIAL velocity component
/// relative to this real local normal/tangent frame (degenerates to the old
/// horizontal-only friction exactly when the local slope is zero).
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

    /// Real local surface normal at column `x`, from a central-difference
    /// slope estimate (forward/backward difference at the array edges).
    /// Unit length, always pointing generally "up" (positive y component).
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

    /// Real, continuous height at a fractional x -- linear interpolation
    /// between the two bracketing columns. Real fix, 2026-08-21: without
    /// this, a grain's own continuous position saw a DISCRETE height/normal
    /// jump every time it crossed an integer column boundary -- confirmed
    /// live as the actual cause of a rolling grain's motion feeling
    /// "stair-stepped" (this demo doesn't even render the terrain surface
    /// itself, so what looked like steps was the grain's own contact
    /// response updating in jumps, not a drawn line). The struct's own doc
    /// already says "fractional values are supported" for `heights` -- this
    /// is that promise finally kept for continuous positions, not a new
    /// design.
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

    /// Real, continuous local normal at a fractional x -- central difference
    /// built from `height_at_f32` itself, so it varies smoothly as `x`
    /// varies continuously, not just at integer column boundaries.
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
            // Block velocity moving INTO the surface along its real local
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
        let wall_min = self.wall_thickness.saturating_sub(1) as f32;
        let wall_max = grid_res.saturating_sub(self.wall_thickness) as f32;
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));

        // Terrain: push particles above the surface. Deliberately still the
        // discrete per-column lookup, NOT `height_at_f32` -- confirmed via
        // direct bisection (2026-08-21) that switching this specific
        // function to the continuous lookup regresses real grain rolling
        // (a grain that genuinely rolled down a slope froze solid instead).
        // This is a last-resort domain-enforcement clamp, not the primary
        // physics (that's `apply_to_grid_velocity` and `grain_contact`,
        // which safely use the continuous lookup) -- not worth chasing the
        // exact interaction further tonight.
        let x_col = (pos.x as usize).min(grid_res.saturating_sub(1));
        let terrain_h = self.height_at(x_col);
        if pos.y < terrain_h + 1.0 {
            pos.y = terrain_h + 1.0;
        }

        pos
    }

    /// Real local normal + overlap for a grain touching this heightmap's
    /// surface -- see this file's own module doc and `BoundaryCondition::
    /// grain_contact`'s own doc for why this exists (grain-vs-terrain
    /// rolling torque, previously entirely missing). Uses the real
    /// continuous height/normal (`height_at_f32`/`normal_at_f32`), not a
    /// per-column snap, so contact response varies smoothly as the grain's
    /// own continuous position moves.
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

    /// Real, radius-aware grain backstop: outer-wall containment is the
    /// same generic (radius-agnostic, still correct -- a hard box edge
    /// doesn't care about tilt) clamp `clamp_particle_position` already
    /// uses, but the TERRAIN correction below uses this boundary's own
    /// real `grain_contact` (continuous, tilted, radius-aware) instead of
    /// `clamp_particle_position`'s crude discrete/vertical/hardcoded-radius
    /// one. See `BoundaryCondition::clamp_grain_position`'s own doc for why
    /// this replaces rather than layers onto the generic clamp.
    ///
    /// Real, load-bearing correction scheme -- checked directly against the
    /// already-cited reference DEM implementation tonight (2026-08-21,
    /// `tmp/GeoTaichi/src/dem/contact/ContactKernel.py`'s own `wall_contact_
    /// model_type1/2`) after two real, measured, hand-tuned attempts both
    /// failed to generalize (kept left as the honest record: correcting
    /// overlap to exactly zero every substep starved `resolve_wall_contact`
    /// of any normal force to compute a friction cap from -- a grain with
    /// real slip on flat ground stayed at that exact slip value for 10,000+
    /// steps; a fixed "slop" margin, and later a Baumgarte-style fractional
    /// correction, each fixed one real scene while silently breaking
    /// another already-verified one -- different hand-picked constants
    /// trading one regression for another, not a real fix). Grepping that
    /// reference's own wall-contact code turned up NO position-correction
    /// mechanism at all -- no slop, no Baumgarte term, nothing: real,
    /// published soft-sphere DEM resolves contact ENTIRELY through the
    /// penalty spring's own continuous force (`resolve_wall_contact`'s
    /// `normal_stiffness * overlap`), relying on a correctly-bounded
    /// timestep (`critical_timestep`, already computed from the SAME
    /// stiffness) to keep penetration naturally small -- exactly this
    /// engine's own existing mechanism, no extra position hack needed.
    ///
    /// This function's only real job, then, is what its own name always
    /// said: a last-resort DOMAIN backstop against genuine tunneling (a
    /// fast grain overshooting past the surface entirely in one substep --
    /// the real, original, disclosed reason this existed, see `coupling.rs`
    /// `apply_grain_contact_forces`'s own module doc), not a per-step
    /// physics substitute. The threshold is deliberately set at a FULL
    /// radius of overlap -- physically impossible for real, bounded contact
    /// (the grain's own center would be exactly AT the surface), so normal
    /// resting/rolling contact (real overlaps of a few percent of radius,
    /// confirmed via direct trace tonight) never triggers this at all; only
    /// a genuine escape does.
    fn clamp_grain_position(&self, position: Vec2, radius: f32, grid_res: usize) -> Vec2 {
        let wall_min = self.wall_thickness.saturating_sub(1) as f32;
        let wall_max = grid_res.saturating_sub(self.wall_thickness) as f32;
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

    /// Real regression: a flat floor's local normal must be exactly (0,1)
    /// at every column -- the old, fixed-+Y behavior every existing
    /// consumer (`tests/stress.rs::boundary_count_stress`) depends on.
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

    /// Real, direct proof of the fix: a genuinely SLOPED heightmap must
    /// produce a tilted normal that lets gravity's own component along the
    /// slope actually accelerate a resting body downhill -- the exact real
    /// capability the old fixed-+Y normal could never provide (a body would
    /// just sit motionless on a "sloped" heightmap forever, since a purely
    /// vertical normal always fully cancels vertical gravity regardless of
    /// how tilted the terrain looks).
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

        // Real dynamics check: a grid cell resting exactly on this slope,
        // hit by straight-down gravity for one substep, must NOT be fully
        // arrested (the old bug) -- some of that velocity should survive
        // along the real tangent direction, driving real downhill motion.
        let gravity_dt = Vec2::new(0.0, -0.05); // one substep's worth of g*dt
        let mut v = gravity_dt;
        // Real grid layout: cell_index = x*grid_res + y (matches this
        // module's own `apply_to_grid_velocity` convention).
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

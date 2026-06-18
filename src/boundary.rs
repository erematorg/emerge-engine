use glam::Vec2;

use crate::particle::Particles;

pub trait BoundaryCondition: Send + Sync + core::fmt::Debug {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2);
    /// Clamp particle position to the valid domain after G2P.
    /// Not a physical force — last-resort domain enforcement so particles never escape the grid.
    /// Proper no-penetration physics lives in `apply_to_grid_velocity`.
    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2;
    fn post_g2p_particle(&self, _particles: &mut Particles, _i: usize, _grid_res: usize, _dt: f32) {
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlipBoundary {
    pub thickness: usize,
}

impl SlipBoundary {
    pub fn new(thickness: usize) -> Self {
        Self { thickness }
    }
}

impl BoundaryCondition for SlipBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        apply_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

/// Grid-level slip boundary with a tighter inner keep-out zone.
///
/// Identical physics to `SlipBoundary` — no-penetration enforced on grid velocities.
/// `predictive_wall_min` shrinks the safe zone so fast particles hitting the boundary
/// layer are caught by `clamp_particle_position` before they can escape. The actual
/// wall physics is still the grid-level normal-zeroing, not a particle-level correction.
#[derive(Debug, Clone, Copy)]
pub struct PredictiveBoundary {
    pub thickness: usize,
    pub predictive_wall_min: f32,
}

impl PredictiveBoundary {
    pub fn new(thickness: usize, predictive_wall_min: f32) -> Self {
        Self {
            thickness,
            predictive_wall_min,
        }
    }
}

impl BoundaryCondition for PredictiveBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        apply_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        let min = self.predictive_wall_min;
        let max = (grid_res as f32 - 1.0) - self.predictive_wall_min;
        position.clamp(Vec2::splat(min), Vec2::splat(max))
    }
}

/// Grid-level Coulomb wall boundary.
///
/// No-penetration (normal zeroed) + Coulomb friction on tangential component,
/// applied to grid cell velocities during grid update. Matches the Lagrangian
/// particle experience to first order — this is the standard MPM friction model.
///
/// `friction_coefficient = 0.0` → pure slip (same as SlipBoundary).
/// `friction_coefficient = 1.0` → strong friction.
/// IRL µ values: rock-on-rock ≈ 0.6, wet clay ≈ 0.2, ice ≈ 0.05.
///
/// # Note
/// This is grid-level friction (applied to grid cell velocities during grid update),
/// which is the standard MPM friction model. It matches the Lagrangian particle
/// experience to first order but is not per-surface-element friction.
#[derive(Debug, Clone, Copy)]
pub struct FrictionBoundary {
    pub thickness: usize,
    /// Coulomb friction coefficient µ ∈ [0, 1].
    /// 0 = slip (no friction), 1 = strong friction (full tangential damping at normal speed).
    pub friction_coefficient: f32,
}

impl FrictionBoundary {
    pub fn new(thickness: usize, friction_coefficient: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&friction_coefficient),
            "friction_coefficient must be in [0.0, 1.0], got {friction_coefficient}"
        );
        Self {
            thickness,
            friction_coefficient,
        }
    }
}

impl BoundaryCondition for FrictionBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        let t = self.thickness;
        let hi = grid_res.saturating_sub(t + 1);
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let mu = self.friction_coefficient;

        if x < t {
            apply_coulomb_wall(velocity, Vec2::X, mu);
        }
        if x > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_X, mu);
        }
        if y < t {
            apply_coulomb_wall(velocity, Vec2::Y, mu);
        }
        if y > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_Y, mu);
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

/// Heightmap terrain boundary — arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[x]` in grid units for each x-column.
/// All grid cells at (x, y) with `y ≤ heights[x]` are treated as solid terrain.
/// The terrain surface normal is +Y (pointing up). Coulomb friction is applied on the
/// tangential (horizontal) velocity component at the surface.
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
    pub fn new(heights: Vec<f32>, friction: f32, wall_thickness: usize) -> Self {
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
            // Block downward (into terrain) velocity component.
            if velocity.y < 0.0 {
                let v_n = velocity.y.abs();
                velocity.y = 0.0;
                // Coulomb friction on tangential (horizontal) component.
                if self.friction > 0.0 {
                    let friction_impulse = self.friction * v_n;
                    let v_t = velocity.x.abs();
                    velocity.x = if v_t > friction_impulse {
                        velocity.x * (1.0 - friction_impulse / v_t)
                    } else {
                        0.0
                    };
                }
            }
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

/// Apply Coulomb wall friction along one wall face.
///
/// `outward_normal`: unit vector pointing away from the wall into the domain.
/// When the velocity has a component moving INTO the wall (v · outward_normal < 0),
/// zero the normal component and damp the tangential component by µ × |v_normal|.
fn apply_coulomb_wall(velocity: &mut Vec2, outward_normal: Vec2, mu: f32) {
    let v_n_scalar = velocity.dot(outward_normal);
    // Only act when moving into the wall.
    if v_n_scalar >= 0.0 {
        return;
    }
    let normal_speed = v_n_scalar.abs();
    let v_t = *velocity - v_n_scalar * outward_normal;
    let v_t_len = v_t.length();
    let friction_impulse = mu * normal_speed;
    // Tangential speed after friction: max(|v_t| − µ|v_n|, 0), direction preserved.
    *velocity = if v_t_len > friction_impulse {
        v_t * ((v_t_len - friction_impulse) / v_t_len)
    } else {
        Vec2::ZERO
    };
}

fn apply_slip_wall_velocity(
    thickness: usize,
    cell_index: usize,
    grid_res: usize,
    velocity: &mut Vec2,
) {
    let hi = grid_res - (thickness + 1);
    let x = cell_index / grid_res;
    let y = cell_index % grid_res;
    // Only block the inward component — let outward (escape) velocity pass through.
    // Standard MPM slip: no-penetration, free tangential slip.
    if x < thickness {
        velocity.x = velocity.x.max(0.0);
    }
    if x > hi {
        velocity.x = velocity.x.min(0.0);
    }
    if y < thickness {
        velocity.y = velocity.y.max(0.0);
    }
    if y > hi {
        velocity.y = velocity.y.min(0.0);
    }
}

fn clamp_position_inside_grid(thickness: usize, position: Vec2, grid_res: usize) -> Vec2 {
    let min = thickness.saturating_sub(1) as f32;
    let max = grid_res.saturating_sub(thickness) as f32;
    position.clamp(Vec2::splat(min), Vec2::splat(max))
}

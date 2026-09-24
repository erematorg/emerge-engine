use glam::Vec2;

use super::{BoundaryCondition, apply_slip_wall_velocity};

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
    pub const fn new(thickness: usize, predictive_wall_min: f32) -> Self {
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

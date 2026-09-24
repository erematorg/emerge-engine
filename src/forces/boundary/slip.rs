use glam::Vec2;

use super::{BoundaryCondition, apply_slip_wall_velocity, clamp_position_inside_grid};

#[derive(Debug, Clone, Copy)]
pub struct SlipBoundary {
    pub thickness: usize,
}

impl SlipBoundary {
    pub const fn new(thickness: usize) -> Self {
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

    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        true
    }
}

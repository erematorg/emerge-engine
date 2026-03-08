use glam::Vec2;

use crate::state::particle::Particle;

pub trait BoundaryCondition: Send + Sync + core::fmt::Debug {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2);
    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2;
    fn post_g2p_particle(&self, _particle: &mut Particle, _grid_res: usize) {}
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
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }

    fn post_g2p_particle(&self, particle: &mut Particle, grid_res: usize) {
        let wall_min = self.predictive_wall_min;
        let wall_max = (grid_res as f32 - 1.0) - wall_min;
        let next_pos = particle.x + particle.v;

        if next_pos.x < wall_min {
            particle.v.x += wall_min - next_pos.x;
        }
        if next_pos.x > wall_max {
            particle.v.x += wall_max - next_pos.x;
        }
        if next_pos.y < wall_min {
            particle.v.y += wall_min - next_pos.y;
        }
        if next_pos.y > wall_max {
            particle.v.y += wall_max - next_pos.y;
        }
    }
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
    if x < thickness || x > hi {
        velocity.x = 0.0;
    }
    if y < thickness || y > hi {
        velocity.y = 0.0;
    }
}

fn clamp_position_inside_grid(thickness: usize, position: Vec2, grid_res: usize) -> Vec2 {
    let min = thickness.saturating_sub(1) as f32;
    let max = grid_res.saturating_sub(thickness) as f32;
    position.clamp(Vec2::splat(min), Vec2::splat(max))
}

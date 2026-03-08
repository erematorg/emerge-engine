use glam::IVec2;

use crate::solver::transfer::scatter_particle_mass;
use crate::state::{grid::Grid, kernel::quadratic_weights, particle::Particle};

pub fn estimate_initial_particle_volumes(particles: &mut [Particle], grid: &mut Grid) {
    estimate_density_and_volume_impl(particles, grid, true);
}

pub fn estimate_particle_density_and_volume(particles: &mut [Particle], grid: &mut Grid) {
    estimate_density_and_volume_impl(particles, grid, false);
}

fn estimate_density_and_volume_impl(
    particles: &mut [Particle],
    grid: &mut Grid,
    write_initial_volume: bool,
) {
    grid.clear();
    scatter_particle_mass(particles, grid);

    for p in particles.iter_mut() {
        let weights = quadratic_weights(p.x);
        let mut density = 0.0;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                density += grid.mass_at(cell_pos) * weight;
            }
        }

        if density > f32::EPSILON {
            p.density = density;
            p.volume = p.mass / density;
            if write_initial_volume {
                p.initial_volume = p.volume;
            }
        }
    }
}

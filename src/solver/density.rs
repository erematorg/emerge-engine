use glam::IVec2;

use crate::transfer::scatter_particle_mass;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// Export the mass-density field as a flat `grid_res × grid_res` buffer.
///
/// Each cell value is Σ(w_ij · mass_j) — the same mass accumulation used internally
/// by P2G. Values are NOT normalized by cell volume; callers can divide by
/// `grid_cell_size²` if physical units are needed.
///
/// # Use case — LP metaball surface rendering
/// Call once per render frame (not per substep) after `solver.step()`.
/// Upload the result to a `wgpu::Texture` and threshold in a fragment shader
/// to get a particle density surface.
///
/// Layout: column-major, index = x * grid_res + y — matches mechanics grid.
pub fn compute_density_grid(particles: &Particles, grid_res: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; grid_res * grid_res];
    let res = grid_res as i32;
    for i in 0..particles.len() {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let w = quadratic_weights(x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                    continue;
                }
                let weight = w.wx[gx as usize] * w.wy[gy as usize];
                buf[(cell.x * res + cell.y) as usize] += weight * mass;
            }
        }
    }
    buf
}

pub fn estimate_initial_particle_volumes(particles: &mut Particles, grid: &mut Grid) {
    estimate_density_and_volume_impl(particles, grid, true);
}

pub fn estimate_particle_density_and_volume(particles: &mut Particles, grid: &mut Grid) {
    estimate_density_and_volume_impl(particles, grid, false);
}

fn estimate_density_and_volume_impl(
    particles: &mut Particles,
    grid: &mut Grid,
    write_initial_volume: bool,
) {
    grid.clear();
    scatter_particle_mass(particles, grid);

    for i in 0..particles.len() {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        let mut density = 0.0;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                density += grid.mass_at(cell_pos) * weight;
            }
        }

        if density > f32::EPSILON {
            particles.density[i] = density;
            particles.volume[i] = mass / density;
            if write_initial_volume {
                particles.initial_volume[i] = particles.volume[i];
            }
        }
    }
}

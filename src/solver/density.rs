use glam::{IVec2, Vec2};

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

/// Compute density and volume for `count` particles (scatter + gather).
/// Pass `write_initial = true` at spawn time to also set `initial_volume`.
pub fn estimate_particle_volumes(
    particles: &mut Particles,
    grid: &mut Grid,
    count: usize,
    write_initial: bool,
) {
    grid.clear();
    scatter_particle_mass(particles, grid, count);

    for i in 0..count {
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
            if write_initial {
                particles.initial_volume[i] = particles.volume[i];
            }
        }
    }
}

/// Compute density and volume only for particles in `[new_start..active_count]`.
///
/// Scatters only particles whose positions fall within the AABB of the new group
/// expanded by 3 grid cells (the quadratic B-spline influence radius). All other
/// active particles are ignored — their density contribution to the new group is zero.
///
/// O(active_count) scan but O(local × stencil) grid work — fast for sparse spawns.
pub fn estimate_particle_volumes_local(
    particles: &mut Particles,
    grid: &mut Grid,
    active_count: usize,
    new_start: usize,
    write_initial: bool,
) {
    if new_start >= active_count {
        return;
    }

    // AABB of new particles in grid coords.
    let mut lo = Vec2::splat(f32::MAX);
    let mut hi = Vec2::splat(f32::MIN);
    for i in new_start..active_count {
        lo = lo.min(particles.x[i]);
        hi = hi.max(particles.x[i]);
    }
    // Expand by 3 cells: quadratic stencil reaches 1.5 cells per side,
    // and we need to capture particles that contribute mass to those cells.
    const MARGIN: f32 = 3.0;
    lo -= Vec2::splat(MARGIN);
    hi += Vec2::splat(MARGIN);

    grid.clear();
    for i in 0..active_count {
        let x = particles.x[i];
        if x.x < lo.x || x.y < lo.y || x.x > hi.x || x.y > hi.y {
            continue;
        }
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * mass, Vec2::ZERO);
            }
        }
    }

    for i in new_start..active_count {
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
            if write_initial {
                particles.initial_volume[i] = particles.volume[i];
            }
        }
    }
}

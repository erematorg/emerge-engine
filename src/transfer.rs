use glam::{IVec2, Mat2, Vec2};

use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::solver::config::KERNEL_D_INVERSE;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// P2G: scatter particle mass, momentum, and stress forces onto the grid (MLS-MPM, Hu 2018 §4).
///
/// Stress is pre-integrated as a momentum impulse so the grid needs one accumulation pass.
/// The APIC affine term conserves angular momentum without a correction step.
pub fn scatter_particles_to_grid(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
) {
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let tau = material.kirchhoff_stress(particles, i);
        let coeff = material.activation_scale();
        let stress = if particles.activation[i] > 0.0 && coeff > 0.0 {
            let n = particles.activation_dir[i];
            let len_sq = n.dot(n);
            let tau_active = if len_sq > f32::EPSILON {
                let n0 = n / len_sq.sqrt();
                let n_outer = Mat2::from_cols(n0 * n0.x, n0 * n0.y);
                let a_mat = n_outer * (particles.activation[i] * coeff);
                let f = particles.deformation_gradient[i];
                f * a_mat * f.transpose()
            } else {
                Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
            };
            tau + tau_active
        } else {
            tau
        };
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                grid.add_mass_momentum(cell_pos, weight * mass_i, momentum);
            }
        }
    }
}

/// G2P: read grid velocities back into particles, advance state, apply boundaries.
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    vel_limit: f32,
    apic_blend: f32,
    active_count: usize,
) -> usize {
    let mut clamp_count = 0usize;
    let grid_res = grid.resolution();
    for i in 0..active_count {
        let x = particles.x[i];
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);

        let weights = quadratic_weights(x);
        let mut v = Vec2::ZERO;
        let mut b = Mat2::ZERO;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let weighted_velocity = grid.velocity_at(cell_pos) * weight;
                let term = Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                b += term;
                v += weighted_velocity;
            }
        }

        // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
        // This fires only when CFL is violated despite the timestep limiter (e.g. first
        // substep of a high-energy spawn). Magnitude clamp preserves direction; no
        // anisotropic bias unlike per-component clamping.
        let spd = v.length();
        if spd > vel_limit {
            v *= vel_limit / spd;
            clamp_count += 1;
        }

        particles.v[i] = v;
        particles.velocity_gradient[i] = b * KERNEL_D_INVERSE * apic_blend;

        // Apply all boundaries in order: position clamp then post-G2P hook.
        let mut new_pos = x + v * dt;
        for boundary in boundaries.iter() {
            new_pos = boundary.clamp_particle_position(new_pos, grid_res);
        }
        particles.x[i] = new_pos;

        material.update_particle(particles, i, dt);
        for boundary in boundaries.iter() {
            boundary.post_g2p_particle(particles, i, grid_res, dt);
        }
    }
    clamp_count
}

pub fn scatter_particle_mass(particles: &Particles, grid: &mut Grid, active_count: usize) {
    for i in 0..active_count {
        let x = particles.x[i];
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
}

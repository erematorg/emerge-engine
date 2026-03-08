use glam::{IVec2, Mat2, Vec2};

use crate::solver::{boundary::BoundaryCondition, material_registry::MaterialRegistry};
use crate::state::{grid::Grid, kernel::quadratic_weights, particle::Particle};

/// P2G: scatter particle mass, momentum, and internal stress forces onto the grid.
///
/// Stress is pre-integrated as a momentum impulse (`-volume * D_inv * dt * σ * r`)
/// so the grid needs only one accumulation pass before normalization.
/// The APIC affine term (`C * cell_dist`) gives the grid a locally-varying velocity field,
/// which is what lets MLS-MPM conserve angular momentum without an extra correction step.
pub fn scatter_particles_to_grid(
    particles: &[Particle],
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    d_inverse: f32,
) {
    for p in particles.iter().copied() {
        let material = materials.get(p.material_id);
        let weights = quadratic_weights(p.x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - p.x + Vec2::splat(0.5);
                let mass_contrib = weight * p.mass;
                let velocity_contrib = p.v + p.affine * cell_dist;
                grid.add_mass_momentum(cell_pos, mass_contrib, mass_contrib * velocity_contrib);
                let stress = material.kirchhoff_stress(&p);
                let volume = material.stress_volume(&p);
                let stress_momentum = (-volume * d_inverse * dt) * (stress * cell_dist) * weight;
                grid.add_mass_momentum(cell_pos, 0.0, stress_momentum);
            }
        }
    }
}

/// G2P: read grid velocities back into particles, then advance each particle's state.
///
/// The B-matrix accumulation (`b += w * v_grid ⊗ dist`) reconstructs the local velocity
/// gradient. Multiplied by D_inv it gives the APIC C matrix, which the next P2G will use
/// to smear a richer velocity field — this is the key accuracy gain over standard FLIP/PIC.
pub fn gather_grid_to_particles(
    particles: &mut [Particle],
    grid: &Grid,
    dt: f32,
    boundary: &dyn BoundaryCondition,
    materials: &MaterialRegistry,
    d_inverse: f32,
) {
    for p in particles.iter_mut() {
        let material = materials.get(p.material_id);
        p.v = Vec2::ZERO;
        let weights = quadratic_weights(p.x);
        let mut b = Mat2::ZERO;

        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let dist = cell_pos.as_vec2() - p.x + Vec2::splat(0.5);
                let weighted_velocity = grid.velocity_at(cell_pos) * weight;
                let term = Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                b += term;
                p.v += weighted_velocity;
            }
        }

        p.affine = b * d_inverse;
        p.x = boundary.clamp_particle_position(p.x + p.v * dt, grid.resolution());
        material.update_particle(p, dt);
        boundary.post_g2p_particle(p, grid.resolution());
    }
}

pub fn scatter_particle_mass(particles: &[Particle], grid: &mut Grid) {
    for p in particles.iter().copied() {
        let weights = quadratic_weights(p.x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * p.mass, Vec2::ZERO);
            }
        }
    }
}

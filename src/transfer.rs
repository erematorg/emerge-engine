use glam::{IVec2, Mat2, Vec2};

use crate::solver::config::KERNEL_D_INVERSE;
use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// P2G: scatter particle mass, momentum, and internal stress forces onto the grid.
///
/// Implements the MLS-MPM transfer operator from Hu et al. 2018
/// "A Moving Least Squares Material Point Method with Displacement Discontinuity
/// and Two-Way Rigid Body Coupling" (SIGGRAPH 2018), §4.
///
/// Key constants (verified against paper and sparkl):
///   D_inv = 4.0  (quadratic B-spline: D = h²/4, D_inv = 4/h² with h=1)
///   force_scale = -D_inv  (negative: force opposes stress gradient)
///
/// Stress is pre-integrated as a momentum impulse (`-volume * D_inv * dt * σ * r`)
/// so the grid needs only one accumulation pass before normalization.
/// The APIC affine term (`C * cell_dist`) gives the grid a locally-varying velocity field,
/// which is what lets MLS-MPM conserve angular momentum without an extra correction step.
pub fn scatter_particles_to_grid(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
) {
    for i in 0..particles.len() {
        let p = particles.get(i);
        let material = materials.get(p.material_id);
        let weights = quadratic_weights(p.x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                // cell_dist: vector from particle to cell center, in grid coordinates.
                // +0.5 because cells are centered at (i+0.5, j+0.5) on a collocated grid
                // (particles live in [0, grid_res], cells cover unit squares around integer nodes).
                let cell_dist = cell_pos.as_vec2() - p.x + Vec2::splat(0.5);
                let mass_contrib = weight * p.mass;
                let velocity_contrib = p.v + p.velocity_gradient * cell_dist;
                grid.add_mass_momentum(cell_pos, mass_contrib, mass_contrib * velocity_contrib);
                let stress = {
                    let tau = material.kirchhoff_stress(&p);
                    let coeff = material.activation_scale();
                    if p.activation > 0.0 && coeff > 0.0 {
                        let n = p.activation_dir;
                        let len_sq = n.dot(n);
                        let tau_active = if len_sq > f32::EPSILON {
                            // Directional active stress — SoftZoo / DiffTaichi formulation.
                            // A = activation · coeff · (n₀ ⊗ n₀)  in reference (material) frame.
                            // τ_active = F · A · Fᵀ  pushes A forward into current config.
                            // Contracts along fiber n₀, follows body deformation automatically.
                            // Reference: Hu et al. 2019 ChainQueen; Wang et al. 2022 SoftZoo.
                            let n0 = n / len_sq.sqrt();
                            let n_outer = Mat2::from_cols(n0 * n0.x, n0 * n0.y);
                            let a_mat = n_outer * (p.activation * coeff);
                            let f = p.deformation_gradient;
                            f * a_mat * f.transpose()
                        } else {
                            // No fiber direction set — isotropic contractile fallback.
                            Mat2::from_diagonal(glam::Vec2::splat(p.activation * coeff))
                        };
                        tau + tau_active
                    } else {
                        tau
                    }
                };
                let volume = material.stress_volume(&p);
                let stress_momentum =
                    (-volume * KERNEL_D_INVERSE * dt) * (stress * cell_dist) * weight;
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
///
/// Serial implementation. Future: `par_iter_mut()` for particle counts > 100k.
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    vel_limit: f32,
    apic_blend: f32,
) -> usize {
    let mut clamp_count = 0usize;
    let grid_res = grid.resolution();
    for i in 0..particles.len() {
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

        // material.update_particle and post_g2p_particle need a Particle view.
        let mut p = particles.get(i);
        material.update_particle(&mut p, dt);
        for boundary in boundaries.iter() {
            boundary.post_g2p_particle(&mut p, grid_res, dt);
        }
        particles.set(i, p);
    }
    clamp_count
}

pub fn scatter_particle_mass(particles: &Particles, grid: &mut Grid) {
    for i in 0..particles.len() {
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

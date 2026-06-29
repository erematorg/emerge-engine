use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{ConstitutiveModel, MaterialModel};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::{grid::Grid, grid::kernel::quadratic_weights, particle::Particles};

/// Elastic/plastic Kirchhoff stress plus the active-stress (muscle contraction) term, if any.
///
/// Single source of truth for "what stress does this particle contribute to P2G" — shared by
/// `scatter_particles_to_grid` and tests, so the two can never drift apart. Mirrors the GPU
/// shader's post-switch active-stress block in `p2g.wgsl` exactly: Viscoelastic uses an
/// isotropic contractile term (matches its own Kelvin-Voigt formulation), every other elastic
/// model uses the directional F·(n₀⊗n₀)·Fᵀ fiber form (follows material deformation).
pub(crate) fn combined_kirchhoff_stress(
    material: &dyn MaterialModel,
    particles: &Particles,
    i: usize,
) -> Mat2 {
    let tau = material.kirchhoff_stress(particles, i);
    let coeff = material.activation_scale();
    if particles.activation[i] <= 0.0 || coeff <= 0.0 {
        return tau;
    }
    let isotropic = material.constitutive_model() == ConstitutiveModel::Viscoelastic;
    let tau_active = if isotropic {
        Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
    } else {
        let n = particles.activation_dir[i];
        let len_sq = n.dot(n);
        if len_sq > f32::EPSILON {
            let n0 = n / len_sq.sqrt();
            let n_outer = Mat2::from_cols(n0 * n0.x, n0 * n0.y);
            let a_mat = n_outer * (particles.activation[i] * coeff);
            let f = particles.deformation_gradient[i];
            f * a_mat * f.transpose()
        } else {
            Mat2::from_diagonal(Vec2::splat(particles.activation[i] * coeff))
        }
    };
    tau + tau_active
}

/// P2G: scatter particle mass, momentum, and stress forces onto the grid (MLS-MPM, Hu 2018 §4).
///
/// Stress is pre-integrated as a momentum impulse so the grid needs one accumulation pass.
/// The APIC affine term conserves angular momentum without a correction step.
///
/// NOT parallelized (unlike G2P below): multiple particles write to the same grid cell (3×3
/// B-spline stencils overlap), so summing their contributions requires either a shared mutable
/// map (unsound across threads — `HashMap::entry()` can trigger a resize) or a thread-local
/// fold/reduce merge. The latter was attempted and reverted 2026-06-20: it's safe and compiles
/// clean, but changes floating-point summation order across particles sharing a cell, and that
/// shifted results enough to break `fluid_spreads_more_than_elastic_under_gravity` (a 600-step
/// chaotic simulation) — confirmed by isolated A/B, not assumed. Reverted rather than accepted
/// the correctness risk for an unmeasured gain.
///
/// Parallelized via rayon `fold`/`reduce`: multiple particles write to the same grid cell
/// (3×3 B-spline stencils overlap), so a single shared `HashMap` cannot be mutated
/// concurrently — even with disjoint keys, `HashMap::entry()` can trigger an internal bucket
/// resize, which is unsound across threads without unsafe code or a concurrent map crate.
/// Instead, each rayon task accumulates into its own thread-local `CellMap` (no cross-thread
/// aliasing), and `reduce` merges those local maps pairwise — each merge step owns both maps
/// exclusively, so it's plain safe Rust. The final merge into the real `Grid` is sequential
/// and O(touched cells), not O(particles).
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

        let stress = combined_kirchhoff_stress(material, particles, i);
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight
                    * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                grid.add_mass_momentum(cell_pos, weight * mass_i, momentum);
            }
        }
    }
}

pub struct G2PParams {
    pub vel_limit: f32,
    pub apic_blend: f32,
    pub active_count: usize,
}

/// G2P: read grid velocities back into particles, advance state, apply boundaries.
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    params: G2PParams,
) -> usize {
    let G2PParams {
        vel_limit,
        apic_blend,
        active_count,
    } = params;
    let grid_res = grid.resolution();

    // Phase 1 (parallel): grid gather -> v, velocity_gradient, position advance + boundary
    // position clamp. Pure math over read-only grid/boundary state, writing only the calling
    // particle's own x/v/velocity_gradient — no cross-particle data dependency, so disjoint
    // per-field slices can be processed concurrently (gather passes are race-free by
    // construction; see Gao et al. 2018, "GPU Optimization of Material Point Methods").
    let xs = &mut particles.x[..active_count];
    let vs = &mut particles.v[..active_count];
    let vgs = &mut particles.velocity_gradient[..active_count];

    let clamp_count: usize = xs
        .par_iter_mut()
        .zip(vs.par_iter_mut())
        .zip(vgs.par_iter_mut())
        .map(|((x, v), vg)| {
            let weights = quadratic_weights(*x);
            let mut new_v = Vec2::ZERO;
            let mut b = Mat2::ZERO;

            for gx in 0..3 {
                for gy in 0..3 {
                    let weight = weights.wx[gx] * weights.wy[gy];
                    let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    let dist = cell_pos.as_vec2() - *x + Vec2::splat(0.5);
                    let weighted_velocity = grid.velocity_at(cell_pos) * weight;
                    let term =
                        Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                    b += term;
                    new_v += weighted_velocity;
                }
            }

            // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
            // This fires only when CFL is violated despite the timestep limiter (e.g. first
            // substep of a high-energy spawn). Magnitude clamp preserves direction; no
            // anisotropic bias unlike per-component clamping.
            let spd = new_v.length();
            let clamped = if spd > vel_limit {
                new_v *= vel_limit / spd;
                1
            } else {
                0
            };

            // Apply all boundaries' position clamp (pure function, no particle-struct access).
            let mut new_pos = *x + new_v * dt;
            for boundary in boundaries.iter() {
                new_pos = boundary.clamp_particle_position(new_pos, grid_res);
            }

            *v = new_v;
            *vg = b * KERNEL_D_INVERSE * apic_blend;
            *x = new_pos;
            clamped
        })
        .sum();

    // Phase 2 (sequential): plasticity update + boundary post-hooks need whole-`Particles`
    // mutable access (deformation_gradient, hardening_scale, etc. per material) — not
    // split-borrow-friendly without a larger `MaterialModel` trait redesign, so kept sequential.
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
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

#[cfg(test)]
mod activation_tests {
    use super::combined_kirchhoff_stress;
    use crate::materials::{NeoHookeanMaterial, ViscoelasticMaterial};
    use crate::particle::{Particle, Particles};
    use glam::{Mat2, Vec2};

    fn particle_at_rest() -> Particle {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p.deformation_gradient = Mat2::IDENTITY; // undeformed: passive elastic stress is exactly zero
        p
    }

    /// Directional materials (everything except Viscoelastic): active stress follows the fiber
    /// direction exactly — `activation * coeff` along the fiber axis, zero perpendicular to it.
    #[test]
    fn directional_active_stress_follows_fiber_axis() {
        let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);

        assert!(
            (tau.x_axis.x - 10.0).abs() < 1e-5,
            "tau_xx should be activation*coeff=10: {tau:?}"
        );
        assert!(
            tau.y_axis.y.abs() < 1e-5,
            "tau_yy should stay ~0 (perpendicular to fiber): {tau:?}"
        );
    }

    /// Viscoelastic uses an isotropic active term (matches its Kelvin-Voigt formulation and the
    /// GPU shader's `model == 9u` special case) — equal on both diagonal axes, regardless of
    /// `activation_dir`.
    #[test]
    fn viscoelastic_active_stress_is_isotropic() {
        let mut mat = ViscoelasticMaterial::new(100.0, 200.0, 0.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X; // must NOT bias the result toward x for this material

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);

        assert!(
            (tau.x_axis.x - 10.0).abs() < 1e-5,
            "tau_xx should be activation*coeff=10: {tau:?}"
        );
        assert!(
            (tau.y_axis.y - 10.0).abs() < 1e-5,
            "tau_yy should equal tau_xx (isotropic, not directional): {tau:?}"
        );
    }

    /// Regression: `ViscoelasticMaterial::kirchhoff_stress` used to add its own isotropic active
    /// term directly AND report a non-zero `activation_scale()`, so the shared P2G path
    /// (`combined_kirchhoff_stress`) added a second active term on top — silently doubling muscle
    /// stress for any Viscoelastic creature body. Pin the total to exactly one contribution.
    #[test]
    fn viscoelastic_active_stress_is_not_double_counted() {
        let mut mat = ViscoelasticMaterial::new(100.0, 200.0, 0.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 1.0;
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);
        let expected_single = 10.0; // activation(1.0) * coeff(10.0), applied exactly once
        assert!(
            (tau.x_axis.x - expected_single).abs() < 1e-5,
            "active stress must be applied exactly once, not doubled: tau_xx={}, expected={expected_single}",
            tau.x_axis.x
        );
    }

    #[test]
    fn zero_activation_leaves_stress_unchanged() {
        let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
        mat.active_stress_coeff = 10.0;
        let mut p = particle_at_rest();
        p.activation = 0.0; // off — must be a true no-op regardless of coeff
        p.activation_dir = Vec2::X;

        let soa = Particles::from(vec![p]);
        let tau = combined_kirchhoff_stress(&mat, &soa, 0);
        assert!(
            tau.x_axis.x.abs() < 1e-6 && tau.y_axis.y.abs() < 1e-6,
            "activation=0.0 must produce zero stress on an undeformed particle: {tau:?}"
        );
    }
}

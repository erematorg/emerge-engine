//! P2G test suite for `transfer.rs` -- split out of the original combined
//! `transfer_tests.rs` (2026-07-19, mirrors `transfer.rs`'s own P2G/G2P
//! phase split) by pure mechanical line-range extraction, not retyped, to
//! eliminate transcription risk in adjoint math this precise (every VJP
//! here is verified against central-difference numerical gradients --
//! exactly the code where a silent copy error would be hardest to notice).

use super::*;

#[cfg(test)]
mod p2g_stress_vjp_tests {
    use super::*;

    /// Recomputes just the stress-scatter contribution `scatter_particles_to_grid`
    /// itself computes for each of the 9 stencil cells, at a given `stress` --
    /// the exact forward formula `p2g_stress_vjp` is the adjoint of, isolated
    /// from mass/velocity/C so the finite-difference check exercises only the
    /// piece being verified.
    fn stress_contributions(x: Vec2, stress_coeff: f32, stress: Mat2) -> [[Vec2; 3]; 3] {
        let weights = quadratic_weights(x);
        let mut out = [[Vec2::ZERO; 3]; 3];
        for (gx, row) in out.iter_mut().enumerate() {
            for (gy, cell) in row.iter_mut().enumerate() {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                *cell = weight * stress_coeff * (stress * cell_dist);
            }
        }
        out
    }

    /// Scalar loss L(stress) = sum_c g_c . contribution_c(stress) -- the
    /// standard way to check a matrix-to-many-vectors function one scalar
    /// component at a time via central differences.
    fn loss(x: Vec2, stress_coeff: f32, stress: Mat2, g: &[[Vec2; 3]; 3]) -> f32 {
        let contributions = stress_contributions(x, stress_coeff, stress);
        let mut total = 0.0;
        for (row, contrib_row) in g.iter().zip(contributions.iter()) {
            for (gv, cv) in row.iter().zip(contrib_row.iter()) {
                total += gv.dot(*cv);
            }
        }
        total
    }

    /// Bundles the fixed context (position, stress_coeff, base stress, incoming
    /// gradients, step size) shared by every one of F's 4 components' checks.
    struct FiniteDiffContext {
        x: Vec2,
        stress_coeff: f32,
        stress: Mat2,
        g: [[Vec2; 3]; 3],
        h: f32,
    }

    fn check_component(
        ctx: &FiniteDiffContext,
        label: &str,
        analytic_val: f32,
        base: f32,
        set: impl Fn(&mut Mat2, f32),
    ) {
        let mut s_plus = ctx.stress;
        set(&mut s_plus, base + ctx.h);
        let mut s_minus = ctx.stress;
        set(&mut s_minus, base - ctx.h);

        let numeric = (loss(ctx.x, ctx.stress_coeff, s_plus, &ctx.g)
            - loss(ctx.x, ctx.stress_coeff, s_minus, &ctx.g))
            / (2.0 * ctx.h);

        let diff = (numeric - analytic_val).abs();
        let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "p2g_stress_vjp mismatch at {label}: analytic={analytic_val:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e} (x={:?})",
            diff / scale,
            ctx.x
        );
    }

    fn check_matches_finite_difference(
        x: Vec2,
        stress_coeff: f32,
        stress: Mat2,
        g: [[Vec2; 3]; 3],
    ) {
        let analytic = p2g_stress_vjp(x, stress_coeff, &g);
        let ctx = FiniteDiffContext {
            x,
            stress_coeff,
            stress,
            g,
            h: 1.0e-3_f32,
        };

        check_component(
            &ctx,
            "F[0][0]",
            analytic.x_axis.x,
            stress.x_axis.x,
            |m, v| m.x_axis.x = v,
        );
        check_component(
            &ctx,
            "F[1][0]",
            analytic.x_axis.y,
            stress.x_axis.y,
            |m, v| m.x_axis.y = v,
        );
        check_component(
            &ctx,
            "F[0][1]",
            analytic.y_axis.x,
            stress.y_axis.x,
            |m, v| m.y_axis.x = v,
        );
        check_component(
            &ctx,
            "F[1][1]",
            analytic.y_axis.y,
            stress.y_axis.y,
            |m, v| m.y_axis.y = v,
        );
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check_matches_finite_difference(
            Vec2::new(10.0, 10.0),
            -0.5,
            Mat2::from_cols(Vec2::new(3.0, 0.5), Vec2::new(0.5, -2.0)),
            [[Vec2::new(1.0, 0.5); 3]; 3],
        );
    }

    #[test]
    fn matches_finite_difference_off_center_with_varied_gradients() {
        // Off-center position (nonzero fractional offset within its cell) and a
        // different, non-uniform incoming gradient per stencil cell -- exercises
        // real per-cell weight/cell_dist variation, not just a symmetric case.
        let g = [
            [
                Vec2::new(0.3, -0.7),
                Vec2::new(1.1, 0.2),
                Vec2::new(-0.4, 0.9),
            ],
            [
                Vec2::new(0.8, 0.1),
                Vec2::new(-0.2, -0.5),
                Vec2::new(0.6, 1.3),
            ],
            [
                Vec2::new(-1.0, 0.4),
                Vec2::new(0.2, -0.9),
                Vec2::new(0.5, 0.5),
            ],
        ];
        check_matches_finite_difference(
            Vec2::new(15.35, 22.78),
            0.8,
            Mat2::from_cols(Vec2::new(-1.5, 2.0), Vec2::new(0.9, 1.2)),
            g,
        );
    }

    #[test]
    fn chains_correctly_into_neohookean_kirchhoff_stress_vjp() {
        // Real end-to-end check: P2G's stress gradient feeds NeoHookean's own
        // F-adjoint, and the composed result still matches a finite-difference
        // taken all the way from F, through stress, through the P2G scatter --
        // proves the two pieces compose correctly, not just individually.
        use crate::materials::NeoHookeanMaterial;
        use crate::particle::{Particle, Particles};

        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let f = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.9));
        let x = Vec2::new(8.4, 12.6);
        let stress_coeff = -0.3;
        let g = [[Vec2::new(0.4, -0.6); 3]; 3];

        let particle_with_f = |f: Mat2| -> Particles {
            let mut particles = Particles::default();
            particles.push(Particle {
                x,
                v: Vec2::ZERO,
                velocity_gradient: Mat2::ZERO,
                deformation_gradient: f,
                mass: 1.0,
                initial_volume: 1.0,
                volume: 1.0,
                density: 1.0,
                material_id: 0,
                plastic_volume_ratio: 1.0,
                hardening_scale: 1.0,
                friction_hardening: 0.0,
                log_volume_strain: 0.0,
                temperature: 0.0,
                scalar_field: 0.0,
                user_tag: 0,
                activation: 0.0,
                activation_dir: Vec2::ZERO,
                muscle_group_id: 0,
                contact_group: 0,
                sleeping: 0,
                pinned: 0,
                _pad: 0,
            });
            particles
        };

        let end_to_end_loss = |f: Mat2| -> f32 {
            let particles = particle_with_f(f);
            let stress = mat.kirchhoff_stress(&particles, 0);
            let contributions = stress_contributions(x, stress_coeff, stress);
            let mut total = 0.0;
            for (row, contrib_row) in g.iter().zip(contributions.iter()) {
                for (gv, cv) in row.iter().zip(contrib_row.iter()) {
                    total += gv.dot(*cv);
                }
            }
            total
        };

        // Composed analytic gradient: P2G adjoint -> NeoHookean adjoint.
        let particles = particle_with_f(f);
        let stress = mat.kirchhoff_stress(&particles, 0);
        let d_loss_d_stress = p2g_stress_vjp(x, stress_coeff, &g);
        let composed = mat.kirchhoff_stress_vjp(&particles, 0, d_loss_d_stress);
        let _ = stress; // used only to construct d_loss_d_stress's context above

        let h = 1.0e-3_f32;
        let mut f_plus = f;
        f_plus.x_axis.x += h;
        let mut f_minus = f;
        f_minus.x_axis.x -= h;
        let numeric = (end_to_end_loss(f_plus) - end_to_end_loss(f_minus)) / (2.0 * h);

        let diff = (numeric - composed.x_axis.x).abs();
        let scale = numeric.abs().max(composed.x_axis.x.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "composed P2G+NeoHookean adjoint must match end-to-end finite difference: \
             analytic={:.6} numeric={numeric:.6} relative_diff={:.2e}",
            composed.x_axis.x,
            diff / scale
        );
    }
}

#[cfg(test)]
mod p2g_position_vjp_tests {
    use super::*;

    /// Forward function reconstructing scatter_particles_to_grid's EXACT
    /// per-cell formula (mass_contrib, momentum_contrib), taking the
    /// particle state directly instead of a real Particles/Grid -- isolates
    /// the position-dependence being verified from everything else.
    fn contributions(x: Vec2, state: &P2GParticleState) -> ([[Vec2; 3]; 3], [[f32; 3]; 3]) {
        let weights = quadratic_weights(x);
        let m = state.mass * state.c + state.stress_coeff * state.stress;
        let mut momentum = [[Vec2::ZERO; 3]; 3];
        let mut mass = [[0.0f32; 3]; 3];
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let a = state.mass * state.v + m * cell_dist;
                momentum[gx][gy] = weight * a;
                mass[gx][gy] = weight * state.mass;
            }
        }
        (momentum, mass)
    }

    fn loss(
        x: Vec2,
        state: &P2GParticleState,
        g_momentum: &[[Vec2; 3]; 3],
        g_mass: &[[f32; 3]; 3],
    ) -> f32 {
        let (momentum, mass) = contributions(x, state);
        let mut total = 0.0;
        for gx in 0..3 {
            for gy in 0..3 {
                total += g_momentum[gx][gy].dot(momentum[gx][gy]) + g_mass[gx][gy] * mass[gx][gy];
            }
        }
        total
    }

    fn check(x: Vec2, state: P2GParticleState, g_momentum: [[Vec2; 3]; 3], g_mass: [[f32; 3]; 3]) {
        let analytic = p2g_position_vjp(x, &state, &g_momentum, &g_mass);
        let h = 1.0e-3_f32;

        let numeric_x = (loss(x + Vec2::new(h, 0.0), &state, &g_momentum, &g_mass)
            - loss(x - Vec2::new(h, 0.0), &state, &g_momentum, &g_mass))
            / (2.0 * h);
        let numeric_y = (loss(x + Vec2::new(0.0, h), &state, &g_momentum, &g_mass)
            - loss(x - Vec2::new(0.0, h), &state, &g_momentum, &g_mass))
            / (2.0 * h);

        for (label, analytic_val, numeric) in
            [("x", analytic.x, numeric_x), ("y", analytic.y, numeric_y)]
        {
            let diff = (numeric - analytic_val).abs();
            let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "p2g_position_vjp mismatch at {label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e} (x={x:?})",
                diff / scale
            );
        }
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(
            Vec2::new(20.0, 20.0),
            P2GParticleState {
                mass: 1.5,
                v: Vec2::new(0.3, -0.2),
                c: Mat2::from_cols(Vec2::new(0.1, 0.05), Vec2::new(-0.05, 0.1)),
                stress: Mat2::from_cols(Vec2::new(3.0, 0.5), Vec2::new(0.5, -2.0)),
                stress_coeff: -0.4,
            },
            [[Vec2::new(0.6, -0.3); 3]; 3],
            [[0.2; 3]; 3],
        );
    }

    #[test]
    fn matches_finite_difference_off_center_with_varied_gradients() {
        let g_momentum = [
            [
                Vec2::new(0.4, -0.5),
                Vec2::new(0.9, 0.1),
                Vec2::new(-0.3, 0.6),
            ],
            [
                Vec2::new(0.6, 0.2),
                Vec2::new(-0.1, -0.4),
                Vec2::new(0.5, 0.9),
            ],
            [
                Vec2::new(-0.8, 0.3),
                Vec2::new(0.2, -0.7),
                Vec2::new(0.4, 0.4),
            ],
        ];
        let g_mass = [[0.3, -0.2, 0.5], [-0.4, 0.6, 0.1], [0.2, -0.3, 0.4]];
        check(
            Vec2::new(9.35, 21.78),
            P2GParticleState {
                mass: 0.8,
                v: Vec2::new(-0.5, 0.4),
                c: Mat2::from_cols(Vec2::new(-0.2, 0.15), Vec2::new(0.1, -0.25)),
                stress: Mat2::from_cols(Vec2::new(-1.5, 2.0), Vec2::new(0.9, 1.2)),
                stress_coeff: 0.6,
            },
            g_momentum,
            g_mass,
        );
    }
}

//! G2P test suite for `transfer.rs` -- split out of the original combined
//! `transfer_tests.rs` (2026-07-19, mirrors `transfer.rs`'s own P2G/G2P
//! phase split) by pure mechanical line-range extraction, not retyped, to
//! eliminate transcription risk in adjoint math this precise (every VJP
//! here is verified against central-difference numerical gradients --
//! exactly the code where a silent copy error would be hardest to notice).

use super::*;

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

#[cfg(test)]
mod g2p_velocity_vjp_tests {
    use super::*;

    /// Forward formula exactly matching G2P's own `new_v` computation (the
    /// weighted sum over the 3x3 stencil), taking the 9 grid velocities
    /// directly as an array instead of reading a real `Grid` -- isolates the
    /// weighted-sum math being verified from grid storage/lookup entirely.
    fn gather_velocity(x: Vec2, v_grid: &[[Vec2; 3]; 3]) -> Vec2 {
        let weights = quadratic_weights(x);
        let mut new_v = Vec2::ZERO;
        for (row, wx) in v_grid.iter().zip(weights.wx.iter()) {
            for (v_cell, wy) in row.iter().zip(weights.wy.iter()) {
                new_v += (wx * wy) * *v_cell;
            }
        }
        new_v
    }

    fn loss(x: Vec2, v_grid: &[[Vec2; 3]; 3], g: Vec2) -> f32 {
        g.dot(gather_velocity(x, v_grid))
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(Vec2::new(20.0, 20.0), Vec2::new(0.6, -0.4));
    }

    #[test]
    fn matches_finite_difference_off_center() {
        check(Vec2::new(7.35, 41.82), Vec2::new(-1.1, 0.9));
    }

    /// Checks every one of the 9 stencil cells' 2 velocity components (18
    /// scalars total) against central differences -- the full adjoint output,
    /// not just a sample of it.
    fn check(x: Vec2, g: Vec2) {
        let v_grid = [
            [
                Vec2::new(0.3, 0.1),
                Vec2::new(-0.2, 0.5),
                Vec2::new(0.7, -0.6),
            ],
            [
                Vec2::new(-0.4, 0.2),
                Vec2::new(0.1, -0.3),
                Vec2::new(0.5, 0.4),
            ],
            [
                Vec2::new(0.2, -0.5),
                Vec2::new(-0.6, 0.3),
                Vec2::new(0.4, 0.1),
            ],
        ];
        let analytic = g2p_velocity_vjp(x, g);
        let h = 1.0e-3_f32;

        for gx in 0..3 {
            for gy in 0..3 {
                for (axis, label) in [(0, "x"), (1, "y")] {
                    let mut v_plus = v_grid;
                    let mut v_minus = v_grid;
                    if axis == 0 {
                        v_plus[gx][gy].x += h;
                        v_minus[gx][gy].x -= h;
                    } else {
                        v_plus[gx][gy].y += h;
                        v_minus[gx][gy].y -= h;
                    }
                    let numeric = (loss(x, &v_plus, g) - loss(x, &v_minus, g)) / (2.0 * h);
                    let analytic_val = if axis == 0 {
                        analytic[gx][gy].x
                    } else {
                        analytic[gx][gy].y
                    };
                    let diff = (numeric - analytic_val).abs();
                    let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                    assert!(
                        diff / scale < 1.0e-2,
                        "g2p_velocity_vjp mismatch at cell[{gx}][{gy}].{label}: \
                         analytic={analytic_val:.6} numeric={numeric:.6} \
                         relative_diff={:.2e} (x={x:?})",
                        diff / scale
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod f_update_vjp_tests {
    use super::*;

    fn f_new(c: Mat2, f_old: Mat2, dt: f32) -> Mat2 {
        (Mat2::IDENTITY + dt * c) * f_old
    }

    fn loss(c: Mat2, f_old: Mat2, dt: f32, g: Mat2) -> f32 {
        let fnew = f_new(c, f_old, dt);
        g.x_axis.x * fnew.x_axis.x
            + g.x_axis.y * fnew.x_axis.y
            + g.y_axis.x * fnew.y_axis.x
            + g.y_axis.y * fnew.y_axis.y
    }

    /// Bundles the fixed context shared by every component check.
    struct FUpdateContext {
        c: Mat2,
        f_old: Mat2,
        dt: f32,
        g: Mat2,
        h: f32,
    }

    /// Central-difference check on one scalar component of either C or
    /// F_old (whichever `set`/`get` target), holding the other input fixed.
    fn check_one_component(
        ctx: &FUpdateContext,
        label: &str,
        analytic_val: f32,
        vary_c: bool,
        set: impl Fn(&mut Mat2, f32),
        get: impl Fn(Mat2) -> f32,
    ) {
        let (mut c_plus, mut f_plus) = (ctx.c, ctx.f_old);
        let (mut c_minus, mut f_minus) = (ctx.c, ctx.f_old);
        if vary_c {
            let base = get(ctx.c);
            set(&mut c_plus, base + ctx.h);
            set(&mut c_minus, base - ctx.h);
        } else {
            let base = get(ctx.f_old);
            set(&mut f_plus, base + ctx.h);
            set(&mut f_minus, base - ctx.h);
        }
        let numeric = (loss(c_plus, f_plus, ctx.dt, ctx.g) - loss(c_minus, f_minus, ctx.dt, ctx.g))
            / (2.0 * ctx.h);
        let diff = (numeric - analytic_val).abs();
        let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "f_update_vjp mismatch at {label}: analytic={analytic_val:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            diff / scale
        );
    }

    /// Checks all 4 scalar components of one input matrix (either C or
    /// F_old), holding the other fixed -- reused for both outputs of
    /// `f_update_vjp`.
    fn check_matrix_input(ctx: &FUpdateContext, label_prefix: &str, analytic: Mat2, vary_c: bool) {
        check_one_component(
            ctx,
            &format!("{label_prefix}[0][0]"),
            analytic.x_axis.x,
            vary_c,
            |m, v| m.x_axis.x = v,
            |m| m.x_axis.x,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[1][0]"),
            analytic.x_axis.y,
            vary_c,
            |m, v| m.x_axis.y = v,
            |m| m.x_axis.y,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[0][1]"),
            analytic.y_axis.x,
            vary_c,
            |m, v| m.y_axis.x = v,
            |m| m.y_axis.x,
        );
        check_one_component(
            ctx,
            &format!("{label_prefix}[1][1]"),
            analytic.y_axis.y,
            vary_c,
            |m, v| m.y_axis.y = v,
            |m| m.y_axis.y,
        );
    }

    fn check(c: Mat2, f_old: Mat2, dt: f32, g: Mat2) {
        let (d_loss_d_c, d_loss_d_f_old) = f_update_vjp(c, f_old, dt, g);
        let ctx = FUpdateContext {
            c,
            f_old,
            dt,
            g,
            h: 1.0e-3_f32,
        };
        check_matrix_input(&ctx, "d_loss_d_c", d_loss_d_c, true);
        check_matrix_input(&ctx, "d_loss_d_f_old", d_loss_d_f_old, false);
    }

    #[test]
    fn matches_finite_difference_small_dt() {
        check(
            Mat2::from_cols(Vec2::new(0.2, -0.1), Vec2::new(0.05, 0.15)),
            Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(-0.02, 0.95)),
            0.001,
            Mat2::from_cols(Vec2::new(0.6, -0.3), Vec2::new(0.4, 0.8)),
        );
    }

    #[test]
    fn matches_finite_difference_larger_dt_and_deformation() {
        check(
            Mat2::from_cols(Vec2::new(-0.5, 0.3), Vec2::new(0.2, 0.4)),
            Mat2::from_cols(Vec2::new(1.4, 0.2), Vec2::new(-0.15, 0.8)),
            0.05,
            Mat2::from_cols(Vec2::new(-0.7, 1.1), Vec2::new(0.9, -0.4)),
        );
    }
}

#[cfg(test)]
mod g2p_affine_vjp_tests {
    use super::*;

    /// Forward formula exactly matching G2P's own `new_c`/`velocity_gradient`
    /// computation (the weighted outer-product sum), taking the 9 grid
    /// velocities directly as an array instead of reading a real `Grid`.
    fn gather_affine(x: Vec2, v_grid: &[[Vec2; 3]; 3], scale: f32) -> Mat2 {
        let weights = quadratic_weights(x);
        let mut b = Mat2::ZERO;
        for (gx, (row, wx)) in v_grid.iter().zip(weights.wx.iter()).enumerate() {
            for (gy, (v_cell, wy)) in row.iter().zip(weights.wy.iter()).enumerate() {
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let weighted = *v_cell * (wx * wy);
                b += Mat2::from_cols(weighted * dist.x, weighted * dist.y);
            }
        }
        b * scale
    }

    fn loss(x: Vec2, v_grid: &[[Vec2; 3]; 3], scale: f32, g: Mat2) -> f32 {
        let c = gather_affine(x, v_grid, scale);
        g.x_axis.x * c.x_axis.x
            + g.x_axis.y * c.x_axis.y
            + g.y_axis.x * c.y_axis.x
            + g.y_axis.y * c.y_axis.y
    }

    #[test]
    fn matches_finite_difference_at_cell_center() {
        check(
            Vec2::new(30.0, 30.0),
            4.0,
            0.9,
            Mat2::from_cols(Vec2::new(0.5, -0.3), Vec2::new(0.2, 0.7)),
        );
    }

    #[test]
    fn matches_finite_difference_off_center() {
        check(
            Vec2::new(12.6, 5.9),
            4.0,
            0.75,
            Mat2::from_cols(Vec2::new(-0.4, 0.6), Vec2::new(1.0, -0.2)),
        );
    }

    /// Checks every one of the 9 stencil cells' 2 velocity components (18
    /// scalars total) against central differences.
    fn check(x: Vec2, kernel_d_inverse: f32, apic_blend: f32, g: Mat2) {
        let v_grid = [
            [
                Vec2::new(0.2, -0.4),
                Vec2::new(0.6, 0.1),
                Vec2::new(-0.3, 0.5),
            ],
            [
                Vec2::new(0.4, 0.3),
                Vec2::new(-0.1, -0.6),
                Vec2::new(0.2, 0.4),
            ],
            [
                Vec2::new(-0.5, 0.2),
                Vec2::new(0.3, -0.4),
                Vec2::new(0.1, 0.6),
            ],
        ];
        let scale = kernel_d_inverse * apic_blend;
        let analytic = g2p_affine_vjp(x, kernel_d_inverse, apic_blend, g);
        let h = 1.0e-3_f32;

        for gx in 0..3 {
            for gy in 0..3 {
                for (axis, label) in [(0, "x"), (1, "y")] {
                    let mut v_plus = v_grid;
                    let mut v_minus = v_grid;
                    if axis == 0 {
                        v_plus[gx][gy].x += h;
                        v_minus[gx][gy].x -= h;
                    } else {
                        v_plus[gx][gy].y += h;
                        v_minus[gx][gy].y -= h;
                    }
                    let numeric =
                        (loss(x, &v_plus, scale, g) - loss(x, &v_minus, scale, g)) / (2.0 * h);
                    let analytic_val = if axis == 0 {
                        analytic[gx][gy].x
                    } else {
                        analytic[gx][gy].y
                    };
                    let diff = (numeric - analytic_val).abs();
                    let scale_denom = numeric.abs().max(analytic_val.abs()).max(1.0);
                    assert!(
                        diff / scale_denom < 1.0e-2,
                        "g2p_affine_vjp mismatch at cell[{gx}][{gy}].{label}: \
                         analytic={analytic_val:.6} numeric={numeric:.6} \
                         relative_diff={:.2e} (x={x:?})",
                        diff / scale_denom
                    );
                }
            }
        }
    }

    /// Real end-to-end check: combines g2p_velocity_vjp and g2p_affine_vjp
    /// (the two halves of G2P's actual joint computation, gathered from the
    /// SAME 9 grid velocities in the same pass) and verifies the SUMMED
    /// gradient matches a finite difference taken through the true combined
    /// loss L = g_v . new_v + g_c : new_c -- proves the two adjoints compose
    /// correctly when G2P's real output (both v and C) feeds a real loss,
    /// not just that each is independently correct in isolation.
    #[test]
    fn composes_correctly_with_g2p_velocity_vjp() {
        let x = Vec2::new(18.3, 9.7);
        let kernel_d_inverse = 4.0;
        let apic_blend = 1.0;
        let scale = kernel_d_inverse * apic_blend;
        let g_v = Vec2::new(0.4, -0.6);
        let g_c = Mat2::from_cols(Vec2::new(0.3, 0.5), Vec2::new(-0.7, 0.2));

        let v_grid = [
            [
                Vec2::new(0.1, 0.2),
                Vec2::new(-0.3, 0.4),
                Vec2::new(0.5, -0.1),
            ],
            [
                Vec2::new(0.2, -0.2),
                Vec2::new(0.4, 0.3),
                Vec2::new(-0.4, 0.1),
            ],
            [
                Vec2::new(-0.1, 0.5),
                Vec2::new(0.2, -0.3),
                Vec2::new(0.3, 0.2),
            ],
        ];

        let combined_loss = |v_grid: &[[Vec2; 3]; 3]| -> f32 {
            let weights = quadratic_weights(x);
            let mut new_v = Vec2::ZERO;
            let mut b = Mat2::ZERO;
            for (gxi, (row, wx)) in v_grid.iter().zip(weights.wx.iter()).enumerate() {
                for (gyi, (v_cell, wy)) in row.iter().zip(weights.wy.iter()).enumerate() {
                    let weight = wx * wy;
                    let cell_pos = weights.base_cell + IVec2::new(gxi as i32 - 1, gyi as i32 - 1);
                    let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                    let weighted = *v_cell * weight;
                    new_v += weighted;
                    b += Mat2::from_cols(weighted * dist.x, weighted * dist.y);
                }
            }
            let new_c = b * scale;
            g_v.dot(new_v)
                + g_c.x_axis.x * new_c.x_axis.x
                + g_c.x_axis.y * new_c.x_axis.y
                + g_c.y_axis.x * new_c.y_axis.x
                + g_c.y_axis.y * new_c.y_axis.y
        };

        let from_v = g2p_velocity_vjp(x, g_v);
        let from_c = g2p_affine_vjp(x, kernel_d_inverse, apic_blend, g_c);

        let h = 1.0e-3_f32;
        // Check cell [1][1] (center of stencil) as a representative sample.
        let mut v_plus = v_grid;
        v_plus[1][1].x += h;
        let mut v_minus = v_grid;
        v_minus[1][1].x -= h;
        let numeric = (combined_loss(&v_plus) - combined_loss(&v_minus)) / (2.0 * h);

        let combined_analytic = from_v[1][1].x + from_c[1][1].x;
        let diff = (numeric - combined_analytic).abs();
        let scale_denom = numeric.abs().max(combined_analytic.abs()).max(1.0);
        assert!(
            diff / scale_denom < 1.0e-2,
            "composed g2p_velocity_vjp + g2p_affine_vjp must match end-to-end finite \
             difference: analytic={combined_analytic:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            diff / scale_denom
        );
    }
}

#[cfg(test)]
mod active_stress_vjp_tests {
    use super::*;

    fn tau_active(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2) -> Mat2 {
        let len_sq = fiber_dir.dot(fiber_dir);
        if len_sq <= f32::EPSILON || activation <= 0.0 || coeff <= 0.0 {
            return Mat2::ZERO;
        }
        let n0 = fiber_dir / len_sq.sqrt();
        let a_mat = Mat2::from_cols(n0 * n0.x, n0 * n0.y) * (activation * coeff);
        f * a_mat * f.transpose()
    }

    fn loss(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2, g: Mat2) -> f32 {
        let tau = tau_active(f, activation, coeff, fiber_dir);
        g.x_axis.x * tau.x_axis.x
            + g.x_axis.y * tau.x_axis.y
            + g.y_axis.x * tau.y_axis.x
            + g.y_axis.y * tau.y_axis.y
    }

    fn check(f: Mat2, activation: f32, coeff: f32, fiber_dir: Vec2, g: Mat2) {
        let (analytic_d_f, analytic_d_activation) =
            active_stress_vjp(f, activation, coeff, fiber_dir, g);
        let h = 1.0e-3_f32;

        // Activation (scalar).
        let numeric_activation = (loss(f, activation + h, coeff, fiber_dir, g)
            - loss(f, activation - h, coeff, fiber_dir, g))
            / (2.0 * h);
        let diff = (numeric_activation - analytic_d_activation).abs();
        let scale = numeric_activation
            .abs()
            .max(analytic_d_activation.abs())
            .max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "active_stress_vjp activation mismatch: analytic={analytic_d_activation:.6} \
             numeric={numeric_activation:.6} relative_diff={:.2e}",
            diff / scale
        );

        // F (4 components).
        let check_f_component =
            |set: fn(&mut Mat2, f32), get: fn(Mat2) -> f32, analytic_val: f32| {
                let mut f_plus = f;
                set(&mut f_plus, get(f) + h);
                let mut f_minus = f;
                set(&mut f_minus, get(f) - h);
                let numeric = (loss(f_plus, activation, coeff, fiber_dir, g)
                    - loss(f_minus, activation, coeff, fiber_dir, g))
                    / (2.0 * h);
                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "active_stress_vjp F mismatch: analytic={analytic_val:.6} numeric={numeric:.6} \
                 relative_diff={:.2e}",
                    diff / scale
                );
            };
        check_f_component(|m, v| m.x_axis.x = v, |m| m.x_axis.x, analytic_d_f.x_axis.x);
        check_f_component(|m, v| m.x_axis.y = v, |m| m.x_axis.y, analytic_d_f.x_axis.y);
        check_f_component(|m, v| m.y_axis.x = v, |m| m.y_axis.x, analytic_d_f.y_axis.x);
        check_f_component(|m, v| m.y_axis.y = v, |m| m.y_axis.y, analytic_d_f.y_axis.y);
    }

    #[test]
    fn matches_finite_difference_axis_aligned_fiber() {
        check(
            Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.9)),
            0.6,
            10.0,
            Vec2::X,
            Mat2::from_cols(Vec2::new(0.4, -0.6), Vec2::new(0.3, 0.5)),
        );
    }

    #[test]
    fn matches_finite_difference_off_axis_fiber() {
        check(
            Mat2::from_cols(Vec2::new(0.95, -0.15), Vec2::new(0.2, 1.1)),
            0.8,
            15.0,
            Vec2::new(0.6, 0.8),
            Mat2::from_cols(Vec2::new(-0.5, 0.9), Vec2::new(0.7, -0.2)),
        );
    }
}

#[cfg(test)]
mod multistep_backprop_tests {
    use super::*;
    use crate::materials::NeoHookeanMaterial;
    use crate::particle::{Particle, Particles};

    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
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
    }

    struct SubstepConfig {
        x: Vec2,
        mass: f32,
        stress_coeff: f32,
        dt: f32,
        kernel_d_inverse: f32,
        apic_blend: f32,
    }

    /// One real MLS-MPM substep (P2G scatter -> grid velocity update -> G2P
    /// gather -> F update) for a SINGLE particle at a FIXED position --
    /// position held fixed to match every adjoint above's own scoping
    /// (`p2g_stress_vjp`, `g2p_velocity_vjp`, etc. all defer the
    /// kernel-weight/position-dependence gap for the same reason; only
    /// `p2g_position_vjp` handles it, and isn't exercised here since this
    /// proof targets the OTHER remaining gap -- chaining substeps together).
    /// With one particle and fixed position, the 9-cell stencil can be
    /// tracked as a plain local array instead of a real `Grid`.
    fn substep_forward(
        f_old: Mat2,
        v_old: Vec2,
        c_old: Mat2,
        mat: &NeoHookeanMaterial,
        cfg: &SubstepConfig,
    ) -> (Mat2, Vec2, Mat2) {
        let particles = particle_with_f(f_old);
        let stress = mat.kirchhoff_stress(&particles, 0);
        let weights = quadratic_weights(cfg.x);

        let mut new_v = Vec2::ZERO;
        let mut b = Mat2::ZERO;
        for (gx, wx) in weights.wx.iter().enumerate() {
            for (gy, wy) in weights.wy.iter().enumerate() {
                let weight = wx * wy;
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - cfg.x + Vec2::splat(0.5);
                let momentum = weight
                    * (cfg.mass * (v_old + c_old * cell_dist)
                        + cfg.stress_coeff * (stress * cell_dist));
                let mass_c = weight * cfg.mass;
                let velocity_c = momentum / mass_c;
                let weighted = velocity_c * weight;
                new_v += weighted;
                b += Mat2::from_cols(weighted * cell_dist.x, weighted * cell_dist.y);
            }
        }
        let new_c = b * (cfg.kernel_d_inverse * cfg.apic_blend);
        let new_f = (Mat2::IDENTITY + cfg.dt * new_c) * f_old;
        (new_f, new_v, new_c)
    }

    /// The adjoint of `substep_forward`, built ENTIRELY from functions
    /// already shipped and individually verified above -- no new production
    /// math, just composition. This is the actual proof that
    /// backprop-through-multiple-substeps works: `F_old` feeds forward two
    /// ways (through `stress = kirchhoff_stress(F_old)` AND directly as
    /// `f_update_vjp`'s own `F_old` multiplicand), so its total gradient is a
    /// SUM of both paths -- the standard multivariable chain rule, not
    /// special-cased per path.
    ///
    /// `p2g_stress_vjp` is reused twice: once with the real `stress_coeff`
    /// for the stress->F path, once with `mass` standing in for that same
    /// scalar for the `c_old`->grid path -- both are the identical
    /// `weight*scalar*(tensor*cell_dist)` shape `scatter_particles_to_grid`
    /// computes, so the existing adjoint applies unchanged.
    fn substep_backward(
        f_old: Mat2,
        new_c: Mat2,
        mat: &NeoHookeanMaterial,
        cfg: &SubstepConfig,
        g_f_new: Mat2,
        g_v_new: Vec2,
        g_c_new: Mat2,
    ) -> (Mat2, Vec2, Mat2) {
        // F_new = (I + dt*new_c) * F_old
        let (g_c_from_f, g_f_old_a) = f_update_vjp(new_c, f_old, cfg.dt, g_f_new);
        let g_c_total = g_c_new + g_c_from_f;

        // new_v and new_c are both gathered from the same 9 grid velocities.
        let g_vel_from_c = g2p_affine_vjp(cfg.x, cfg.kernel_d_inverse, cfg.apic_blend, g_c_total);
        let g_vel_from_v = g2p_velocity_vjp(cfg.x, g_v_new);

        let weights = quadratic_weights(cfg.x);
        let mut g_momentum = [[Vec2::ZERO; 3]; 3];
        let mut g_v_old = Vec2::ZERO;
        for (gx, wx) in weights.wx.iter().enumerate() {
            for (gy, wy) in weights.wy.iter().enumerate() {
                let weight = wx * wy;
                let mass_c = weight * cfg.mass;
                let g_v_cell = g_vel_from_c[gx][gy] + g_vel_from_v[gx][gy];
                // update_velocities_vjp's d_loss_d_momentum output doesn't
                // depend on the forward momentum value (only on mass), so
                // the placeholder Vec2::ZERO here is exact, not an
                // approximation -- confirmed against the function's own
                // formula (see `grid/mod.rs`).
                let (g_m, _g_mass) = Grid::update_velocities_vjp(Vec2::ZERO, mass_c, g_v_cell);
                g_momentum[gx][gy] = g_m;
                g_v_old += weight * cfg.mass * g_m;
            }
        }

        let g_stress = p2g_stress_vjp(cfg.x, cfg.stress_coeff, &g_momentum);
        let g_c_old = p2g_stress_vjp(cfg.x, cfg.mass, &g_momentum);

        let particles = particle_with_f(f_old);
        let g_f_old_b = mat.kirchhoff_stress_vjp(&particles, 0, g_stress);

        (g_f_old_a + g_f_old_b, g_v_old, g_c_old)
    }

    /// The actual milestone: chain TWO real substeps forward, then backprop
    /// the whole thing back to the very first `F`, and check the result
    /// against a finite difference taken through the ENTIRE two-substep
    /// forward pass -- not just one isolated function. This is what "diff-MPM
    /// chain must finish first" concretely means: not more individually-
    /// verified pieces, but proof they compose across time.
    #[test]
    fn chains_two_substeps_matches_finite_difference() {
        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let cfg = SubstepConfig {
            x: Vec2::new(12.4, 7.8),
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
        };
        let target = Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(-0.03, 0.95));
        let f0_start = Mat2::from_cols(Vec2::new(1.15, 0.08), Vec2::new(-0.06, 0.9));

        let forward_two = |f0: Mat2| -> Mat2 {
            let (f1, v1, c1) = substep_forward(f0, Vec2::ZERO, Mat2::ZERO, &mat, &cfg);
            let (f2, _v2, _c2) = substep_forward(f1, v1, c1, &mat, &cfg);
            f2
        };

        let loss = |f0: Mat2| -> f32 {
            let d = forward_two(f0) - target;
            0.5 * (d.x_axis.x * d.x_axis.x
                + d.x_axis.y * d.x_axis.y
                + d.y_axis.x * d.y_axis.x
                + d.y_axis.y * d.y_axis.y)
        };

        // Analytic: forward, keeping intermediates, then backward from the
        // final loss all the way to f0_start.
        let (f1, v1, c1) = substep_forward(f0_start, Vec2::ZERO, Mat2::ZERO, &mat, &cfg);
        let (f2, _, c2) = substep_forward(f1, v1, c1, &mat, &cfg);
        let g_f2 = f2 - target; // dL/dF2 for L = 0.5*||F2-target||^2

        let (g_f1, g_v1, g_c1) = substep_backward(f1, c2, &mat, &cfg, g_f2, Vec2::ZERO, Mat2::ZERO);
        let (g_f0, _g_v0, _g_c0) = substep_backward(f0_start, c1, &mat, &cfg, g_f1, g_v1, g_c1);

        let h = 1.0e-3_f32;

        // Central-difference check on one scalar component of f0_start.
        let check_component =
            |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
                let mut f_plus = f0_start;
                set(&mut f_plus, base + h);
                let mut f_minus = f0_start;
                set(&mut f_minus, base - h);
                let numeric = (loss(f_plus) - loss(f_minus)) / (2.0 * h);

                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "two-substep chained adjoint mismatch at F{label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
                    diff / scale
                );
            };

        check_component("[0][0]", g_f0.x_axis.x, f0_start.x_axis.x, |m, v| {
            m.x_axis.x = v
        });
        check_component("[1][0]", g_f0.x_axis.y, f0_start.x_axis.y, |m, v| {
            m.x_axis.y = v
        });
        check_component("[0][1]", g_f0.y_axis.x, f0_start.y_axis.x, |m, v| {
            m.y_axis.x = v
        });
        check_component("[1][1]", g_f0.y_axis.y, f0_start.y_axis.y, |m, v| {
            m.y_axis.y = v
        });
    }

    /// Scales the two-substep proof above to a real rollout length (5
    /// substeps) via a plain loop over the same `substep_forward` /
    /// `substep_backward` functions -- no new math, just more of it. Proves
    /// the chain doesn't silently degrade (error accumulation, sign flips)
    /// over a longer horizon closer to what an actual trainer would run.
    #[test]
    fn chains_five_substeps_matches_finite_difference() {
        let mat = NeoHookeanMaterial::new(900.0, 700.0);
        let cfg = SubstepConfig {
            x: Vec2::new(4.6, 18.2),
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
        };
        let target = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(-0.05, 0.85));
        let f0_start = Mat2::from_cols(Vec2::new(1.05, 0.03), Vec2::new(-0.02, 0.97));
        const STEPS: usize = 5;

        let forward_n = |f0: Mat2| -> Mat2 {
            let (mut f, mut v, mut c) = (f0, Vec2::ZERO, Mat2::ZERO);
            for _ in 0..STEPS {
                let (f_new, v_new, c_new) = substep_forward(f, v, c, &mat, &cfg);
                f = f_new;
                v = v_new;
                c = c_new;
            }
            f
        };

        let loss = |f0: Mat2| -> f32 {
            let d = forward_n(f0) - target;
            0.5 * (d.x_axis.x * d.x_axis.x
                + d.x_axis.y * d.x_axis.y
                + d.y_axis.x * d.y_axis.x
                + d.y_axis.y * d.y_axis.y)
        };

        // Forward, keeping every intermediate (f, v, c) so the backward
        // pass has what each substep's own backward call needs.
        let mut states = Vec::with_capacity(STEPS + 1);
        states.push((f0_start, Vec2::ZERO, Mat2::ZERO));
        for _ in 0..STEPS {
            let (f, v, c) = *states.last().unwrap();
            states.push(substep_forward(f, v, c, &mat, &cfg));
        }
        let f_final = states[STEPS].0;
        let mut g_f = f_final - target; // dL/dF_final
        let mut g_v = Vec2::ZERO;
        let mut g_c = Mat2::ZERO;

        for step in (0..STEPS).rev() {
            let (f_old, _, _) = states[step];
            let (_, _, c_new) = states[step + 1];
            let (next_g_f, next_g_v, next_g_c) =
                substep_backward(f_old, c_new, &mat, &cfg, g_f, g_v, g_c);
            g_f = next_g_f;
            g_v = next_g_v;
            g_c = next_g_c;
        }
        let g_f0 = g_f;

        let h = 1.0e-3_f32;
        let check_component =
            |label: &str, analytic_val: f32, base: f32, set: fn(&mut Mat2, f32)| {
                let mut f_plus = f0_start;
                set(&mut f_plus, base + h);
                let mut f_minus = f0_start;
                set(&mut f_minus, base - h);
                let numeric = (loss(f_plus) - loss(f_minus)) / (2.0 * h);

                let diff = (numeric - analytic_val).abs();
                let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
                assert!(
                    diff / scale < 1.0e-2,
                    "{STEPS}-substep chained adjoint mismatch at F{label}: \
                     analytic={analytic_val:.6} numeric(central-diff)={numeric:.6} \
                     relative_diff={:.2e}",
                    diff / scale
                );
            };

        check_component("[0][0]", g_f0.x_axis.x, f0_start.x_axis.x, |m, v| {
            m.x_axis.x = v
        });
        check_component("[1][0]", g_f0.x_axis.y, f0_start.x_axis.y, |m, v| {
            m.x_axis.y = v
        });
        check_component("[0][1]", g_f0.y_axis.x, f0_start.y_axis.x, |m, v| {
            m.y_axis.x = v
        });
        check_component("[1][1]", g_f0.y_axis.y, f0_start.y_axis.y, |m, v| {
            m.y_axis.y = v
        });
    }
}

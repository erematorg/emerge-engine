//! Test suite for `NeoHookeanMaterial` -- split out of `elastic.rs` (was ~440 of
//! its ~770 lines), same reasoning as `gpu/solver/device_lost_tests.rs`: the
//! constitutive-model file itself should read as the model, not scroll past
//! three elaborate verification suites (Hooke's-law limit, damage softening,
//! finite-difference VJP checks) to get there.

use super::*;

#[cfg(test)]
mod small_strain_linear_elasticity_tests {
    use super::*;
    use crate::Particle;
    use glam::Vec2;

    /// **Small-strain limit must recover exact linear elasticity (Hooke's law).**
    ///
    /// `NeoHookeanMaterial` had zero test comparing its stress-strain response to
    /// any real/analytical elasticity result (confirmed via a full test-file
    /// audit, 2026-07-07) -- only stability (J>0, symmetry) and damage-direction
    /// checks existed. Every well-formed hyperelastic model must reduce to
    /// isotropic linear elasticity as strain -> 0: sigma = lambda*tr(eps)*I +
    /// 2*mu*eps for infinitesimal strain eps. Derivation for THIS model's exact
    /// formula (tau = (mu/J)*dev(B) + (k/2)*(J^2-1)*I, k=lambda+mu, B=F*F^T):
    /// for F = I + delta*E (E symmetric, delta small), linearizing to O(delta)
    /// gives tau ~= 2*mu*delta*dev(E) + (lambda+mu)*delta*tr(E)*I, which is
    /// EXACTLY sigma = lambda*tr(eps)*I + 2*mu*eps with eps=delta*E (the plane-
    /// strain form, matching this material's own k=lambda+mu bulk modulus
    /// fix). Verified numerically here, not just derived by hand.
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

    /// Real analytical Hooke's law prediction: sigma = lambda*tr(eps)*I + 2*mu*eps.
    fn linear_elastic_prediction(lambda: f32, mu: f32, eps: Mat2) -> Mat2 {
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps
    }

    #[test]
    fn small_uniaxial_strain_matches_hookes_law() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        // Uniaxial strain: stretch in x, zero in y (E = diag(1, 0)).
        let e = Mat2::from_diagonal(Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain NeoHookean stress should match linear elasticity (Hooke's law) \
             to O(delta^2): predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    #[test]
    fn small_shear_strain_matches_hookes_law() {
        let lambda = 500.0;
        let mu = 1200.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        // Pure shear strain (symmetric, zero trace): E = [[0, 1], [1, 0]].
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain NeoHookean shear stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// Confirms the match is a real convergence to exact linear elasticity as
    /// strain shrinks, not a coincidence at one specific delta.
    ///
    /// The ABSOLUTE residual (actual minus predicted stress) is genuine O(delta^2)
    /// -- hand-derived and hand-verified against the exact numbers at delta=0.01
    /// (predicted residual from the O(delta^2) term in this model's own
    /// linearization matched the measured residual to within rounding). But
    /// RELATIVE error here is that O(delta^2) absolute residual divided by the
    /// O(delta) leading-order predicted stress, so it correctly scales as
    /// O(delta^2)/O(delta) = O(delta) -- linear, roughly 2x per halving, NOT 4x.
    /// (First version of this test wrongly expected 4x by conflating absolute
    /// and relative error order -- fixed after the measured ~2x ratio held
    /// consistently across delta=1e-2 down to 5e-4, five halvings, before f32
    /// precision noise took over below that.)
    #[test]
    fn hookes_law_match_improves_as_strain_shrinks() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = NeoHookeanMaterial::new(lambda, mu);
        let e = Mat2::from_diagonal(Vec2::new(1.0, -0.3));

        let rel_err_at = |delta: f32| -> f32 {
            let f = Mat2::IDENTITY + delta * e;
            let particles = particle_with_f(f);
            let tau = mat.kirchhoff_stress(&particles, 0);
            let predicted = linear_elastic_prediction(lambda, mu, delta * e);
            let diff = tau - predicted;
            let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
            let scale =
                (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
            err / scale
        };

        let err_large = rel_err_at(1.0e-2);
        let err_small = rel_err_at(5.0e-3);
        assert!(
            err_small < err_large * 0.7 && err_small > err_large * 0.3,
            "halving strain should roughly halve relative error (O(delta) relative \
             error from an O(delta^2) absolute residual over an O(delta) leading term): \
             err(1e-2)={err_large:.2e} err(5e-3)={err_small:.2e} ratio={:.2}",
            err_small / err_large
        );
    }
}

#[cfg(test)]
mod damage_softening_tests {
    use super::*;
    use crate::Particle;

    fn particle_with(deformation_gradient: Mat2, friction_hardening: f32) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening,
            log_volume_strain: 0.0,
            temperature: 0.0,
            scalar_field: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: 0,
        });
        particles
    }

    #[test]
    fn zero_softening_rate_matches_undamaged_stress() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.0;

        let undamaged = particle_with(f, 0.0);
        let damaged = particle_with(f, 5.0);
        let tau_undamaged = mat.kirchhoff_stress(&undamaged, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert_eq!(
            tau_undamaged, tau_damaged,
            "rate=0.0 must leave stress unaffected by damage (backward compatible default)"
        );
    }

    #[test]
    fn damage_softens_stress_magnitude() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.5;

        let healthy = particle_with(f, 0.0);
        let damaged = particle_with(f, 3.0);
        let tau_healthy = mat.kirchhoff_stress(&healthy, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert!(
            tau_damaged.x_axis.length() < tau_healthy.x_axis.length(),
            "damaged tissue must produce weaker stress for the same deformation: \
             healthy={:?} damaged={:?}",
            tau_healthy,
            tau_damaged
        );
    }

    #[test]
    fn severe_damage_approaches_near_zero_stiffness() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 1.0;

        let severely_damaged = particle_with(f, 20.0); // exp(-20) ~ 2e-9, near-total loss
        let tau = mat.kirchhoff_stress(&severely_damaged, 0);
        assert!(
            tau.x_axis.length() < 1.0e-3,
            "severe damage must drive stiffness (and thus stress) toward zero, got {:?}",
            tau
        );
    }
}

#[cfg(test)]
mod kirchhoff_stress_vjp_tests {
    use super::*;
    use crate::Particle;

    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
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
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            _pad: 0,
        });
        particles
    }

    /// Scalar loss L(F) = tau(F) : g (Frobenius inner product with a fixed g) --
    /// the standard way to reduce a matrix-to-matrix function to something a
    /// central-difference check can validate one scalar output at a time.
    fn loss(mat: &NeoHookeanMaterial, f: Mat2, g: Mat2) -> f32 {
        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        tau.x_axis.x * g.x_axis.x
            + tau.x_axis.y * g.x_axis.y
            + tau.y_axis.x * g.y_axis.x
            + tau.y_axis.y * g.y_axis.y
    }

    /// Central-difference numerical gradient of `loss` w.r.t. each of F's 4
    /// components, compared against the analytic `kirchhoff_stress_vjp`.
    ///
    /// This is the real verification the hand derivation needed -- matching
    /// this project's standing "verify numerically" discipline for anything
    /// hand-derived, doubly so for tensor calculus where sign/transpose
    /// errors are exactly the class of mistake that doesn't show up as a
    /// compile error or a crash, only as silently wrong gradients.
    /// Perturbs one scalar component of F by ±h, returns the central-difference
    /// numerical derivative of `loss` w.r.t. that component.
    fn numeric_grad_component(
        mat: &NeoHookeanMaterial,
        mut f: Mat2,
        g: Mat2,
        h: f32,
        set: impl Fn(&mut Mat2, f32),
        get: impl Fn(Mat2) -> f32,
    ) -> f32 {
        let base = get(f);
        set(&mut f, base + h);
        let loss_plus = loss(mat, f, g);
        set(&mut f, base - h);
        let loss_minus = loss(mat, f, g);
        (loss_plus - loss_minus) / (2.0 * h)
    }

    fn check_vjp_matches_finite_difference(mat: &NeoHookeanMaterial, f: Mat2, g: Mat2) {
        let analytic = {
            let particles = particle_with_f(f);
            mat.kirchhoff_stress_vjp(&particles, 0, g)
        };

        let h = 1.0e-3_f32;
        // glam's Mat2 stores columns (x_axis, y_axis); x_axis.y is row 1 of
        // column 0, i.e. F[1][0] in row-major reading.
        let checks: [(&str, f32); 4] = [
            (
                "F[0][0]",
                numeric_grad_component(mat, f, g, h, |m, v| m.x_axis.x = v, |m| m.x_axis.x),
            ),
            (
                "F[1][0]",
                numeric_grad_component(mat, f, g, h, |m, v| m.x_axis.y = v, |m| m.x_axis.y),
            ),
            (
                "F[0][1]",
                numeric_grad_component(mat, f, g, h, |m, v| m.y_axis.x = v, |m| m.y_axis.x),
            ),
            (
                "F[1][1]",
                numeric_grad_component(mat, f, g, h, |m, v| m.y_axis.y = v, |m| m.y_axis.y),
            ),
        ];
        let analytic_vals = [
            analytic.x_axis.x,
            analytic.x_axis.y,
            analytic.y_axis.x,
            analytic.y_axis.y,
        ];

        for ((label, numeric), analytic_val) in checks.iter().zip(analytic_vals) {
            let diff = (numeric - analytic_val).abs();
            let scale = numeric.abs().max(analytic_val.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "kirchhoff_stress_vjp mismatch at {label}: analytic={analytic_val:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e} \
                 (F={f:?}, g={g:?})",
                diff / scale
            );
        }
    }

    #[test]
    fn vjp_matches_finite_difference_at_identity() {
        let mat = NeoHookeanMaterial::new(1000.0, 800.0);
        check_vjp_matches_finite_difference(&mat, Mat2::IDENTITY, Mat2::IDENTITY);
    }

    #[test]
    fn vjp_matches_finite_difference_under_stretch() {
        let mat = NeoHookeanMaterial::new(1000.0, 800.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.05), glam::Vec2::new(-0.02, 0.9));
        let g = Mat2::from_cols(glam::Vec2::new(0.7, -0.3), glam::Vec2::new(0.4, 1.1));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_matches_finite_difference_under_shear() {
        let mat = NeoHookeanMaterial::new(500.0, 1200.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.0, 0.4), glam::Vec2::new(0.15, 1.05));
        let g = Mat2::from_cols(glam::Vec2::new(-0.5, 0.9), glam::Vec2::new(0.2, -0.6));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_matches_finite_difference_with_nonsymmetric_g() {
        // g need not be symmetric in general (only the dev(B)-derived internal
        // adjoint happens to be) -- confirms the derivation handles the fully
        // general case, not just the symmetric one it happens to be called
        // with in a real P2G force-scatter backward pass.
        let mat = NeoHookeanMaterial::new(800.0, 800.0);
        let f = Mat2::from_cols(glam::Vec2::new(1.1, -0.1), glam::Vec2::new(0.2, 0.95));
        let g = Mat2::from_cols(glam::Vec2::new(0.3, 1.2), glam::Vec2::new(-0.8, 0.1));
        check_vjp_matches_finite_difference(&mat, f, g);
    }

    #[test]
    fn vjp_respects_thermal_and_damage_scaling() {
        let mut mat = NeoHookeanMaterial::new(900.0, 700.0);
        mat.thermal_expansion = -0.01;
        mat.damage_softening_rate = 0.3;

        let f = Mat2::from_cols(glam::Vec2::new(1.15, 0.08), glam::Vec2::new(-0.05, 0.92));
        let temperature = 12.0;
        let friction_hardening = 2.0;

        let particle_with = |f: Mat2| -> Particles {
            let mut particles = particle_with_f(f);
            particles.temperature[0] = temperature;
            particles.friction_hardening[0] = friction_hardening;
            particles
        };

        let g = Mat2::from_cols(glam::Vec2::new(0.6, -0.4), glam::Vec2::new(0.5, 0.7));
        let analytic = mat.kirchhoff_stress_vjp(&particle_with(f), 0, g);

        let h = 1.0e-3_f32;
        let mut f_plus = f;
        f_plus.x_axis.x += h;
        let mut f_minus = f;
        f_minus.x_axis.x -= h;

        let tau_plus = mat.kirchhoff_stress(&particle_with(f_plus), 0);
        let tau_minus = mat.kirchhoff_stress(&particle_with(f_minus), 0);
        let dot = |t: Mat2| {
            t.x_axis.x * g.x_axis.x
                + t.x_axis.y * g.x_axis.y
                + t.y_axis.x * g.y_axis.x
                + t.y_axis.y * g.y_axis.y
        };
        let numeric = (dot(tau_plus) - dot(tau_minus)) / (2.0 * h);

        let diff = (numeric - analytic.x_axis.x).abs();
        let scale = numeric.abs().max(analytic.x_axis.x.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "vjp must still match finite-difference with thermal/damage scaling active: \
             analytic={:.6} numeric={numeric:.6} relative_diff={:.2e}",
            analytic.x_axis.x,
            diff / scale
        );
    }
}

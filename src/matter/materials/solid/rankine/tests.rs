//! Tests of the Rankine brittle material.

#[cfg(test)]
mod marginal_yield_tests {
    use super::super::*;
    use crate::Particle;

    /// Checks that `update_particle`'s return mapping matches this material's
    /// documented tensile cutoff (`max(tau1,tau2) <= t_eff`) exactly, like
    /// `sand.rs`/`von_mises.rs`'s `marginal_yield_tests`.
    fn run_one_step(mat: &RankineMaterial, sigma: Vec2, damage: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = damage;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    fn run_rate_step(
        mat: &RankineMaterial,
        f: Mat2,
        velocity_gradient: Mat2,
        dt: f32,
        damage: f32,
    ) -> (Mat2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.velocity_gradient = velocity_gradient;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = damage;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), dt);
        (
            particles.deformation_gradient[0],
            particles.friction_hardening[0],
        )
    }

    /// Given eps.y=0, tau.x = a*eps.x (a=2*mu+lambda), tau.y = lambda*eps.x --
    /// solving for the eps.x that puts tau.x EXACTLY at a target tensile stress.
    fn eps_x_for_target_tau_x(mat: &RankineMaterial, target_tau_x: f32) -> f32 {
        let a = 2.0 * mat.mu + mat.lambda;
        target_tau_x / a
    }

    #[test]
    fn marginal_state_at_tensile_strength_does_not_yield() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 0.99 * mat.tensile_strength);
        let sigma = Vec2::new(eps_x.exp(), 1.0); // eps.y = ln(1.0) = 0

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);
        assert!(
            (sigma_after - sigma).length() < 1.0e-4,
            "state inside the tensile-cutoff surface should stay elastic (no change): \
             sigma={sigma:?} sigma_after={sigma_after:?}"
        );
        assert_eq!(
            damage_after, 0.0,
            "damage must not accumulate on an elastic step"
        );
    }

    #[test]
    fn marginal_state_beyond_tensile_strength_projects_exactly_to_the_yield_surface() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength);
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);

        // The projected principal stress must land EXACTLY at tensile_strength
        // (damage = 0, so t_eff = tensile_strength, no floor or saturation).
        let a = 2.0 * mat.mu + mat.lambda;
        let eps_after_x = sigma_after.x.ln();
        let eps_after_y = sigma_after.y.ln();
        let tau_x_after = a * eps_after_x + mat.lambda * eps_after_y;
        let tau_y_after = mat.lambda * eps_after_x + a * eps_after_y;
        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3,
            "projected principal stress should land EXACTLY on the tensile-cutoff \
             surface (t_eff={}), got {tau_x_after:.6}",
            mat.tensile_strength
        );

        // The real invariant for the UNAFFECTED principal direction is the
        // STRESS tau.y (not the strain sigma.y/eps.y) staying fixed -- this is
        // a single-component projection in STRESS space (project_stress's
        // (true,false) branch only rewrites tau.x). Because stress and strain
        // are coupled through lambda, inverting back to strain space changes
        // BOTH eps.x and eps.y even though only tau.x was projected -- eps.y
        // changing is correct coupled elasticity, not a bug, so the invariant
        // to check is tau.y, not sigma.y.
        let original_tau_y = mat.lambda * eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength)
            + a * sigma.y.ln();
        assert!(
            (tau_y_after - original_tau_y).abs() < 1.0e-3,
            "the non-yielding principal STRESS (tau.y) must be untouched: \
             expected {original_tau_y}, got {tau_y_after}"
        );

        assert!(
            damage_after > 0.0,
            "damage must accumulate on a plastic (yielding) step"
        );
    }

    #[test]
    fn biaxial_tension_projects_both_components_to_the_corner() {
        // Both principal stresses exceed t_eff simultaneously -- the documented
        // "corner return" case (project_stress's (true,true) branch).
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let a = 2.0 * mat.mu + mat.lambda;
        // Symmetric biaxial tension: eps.x = eps.y = e, giving tau.x=tau.y=(a+lambda)*e.
        let e = (1.5 * mat.tensile_strength) / (a + mat.lambda);
        let sigma = Vec2::new(e.exp(), e.exp());

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);
        let eps_after_x = sigma_after.x.ln();
        let eps_after_y = sigma_after.y.ln();
        let tau_x_after = a * eps_after_x + mat.lambda * eps_after_y;
        let tau_y_after = mat.lambda * eps_after_x + a * eps_after_y;

        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3
                && (tau_y_after - mat.tensile_strength).abs() < 1.0e-3,
            "biaxial tension must project BOTH components exactly to t_eff={}: \
             got tau_x={tau_x_after:.6} tau_y={tau_y_after:.6}",
            mat.tensile_strength
        );
        assert!(
            damage_after > 0.0,
            "damage must accumulate on the corner-return case"
        );
    }

    /// The defining, category-specific behavior of a BRITTLE material (vs.
    /// VonMises's non-softening ductile plasticity): damage makes the body
    /// genuinely WEAKER, not just permanently deformed. A stress level
    /// comfortably under the VIRGIN tensile strength must still fail once the
    /// particle already carries real damage -- softening = real loss of
    /// load-bearing capacity, not bookkeeping. No prior test exercised
    /// `run_one_step` with `damage > 0` at all (confirmed via a read of every
    /// call site in this module before writing this one).
    #[test]
    fn accumulated_damage_lowers_the_yield_threshold_real_softening() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 2.0);
        let damage = 0.5; // well under this softening_rate's saturation point (~1.5)
        let t_eff_damaged = mat.tensile_strength * (-mat.softening_rate * damage).exp();

        // 70% of virgin strength: comfortably elastic at damage=0 (this
        // module's own `marginal_state_at_tensile_strength_does_not_yield`
        // treats 99% as still-elastic), but above the damaged threshold.
        let target_tau_x = 0.7 * mat.tensile_strength;
        assert!(
            target_tau_x > t_eff_damaged,
            "test setup sanity: target stress must exceed the damaged threshold"
        );
        let eps_x = eps_x_for_target_tau_x(&mat, target_tau_x);
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, damage);

        assert!(
            (sigma_after - sigma).length() > 1.0e-4,
            "a damaged particle must yield at a stress the VIRGIN material \
             would have carried elastically: target_tau_x={target_tau_x} \
             t_eff_damaged={t_eff_damaged}"
        );
        let a = 2.0 * mat.mu + mat.lambda;
        let tau_x_after = a * sigma_after.x.ln() + mat.lambda * sigma_after.y.ln();
        assert!(
            (tau_x_after - t_eff_damaged).abs() < 1.0e-3,
            "projected stress should land exactly at the DAMAGED threshold \
             ({t_eff_damaged:.4}), not the virgin one ({}): got {tau_x_after:.6}",
            mat.tensile_strength
        );
        assert!(
            damage_after > damage,
            "damage must keep accumulating on repeated yielding"
        );
    }

    #[test]
    fn exponential_trial_preserves_rigid_rotation_without_false_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let dt = 0.25;
        let omega = 0.8;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
        let (f_after, damage_after) = run_rate_step(&mat, Mat2::IDENTITY, spin, dt, 0.0);
        let expected = Mat2::from_angle(omega * dt);
        let error = (f_after.x_axis - expected.x_axis)
            .abs()
            .max_element()
            .max((f_after.y_axis - expected.y_axis).abs().max_element());
        assert!(
            error < 1.0e-6,
            "rigid spin must integrate to a rotation: {f_after:?}"
        );
        assert!((f_after.determinant() - 1.0).abs() < 1.0e-6);
        assert_eq!(
            damage_after, 0.0,
            "rigid rotation must not create Rankine damage"
        );
    }

    #[test]
    fn opposite_subthreshold_rates_are_reversible_without_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 0.5 * mat.tensile_strength);
        let rate = Mat2::from_diagonal(Vec2::new(eps_x, 0.0));
        let (f_loaded, damage_loaded) = run_rate_step(&mat, Mat2::IDENTITY, rate, 1.0, 0.0);
        assert_eq!(damage_loaded, 0.0);
        let (f_unloaded, damage_unloaded) =
            run_rate_step(&mat, f_loaded, -rate, 1.0, damage_loaded);
        let error = (f_unloaded.x_axis - Mat2::IDENTITY.x_axis)
            .abs()
            .max_element()
            .max(
                (f_unloaded.y_axis - Mat2::IDENTITY.y_axis)
                    .abs()
                    .max_element(),
            );
        assert!(
            error < 1.0e-6,
            "opposite rates must restore F exactly: {f_unloaded:?}"
        );
        assert_eq!(damage_unloaded, 0.0);
    }

    #[test]
    fn nonzero_rate_trial_projects_to_rankine_surface_and_accumulates_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength);
        let rate = Mat2::from_diagonal(Vec2::new(eps_x, 0.0));
        let (f_after, damage_after) = run_rate_step(&mat, Mat2::IDENTITY, rate, 1.0, 0.0);
        let (_, sigma_after, _) = svd2(f_after);
        let eps_after = hencky_strains(sigma_after);
        let a = 2.0 * mat.mu + mat.lambda;
        let tau_x_after = a * eps_after.x + mat.lambda * eps_after.y;
        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3,
            "rate-generated trial state must return to the tensile surface: {tau_x_after}"
        );
        assert!(damage_after > 0.0);
    }
}

#[cfg(test)]
mod damping_tests {
    use super::super::*;
    use crate::Particle;

    /// `q_factor_elastic_viscosity_pa_s` must return a finite,
    /// positive SI viscosity for the cited real ice Q-factor -- basic
    /// sanity floor before trusting the constant in a live scene.
    #[test]
    fn ice_quality_factor_viscosity_is_finite_and_positive() {
        let shear_modulus_pa = 9.0e9 / (2.0 * (1.0 + 0.3));
        let eta = q_factor_elastic_viscosity_pa_s(
            shear_modulus_pa,
            ICE_QUALITY_FACTOR_Q,
            ICE_Q_REFERENCE_FREQUENCY_HZ,
        );
        assert!(
            eta.is_finite() && eta > 0.0,
            "real ice Q-factor must produce a finite, positive Pa.s viscosity, got {eta}"
        );
    }

    /// A higher Q (less damping, real physical meaning: colder/purer ice)
    /// must produce a LOWER viscosity -- the inverse relationship
    /// `eta = G/(Q*omega)`, not an accidental monotonic-the-wrong-way bug.
    #[test]
    fn higher_quality_factor_means_less_damping() {
        let g = 4.0e9;
        let low_q_eta = q_factor_elastic_viscosity_pa_s(g, 100.0, 136.0);
        let high_q_eta = q_factor_elastic_viscosity_pa_s(g, 1700.0, 136.0);
        assert!(
            high_q_eta < low_q_eta,
            "higher Q (less real damping) must give lower viscosity: \
             Q=100 -> {low_q_eta}, Q=1700 -> {high_q_eta}"
        );
    }

    /// Measures `elastic_viscosity`'s damping against its cited target Q. One
    /// particle goes through a full cycle of pure shear (`F(t) = I +
    /// A*sin(wt)*E`, E symmetric and traceless so the corotated rotation is
    /// exactly I, as in `CorotatedMaterial`'s `small_shear_strain_matches_
    /// hookes_law`); the energy dissipated per cycle is integrated from
    /// `tau:D` (the elastic part integrates to zero over a cycle), and `Q =
    /// 2*pi*E_max/delta_E_cycle`. It must match the target: the textbook
    /// conversion `G/(Q*omega)` gives twice the Q (see
    /// `q_factor_elastic_viscosity_pa_s`).
    #[test]
    fn measured_q_factor_matches_target_after_the_conversion_fix() {
        let mu = 4.0e9_f32;
        let lambda = 4.0e9_f32;
        let target_q = 65.0_f32;
        let reference_freq_hz = 136.0_f32;
        let omega = 2.0 * std::f32::consts::PI * reference_freq_hz;
        let eta = q_factor_elastic_viscosity_pa_s(mu, target_q, reference_freq_hz);

        let mat = RankineMaterial {
            elastic_viscosity: eta,
            ..RankineMaterial::new(lambda, mu, f32::MAX, 0.0)
        };

        let amplitude = 1.0e-4_f32; // small-strain regime
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)); // pure shear, traceless, R=I exactly

        let n_samples = 2000_u32;
        let period = 2.0 * std::f32::consts::PI / omega;
        let dt = period / n_samples as f32;

        let mut dissipated = 0.0_f64;
        for k in 0..n_samples {
            let t = k as f32 * dt;
            let delta = amplitude * (omega * t).sin();
            let delta_dot = amplitude * omega * (omega * t).cos();

            let mut p = Particle::zeroed();
            p.deformation_gradient = Mat2::IDENTITY + delta * e;
            p.velocity_gradient = delta_dot * e;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            let particles = Particles::from(vec![p]);

            let tau = mat.kirchhoff_stress(&particles, 0);
            let d = delta_dot * e; // already symmetric (E is symmetric)
            let power = (tau.x_axis.x * d.x_axis.x
                + tau.x_axis.y * d.x_axis.y
                + tau.y_axis.x * d.y_axis.x
                + tau.y_axis.y * d.y_axis.y) as f64;
            dissipated += power * dt as f64;
        }

        let e_max = 2.0 * (mu as f64) * (amplitude as f64).powi(2); // peak elastic energy density, pure shear
        let measured_q = 2.0 * std::f64::consts::PI * e_max / dissipated;

        assert!(
            (measured_q - target_q as f64).abs() / (target_q as f64) < 0.05,
            "measured Q from actual kirchhoff_stress output must match the \
             cited target Q to within numerical error: target_q={target_q}, \
             got measured_q={measured_q:.3}"
        );
    }

    /// The actual mechanism this was added for: `elastic_viscosity > 0.0`
    /// must make `kirchhoff_stress` respond to the particle's velocity
    /// gradient (viscous stress), not just its deformation gradient
    /// (elastic stress) -- confirms the Kelvin-Voigt term actually engages,
    /// not just that the field exists.
    #[test]
    fn nonzero_elastic_viscosity_adds_a_real_viscous_stress_term() {
        let elastic_only = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let mut damped = elastic_only;
        damped.elastic_viscosity = 50.0;

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        // A pure shearing velocity gradient -- nonzero deviatoric strain
        // rate, the exact quantity the Kelvin-Voigt term reacts to.
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let particles = Particles::from(vec![p]);

        let tau_elastic = elastic_only.kirchhoff_stress(&particles, 0);
        let tau_damped = damped.kirchhoff_stress(&particles, 0);

        let diff = tau_damped - tau_elastic;
        let max_abs = diff
            .x_axis
            .abs()
            .max_element()
            .max(diff.y_axis.abs().max_element());
        assert!(
            max_abs > 1.0e-6,
            "nonzero elastic_viscosity under a real velocity gradient must \
             change the Kirchhoff stress: elastic={tau_elastic:?} damped={tau_damped:?}"
        );
    }

    /// `elastic_viscosity == 0.0` (every preset's default) reproduces the pure
    /// elastic stress.
    #[test]
    fn zero_elastic_viscosity_is_bit_identical_to_pure_elastic_stress() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        assert_eq!(mat.elastic_viscosity, 0.0);

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(0.01, 0.97));
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4));
        let particles = Particles::from(vec![p]);

        let tau = mat.kirchhoff_stress(&particles, 0);
        let expected =
            corotated_elastic_stress(particles.deformation_gradient[0], mat.lambda, mat.mu);
        assert_eq!(
            tau, expected,
            "elastic_viscosity=0.0 must be bit-identical to the pure elastic path"
        );
    }
}

#[cfg(test)]
mod rock_preset_tests {
    use super::super::*;
    use crate::Particle;

    /// The rock presets are different materials, not renamed numbers:
    /// limestone is softer (lower E) than sandstone and stronger in tension
    /// per unit stiffness (higher tensile-to-modulus ratio, Xu 2016).
    #[test]
    fn sandstone_and_limestone_presets_are_genuinely_distinct() {
        let sandstone = RankineMaterial::sandstone(20.0e9, 0.25);
        let limestone = RankineMaterial::limestone(8.0e9, 0.25);

        assert!(
            limestone.tensile_strength / limestone.lambda.max(1.0)
                != sandstone.tensile_strength / sandstone.lambda.max(1.0),
            "sandstone and limestone presets must not collapse to the same ratio"
        );
        let sandstone_ratio = sandstone.tensile_strength / 20.0e9;
        let limestone_ratio = limestone.tensile_strength / 8.0e9;
        assert!(
            limestone_ratio > sandstone_ratio,
            "limestone's real tensile-to-modulus ratio should be higher than \
             sandstone's (Xu 2016): limestone={limestone_ratio:.2e} sandstone={sandstone_ratio:.2e}"
        );
    }

    /// All 6 Rankine presets (bone/rock/ice family) must produce finite, positive
    /// tensile strengths at a representative modulus -- a basic sanity floor
    /// before trusting any of them in a live scene.
    #[test]
    fn all_rock_and_bone_presets_produce_finite_positive_tensile_strength() {
        let e = 30.0e9;
        let nu = 0.25;
        for (name, mat) in [
            ("stiff_brittle", RankineMaterial::stiff_brittle(e, nu)),
            ("high_tensile", RankineMaterial::high_tensile(e, nu)),
            ("sandstone", RankineMaterial::sandstone(e, nu)),
            ("limestone", RankineMaterial::limestone(e, nu)),
            ("shale", RankineMaterial::shale(e, nu)),
            ("ice", RankineMaterial::ice(e, nu)),
        ] {
            assert!(
                mat.tensile_strength.is_finite() && mat.tensile_strength > 0.0,
                "{name}: tensile_strength must be finite and positive, got {}",
                mat.tensile_strength
            );
        }
    }

    /// Emergent fracture, not only preset parameters, through
    /// `update_particle`'s return mapping (the technique of
    /// `marginal_yield_tests::run_one_step`), as `rock_fracture.rs` shows it
    /// on screen.
    ///
    /// `sandstone` and `limestone` go through the same `from_young_modulus` at
    /// the same (E, nu), so share lambda and mu; only `tensile_strength`
    /// differs (ratio 1.0e-3 against limestone's 3.1e-3, Xu 2016): at E = 20
    /// GPa, 20 MPa and 62 MPa. One tensile stress state of 35 MPa, between
    /// them, must damage sandstone and leave limestone elastic.
    #[test]
    fn weaker_rock_fractures_under_a_load_stiffer_rock_survives() {
        let e = 20.0e9;
        let nu = 0.25;
        let sandstone = RankineMaterial::sandstone(e, nu);
        let limestone = RankineMaterial::limestone(e, nu);
        assert!(
            sandstone.tensile_strength < limestone.tensile_strength,
            "test setup sanity: sandstone must be the weaker preset here"
        );

        let target_tau_x = 0.5 * (sandstone.tensile_strength + limestone.tensile_strength);
        assert!(
            target_tau_x > sandstone.tensile_strength && target_tau_x < limestone.tensile_strength,
            "test setup sanity: target stress must sit strictly between the \
             two real tensile strengths"
        );

        // Same real construction `run_one_step`/`eps_x_for_target_tau_x`
        // (marginal_yield_tests, this file) use: eps.y=0, tau.x=a*eps.x
        // with a=2*mu+lambda (IDENTICAL for both presets here).
        let a = 2.0 * sandstone.mu + sandstone.lambda;
        assert_eq!(
            a,
            2.0 * limestone.mu + limestone.lambda,
            "sandstone and limestone must share identical lambda/mu at the \
             same (E, nu) -- only tensile_strength should differ"
        );
        let eps_x = target_tau_x / a;
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let one_step = |mat: &RankineMaterial| -> (Vec2, f32) {
            let mut p = Particle::zeroed();
            p.deformation_gradient =
                Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
            p.mass = 1.0;
            p.initial_volume = 1.0;
            let mut particles = Particles::from(vec![p]);
            mat.update_particle(&mut particles.update_ctx(0), 1.0);
            let f = particles.deformation_gradient[0];
            (
                Vec2::new(f.x_axis.x, f.y_axis.y),
                particles.friction_hardening[0],
            )
        };

        let (sigma_sandstone, damage_sandstone) = one_step(&sandstone);
        let (sigma_limestone, damage_limestone) = one_step(&limestone);

        assert!(
            (sigma_sandstone - sigma).length() > 1.0e-4 && damage_sandstone > 0.0,
            "sandstone (weaker, tensile_strength={:.3e}) must fracture under \
             {target_tau_x:.3e} Pa: sigma_after={sigma_sandstone:?} damage={damage_sandstone}",
            sandstone.tensile_strength
        );
        assert!(
            (sigma_limestone - sigma).length() < 1.0e-6 && damage_limestone == 0.0,
            "limestone (stronger, tensile_strength={:.3e}) must stay ELASTIC \
             under the SAME {target_tau_x:.3e} Pa: sigma_after={sigma_limestone:?} \
             damage={damage_limestone}",
            limestone.tensile_strength
        );
    }
}

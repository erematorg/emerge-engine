//! Core `DruckerPragerMaterial` tests: presets, the analytical
//! marginal-yield-surface derivation, the scale contract. The Nonlocal
//! Granular Fluidity investigation's tests are in `sand_ngf_tests.rs`.

use super::*;

#[cfg(test)]
mod preset_tests {
    use super::*;

    /// `gravel()` sets the friction and dilatancy angles its constructor
    /// documents.
    #[test]
    fn gravel_preset_matches_its_own_documented_real_angles() {
        let g = DruckerPragerMaterial::gravel(1.0e5, 0.2);
        assert!(
            (g.friction_angle.to_degrees() - 42.0).abs() < 1.0e-4,
            "expected real, cited 42deg friction angle, got {}",
            g.friction_angle.to_degrees()
        );
        assert!(
            (g.dilatancy_angle.to_degrees() - 8.0).abs() < 1.0e-4,
            "expected real, disclosed 8deg dilatancy, got {}",
            g.dilatancy_angle.to_degrees()
        );
        // Gravel's friction angle spans 30-48 degrees; 42 must lie inside.
        assert!(
            (30.0..=48.0).contains(&g.friction_angle.to_degrees()),
            "gravel's friction angle must fall inside the real, cited 30-48deg range"
        );
    }

    /// For the cohesionless presets, `predicted_repose_angle_deg()` equals the
    /// friction angle each preset documents (Coulomb 1776, see the method),
    /// exactly.
    #[test]
    fn predicted_repose_angle_matches_friction_angle_exactly_for_every_preset() {
        let cases: [(DruckerPragerMaterial, f32); 4] = [
            (DruckerPragerMaterial::cohesionless(1.0e5, 0.2), 35.0),
            (DruckerPragerMaterial::low_friction(1.0e5, 0.2), 25.0),
            (DruckerPragerMaterial::dilatant(1.0e5, 0.2), 38.0),
            (DruckerPragerMaterial::gravel(1.0e5, 0.2), 42.0),
        ];
        for (material, expected_deg) in cases {
            let predicted = material.predicted_repose_angle_deg();
            assert!(
                (predicted - expected_deg).abs() < 1.0e-3,
                "expected predicted_repose_angle_deg()={expected_deg}, got {predicted}"
            );
        }
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::particle::Particles;

    /// Isolates whether `project()` itself matches the analytically-derived 2D
    /// Mohr-Coulomb marginal-yield condition, bypassing MPM's grid/transfer pipeline
    /// entirely (no P2G, no gravity, no free surface -- a single particle, a single
    /// hand-built deformation gradient, called directly).
    ///
    /// Derivation: converting this 2D log-strain DP
    /// return mapping into principal Cauchy stress shows elastic moduli cancel exactly,
    /// giving a universal relation sin(phi_eff) = sqrt(2) * alpha(q), independent of
    /// lambda/mu. For the default Klar 2016 params at phi_in=35 deg, alpha(q_init) =
    /// 0.386019, predicting phi_eff = 33.087 deg.
    ///
    /// This test builds a deformation gradient at EXACTLY that marginal angle and checks:
    /// slightly inside (less shear) => elastic (no change). slightly outside (more shear)
    /// => plastic (state changes). If this holds, the constitutive code matches the math
    /// and the repose-angle gap lives in MPM's grid transfer, not here.
    /// Builds a strain state whose underlying STRESS state (sigma_i = 2*mu*eps_i +
    /// lambda*tr(eps)) sits at exactly Mohr-Coulomb angle `phi_test_deg`. Strain-space
    /// and stress-space deviatoric/volumetric ratios differ by the elastic `ratio` factor
    /// (dev(stress)/-tr(stress) = (1/ratio) * dev(strain)/-tr(strain)), so this must
    /// multiply by `ratio`, not just `sin(phi)/sqrt(2)` directly in strain space.
    fn marginal_state_at_phi_eff(ratio: f32, trace: f32, phi_test_deg: f32) -> (Vec2, f32) {
        let phi_test = phi_test_deg.to_radians();
        let dev_norm = -trace * ratio * phi_test.sin() / std::f32::consts::SQRT_2;
        let diff = dev_norm * std::f32::consts::SQRT_2; // |eps1 - eps2|
        let eps1 = (trace + diff) * 0.5;
        let eps2 = (trace - diff) * 0.5;
        (Vec2::new(eps1.exp(), eps2.exp()), dev_norm)
    }

    fn run_one_step(sand: &DruckerPragerMaterial, sigma: Vec2, q: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = q;
        let mut particles = Particles::from(vec![p]);
        sand.update_particle(&mut particles.update_ctx(0), 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    fn run_rate_step(
        sand: &DruckerPragerMaterial,
        particles: &mut Particles,
        velocity_gradient: Mat2,
        dt: f32,
    ) {
        let mut ctx = particles.update_ctx(0);
        *ctx.velocity_gradient = velocity_gradient;
        sand.update_particle(&mut ctx, dt);
    }

    fn rate_particle(f: Mat2, sand: &DruckerPragerMaterial) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = sand.friction_residual / sand.hardening_peak;
        Particles::from(vec![p])
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

    #[test]
    fn rigid_rotation_preserves_volume_and_plastic_history() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        let mut particles = rate_particle(Mat2::IDENTITY, &sand);
        let q_before = particles.friction_hardening[0];
        let omega = 2.3;
        let dt = 0.2;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));

        run_rate_step(&sand, &mut particles, spin, dt);

        let expected = Mat2::from_angle(omega * dt);
        assert!(matrix_error(particles.deformation_gradient[0], expected) < 3.0e-6);
        assert!((particles.deformation_gradient[0].determinant() - 1.0).abs() < 2.0e-6);
        assert!((particles.friction_hardening[0] - q_before).abs() < 1.0e-6);
        assert!(particles.log_volume_strain[0].abs() < 1.0e-6);
    }

    #[test]
    fn opposite_subyield_rates_are_reversible_without_history_ratchet() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        // Confinement gives the DP cone a finite elastic shear domain. The
        // small isochoric rate stays comfortably inside that domain.
        let baseline = Mat2::from_diagonal(Vec2::splat(0.99));
        let mut particles = rate_particle(baseline, &sand);
        let q_before = particles.friction_hardening[0];
        let rate = Mat2::from_diagonal(Vec2::new(0.001, -0.001));

        run_rate_step(&sand, &mut particles, rate, 1.0);
        assert!((particles.friction_hardening[0] - q_before).abs() < 1.0e-6);
        assert!(particles.log_volume_strain[0].abs() < 1.0e-6);
        run_rate_step(&sand, &mut particles, -rate, 1.0);

        assert!(matrix_error(particles.deformation_gradient[0], baseline) < 3.0e-6);
        assert!((particles.friction_hardening[0] - q_before).abs() < 1.0e-6);
        assert!(particles.log_volume_strain[0].abs() < 1.0e-6);
    }

    /// A purely isotropic compression has a zero deviator but sits on the
    /// cone's axis, inside it: Klar et al. 2016 (sec. 7.1) return it
    /// unchanged (their Case I is tested before the tip's Case II). Sent to
    /// the tip instead, it came back as `F = I`, stress-free, with its
    /// compression booked as plastic `log_volume_strain`.
    #[test]
    fn isotropic_compression_at_rest_stays_elastic() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        let compressed = Mat2::from_diagonal(Vec2::splat(0.99));
        let mut particles = rate_particle(compressed, &sand);
        let q_before = particles.friction_hardening[0];

        run_rate_step(&sand, &mut particles, Mat2::ZERO, 1.0);

        assert_eq!(particles.deformation_gradient[0], compressed);
        assert_eq!(particles.friction_hardening[0], q_before);
        assert_eq!(particles.log_volume_strain[0], 0.0);
    }

    #[test]
    fn nonzero_rate_crossing_yield_updates_q_and_preserves_positive_volume() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        let baseline = Mat2::from_diagonal(Vec2::splat(0.99));
        let mut particles = rate_particle(baseline, &sand);
        let q_before = particles.friction_hardening[0];
        let rate = Mat2::from_diagonal(Vec2::new(0.2, -0.2));

        run_rate_step(&sand, &mut particles, rate, 1.0);

        assert!(particles.friction_hardening[0] > q_before);
        assert!(particles.deformation_gradient[0].determinant() > 0.0);
        assert!(particles.deformation_gradient[0].is_finite());
        assert!(particles.log_volume_strain[0].is_finite());
    }

    #[test]
    fn marginal_30deg_state_does_not_yield_for_35deg_friction() {
        let sand = DruckerPragerMaterial::new(2000.0, 3000.0);
        let q_init = sand.friction_residual / sand.hardening_peak;
        let ratio = (sand.lambda + sand.mu) / sand.mu;
        let phi_eff_deg = 33.087_f32; // sqrt(2)*alpha(q_init) for phi_in=35deg

        // Comfortably INSIDE the predicted yield surface (25 deg < 33.087 deg effective).
        let (sigma_in, _) = marginal_state_at_phi_eff(ratio, -0.01, 25.0);
        let (sigma_after, q_after) = run_one_step(&sand, sigma_in, q_init);
        assert!(
            (sigma_after - sigma_in).length() < 1.0e-6,
            "25 deg state (inside 33.087 deg yield surface) should stay elastic: \
             sigma_in={sigma_in:?} sigma_after={sigma_after:?}"
        );
        assert!(
            (q_after - q_init).abs() < 1.0e-6,
            "q should not change on an elastic step: q_init={q_init} q_after={q_after}"
        );

        // Comfortably OUTSIDE the predicted yield surface (40 deg > 33.087 deg effective).
        let (sigma_out, _) = marginal_state_at_phi_eff(ratio, -0.01, 40.0);
        let (sigma_after2, q_after2) = run_one_step(&sand, sigma_out, q_init);
        assert!(
            (sigma_after2 - sigma_out).length() > 1.0e-6,
            "40 deg state (outside 33.087 deg yield surface) should yield (state should \
             change): sigma_out={sigma_out:?} sigma_after2={sigma_after2:?}"
        );
        assert!(
            q_after2 > q_init,
            "q should increase on a plastic step: q_init={q_init} q_after2={q_after2}"
        );

        println!("phi_eff prediction = {phi_eff_deg} deg (informational, not asserted directly)");
    }
}

/// Headless measurement of the Nonlocal Granular Fluidity coupling: a
/// diagnostic that reports numbers, not a pass/fail check.
///
/// Runs on `SimConfig::earth` (SI throughout) rather than the arbitrary
/// units of `tests/accuracy.rs`'s Lajeunesse benchmark, because
/// `GranularFluidityField::apply` divides by `t0_s` (seconds) and multiplies
/// by `sub_dt`. SI sand of Haeri & Skonieczny 2022's excavation case (E = 15
/// MPa, nu = 0.3; their B = 12.5 MPa gives nu = 0.3 through K = E/(3(1-2nu))),
/// with its own cohesionless baseline under the same config.
///
/// Internal (not in `tests/accuracy.rs`) because the pressure/stress-ratio
/// closure needs `svd2` and the Hencky-strain formula, both crate-internal,
/// like `marginal_yield_tests` above.
/// Ties `scale_contract`'s REV-derived grid-resolution check to this
/// material's grain diameter.
#[cfg(test)]
mod scale_contract_integration {
    use super::*;
    use crate::materials::granular::scale_contract::{
        dx_in_valid_granular_range, granular_dx_window,
    };

    #[test]
    fn lp_cell_size_validity_for_real_sand_scene_scales() {
        const CELL_M: f32 = 0.01;

        // A 1m macro feature (real terrain scale) must have a genuine,
        // non-empty valid REV window for real dry-sand grain size --
        // otherwise no `dx` could ever make this material a valid continuum
        // at any resolution, which would be a modeling dead end.
        let window = granular_dx_window(GRAIN_DIAMETER_M, 1.0);
        assert!(
            window.is_some(),
            "a 1m macro feature should have a valid REV window for grain_diameter_m={GRAIN_DIAMETER_M}"
        );
        let (lo, hi) = window.unwrap();
        assert!(lo < hi);

        // Informational, not asserted: callers decide what a `false` means
        // (see `scale_contract`). Reports whether small collapsed-pile scenes
        // (cell_m = 0.01, pile height ~0.12 m) sit inside the valid window.
        let small_pile_valid = dx_in_valid_granular_range(CELL_M, GRAIN_DIAMETER_M, 0.12);
        println!(
            "scale_contract: cell_m={CELL_M} grain_diameter_m={GRAIN_DIAMETER_M} \
             1m_terrain_window=({lo:.4},{hi:.4}) small_pile(0.12m)_valid={small_pile_valid}"
        );
    }
}

#[cfg(test)]
mod saturation_cohesion_tests {
    use super::*;
    use crate::materials::MaterialModel;

    /// `saturation_cohesion_coeff` starts at 0.0, so `cohesion_bonus_pa` is
    /// exactly 0.0 at any saturation.
    #[test]
    fn cohesion_bonus_is_inert_by_default() {
        let dp = DruckerPragerMaterial::cohesionless(1.0e5, 0.2);
        assert_eq!(dp.saturation_cohesion_coeff, 0.0);
        for saturation in [0.0, 0.1, 0.3, 0.5, 1.0] {
            assert_eq!(
                dp.cohesion_bonus_pa(saturation),
                0.0,
                "saturation={saturation} must be inert when saturation_cohesion_coeff==0.0"
            );
        }
    }

    /// The pendular-regime shape: rises with saturation up to
    /// `pendular_regime_ceiling`, then plateaus (no post-peak decline, see
    /// `cohesion_bonus_pa`).
    #[test]
    fn cohesion_bonus_rises_through_pendular_regime_then_plateaus() {
        let dp = DruckerPragerMaterial {
            saturation_cohesion_coeff: 1000.0,
            pendular_regime_ceiling: 0.3,
            ..DruckerPragerMaterial::cohesionless(1.0e5, 0.2)
        };

        let dry = dp.cohesion_bonus_pa(0.0);
        let damp = dp.cohesion_bonus_pa(0.15);
        let at_ceiling = dp.cohesion_bonus_pa(0.3);
        let past_ceiling = dp.cohesion_bonus_pa(0.7);
        let fully_saturated = dp.cohesion_bonus_pa(1.0);

        assert_eq!(dry, 0.0, "bone-dry sand has zero apparent cohesion");
        assert!(
            damp > dry,
            "cohesion must rise with saturation in the pendular regime"
        );
        assert!(
            (at_ceiling - dp.saturation_cohesion_coeff).abs() < 1.0e-4,
            "at the ceiling, bonus should equal the full coefficient, got {at_ceiling}"
        );
        // The post-peak decline of the literature is not modelled: it plateaus.
        assert_eq!(
            past_ceiling, at_ceiling,
            "past the pendular ceiling this simplified core plateaus, doesn't decline"
        );
        assert_eq!(fully_saturated, at_ceiling);
    }

    /// The load-bearing proof, not just a check on the raw formula in
    /// isolation: a trial strain state that WOULD yield when dry must NOT
    /// yield once wet enough, because `saturation_cohesion_term` genuinely
    /// raises the yield threshold inside `project()` -- confirms the wiring
    /// (ProjectInputs -> project()'s yield check) actually works end to end,
    /// not just that the standalone formula returns a plausible number.
    #[test]
    fn wet_sand_resists_yielding_that_dry_sand_would_not() {
        let dp = DruckerPragerMaterial {
            saturation_cohesion_coeff: 5.0e4,
            pendular_regime_ceiling: 0.3,
            ..DruckerPragerMaterial::cohesionless(1.0e5, 0.2)
        };

        // A marginal trial state: small deviatoric strain, near-zero
        // trace, chosen so the dry case yields (gamma > 0) but
        // isn't buried so deep past the surface that the cohesion bonus
        // couldn't plausibly matter.
        let sigma = Vec2::new(1.003, 0.997);

        let dry_inputs = |cohesion_bonus_pa: f32| ProjectInputs {
            sigma,
            log_volume_strain: 0.0,
            q: 0.0,
            dt: 0.01,
            nonlocal_fluidity: 0.0,
            strain_rate_norm: 0.0,
            cosserat_curvature: Vec2::ZERO,
            cohesion_bonus_pa,
            eps_pl_vol_pradhana: 0.0,
        };

        let dry_result = dp.project(dry_inputs(dp.cohesion_bonus_pa(0.0)));
        assert!(
            dry_result.is_some(),
            "test setup invalid: dry case must actually yield for this to be a real check"
        );

        let wet_result = dp.project(dry_inputs(dp.cohesion_bonus_pa(0.3)));
        assert!(
            wet_result.is_none(),
            "wet sand with real apparent cohesion must resist the SAME trial strain \
             that yields dry sand -- if this fails, the cohesion bonus isn't reaching \
             the yield check"
        );
    }
}

/// `MaterialModel::current_friction_coefficient` (Material-Induced Boundary
/// Friction, Blatny & Gaume 2025) for this material.
#[cfg(test)]
mod current_friction_coefficient_tests {
    use super::*;
    use crate::materials::MaterialModel;

    fn particle_with_q(q: f32) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = q;
        Particles::from(vec![p])
    }

    /// At the default `compaction_sensitivity = 0.0`,
    /// `current_friction_coefficient` equals a hand-called `tan(phi(q, 0.0,
    /// 0.0))`, not `alpha(q, 0.0)` (the cone's coefficient, see the method).
    /// Checked at `q = 0.696`, the `cohesionless` preset's rest hardening
    /// (`friction_residual/hardening_peak`), against `tan(35deg) ~= 0.700`.
    #[test]
    fn matches_tan_phi_at_zero_compaction_sensitivity() {
        let dp = DruckerPragerMaterial::cohesionless(1.0e5, 0.2);
        assert_eq!(
            dp.compaction_sensitivity, 0.0,
            "test assumes the real default"
        );
        for q in [0.0, 0.3, 0.696, 1.5] {
            let particles = particle_with_q(q);
            let expected = dp.phi(q, 0.0, 0.0).tan();
            let got = dp.current_friction_coefficient(&particles, 0);
            assert_eq!(
                got,
                Some(expected),
                "current_friction_coefficient must match tan(phi(q, 0.0, 0.0)) exactly at q={q}"
            );
        }
        // At the rest state q = friction_residual/hardening_peak, phi(q) =
        // friction_angle exactly (see `init_particle`): 35 degrees for
        // `cohesionless`, so tan(phi) lands near 0.700, not alpha's ~0.386.
        let q_rest = dp.friction_residual / dp.hardening_peak;
        let at_rest = dp
            .current_friction_coefficient(&particle_with_q(q_rest), 0)
            .unwrap();
        assert!(
            (at_rest - 0.700).abs() < 1.0e-3,
            "at this preset's own real 35deg rest friction angle, tan(phi) must be ~0.700 \
             (matching FrictionBoundary's own real tan(35deg) default), got {at_rest}"
        );
    }

    /// With `compaction_sensitivity != 0.0` the coefficient needs a `trace`
    /// this does not recompute, so it returns `None` (the boundary keeps its
    /// own coefficient).
    #[test]
    fn returns_none_when_compaction_sensitivity_is_nonzero() {
        let mut dp = DruckerPragerMaterial::cohesionless(1.0e5, 0.2);
        dp.compaction_sensitivity = 0.1;
        let particles = particle_with_q(0.3);
        assert_eq!(dp.current_friction_coefficient(&particles, 0), None);
    }
}

/// Controlled measurement of `use_pradhana` on a synthetic particle, not the
/// poured-pile scene (`tests/accuracy.rs`'s `sand_pile_built_by_slow_pour_*`,
/// expensive and with a separate confound): checks the mechanism's sign
/// first.
///
/// A first design shifted the tension-cutoff trigger by an accumulated debt
/// and overcorrected: under sustained load the baseline settles at
/// `log_volume_strain = +0.002`, that design drifted to -0.625 and still
/// falling. Blatny's reference has four branches, one of them shear yield
/// with volumetric correction; this material's two (tension cutoff and
/// trace-preserving shear yield, "Case III") give the debt nowhere to live.
///
/// The current design acts on the projection: every tension-cutoff firing
/// sets `log_volume_strain` to `ln(prev_det)` with no memory (the
/// projection itself matches `tmp/sparkl`'s textbook DP), so
/// `eps_pl_vol_pradhana` is a bounded flag: the first firing since the
/// particle was last non-yielding applies the correction and sets it; later
/// firings are no-ops (`ProjectedBranch::DebtBlocked`) until a non-yielding
/// step. Under the same sustained load `corrected` settles at
/// `log_volume_strain ~= 1.5e-8` against the baseline's `+0.002`
/// (`pradhana_holds_near_zero_volumetric_drift_under_sustained_load`).
///
/// Sustained load is one long compaction cycle, where a per-cycle flag
/// helps; a poured pile has separate impacts with rest between them, which
/// `pradhana_effect_across_repeated_separate_impact_episodes` measures.
#[cfg(test)]
mod pradhana_correction_tests {
    use super::*;
    use crate::materials::MaterialModel;
    use glam::Mat2;

    fn fresh_particle(dp: &DruckerPragerMaterial) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.deformation_gradient = Mat2::IDENTITY;
        dp.init_particle(&mut p);
        Particles::from(vec![p])
    }

    /// Drives one particle through an anisotropic expansion impact (a purely
    /// isotropic `diag(a,a)` keeps `dev_norm` at exactly 0.0 the whole run,
    /// so nothing but the cutoff ever acts on it), then holds it under
    /// sustained anisotropic compression (not `L=0`, which leaves `F` at the
    /// tension cutoff's rotation-only output: zero strain, an elastic step
    /// every substep, nothing for the debt to act on).
    /// Returns the full `(log_volume_strain, friction_hardening)` trajectory.
    fn run_impact_then_sustained_load(use_pradhana: bool, settle_steps: usize) -> Vec<(f32, f32)> {
        let dp = DruckerPragerMaterial {
            use_pradhana,
            ..DruckerPragerMaterial::cohesionless(1.0e5, 0.2)
        };
        let mut particles = fresh_particle(&dp);
        let dt = 0.001;
        let impact = Mat2::from_cols(Vec2::new(50.0, 5.0), Vec2::new(5.0, 45.0));
        let sustained_compression = Mat2::from_cols(Vec2::new(-3.0, 0.4), Vec2::new(0.4, -2.5));
        let mut log = Vec::new();
        for _ in 0..5 {
            particles.velocity_gradient[0] = impact;
            dp.update_particle(&mut particles.update_ctx(0), dt);
            log.push((
                particles.log_volume_strain[0],
                particles.friction_hardening[0],
            ));
        }
        for _ in 0..settle_steps {
            particles.velocity_gradient[0] = sustained_compression;
            dp.update_particle(&mut particles.update_ctx(0), dt);
            log.push((
                particles.log_volume_strain[0],
                particles.friction_hardening[0],
            ));
        }
        log
    }

    /// Reports the divergence of the bounded-flag design under sustained load:
    /// `corrected` settles at floating-point zero, below the baseline's small
    /// drift.
    #[test]
    fn pradhana_holds_near_zero_volumetric_drift_under_sustained_load() {
        let baseline = run_impact_then_sustained_load(false, 200);
        let corrected = run_impact_then_sustained_load(true, 200);
        let (baseline_final, corrected_final) = (
            baseline.last().copied().unwrap(),
            corrected.last().copied().unwrap(),
        );
        println!(
            "baseline final (lvs, q) = {baseline_final:?}, corrected final (lvs, q) = {corrected_final:?}"
        );
        assert!(
            baseline_final.0.abs() < 0.01,
            "test setup invalid: baseline must reach a genuine near-zero elastic \
             equilibrium under sustained load for this comparison to be meaningful, \
             got log_volume_strain={}",
            baseline_final.0
        );
        assert!(
            corrected_final.0.abs() < baseline_final.0.abs() + 1.0e-4,
            "the bounded-flag Pradhana design should hold volumetric drift AT LEAST as \
             well as the uncorrected baseline under sustained load, not overcorrect past \
             it (that was the FIRST, already-reverted design's own real failure mode) -- \
             baseline={}, corrected={}",
            baseline_final.0,
            corrected_final.0
        );
    }

    /// A poured pile's impacts are separate events with rest between them,
    /// not one compaction cycle: does the per-cycle flag still help, or does
    /// each impact get a fresh free correction? Drives one particle through
    /// several impact-then-rest episodes and compares the total
    /// `log_volume_strain` drift, baseline against corrected.
    ///
    /// Each rest phase starts with a brief compression (the gradient that
    /// brings the particle to elastic equilibrium by step ~91 in the
    /// sustained-load test) before `L = 0`, so the rest holds a loaded
    /// elastic state rather than the stress-free pure rotation the tension
    /// cutoff leaves.
    #[test]
    #[ignore = "premise no longer holds: the baseline it compares against was f32 round-off in the F product. Baseline log_volume_strain over 15 episodes read a small positive number with the plain product, -2.19e-8 with the carried-volume rescale and +1.91e-8 with the exp - I form of advance_deformation_gradient: round-off of either sign, not the physical volume gain Pradhana corrects. Needs a scene where that gain is real."]
    fn pradhana_effect_across_repeated_separate_impact_episodes() {
        fn run_repeated_episodes(use_pradhana: bool, episodes: usize) -> f32 {
            let dp = DruckerPragerMaterial {
                use_pradhana,
                ..DruckerPragerMaterial::cohesionless(1.0e5, 0.2)
            };
            let mut particles = fresh_particle(&dp);
            let dt = 0.001;
            let impact = Mat2::from_cols(Vec2::new(50.0, 5.0), Vec2::new(5.0, 45.0));
            let settle_to_equilibrium = Mat2::from_cols(Vec2::new(-3.0, 0.4), Vec2::new(0.4, -2.5));
            for _ in 0..episodes {
                for _ in 0..5 {
                    particles.velocity_gradient[0] = impact;
                    dp.update_particle(&mut particles.update_ctx(0), dt);
                }
                for _ in 0..120 {
                    particles.velocity_gradient[0] = settle_to_equilibrium;
                    dp.update_particle(&mut particles.update_ctx(0), dt);
                }
                // Genuine full rest -- safe now that F sits at a real,
                // non-degenerate elastic equilibrium (dev_norm != 0), long
                // enough to confirm a non-yielding steady state,
                // matching a pour's own real settling between batches.
                for _ in 0..380 {
                    particles.velocity_gradient[0] = Mat2::ZERO;
                    dp.update_particle(&mut particles.update_ctx(0), dt);
                }
            }
            particles.log_volume_strain[0]
        }

        let baseline = run_repeated_episodes(false, 15);
        let corrected = run_repeated_episodes(true, 15);
        println!(
            "15 separate impact-then-full-rest episodes: baseline log_volume_strain={baseline:.6}, \
             corrected={corrected:.6}"
        );
        assert!(
            baseline > 0.0,
            "test setup invalid: baseline must show real positive volume-gain drift across \
             repeated episodes for this comparison to be meaningful, got {baseline}"
        );
    }
}

/// Where Drucker-Prager's per-particle plasticity update spends its time.
/// A 60x60-cell block of sand falls and spreads for 40 steps, so its
/// particles carry real deformation states; then, on one thread, the whole
/// `update_particle` and each of its pieces (the F advance, the SVD,
/// `project`, the rebuild of F) are timed alone over every particle, and
/// `project`'s branches are counted. Optimised profile only:
///
///   cargo test --profile quick --lib dp_update_cost_breakdown -- --ignored --nocapture
///
/// Knob: DP_PROBE_STEPS (40), the steps run before timing. The block lands
/// at about step 73; 80 catches the impact, 300 a settled pile.
///
/// First reading: the whole update 315-420 ns per particle, of which the F
/// advance 41, `svd2` 37-104, `project` 80-205, the rebuild 6. Its branch
/// counts were the bigger finding: after the impact the tension cutoff takes
/// 95 % of the particles each substep, and still 70 % of a settled pile at
/// step 300. 83 % of those particles carry a positive `log_volume_strain`,
/// and for 40 % of all particles it alone tips an elastically compressed
/// state into the cutoff: each cutoff adds the expansion it removed to that
/// history (`volume_correction = 1`), so the particle stays stress-free until
/// a compression pays it back.
///
/// Controls on the repose column and the Lube column collapse (64- and
/// 192-cell grids, 128-substep cap): defaults 26.3 deg and a runout 0.49 of
/// the experiment's; `volume_correction = 0` 26.9 deg and 0.49; `use_pradhana`
/// 76.4 deg and a column that does not collapse (runout -0.01). The repeated
/// cutoff is what lets this sand flow at all, not the cause of either gap.
#[cfg(test)]
mod dp_update_cost_breakdown {
    use super::*;
    use crate::materials::svd::svd2;
    use crate::materials::utils::advance_deformation_gradient;
    use crate::{FrictionBoundary, SimConfig, Simulation, SpawnRegion};
    use glam::IVec2;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    #[ignore = "timing probe kept for reruns, not part of the CI suite"]
    fn dp_update_cost_breakdown() {
        const REPS: usize = 20;
        let config = SimConfig {
            max_substeps_per_step: 256,
            ..SimConfig::standard(128, 0.1, Vec2::new(0.0, -0.3))
        };
        let block = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(60, 60),
            box_center: Vec2::new(64.0, 40.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        let mut sim = Simulation::new(config, block)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let steps: usize = std::env::var("DP_PROBE_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40);
        for _ in 0..steps {
            sim.step();
        }
        let substeps = sim.diagnostics_snapshot().substeps_last_step.max(1);
        let dt = config.dt / substeps as f32;
        let particles = sim.particles().clone();
        let n = particles.len();
        let per_particle_ns = |total: f64| total * 1e9 / (REPS * n) as f64;

        let mut whole = 0.0;
        for _ in 0..REPS {
            let mut copy = particles.clone();
            let t0 = Instant::now();
            for i in 0..n {
                sand.update_particle(&mut copy.update_ctx(i), dt);
            }
            whole += t0.elapsed().as_secs_f64();
        }

        let t0 = Instant::now();
        for _ in 0..REPS {
            for (&f, &l) in particles
                .deformation_gradient
                .iter()
                .zip(&particles.velocity_gradient)
            {
                black_box(advance_deformation_gradient(f, dt * l));
            }
        }
        let advance = t0.elapsed().as_secs_f64();

        let trials: Vec<Mat2> = particles
            .deformation_gradient
            .iter()
            .zip(&particles.velocity_gradient)
            .map(|(&f, &l)| advance_deformation_gradient(f, dt * l))
            .collect();
        let t0 = Instant::now();
        for _ in 0..REPS {
            for &f in &trials {
                black_box(svd2(f));
            }
        }
        let svd = t0.elapsed().as_secs_f64();

        let decomposed: Vec<(Mat2, Vec2, Mat2)> = trials.iter().map(|&f| svd2(f)).collect();
        let inputs = |i: usize| ProjectInputs {
            sigma: decomposed[i].1,
            log_volume_strain: particles.log_volume_strain[i],
            q: particles.friction_hardening[i],
            dt,
            nonlocal_fluidity: 0.0,
            strain_rate_norm: 0.0,
            cosserat_curvature: Vec2::ZERO,
            cohesion_bonus_pa: 0.0,
            eps_pl_vol_pradhana: 0.0,
        };
        let t0 = Instant::now();
        for _ in 0..REPS {
            for i in 0..n {
                black_box(sand.project(inputs(i)));
            }
        }
        let project = t0.elapsed().as_secs_f64();
        let (mut elastic, mut tension, mut shear, mut other) = (0usize, 0usize, 0usize, 0usize);
        for i in 0..n {
            match sand.project(inputs(i)) {
                None => elastic += 1,
                Some((_, _, ProjectedBranch::TensionCutoff)) => tension += 1,
                Some((_, _, ProjectedBranch::ShearYield)) => shear += 1,
                Some(_) => other += 1,
            }
        }

        let t0 = Instant::now();
        for _ in 0..REPS {
            for &(u, sigma, vt) in &decomposed {
                let s = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
                black_box(u * s * vt);
            }
        }
        let rebuild = t0.elapsed().as_secs_f64();

        println!("{n} particles of sand after {steps} steps, substep {dt:.2e}, ns per particle:");
        println!("  whole update_particle     {:7.1}", per_particle_ns(whole));
        println!(
            "  advance F                 {:7.1}",
            per_particle_ns(advance)
        );
        println!("  svd2                      {:7.1}", per_particle_ns(svd));
        println!(
            "  project                   {:7.1}",
            per_particle_ns(project)
        );
        println!(
            "  rebuild F = U S V^T       {:7.1}",
            per_particle_ns(rebuild)
        );
        println!(
            "  project branches: elastic {elastic}, tension cutoff {tension}, shear yield {shear}, other {other}"
        );
        // Why tension: `project` tests the trace of ln(sigma) plus the
        // carried volumetric history `log_volume_strain`.
        let (mut history_positive, mut tension_from_history) = (0usize, 0usize);
        let mut history: Vec<f32> = particles.log_volume_strain.clone();
        for (i, &(_, sigma, _)) in decomposed.iter().enumerate() {
            let lvs = particles.log_volume_strain[i];
            if lvs > 0.0 {
                history_positive += 1;
            }
            let s = sigma
                .abs()
                .max(Vec2::splat(crate::materials::utils::LOG_CLAMP));
            let elastic_trace = s.x.ln() + s.y.ln();
            if elastic_trace <= 0.0 && elastic_trace + lvs > 0.0 {
                tension_from_history += 1;
            }
        }
        history.sort_by(f32::total_cmp);
        println!(
            "  log_volume_strain > 0 on {history_positive}; tension only through it on {tension_from_history}; median {:.2e}, p10 {:.2e}, p90 {:.2e}",
            history[n / 2],
            history[n / 10],
            history[9 * n / 10]
        );
    }
}

/// Where a settled pile keeps creeping, and in which state. The same block
/// as `dp_update_cost_breakdown` falls, spreads and settles for
/// `CREEP_PROBE_STEPS` steps; then every particle is binned by its depth
/// below the pile's local surface (the highest particle within half a cell of
/// its x), and each depth band reports how many particles take each branch
/// of `project` and their mean speed. Knobs: CREEP_PROBE_STEPS (400),
/// CREEP_PROBE_VOLUME_CORRECTION (the material's 1.0). Ignored, read-only:
///
///   cargo test --profile quick --lib settled_pile_creep_by_depth -- --ignored --nocapture
///
/// First reading, step 400: speed does not depend on the branch within a
/// band (1.4e-3 cells/s in the top half cell for all three, 4-6e-4 below 8
/// cells), so the whole pile creeps, about 3 times faster at its surface,
/// rather than its stress-free particles sliding. The tension cutoff takes
/// 66 % of the particles even more than 8 cells deep, under the pile's
/// weight. With `volume_correction = 0` (no history carried from a cutoff)
/// only 1.2 % of the deep particles take the cutoff, 91 % stay elastic, and
/// the deep pile creeps about 4 times slower (elastic particles 9.6e-5
/// cells/s); the branch then sets the speed, cutoff and shear particles
/// moving fastest. The deep cutoff is the carried history, not creep noise.
#[cfg(test)]
mod settled_pile_creep_by_depth {
    use super::*;
    use crate::materials::svd::svd2;
    use crate::materials::utils::advance_deformation_gradient;
    use crate::{FrictionBoundary, SimConfig, Simulation, SpawnRegion};
    use glam::IVec2;

    #[test]
    #[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
    fn settled_pile_creep_by_depth() {
        let config = SimConfig {
            max_substeps_per_step: 256,
            ..SimConfig::standard(128, 0.1, Vec2::new(0.0, -0.3))
        };
        let block = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(60, 60),
            box_center: Vec2::new(64.0, 40.0),
            material_id: 0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        if let Some(c) = std::env::var("CREEP_PROBE_VOLUME_CORRECTION")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            sand.volume_correction = c;
        }
        let mut sim = Simulation::new(config, block)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
        let steps: usize = std::env::var("CREEP_PROBE_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400);
        for _ in 0..steps {
            sim.step();
        }
        let substeps = sim.diagnostics_snapshot().substeps_last_step.max(1);
        let dt = config.dt / substeps as f32;
        let p = sim.particles();
        let n = p.len();

        // Local surface: highest particle per half-cell column.
        let columns = 2 * config.grid_res;
        let mut surface = vec![f32::MIN; columns];
        for x in &p.x {
            let c = ((x.x * 2.0) as usize).min(columns - 1);
            surface[c] = surface[c].max(x.y);
        }
        // Depth bands in cells below the local surface.
        const BANDS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, f32::INFINITY];
        let mut count = [[0usize; 3]; 6]; // [band][elastic, tension, shear]
        let mut speed = [[0.0f64; 3]; 6];
        for i in 0..n {
            let c = ((p.x[i].x * 2.0) as usize).min(columns - 1);
            let depth = surface[c] - p.x[i].y;
            let band = BANDS.iter().position(|&b| depth < b).unwrap_or(5);
            let trial = advance_deformation_gradient(
                p.deformation_gradient[i],
                dt * p.velocity_gradient[i],
            );
            let (_, sigma, _) = svd2(trial);
            let branch = match sand.project(ProjectInputs {
                sigma,
                log_volume_strain: p.log_volume_strain[i],
                q: p.friction_hardening[i],
                dt,
                nonlocal_fluidity: 0.0,
                strain_rate_norm: 0.0,
                cosserat_curvature: Vec2::ZERO,
                cohesion_bonus_pa: 0.0,
                eps_pl_vol_pradhana: 0.0,
            }) {
                None => 0,
                Some((_, _, ProjectedBranch::TensionCutoff)) => 1,
                Some(_) => 2,
            };
            count[band][branch] += 1;
            speed[band][branch] += f64::from(p.v[i].length());
        }
        println!(
            "{n} particles after {steps} steps; speed in cells/s; depth below the local surface:"
        );
        let mut lower = 0.0f32;
        for (band, &upper) in BANDS.iter().enumerate() {
            let cell = |k: usize| {
                let c = count[band][k];
                let s = if c > 0 {
                    speed[band][k] / c as f64
                } else {
                    0.0
                };
                format!("{c:5} at {s:.2e}")
            };
            println!(
                "  {lower:4.1}-{upper:4.1}: elastic {}, tension cutoff {}, shear yield {}",
                cell(0),
                cell(1),
                cell(2)
            );
            lower = upper;
        }
    }
}

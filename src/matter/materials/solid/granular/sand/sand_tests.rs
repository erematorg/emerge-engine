//! Core `DruckerPragerMaterial` test suite -- split out of `sand.rs` (2026-08-05),
//! same reasoning as `elastic/elastic_tests.rs`: the constitutive-model file should
//! read as the model, not scroll past its own test suites to get there. This file
//! holds the material's own always-relevant correctness checks (presets, the
//! analytical marginal-yield-surface derivation, scale-contract integration).
//! The much larger, distinct Nonlocal Granular Fluidity research investigation's
//! own tests live separately in `sand_ngf_tests.rs` -- a genuinely different
//! research topic, not force-merged into one file just because both touch sand.

use super::*;

#[cfg(test)]
mod preset_tests {
    use super::*;

    /// Real, direct check that `gravel()` sets the exact real, cited
    /// friction/dilatancy angles documented on the constructor -- not just
    /// "compiles and returns something."
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
        // Real, standard geotechnical fact this preset is grounded in
        // (verified 2026-08-04): gravel's real friction angle spans
        // 30-48 degrees -- 42 must sit inside that real range, not just
        // match itself.
        assert!(
            (30.0..=48.0).contains(&g.friction_angle.to_degrees()),
            "gravel's friction angle must fall inside the real, cited 30-48deg range"
        );
    }

    /// Real, closed-form check: for the standard cohesionless presets,
    /// `predicted_repose_angle_deg()` must exactly equal the real friction
    /// angle each preset already documents (Coulomb 1776's own real
    /// relationship for a cohesionless pile, see the method's own doc) --
    /// not an approximation, an exact identity by construction.
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
    /// entirely (no P2G, no gravity, no free surface — a single particle, a single
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
    /// and the real repose-angle gap lives in MPM's grid transfer, not here.
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

/// Real, headless first measurement of the Nonlocal Granular Fluidity
/// coupling (Phase 3 of the NGF plan) -- DIAGNOSTIC, not yet a pass/fail
/// regression: the real outcome isn't known ahead of time, so this reports
/// honest numbers rather than asserting a threshold picked in advance.
///
/// Runs on `SimConfig::earth` (real SI throughout: `dx_meters`, `dt_seconds`,
/// gravity) rather than the arbitrary-unit convention `tests/accuracy.rs`'s
/// own Lajeunesse benchmark uses, because `GranularFluidityField::apply`'s
/// reaction step divides by `t0_s` (real seconds) and multiplies by
/// `sub_dt` -- mixing a real-seconds `t0_s` against an arbitrary substep
/// time unit is a genuine units error. Real SI sand properties (E=15MPa,
/// nu=0.3, matching Haeri & Skonieczny 2022's own Excavation case,
/// cross-checked internally consistent: their bulk modulus B=12.5MPa at
/// E=15MPa implies nu=0.3 exactly via K=E/(3(1-2nu))). Since this is a
/// genuinely different (real-SI) scene than `tests/accuracy.rs`'s own
/// Lajeunesse benchmark, this test measures its own fresh cohesionless
/// baseline under the same config rather than reusing that benchmark's
/// arbitrary-unit numbers.
///
/// Internal (not `tests/accuracy.rs`) because the pressure/stress-ratio
/// closure needs `svd2` and the real Hencky-strain formula, both
/// crate-internal -- same reason `marginal_yield_tests` above lives here.
/// Ties `scale_contract`'s REV-derived grid-resolution check to this
/// material's own real grain diameter, so the module is exercised against a
/// real material's real constant rather than sitting wired to nothing but
/// its own standalone unit tests.
#[cfg(test)]
mod scale_contract_integration {
    use super::*;
    use crate::materials::solid::granular::scale_contract::{
        dx_in_valid_granular_range, granular_dx_window,
    };

    #[test]
    fn lp_cell_size_validity_for_real_sand_scene_scales() {
        const CELL_M: f32 = 0.01;

        // A 1m macro feature (real terrain scale) must have a genuine,
        // non-empty valid REV window for real dry-sand grain size --
        // otherwise no `dx` could ever make this material a valid continuum
        // at any resolution, which would be a real modeling dead end.
        let window = granular_dx_window(GRAIN_DIAMETER_M, 1.0);
        assert!(
            window.is_some(),
            "a 1m macro feature should have a valid REV window for grain_diameter_m={GRAIN_DIAMETER_M}"
        );
        let (lo, hi) = window.unwrap();
        assert!(lo < hi);

        // Informational, not asserted pass/fail -- per `scale_contract`'s own
        // doc, callers decide what to do with a `false` result. Reports
        // whether this session's own small collapsed-pile scenes (cell_m=0.01,
        // pile height ~0.12m) sit inside the physically valid window.
        let small_pile_valid = dx_in_valid_granular_range(CELL_M, GRAIN_DIAMETER_M, 0.12);
        println!(
            "scale_contract: cell_m={CELL_M} grain_diameter_m={GRAIN_DIAMETER_M} \
             1m_terrain_window=({lo:.4},{hi:.4}) small_pile(0.12m)_valid={small_pile_valid}"
        );
    }
}

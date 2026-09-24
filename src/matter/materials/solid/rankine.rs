use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BrittleProps, FromSI, scale_lame, scale_stress};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, RANKINE_MIN_RESIDUAL_TENSILE_FRACTION, elastic_wave_dt, hencky_strains, lame_from_young,
    rankine_damage_saturation_point, reconstruct_f, stress_to_hencky,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{ParticleUpdateCtx, Particles};

/// Rankine (maximum principal stress) elastoplastic material — brittle tensile failure.
///
/// Elastic response: corotated linear elastic (same as DruckerPragerMaterial / VonMisesMaterial).
/// Yield criterion: max(τ₁, τ₂) ≤ σ_t_eff, where τᵢ are principal Kirchhoff stresses and
///   σ_t_eff = max(tensile_strength · exp(−softening_rate · damage), tensile_strength · 5%)
///   (exponential softening, floored at a small residual so damage saturates under
///   sustained loading instead of ratcheting forever -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`).
///
/// Return mapping: when a principal stress exceeds σ_t_eff, it is projected back to the
/// tensile cutoff surface; the remaining stress component is unaffected (1D projection).
/// Biaxial tension (both τ₁ > σ_t AND τ₂ > σ_t) projects at the corner — both set to σ_t.
///
/// Damage accumulates in `Particle::friction_hardening` (repurposed as damage), bounded
/// above by `rankine_damage_saturation_point(softening_rate)` -- the point past which
/// `t_eff` is already at its residual floor, so further stress cannot lower it any
/// more and additional accumulation would be pure bookkeeping, not physical (see
/// `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION` doc). `softening_rate <= 0` (hard cutoff,
/// no softening) has no saturation point -- damage can grow unbounded in that case,
/// same as before.
/// Softening reduces effective tensile strength exponentially toward a small residual.
///
/// Suitable for: brittle rock, bone, eggshell, chitin, ice with fracture.
/// At zero softening rate: perfect tensile cutoff (material can never exceed σ_t).
///
/// References: Rankine 1876 (original criterion); Wolper et al. 2019 (MPM brittle fracture);
/// sparkl `RankinePlasticity` (Rust open-source reference, Apache-2.0).
#[derive(Debug, Clone, Copy)]
pub struct RankineMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Maximum tensile principal Kirchhoff stress (compressive stress is unlimited).
    /// Rock: 1e3–1e4. Bone: 1e4–1e5. Ice: 1e2–1e3.
    pub tensile_strength: f32,
    /// Exponential softening rate. 0.0 = no softening (hard cutoff).
    /// Positive values reduce σ_t as damage accumulates.
    /// Typical: 0.5–5.0 — higher = more brittle (strength collapses fast after first crack).
    pub softening_rate: f32,
}

impl RankineMaterial {
    pub const fn new(lambda: f32, mu: f32, tensile_strength: f32, softening_rate: f32) -> Self {
        Self {
            lambda,
            mu,
            tensile_strength,
            softening_rate,
        }
    }

    pub fn from_young_modulus(
        young_modulus: f32,
        poisson_ratio: f32,
        tensile_strength: f32,
        softening_rate: f32,
    ) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, tensile_strength, softening_rate)
    }

    /// Brittle rock regime: tensile strength as a real FRACTION of the caller's own
    /// `young_modulus`, not a hardcoded absolute number -- a fixed absolute value only
    /// "means" rock at one specific implicit E, silently wrong at any other (a hardcoded
    /// tensile=500 gives an 18-50% tensile/E ratio at the values this engine's own tests
    /// pass it, vs. real brittle rock's tensile-to-modulus ratio of ~2-3e-4 --
    /// granite/basalt: E~50 GPa, tensile strength~10-15 MPa (Goodman 1989, "Introduction
    /// to Rock Mechanics"). Real, fast softening_rate=2.0 (brittle failure propagates
    /// quickly) unchanged.
    pub fn stiff_brittle(young_modulus: f32, poisson_ratio: f32) -> Self {
        const ROCK_TENSILE_TO_MODULUS_RATIO: f32 = 2.5e-4;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * ROCK_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Bone regime: tensile strength as a real fraction of `young_modulus`, same fix as
    /// `stiff_brittle` above. Real cortical bone tolerates a much higher tensile-to-
    /// modulus ratio than rock (tougher composite material): E~15-20 GPa, tensile
    /// strength~100-150 MPa, ratio ~7e-3 (Currey 2002, "Bones: Structure and
    /// Mechanics"). Real, slower softening_rate=1.0 (bone fails less abruptly than
    /// rock) unchanged.
    pub fn high_tensile(young_modulus: f32, poisson_ratio: f32) -> Self {
        const BONE_TENSILE_TO_MODULUS_RATIO: f32 = 7.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * BONE_TENSILE_TO_MODULUS_RATIO,
            1.0,
        )
    }

    /// Sandstone regime: sedimentary clastic rock, same ratio-not-absolute fix as
    /// `stiff_brittle`. Real E range 11.3-40 GPa (avg ~19.9 GPa), tensile strength
    /// 19.17-65.66 MPa (Xu 2016, "Characterization of Rock Mechanical Properties
    /// Using Lab Tests and Numerical Interpretation Model of Well Logs") -- huge
    /// real spread from cementation/porosity, disclosed not hidden. Representative
    /// pick near the lower/typical end of both ranges (E~20 GPa, tensile~20 MPa).
    /// Same softening_rate=2.0 as `stiff_brittle` -- still real brittle failure,
    /// no separately-cited reason to differ.
    pub fn sandstone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SANDSTONE_TENSILE_TO_MODULUS_RATIO: f32 = 1.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SANDSTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Limestone regime: sedimentary chemical rock. Real E range 4.6-12 GPa,
    /// tensile strength 18.00-38.76 MPa (same Xu 2016 source as `sandstone`) --
    /// genuinely softer AND relatively stronger-in-tension-per-modulus than
    /// sandstone, a real distinguishing feature, not the same rock renamed.
    /// Representative pick E~8 GPa, tensile~25 MPa.
    pub fn limestone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const LIMESTONE_TENSILE_TO_MODULUS_RATIO: f32 = 3.1e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * LIMESTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Shale regime: sedimentary clastic, fissile/foliated. Real E range 15-36.9 GPa
    /// (avg ~27 GPa, foliated), tensile strength ~168 MPa average ACROSS foliation
    /// (same Xu 2016 source) -- real, cited, but an HONEST, DISCLOSED limitation:
    /// real shale is strongly anisotropic (splits far more easily ALONG bedding
    /// planes than across them; the source's own "laminated shale shows lower
    /// values" note, exact number not given). This preset is isotropic (this
    /// material's yield surface has no per-particle orientation field), so it
    /// necessarily represents the ACROSS-foliation (stronger) direction -- real
    /// bedding-plane weakness is a genuinely separate, not-yet-built mechanism
    /// (see the geosphere-taxonomy memory's "anisotropic foliated rock" gap), not
    /// something this single-number preset can honestly claim to capture.
    pub fn shale(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SHALE_TENSILE_TO_MODULUS_RATIO: f32 = 6.2e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SHALE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Effective tensile strength after damage softening. Floored at a small
    /// residual fraction of virgin strength -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`
    /// doc for why an unfloored exponential decay is an unbounded damage ratchet.
    #[inline]
    fn tensile_strength_eff(&self, damage: f32) -> f32 {
        (self.tensile_strength * (-self.softening_rate * damage).exp())
            .max(self.tensile_strength * RANKINE_MIN_RESIDUAL_TENSILE_FRACTION)
    }
}

impl FromSI<BrittleProps> for RankineMaterial {
    fn from_physical(props: &BrittleProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        let ts = scale_stress(props.tensile_strength_pa, props.elastic.rho_kg_m3, config);
        Self::new(lambda, mu, ts, props.softening_rate)
    }
}

impl RankineMaterial {
    /// Rankine return mapping in 2D principal stress space.
    ///
    /// Returns (projected_tau, yielded) — `yielded` is true if any projection occurred.
    #[inline]
    fn project_stress(&self, tau: Vec2, t_eff: f32) -> (Vec2, bool) {
        let t1 = tau.x > t_eff;
        let t2 = tau.y > t_eff;
        match (t1, t2) {
            (false, false) => (tau, false),
            (true, false) => (Vec2::new(t_eff, tau.y), true),
            (false, true) => (Vec2::new(tau.x, t_eff), true),
            (true, true) => (Vec2::splat(t_eff), true), // biaxial tension corner return
        }
    }
}

impl MaterialModel for RankineMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Rankine
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }
        let r = polar_decomposition_2d(f);
        2.0 * self.mu * (f - r) * f.transpose() + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);

        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );

        let damage = *ctx.friction_hardening;
        let t_eff = self.tensile_strength_eff(damage);

        let (tau_proj, yielded) = self.project_stress(tau, t_eff);

        let sigma_new = if yielded {
            let eps_proj = stress_to_hencky(tau_proj, self.lambda, self.mu);
            let eps_trial = stress_to_hencky(tau, self.lambda, self.mu);
            *ctx.friction_hardening = (damage + (eps_trial - eps_proj).length())
                .min(rankine_damage_saturation_point(self.softening_rate));
            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        *ctx.deformation_gradient = reconstruct_f(u, sigma_new, vt);
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Rankine as u32,
            lambda: self.lambda,
            mu: self.mu,
            // tensile_strength → hardening_exponent slot (union layout, CPU-only plasticity)
            hardening_exponent: self.tensile_strength,
            // softening_rate → hardening_modulus slot
            hardening_modulus: self.softening_rate,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        )
    }

    fn needs_cpu_update(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::Particle;

    /// Isolates whether `update_particle`'s return mapping matches this
    /// material's OWN documented tensile-cutoff criterion (`max(tau1,tau2) <=
    /// t_eff`) exactly -- same discipline as `sand.rs`/`von_mises.rs`'s own
    /// `marginal_yield_tests`. `RankineMaterial` had zero test comparing its
    /// return mapping to an exact analytical prediction before this (only
    /// stability + softening-direction checks existed, confirmed via the
    /// 2026-07-07 citation audit).
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
        // (damage=0, so t_eff=tensile_strength exactly -- no floor/saturation
        // complication from the 2026-07-07 ratchet fix).
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
        // changing is real, correct coupled elasticity, not a bug (confirmed:
        // an earlier version of this test wrongly asserted sigma.y itself must
        // stay fixed, and failed -- the fix is checking the right invariant).
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
}

#[cfg(test)]
mod rock_preset_tests {
    use super::*;

    /// Real, cited rock presets must be genuinely DIFFERENT materials, not the
    /// same numbers under different names -- checks the real distinguishing
    /// feature each preset's own doc claims: limestone is softer (lower E) than
    /// sandstone AND relatively stronger in tension per unit stiffness (higher
    /// tensile-to-modulus ratio), a real geotechnical distinction (Xu 2016), not
    /// an assumption.
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

    /// All 5 Rankine presets (bone/rock family) must produce finite, positive
    /// tensile strengths at a real representative modulus -- a basic sanity floor
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
        ] {
            assert!(
                mat.tensile_strength.is_finite() && mat.tensile_strength > 0.0,
                "{name}: tensile_strength must be finite and positive, got {}",
                mat.tensile_strength
            );
        }
    }
}

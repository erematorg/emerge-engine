use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BrittleProps, FromSI, scale_lame, scale_stress};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, RANKINE_MIN_RESIDUAL_TENSILE_FRACTION, elastic_wave_dt, hencky_strains, lame_from_young,
    rankine_damage_saturation_point, reconstruct_f, stress_to_hencky,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particles;

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
    pub fn new(lambda: f32, mu: f32, tensile_strength: f32, softening_rate: f32) -> Self {
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
    /// "means" rock at one specific implicit E, silently wrong at any other (found via
    /// 2026-07-02 audit: this preset's old hardcoded tensile=500 gave an 18-50% tensile/E
    /// ratio at the values this engine's own tests pass it, vs. real brittle rock's
    /// tensile-to-modulus ratio of ~2-3e-4 -- granite/basalt: E~50 GPa, tensile
    /// strength~10-15 MPa (Goodman 1989, "Introduction to Rock Mechanics"). Real, fast
    /// softening_rate=2.0 (brittle failure propagates quickly) unchanged.
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

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particles.velocity_gradient[i])
            * particles.deformation_gradient[i];
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);

        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );

        let damage = particles.friction_hardening[i];
        let t_eff = self.tensile_strength_eff(damage);

        let (tau_proj, yielded) = self.project_stress(tau, t_eff);

        let sigma_new = if yielded {
            let eps_proj = stress_to_hencky(tau_proj, self.lambda, self.mu);
            let eps_trial = stress_to_hencky(tau, self.lambda, self.mu);
            particles.friction_hardening[i] = (damage + (eps_trial - eps_proj).length())
                .min(rankine_damage_saturation_point(self.softening_rate));
            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        particles.deformation_gradient[i] = reconstruct_f(u, sigma_new, vt);
        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
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
        mat.update_particle(&mut particles, 0, 1.0);
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
}

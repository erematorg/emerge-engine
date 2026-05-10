use glam::{Mat2, Vec2};

use crate::materials::utils::{
    MIN_J, elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f, stress_to_hencky,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particle;
use crate::materials::svd::svd2;

/// Rankine (maximum principal stress) elastoplastic material — brittle tensile failure.
///
/// Elastic response: corotated linear elastic (same as SandMaterial / VonMisesMaterial).
/// Yield criterion: max(τ₁, τ₂) ≤ σ_t_eff, where τᵢ are principal Kirchhoff stresses and
///   σ_t_eff = tensile_strength · exp(−softening_rate · damage)  (exponential softening).
///
/// Return mapping: when a principal stress exceeds σ_t_eff, it is projected back to the
/// tensile cutoff surface; the remaining stress component is unaffected (1D projection).
/// Biaxial tension (both τ₁ > σ_t AND τ₂ > σ_t) projects at the corner — both set to σ_t.
///
/// Damage accumulates in `Particle::friction_hardening` (repurposed as damage ∈ [0, ∞]).
/// Softening reduces effective tensile strength exponentially toward zero.
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
        Self { lambda, mu, tensile_strength, softening_rate }
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

    /// Brittle rock preset: high stiffness, low tensile strength, fast softening.
    pub fn rock(lambda: f32, mu: f32) -> Self {
        Self::new(lambda, mu, 500.0, 2.0)
    }

    /// Bone preset: stiff, moderate tensile strength, moderate softening.
    pub fn bone(lambda: f32, mu: f32) -> Self {
        Self::new(lambda, mu, 2000.0, 1.0)
    }

    /// Effective tensile strength after damage softening.
    #[inline]
    fn tensile_strength_eff(&self, damage: f32) -> f32 {
        self.tensile_strength * (-self.softening_rate * damage).exp()
    }

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

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= MIN_J { return Mat2::ZERO; }
        let r = polar_decomposition_2d(f);
        2.0 * self.mu * (f - r) * f.transpose() + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particle.velocity_gradient) * particle.deformation_gradient;
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);

        // Principal Kirchhoff stresses via corotated elastic stiffness tensor.
        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );

        let damage = particle.friction_hardening;
        let t_eff = self.tensile_strength_eff(damage);

        let (tau_proj, yielded) = self.project_stress(tau, t_eff);

        let sigma_new = if yielded {
            // Convert both trial and projected stresses to Hencky strain space,
            // then take the norm of the strain increment as the damage increment.
            // This gives a dimensionless, ν-independent damage rate — dividing by
            // (2µ+λ) was wrong (off by det/a² depending on ν). Strain norm is correct.
            let eps_proj = stress_to_hencky(tau_proj, self.lambda, self.mu);
            let eps_trial = stress_to_hencky(tau, self.lambda, self.mu);
            particle.friction_hardening = damage + (eps_trial - eps_proj).length();

            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        particle.deformation_gradient = reconstruct_f(u, sigma_new, vt);
        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.sync_volume_and_density(j);
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

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        elastic_wave_dt(self.lambda, self.mu, 1.0, particle.density, MIN_J, cell_width, material_cfl)
    }

    fn needs_cpu_update(&self) -> bool { true }
}

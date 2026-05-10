use glam::{Mat2, Vec2};

use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particle;
use crate::materials::svd::svd2;

/// Snow constitutive model: corotated elasticity + SVD-based plasticity.
///
/// On elastic deformation the stress is corotated linear elastic (same form as CorotatedMaterial).
/// On each step, plastic flow is applied via SVD decomposition of F:
///   - Singular values are clamped to [1 - theta_c, 1 + theta_s]
///   - The plastic Jacobian Jp accumulates the volume change that was plastically removed
///   - Elastic hardening h = exp(xi * (1 - Jp)) scales stiffness (compressed snow is stiffer)
///
/// Reference: Stomakhin et al. 2013, §4.2. Identical in sparkl, taichi128, Genesis.
#[derive(Debug, Clone, Copy)]
pub struct SnowMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Hardening exponent. Higher = more stiffness gain as snow compacts.
    pub hardening_exponent: f32,
    /// Max compression before plastic flow triggers: Δσ < 1 − θ_c → plastic. (Stomakhin 2013 θ_c)
    pub compression_limit: f32,
    /// Max stretch before plastic flow triggers: Δσ > 1 + θ_s → plastic. (Stomakhin 2013 θ_s)
    pub stretch_limit: f32,
    /// Lower bound on Jp. Prevents wave speed from exploding as h → ∞.
    pub min_plastic_jacobian: f32,
    /// Upper bound on Jp (slight stretch plasticity allowed).
    pub max_plastic_jacobian: f32,
    /// Cohesion pressure: τ += −c · max(1−Jp, 0) · I.
    /// Creates attractive stress in plastically compacted snow (Jp < 1).
    /// 0.0 = no cohesion (Stomakhin 2013 default — powder, loose snow).
    /// ~500–2000 for packed/wet snow that sticks after impact.
    pub cohesion_coeff: f32,
}

impl SnowMaterial {
    pub fn new(
        lambda: f32,
        mu: f32,
        hardening_exponent: f32,
        compression_limit: f32,
        stretch_limit: f32,
        min_plastic_jacobian: f32,
        max_plastic_jacobian: f32,
    ) -> Self {
        Self {
            lambda,
            mu,
            hardening_exponent,
            compression_limit,
            stretch_limit,
            min_plastic_jacobian,
            max_plastic_jacobian,
            cohesion_coeff: 0.0,
        }
    }

    pub fn with_cohesion(mut self, coeff: f32) -> Self {
        self.cohesion_coeff = coeff;
        self
    }

    /// Construct with Stomakhin 2013 default plasticity params and E/ν inputs.
    ///
    /// Canonical: E = 1.4e5, ν = 0.2 — matches MPM2D reference and sparkl snow demos.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(
            lambda,
            mu,
            10.0,   // hardening_exponent ξ — Stomakhin 2013 §4
            0.025,  // compression_limit θ_c — Stomakhin 2013 §4 canonical (was 0.02)
            0.0075, // stretch_limit θ_s — Stomakhin 2013 §4 canonical (was 0.006)
            0.6,  // min_plastic_jacobian — Jp floor (prevents volume explosion)
            20.0, // max_plastic_jacobian — Jp ceiling. Matches Taichi MPM88/MPM128.
        )
    }
}

impl MaterialModel for SnowMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Snow
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        // h = particle.hardening_scale, updated each step by plasticity
        let h = particle.hardening_scale;
        let mu_eff = self.mu * h;
        let lambda_eff = self.lambda * h;

        // τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
        let f_t = f.transpose();
        let mut tau = 2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY;

        // Cohesion: compacted snow (Jp < 1) resists elastic re-expansion (J > 1).
        // Fires only when J_e > 1 AND Jp < 1 — no feedback loop, stable.
        // τ_coh = +c · Jp · (J−1) · J · I  →  positive Kirchhoff = inward P2G impulse = attraction.
        // Modulated by Jp so more-compressed snow is more cohesive.
        if self.cohesion_coeff > 0.0 && particle.plastic_volume_ratio < 1.0 && j > 1.0 {
            tau += self.cohesion_coeff * particle.plastic_volume_ratio * (j - 1.0) * j * Mat2::IDENTITY;
        }
        tau
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        // 1. Update deformation gradient
        let f_trial =
            (Mat2::IDENTITY + dt * particle.velocity_gradient) * particle.deformation_gradient;

        // 2. SVD of trial F
        let (u, sigma, vt) = svd2(f_trial);

        // 3. Clamp singular values to elastic range
        let sigma_c = Vec2::new(
            sigma
                .x
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            sigma
                .y
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
        );

        // 4. Accumulate plastic Jacobian: Jp *= det(sigma) / det(sigma_c)
        let jp_new = particle.plastic_volume_ratio * (sigma.x * sigma.y) / (sigma_c.x * sigma_c.y);
        particle.plastic_volume_ratio =
            jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);

        // 5. Update elastic hardening: h = exp(ξ(1−Jp)), clamped [0.1, 7.0].
        //    Upper bound is CFL-driven: at E=5000, h=7 → c_P≈99 cells/s → sub_dt≈0.005 → ~20 substeps.
        //    Sparkl has no clamp (runs 50 substeps); Taichi MPM128 clamps implicitly via substep budget.
        //    h=7 gives E_eff_max=35,000 under heavy compression — enough contrast between soft/packed.
        particle.hardening_scale =
            (self.hardening_exponent * (1.0 - particle.plastic_volume_ratio)).exp()
                .clamp(0.1, 7.0);

        // 6. Write back elastic F (plastic part absorbed into Jp + h)
        particle.deformation_gradient = u * Mat2::from_diagonal(sigma_c) * vt;

        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.sync_volume_and_density(j);
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Snow as u32,
            lambda: self.lambda,
            mu: self.mu,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            volume_ratio_min: self.min_plastic_jacobian,
            volume_ratio_max: self.max_plastic_jacobian,
            cohesion_coeff: self.cohesion_coeff,
            ..Default::default()
        }
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        // h grows when snow compresses — accounts for stiffening in CFL bound
        elastic_wave_dt(self.lambda, self.mu, particle.hardening_scale, particle.density, MIN_J, cell_width, material_cfl)
    }
}

use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{Particle, Particles};

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
            lambda, mu, 10.0,   // hardening_exponent ξ — Stomakhin 2013 §4
            0.025,  // compression_limit θ_c — Stomakhin 2013 §4 canonical (was 0.02)
            0.0075, // stretch_limit θ_s — Stomakhin 2013 §4 canonical (was 0.006)
            0.6,    // min_plastic_jacobian — Jp floor (prevents volume explosion)
            20.0,   // max_plastic_jacobian — Jp ceiling. Matches Taichi MPM88/MPM128.
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

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        // h = particles.hardening_scale[i], updated each step by plasticity
        let h = particles.hardening_scale[i];
        let mu_eff = self.mu * h;
        let lambda_eff = self.lambda * h;

        // τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
        let f_t = f.transpose();
        let mut tau = 2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY;

        // Cohesion: compacted snow (Jp < 1) resists elastic re-expansion (J > 1).
        if self.cohesion_coeff > 0.0 && particles.plastic_volume_ratio[i] < 1.0 && j > 1.0 {
            tau += self.cohesion_coeff
                * particles.plastic_volume_ratio[i]
                * (j - 1.0)
                * j
                * Mat2::IDENTITY;
        }
        tau
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particles.velocity_gradient[i])
            * particles.deformation_gradient[i];

        let (u, sigma, vt) = svd2(f_trial);

        let sigma_c = Vec2::new(
            sigma
                .x
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            sigma
                .y
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
        );

        let jp_new =
            particles.plastic_volume_ratio[i] * (sigma.x * sigma.y) / (sigma_c.x * sigma_c.y);
        particles.plastic_volume_ratio[i] =
            jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);

        // h clamped [0.1, 7.0]: upper bound is CFL-driven (h=7 → E_eff=35k → ~20 substeps).
        particles.hardening_scale[i] = (self.hardening_exponent
            * (1.0 - particles.plastic_volume_ratio[i]))
            .exp()
            .clamp(0.1, 7.0);

        particles.deformation_gradient[i] = u * Mat2::from_diagonal(sigma_c) * vt;

        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
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

    fn timestep_bound(
        &self,
        particles: &Particles,
        i: usize,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        // h grows when snow compresses — accounts for stiffening in CFL bound
        elastic_wave_dt(
            self.lambda,
            self.mu,
            particles.hardening_scale[i],
            particles.density[i],
            MIN_J,
            cell_width,
            material_cfl,
        )
    }
}

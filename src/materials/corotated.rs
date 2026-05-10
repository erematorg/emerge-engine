use glam::Mat2;

use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particle;

/// Corotated linear elasticity.
///
/// Kirchhoff stress: τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
/// R is the rotation from 2D polar decomposition (analytical — no SVD needed in 2D).
/// h = particle.hardening_scale (1.0 baseline; snow plasticity scales this up on compression).
/// Reference: Stomakhin et al. 2013, eq. (5)–(8). Used as the elastic base for snow.
/// Also the elastic component of Drucker-Prager (Klar et al. 2016).
#[derive(Debug, Clone, Copy)]
pub struct CorotatedMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Thermal modulus scale: µ_eff = µ·h·(1 + thermal_expansion·T), same for λ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour (same semantics as NeoHookean).
    /// τ_total = τ_elastic + activation × coeff × F·(n₀⊗n₀)·Fᵀ  (fiber-directional contraction).
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
}

impl CorotatedMaterial {
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self { lambda, mu, thermal_expansion: 0.0, active_stress_coeff: 0.0 }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for CorotatedMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Corotated
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.hardening_scale = 1.0;
        particle.plastic_volume_ratio = 1.0;
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        let h = particle.hardening_scale;
        let t_scale = 1.0 + self.thermal_expansion * particle.temperature;
        let mu_eff = self.mu * h * t_scale;
        let lambda_eff = self.lambda * h * t_scale;

        let f_t = f.transpose();
        2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particle.velocity_gradient;
        particle.deformation_gradient = fp_new * particle.deformation_gradient;
        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.sync_volume_and_density(j);
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Corotated as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            ..Default::default()
        }
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        elastic_wave_dt(self.lambda, self.mu, particle.hardening_scale, particle.density, MIN_J, cell_width, material_cfl)
    }
}

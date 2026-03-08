use glam::{Mat2, Vec2};

use crate::solver::materials::{ConstitutiveModel, MaterialModel};
use crate::solver::svd::svd2;
use crate::state::particle::Particle;

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
    pub hardening_xi: f32,
    /// Max compression before plastic flow triggers: Δσ < 1 - theta_c → plastic.
    pub compression_limit: f32,
    /// Max stretch before plastic flow triggers: Δσ > 1 + theta_s → plastic.
    pub stretch_limit: f32,
    /// Lower bound on Jp. Prevents wave speed from exploding as h → ∞.
    pub min_plastic_jacobian: f32,
    /// Upper bound on Jp (slight stretch plasticity allowed).
    pub max_plastic_jacobian: f32,
    min_j: f32,
}

impl SnowMaterial {
    pub fn new(
        lambda: f32,
        mu: f32,
        hardening_xi: f32,
        compression_limit: f32,
        stretch_limit: f32,
        min_plastic_jacobian: f32,
        max_plastic_jacobian: f32,
    ) -> Self {
        Self {
            lambda,
            mu,
            hardening_xi,
            compression_limit,
            stretch_limit,
            min_plastic_jacobian,
            max_plastic_jacobian,
            min_j: 1.0e-6,
        }
    }
}

impl MaterialModel for SnowMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Snow
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= self.min_j {
            return Mat2::ZERO;
        }

        // 2D polar decomposition (same as CorotatedMaterial — no SVD needed for stress)
        let a = f.x_axis.x;
        let c = f.x_axis.y;
        let b = f.y_axis.x;
        let d = f.y_axis.y;
        let x = a + d;
        let y = c - b;
        let norm = (x * x + y * y).sqrt();
        let r = if norm > f32::EPSILON {
            Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
        } else {
            Mat2::IDENTITY
        };

        // h = particle.elastic_hardening, updated each step by plasticity
        let h = particle.elastic_hardening;
        let mu_eff = self.mu * h;
        let lambda_eff = self.lambda * h;

        // τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
        let f_t = f.transpose();
        2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        // 1. Update deformation gradient
        let f_trial = (Mat2::IDENTITY + dt * particle.affine) * particle.deformation_gradient;

        // 2. SVD of trial F
        let (u, sigma, vt) = svd2(f_trial);

        // 3. Clamp singular values to elastic range
        let sigma_c = Vec2::new(
            sigma.x.clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            sigma.y.clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
        );

        // 4. Accumulate plastic Jacobian: Jp *= det(sigma) / det(sigma_c)
        let jp_new = particle.plastic_jacobian * (sigma.x * sigma.y) / (sigma_c.x * sigma_c.y);
        particle.plastic_jacobian =
            jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);

        // 5. Update elastic hardening: h = exp(xi * (1 - Jp))
        //    Jp < 1 (compressed) → h > 1 (stiffer). Jp > 1 (stretched) → h < 1 (softer).
        particle.elastic_hardening = (self.hardening_xi * (1.0 - particle.plastic_jacobian)).exp();

        // 6. Write back elastic F (plastic part absorbed into Jp + h)
        particle.deformation_gradient =
            u * Mat2::from_diagonal(sigma_c) * vt;

        // 7. Track volume and density from elastic J
        let j = particle.deformation_gradient.determinant().max(self.min_j);
        particle.volume = (particle.initial_volume * j).max(1.0e-6);
        particle.density = particle.mass / particle.volume;
    }

    fn timestep_bound(
        &self,
        particle: &Particle,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        let density = particle.density.max(1.0e-6);
        // h can increase stiffness significantly when snow is compressed — account for it
        let h = particle.elastic_hardening;
        let elastic_modulus = ((self.lambda + 2.0 * self.mu) * h).max(0.0);
        if elastic_modulus <= f32::EPSILON {
            return f32::INFINITY;
        }
        let wave_speed = (elastic_modulus / density).sqrt();
        if wave_speed <= f32::EPSILON {
            return f32::INFINITY;
        }
        material_cfl * cell_width / wave_speed
    }
}

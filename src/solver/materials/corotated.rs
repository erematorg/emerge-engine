use glam::{Mat2, Vec2};

use crate::solver::materials::{ConstitutiveModel, MaterialModel};
use crate::state::particle::Particle;

/// Corotated linear elasticity.
/// τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
/// R is the rotation from 2D polar decomposition (no SVD needed).
/// h = particle.elastic_hardening (1.0 = no hardening; scaled by snow plasticity).
#[derive(Debug, Clone, Copy)]
pub struct CorotatedMaterial {
    pub elastic_lambda: f32,
    pub elastic_mu: f32,
    pub min_j: f32,
}

impl CorotatedMaterial {
    pub fn new(elastic_lambda: f32, elastic_mu: f32) -> Self {
        Self { elastic_lambda, elastic_mu, min_j: 1.0e-6 }
    }
}

impl MaterialModel for CorotatedMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Corotated
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= self.min_j {
            return Mat2::ZERO;
        }

        // 2D polar decomposition: F = R·S, R is rotation.
        // For F = [[a,b],[c,d]] (row-major notation):
        //   x = a+d, y = c-b, norm = sqrt(x²+y²)
        //   R = [[x,-y],[y,x]] / norm
        // In glam column-major: col0=[a,c], col1=[b,d]
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

        let h = particle.elastic_hardening;
        let mu_eff = self.elastic_mu * h;
        let lambda_eff = self.elastic_lambda * h;

        let f_t = f.transpose();
        2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particle.c;
        particle.deformation_gradient = fp_new * particle.deformation_gradient;
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
        let h = particle.elastic_hardening;
        let elastic_modulus = ((self.elastic_lambda + 2.0 * self.elastic_mu) * h).max(0.0);
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

use glam::Mat2;

use crate::solver::materials::MaterialModel;
use crate::state::particle::Particle;

// --- Neo-Hookean ---

#[derive(Debug, Clone, Copy)]
pub struct NeoHookeanMaterial {
    pub elastic_lambda: f32,
    pub elastic_mu: f32,
    pub min_density: f32,
    pub min_j: f32,
}

impl NeoHookeanMaterial {
    pub fn new(elastic_lambda: f32, elastic_mu: f32) -> Self {
        Self {
            elastic_lambda,
            elastic_mu,
            min_density: 1.0e-6,
            min_j: 1.0e-6,
        }
    }
}

impl MaterialModel for NeoHookeanMaterial {
    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= self.min_j {
            return Mat2::ZERO;
        }

        let f_t = f.transpose();
        let f_inv_t = f_t.inverse();
        let p_term_0 = self.elastic_mu * (f - f_inv_t);
        let p_term_1 = self.elastic_lambda * j.ln() * f_inv_t;
        let p = p_term_0 + p_term_1;
        (1.0 / j) * (p * f_t)
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        let j = particle.deformation_gradient.determinant().max(self.min_j);
        particle.initial_volume * j
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particle.c;
        particle.deformation_gradient = fp_new * particle.deformation_gradient;
        let j = particle.deformation_gradient.determinant().max(self.min_j);
        particle.volume = (particle.initial_volume * j).max(1.0e-6);
        // Density tracks volume: ρ = m/V. Required for correct wave speed in timestep_bound.
        particle.density = particle.mass / particle.volume;
    }

    fn timestep_bound(
        &self,
        particle: &Particle,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        // Explicit elasticity update is bounded by elastic wave speed.
        let density = particle.density.max(self.min_density);
        let elastic_modulus = (self.elastic_lambda + 2.0 * self.elastic_mu).max(0.0);
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

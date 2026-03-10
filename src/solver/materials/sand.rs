use glam::{Mat2, Vec2};

use crate::solver::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::solver::svd::svd2;
use crate::state::particle::Particle;

/// Drucker-Prager elastoplastic sand/soil.
///
/// Elastic response: corotated linear elasticity (τ = 2µ(F−R)Fᵀ + λ(J−1)J·I).
/// Plasticity: Drucker-Prager yield surface with friction-angle hardening.
///
/// The yield criterion is checked in logarithmic strain space (Hencky strain).
/// When the trial strain exceeds the cone, singular values of F are projected
/// back to the yield surface via return mapping.
///
/// Reference: Klar et al. 2016 "Drucker-Prager Elastoplasticity for Sand Animation".
/// Implementation mirrors sparkl's `DruckerPragerPlasticity`, adapted to 2D.
#[derive(Debug, Clone, Copy)]
pub struct SandMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Initial friction angle (radians). Dry sand ≈ 35° = 0.611 rad.
    pub h0: f32,
    /// Friction hardening sensitivity.
    pub h1: f32,
    /// Friction hardening decay rate.
    pub h2: f32,
    /// Residual friction angle (radians). ≈ 10° = 0.175 rad.
    pub h3: f32,
    /// Volume correction factor. 1.0 = full sparkl correction, 0.0 = none.
    pub volume_correction: f32,
    min_j: f32,
}

impl SandMaterial {
    /// Construct with default friction-angle hardening for dry sand.
    /// `lambda`, `mu` are Lamé parameters (SI or grid units).
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            h0: 35.0_f32.to_radians(),
            h1: 9.0_f32.to_radians(),
            h2: 0.2,
            h3: 10.0_f32.to_radians(),
            volume_correction: 1.0,
            min_j: 1.0e-6,
        }
    }

    /// Friction coefficient α(q) derived from friction angle φ(q).
    /// φ(q) = h0 + (h1·q − h3)·exp(−h2·q)
    /// α(q) = √(2/3) · 2·sin(φ) / (3 − sin(φ))
    fn alpha(&self, q: f32) -> f32 {
        let phi = self.h0 + (self.h1 * q - self.h3) * (-self.h2 * q).exp();
        let s = phi.sin();
        (2.0_f32 / 3.0).sqrt() * (2.0 * s) / (3.0 - s)
    }

    /// Drucker-Prager return mapping in log-strain (Hencky) space.
    ///
    /// Returns `Some((projected_sigma, delta_q))` if projection occurred (plastic step),
    /// `None` if the trial state is inside the yield surface (elastic step).
    ///
    /// `sigma`: trial singular values of F (σ₁, σ₂, both > 0).
    /// `log_vol_gain`: cumulative volumetric plastic strain offset (particle field).
    /// `alpha`: current friction coefficient from `self.alpha(q)`.
    fn project(&self, sigma: Vec2, log_vol_gain: f32, alpha: f32) -> Option<(Vec2, f32)> {
        // Hencky (logarithmic) strain, shifted by the accumulated volumetric offset.
        let eps = Vec2::new(
            sigma.x.ln() + log_vol_gain * 0.5,
            sigma.y.ln() + log_vol_gain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();

        // Tension cutoff or purely volumetric deformation: project to identity (σ = 1).
        if dev_norm == 0.0 || trace > 0.0 {
            return Some((Vec2::ONE, eps.length()));
        }

        // Yield function: γ = |dev_strain| + (d·λ + 2µ)/(2µ) · tr · α, with d=2.
        let ratio = (2.0 * self.lambda + 2.0 * self.mu) / (2.0 * self.mu);
        let gamma = dev_norm + ratio * trace * alpha;

        if gamma <= 0.0 {
            return None; // Inside yield surface — elastic step.
        }

        // Project onto yield surface in log-strain space, then exponentiate.
        let h = eps - gamma * (dev / dev_norm);
        Some((Vec2::new(h.x.exp(), h.y.exp()), gamma))
    }
}

impl MaterialModel for SandMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::DruckerPrager
    }

    /// Corotated elastic Kirchhoff stress: τ = 2µ(F−R)Fᵀ + λ(J−1)J·I
    /// R is the rotation from 2D polar decomposition of F.
    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= self.min_j {
            return Mat2::ZERO;
        }

        // 2D polar decomp: x = F₀₀+F₁₁, y = F₁₀−F₀₁ (column-major layout)
        let x = f.x_axis.x + f.y_axis.y;
        let y = f.x_axis.y - f.y_axis.x;
        let norm = (x * x + y * y).sqrt();
        let r = if norm > f32::EPSILON {
            Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
        } else {
            Mat2::IDENTITY
        };

        let f_t = f.transpose();
        2.0 * self.mu * (f - r) * f_t + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        // 1. Trial elastic deformation gradient.
        let f_trial = (Mat2::IDENTITY + dt * particle.affine) * particle.deformation_gradient;

        // 2. SVD: F_trial = U · diag(σ) · Vt
        let (u, sigma, vt) = svd2(f_trial);

        // 3. Drucker-Prager return mapping.
        let alpha = self.alpha(particle.plastic_hardening);
        let new_sigma = if let Some((proj_sigma, dq)) =
            self.project(sigma, particle.log_vol_gain, alpha)
        {
            let prev_det = sigma.x * sigma.y;
            let new_det = proj_sigma.x * proj_sigma.y;
            let diff = new_det - prev_det;
            // Volume correction: attenuate volume loss from projection.
            let corrected_det = if diff > 0.0 {
                new_det
            } else {
                prev_det + diff * self.volume_correction
            };

            particle.log_vol_gain += prev_det.ln() - corrected_det.ln();
            particle.plastic_hardening += dq;
            proj_sigma
        } else {
            sigma
        };

        // 4. Recompose F from projected singular values: F = U · diag(σ_new) · Vt
        let sigma_mat = Mat2::from_cols(
            Vec2::new(new_sigma.x, 0.0),
            Vec2::new(0.0, new_sigma.y),
        );
        particle.deformation_gradient = u * sigma_mat * vt;

        let j = particle.deformation_gradient.determinant().max(self.min_j);
        particle.volume = (particle.initial_volume * j).max(1.0e-6);
        particle.density = particle.mass / particle.volume;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::DruckerPrager as u32,
            lambda: self.lambda,
            mu: self.mu,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        particle: &Particle,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        let density = particle.density.max(1.0e-6);
        let elastic_modulus = (self.lambda + 2.0 * self.mu).max(0.0);
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

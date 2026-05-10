use glam::{Mat2, Vec2};

use crate::materials::utils::{
    LOG_CLAMP, MIN_J, elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particle;
use crate::materials::svd::svd2;

/// Von Mises elastoplastic material: J2 plasticity with optional linear isotropic hardening.
///
/// Elastic response: corotated linear elastic (τ = 2µ(F−R)Fᵀ + λ(J−1)J·I).
/// Yield criterion: 2µ·|dev(ε)| ≤ σ_Y(κ), where ε is the Hencky (log) strain and
///   σ_Y(κ) = yield_stress + hardening_modulus·κ  (κ = accumulated equivalent plastic strain).
/// Return mapping: scale deviatoric log-strain back to the yield surface; volumetric
/// strain is preserved exactly (incompressible plastic flow assumption).
///
/// `hardening_modulus = 0.0` → perfect plasticity (original behaviour, backward compatible).
/// `hardening_modulus > 0` → linear isotropic hardening (metals, biological tissue stiffening).
///
/// Suitable for: lava flows, ductile metals, clay, soft rock under shear, biological tissue.
///
/// GPU note: `yield_stress` is stored in `hardening_exponent` in `MaterialParams` (union layout).
/// `hardening_modulus` is stored in `MaterialParams::hardening_modulus`.
/// `κ` is accumulated into `Particle::friction_hardening` each substep.
#[derive(Debug, Clone, Copy)]
pub struct VonMisesMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Initial yield stress σ_Y₀ in simulation stress units (same scale as λ/µ).
    /// Flow begins when 2µ|dev(ε)| > σ_Y₀ + H·κ.
    pub yield_stress: f32,
    /// Linear isotropic hardening modulus H.
    /// σ_Y(κ) = yield_stress + H·κ. Set 0.0 for perfect plasticity (default).
    pub hardening_modulus: f32,
}

impl VonMisesMaterial {
    /// Construct from Young's modulus E, Poisson's ratio ν, and yield stress σ_Y.
    ///
    /// Typical values for lava/clay: E = 5e4–1e5, ν = 0.3–0.4, σ_Y = 1e2–1e3.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32, yield_stress: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, yield_stress)
    }

    /// Perfect plasticity (no hardening).
    pub fn new(lambda: f32, mu: f32, yield_stress: f32) -> Self {
        Self { lambda, mu, yield_stress, hardening_modulus: 0.0 }
    }

    /// Linear isotropic hardening. `hardening_modulus` > 0 makes the material stiffen
    /// as it deforms plastically — yield stress grows as `yield_stress + H·κ`.
    pub fn with_hardening(lambda: f32, mu: f32, yield_stress: f32, hardening_modulus: f32) -> Self {
        assert!(hardening_modulus >= 0.0, "hardening_modulus must be ≥ 0");
        Self { lambda, mu, yield_stress, hardening_modulus }
    }
}

impl MaterialModel for VonMisesMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::VonMises
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
        let tr = eps.x + eps.y;
        let dev = eps - Vec2::splat(tr * 0.5);
        let dev_norm = dev.length();

        // J2 yield: 2µ|dev(ε)| ≤ yield_stress + H·κ
        let kappa = particle.friction_hardening;
        let effective_yield = self.yield_stress + self.hardening_modulus * kappa;
        let elastic_dev = 2.0 * self.mu * dev_norm;

        let sigma_new = if elastic_dev > effective_yield && dev_norm > LOG_CLAMP {
            // Return mapping: project deviatoric strain onto yield surface.
            // γ = (elastic_dev − σ_Y) / (2µ + H)  →  new κ = κ + γ
            let denom = 2.0 * self.mu + self.hardening_modulus;
            let gamma = if denom > f32::EPSILON { (elastic_dev - effective_yield) / denom } else { 0.0 };
            particle.friction_hardening = kappa + gamma;
            let eps_proj = dev * (effective_yield / elastic_dev) + Vec2::splat(tr * 0.5);
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
            model: ConstitutiveModel::VonMises as u32,
            lambda: self.lambda,
            mu: self.mu,
            hardening_exponent: self.yield_stress,
            hardening_modulus: self.hardening_modulus,
            ..Default::default()
        }
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        elastic_wave_dt(self.lambda, self.mu, 1.0, particle.density, MIN_J, cell_width, material_cfl)
    }
}

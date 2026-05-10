use glam::Mat2;

use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particle;

/// Compressible Neo-Hookean hyperelastic solid (jelly, soft tissue).
///
/// Free energy: Ψ = µ/2·(tr(FᵀF)−d) − µ·ln(J) + λ/2·ln(J)²
/// Kirchhoff stress: τ = µ(FFᵀ − I) + λ·ln(J)·I
///   (derived via P = ∂Ψ/∂F, τ = (1/J)·P·Fᵀ — simplified to avoid F⁻¹)
/// Reference: standard hyperelasticity; used in Stomakhin et al. 2013 (snow paper) §2.
#[derive(Debug, Clone, Copy)]
pub struct NeoHookeanMaterial {
    pub lambda: f32,
    pub mu: f32,
    pub min_density: f32,
    /// Thermal modulus scale: λ_eff = λ·(1 + thermal_expansion·T), same for µ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour.
    /// τ_total = τ_elastic + activation × coeff × I  (contractile: pulls inward like a muscle).
    /// Independent of elastic state — generates force even at rest.
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
}

impl NeoHookeanMaterial {
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            min_density: 1.0e-6,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Canonical values: E = 5e6, ν = 0.2 (wgsparkl elasticity2 — stiff soft solid).
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for NeoHookeanMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::NeoHookean
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        // Thermal modulus scaling: λ_eff = λ·(1 + α·T), same for µ.
        let t_scale = 1.0 + self.thermal_expansion * particle.temperature;
        let mu = self.mu * t_scale;
        let lambda = self.lambda * t_scale;

        // Simo-Pister volumetric-deviatoric split (Apache-2.0 reference: sparkl).
        // B = F·Fᵀ (left Cauchy-Green), d = 2 in 2D.
        // Deviatoric Kirchhoff: µ · J^{-2/d} · dev(B)  with d=2 → µ/J · dev(B)
        //   dev(B) = B − (tr(B)/2)·I  (2D traceless part)
        // Volumetric Kirchhoff: k/2 · (J²−1) · I
        //   k = 2/3·µ + λ  (bulk modulus, Simo form — matches sparkl exactly)
        // Reference: Simo & Pister 1984; Bonet & Wood §6.4.
        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(glam::Vec2::splat(tr_b * 0.5));
        let k = (2.0 / 3.0) * mu + lambda;

        let dev_stress = (mu / j) * dev_b;
        let vol_stress = (k * 0.5 * (j * j - 1.0)) * Mat2::IDENTITY;

        dev_stress + vol_stress
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        // Kirchhoff stress is returned directly → scatter with V₀, not current volume.
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
            model: ConstitutiveModel::NeoHookean as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            ..Default::default()
        }
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        elastic_wave_dt(self.lambda, self.mu, 1.0, particle.density, self.min_density, cell_width, material_cfl)
    }
}

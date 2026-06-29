use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, GranularProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, Particles};

/// µ(I)-rheology sand — rate-dependent Drucker-Prager (Blatny / matter "DPMui").
///
/// Extends plain Drucker-Prager by making the friction coefficient pressure- and
/// rate-dependent via the inertial number I = γ̇·d/√(p/ρₛ):
///
///   µ(I) = µ₁ + (µ₂ − µ₁) / (Q·√p / γ̇ + 1)
///
/// where:
///   µ₁   — static friction coefficient  (slow/quasi-static flows)
///   µ₂   — dynamic friction coefficient (rapid granular flows)
///   Q    = I₀ / (d · √ρₛ)  — single merged inertial rate parameter
///
/// At low shear rate (γ̇ → 0): µ(I) → µ₁  — material resists flow like dry sand.
/// At high shear rate (γ̇ → ∞): µ(I) → µ₂ — material flows more easily.
///
/// The plastic multiplier γ̇ is solved analytically via a quadratic at each step.
/// This gives rate-softening without requiring a Newton iteration.
///
/// Yield surface (no cohesion, no dilation in this formulation):
///   q ≤ µ(I) · p   (Drucker-Prager cone, µ is rate-dependent)
///
/// `Particle::friction_hardening` is repurposed to store the current µ(I) value,
/// useful for visualising the local flow regime (µ₁ = quasi-static, µ₂ = rapid).
///
/// Reference: Blatny 2022, Blatny et al. 2021; matter/src/simulation/plasticity.cpp DPMui.
/// Canonical parameters: µ₁=tan(20.9°), µ₂=tan(32.8°), I₀=0.279, d=1mm, ρₛ=2500 kg/m³
/// → Q = 0.279 / (0.001 · √2500) ≈ 5.58.
#[derive(Debug, Clone, Copy)]
pub struct MuIRheologyMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// µ₁ = tan(φ_static). Quasi-static friction at vanishing shear rate.
    /// matter default: tan(20.9°) ≈ 0.382.
    pub mu_static: f32,
    /// µ₂ = tan(φ_dynamic). Friction at infinite shear rate.
    /// matter default: tan(32.8°) ≈ 0.644.
    pub mu_dynamic: f32,
    /// Q = I₀ / (d · √ρₛ).  Higher Q → rate effects kick in at lower γ̇.
    /// Fine sand (d=1mm, ρₛ=2500, I₀=0.279): Q ≈ 5.58.
    /// Coarse sand (d=5mm, ρₛ=2500, I₀=0.279): Q ≈ 1.12.
    pub inertial_q: f32,
}

impl MuIRheologyMaterial {
    /// Construct with Lamé parameters. Defaults to matter's canonical µ(I) parameters
    /// (µ₁=tan20.9°, µ₂=tan32.8°, Q=5.58 for 1mm sand grains).
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            mu_static: 20.9_f32.to_radians().tan(),
            mu_dynamic: 32.8_f32.to_radians().tan(),
            inertial_q: 5.58,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν (same API as DruckerPragerMaterial).
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Large-grain: inertial_q=1.12, less rate-sensitive. Coarse sand regime (d≈5mm).
    pub fn large_grain(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            inertial_q: 1.12,
            ..Self::new(lambda, mu)
        }
    }

    /// Small-grain: default inertial_q, more rate-sensitive. Fine sand regime (d≈1mm).
    pub fn small_grain(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::from_young_modulus(young_modulus, poisson_ratio)
    }

    /// Dense-packed: higher static + dynamic friction, larger µ₂-µ₁ gap.
    pub fn dense_packed(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu_static: 30.0_f32.to_radians().tan(),
            mu_dynamic: 40.0_f32.to_radians().tan(),
            ..Self::new(lambda, mu)
        }
    }
}

impl FromSI<GranularProps> for MuIRheologyMaterial {
    /// µ(I) friction params (µ₁, µ₂, I₀) fixed to canonical fine-sand values.
    /// Use `irl::DRY_SAND`, `irl::LOOSE_SAND`, or `irl::DENSE_SAND` as props.
    fn from_physical(props: &GranularProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        Self {
            mu_static: props.friction_angle_deg.to_radians().tan(),
            mu_dynamic: (props.friction_angle_deg + 12.0).to_radians().tan(),
            ..Self::new(lambda, mu)
        }
    }
}

impl MaterialModel for MuIRheologyMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::DruckerPragerMuI
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }
        let r = crate::materials::polar_decomposition_2d(f);
        2.0 * self.mu * (f - r) * f.transpose() + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn init_particle(&self, particle: &mut Particle) {
        // Initialise stored µ to static friction (quasi-static at rest).
        particle.friction_hardening = self.mu_static;
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particles.velocity_gradient[i])
            * particles.deformation_gradient[i];
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);
        let tr = eps.x + eps.y;

        let k_2d = self.lambda + self.mu;
        let p_trial = -k_2d * tr;

        if p_trial <= 0.0 {
            particles.deformation_gradient[i] = reconstruct_f(u, Vec2::ONE, vt);
            particles.friction_hardening[i] = self.mu_static;
            let j = particles.deformation_gradient[i].determinant().max(MIN_J);
            let v = (particles.initial_volume[i] * j).max(1.0e-6);
            particles.volume[i] = v;
            particles.density[i] = particles.mass[i] / v;
            return;
        }

        let dev = eps - Vec2::splat(tr * 0.5);
        let dev_norm = dev.length();
        let q_trial = std::f32::consts::SQRT_2 * self.mu * dev_norm;
        let q_yield = self.mu_static * p_trial;

        if q_trial <= q_yield || dev_norm < f32::EPSILON {
            particles.deformation_gradient[i] = reconstruct_f(u, sigma, vt);
            particles.friction_hardening[i] = self.mu_static;
            let j = particles.deformation_gradient[i].determinant().max(MIN_J);
            let v = (particles.initial_volume[i] * j).max(1.0e-6);
            particles.volume[i] = v;
            particles.density[i] = particles.mass[i] / v;
            return;
        }

        let delta_q = q_trial - q_yield;
        let sqrt_p = p_trial.sqrt();

        let a = self.mu * dt;
        let b =
            p_trial * (self.mu_dynamic - self.mu_static) + a * self.inertial_q * sqrt_p - delta_q;
        let c = -delta_q * self.inertial_q * sqrt_p;

        let gamma_dot = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
        let gamma_dot = gamma_dot.max(0.0);

        let mu_i = if gamma_dot > f32::EPSILON {
            self.mu_static
                + (self.mu_dynamic - self.mu_static) / (self.inertial_q * sqrt_p / gamma_dot + 1.0)
        } else {
            self.mu_static
        };

        let delta_gamma = gamma_dot * dt;
        let n_hat = dev / dev_norm;
        let eps_new = eps - n_hat * (delta_gamma / std::f32::consts::SQRT_2);

        let sigma_new = Vec2::new(eps_new.x.exp(), eps_new.y.exp());
        particles.deformation_gradient[i] = reconstruct_f(u, sigma_new, vt);
        particles.friction_hardening[i] = mu_i;

        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::DruckerPragerMuI as u32,
            lambda: self.lambda,
            mu: self.mu,
            // Reuse DP slots for µ(I) params (CPU-only for now).
            dp_h0: self.mu_static,
            dp_h1: self.mu_dynamic,
            dp_h2: self.inertial_q,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        )
    }

    fn needs_cpu_update(&self) -> bool {
        false
    }
}

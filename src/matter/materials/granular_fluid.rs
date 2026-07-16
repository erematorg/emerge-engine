use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young, polar_decomposition_2d};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, Particles};

/// Granular-fluid mixture: Tait EOS bulk pressure + corotated elastic deviatoric + SVD plasticity.
///
/// Constitutive law (Dunatunga & Kamrin 2015, §3):
///   τ = τ_EOS + τ_corotated_dev
///   τ_EOS  = −k·((ρ/ρ₀)^γ − 1)·I                 — weakly-compressible fluid bulk (Tait EOS)
///   τ_dev  = 2µ·h·dev[(F−R)·Fᵀ] + λ·h·(J−1)·J·I  — corotated elastic (shape-restoring + vol)
///
/// Plasticity: SVD clamp on F singular values (Stomakhin 2013 §4 — identical to StomakhinMaterial).
///   Jp accumulates plastic volume change. h = exp(ξ·(1−Jp)) hardens on compression.
///
/// This differs from:
///   - `BinghamFluidMaterial` (purely fluid, no elastic restoring force)
///   - `DruckerPragerMaterial` (no EOS pressure, rate-independent DP)
///   - `StomakhinMaterial` (no EOS, purely elastic+plastic)
///
/// Use for: wet terrain substrates, wet granular flows, biological cell matrices.
/// Ref: Kamrin 2015 granular-fluid; SoftZoo's own mud material (`mud.py`) independently
/// confirms the same fluid-EOS + corotated blend CONCEPT -- but its own specific
/// parameters (a single fixed set, linear not Tait EOS, θ_c=0.025) do NOT match this
/// file's three presets below; see each preset's own honest-disclosure doc comment.
#[derive(Debug, Clone, Copy)]
pub struct GranularFluidMaterial {
    /// Elastic shear modulus µ — corotated deviatoric stiffness.
    pub mu: f32,
    /// Elastic first Lamé λ — volumetric elastic contribution.
    pub lambda: f32,
    /// Rest density ρ₀. EOS pressure is zero when ρ = ρ₀.
    pub rest_density: f32,
    /// EOS bulk stiffness k. Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. 7 for near-incompressible; 1–3 for compressible granular flow.
    pub eos_power: f32,
    /// Hardening exponent ξ. h = exp(ξ·(1−Jp)). 0 = perfect plasticity.
    pub hardening_exponent: f32,
    /// Compression limit θ_c — singular values clamped at (1−θ_c).
    pub compression_limit: f32,
    /// Stretch limit θ_s — singular values clamped at (1+θ_s).
    pub stretch_limit: f32,
    /// Jp lower bound — prevents h from exploding under sustained compression.
    pub min_plastic_jacobian: f32,
    /// Jp upper bound — limits plastic volume expansion.
    pub max_plastic_jacobian: f32,
    /// EOS pressure floor. 0.0 = no tensile (stable free surface).
    pub pressure_floor: f32,
}

impl GranularFluidMaterial {
    /// Saturated loam: eos_stiffness=200, ξ=5, θ_c=0.4 — yields easily, flows under load.
    ///
    /// HONEST DISCLOSURE (audit 2026-07-17): the constitutive LAW above (Tait EOS +
    /// corotated elastic + Stomakhin SVD plasticity) is real and cited. These specific
    /// shape-parameter VALUES (eos_stiffness, hardening_exponent, compression_limit,
    /// stretch_limit, plastic-Jacobian bounds, rest_density) are NOT — checked directly
    /// against SoftZoo's own mud material (`mud.py`, the file this module's top doc
    /// pointed to) and they don't trace to it: SoftZoo uses one fixed parameter set
    /// (not three material variants), a different (linear, not Tait power-law) EOS
    /// form, and its compression limit (θ_c=0.025) is off by ~12-24x from this preset's
    /// 0.4. They also don't trace to Dunatunga & Kamrin 2015 (a granular-only paper,
    /// no mud/fluid blend or these numbers). This preset's real-world name ("saturated
    /// loam") is illustrative/hand-tuned, not a measured real-loam value -- same
    /// honesty standard as `FORAGING_RECOVERY_RATE` elsewhere in this codebase: the
    /// mechanism is real, this specific calibration is not yet, and shouldn't be
    /// presented as if it were. Needs a real geotechnical/soil-mechanics source before
    /// any claim of "this is real loam" would be honest.
    pub fn saturated_loam(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 200.0,
            eos_power: 7.0,
            hardening_exponent: 5.0,
            compression_limit: 0.4,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
        }
    }

    /// Consolidated clay: eos_stiffness=500, ξ=3, θ_c=0.3 — higher stiffness, slower creep.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real consolidated-
    /// clay geotechnical data.
    pub fn consolidated_clay(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.2,
            eos_stiffness: 500.0,
            eos_power: 7.0,
            hardening_exponent: 3.0,
            compression_limit: 0.3,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.3,
            max_plastic_jacobian: 2.5,
            pressure_floor: 0.0,
        }
    }

    /// Cytoplasmic matrix: eos_stiffness=50, ξ=1, large yield surface.
    /// Use for biological cell interiors and soft tissue matrices.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real cytoplasm
    /// rheology literature.
    pub fn cytoplasmic(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 50.0,
            eos_power: 7.0,
            hardening_exponent: 1.0,
            compression_limit: 0.6,
            stretch_limit: 0.05,
            min_plastic_jacobian: 0.1,
            max_plastic_jacobian: 5.0,
            pressure_floor: 0.0,
        }
    }
}

impl MaterialModel for GranularFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::GranularFluid
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant().max(MIN_J);

        let density = (self.rest_density / j).max(1.0e-6);
        let ratio = (density / self.rest_density.max(1.0e-6)).max(1.0e-6);
        let pressure =
            (self.eos_stiffness * (ratio.powf(self.eos_power) - 1.0)).max(self.pressure_floor);

        let h = particles.hardening_scale[i];
        let r = polar_decomposition_2d(f);
        let mu_eff = self.mu * h;
        let coro = 2.0 * mu_eff * (f - r) * f.transpose();
        let tr = coro.x_axis.x + coro.y_axis.y;
        let dev_coro = coro - Mat2::from_diagonal(Vec2::splat(tr * 0.5));

        let lam_vol = self.lambda * h * (j - 1.0) * j * Mat2::IDENTITY;

        Mat2::from_diagonal(Vec2::splat(-pressure)) + dev_coro + lam_vol
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(1.0e-6)
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particles.velocity_gradient[i])
            * particles.deformation_gradient[i];

        if self.compression_limit > 0.0 || self.stretch_limit > 0.0 {
            let (u, sigma, vt) = svd2(f_trial);
            let sigma_c = Vec2::new(
                sigma
                    .x
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
                sigma
                    .y
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            );
            let jp_new = particles.plastic_volume_ratio[i] * (sigma.x * sigma.y)
                / (sigma_c.x * sigma_c.y).max(1.0e-10);
            particles.plastic_volume_ratio[i] =
                jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);
            particles.hardening_scale[i] = (self.hardening_exponent
                * (1.0 - particles.plastic_volume_ratio[i]))
                .exp()
                .clamp(0.1, 7.0);
            particles.deformation_gradient[i] = u * Mat2::from_diagonal(sigma_c) * vt;
        } else {
            particles.deformation_gradient[i] = f_trial;
        }

        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::GranularFluid as u32,
            mu: self.mu,
            lambda: self.lambda,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            volume_ratio_min: self.min_plastic_jacobian,
            volume_ratio_max: self.max_plastic_jacobian,
            pressure_floor: self.pressure_floor,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        elastic_wave_dt(
            self.lambda,
            self.mu,
            hardening_scale,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        )
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }
}

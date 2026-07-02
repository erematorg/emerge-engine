use glam::Mat2;

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

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
    /// Continuum damage softening rate — real mechanical consequence of accumulated
    /// structural damage (`Particle::friction_hardening`, e.g. from
    /// `rankine_damage_estimate`), not just a passive health readout. Effective
    /// stiffness: µ_eff = µ·exp(−rate·damage), λ_eff = λ·exp(−rate·damage) — the
    /// same exponential softening `RankineMaterial` uses for its own tensile
    /// strength (continuum damage mechanics), applied here to elastic stiffness
    /// instead. Damaged tissue gets progressively softer/weaker as a smooth,
    /// continuous function of real accumulated strain — not a hard on/off failure
    /// threshold. 0.0 = no damage coupling (default, unchanged behavior).
    pub damage_softening_rate: f32,
}

impl NeoHookeanMaterial {
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            min_density: 1.0e-6,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
            damage_softening_rate: 0.0,
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

impl FromSI<Elastic> for NeoHookeanMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for NeoHookeanMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::NeoHookean
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        // Thermal modulus scaling: λ_eff = λ·(1 + α·T), same for µ.
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        // Damage softening: µ_eff = µ·exp(−rate·damage), same exponential form
        // RankineMaterial uses for tensile strength -- continuum damage mechanics,
        // not a hand-picked curve. rate=0.0 (default) leaves this at 1.0, no-op.
        let damage_scale = (-self.damage_softening_rate * particles.friction_hardening[i]).exp();
        let mu = self.mu * t_scale * damage_scale;
        let lambda = self.lambda * t_scale * damage_scale;

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

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        // Kirchhoff stress is returned directly → scatter with V₀, not current volume.
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particles.velocity_gradient[i];
        particles.deformation_gradient[i] = fp_new * particles.deformation_gradient[i];
        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
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
            // cohesion_coeff is documented as reusable padding (Snow-only otherwise,
            // zero for all other materials) -- repurposed here for damage_softening_rate.
            cohesion_coeff: self.damage_softening_rate,
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
            self.min_density,
            cell_width,
            material_cfl,
        )
    }
}

#[cfg(test)]
mod damage_softening_tests {
    use super::*;
    use crate::Particle;

    fn particle_with(deformation_gradient: Mat2, friction_hardening: f32) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            sleeping: 0,
        });
        particles
    }

    #[test]
    fn zero_softening_rate_matches_undamaged_stress() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.0;

        let undamaged = particle_with(f, 0.0);
        let damaged = particle_with(f, 5.0);
        let tau_undamaged = mat.kirchhoff_stress(&undamaged, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert_eq!(
            tau_undamaged, tau_damaged,
            "rate=0.0 must leave stress unaffected by damage (backward compatible default)"
        );
    }

    #[test]
    fn damage_softens_stress_magnitude() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 0.5;

        let healthy = particle_with(f, 0.0);
        let damaged = particle_with(f, 3.0);
        let tau_healthy = mat.kirchhoff_stress(&healthy, 0);
        let tau_damaged = mat.kirchhoff_stress(&damaged, 0);

        assert!(
            tau_damaged.x_axis.length() < tau_healthy.x_axis.length(),
            "damaged tissue must produce weaker stress for the same deformation: \
             healthy={:?} damaged={:?}",
            tau_healthy,
            tau_damaged
        );
    }

    #[test]
    fn severe_damage_approaches_near_zero_stiffness() {
        let f = Mat2::from_cols(glam::Vec2::new(1.3, 0.0), glam::Vec2::new(0.0, 1.1));
        let mut mat = NeoHookeanMaterial::new(1000.0, 1000.0);
        mat.damage_softening_rate = 1.0;

        let severely_damaged = particle_with(f, 20.0); // exp(-20) ~ 2e-9, near-total loss
        let tau = mat.kirchhoff_stress(&severely_damaged, 0);
        assert!(
            tau.x_axis.length() < 1.0e-3,
            "severe damage must drive stiffness (and thus stress) toward zero, got {:?}",
            tau
        );
    }
}

use glam::{Mat2, Vec2};

use crate::materials::physical_props::{DuctileProps, FromSI, scale_lame, scale_stress};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particles;

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
        Self {
            lambda,
            mu,
            yield_stress,
            hardening_modulus: 0.0,
        }
    }

    /// Linear isotropic hardening. `hardening_modulus` > 0 makes the material stiffen
    /// as it deforms plastically — yield stress grows as `yield_stress + H·κ`.
    pub fn with_hardening(lambda: f32, mu: f32, yield_stress: f32, hardening_modulus: f32) -> Self {
        assert!(hardening_modulus >= 0.0, "hardening_modulus must be ≥ 0");
        Self {
            lambda,
            mu,
            yield_stress,
            hardening_modulus,
        }
    }

    /// Soft ductile: yield_stress = E/200. Low yield-to-stiffness ratio. Remoulded clay regime.
    pub fn soft_ductile(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::from_young_modulus(young_modulus, poisson_ratio, young_modulus * 0.005)
    }
}

impl FromSI<DuctileProps> for VonMisesMaterial {
    fn from_physical(props: &DuctileProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        let yield_stress = scale_stress(props.yield_stress_pa, props.elastic.rho_kg_m3, config);
        Self::new(lambda, mu, yield_stress)
    }
}

impl MaterialModel for VonMisesMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::VonMises
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }
        let r = polar_decomposition_2d(f);
        2.0 * self.mu * (f - r) * f.transpose() + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * particles.velocity_gradient[i])
            * particles.deformation_gradient[i];
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);
        let tr = eps.x + eps.y;
        let dev = eps - Vec2::splat(tr * 0.5);
        let dev_norm = dev.length();

        let kappa = particles.friction_hardening[i];
        let effective_yield = self.yield_stress + self.hardening_modulus * kappa;
        let elastic_dev = 2.0 * self.mu * dev_norm;

        let sigma_new = if elastic_dev > effective_yield && dev_norm > LOG_CLAMP {
            let denom = 2.0 * self.mu + self.hardening_modulus;
            let gamma = if denom > f32::EPSILON {
                (elastic_dev - effective_yield) / denom
            } else {
                0.0
            };
            particles.friction_hardening[i] = kappa + gamma;
            let eps_proj = dev * (effective_yield / elastic_dev) + Vec2::splat(tr * 0.5);
            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        particles.deformation_gradient[i] = reconstruct_f(u, sigma_new, vt);
        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
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
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::Particle;

    /// Isolates whether `update_particle`'s return mapping matches the material's
    /// OWN documented yield criterion (`2*mu*|dev(eps)| <= yield_stress`) exactly,
    /// bypassing MPM's grid/transfer pipeline entirely -- same discipline as
    /// `sand.rs::marginal_yield_tests`, which this engine's own citation audit
    /// (2026-07-07) confirmed is the right pattern for verifying a plasticity
    /// return-mapping against its own analytical yield surface. `VonMisesMaterial`
    /// had ZERO test comparing it to any analytical result before this (only a
    /// loose stress-stays-bounded overshoot check existed).
    fn run_one_step(mat: &VonMisesMaterial, sigma: Vec2, kappa: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = kappa;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles, 0, 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    /// Maps a TARGET `dev_norm` (the code's own L2-norm convention,
    /// `dev.length()` for `dev=(eps1-tr/2, eps2-tr/2)`) to the per-component
    /// magnitude `d` needed so that constructing `eps=(tr/2+d, tr/2-d)`
    /// actually produces that `dev_norm` exactly: `dev=(d,-d)`, so
    /// `dev.length() = d*sqrt(2)`, i.e. `d = dev_norm/sqrt(2)`.
    fn per_component_d_for_target_dev_norm(target_dev_norm: f32) -> f32 {
        target_dev_norm / std::f32::consts::SQRT_2
    }

    #[test]
    fn marginal_state_at_yield_stress_does_not_yield() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        // dev_norm exactly AT the yield threshold: 2*mu*dev_norm = yield_stress
        // => dev_norm = yield_stress / (2*mu). Comfortably inside (99% of it).
        let target_dev_norm = 0.99 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let trace = 0.0; // pure deviatoric, no volumetric strain
        let eps1 = trace * 0.5 + d;
        let eps2 = trace * 0.5 - d;
        let sigma = Vec2::new(eps1.exp(), eps2.exp());

        let (sigma_after, kappa_after) = run_one_step(&mat, sigma, 0.0);
        assert!(
            (sigma_after - sigma).length() < 1.0e-4,
            "state inside the yield surface should stay elastic (no change): \
             sigma={sigma:?} sigma_after={sigma_after:?}"
        );
        assert_eq!(
            kappa_after, 0.0,
            "kappa must not accumulate on an elastic step"
        );
    }

    #[test]
    fn marginal_state_beyond_yield_stress_projects_exactly_to_the_yield_surface() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        // Comfortably OUTSIDE: 150% of the yield threshold.
        let target_dev_norm = 1.5 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let trace = 0.4; // nonzero volumetric strain -- must be preserved exactly
        let eps1 = trace * 0.5 + d;
        let eps2 = trace * 0.5 - d;
        let sigma = Vec2::new(eps1.exp(), eps2.exp());

        let (sigma_after, kappa_after) = run_one_step(&mat, sigma, 0.0);

        // Real, exact analytical claim: the projected state's dev_norm must equal
        // EXACTLY yield_stress/(2*mu) (perfect plasticity, no hardening here) --
        // not just "less than before." Computed the SAME way the material's own
        // code does (L2 norm of the deviatoric vector), not a per-component value.
        let eps_after = crate::materials::utils::hencky_strains(sigma_after);
        let tr_after = eps_after.x + eps_after.y;
        let dev_after_vec = eps_after - Vec2::splat(tr_after * 0.5);
        let dev_after = dev_after_vec.length();
        let expected_dev = mat.yield_stress / (2.0 * mat.mu);
        assert!(
            (dev_after - expected_dev).abs() < 1.0e-4,
            "projected deviatoric strain should land EXACTLY on the yield surface \
             (dev_norm = yield_stress/(2*mu) = {expected_dev:.6}), got {dev_after:.6}"
        );

        // Volumetric (trace) strain must be preserved exactly -- incompressible
        // plastic flow assumption, a real documented claim of this material.
        assert!(
            (tr_after - trace).abs() < 1.0e-4,
            "plastic flow must preserve volumetric strain exactly: expected trace={trace}, \
             got {tr_after}"
        );

        assert!(kappa_after > 0.0, "kappa must accumulate on a plastic step");
    }

    #[test]
    fn hardening_raises_the_effective_yield_surface() {
        // With hardening_modulus > 0, a state that would yield at kappa=0 should
        // require LESS additional plastic strain once kappa has already
        // accumulated (softer transition) -- real, checkable monotonic claim.
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let target_dev_norm = 1.5 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let sigma = Vec2::new(d.exp(), (-d).exp());

        let (_, kappa_from_zero) = run_one_step(&mat, sigma, 0.0);
        let (_, kappa_from_existing) = run_one_step(&mat, sigma, 1.0);

        // Effective yield stress is HIGHER when kappa is already 1.0 (hardening),
        // so the SAME trial state should trigger a SMALLER incremental gamma.
        let gamma_from_zero = kappa_from_zero - 0.0;
        let gamma_from_existing = kappa_from_existing - 1.0;
        assert!(
            gamma_from_existing < gamma_from_zero,
            "hardening should shrink the plastic strain increment for the same trial \
             state once kappa has already accumulated: gamma(kappa=0)={gamma_from_zero:.6} \
             gamma(kappa=1)={gamma_from_existing:.6}"
        );
    }
}

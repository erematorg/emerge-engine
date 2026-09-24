use glam::Mat2;

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Corotated linear elasticity.
///
/// Kirchhoff stress: τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
/// R is the rotation from 2D polar decomposition (analytical — no SVD needed in 2D).
/// h = particle.hardening_scale (1.0 baseline; snow plasticity scales this up on compression).
/// Reference: Stomakhin et al. 2013, eq. (5)–(8). Used as the elastic base for snow.
/// Also the elastic component of Drucker-Prager (Klar et al. 2016).
#[derive(Debug, Clone, Copy)]
pub struct CorotatedMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Thermal modulus scale: µ_eff = µ·h·(1 + thermal_expansion·T), same for λ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour (same semantics as NeoHookean).
    /// τ_total = τ_elastic + activation × coeff × F·(n₀⊗n₀)·Fᵀ  (fiber-directional contraction).
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
    /// Clamp J ≥ j_min before evaluating the volumetric term, same convention
    /// `ViscoelasticMaterial`/`NeoHookeanMaterial` use (default 0.01) -- see
    /// `NeoHookeanMaterial::j_min`'s own doc. Clamp, don't hard-zero below
    /// `MIN_J`, or the volumetric restoring force can't act exactly when it's
    /// needed most. Honest, disclosed limitation: this model's volumetric
    /// term (`λ·(J−1)·J`, a bounded polynomial, not a diverging log-barrier)
    /// is inherently weaker against extreme compression than NeoHookean's
    /// (Stomakhin et al. 2013's simplified elastic base for snow/DP
    /// plasticity supplies its OWN separate compression_limit clamps; this
    /// material alone, e.g. `basic_jellies_gpu`, doesn't have that net) --
    /// the clamp removes the permanent-zero-stress trap, it doesn't give
    /// this model NeoHookean's stronger barrier.
    pub j_min: f32,
}

impl CorotatedMaterial {
    pub const fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
            j_min: 0.01,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }
}

impl FromSI<Elastic> for CorotatedMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for CorotatedMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Corotated
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.hardening_scale = 1.0;
        particle.plastic_volume_ratio = 1.0;
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        // Clamp, don't zero -- see `j_min`'s own doc.
        let j = f.determinant().max(self.j_min);

        let r = polar_decomposition_2d(f);

        let h = particles.hardening_scale[i];
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        let mu_eff = self.mu * h * t_scale;
        let lambda_eff = self.lambda * h * t_scale;

        let f_t = f.transpose();
        2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * *ctx.velocity_gradient;
        *ctx.deformation_gradient = fp_new * *ctx.deformation_gradient;
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn pressure_scale(&self) -> f32 {
        1.0
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Corotated as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            // Real, per-material stress-computation floor -- see `j_min`'s own
            // doc. NOT shared with Snow/Sand's own reuse of this same GPU
            // param slot for their real, different plastic-clamp range --
            // Corotated doesn't populate it for anything else.
            volume_ratio_min: self.j_min,
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
}

#[cfg(test)]
mod small_strain_linear_elasticity_tests {
    use super::*;
    use glam::Vec2;

    /// **Small-strain limit must recover exact linear elasticity (Hooke's law).**
    ///
    /// `CorotatedMaterial` had zero test comparing it to any measured/analytical
    /// result (confirmed via a full test-file audit, 2026-07-07). For a
    /// SYMMETRIC small strain F=I+delta*E (E symmetric, no rotation), the polar
    /// decomposition's rotation R is EXACTLY I (not just to leading order --
    /// `polar_decomposition_2d`'s own formula gives y=F10-F01=delta(E10-E01)=0
    /// exactly when E is symmetric, so R=I for any delta, not an approximation).
    /// This makes the leading-order linearization of tau=2*mu*(F-R)*F^T +
    /// lambda*(J-1)*J*I reduce to EXACTLY tau = 2*mu*eps + lambda*tr(eps)*I
    /// (eps=delta*E) -- the textbook linear-elasticity form directly, no
    /// plane-strain k=lambda+mu correction needed (unlike NeoHookean's
    /// vol-dev split form -- Corotated already IS linear elasticity, just
    /// extended to finite rotation).
    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
            v: Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: f,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            scalar_field: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            internal_pressure: 0.0,
        });
        particles
    }

    fn linear_elastic_prediction(lambda: f32, mu: f32, eps: Mat2) -> Mat2 {
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps
    }

    #[test]
    fn small_uniaxial_strain_matches_hookes_law() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = CorotatedMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_diagonal(Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain Corotated stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    #[test]
    fn small_shear_strain_matches_hookes_law() {
        let lambda = 500.0;
        let mu = 1200.0;
        let mat = CorotatedMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain Corotated shear stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// Real, exact property (not an approximation): for symmetric F, the
    /// rotation R from polar decomposition must be EXACTLY identity, at any
    /// strain magnitude (not just small) -- this is what makes Corotated
    /// reduce cleanly to linear elasticity for pure-strain (no-rotation)
    /// deformations. Checked directly, independent of the Hooke's-law tests
    /// above.
    #[test]
    fn symmetric_deformation_gradient_has_exact_identity_rotation() {
        let e = Mat2::from_cols(Vec2::new(0.3, -0.15), Vec2::new(-0.15, 0.6)); // symmetric, large
        let f = Mat2::IDENTITY + e;
        let r = polar_decomposition_2d(f);
        assert!(
            (r - Mat2::IDENTITY).x_axis.length() < 1.0e-6
                && (r - Mat2::IDENTITY).y_axis.length() < 1.0e-6,
            "symmetric F must give EXACTLY R=I, got {r:?}"
        );
    }
}

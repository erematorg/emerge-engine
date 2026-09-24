use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, SnowProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Snow constitutive model: corotated elasticity + SVD-based plasticity.
///
/// On elastic deformation the stress is corotated linear elastic (same form as CorotatedMaterial).
/// On each step, plastic flow is applied via SVD decomposition of F:
///   - Singular values are clamped to [1 - theta_c, 1 + theta_s]
///   - The plastic Jacobian Jp accumulates the volume change that was plastically removed
///   - Elastic hardening h = exp(xi * (1 - Jp)) scales stiffness (compressed snow is stiffer)
///
/// Reference: Stomakhin et al. 2013, §4.2. Identical in sparkl, taichi128, Genesis.
#[derive(Debug, Clone, Copy)]
pub struct StomakhinMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Hardening exponent. Higher = more stiffness gain as snow compacts.
    pub hardening_exponent: f32,
    /// Max compression before plastic flow triggers: Δσ < 1 − θ_c → plastic. (Stomakhin 2013 θ_c)
    pub compression_limit: f32,
    /// Max stretch before plastic flow triggers: Δσ > 1 + θ_s → plastic. (Stomakhin 2013 θ_s)
    pub stretch_limit: f32,
    /// Lower bound on Jp. Prevents wave speed from exploding as h → ∞.
    pub min_plastic_jacobian: f32,
    /// Upper bound on Jp (slight stretch plasticity allowed).
    pub max_plastic_jacobian: f32,
    /// Cohesion pressure: τ += −c · max(1−Jp, 0) · I.
    /// Creates attractive stress in plastically compacted snow (Jp < 1).
    /// 0.0 = no cohesion (Stomakhin 2013 default — powder, loose snow).
    /// ~500–2000 for packed/wet snow that sticks after impact.
    pub cohesion_coeff: f32,
}

impl StomakhinMaterial {
    pub const fn new(
        lambda: f32,
        mu: f32,
        hardening_exponent: f32,
        compression_limit: f32,
        stretch_limit: f32,
        min_plastic_jacobian: f32,
        max_plastic_jacobian: f32,
    ) -> Self {
        Self {
            lambda,
            mu,
            hardening_exponent,
            compression_limit,
            stretch_limit,
            min_plastic_jacobian,
            max_plastic_jacobian,
            cohesion_coeff: 0.0,
        }
    }

    pub const fn with_cohesion(mut self, coeff: f32) -> Self {
        self.cohesion_coeff = coeff;
        self
    }

    /// Stomakhin 2013 canonical plasticity: ξ=10, θ_c=0.025, θ_s=0.0075.
    /// Canonical: E = 1.4e5, ν = 0.2 — matches MPM2D reference and sparkl snow demos.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, 10.0, 0.025, 0.0075, 0.6, 20.0)
    }

    /// Low cohesion: ξ=5, tight compression (θ_c=0.01), tight stretch (θ_s=0.003). Loose powder regime.
    pub fn low_cohesion(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, 5.0, 0.01, 0.003, 0.5, 20.0)
    }

    /// High cohesion: ξ=15, relaxed compression (θ_c=0.04), minimal stretch (θ_s=0.005). Packed/wet regime.
    pub fn high_cohesion(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, 15.0, 0.04, 0.005, 0.6, 20.0).with_cohesion(800.0)
    }
}

impl FromSI<SnowProps> for StomakhinMaterial {
    /// Plasticity params fixed to Stomakhin 2013: ξ=10, θ_c=0.025, θ_s=0.0075.
    fn from_physical(props: &SnowProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        Self::new(lambda, mu, 10.0, 0.025, 0.0075, 0.6, 20.0)
    }
}

impl MaterialModel for StomakhinMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Snow
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        // h = particles.hardening_scale[i], updated each step by plasticity
        let h = particles.hardening_scale[i];
        let mu_eff = self.mu * h;
        let lambda_eff = self.lambda * h;

        // τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
        let f_t = f.transpose();
        let mut tau = 2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY;

        // Cohesion pressure: τ += -c * max(1-Jp, 0) * I -- matches this struct's
        // own doc on `cohesion_coeff`. An addition beyond Stomakhin 2013's base
        // model (disclosed on `cohesion_coeff`'s own doc: "0.0 = no cohesion,
        // Stomakhin 2013 default").
        if self.cohesion_coeff > 0.0 && particles.plastic_volume_ratio[i] < 1.0 {
            tau -= self.cohesion_coeff * (1.0 - particles.plastic_volume_ratio[i]) * Mat2::IDENTITY;
        }
        tau
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;

        let (u, sigma, vt) = svd2(f_trial);

        let sigma_c = Vec2::new(
            sigma
                .x
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            sigma
                .y
                .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
        );

        let jp_new = *ctx.plastic_volume_ratio * (sigma.x * sigma.y) / (sigma_c.x * sigma_c.y);
        // Known: Jp drifts slowly over thousands of substeps due to cumulative SVD rounding.
        // Clamp prevents blow-up but doesn't eliminate drift. Acceptable for LP timescales.
        *ctx.plastic_volume_ratio =
            jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);

        // h clamped [0.1, 7.0]: upper bound is CFL-driven (h=7 → E_eff=35k → ~20 substeps).
        *ctx.hardening_scale = (self.hardening_exponent * (1.0 - *ctx.plastic_volume_ratio))
            .exp()
            .clamp(0.1, 7.0);

        *ctx.deformation_gradient = u * Mat2::from_diagonal(sigma_c) * vt;

        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Snow as u32,
            lambda: self.lambda,
            mu: self.mu,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            volume_ratio_min: self.min_plastic_jacobian,
            volume_ratio_max: self.max_plastic_jacobian,
            cohesion_coeff: self.cohesion_coeff,
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
        // h grows when snow compresses — accounts for stiffening in CFL bound
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
mod analytical_validation_tests {
    use super::*;

    fn particle_with(f: Mat2, hardening_scale: f32, plastic_volume_ratio: f32) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.hardening_scale = hardening_scale;
        p.plastic_volume_ratio = plastic_volume_ratio;
        Particles::from(vec![p])
    }

    /// **Small-strain limit must recover exact linear elasticity.** Snow's
    /// elastic stress formula is IDENTICAL to `CorotatedMaterial`'s (same
    /// documented reference, Stomakhin 2013 eq 5-8), just scaled by the
    /// hardening factor `h` -- at `h=1.0` (undamaged/uncompressed default),
    /// the same small-strain-recovers-Hooke's-law argument applies directly
    /// (see `corotated.rs`'s own test of the same claim). `StomakhinMaterial`
    /// had zero test comparing it to any analytical result before this.
    #[test]
    fn small_strain_matches_hookes_law_at_unit_hardening() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = StomakhinMaterial::new(lambda, mu, 10.0, 0.025, 0.0075, 0.6, 20.0);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_diagonal(Vec2::new(1.0, -0.4));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with(f, 1.0, 1.0);
        let tau = mat.kirchhoff_stress(&particles, 0);

        let eps = delta * e;
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        let predicted = Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps;

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain snow stress (h=1.0) should match linear elasticity: \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// **SVD-clamp plasticity must match its own documented bounds exactly.**
    /// Unlike the strain-space return mappings in sand/rankine/von_mises, this
    /// plasticity is a DIRECT clamp on singular values -- a marginal test here
    /// is a straightforward, exact check: a singular value just inside
    /// [1-theta_c, 1+theta_s] must pass through unchanged; one just outside
    /// must clamp EXACTLY to the boundary.
    #[test]
    fn singular_value_marginally_inside_compression_limit_is_unclamped() {
        let mat = StomakhinMaterial::new(1000.0, 800.0, 10.0, 0.025, 0.0075, 0.6, 20.0);
        let sigma_x = 1.0 - 0.99 * mat.compression_limit; // just inside the floor
        let f = Mat2::from_diagonal(Vec2::new(sigma_x, 1.0));
        let mut particles = particle_with(f, 1.0, 1.0);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        let f_after = particles.deformation_gradient[0];
        assert!(
            (f_after.x_axis.x - sigma_x).abs() < 1.0e-5,
            "singular value inside the compression limit must pass through unchanged: \
             expected {sigma_x}, got {}",
            f_after.x_axis.x
        );
        assert_eq!(
            particles.plastic_volume_ratio[0], 1.0,
            "Jp must not change on a purely elastic step"
        );
    }

    #[test]
    fn singular_value_beyond_compression_limit_clamps_exactly_to_the_boundary() {
        let mat = StomakhinMaterial::new(1000.0, 800.0, 10.0, 0.025, 0.0075, 0.6, 20.0);
        let sigma_x = 1.0 - 1.5 * mat.compression_limit; // comfortably beyond the floor
        let f = Mat2::from_diagonal(Vec2::new(sigma_x, 1.0));
        let mut particles = particle_with(f, 1.0, 1.0);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        let f_after = particles.deformation_gradient[0];

        let expected_clamped = 1.0 - mat.compression_limit;
        assert!(
            (f_after.x_axis.x - expected_clamped).abs() < 1.0e-5,
            "singular value beyond the compression limit must clamp EXACTLY to \
             1-compression_limit={expected_clamped}, got {}",
            f_after.x_axis.x
        );

        // Real, exact analytical claim: Jp_new = Jp_old * (sigma.x*sigma.y) /
        // (sigma_c.x*sigma_c.y) -- with sigma.y=1 unclamped, this reduces to
        // Jp_new = sigma_x / expected_clamped exactly.
        let expected_jp = sigma_x / expected_clamped;
        assert!(
            (particles.plastic_volume_ratio[0] - expected_jp).abs() < 1.0e-5,
            "Jp should update EXACTLY per its own documented formula: expected \
             {expected_jp}, got {}",
            particles.plastic_volume_ratio[0]
        );
    }

    /// **Cohesion must match its own documented formula exactly** -- pins the
    /// formula down numerically so it can't silently drift from `cohesion_coeff`'s
    /// own struct doc.
    #[test]
    fn cohesion_matches_documented_formula_and_is_gated_on_compaction_only() {
        let with_cohesion = StomakhinMaterial::new(1000.0, 800.0, 10.0, 0.025, 0.0075, 0.6, 20.0)
            .with_cohesion(500.0);
        let no_cohesion = StomakhinMaterial::new(1000.0, 800.0, 10.0, 0.025, 0.0075, 0.6, 20.0);
        let expected = -with_cohesion.cohesion_coeff * (1.0 - 0.9);

        // Isolate the cohesion CONTRIBUTION by diffing against the same F/Jp
        // with cohesion off -- the background elastic stress (real, nonzero
        // whenever F != I) must not be mistaken for the cohesion term itself.
        let cohesion_delta = |f: Mat2, jp: f32| -> Mat2 {
            let p_on = particle_with(f, 1.0, jp);
            let p_off = particle_with(f, 1.0, jp);
            with_cohesion.kirchhoff_stress(&p_on, 0) - no_cohesion.kirchhoff_stress(&p_off, 0)
        };

        // Compacted (Jp=0.9) AND currently stretched (j>1).
        let f_stretched = Mat2::from_diagonal(Vec2::new(1.1, 1.0));
        let delta_stretched = cohesion_delta(f_stretched, 0.9);
        assert!(
            (delta_stretched.x_axis.x - expected).abs() < 1.0e-3
                && (delta_stretched.y_axis.y - expected).abs() < 1.0e-3,
            "cohesion contribution should be an isotropic -c*(1-Jp) addition: expected \
             diag={expected}, got {delta_stretched:?}"
        );

        // Compacted (Jp=0.9) but currently AT rest (F=I) -- real cohesion (a
        // bonding pressure) must still apply; it is gated on Jp alone, not
        // on the current elastic state.
        let delta_rest = cohesion_delta(Mat2::IDENTITY, 0.9);
        assert!(
            (delta_rest.x_axis.x - expected).abs() < 1.0e-3,
            "cohesion must apply regardless of current stretch state (only Jp<1 gates it): \
             expected diag={expected}, got {delta_rest:?}"
        );

        // Never compacted (Jp=1.0) -- zero cohesion contribution, any state.
        let delta_uncompacted = cohesion_delta(f_stretched, 1.0);
        assert!(
            delta_uncompacted.x_axis.x.abs() < 1.0e-5 && delta_uncompacted.y_axis.y.abs() < 1.0e-5,
            "Jp=1.0 (never compacted) must produce zero cohesion contribution, got \
             {delta_uncompacted:?}"
        );
    }

    /// **Real, dynamic compaction-hardening test** -- the category-defining
    /// snow behavior ("compressed snow is stiffer," this file's own module
    /// doc, Stomakhin 2013 §4.2) had zero test driving it through real
    /// `update_particle` substeps before this; every existing test above
    /// hand-sets `hardening_scale`/`plastic_volume_ratio` directly rather
    /// than letting them accumulate from real sustained compression. Real,
    /// dynamic mirror of `VonMises`'s permanent-set test and `Rankine`'s
    /// softening test (2026-08-04) -- snow's own real contrast is
    /// HARDENING, the opposite sign of Rankine's softening.
    #[test]
    fn repeated_compaction_genuinely_stiffens_snow_real_hardening_dynamics() {
        // Real Stomakhin 2013 canonical params (xi=10, theta_c=0.025).
        let mat = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);

        // Drive ONE particle through repeated compressive substeps via the
        // real `update_particle` path (Jp/hardening_scale accumulate
        // naturally, not hand-set) -- a sustained uniaxial compression rate,
        // the same real mechanism a footstep/snowball packing would apply.
        let mut particles = particle_with(Mat2::IDENTITY, 1.0, 1.0);
        let dt = 0.01;
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_diagonal(Vec2::new(-0.5, -0.5));
            for _ in 0..20 {
                mat.update_particle(&mut ctx, dt);
            }
        }

        let jp_after = particles.plastic_volume_ratio[0];
        let h_after = particles.hardening_scale[0];
        assert!(
            jp_after < 1.0,
            "sustained real compression must genuinely compact the material (Jp<1), got {jp_after}"
        );
        assert!(
            h_after > 1.0,
            "genuinely compacted snow must be stiffer (hardening_scale>1), got {h_after}"
        );

        // Real stress-stiffening proof: apply the exact same current
        // deformation to a compacted particle vs a fresh (h=1, Jp=1)
        // particle AT THE SAME F -- compacted must produce a LARGER stress
        // for the identical deformation, the real "packed snow resists
        // further compression more" signature, not just an unused number.
        let f_current = particles.deformation_gradient[0];
        let compacted = particle_with(f_current, h_after, jp_after);
        let fresh = particle_with(f_current, 1.0, 1.0);
        let tau_compacted = mat.kirchhoff_stress(&compacted, 0);
        let tau_fresh = mat.kirchhoff_stress(&fresh, 0);
        let mag = |t: Mat2| (t.x_axis.length_squared() + t.y_axis.length_squared()).sqrt();
        assert!(
            mag(tau_compacted) > mag(tau_fresh),
            "compacted snow must show a stiffer (larger-magnitude) stress response to the \
             identical deformation than fresh snow: compacted={:.3}, fresh={:.3}",
            mag(tau_compacted),
            mag(tau_fresh)
        );
    }
}

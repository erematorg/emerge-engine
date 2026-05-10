use glam::{Mat2, Vec2};

use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::Particle;
use crate::materials::svd::svd2;

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
///
/// # Visual quality note
/// Realistic sand piles require ≥4 particles/cell. At 1–2 ppc the elastic regime
/// is visible between plastic projections. The GPU path enables the ppc needed for
/// production-quality results; on CPU keep ppc=2 and accept some elastic feel.
#[derive(Debug, Clone, Copy)]
pub struct SandMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// φ₀: Initial friction angle (radians). Dry sand ≈ 35° = 0.611 rad. (Klar 2016 h₀)
    pub friction_angle: f32,
    /// φ₁: Friction hardening sensitivity — slope of φ(q) near q=0. (Klar 2016 h₁)
    pub hardening_peak: f32,
    /// φ₂: Hardening decay rate — exponential falloff coefficient. (Klar 2016 h₂)
    pub hardening_decay: f32,
    /// φ_r: Residual friction angle (radians). ≈ 10° = 0.175 rad. (Klar 2016 h₃)
    pub friction_residual: f32,
    /// Volume correction factor. 1.0 = full sparkl correction, 0.0 = none.
    pub volume_correction: f32,
    /// Reynolds dilatancy angle ψ (radians). Dense sand ≈ 10–15°.
    ///
    /// When ψ > 0, plastic shear increments drive volumetric expansion:
    /// δεᵥᵖ = sin(ψ) · dq. Physical for dense/compacted sand;
    /// set to 0 for loose/loose-packed sand.
    pub dilatancy_angle: f32,
}

impl SandMaterial {
    /// Construct with Lamé parameters and default Klar 2016 friction-angle hardening.
    ///
    /// Use [`from_young_modulus`](Self::from_young_modulus) if you prefer E/ν inputs.
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            friction_angle: 35.0_f32.to_radians(),
            hardening_peak: 9.0_f32.to_radians(),
            hardening_decay: 0.2,
            friction_residual: 10.0_f32.to_radians(),
            volume_correction: 1.0,
            dilatancy_angle: 0.0,
            
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Matches sparkl/wgsparkl API: `DruckerPragerPlasticity::new(E, nu)`.
    /// Canonical demo value (sparkl basic2): E = 1e5, ν = 0.2.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Preset: dry cohesionless sand, φ=35°, no dilatancy.
    /// Equivalent to Klar 2016 default parameters at the given stiffness.
    pub fn dry_sand(lambda: f32, mu: f32) -> Self {
        Self::new(lambda, mu) // all defaults are already dry-sand correct
    }

    /// Preset: loose sand / silty soil, φ=25°, weaker hardening.
    /// Flows more readily than dry sand — suitable for wet or disturbed terrain.
    pub fn loose_sand(lambda: f32, mu: f32) -> Self {
        Self {
            friction_angle: 25.0_f32.to_radians(),
            hardening_peak: 4.0_f32.to_radians(),
            hardening_decay: 0.1,
            friction_residual: 5.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Preset: dense compacted sand with Reynolds dilatancy (φ=38°, ψ=12°).
    /// Expands under shear — produces more pronounced pile shoulders.
    pub fn dense_sand(lambda: f32, mu: f32) -> Self {
        Self {
            friction_angle: 38.0_f32.to_radians(),
            dilatancy_angle: 12.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Friction coefficient α(q) derived from friction angle φ(q).
    /// φ(q) = friction_angle + (hardening_peak·q − friction_residual)·exp(−hardening_decay·q)
    /// α(q) = √(2/3) · 2·sin(φ) / (3 − sin(φ))
    fn alpha(&self, q: f32) -> f32 {
        let phi = self.friction_angle
            + (self.hardening_peak * q - self.friction_residual)
                * (-self.hardening_decay * q).exp();
        let s = phi.sin();
        (2.0_f32 / 3.0).sqrt() * (2.0 * s) / (3.0 - s)
    }

    /// Drucker-Prager return mapping in log-strain (Hencky) space.
    ///
    /// Returns `Some((projected_sigma, delta_q))` if projection occurred (plastic step),
    /// `None` if the trial state is inside the yield surface (elastic step).
    fn project(&self, sigma: Vec2, log_volume_strain: f32, alpha: f32) -> Option<(Vec2, f32)> {
        // Hencky (logarithmic) strain, shifted by the accumulated volumetric offset.
        let eps = Vec2::new(
            sigma.x.ln() + log_volume_strain * 0.5,
            sigma.y.ln() + log_volume_strain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();

        // Tension cutoff or purely volumetric deformation: project to identity (σ = 1).
        // dq = dev_norm only — friction hardening is driven by shear, not volumetric expansion.
        // Using eps.length() here would include the log_volume_strain offset and cause
        // unbounded q growth in static/settled sand (confirmed by simulation audit 2026-04-18).
        if dev_norm == 0.0 || trace > 0.0 {
            return Some((Vec2::ONE, dev_norm));
        }

        // Yield function: γ = |dev_ε| + ratio · tr · α.
        // Klar 2016 eq. 25, d=2: (d·λ + 2µ)/(2µ) = (2λ+2µ)/(2µ) = (λ+µ)/µ.
        // Verified against sparkl DruckerPragerPlasticity::project and wgsparkl drucker_prager.wgsl.
        let ratio = (self.lambda + self.mu) / self.mu;
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
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let r = polar_decomposition_2d(f);

        let f_t = f.transpose();
        2.0 * self.mu * (f - r) * f_t + self.lambda * (j - 1.0) * j * Mat2::IDENTITY
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn init_particle(&self, particle: &mut Particle) {
        // q=0 gives φ = h0 − h3 = 25° (too weak). The neutral point where
        // φ(q) = h0 exactly is q = h3/h1. Matches sparkl's plastic_hardening=1.0
        // default (which gives φ ≈ 34.2°). At q = h3/h1 the hardening term = 0.
        particle.friction_hardening = if self.hardening_peak > 0.0 {
            self.friction_residual / self.hardening_peak
        } else {
            0.0
        };
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        // 1. Trial elastic deformation gradient.
        let f_trial =
            (Mat2::IDENTITY + dt * particle.velocity_gradient) * particle.deformation_gradient;

        // 2. SVD: F_trial = U · diag(σ) · Vt
        let (u, sigma, vt) = svd2(f_trial);

        // 3. Drucker-Prager return mapping.
        let alpha = self.alpha(particle.friction_hardening);
        let new_sigma = if let Some((proj_sigma, dq)) =
            self.project(sigma, particle.log_volume_strain, alpha)
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

            particle.log_volume_strain += prev_det.ln() - corrected_det.ln();
            // phi(q) asymptotes to friction_angle at large q — cap where the
            // hardening term (h1·q·exp(-h2·q)) is negligible (< 0.01 rad ≈ 0.6°).
            // Prevents unbounded accumulation in long-settled sims.
            let q_max = 5.0 / self.hardening_decay.max(1e-6);
            particle.friction_hardening = (particle.friction_hardening + dq).min(q_max);
            // Reynolds dilatancy: dense sand expands under shear.
            if self.dilatancy_angle > 0.0 {
                particle.log_volume_strain += self.dilatancy_angle.sin() * dq;
            }
            proj_sigma
        } else {
            sigma
        };

        // 4. Recompose F from projected singular values: F = U · diag(σ_new) · Vt
        let sigma_mat = Mat2::from_cols(Vec2::new(new_sigma.x, 0.0), Vec2::new(0.0, new_sigma.y));
        particle.deformation_gradient = u * sigma_mat * vt;

        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.sync_volume_and_density(j);
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::DruckerPrager as u32,
            lambda: self.lambda,
            mu: self.mu,
            dp_h0: self.friction_angle,
            dp_h1: self.hardening_peak,
            dp_h2: self.hardening_decay,
            dp_h3: self.friction_residual,
            // compression_limit repurposed for DP: stores dilatancy angle ψ (radians).
            // Snow uses compression_limit for its singular-value clamp (model 4 only).
            compression_limit: self.dilatancy_angle,
            ..Default::default()
        }
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        elastic_wave_dt(self.lambda, self.mu, 1.0, particle.density, MIN_J, cell_width, material_cfl)
    }
}

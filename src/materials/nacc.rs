use glam::{Mat2, Vec2};

use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particle;
use crate::materials::svd::svd2;

/// Non-Associated Cam-Clay (NACC) elastoplastic solid.
///
/// Elastic energy: Neo-Hookean (κ bulk, µ shear).
/// Yield surface: ellipse in (p, q) space — q² + M²·(p + β·p₀)·(p − p₀) ≤ 0
///   p = −tr(σ)/d  (mean pressure, positive in compression)
///   q = deviatoric stress magnitude
///   p₀ = preconsolidation pressure (hardens under plastic volumetric compression)
///   M  = friction slope (tan of friction angle)
///   β  = ellipse shift (cohesion term; 0 = no tensile strength)
///
/// Plastic variable: `nacc_alpha` (stored in `particle.log_volume_strain`).
///   α tracks accumulated plastic volumetric compression.
///   Positive α = plastic compression → larger p₀ → harder material.
///   Init: α = 0 (unstressed, reference state).
///
/// Unlike Drucker-Prager (cone), NACC has a *cap* — it limits compression too.
/// This captures preconsolidation: previously consolidated soils yield at lower stress.
///
/// Reference: Klar et al. 2016; sparkl `plasticity_nacc.rs`.
///
/// # Natural phenomena
/// - Saturated clay / soft sediment: κ ≈ 1e4–1e5, M ≈ 1.2–1.8
/// - Wet compressed soil (paddy fields, river banks): M ≈ 0.8–1.2
/// - Biological soft tissue under large compression: κ ≈ 1e3–1e4
/// - Reconstituted (remoulded) clay: hardening_factor ξ ≈ 1–5
#[derive(Debug, Clone, Copy)]
pub struct NaccMaterial {
    /// Shear modulus µ.
    pub mu: f32,
    /// Bulk modulus κ. Note: λ (Lamé) = κ − 2µ/3.
    pub kappa: f32,
    /// Friction slope M — controls yield surface width in q direction.
    /// Related to friction angle φ: M = (6 sin φ)/(3 − sin φ) · √((6−d)/2) in 2D.
    /// Typical: 1.0–2.0.
    pub friction: f32,
    /// Cohesion (beta β) — shifts yield surface min tip.
    /// 0.0 = no tensile strength (standard). 1.0 = symmetric around p=0.
    pub cohesion: f32,
    /// Hardening factor ξ. Controls how fast p₀ grows: p₀ = κ·(1e-5 + sinh(ξ·max(−α,0))).
    /// 0.0 = no hardening (perfect plasticity cap). Typical: 1.0–5.0.
    pub hardening_factor: f32,
    /// Enable volumetric hardening. If false, p₀ stays fixed (perfect plasticity cap).
    pub hardening_enabled: bool,
    pub min_density: f32,
}

impl NaccMaterial {
    pub fn new(mu: f32, kappa: f32, friction: f32, cohesion: f32, hardening_factor: f32) -> Self {
        Self {
            mu,
            kappa,
            friction,
            cohesion,
            hardening_factor,
            hardening_enabled: hardening_factor > 0.0,
            min_density: 1.0e-6,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    /// Friction slope M and cohesion β set separately.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32, friction: f32, cohesion: f32, hardening_factor: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        let kappa = lambda + (2.0 / 3.0) * mu;
        Self::new(mu, kappa, friction, cohesion, hardening_factor)
    }

    /// Saturated soft clay. κ = 2e4, M = 1.2, β = 0, ξ = 2.
    pub fn soft_clay() -> Self {
        let (lambda, mu) = lame_from_young(5.0e4, 0.3);
        let kappa = lambda + (2.0 / 3.0) * mu;
        Self::new(mu, kappa, 1.2, 0.0, 2.0)
    }

    /// Wet compressed soil (paddy field, river bank). Stiffer, higher friction.
    pub fn wet_soil() -> Self {
        let (lambda, mu) = lame_from_young(2.0e5, 0.35);
        let kappa = lambda + (2.0 / 3.0) * mu;
        Self::new(mu, kappa, 1.0, 0.0, 3.0)
    }

    /// Biological soft tissue under large compression. Very soft, no tensile strength.
    pub fn soft_tissue() -> Self {
        let (lambda, mu) = lame_from_young(5.0e3, 0.45);
        let kappa = lambda + (2.0 / 3.0) * mu;
        Self::new(mu, kappa, 1.5, 0.0, 1.0)
    }

    /// NACC yield surface projection. Returns updated (F, alpha).
    ///
    /// Three cases from sparkl canonical:
    ///   A — p_trial > p₀:          compress past preconsolidation → project to max cap
    ///   B — p_trial < −β·p₀:       pull past tensile limit → project to min tip
    ///   C — yield surface exceeded: project onto ellipse
    ///   elastic: inside yield surface → no projection
    fn project(&self, f: Mat2, mut alpha: f32) -> (Mat2, f32) {
        let xi = self.hardening_factor;
        let beta = self.cohesion;
        let m = self.friction;

        let (u, sigma, vt) = svd2(f);

        let sv = Vec2::new(sigma.x, sigma.y);
        let sv_sq = sv * sv;
        let sv_sq_trace = sv_sq.x + sv_sq.y;

        // Current preconsolidation pressure.
        let p0 = self.kappa * (1.0e-5 + (xi * (-alpha).max(0.0)).sinh());

        // J = det(F) = product of singular values.
        let j_e_tr = (sv.x * sv.y).max(1.0e-6_f32);

        // Trial deviatoric stress: s_tr = µ·J^(−2/2)·dev(B_eigenvalues)
        // In 2D: J^(-1) · dev(σ²)
        let s_tr = self.mu * j_e_tr.recip() * (sv_sq - Vec2::splat(sv_sq_trace * 0.5));

        // Trial pressure p = −κ/2·(J − 1/J)·J = −κ/2·(J²−1)
        let psi_kappa = self.kappa * 0.5 * (j_e_tr - j_e_tr.recip());
        let p_tr = -psi_kappa * j_e_tr;

        // Case A: past max cap (over-consolidation / compressive failure).
        if p_tr > p0 {
            let j_n1 = (-2.0 * p0 / self.kappa + 1.0).max(1.0e-8_f32).sqrt();
            let sv_new = j_n1.powf(0.5); // J^(1/2) since d=2
            let sigma_new = Vec2::splat(sv_new);
            if self.hardening_enabled {
                alpha += (j_e_tr / j_n1).ln();
            }
            return (reconstruct(u, sigma_new, vt), alpha);
        }

        // Case B: past min tip (tensile failure).
        if p_tr < -beta * p0 {
            let j_n1 = (2.0 * beta * p0 / self.kappa + 1.0).max(1.0e-8_f32).sqrt();
            let sv_new = j_n1.powf(0.5);
            let sigma_new = Vec2::splat(sv_new);
            if self.hardening_enabled {
                alpha += (j_e_tr / j_n1).ln();
            }
            return (reconstruct(u, sigma_new, vt), alpha);
        }

        // Yield function: y = (1+2β)·(6−2)/2·‖s_tr‖² + M²·(p_tr+β·p₀)·(p_tr−p₀)
        // In 2D: d=2, factor = (6−d)/2 = 2.
        let y0 = (1.0 + 2.0 * beta) * 2.0_f32; // (6-d)/2 with d=2
        let y1 = m * m * (p_tr + beta * p0) * (p_tr - p0);
        let s_norm_sq = s_tr.x * s_tr.x + s_tr.y * s_tr.y;
        let y = y0 * s_norm_sq + y1;

        if y < 1.0e-4 {
            // Inside yield surface — elastic, no projection.
            return (f, alpha);
        }

        // Hardening: move p₀ to reduce y to zero.
        if self.hardening_enabled && p0 > 1.0e-4 && p_tr < p0 - 1.0e-4 && p_tr > -beta * p0 + 1.0e-4 {
            let p_c = (1.0 - beta) * p0 * 0.5;
            let q_tr = (2.0_f32).sqrt() * s_tr.length();
            let dir = Vec2::new(p_c - p_tr, -q_tr);
            let dir = dir.normalize_or_zero();
            let c = m * m * (p_c + beta * p0) * (p_c - p0);
            let b = m * m * dir.x * (2.0 * p_c - p0 + beta * p0);
            let a = m * m * dir.x * dir.x + (1.0 + 2.0 * beta) * dir.y * dir.y;
            let discr = (b * b - 4.0 * a * c).max(0.0).sqrt();
            let l1 = (-b + discr) / (2.0 * a);
            let l2 = (-b - discr) / (2.0 * a);
            let p1 = p_c + l1 * dir.x;
            let p2 = p_c + l2 * dir.x;
            let p_x = if (p_tr - p_c) * (p1 - p_c) > 0.0 { p1 } else { p2 };
            let j_e_x = (-2.0 * p_x / self.kappa + 1.0).abs().max(1.0e-8_f32).sqrt();
            if j_e_x > 1.0e-4 {
                alpha += (j_e_tr / j_e_x).ln();
            }
        }

        // Case C: project onto yield surface.
        // B_n1 eigenvalues: solve for scaled deviatoric + isotropic.
        let b_n1 = (-y1 / y0.max(1.0e-10_f32)).max(0.0).sqrt()
            * (j_e_tr.powf(2.0 / 2.0) / self.mu)
            * s_tr.normalize_or_zero()
            + Vec2::splat(sv_sq_trace / 2.0);

        let sv_new = Vec2::new(b_n1.x.max(1.0e-8_f32).sqrt(), b_n1.y.max(1.0e-8_f32).sqrt());
        (reconstruct(u, sv_new, vt), alpha)
    }
}

#[inline]
fn reconstruct(u: Mat2, sigma: Vec2, vt: Mat2) -> Mat2 {
    u * Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y)) * vt
}

impl MaterialModel for NaccMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Nacc
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = f.determinant().max(MIN_J);

        // NeoHookean Simo-Pister vol-dev split, with κ = λ + 2µ/3.
        // Identical to NeoHookeanMaterial but expressed in (κ, µ) terms.
        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(Vec2::splat(tr_b * 0.5));

        let dev_stress = (self.mu / j) * dev_b;
        let vol_stress = (self.kappa * 0.5 * (j * j - 1.0)) * Mat2::IDENTITY;

        dev_stress + vol_stress
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.volume
    }

    fn update_particle(&self, particle: &mut Particle, _dt: f32) {
        let alpha = particle.log_volume_strain; // nacc_alpha reuses this field
        let (new_f, new_alpha) = self.project(particle.deformation_gradient, alpha);
        particle.deformation_gradient = new_f;
        particle.log_volume_strain = new_alpha;

        // Keep volume/density consistent with updated F.
        let j = new_f.determinant().max(MIN_J);
        particle.volume = particle.initial_volume * j;
        particle.density = if particle.volume > MIN_J {
            particle.mass / particle.volume
        } else {
            particle.density
        };
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.log_volume_strain = 0.0; // nacc_alpha starts at 0 (unstressed)
    }

    fn needs_cpu_update(&self) -> bool {
        true
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: f32, material_cfl: f32, _viscous_cfl: f32) -> f32 {
        let lambda = self.kappa - (2.0 / 3.0) * self.mu;
        elastic_wave_dt(lambda, self.mu, particle.hardening_scale, particle.density, self.min_density, cell_width, material_cfl)
    }

    fn params(&self) -> MaterialParams {
        // GPU uses NeoHookean stress (model 2) with λ = κ − 2µ/3, µ = µ.
        // Plasticity runs CPU-only via needs_cpu_update=true.
        let lambda = self.kappa - (2.0 / 3.0) * self.mu;
        MaterialParams {
            model: ConstitutiveModel::NeoHookean as u32,
            lambda,
            mu: self.mu,
            hardening_exponent: self.hardening_factor,
            compression_limit: self.cohesion,  // β
            stretch_limit: self.friction,       // M
            ..Default::default()
        }
    }
}

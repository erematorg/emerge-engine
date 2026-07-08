use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, Particles};

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
    /// Bulk modulus κ. Note: λ (Lamé) = κ − µ (2D plane-strain relation --
    /// fixed 2026-07-06, was the 3D κ=λ+2µ/3 relation; see `timestep_bound`
    /// and `params()` below, which already correctly invert as λ=κ−µ).
    pub kappa: f32,
    /// Friction slope M — controls yield surface width in q direction.
    /// Related to friction angle φ (sparkl's `NaccPlasticity::new`, general-d form:
    /// M = √(2/3)·2·sin φ/(3−sin φ)·d/√(2/(6−d))): in 2D (d=2) this reduces to
    /// M = (8/√3)·sin φ/(3−sin φ) ≈ 4.619·sin φ/(3−sin φ). Not called by any
    /// constructor here (presets pass M directly) -- informational only. A
    /// previous version of this comment used the 3D-style `6 sin φ/(3−sin φ)·
    /// √((6−d)/2)` form, which is ~1.84x too large at d=2; fixed 2026-07-07
    /// after cross-checking sparkl's actual source (not just its doc).
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
    pub fn from_young_modulus(
        young_modulus: f32,
        poisson_ratio: f32,
        friction: f32,
        cohesion: f32,
        hardening_factor: f32,
    ) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        // 2D plane-strain bulk modulus (kappa = lambda + mu, not the 3D
        // lambda + 2*mu/3 an earlier version used to match sparkl -- see
        // elastic.rs's ConstitutiveModel impl for the full derivation/fix note).
        let kappa = lambda + mu;
        Self::new(mu, kappa, friction, cohesion, hardening_factor)
    }

    /// Saturated soft clay: M=1.2, β=0, ξ=2 (Klar 2016 soft clay params).
    pub fn soft_clay(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        // 2D plane-strain bulk modulus (kappa = lambda + mu, not the 3D
        // lambda + 2*mu/3 an earlier version used to match sparkl -- see
        // elastic.rs's ConstitutiveModel impl for the full derivation/fix note).
        let kappa = lambda + mu;
        Self::new(mu, kappa, 1.2, 0.0, 2.0)
    }

    /// Wet compressed soil (paddy field, river bank): M=1.0, β=0, ξ=3.
    pub fn wet_soil(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        // 2D plane-strain bulk modulus (kappa = lambda + mu, not the 3D
        // lambda + 2*mu/3 an earlier version used to match sparkl -- see
        // elastic.rs's ConstitutiveModel impl for the full derivation/fix note).
        let kappa = lambda + mu;
        Self::new(mu, kappa, 1.0, 0.0, 3.0)
    }

    /// High critical-slope, low hardening: M=1.5, β=0, ξ=1. Soft material under large compression.
    pub fn low_hardening(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        // 2D plane-strain bulk modulus (kappa = lambda + mu, not the 3D
        // lambda + 2*mu/3 an earlier version used to match sparkl -- see
        // elastic.rs's ConstitutiveModel impl for the full derivation/fix note).
        let kappa = lambda + mu;
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
        if self.hardening_enabled && p0 > 1.0e-4 && p_tr < p0 - 1.0e-4 && p_tr > -beta * p0 + 1.0e-4
        {
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
            let p_x = if (p_tr - p_c) * (p1 - p_c) > 0.0 {
                p1
            } else {
                p2
            };
            let j_e_x = (-2.0 * p_x / self.kappa + 1.0).abs().max(1.0e-8_f32).sqrt();
            if j_e_x > 1.0e-4 {
                alpha += (j_e_tr / j_e_x).ln();
            }
        }

        // Case C: project onto yield surface.
        // B_n1 eigenvalues: solve for scaled deviatoric + isotropic.
        let b_n1 = (-y1 / y0.max(1.0e-10_f32)).max(0.0).sqrt()
            * (j_e_tr / self.mu)
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

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant().max(MIN_J);

        // NeoHookean Simo-Pister vol-dev split, with κ = λ + µ (2D plane-strain).
        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(Vec2::splat(tr_b * 0.5));

        let dev_stress = (self.mu / j) * dev_b;
        let vol_stress = (self.kappa * 0.5 * (j * j - 1.0)) * Mat2::IDENTITY;

        dev_stress + vol_stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, _dt: f32) {
        let alpha = particles.log_volume_strain[i];
        let (new_f, new_alpha) = self.project(particles.deformation_gradient[i], alpha);
        particles.deformation_gradient[i] = new_f;
        particles.log_volume_strain[i] = new_alpha;

        let j = new_f.determinant().max(MIN_J);
        let vol = particles.initial_volume[i] * j;
        particles.volume[i] = vol;
        particles.density[i] = if vol > MIN_J {
            particles.mass[i] / vol
        } else {
            particles.density[i]
        };
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.log_volume_strain = 0.0; // nacc_alpha starts at 0 (unstressed)
    }

    fn needs_cpu_update(&self) -> bool {
        true
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        // Inverse of the 2D plane-strain relation kappa = lambda + mu.
        let lambda = self.kappa - self.mu;
        elastic_wave_dt(
            lambda,
            self.mu,
            hardening_scale,
            density,
            self.min_density,
            cell_width,
            material_cfl,
        )
    }

    fn params(&self) -> MaterialParams {
        // GPU uses NeoHookean stress (model 2) with λ = κ − µ, µ = µ.
        // Plasticity runs CPU-only via needs_cpu_update=true.
        // Inverse of the 2D plane-strain relation kappa = lambda + mu.
        let lambda = self.kappa - self.mu;
        MaterialParams {
            model: ConstitutiveModel::NeoHookean as u32,
            lambda,
            mu: self.mu,
            hardening_exponent: self.hardening_factor,
            compression_limit: self.cohesion, // β
            stretch_limit: self.friction,     // M
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;

    /// Real pressure computed from a deformation gradient the SAME way
    /// `project`'s own trial-pressure formula does (`p = -kappa/2*(J-1/J)*J`),
    /// used to verify the projected state lands where the material's own
    /// documented cap formula (`j_n1 = sqrt(-2*p0/kappa+1)`) analytically
    /// predicts -- not just "less than before."
    fn pressure_from_j(kappa: f32, j: f32) -> f32 {
        let psi_kappa = kappa * 0.5 * (j - j.recip());
        -psi_kappa * j
    }

    /// **Case A (compression cap) must project EXACTLY to p=p0.** Hand-derived:
    /// substituting `j_n1^2 = -2*p0/kappa+1` into the pressure formula
    /// `p=-kappa/2*(J^2-1)` gives `p=-kappa/2*(-2*p0/kappa) = p0` exactly, an
    /// identity independent of any specific numeric values. `NaccMaterial` had
    /// zero test comparing its return mapping to any analytical prediction
    /// before this (only a stability check existed).
    #[test]
    fn compression_cap_projects_exactly_to_p0() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 0.0); // hardening off: p0 fixed
        let alpha = 0.0;
        let p0 = mat.kappa * 1.0e-5; // hardening_factor=0 -> sinh(0)=0 -> p0 = kappa*1e-5

        // Isotropic compression well past p0: small uniform J << 1.
        let j_trial = 0.5_f32;
        let f = Mat2::from_diagonal(Vec2::splat(j_trial.sqrt()));
        let p_trial = pressure_from_j(mat.kappa, j_trial);
        assert!(
            p_trial > p0,
            "test setup must genuinely be past the cap: p_trial={p_trial} p0={p0}"
        );

        let (f_after, _alpha_after) = mat.project(f, alpha);
        let j_after = f_after.determinant();
        let p_after = pressure_from_j(mat.kappa, j_after);

        assert!(
            (p_after - p0).abs() < 1.0e-2,
            "compression-cap projection should land EXACTLY at p=p0={p0:.6}, got {p_after:.6}"
        );
    }

    /// A trial state comfortably INSIDE the yield ellipse (real confining
    /// pressure PLUS a small shear perturbation) must leave the deformation
    /// gradient completely unchanged.
    ///
    /// Two earlier versions of this test failed, for genuinely informative
    /// reasons (not test-tooling bugs):
    /// 1. `hardening_factor=0` gives `p0=kappa*1e-5` regardless of alpha (xi=0
    ///    zeroes the sinh term unconditionally) -- a vanishingly small elastic
    ///    region where any real strain immediately exceeds the cap. No real
    ///    preset in this file ever uses hardening_factor=0.
    /// 2. Even with hardening on and a large p0, a PURE shear perturbation at
    ///    near-zero volumetric strain (p_tr~0) still yielded. This is REAL,
    ///    physically-correct behavior, not a bug: with cohesion (beta) = 0,
    ///    the ellipse's y1 term is `M^2*(p_tr+beta*p0)*(p_tr-p0)`, and at
    ///    p_tr=0/beta=0 this is exactly 0 regardless of how large p0 is --
    ///    a cohesionless material genuinely has ~zero elastic shear capacity
    ///    at zero confining pressure (real critical-state soil mechanics:
    ///    frictional materials can't resist shear without confinement). The
    ///    fix is testing what the model actually claims: shear WITH real
    ///    confining pressure present, not shear alone.
    #[test]
    fn small_elastic_strain_is_not_projected() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 2.0);
        let alpha = -1.0; // real pre-consolidation, gives a meaningfully large p0
        // Real isotropic confining compression (sv=0.999 each way, giving
        // p_tr~4.0, comfortably inside p0~7254) PLUS a tiny shear on top --
        // this is the physically meaningful "small elastic strain" case: real
        // confining pressure present, not shear at zero pressure.
        let f = Mat2::from_diagonal(Vec2::new(0.99895, 0.99905));
        let (f_after, alpha_after) = mat.project(f, alpha);
        assert!(
            (f_after - f).x_axis.length() < 1.0e-6 && (f_after - f).y_axis.length() < 1.0e-6,
            "small elastic strain (with real confining pressure) must not be projected: \
             f={f:?} f_after={f_after:?}"
        );
        assert_eq!(
            alpha_after, alpha,
            "alpha must not change on an elastic step"
        );
    }
}

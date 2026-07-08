use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, GranularProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, Particles};

/// µ(I)-rheology sand — rate-dependent Drucker-Prager (Cicoira et al. / matter "DPMui").
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
/// Reference: Cicoira, Blatny, Li, Trottet & Gaume 2022, "Towards a predictive
/// multi-phase model for alpine mass movements and process cascades," Engineering
/// Geology 310:106866 (Blatny is a co-author, not first author -- an earlier
/// version of this comment misattributed it); the µ(I) functional form itself
/// traces to Jop, Forterre & Pouliquen 2006, Nature 441:727-730. Also:
/// matter/src/simulation/plasticity.cpp DPMui.
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

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;

    fn run_one_step(mat: &MuIRheologyMaterial, sigma: Vec2, dt: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        mat.init_particle(&mut p);
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles, 0, dt);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    /// **Tension cutoff**: `p_trial <= 0` (net expansion/tension) must project
    /// EXACTLY to identity (real dry granular material has no tensile
    /// strength) and reset friction_hardening to mu_static. `MuIRheologyMaterial`
    /// had zero test comparing its return mapping to any analytical claim
    /// before this (only a bound-check on friction_hardening's own range
    /// existed, self-referential to the material's own configured values).
    #[test]
    fn tension_projects_exactly_to_identity() {
        let mat = MuIRheologyMaterial::new(2000.0, 3000.0);
        let sigma = Vec2::new(1.05, 1.02); // net expansion -- p_trial < 0
        let (sigma_after, mu_after) = run_one_step(&mat, sigma, 0.1);
        assert!(
            (sigma_after - Vec2::ONE).length() < 1.0e-5,
            "tension (p_trial<=0) must project exactly to identity, got {sigma_after:?}"
        );
        assert_eq!(
            mu_after, mat.mu_static,
            "tension resets friction_hardening to mu_static"
        );
    }

    /// **Marginal state at the quasi-static yield surface (q=mu_static*p) must
    /// stay elastic.** Comfortably inside (99%).
    #[test]
    fn marginal_state_at_quasistatic_yield_does_not_flow() {
        let mat = MuIRheologyMaterial::new(2000.0, 3000.0);
        // eps.y=0, so p_trial=-k_2d*eps.x, q_trial=sqrt(2)*mu*|dev|=sqrt(2)*mu*|eps.x/2|.
        // Solve eps.x (negative, compressive) for q_trial = 0.99*mu_static*p_trial.
        let k_2d = mat.lambda + mat.mu;
        // p_trial = -k_2d*eps.x (eps.x negative -> p_trial positive)
        // q_trial = sqrt(2)*mu*(|eps.x|/2) = mu*|eps.x|/sqrt(2)
        // Set q_trial = 0.99*mu_static*p_trial and solve for eps.x < 0:
        // mu*(-eps.x)/sqrt(2) = 0.99*mu_static*(-k_2d*eps.x)
        // mu/sqrt(2) = 0.99*mu_static*k_2d  <-- this doesn't depend on eps.x's magnitude,
        // it's a RATIO condition -- so scale eps.x arbitrarily small and check the ratio
        // holds by construction via direct sigma construction instead.
        let eps_x = -0.01_f32;
        let p_trial = -k_2d * eps_x;
        let target_q = 0.99 * mat.mu_static * p_trial;
        // q_trial = mu*|eps.x|/sqrt(2) is FIXED by eps_x and mu -- to hit an
        // arbitrary target_q instead, construct eps.y independently: dev_norm
        // needed = target_q/(sqrt(2)*mu), then eps.y = tr - eps.x where
        // tr = 2*dev_norm_signed... simplest: solve eps.y directly from
        // dev = (eps.x-eps.y)/2 (for this 2-component system, dev_norm=|eps.x-eps.y|/sqrt(2)).
        let dev_norm_needed = target_q / (std::f32::consts::SQRT_2 * mat.mu);
        // dev = eps - tr/2 * I; for eps=(ex,ey), dev.x=(ex-ey)/2, dev.y=(ey-ex)/2,
        // dev_norm = |ex-ey|/sqrt(2). Solve ey given ex and desired dev_norm:
        let ex = eps_x;
        let ey = ex - dev_norm_needed * std::f32::consts::SQRT_2;
        let sigma = Vec2::new(ex.exp(), ey.exp());

        let (sigma_after, mu_after) = run_one_step(&mat, sigma, 0.1);
        assert!(
            (sigma_after - sigma).length() < 1.0e-4,
            "state inside the quasi-static yield surface should stay elastic: \
             sigma={sigma:?} sigma_after={sigma_after:?}"
        );
        assert_eq!(
            mu_after, mat.mu_static,
            "mu must stay at mu_static on an elastic step"
        );
    }

    /// **Beyond yield: the analytically-solved `gamma_dot` must be a genuine
    /// root of the material's own documented quadratic** `a*g^2+b*g+c=0`
    /// (checked by direct substitution, not just trusted from the derivation).
    #[test]
    fn gamma_dot_is_a_real_root_of_its_own_quadratic() {
        let mat = MuIRheologyMaterial::new(2000.0, 3000.0);
        let dt = 0.1_f32;
        // Comfortably past yield: NET COMPRESSION (tr<0, real p_trial>0) plus
        // enough shear to exceed mu_static*p_trial. First version used
        // eps_x=-0.05,eps_y=0.05 (zero net trace), which hit the p_trial<=0
        // tension-cutoff branch instead of real yield -- fixed by using
        // asymmetric values with genuine net compression.
        let eps_x = -0.08_f32;
        let eps_y = 0.02_f32;
        let sigma = Vec2::new(eps_x.exp(), eps_y.exp());

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        mat.init_particle(&mut p);
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles, 0, dt);

        // Recompute a, b, c the SAME way `update_particle` does, to verify the
        // ACTUAL solved gamma_dot (recovered from the friction_hardening
        // output + the known mu(I) formula) satisfies a*g^2+b*g+c=0.
        let f_trial = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        let (_, sigma_svd, _) = svd2(f_trial);
        let eps = hencky_strains(sigma_svd);
        let tr = eps.x + eps.y;
        let k_2d = mat.lambda + mat.mu;
        let p_trial = -k_2d * tr;
        let dev = eps - Vec2::splat(tr * 0.5);
        let dev_norm = dev.length();
        let q_trial = std::f32::consts::SQRT_2 * mat.mu * dev_norm;
        let q_yield = mat.mu_static * p_trial;
        let delta_q = q_trial - q_yield;
        let sqrt_p = p_trial.sqrt();
        let a = mat.mu * dt;
        let b = p_trial * (mat.mu_dynamic - mat.mu_static) + a * mat.inertial_q * sqrt_p - delta_q;
        let c = -delta_q * mat.inertial_q * sqrt_p;

        // Recover gamma_dot from the OUTPUT mu_i via the documented formula:
        // mu_i = mu_static + (mu_dynamic-mu_static)/(Q*sqrt_p/gamma_dot + 1)
        // => Q*sqrt_p/gamma_dot = (mu_dynamic-mu_static)/(mu_i-mu_static) - 1
        let mu_i = particles.friction_hardening[0];
        assert!(
            mu_i > mat.mu_static,
            "beyond yield, mu_i must exceed mu_static"
        );
        let denom = (mat.mu_dynamic - mat.mu_static) / (mu_i - mat.mu_static) - 1.0;
        let gamma_dot_recovered = mat.inertial_q * sqrt_p / denom;

        let residual = a * gamma_dot_recovered * gamma_dot_recovered + b * gamma_dot_recovered + c;
        // RELATIVE tolerance: a/b/c here are O(1e2-1e4), so a fixed absolute
        // tolerance is meaningless -- compare against the largest term's scale.
        let scale = (a * gamma_dot_recovered * gamma_dot_recovered)
            .abs()
            .max((b * gamma_dot_recovered).abs())
            .max(c.abs());
        assert!(
            residual.abs() / scale < 1.0e-3,
            "recovered gamma_dot={gamma_dot_recovered:.6} should satisfy the material's \
             own quadratic a*g^2+b*g+c=0 (a={a:.4} b={b:.4} c={c:.4}), residual={residual:.6} \
             relative={:.2e}",
            residual.abs() / scale
        );
    }
}

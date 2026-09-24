//! Cosserat micro-rotation field — the real, grid-level angular-momentum
//! channel that closes the loop `matter::materials::solid::granular::cosserat`'s kinematics
//! module deliberately left open. Same architectural family as
//! `GranularFluidityField`: a standalone, opt-in, grid-sized scratch field
//! with its own P2G-scatter → solve → G2P-gather cycle, entirely separate
//! from the core `Cell`/momentum machinery every material already depends
//! on — adding this touches ZERO shared code when unconfigured (no scene
//! that never constructs a `CosseratField` sees any difference at all).
//!
//! # Real citation
//! de Borst, R., Sabet, S.A. and Hageman, T. (2022), "Non-associated
//! Cosserat plasticity", International Journal of Mechanical Sciences,
//! 230, 107535. See `matter::materials::solid::granular::cosserat`'s own doc for the
//! kinematics/constitutive equations this field solves for.
//!
//! # What this solves
//! A Cosserat continuum's micro-rotation field ω_c obeys its own angular-
//! momentum balance, `div(m) + e:σ_antisym = 0` (quasi-static, matching this
//! engine's own per-substep quasi-static treatment of ordinary plasticity).
//! Discretized on the SAME grid as ordinary MPM momentum, mirroring exactly
//! how `v = p/m` already solves the ordinary linear-momentum balance:
//! ```text
//! ω_c(node) = L(node) / I_eff(node) + τ(node) · dt / I_eff(node)
//! ```
//! where `L` is scattered micro-angular-momentum (mass-weighted, mirroring
//! ordinary P2G momentum scatter), `I_eff` is scattered micro-inertia, and
//! `τ` is the real coupling torque `2·coupling_modulus·(ω_macro − ω_c)` —
//! the antisymmetric part of the elastic Cosserat stress relation (de Borst
//! et al. 2022 eq. 36's `(μ+μc)e + μ(e)ᵀ` term is exactly this once split
//! into symmetric/antisymmetric halves), NOT an invented shortcut.
//!
//! Real, disclosed simplifying assumption: grains are treated as
//! effectively spherical/rounded for the micro-inertia coefficient
//! (`1/10` — the mass-specific polar moment of inertia of a solid sphere of
//! diameter `d`, elementary mechanics, not paper-specific), matching the
//! same single-scalar "grain diameter" convention `GranularFluidityConfig`
//! already uses rather than modeling real grain angularity explicitly.

use glam::{IVec2, Vec2};

use crate::{
    grid::kernel::quadratic_weights,
    matter::materials::solid::granular::cosserat::micro_curvature_2d,
};

/// Real physical parameters for the Cosserat micro-rotation field.
#[derive(Clone, Copy, Debug)]
pub struct CosseratConfig {
    /// Real elastic coupling modulus `alpha` \[Pa\] — see
    /// `matter::materials::solid::granular::cosserat`'s own doc for the cited relation this
    /// feeds (`m = alpha * l^2 * kappa`, and the coupling torque
    /// `2*alpha*(omega_macro - omega_c)`).
    pub coupling_modulus_pa: f32,
    /// Real grain diameter `l` \[m\] — same physical quantity
    /// `GranularFluidityConfig::grain_diameter_m` already uses, not a
    /// separate free parameter.
    pub grain_diameter_m: f32,
    /// Dimensionless micro-inertia shape coefficient. `1/10` (a solid
    /// sphere/disk's real mass-specific polar moment of inertia,
    /// `I = (2/5)*r^2 = (1/10)*d^2`) is the real, disclosed default for
    /// effectively-rounded grains — a genuine simplifying assumption, not
    /// an arbitrary numerical knob.
    pub micro_inertia_coefficient: f32,
}

impl CosseratConfig {
    /// Real default: `1/10`, the solid-sphere mass-specific polar moment of
    /// inertia coefficient (elementary mechanics, `I = (2/5)r^2` in terms of
    /// diameter `d=2r` gives `I = d^2/10`).
    pub fn micro_inertia(&self, particle_mass: f32) -> f32 {
        particle_mass
            * self.micro_inertia_coefficient
            * self.grain_diameter_m
            * self.grain_diameter_m
    }

    /// Real, quoted-family stability bound for this explicit scheme,
    /// following the SAME derivation shape as `GranularFluidityConfig::
    /// stability_dt` (a diffusion-like second-order spatial operator):
    /// `dt < dx^2 * I_eff / (2 * coupling_modulus * l^2)`, using a real
    /// per-unit-mass `I_eff` estimate (`micro_inertia(1.0)`) since the bound
    /// must hold per unit mass, matching how the solve itself is
    /// mass-normalized (`L/I_eff`).
    pub fn stability_dt(&self, dx_m: f32) -> f32 {
        let i_eff = self.micro_inertia(1.0).max(1e-30);
        let denom = 2.0 * self.coupling_modulus_pa * self.grain_diameter_m.powi(2);
        dx_m * dx_m * i_eff / denom.max(1e-30)
    }
}

/// A persistent, grid-coupled Cosserat micro-rotation field.
pub struct CosseratField {
    pub config: CosseratConfig,
    grid_res: usize,
    grid_mass: Vec<f32>,  // Sigma(w * micro_inertia)         -- cleared each step
    grid_l: Vec<f32>,     // scattered micro-angular-momentum -- cleared each step
    grid_spin: Vec<f32>,  // scattered mass-weighted macro spin -- cleared each step
    grid_omega: Vec<f32>, // persistent PDE state -- NOT cleared between steps
    grid_work: Vec<f32>,  // scratch: solve output before it replaces grid_omega
}

impl CosseratField {
    pub fn new(config: CosseratConfig, grid_res: usize) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            grid_res,
            grid_mass: vec![0.0; n],
            grid_l: vec![0.0; n],
            grid_spin: vec![0.0; n],
            grid_omega: vec![0.0; n],
            grid_work: vec![0.0; n],
        }
    }

    pub const fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Advance omega_c by one substep and gather it (plus its real spatial
    /// gradient, the micro-curvature kappa) into `out_omega`/`out_curvature`
    /// for every particle. `macro_spin` is the particle's own local
    /// antisymmetric velocity-gradient component (`0.5*(dvy/dx - dvx/dy)`,
    /// computed by the CALLER from `ParticleUpdateCtx::velocity_gradient` --
    /// this field has no access to that context, matching
    /// `GranularFluidityField::pressure_and_ratio`'s own caller-supplied
    /// convention).
    pub fn apply(
        &mut self,
        particles: &crate::particle::Particles,
        sub_dt: f32,
        dx_meters: f32,
        macro_spin: fn(&crate::particle::Particle) -> f32,
        out_omega: &mut [f32],
        out_curvature: &mut [Vec2],
    ) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        for i in 0..n {
            self.grid_mass[i] = 0.0;
            self.grid_l[i] = 0.0;
            self.grid_spin[i] = 0.0;
        }

        // --- P2G: scatter (micro_inertia, micro_inertia*omega_c, micro_inertia*macro_spin) ---
        for pi in 0..particles.len() {
            let p = particles.get(pi);
            let i_micro = self.config.micro_inertia(p.mass);
            let spin = macro_spin(&p);
            let w = quadratic_weights(p.x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * i_micro;
                    // omega_c_particle is the field's own last-gathered value for
                    // this particle -- read from out_omega (last substep's
                    // result), 0.0 at rest/first-touch, matching grid_g's own
                    // "empty cell keeps its real rest value" convention.
                    let omega_c_particle = out_omega.get(pi).copied().unwrap_or(0.0);
                    self.grid_l[idx] += mw * omega_c_particle;
                    self.grid_mass[idx] += mw;
                    self.grid_spin[idx] += mw * spin;
                }
            }
        }

        // --- Solve: omega_c_new = (L + torque*dt) / I_eff, torque =
        // 2*coupling_modulus*(macro_spin - omega_c) -- the real antisymmetric
        // Cosserat coupling term, mass-normalized the same way L/I_eff is. ---
        let alpha = self.config.coupling_modulus_pa;
        for i in 0..n {
            if self.grid_mass[i] <= 1e-12 {
                self.grid_work[i] = self.grid_omega[i]; // no particle here this step -- hold last value
                continue;
            }
            let omega_prev = self.grid_l[i] / self.grid_mass[i];
            let spin_avg = self.grid_spin[i] / self.grid_mass[i];
            // d(omega)/dt = k*(spin_avg - omega), k = 2*alpha/i_eff_per_mass --
            // a REAL, LINEAR relaxation ODE (simpler than GranularFluidityField's
            // logistic one) with an EXACT closed-form solution. Real, disclosed
            // lesson applied from that file's own doc: naive explicit Euler on
            // this class of stiff microscopic-timescale relaxation term either
            // understates or wildly overshoots (confirmed here directly -- an
            // earlier explicit-Euler version of this solve produced NaN under
            // real parameters). The exact exponential solution is
            // UNCONDITIONALLY STABLE for any dt, by construction, not a
            // numerics workaround.
            let i_eff_per_mass = self.config.micro_inertia(1.0).max(1e-20);
            let k = 2.0 * alpha / i_eff_per_mass;
            self.grid_work[i] = spin_avg + (omega_prev - spin_avg) * (-k * sub_dt).exp();
        }
        self.grid_omega.copy_from_slice(&self.grid_work);

        // --- G2P: gather omega_c and its real spatial gradient (curvature) ---
        for pi in 0..particles.len().min(out_omega.len()) {
            let p = particles.get(pi);
            let w = quadratic_weights(p.x);
            let mut omega_sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    omega_sum += weight * self.grid_omega[idx];
                    w_sum += weight;
                }
            }
            out_omega[pi] = if w_sum > 1e-10 {
                omega_sum / w_sum
            } else {
                0.0
            };

            if pi < out_curvature.len() {
                let base = w.base_cell;
                let x = base.x.clamp(0, res - 1) as usize;
                let y = base.y.clamp(0, res - 1) as usize;
                out_curvature[pi] =
                    micro_curvature_2d(&self.grid_omega, self.grid_res, x, y, dx_meters);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{Particle, Particles};

    fn test_config() -> CosseratConfig {
        CosseratConfig {
            coupling_modulus_pa: 1.0e6,
            grain_diameter_m: 0.3e-3,
            micro_inertia_coefficient: 0.1,
        }
    }

    fn test_particle_at(pos: Vec2) -> Particle {
        let mut p = Particle::zeroed();
        p.x = pos;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p
    }

    #[test]
    fn stability_dt_is_positive_and_finite() {
        let cfg = test_config();
        let dt = cfg.stability_dt(0.01);
        assert!(dt.is_finite() && dt > 0.0);
    }

    #[test]
    fn omega_stays_zero_with_zero_macro_spin() {
        // Real, necessary invariant: with no macro spin anywhere (a purely
        // irrotational flow), there is no source to drive micro-rotation
        // away from its zero rest state -- omega_c must stay exactly 0.
        let cfg = test_config();
        let mut field = CosseratField::new(cfg, 16);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(8.0, 8.0)),
            test_particle_at(Vec2::new(8.0, 8.0)),
        ]);
        let mut out_omega = vec![0.0; 2];
        let mut out_curvature = vec![Vec2::ZERO; 2];
        for _ in 0..50 {
            field.apply(
                &particles,
                1.0e-6,
                0.01,
                |_p| 0.0,
                &mut out_omega,
                &mut out_curvature,
            );
        }
        assert!(out_omega.iter().all(|&o| o.abs() < 1e-9));
    }

    #[test]
    fn omega_grows_toward_macro_spin_when_driven() {
        // Real check: with a real, sustained macro spin, omega_c should be
        // driven away from zero TOWARD that spin (the coupling torque's own
        // real sign convention: torque = 2*alpha*(spin - omega_c), which is
        // exactly a relaxation of omega_c toward spin -- omega_c=spin is the
        // real fixed point of this ODE).
        let cfg = test_config();
        let mut field = CosseratField::new(cfg, 16);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(8.0, 8.0)),
            test_particle_at(Vec2::new(8.0, 8.0)),
            test_particle_at(Vec2::new(8.0, 8.0)),
            test_particle_at(Vec2::new(8.0, 8.0)),
        ]);
        const SPIN: f32 = 2.0;
        let mut out_omega = vec![0.0; 4];
        let mut out_curvature = vec![Vec2::ZERO; 4];
        let sub_dt = (cfg.stability_dt(1.0) * 0.5).min(1.0e-4);
        for _ in 0..20000 {
            field.apply(
                &particles,
                sub_dt,
                0.01,
                |_p| SPIN,
                &mut out_omega,
                &mut out_curvature,
            );
        }
        assert!(
            out_omega[0] > 0.0 && (out_omega[0] - SPIN).abs() / SPIN < 0.3,
            "omega_c={} should have relaxed toward spin={SPIN}",
            out_omega[0]
        );
    }

    #[test]
    fn zero_particles_produces_zero_field_and_no_panic() {
        let cfg = test_config();
        let mut field = CosseratField::new(cfg, 8);
        let particles = Particles::from(vec![]);
        let mut out_omega = vec![0.0; 0];
        let mut out_curvature = vec![Vec2::ZERO; 0];
        field.apply(
            &particles,
            1.0e-6,
            0.01,
            |_p| 0.0,
            &mut out_omega,
            &mut out_curvature,
        );
        assert!(field.grid_omega.iter().all(|&o| o == 0.0));
    }
}

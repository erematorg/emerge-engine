use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, Viscoelastic, scale_lame};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{ParticleUpdateCtx, Particles};

/// Kelvin-Voigt viscoelastic solid.
///
/// Stress = elastic (NeoHookean) + viscous (Newtonian dashpot in parallel).
///   τ = τ_elastic(F) + η · D_dev
/// where D = (C + Cᵀ)/2 is the symmetric strain-rate from the APIC velocity gradient.
///
/// This is the Kelvin-Voigt model: spring and dashpot in **parallel**.
/// The material deforms elastically AND dissipates energy simultaneously.
/// Creep under constant stress eventually stops (unlike Maxwell).
///
/// # Physical regime
/// - Biological soft tissue (tendon, cartilage, ligament): E=1–100 MPa, ν≈0.45, η=1–100 Pa·s
/// - Cell cytoskeleton: E=1–10 kPa, η=0.01–0.1 Pa·s
/// - Gelatin / hydrogel: E=0.1–10 kPa, η=0.01–10 Pa·s
/// - Rubber dampers: E=1–10 MPa, η=100–10000 Pa·s
///
/// # Why Kelvin-Voigt, not Maxwell?
/// Maxwell (spring + dashpot in series) has stress relaxation to zero — models fluids/polymers.
/// KV (parallel) has strain creep that stops — models biological solids correctly.
/// Biological tissues (Fung 1993) are better approximated by KV at the MPM particle scale.
/// Maxwell requires storing Fₑ per particle (new Mat2 field); KV uses only existing `velocity_gradient`.
///
/// # Reference
/// Kelvin-Voigt: Christensen 1982, "Theory of Viscoelasticity".
/// NeoHookean base: Bonet & Wood 2008, §6.4.
/// MPM viscoelastic: Stomakhin et al. 2014, §3 (foam); Fang et al. 2019 (MPM-DEM).
#[derive(Debug, Clone, Copy)]
pub struct ViscoelasticMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Kelvin-Voigt viscosity η (Pa·s in SI, sim-units²/s at emerge scale).
    /// Larger = more dissipation, stiffer rate response.
    pub viscosity: f32,
    /// Clamp J ∈ [j_min, 1/j_min] to prevent stress explosion on extreme compression.
    pub j_min: f32,
    /// Active stress coefficient for motile-cell / amoeba behaviour.
    /// τ_total = τ_elastic + τ_viscous + activation × coeff × I  (isotropic contractile pressure).
    /// 0.0 = passive (default). Tune to order of µ for visible deformation.
    pub active_stress_coeff: f32,
    /// Thermal softening: λ_eff = λ·(1 + thermal_expansion·T), same for µ.
    /// Negative = soften on heat (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
}

impl ViscoelasticMaterial {
    pub const fn new(lambda: f32, mu: f32, viscosity: f32) -> Self {
        Self {
            lambda,
            mu,
            viscosity,
            j_min: 0.01,
            active_stress_coeff: 0.0,
            thermal_expansion: 0.0,
        }
    }

    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32, viscosity: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, viscosity)
    }

    /// Nearly incompressible viscoelastic: ν=0.45. Soft tissue regime (tendon, cartilage).
    pub fn near_incompressible(young_modulus: f32, viscosity: f32) -> Self {
        Self::from_young_modulus(young_modulus, 0.45, viscosity)
    }

    /// Moderately compressible viscoelastic: ν=0.40. Cytoskeletal network regime.
    pub fn moderately_compressible(young_modulus: f32, viscosity: f32) -> Self {
        Self::from_young_modulus(young_modulus, 0.40, viscosity)
    }

    /// Dense rubber-like material (e.g. intervertebral disc): ν=0.45.
    pub fn rubber_damper(young_modulus: f32, viscosity: f32) -> Self {
        Self::from_young_modulus(young_modulus, 0.45, viscosity)
    }
}

impl FromSI<Viscoelastic> for ViscoelasticMaterial {
    fn from_physical(props: &Viscoelastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        // Real fix (2026-08-17), same class as `NewtonianFluidMaterial::
        // from_physical`'s own regression (see that method's own doc for
        // the full derivation/git-history): viscosity multiplies a
        // velocity-gradient-derived strain rate to produce a stress
        // (`tau_viscous = eta*D_dev`, the exact same dimensional shape as
        // the fluid case), so it stays raw SI Pa*s -- `scale_visc`'s
        // `dt^2/(rho*dx^2)`-family scaling does not apply here either. No
        // existing test exercised this specific constructor (every
        // Viscoelastic test uses `::new`/`::from_young_modulus`, grid-unit
        // paths), so this was a real, silent, zero-coverage gap, not a
        // verified-then-broken regression like the fluid case -- fixed
        // proactively while the correct convention was freshly re-derived
        // and confirmed.
        Self::new(lambda, mu, props.eta_pa_s)
    }
}

impl MaterialModel for ViscoelasticMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Viscoelastic
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = (f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x)
            .max(self.j_min)
            .min(1.0 / self.j_min);

        // Thermal modulus scaling.
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        let mu = self.mu * t_scale;
        let lambda = self.lambda * t_scale;

        // NeoHookean elastic: τ = µ·(F·Fᵀ − I) + λ·ln(J)·I
        let b = f * f.transpose();
        let elastic = mu * (b - Mat2::IDENTITY) + Mat2::from_diagonal(Vec2::splat(lambda * j.ln()));

        // Kelvin-Voigt viscous dashpot: τ_v = η · D_dev
        let c = particles.velocity_gradient[i];
        let sym = c + c.transpose();
        let d = sym * 0.5;
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));
        let viscous = self.viscosity * t_scale * d_dev;

        // Active stress is NOT added here — `activation_scale()` below reports
        // `active_stress_coeff` to the shared P2G path (`transfer::combined_kirchhoff_stress`),
        // which applies it isotropically for this model. Adding it here too would double-count.
        elastic + viscous
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * *ctx.velocity_gradient;
        *ctx.deformation_gradient = fp_new * *ctx.deformation_gradient;
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn pressure_scale(&self) -> f32 {
        1.0
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Viscoelastic as u32,
            lambda: self.lambda,
            mu: self.mu,
            dynamic_viscosity: self.viscosity,
            active_stress_coeff: self.active_stress_coeff,
            volume_ratio_min: self.j_min, // GPU uses this to clamp J in ln(J) stress term
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let elastic_dt = elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            1.0e-6,
            cell_width,
            material_cfl,
        );
        let viscous_dt = if self.viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic = self.viscosity / density;
            if kinematic > f32::EPSILON {
                viscous_cfl * cell_width * cell_width / kinematic
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        };
        elastic_dt.min(viscous_dt)
    }
}

#[cfg(test)]
mod analytical_validation_tests {
    use super::*;
    use crate::Particle;

    fn particle_with(f: Mat2, velocity_gradient: Mat2) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.velocity_gradient = velocity_gradient;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        Particles::from(vec![p])
    }

    /// **Small-strain elastic limit must recover exact linear elasticity.**
    /// This material's elastic term is `mu*(F*F^T-I) + lambda*ln(J)*I` -- the
    /// same "simple" NeoHookean form `NeoHookeanMaterial`'s doc describes as
    /// historical context (that material's actual CODE now uses a
    /// plane-strain vol-dev split, `k=lambda+mu` -- see its own doc). For
    /// F=I+delta*E (E symmetric), this linearizes to the STANDARD textbook
    /// form `2*mu*eps + lambda*tr(eps)*I` (same as `CorotatedMaterial`'s own
    /// verified small-strain limit, NOT the `k=lambda+mu` substitution
    /// NeoHookeanMaterial's split form needs).
    /// `ViscoelasticMaterial` had zero test comparing it to any analytical
    /// result before this.
    #[test]
    fn small_strain_elastic_term_matches_hookes_law_at_zero_strain_rate() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = ViscoelasticMaterial::new(lambda, mu, 50.0);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_diagonal(Vec2::new(1.0, -0.4));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with(f, Mat2::ZERO); // zero strain rate -- no viscous term
        let tau = mat.kirchhoff_stress(&particles, 0);

        let eps = delta * e;
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        let predicted = Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps;

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain viscoelastic elastic term should match linear elasticity: \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// **Kelvin-Voigt dashpot term must be EXACTLY `eta*D_dev`** -- this term is
    /// linear in the velocity gradient by construction (no approximation
    /// involved, unlike the elastic term's hyperelastic nonlinearity), so this
    /// should match to full floating-point precision, not just approximately.
    #[test]
    fn viscous_term_matches_eta_times_deviatoric_strain_rate_exactly() {
        let eta = 50.0;
        let mat = ViscoelasticMaterial::new(1000.0, 800.0, eta);

        // Pure shear strain rate C = [[0, g], [g, 0]] -- already symmetric and
        // traceless, so D=C and D_dev=D exactly (same derivation as Bingham's
        // own analytical test).
        let g = 3.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));
        // F=I (zero elastic contribution: mu*(I-I)+lambda*ln(1)*I = 0), isolating
        // the viscous term entirely.
        let particles = particle_with(Mat2::IDENTITY, c);
        let tau = mat.kirchhoff_stress(&particles, 0);

        let predicted = eta * c; // D_dev = C exactly for this pure-shear C
        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-4,
            "viscous term should match eta*D_dev exactly (F=I isolates it from \
             the elastic term): predicted={predicted:?} actual={tau:?}"
        );
    }

    /// Real, checkable claim: elastic and viscous contributions are additive
    /// (Kelvin-Voigt = spring + dashpot IN PARALLEL) -- stress at a state with
    /// both nonzero strain AND nonzero strain rate must equal the sum of each
    /// computed independently, not some coupled/nonlinear combination.
    #[test]
    fn elastic_and_viscous_contributions_are_additive() {
        let lambda = 1000.0;
        let mu = 800.0;
        let eta = 50.0;
        let mat = ViscoelasticMaterial::new(lambda, mu, eta);

        let delta = 0.02_f32; // real (not infinitesimal) strain -- exact formula, no linearization needed
        let e = Mat2::from_diagonal(Vec2::new(1.0, -0.3));
        let f = Mat2::IDENTITY + delta * e;
        let g = 4.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));

        let combined = particle_with(f, c);
        let elastic_only = particle_with(f, Mat2::ZERO);
        let viscous_only = particle_with(Mat2::IDENTITY, c);

        let tau_combined = mat.kirchhoff_stress(&combined, 0);
        let tau_elastic = mat.kirchhoff_stress(&elastic_only, 0);
        let tau_viscous = mat.kirchhoff_stress(&viscous_only, 0);

        let diff = tau_combined - (tau_elastic + tau_viscous);
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-3,
            "elastic and viscous contributions must be exactly additive (parallel \
             Kelvin-Voigt, not a coupled model): combined={tau_combined:?} \
             elastic+viscous={:?}",
            tau_elastic + tau_viscous
        );
    }

    /// **Real regression guard (2026-08-17)**: `from_physical` had zero test
    /// coverage at all before this -- every other test in this file uses
    /// `::new`/`::from_young_modulus` (grid-unit paths), which never
    /// exercised the SI-construction code path where the real bug lived
    /// (see `from_physical`'s own doc for the fix and its shared root cause
    /// with `NewtonianFluidMaterial::from_physical`'s regression). Real,
    /// exact check: SI viscosity must survive unconverted, matching this
    /// solver's own real time/length convention (time in real seconds,
    /// only length rescaled by dx) -- not scaled by any dt/dx/rho factor.
    #[test]
    fn from_physical_preserves_si_viscosity_unconverted() {
        use crate::materials::physical_props::{Elastic, Viscoelastic};
        use crate::solver::config::SimConfig;

        let props = Viscoelastic {
            elastic: Elastic {
                e_pa: 5.0e4,
                nu: 0.45,
                rho_kg_m3: 1100.0,
            },
            eta_pa_s: 10.0,
        };
        // Real, distinct dx/dt from any "1.0" default -- if a dt^2/(rho*dx^2)-
        // style scaling ever creeps back in, these values make it numerically
        // obvious (a 0.02s/0.01m pair scales viscosity by a large factor,
        // not silently ~1.0).
        let config = SimConfig {
            dx_meters: 0.01,
            dt_seconds: 0.02,
            ..SimConfig::default()
        };

        let mat = ViscoelasticMaterial::from_physical(&props, &config);

        assert_eq!(
            mat.viscosity, props.eta_pa_s,
            "from_physical must pass SI viscosity through raw/unconverted: \
             expected {} (the real input Pa*s), got {}",
            props.eta_pa_s, mat.viscosity
        );
    }
}

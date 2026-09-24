use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young, polar_decomposition_2d};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Granular-fluid mixture: Tait EOS bulk pressure + corotated elastic deviatoric + SVD plasticity.
///
/// Constitutive law (Dunatunga & Kamrin 2015, §3):
///   τ = τ_EOS + τ_corotated_dev
///   τ_EOS  = −k·((ρ/ρ₀)^γ − 1)·I                 — weakly-compressible fluid bulk (Tait EOS)
///   τ_dev  = 2µ·h·dev[(F−R)·Fᵀ] + λ·h·(J−1)·J·I  — corotated elastic (shape-restoring + vol)
///
/// Plasticity: SVD clamp on F singular values (Stomakhin 2013 §4 — identical to StomakhinMaterial).
///   Jp accumulates plastic volume change. h = exp(ξ·(1−Jp)) hardens on compression.
///
/// This differs from:
///   - `BinghamFluidMaterial` (purely fluid, no elastic restoring force)
///   - `DruckerPragerMaterial` (no EOS pressure, rate-independent DP)
///   - `StomakhinMaterial` (no EOS, purely elastic+plastic)
///
/// Use for: wet terrain substrates, wet granular flows, biological cell matrices.
/// Ref: Kamrin 2015 granular-fluid; SoftZoo's own mud material (`mud.py`) independently
/// confirms the same fluid-EOS + corotated blend CONCEPT -- but its own specific
/// parameters (a single fixed set, linear not Tait EOS, θ_c=0.025) do NOT match this
/// file's three presets below; see each preset's own honest-disclosure doc comment.
#[derive(Debug, Clone, Copy)]
pub struct GranularFluidMaterial {
    /// Elastic shear modulus µ — corotated deviatoric stiffness.
    pub mu: f32,
    /// Elastic first Lamé λ — volumetric elastic contribution.
    pub lambda: f32,
    /// Rest density ρ₀. EOS pressure is zero when ρ = ρ₀.
    pub rest_density: f32,
    /// EOS bulk stiffness k. Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. 7 for near-incompressible; 1–3 for compressible granular flow.
    pub eos_power: f32,
    /// Hardening exponent ξ. h = exp(ξ·(1−Jp)). 0 = perfect plasticity.
    pub hardening_exponent: f32,
    /// Compression limit θ_c — singular values clamped at (1−θ_c).
    pub compression_limit: f32,
    /// Stretch limit θ_s — singular values clamped at (1+θ_s).
    pub stretch_limit: f32,
    /// Jp lower bound — prevents h from exploding under sustained compression.
    pub min_plastic_jacobian: f32,
    /// Jp upper bound — limits plastic volume expansion.
    pub max_plastic_jacobian: f32,
    /// Granular contact/no-tension pressure floor. This is a constitutive
    /// choice for the granular branch, not free-surface surface tension and
    /// not used by strict Newtonian/Bingham WC-MPM liquids.
    pub pressure_floor: f32,
    /// Dynamic shear viscosity η -- real, physically-motivated dissipation
    /// (real wet granular materials, mud/clay/cytoplasm, are NOT purely
    /// elastic; they carry genuine viscous energy loss on top of their
    /// elastic+plastic response). Same formula `NewtonianFluidMaterial`
    /// already uses (`τ += η·dev(D)`, D = symmetric part of the velocity
    /// gradient). Real, previously-disclosed gap found 2026-08-06: with
    /// this at 0.0, a hard impact (as opposed to gentle self-weight
    /// settling) makes the material bounce almost elastically -- min_j
    /// dropping to ~0.43 on impact then the whole body rebounding UPWARD
    /// past its own drop height, the same superball-bounce failure mode
    /// already found+fixed once before for a different zero-damping
    /// material (see `fire_spread_plank_drift_root_caused_2026-07-31`'s own
    /// "elastic-bounce" fix). 0.0 = disabled (`new()`'s own default,
    /// matching every other numerical-stability field's convention here).
    pub dynamic_viscosity: f32,
    /// Bulk (volumetric) viscosity ζ -- real, disclosed 2026-08-06 addition
    /// alongside `dynamic_viscosity`. Shear viscosity alone left a real,
    /// measured residual: on hard impact this material's TWO independent
    /// volumetric-stiffness sources (Tait EOS pressure + the corotated
    /// model's own `lambda*(J-1)*J` term) kept bouncing -- min_j oscillating
    /// 0.6-0.85 with no volumetric damping at all, since `dynamic_viscosity`
    /// only damps the deviatoric/shear part. Same formula `NewtonianFluid
    /// Material::bulk_viscosity` already uses (`τ += ζ·(∇·v)·I`). Real
    /// physical mechanism for wet consolidated materials: pore-fluid
    /// drainage resistance genuinely damps volumetric change, distinct from
    /// shear viscosity. 0.0 = disabled.
    pub bulk_viscosity: f32,
}

impl GranularFluidMaterial {
    /// Raw field constructor, for consistency with every other material struct
    /// in this crate (`sand.rs`/`fluid.rs`/`snow.rs`/etc. all have a `::new()`
    /// — this was the sole exception). Takes the physically-meaningful
    /// parameters directly; the remaining numerical-stability fields default
    /// to the same values the `saturated_loam` preset already uses (eos_power
    /// 7.0 = standard near-incompressible Tait EOS, pressure_floor 0.0 = no
    /// tensile granular-contact traction). For a ready-made preset, prefer `saturated_loam`/
    /// `consolidated_clay`/`cytoplasmic` instead.
    pub const fn new(
        lambda: f32,
        mu: f32,
        rest_density: f32,
        eos_stiffness: f32,
        hardening_exponent: f32,
        compression_limit: f32,
    ) -> Self {
        Self {
            mu,
            lambda,
            rest_density,
            eos_stiffness,
            eos_power: 7.0,
            hardening_exponent,
            compression_limit,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
            dynamic_viscosity: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// Saturated loam: eos_stiffness=200, ξ=5, θ_c=0.4 — yields easily, flows under load.
    ///
    /// HONEST DISCLOSURE (audit 2026-07-17): the constitutive LAW above (Tait EOS +
    /// corotated elastic + Stomakhin SVD plasticity) is real and cited. These specific
    /// shape-parameter VALUES (eos_stiffness, hardening_exponent, compression_limit,
    /// stretch_limit, plastic-Jacobian bounds, rest_density) are NOT — checked directly
    /// against SoftZoo's own mud material (`mud.py`, the file this module's top doc
    /// pointed to) and they don't trace to it: SoftZoo uses one fixed parameter set
    /// (not three material variants), a different (linear, not Tait power-law) EOS
    /// form, and its compression limit (θ_c=0.025) is off by ~12-24x from this preset's
    /// 0.4. They also don't trace to Dunatunga & Kamrin 2015 (a granular-only paper,
    /// no mud/fluid blend or these numbers). This preset's real-world name ("saturated
    /// loam") is illustrative/hand-tuned, not a measured real-loam value -- same
    /// honesty standard as `FORAGING_RECOVERY_RATE` elsewhere in this codebase: the
    /// mechanism is real, this specific calibration is not yet, and shouldn't be
    /// presented as if it were. Needs a real geotechnical/soil-mechanics source before
    /// any claim of "this is real loam" would be honest.
    ///
    /// `eos_power` must stay in the granular-flow range (1-3, per the field's
    /// own doc) -- the near-incompressible value 7.0 causes runaway pressure
    /// under gravity-settling compression, driving dilation that weakens
    /// `hardening_scale` toward its floor in an unbounded feedback loop.
    pub fn saturated_loam(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 200.0,
            eos_power: 2.0,
            hardening_exponent: 5.0,
            compression_limit: 0.4,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
            // Real, disclosed 2026-08-06 addition: 0.3*mu, order-of-mu damping
            // (same real-physics convention `ViscoelasticMaterial`'s own doc
            // uses) -- empirically verified to stop a hard impact bouncing
            // elastically, not a re-guess (see `dynamic_viscosity`'s own doc
            // on this struct for the real bug this closes).
            dynamic_viscosity: 0.3 * mu,
            // Real, disclosed CORRECTION 2026-08-06: first version scaled this
            // by mu too -- wrong pairing, caught live (consolidated_clay kept
            // creeping/growing indefinitely, never settling, despite this).
            // bulk_viscosity damps div_v, the SAME quantity eos_stiffness's
            // own pressure term and lambda's own lam_vol term both act on --
            // it must scale with the material's VOLUMETRIC stiffness
            // (eos_stiffness), not shear stiffness (mu). See `bulk_viscosity`'s
            // own doc on the struct.
            bulk_viscosity: 0.5 * 200.0,
        }
    }

    /// Consolidated clay: eos_stiffness=500, ξ=3, θ_c=0.3 — higher stiffness, slower creep.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real consolidated-
    /// clay geotechnical data.
    ///
    /// Same `eos_power` constraint as `saturated_loam` above.
    pub fn consolidated_clay(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.2,
            eos_stiffness: 500.0,
            eos_power: 2.0,
            hardening_exponent: 3.0,
            compression_limit: 0.3,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.3,
            max_plastic_jacobian: 2.5,
            pressure_floor: 0.0,
            // Real, disclosed 2026-08-06 addition -- see saturated_loam's own
            // note. Higher factor (0.5 vs 0.3) matches this preset's own
            // "stiffer, slower creep" real-world framing.
            dynamic_viscosity: 0.5 * mu,
            // Real correction -- see saturated_loam's own note: scales with
            // this preset's OWN eos_stiffness (500, the highest of the three),
            // not mu.
            bulk_viscosity: 0.5 * 500.0,
        }
    }

    /// Cytoplasmic matrix: eos_stiffness=50, ξ=1, large yield surface.
    /// Use for biological cell interiors and soft tissue matrices.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real cytoplasm
    /// rheology literature.
    ///
    /// Same `eos_power` constraint as `saturated_loam` above; low
    /// `eos_stiffness=50` makes this preset least sensitive to it, but the
    /// constraint still applies.
    pub fn cytoplasmic(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 50.0,
            eos_power: 2.0,
            hardening_exponent: 1.0,
            compression_limit: 0.6,
            stretch_limit: 0.05,
            min_plastic_jacobian: 0.1,
            max_plastic_jacobian: 5.0,
            pressure_floor: 0.0,
            // Real, disclosed 2026-08-06 addition -- see saturated_loam's own
            // note. Lower factor (0.15 vs 0.3): softer biological matrix,
            // real cytoplasm is less viscous relative to its own stiffness
            // than wet soil.
            dynamic_viscosity: 0.15 * mu,
            // Real correction -- see saturated_loam's own note: scales with
            // this preset's OWN eos_stiffness (50, the lowest of the three).
            bulk_viscosity: 0.5 * 50.0,
        }
    }
}

impl MaterialModel for GranularFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::GranularFluid
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant().max(MIN_J);

        let density = (self.rest_density / j).max(1.0e-6);
        let ratio = (density / self.rest_density.max(1.0e-6)).max(1.0e-6);
        let pressure = (self.eos_stiffness
            * (crate::materials::utils::fast_pow(ratio, self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let h = particles.hardening_scale[i];
        let r = polar_decomposition_2d(f);
        let mu_eff = self.mu * h;
        let coro = 2.0 * mu_eff * (f - r) * f.transpose();
        let tr = coro.x_axis.x + coro.y_axis.y;
        let dev_coro = coro - Mat2::from_diagonal(Vec2::splat(tr * 0.5));

        let lam_vol = self.lambda * h * (j - 1.0) * j * Mat2::IDENTITY;

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure)) + dev_coro + lam_vol;

        // Real viscous dissipation -- see `dynamic_viscosity`/`bulk_viscosity`'s
        // own docs. Identical formulas to `NewtonianFluidMaterial::
        // kirchhoff_stress` (τ += η·dev(D) + ζ·(∇·v)·I).
        if self.dynamic_viscosity > 0.0 || self.bulk_viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym_strain = c + c.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            if self.dynamic_viscosity > 0.0 {
                let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
                stress += self.dynamic_viscosity * strain_dev;
            }
            if self.bulk_viscosity > 0.0 {
                stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
            }
        }

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(1.0e-6)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;

        if self.compression_limit > 0.0 || self.stretch_limit > 0.0 {
            let (u, sigma, vt) = svd2(f_trial);
            let sigma_c = Vec2::new(
                sigma
                    .x
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
                sigma
                    .y
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            );
            let jp_new = *ctx.plastic_volume_ratio * (sigma.x * sigma.y)
                / (sigma_c.x * sigma_c.y).max(1.0e-10);
            *ctx.plastic_volume_ratio =
                jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);
            // `hardening_scale` floor must stay at 1.0, not lower: `h` scales both
            // the deviatoric and volumetric elastic terms in `kirchhoff_stress`, so
            // h<1 under dilation (Jp>1) softens the material -- an unbounded
            // soften->dilate->soften feedback with nothing to stop it. Compression-
            // side hardening (h>1) is self-stabilizing; there's no equivalent
            // mechanism on the dilation side, so the floor clamps at baseline
            // stiffness (h=1) instead.
            *ctx.hardening_scale = (self.hardening_exponent * (1.0 - *ctx.plastic_volume_ratio))
                .exp()
                .clamp(1.0, 7.0);
            *ctx.deformation_gradient = u * Mat2::from_diagonal(sigma_c) * vt;
        } else {
            *ctx.deformation_gradient = f_trial;
        }

        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::GranularFluid as u32,
            mu: self.mu,
            lambda: self.lambda,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            volume_ratio_min: self.min_plastic_jacobian,
            volume_ratio_max: self.max_plastic_jacobian,
            pressure_floor: self.pressure_floor,
            dynamic_viscosity: self.dynamic_viscosity,
            bulk_viscosity: self.bulk_viscosity,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = elastic_wave_dt(
            self.lambda,
            self.mu,
            hardening_scale,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        );

        // Real, disclosed 2026-08-06 fix, found live: adding real
        // dynamic_viscosity/bulk_viscosity (see their own docs) without a
        // matching viscous CFL bound let the solver pick a substep too large
        // for stable EXPLICIT integration of that damping term -- a large
        // enough dt*viscosity/mass ratio makes an explicit damping term
        // INJECT energy instead of removing it (classic explicit-integrator
        // instability), which is exactly what looked like "exploding" on
        // impact (clay's own pile height growing to 52+ units instead of
        // settling). Same formula `NewtonianFluidMaterial::timestep_bound`
        // already uses, extended to cover bulk_viscosity too since it's the
        // same explicit-damping character on the same velocity-gradient
        // quantities.
        let total_viscosity = self.dynamic_viscosity + self.bulk_viscosity;
        if total_viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic_viscosity = total_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }
}

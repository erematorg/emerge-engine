use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, advance_deformation_gradient, carried_volume_ratio, elastic_wave_dt, lame_from_young,
    polar_decomposition_2d,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Granular-fluid mixture: Tait EOS bulk pressure + corotated elastic deviatoric + SVD plasticity.
///
/// Constitutive law (Dunatunga & Kamrin 2015, §3):
///   τ = τ_EOS + τ_corotated_dev
///   τ_EOS  = −k·((ρ/ρ₀)^γ − 1)·I                 -- weakly-compressible fluid bulk (Tait EOS)
///   τ_dev  = 2µ·h·dev[(F−R)·Fᵀ] + λ·h·(J−1)·J·I  -- corotated elastic (shape-restoring + vol)
///
/// Plasticity: SVD clamp on F singular values (Stomakhin 2013 §4 -- identical to StomakhinMaterial).
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
    /// Elastic shear modulus µ -- corotated deviatoric stiffness.
    pub mu: f32,
    /// Elastic first Lamé λ -- volumetric elastic contribution.
    pub lambda: f32,
    /// Rest density ρ₀. EOS pressure is zero when ρ = ρ₀.
    pub rest_density: f32,
    /// EOS bulk stiffness k. Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. 7 for near-incompressible; 1–3 for compressible granular flow.
    pub eos_power: f32,
    /// Hardening exponent ξ. h = exp(ξ·(1−Jp)). 0 = perfect plasticity.
    pub hardening_exponent: f32,
    /// Compression limit θ_c -- singular values clamped at (1−θ_c).
    pub compression_limit: f32,
    /// Stretch limit θ_s -- singular values clamped at (1+θ_s).
    pub stretch_limit: f32,
    /// Jp lower bound -- prevents h from exploding under sustained compression.
    pub min_plastic_jacobian: f32,
    /// Jp upper bound -- limits plastic volume expansion.
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
    /// -- this was the sole exception). Takes the physically-meaningful
    /// parameters directly; the remaining numerical-stability fields default
    /// to the same values EVERY real preset below (`saturated_loam`/
    /// `consolidated_clay`/`cytoplasmic`) actually uses.
    ///
    /// Real bug fixed 2026-09-15 (found via audit, not live-reported): this
    /// used to default `eos_power` to 7.0 while its OWN doc comment claimed
    /// that matched `saturated_loam` -- it doesn't; that preset (and both
    /// others) uses 2.0. Worse, `saturated_loam`'s own doc, right below,
    /// explicitly states 7.0 "causes runaway pressure under gravity-settling
    /// compression... an unbounded feedback loop" for this material's real
    /// granular-flow regime -- so the general-purpose constructor was
    /// shipping a default this very file calls unsuitable. 2.0 is not a
    /// guess: it's the one value every real preset already independently
    /// converged on, and the one the field's own doc says granular flow
    /// requires (1-3 range). `pressure_floor: 0.0` = no tensile granular-
    /// contact traction (this part was already correct).
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
            eos_power: 2.0,
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

    /// Saturated loam: eos_stiffness=200, ξ=5, θ_c=0.4 -- yields easily, flows under load.
    ///
    /// HONEST DISCLOSURE (audit 2026-07-17): the constitutive LAW above (Tait EOS +
    /// corotated elastic + Stomakhin SVD plasticity) is real and cited. These specific
    /// shape-parameter VALUES (eos_stiffness, hardening_exponent, compression_limit,
    /// stretch_limit, plastic-Jacobian bounds, rest_density) are NOT -- checked directly
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

    /// Consolidated clay: eos_stiffness=500, ξ=3, θ_c=0.3 -- higher stiffness, slower creep.
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

    // Real bug fixed 2026-09-15 (found via audit, not live-reported): unlike
    // every sibling EOS-pressure material in this crate
    // (`NewtonianFluidMaterial`, `BinghamFluidMaterial`, `IdealGasMaterial`,
    // `CavitatingFluidMaterial`, `BoilingMixtureMaterial` -- see each one's
    // own `init_particle`), this never seeded `initial_volume`/`volume`/
    // `density` from the real conserved quantity `mass/rest_density`, and
    // never overrode `owns_deformation_volume_state()` (stays at the trait
    // default `false`). Combined, this left density/volume entirely to
    // `SpawnRegion`'s kernel-mass-density gather at spawn -- the exact
    // free-surface-biased measurement `NewtonianFluidMaterial::init_particle`
    // exists specifically to avoid (see that function's own doc) -- and, on
    // the GPU backend, `g2p.wgsl` OVERWRITES density/volume from that same
    // biased gather every substep for any material with
    // `owns_deformation_volume_state==0`, not just at spawn. `update_particle`
    // below already correctly derives `volume = initial_volume * j` and
    // `density = mass / volume` every substep -- it only ever needed a
    // correct `initial_volume` to start from.
    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.initial_volume = particle.mass / self.rest_density.max(1.0e-6);
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    // TRIED, REVERTED (2026-08-26/27): a `GasMaterial`-style
    // `init_particle_from_transition` override (preserve real J relative to
    // `rest_density` instead of the engine's default identity reset -- see
    // `gas.rs`'s own override for that real, working pattern on a DIFFERENT
    // material) was attempted here for the exact same class of bug
    // (`Simulation::apply_phase_transition` resets `deformation_gradient` to
    // IDENTITY unconditionally, so `kirchhoff_stress`'s EOS pressure, which
    // reads `det(deformation_gradient)` directly rather than the stored
    // `Particle::density` field, drops to exactly zero regardless of real
    // load -- confirmed live and reproduced in a controlled diagnostic,
    // `diag_phase_transition_under_load_causes_stress_discontinuity` in
    // `tests/physics_correctness.rs`: 0.023 -> 0.188 max-speed spike in one
    // substep, ~18000x the matched no-transition control).
    //
    // Live-measured result of the fix attempt: WORSE, not better -- same
    // diagnostic went from a 0.16 speed delta to 2.93 (confirmed twice,
    // including after finding and fixing a real bug in the diagnostic's own
    // test setup, which turned out not to be the actual explanation). The
    // isotropic F this override installed does correctly zero out the
    // corotated deviatoric term (an isotropic matrix has zero deviatoric
    // part by construction) and lands J very close to the real prior
    // compression ratio (measured J~1.02, near the material's own
    // stretch_limit clamp) -- so the mechanism is doing roughly what
    // `gas.rs`'s own working version does.
    //
    // RESOLVED, mostly (2026-08-28, see `tests/physics_correctness.rs`'s own
    // doc on `diag_phase_transition_under_load_causes_stress_discontinuity`):
    // the `eos_power` candidate below WAS checked -- the diagnostic's own
    // material had been built via the raw `::new()` constructor, which
    // hardcoded 7.0 (the value flagged above as unsuitable), while the real
    // scene (`sand_water_saturation.rs`'s `make_mixture`) already used 2.0
    // directly and was never actually exposed to the danger. Rebuilding the
    // diagnostic to match `make_mixture` field-for-field dropped the
    // measured spike from +0.1646 to -0.0076 -- the identity-reset
    // mechanism, with the CORRECT eos_power, is not the catastrophic problem
    // it looked like. `new()`'s own default was fixed separately (2026-09-15,
    // see that constructor's own doc) so future callers can't reintroduce
    // this by using the raw constructor instead of a preset. Honest residual
    // scope, per that test's own doc: this explains why the diagnostic
    // OVERSTATED the danger, not necessarily the real scene's own eventual
    // 104,737-frame crash, which may have a slower, separate cause (a real
    // water-side CFL/retry instability per that crash's own panic message)
    // -- a repeated-transition version of the diagnostic remains the real
    // next step there, not more single-transition analysis.
    //
    // Original "not yet checked" note, kept for the historical record: this
    // material's `eos_power` default (7.0 in the raw `new()` constructor, already
    // documented elsewhere in this file as "causes runaway pressure under
    // gravity-settling compression" and NOT the value `sand_water_
    // saturation.rs`'s own `make_mixture` actually uses, 2.0 -- the
    // diagnostic test itself used the raw `new()` default and never
    // re-checked with eos_power=2.0), and/or a real mismatch between
    // DruckerPragerMaterial's own compressive support stress at a loaded
    // particle and what GranularFluidMaterial's Tait EOS can supply at a
    // J this close to 1 by construction (near-incompressible EOS forms are
    // deliberately flat near J=1 -- see this file's own honest-disclosure
    // doc on `saturated_loam` for why eos_power=7.0 is flagged unsuitable
    // for a granular-settling regime in the first place). Reverted rather
    // than shipping a confirmed regression; the diagnostic test stays as a
    // real regression guard for whoever picks this up next.
    //
    // fn init_particle_from_transition(&self, particle: &mut Particle) { ... }

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
        // This increment advances the single F read by both the EOS volume
        // response and the corotated/SVD branch. Exact constant-C integration
        // prevents Euler volume drift from becoming false EOS pressure or
        // permanent Jp/hardening.
        let (f_trial, _) = advance_deformation_gradient(
            *ctx.deformation_gradient,
            dt * *ctx.velocity_gradient,
            carried_volume_ratio(*ctx.volume, ctx.initial_volume),
        );

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
            // Real bug fixed 2026-09-15, second half of the fix on
            // `init_particle`'s own doc: `MaterialParams::
            // owns_deformation_volume_state` is NOT auto-derived from the
            // trait method -- each material's `params()` must forward it
            // explicitly (see that field's own doc), and this one never did,
            // so `..Default::default()` below silently left it at 0/false on
            // GPU even once the trait method itself returned `true`.
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
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

#[cfg(test)]
mod kinematic_projection_tests {
    use super::*;

    fn particle() -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.plastic_volume_ratio = 1.0;
        p.hardening_scale = 1.0;
        Particles::from(vec![p])
    }

    fn run_rate_step(mat: &GranularFluidMaterial, particles: &mut Particles, rate: Mat2, dt: f32) {
        let mut ctx = particles.update_ctx(0);
        *ctx.velocity_gradient = rate;
        mat.update_particle(&mut ctx, dt);
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

    #[test]
    fn rigid_rotation_preserves_eos_volume_and_plastic_history() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let mut particles = particle();
        let omega = 2.1;
        let dt = 0.2;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));

        run_rate_step(&mat, &mut particles, spin, dt);

        let expected = Mat2::from_angle(omega * dt);
        assert!(matrix_error(particles.deformation_gradient[0], expected) < 3.0e-6);
        assert!((particles.deformation_gradient[0].determinant() - 1.0).abs() < 2.0e-6);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
    }

    #[test]
    fn opposite_subclamp_rates_are_reversible_without_false_hardening() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let mut particles = particle();
        let rate = Mat2::from_diagonal(Vec2::new(-0.01, 0.005));
        let dt = 0.1;

        run_rate_step(&mat, &mut particles, rate, dt);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
        run_rate_step(&mat, &mut particles, -rate, dt);

        assert!(matrix_error(particles.deformation_gradient[0], Mat2::IDENTITY) < 3.0e-6);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
    }

    #[test]
    fn exponential_trial_respects_both_sides_of_compression_clamp() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let floor = 1.0 - mat.compression_limit;
        let dt = 0.1;

        let inside_sigma = floor + 1.0e-4;
        let mut inside = particle();
        run_rate_step(
            &mat,
            &mut inside,
            Mat2::from_diagonal(Vec2::new(inside_sigma.ln() / dt, 0.0)),
            dt,
        );
        assert!((inside.deformation_gradient[0].x_axis.x - inside_sigma).abs() < 2.0e-6);
        assert_eq!(inside.plastic_volume_ratio[0], 1.0);

        let outside_sigma = floor - 1.0e-4;
        let mut outside = particle();
        run_rate_step(
            &mat,
            &mut outside,
            Mat2::from_diagonal(Vec2::new(outside_sigma.ln() / dt, 0.0)),
            dt,
        );
        assert!((outside.deformation_gradient[0].x_axis.x - floor).abs() < 2.0e-6);
        let expected_jp = outside_sigma / floor;
        assert!((outside.plastic_volume_ratio[0] - expected_jp).abs() < 2.0e-6);
        assert!(outside.hardening_scale[0] > 1.0);
    }
}

/// Real regression tests, 2026-09-15, for the spawn-state/GPU-density bug
/// found by audit (see `init_particle`'s own doc for the full account): a
/// freshly-spawned particle must get its real, conserved volume/density
/// from `mass/rest_density`, matching every sibling EOS-pressure material,
/// and the material must correctly declare that it owns that state (both
/// the trait method AND its own separate forwarding into `MaterialParams`
/// for the GPU path -- these are NOT automatically linked, a real, second
/// bug this test also catches).
#[cfg(test)]
mod spawn_state_tests {
    use super::*;

    #[test]
    fn init_particle_seeds_real_conserved_volume_and_density() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 2.0;
        mat.init_particle(&mut p);

        let expected_volume = p.mass / mat.rest_density; // = 0.5
        assert!(
            (p.initial_volume - expected_volume).abs() < 1.0e-6,
            "initial_volume must be the real conserved mass/rest_density, \
             got {} expected {expected_volume}",
            p.initial_volume
        );
        assert!(
            (p.volume - expected_volume).abs() < 1.0e-6,
            "volume at spawn (J=1) must equal initial_volume exactly, got {} expected {expected_volume}",
            p.volume
        );
        assert!(
            (p.density - mat.rest_density).abs() < 1.0e-6,
            "density at spawn (J=1) must equal rest_density exactly, got {} expected {}",
            p.density,
            mat.rest_density
        );
    }

    #[test]
    fn init_particle_respects_real_prior_compression() {
        // A particle spawned with F already compressed (J=0.8, e.g. seeded
        // mid-scene by some other mechanism) must get a volume/density
        // consistent with THAT real J, not silently reset to J=1 -- matches
        // `NewtonianFluidMaterial::init_particle`'s own real contract.
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        let mut p = Particle::zeroed();
        let s = 0.8_f32.sqrt();
        p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(s));
        p.mass = 2.0;
        mat.init_particle(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 0.8).abs() < 1.0e-5,
            "test setup sanity: J should be 0.8, got {j}"
        );
        let expected_initial_volume = p.mass / mat.rest_density;
        assert!((p.initial_volume - expected_initial_volume).abs() < 1.0e-6);
        assert!((p.volume - expected_initial_volume * j).abs() < 1.0e-6);
        assert!((p.density - mat.rest_density / j).abs() < 1.0e-6);
    }

    #[test]
    fn owns_deformation_volume_state_is_true_and_reaches_gpu_params() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        assert!(
            MaterialModel::owns_deformation_volume_state(&mat),
            "GranularFluidMaterial must own its volume/density state -- it \
             derives both from its own EOS+deformation-gradient, not a \
             kernel-mass gather"
        );
        // Real, separate check: the trait method alone does NOT reach the
        // GPU -- `params()` must forward it explicitly (see that function's
        // own doc). This would have stayed silently 0/false even with the
        // trait method fixed, if only the trait override existed.
        assert_eq!(
            mat.params().owns_deformation_volume_state,
            1,
            "MaterialParams::owns_deformation_volume_state must be forwarded \
             from the trait method, not left at Default::default()'s 0"
        );
    }
}

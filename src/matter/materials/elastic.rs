use glam::Mat2;

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// Compressible Neo-Hookean hyperelastic solid (jelly, soft tissue).
///
/// STALE DOC FIXED 2026-07-07: this comment described an older, simpler form
/// (`τ = µ(FFᵀ − I) + λ·ln(J)·I`, still what `ViscoelasticMaterial` actually
/// implements for its own elastic term) that this material's CODE no longer
/// matches -- the actual `kirchhoff_stress` below uses a Simo-Pister
/// volumetric-deviatoric split (`k=λ+µ`, the 2D plane-strain bulk modulus,
/// fixed 2026-07-06 for dimensional correctness) instead. See the real,
/// current formula documented directly in `kirchhoff_stress`'s own body below.
/// Free energy: Ψ = µ/2·(tr(FᵀF)−d) − µ·ln(J) + λ/2·ln(J)²
/// Reference: standard hyperelasticity; used in Stomakhin et al. 2013 (snow paper) §2.
#[derive(Debug, Clone, Copy)]
pub struct NeoHookeanMaterial {
    pub lambda: f32,
    pub mu: f32,
    pub min_density: f32,
    /// Thermal modulus scale: λ_eff = λ·(1 + thermal_expansion·T), same for µ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour.
    /// τ_total = τ_elastic + activation × coeff × I  (contractile: pulls inward like a muscle).
    /// Independent of elastic state — generates force even at rest.
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
    /// Continuum damage softening rate — real mechanical consequence of accumulated
    /// structural damage (`Particle::friction_hardening`, e.g. from
    /// `rankine_damage_estimate`), not just a passive health readout. Effective
    /// stiffness: µ_eff = µ·exp(−rate·damage), λ_eff = λ·exp(−rate·damage) — the
    /// same exponential softening `RankineMaterial` uses for its own tensile
    /// strength (continuum damage mechanics), applied here to elastic stiffness
    /// instead. Damaged tissue gets progressively softer/weaker as a smooth,
    /// continuous function of real accumulated strain — not a hard on/off failure
    /// threshold. 0.0 = no damage coupling (default, unchanged behavior).
    pub damage_softening_rate: f32,
    /// EXPERIMENTAL, not yet FD-verified for the differentiable trainer (see
    /// `kirchhoff_stress_vjp` doc): Kelvin-Voigt internal viscosity, same
    /// `η·dev(D)` term `ViscoelasticMaterial` already implements (D = the
    /// symmetric part of the APIC velocity gradient). 0.0 = passive elastic
    /// only (default, unchanged behavior for every existing user).
    pub viscosity: f32,
}

impl NeoHookeanMaterial {
    pub fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            min_density: 1.0e-6,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
            damage_softening_rate: 0.0,
            viscosity: 0.0,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Canonical values: E = 5e6, ν = 0.2 (wgsparkl elasticity2 — stiff soft solid).
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Analytic adjoint (vector-Jacobian product) of `kirchhoff_stress` w.r.t.
    /// the deformation gradient F -- the first real building block toward
    /// differentiable stepping (offline gradient-based controller training,
    /// same real technique ChainQueen/DiffTaichi/SoftZoo use, applied here as
    /// a from-scratch hand derivation rather than a compiler-generated one --
    /// see `project_domain_taxonomy`/locomotion research notes for why no
    /// Rust autodiff crate fits this problem shape).
    ///
    /// Given `d_loss_d_tau` = ∂L/∂τ (the gradient flowing backward from
    /// wherever this particle's stress feeds into a scalar loss), returns
    /// ∂L/∂F.
    ///
    /// Derivation: τ(F) = (µ/J)·dev(B) + k·ln(J)·I, where B = F·Fᵀ,
    /// J = det(F), dev(B) = B − (tr(B)/2)·I (matching `kirchhoff_stress`
    /// exactly -- updated 2026-07-11 alongside the forward formula's
    /// volumetric-term fix, see that function's doc for why). Reverse-mode
    /// chain rule through B → A=dev(B) → τ, and separately through J (using
    /// the standard cofactor identity ∂J/∂F = J·F⁻ᵀ, so ∂ln(J)/∂F = F⁻ᵀ),
    /// gives:
    ///
    ///   B̄ = (µ/J)·dev(Ḡ)
    ///   ∂L/∂F = (B̄ + B̄ᵀ)·F + [k·tr(Ḡ) − (µ/J)·(Ḡ:A)] · F⁻ᵀ
    ///
    /// where Ḡ = ∂L/∂τ, A = dev(B), and Ḡ:A is the Frobenius inner product
    /// (sum of elementwise products). The `B̄ + B̄ᵀ` (NOT `2·B̄`) matters: B̄
    /// is only symmetric when Ḡ itself is, which isn't guaranteed just
    /// because B and A are -- a real derivation bug first-draft code hit
    /// here, caught by the finite-difference tests below, not by inspection.
    /// Verified against central-difference numerical gradients in this
    /// module's own tests -- hand-derived tensor calculus is exactly where
    /// sign/transpose/symmetry-assumption errors hide, so this is not
    /// trusted on derivation alone.
    ///
    /// Covers only the core elastic term (thermal/damage scaling folded into
    /// µ/λ as constants here, matching how `kirchhoff_stress` already treats
    /// them per-call; the active-stress term is additive and its own
    /// gradient is trivial, not yet wired in). Does NOT cover `viscosity`'s
    /// contribution: that term depends on `velocity_gradient` (the APIC C
    /// matrix), not F, so it's simply absent from ∂L/∂F -- correct as long as
    /// `viscosity` stays 0.0 (its default) for any differentiable use. This
    /// is checked with a REAL (non-debug) assert: training runs are exactly
    /// where `--release` gets used, so a `debug_assert` here would silently
    /// compile away and hand back a wrong gradient with no protection at all.
    pub fn kirchhoff_stress_vjp(
        &self,
        particles: &Particles,
        i: usize,
        d_loss_d_tau: Mat2,
    ) -> Mat2 {
        assert_eq!(
            self.viscosity, 0.0,
            "kirchhoff_stress_vjp does not differentiate through the viscosity term; \
             only viscosity=0.0 materials are safe to use with the differentiable trainer"
        );
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        let damage_scale = (-self.damage_softening_rate * particles.friction_hardening[i]).exp();
        let mu = self.mu * t_scale * damage_scale;
        let lambda = self.lambda * t_scale * damage_scale;
        let k = lambda + mu;

        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(glam::Vec2::splat(tr_b * 0.5));

        let g = d_loss_d_tau;
        let tr_g = g.x_axis.x + g.y_axis.y;
        let dev_g = g - Mat2::from_diagonal(glam::Vec2::splat(tr_g * 0.5));
        // Frobenius inner product G:A = sum of elementwise products.
        let g_dot_a = g.x_axis.x * dev_b.x_axis.x
            + g.x_axis.y * dev_b.x_axis.y
            + g.y_axis.x * dev_b.y_axis.x
            + g.y_axis.y * dev_b.y_axis.y;

        let f_inv_t = f.inverse().transpose();

        // B = F·Fᵀ's VJP is (B̄ + B̄ᵀ)·F for a general (not necessarily
        // symmetric) incoming adjoint B̄ = (µ/J)·dev(Ḡ). B itself is always
        // symmetric, but the GRADIENT flowing into it isn't -- Ḡ = ∂L/∂τ has
        // no reason to be symmetric in general (e.g. it won't be once this
        // feeds into an asymmetric downstream operation like a P2G weight).
        // The simplification (B̄+B̄ᵀ)·F = 2·B̄·F only holds when B̄ itself is
        // symmetric, which is NOT guaranteed just because B and A=dev(B) are.
        let b_bar = (mu / j) * dev_g;
        let term1 = (b_bar + b_bar.transpose()) * f;
        let scalar2 = k * tr_g - (mu / j) * g_dot_a;
        let term2 = scalar2 * f_inv_t;

        term1 + term2
    }
}

impl FromSI<Elastic> for NeoHookeanMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for NeoHookeanMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::NeoHookean
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }

        // Thermal modulus scaling: λ_eff = λ·(1 + α·T), same for µ.
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        // Damage softening: µ_eff = µ·exp(−rate·damage), same exponential form
        // RankineMaterial uses for tensile strength -- continuum damage mechanics,
        // not a hand-picked curve. rate=0.0 (default) leaves this at 1.0, no-op.
        let damage_scale = (-self.damage_softening_rate * particles.friction_hardening[i]).exp();
        let mu = self.mu * t_scale * damage_scale;
        let lambda = self.lambda * t_scale * damage_scale;

        // Simo-Pister volumetric-deviatoric split, adapted to plane strain (this
        // solver is 2D-only for now; a 3D bulk term would be revisited if/when
        // that changes -- see project notes).
        // B = F·Fᵀ (left Cauchy-Green), d = 2 in 2D.
        // Deviatoric Kirchhoff: µ · J^{-2/d} · dev(B)  with d=2 → µ/J · dev(B)
        //   dev(B) = B − (tr(B)/2)·I  (2D traceless part)
        // Volumetric Kirchhoff: k · ln(J) · I  (from U(J) = k/2·(ln J)², the
        //   actual Simo & Pister 1984 log-barrier volumetric potential -- NOT
        //   k/2·(J²−1), a bounded polynomial this code used until 2026-07-11.
        //   k = λ + µ  (2D PLANE-STRAIN bulk modulus -- NOT the 3D relation
        //   k=λ+2µ/3, which an earlier version of this code used to match
        //   `sparkl`, a 3D reference engine. Real derivation: linearizing
        //   k·(J−1) against small-strain plane-strain pressure gives k=λ+µ;
        //   the 3D relation is off by µ/3, a real (1−2ν)/3 fractional error
        //   in bulk stiffness -- negligible near ν=0.5 (soft-tissue presets)
        //   but ~20% at ν≈0.2 (compressible/granular-like presets). Fixed
        //   2026-07-06 in favor of dimensional correctness over reference-
        //   engine parity.)
        //
        // REAL BUG FIXED 2026-07-11: `k/2·(J²−1)` is bounded as J→0 (its
        // Kirchhoff contribution approaches a finite `-k/2`, never more), so
        // it supplies only a FINITE ceiling on how hard the material resists
        // further compression, no matter how large k is scaled. A sustained
        // driven load (a creature's own muscle activation, cyclically
        // compressing tissue every gait cycle with nothing to fully release
        // it) can always eventually overpower a finite ceiling given enough
        // cycles -- exactly what a real long-horizon `basic_creature`
        // diagnostic found: net crawl drift collapsed to ~0 while min(J) kept
        // falling and NEVER recovered, and neither raising material stiffness
        // nor adding numerical (APIC) damping fixed it -- both only delayed
        // the same eventual collapse, because neither changes the bounded
        // ceiling itself. The log form `k·ln(J)` has NO such ceiling: as J→0,
        // ln(J)→−∞, so the restoring Kirchhoff stress diverges too -- a
        // genuine physical barrier against total compression, the actual
        // reason Simo & Pister's own 1984 formulation uses `(ln J)²` rather
        // than a bounded polynomial in J. This was a citation/implementation
        // mismatch as much as a stability bug: the doc already cited Simo &
        // Pister for this term while implementing a different, weaker one.
        // Reference: Simo & Pister 1984; Bonet & Wood §6.4 (2D plane-strain form).
        let b = f * f.transpose();
        let tr_b = b.x_axis.x + b.y_axis.y;
        let dev_b = b - Mat2::from_diagonal(glam::Vec2::splat(tr_b * 0.5));
        let k = lambda + mu;

        let dev_stress = (mu / j) * dev_b;
        let vol_stress = (k * j.ln()) * Mat2::IDENTITY;

        let viscous_stress = if self.viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym = c + c.transpose();
            let d = sym * 0.5;
            let d_trace = d.x_axis.x + d.y_axis.y;
            let d_dev = d - Mat2::from_diagonal(glam::Vec2::splat(d_trace * 0.5));
            self.viscosity * d_dev
        } else {
            Mat2::ZERO
        };

        dev_stress + vol_stress + viscous_stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        // Kirchhoff stress is returned directly → scatter with V₀, not current volume.
        particles.initial_volume[i]
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particles.velocity_gradient[i];
        particles.deformation_gradient[i] = fp_new * particles.deformation_gradient[i];
        let j = particles.deformation_gradient[i].determinant().max(MIN_J);
        let v = (particles.initial_volume[i] * j).max(1.0e-6);
        particles.volume[i] = v;
        particles.density[i] = particles.mass[i] / v;
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::NeoHookean as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            // cohesion_coeff is documented as reusable padding (Snow-only otherwise,
            // zero for all other materials) -- repurposed here for damage_softening_rate.
            cohesion_coeff: self.damage_softening_rate,
            dynamic_viscosity: self.viscosity,
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
            self.min_density,
            cell_width,
            material_cfl,
        );
        // Real bug caught 2026-07-11: `viscosity` was added (stress term) without this
        // bound, so a high-viscosity NeoHookean body took substeps sized only for elastic
        // stability -- far too large for the added viscous (parabolic/diffusive) term,
        // which has its own, much stricter stability requirement. Explicit integration of
        // a diffusive term needs dt ~ h²/ν, not h/c (elastic wave speed) -- a real, standard
        // numerical-stability fact, not tuned to this case. Caught by its actual symptom:
        // deformation gradient inverting (J < 0) within ~500 steps at viscosity=150+,
        // identical formula and bound `ViscoelasticMaterial::timestep_bound` already uses.
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

// Test suite split into its own file -- was ~440 of this file's ~770 lines, same
// reasoning as `gpu/solver/device_lost_tests.rs`: the constitutive-model file
// should read as the model, not scroll past three FD-verification suites first.
#[cfg(test)]
mod elastic_tests;

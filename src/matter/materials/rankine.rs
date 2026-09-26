use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BrittleProps, FromSI, scale_lame, scale_stress};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, RANKINE_MIN_RESIDUAL_TENSILE_FRACTION, advance_deformation_gradient,
    carried_volume_ratio, corotated_elastic_stress, elastic_wave_dt, hencky_strains,
    lame_from_young, rankine_damage_saturation_point, reconstruct_f, stress_to_hencky,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{ParticleUpdateCtx, Particles};

/// Real Kelvin-Voigt viscosity (SI Pa.s) for `RankineMaterial::elastic_viscosity`,
/// derived from a material's own measured seismic/ultrasonic quality factor Q --
/// the standard way solid-earth/ice-physics literature reports internal damping
/// (distinct from soil's small-strain damping ratio convention,
/// `granular::sand::small_strain_elastic_viscosity_pa_s`, but the same
/// underlying equivalent-viscous-damping conversion).
///
/// `zeta = 1/(2*Q)` is the standard quality-factor/damping-ratio relation
/// (Aki & Richards, "Quantitative Seismology," 2002). Real, disclosed fix
/// (2026-08-29): this engine's own `kirchhoff_stress` applies the viscous
/// term as `eta*D_dev` (NOT `2*eta*D_dev` -- a deliberate, already-tested
/// convention, see `ViscoelasticMaterial`'s own
/// `viscous_term_matches_eta_times_deviatoric_strain_rate_exactly` test),
/// so under THIS convention the textbook `eta=2*zeta*G/omega` relation
/// (which assumes the fully-matched `sigma=2G*eps+2*eta*D` tensor form)
/// silently produces a measured Q exactly 2x the target -- confirmed both
/// by hand derivation and a direct numeric cyclic-oscillation test
/// (`measured_q_factor_matches_target_after_the_conversion_fix`, this
/// file's own test module), found via independent deep research the same
/// night. Real, corrected relation for THIS engine's
/// `eta*D_dev` convention: `eta = 2*G/(Q*omega)` (double the naive
/// `G/(Q*omega))` -- verified by that same test to reproduce the cited Q
/// exactly, not just approximately.
///
/// `reference_frequency_hz` must be the SAME frequency the cited Q
/// measurement used -- Q is frequency-dependent in real polycrystalline
/// solids (grain-boundary friction loss scales ~linearly with frequency,
/// scattering loss ~quartically), so this conversion is only exact at that
/// reference, same caveat `small_strain_elastic_viscosity_pa_s` carries for
/// its own 1 Hz reference.
///
/// Returns real SI Pa.s. Must be converted with the SAME convention the
/// caller's own `lambda`/`mu` used, since `kirchhoff_stress` adds
/// `elastic_viscosity * d_dev` directly into the same stress tensor those
/// build: raw `lame_from_young`/`from_young_modulus` lambda/mu (density-
/// agnostic) pairs with this value assigned RAW, unconverted; density-
/// normalized `lame_from_si_physical`/`SimConfig::lame_from_si_physical_cfg`
/// lambda/mu pairs with `SimConfig::visc_from_si_physical(eta, rho)`. Mixing
/// the two is wrong either direction.
///
/// Real, disclosed regression found+fixed 2026-08-29: `RankineMaterial::ice`'s
/// real call sites (`examples/cpu/phase_states_gui.rs`,
/// `phase_states_headless.rs`) build `lambda`/`mu` via `ice()` ->
/// `from_young_modulus` -> raw `lame_from_young`, but were WRONGLY paired
/// with `SimConfig::visc_from_si_physical`, an ~917x-too-small division
/// (ice's real density ~917 kg/m^3, dx=1 in that scene) that belongs only
/// with the OTHER (density-normalized) lambda/mu family. Confirmed wrong
/// three ways: (1) dimensionally inconsistent with `ice()`'s own raw,
/// undivided lambda/mu; (2) this file's own
/// `measured_q_factor_matches_target_after_the_conversion_fix` test assigns
/// `elastic_viscosity: eta` raw alongside raw `lambda`/`mu` and empirically
/// measures the correct real Q (that test doesn't discriminate between
/// conventions on its own -- Q is scale-invariant if lambda/mu/eta all scale
/// together -- but it does confirm raw+raw is internally self-consistent,
/// which `ice()`'s own raw lambda/mu requires); (3) it silently explained an
/// earlier same-night finding that doubling Q "had no visible effect" -- the
/// damping was already ~917x too small before the 2x change. Fixed at
/// those two real call sites only (assign this function's return value
/// directly); the helper itself is correct and still needed by every call
/// site that legitimately pairs it with density-normalized lambda/mu (e.g.
/// `examples/cpu/sand_water_saturation.rs`).
pub fn q_factor_elastic_viscosity_pa_s(
    shear_modulus_pa: f32,
    quality_factor: f32,
    reference_frequency_hz: f32,
) -> f32 {
    debug_assert!(
        quality_factor > 0.0,
        "quality factor {quality_factor} must be positive -- Q<=0 is not a real material"
    );
    let omega = 2.0 * std::f32::consts::PI * reference_frequency_hz;
    2.0 * shear_modulus_pa / (quality_factor * omega)
}

/// Real, cited P-wave quality factor for COLD polycrystalline ice (not
/// temperate ice near 0C -- see this constant's own disclosed limitation
/// below). Two independent real sources: Bentley & Kohnen (1976), "Seismic
/// refraction measurements of internal friction in Antarctic ice," Journal
/// of Geophysical Research 81(9):1519-1526, measured Q_P ~= 715 at 136 Hz,
/// Byrd Station, ~-28C, 100-500m depth; Peters et al. (2012), "Seismic
/// attenuation in glacial ice: A proxy for englacial temperature," JGR
/// Earth Surface, independently cross-checks this with Q_P ~ 500-1700 for
/// cold Antarctic ice at the same site. Representative pick near the
/// lower/typical end of both, not the extreme.
///
/// Disclosed limitation: Q drops sharply toward the melting point (grain-
/// boundary sliding/premelting become dominant loss mechanisms) -- the
/// same Peters et al. 2012 review reports Q_P ~ 65 for TEMPERATE ice near
/// 0C (Athabasca Glacier), an order of magnitude lower. This constant
/// represents `RankineMaterial::ice()`'s own -10C reference point (cold,
/// not temperate), so real ice very close to 0C would genuinely dissipate
/// energy faster than this value implies.
pub const ICE_QUALITY_FACTOR_Q: f32 = 700.0;

/// The real measurement frequency Bentley & Kohnen (1976) used -- see
/// `ICE_QUALITY_FACTOR_Q`'s own doc. Pair the two together, never `Q`
/// alone, at a different reference frequency.
pub const ICE_Q_REFERENCE_FREQUENCY_HZ: f32 = 136.0;

/// Rankine (maximum principal stress) elastoplastic material -- brittle tensile failure.
///
/// Elastic response: corotated linear elastic (same as DruckerPragerMaterial / VonMisesMaterial).
/// Yield criterion: max(τ₁, τ₂) ≤ σ_t_eff, where τᵢ are principal Kirchhoff stresses and
///   σ_t_eff = max(tensile_strength · exp(−softening_rate · damage), tensile_strength · 5%)
///   (exponential softening, floored at a small residual so damage saturates under
///   sustained loading instead of ratcheting forever -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`).
///
/// Return mapping: when a principal stress exceeds σ_t_eff, it is projected back to the
/// tensile cutoff surface; the remaining stress component is unaffected (1D projection).
/// Biaxial tension (both τ₁ > σ_t AND τ₂ > σ_t) projects at the corner -- both set to σ_t.
///
/// Damage accumulates in `Particle::friction_hardening` (repurposed as damage), bounded
/// above by `rankine_damage_saturation_point(softening_rate)` -- the point past which
/// `t_eff` is already at its residual floor, so further stress cannot lower it any
/// more and additional accumulation would be pure bookkeeping, not physical (see
/// `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION` doc). `softening_rate <= 0` (hard cutoff,
/// no softening) has no saturation point -- damage can grow unbounded in that case,
/// same as before.
/// Softening reduces effective tensile strength exponentially toward a small residual.
///
/// Suitable for: brittle rock, bone, eggshell, chitin, ice with fracture.
/// At zero softening rate: perfect tensile cutoff (material can never exceed σ_t).
///
/// References: Rankine 1876 (original criterion); Wolper et al. 2019 (MPM brittle fracture);
/// sparkl `RankinePlasticity` (Rust open-source reference, Apache-2.0).
#[derive(Debug, Clone, Copy)]
pub struct RankineMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Maximum tensile principal Kirchhoff stress (compressive stress is unlimited).
    /// Rock: 1e3–1e4. Bone: 1e4–1e5. Ice: 1e2–1e3.
    pub tensile_strength: f32,
    /// Exponential softening rate. 0.0 = no softening (hard cutoff).
    /// Positive values reduce σ_t as damage accumulates.
    /// Typical: 0.5–5.0 -- higher = more brittle (strength collapses fast after first crack).
    pub softening_rate: f32,
    /// Real Kelvin-Voigt viscous damping on the deviatoric elastic strain
    /// rate (SI Pa.s, converted with the SAME convention `lambda`/`mu` used
    /// -- raw if they came from `lame_from_young`, `SimConfig::visc_from_si_physical`
    /// if from `lame_from_si_physical` -- see `q_factor_elastic_viscosity_pa_s`'s
    /// own doc) -- same mechanism, same formula, as
    /// `DruckerPragerMaterial::elastic_viscosity`. Zero cost, zero behavior
    /// change at `0.0` (every preset's default, same convention as sand).
    ///
    /// Without this, a pure elastic-plus-brittle-fracture model has NO
    /// energy dissipation at all below the fracture threshold -- real
    /// solids are never purely elastic (internal friction from dislocation
    /// motion and grain-boundary sliding measurably dissipates energy in
    /// every real material, reported as a seismic/ultrasonic quality
    /// factor Q -- see `q_factor_elastic_viscosity_pa_s`). Confirmed live
    /// 2026-08-28: `ice()` with `elastic_viscosity=0.0` bounces near-
    /// elastically off the ground on any sub-fracture impact, and a
    /// borderline impact can look like a wrong bounce-then-partial-
    /// fracture hybrid (some region locally exceeds the yield surface and
    /// softens while the rest of the body stays perfectly elastic).
    pub elastic_viscosity: f32,
}

impl RankineMaterial {
    /// Construct directly from grid-native Lame parameters and tensile
    /// strength -- NOT SI Pascals (see [`Self::from_young_modulus`] for the
    /// common gotcha and the real SI conversion path).
    pub const fn new(lambda: f32, mu: f32, tensile_strength: f32, softening_rate: f32) -> Self {
        Self {
            lambda,
            mu,
            tensile_strength,
            softening_rate,
            elastic_viscosity: 0.0,
        }
    }

    /// **Grid units, NOT real Pascals** (real disclosure added 2026-09-05,
    /// same finding as `NeoHookeanMaterial::from_young_modulus`'s own doc):
    /// calls [`lame_from_young`] directly, never touches `dx_meters`/
    /// density. For a real, correctly SI-to-grid-converted material use
    /// [`Self::from_physical`] (needs a `&SimConfig` and real `rho_kg_m3`).
    pub fn from_young_modulus(
        young_modulus: f32,
        poisson_ratio: f32,
        tensile_strength: f32,
        softening_rate: f32,
    ) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, tensile_strength, softening_rate)
    }

    /// Brittle rock regime: tensile strength as a real FRACTION of the caller's own
    /// `young_modulus`, not a hardcoded absolute number -- a fixed absolute value only
    /// "means" rock at one specific implicit E, silently wrong at any other (a hardcoded
    /// tensile=500 gives an 18-50% tensile/E ratio at the values this engine's own tests
    /// pass it, vs. real brittle rock's tensile-to-modulus ratio of ~2-3e-4 --
    /// granite/basalt: E~50 GPa, tensile strength~10-15 MPa (Goodman 1989, "Introduction
    /// to Rock Mechanics"). Real, fast softening_rate=2.0 (brittle failure propagates
    /// quickly) unchanged.
    pub fn stiff_brittle(young_modulus: f32, poisson_ratio: f32) -> Self {
        const ROCK_TENSILE_TO_MODULUS_RATIO: f32 = 2.5e-4;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * ROCK_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Bone regime: tensile strength as a real fraction of `young_modulus`, same fix as
    /// `stiff_brittle` above. Real cortical bone tolerates a much higher tensile-to-
    /// modulus ratio than rock (tougher composite material): E~15-20 GPa, tensile
    /// strength~100-150 MPa, ratio ~7e-3 (Currey 2002, "Bones: Structure and
    /// Mechanics"). Real, slower softening_rate=1.0 (bone fails less abruptly than
    /// rock) unchanged.
    pub fn high_tensile(young_modulus: f32, poisson_ratio: f32) -> Self {
        const BONE_TENSILE_TO_MODULUS_RATIO: f32 = 7.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * BONE_TENSILE_TO_MODULUS_RATIO,
            1.0,
        )
    }

    /// Sandstone regime: sedimentary clastic rock, same ratio-not-absolute fix as
    /// `stiff_brittle`. Real E range 11.3-40 GPa (avg ~19.9 GPa), tensile strength
    /// 19.17-65.66 MPa (Xu 2016, "Characterization of Rock Mechanical Properties
    /// Using Lab Tests and Numerical Interpretation Model of Well Logs") -- huge
    /// real spread from cementation/porosity, disclosed not hidden. Representative
    /// pick near the lower/typical end of both ranges (E~20 GPa, tensile~20 MPa).
    /// Same softening_rate=2.0 as `stiff_brittle` -- still real brittle failure,
    /// no separately-cited reason to differ.
    pub fn sandstone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SANDSTONE_TENSILE_TO_MODULUS_RATIO: f32 = 1.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SANDSTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Limestone regime: sedimentary chemical rock. Real E range 4.6-12 GPa,
    /// tensile strength 18.00-38.76 MPa (same Xu 2016 source as `sandstone`) --
    /// genuinely softer AND relatively stronger-in-tension-per-modulus than
    /// sandstone, a real distinguishing feature, not the same rock renamed.
    /// Representative pick E~8 GPa, tensile~25 MPa.
    pub fn limestone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const LIMESTONE_TENSILE_TO_MODULUS_RATIO: f32 = 3.1e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * LIMESTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Shale regime: sedimentary clastic, fissile/foliated. Real E range 15-36.9 GPa
    /// (avg ~27 GPa, foliated), tensile strength ~168 MPa average ACROSS foliation
    /// (same Xu 2016 source) -- real, cited, but an HONEST, DISCLOSED limitation:
    /// real shale is strongly anisotropic (splits far more easily ALONG bedding
    /// planes than across them; the source's own "laminated shale shows lower
    /// values" note, exact number not given). This preset is isotropic (this
    /// material's yield surface has no per-particle orientation field), so it
    /// necessarily represents the ACROSS-foliation (stronger) direction -- real
    /// bedding-plane weakness is a genuinely separate, not-yet-built mechanism
    /// (see the geosphere-taxonomy memory's "anisotropic foliated rock" gap), not
    /// something this single-number preset can honestly claim to capture.
    pub fn shale(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SHALE_TENSILE_TO_MODULUS_RATIO: f32 = 6.2e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SHALE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Ice regime: same ratio-not-absolute pattern as the rock presets above --
    /// the real, defining reason ice belongs HERE, not in `StomakhinMaterial`
    /// (snow): snow's whole constitutive law is compaction-hardening (crushing
    /// trapped air pockets between ice crystals, a real, distinct porous-media
    /// mechanism), which solid, non-porous ice does not have -- real ice is
    /// brittle, it cracks rather than crushes, the same failure MODE this
    /// material already models for rock/bone. Real E for polycrystalline ice:
    /// 9.0-11.2 GPa at -10C (randomly oriented polycrystals ~9.0 GPa at -5C;
    /// granular polycrystalline ice 9.3 GPa at 263K) -- representative pick
    /// E~9.0 GPa, the lower/typical end. Real tensile strength: 0.7-3.1 MPa
    /// over -10C to -20C (Petrovic 2003, "Review: mechanical properties of ice
    /// and snow," J. Materials Science 38), general engineering estimate ~1 MPa
    /// -- ratio ~1.1e-4, the same order of magnitude as rock's 2.5e-4 (both
    /// brittle crystalline solids, a real cross-check, not a coincidence).
    /// softening_rate=2.0 matching rock/sandstone/shale's own "fails fast, no
    /// separately-cited reason to differ" convention -- ice's own real fracture
    /// propagation is brittle/abrupt like rock, not bone's tougher, slower mode.
    ///
    /// Real, disclosed engine limitation this preset inherits, not a new one:
    /// explicit MPM must resolve the elastic wave speed `c = sqrt(E/rho)` --
    /// at E=9 GPa and real ice density this is genuinely fast, ~3130 m/s (the
    /// bar-wave speed from E alone). Real cross-check: this lands almost
    /// exactly on ice's own directly-measured laboratory rod/bar longitudinal
    /// speed, 3163 m/s (Northwood 1947, "Propagation of Elastic Waves in
    /// Ice"), the same wave mode `sqrt(E/rho)` represents -- full-body deep-
    /// ice P-wave measurements run somewhat higher (3410-3878 m/s, a
    /// different wave mode that also depends on Poisson's ratio, not just
    /// E). A scene using this preset needs a correspondingly fine substep
    /// budget, same real stiffness-vs-explicit-timestep tradeoff already
    /// documented for the `stiff_brittle`/`sandstone`/`shale` rock presets
    /// above (which also use real, unreduced GPa-scale stiffness) -- not
    /// reduced here either, for the same reason: this preset represents real
    /// ice, not a demo-scaled stand-in.
    ///
    /// Leaves `elastic_viscosity` at its default `0.0` -- this constructor
    /// (like every preset above) is a unit-agnostic function of `young_modulus`
    /// alone, with no real damping baked in (same reason `DruckerPragerMaterial`'s
    /// own presets never bake in `elastic_viscosity` either -- see
    /// `granular::sand::small_strain_elastic_viscosity_pa_s`'s own doc and
    /// `examples/cpu/sand_water_saturation.rs`'s real call site for the
    /// established pattern). A caller that needs real damping (any scene
    /// putting this preset under real gravity/impacts) should set it
    /// explicitly via struct-update syntax -- assign the raw SI Pa.s value
    /// directly, no `SimConfig` conversion (see
    /// `q_factor_elastic_viscosity_pa_s`'s own doc for why):
    /// ```ignore
    /// let g = young_modulus / (2.0 * (1.0 + poisson_ratio));
    /// let eta_pa_s = q_factor_elastic_viscosity_pa_s(g, ICE_QUALITY_FACTOR_Q, ICE_Q_REFERENCE_FREQUENCY_HZ);
    /// RankineMaterial { elastic_viscosity: eta_pa_s, ..RankineMaterial::ice(young_modulus, poisson_ratio) }
    /// ```
    pub fn ice(young_modulus: f32, poisson_ratio: f32) -> Self {
        const ICE_TENSILE_TO_MODULUS_RATIO: f32 = 1.1e-4;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * ICE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Effective tensile strength after damage softening. Floored at a small
    /// residual fraction of virgin strength -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`
    /// doc for why an unfloored exponential decay is an unbounded damage ratchet.
    #[inline]
    fn tensile_strength_eff(&self, damage: f32) -> f32 {
        (self.tensile_strength * (-self.softening_rate * damage).exp())
            .max(self.tensile_strength * RANKINE_MIN_RESIDUAL_TENSILE_FRACTION)
    }
}

impl FromSI<BrittleProps> for RankineMaterial {
    fn from_physical(props: &BrittleProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        let ts = scale_stress(props.tensile_strength_pa, props.elastic.rho_kg_m3, config);
        Self::new(lambda, mu, ts, props.softening_rate)
    }
}

impl RankineMaterial {
    /// Rankine return mapping in 2D principal stress space.
    ///
    /// Returns (projected_tau, yielded) -- `yielded` is true if any projection occurred.
    #[inline]
    fn project_stress(&self, tau: Vec2, t_eff: f32) -> (Vec2, bool) {
        let t1 = tau.x > t_eff;
        let t2 = tau.y > t_eff;
        match (t1, t2) {
            (false, false) => (tau, false),
            (true, false) => (Vec2::new(t_eff, tau.y), true),
            (false, true) => (Vec2::new(tau.x, t_eff), true),
            (true, true) => (Vec2::splat(t_eff), true), // biaxial tension corner return
        }
    }
}

impl MaterialModel for RankineMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Rankine
    }

    /// Corotated elastic Kirchhoff stress plus a Kelvin-Voigt viscous term
    /// on the deviatoric strain rate -- see `elastic_viscosity`'s own doc.
    /// Zero cost, zero behavior change when `elastic_viscosity == 0.0`
    /// (every existing preset/scene before 2026-08-28). Same formula as
    /// `DruckerPragerMaterial::kirchhoff_stress`'s own Kelvin-Voigt dashpot,
    /// and `ViscoelasticMaterial::kirchhoff_stress`'s own -- `tau_v =
    /// eta*D_dev` (NOT `2*eta*D_dev`) is this engine's one, deliberate,
    /// already-tested convention everywhere (see `ViscoelasticMaterial`'s
    /// own `viscous_term_matches_eta_times_deviatoric_strain_rate_exactly`
    /// test) -- do NOT add a factor of 2 here; a real Q-vs-target
    /// discrepancy found 2026-08-29 (independent deep research +
    /// independent hand derivation + a direct numeric cyclic-oscillation
    /// test, `measured_q_factor_matches_target_after_the_conversion_fix`
    /// below) was fixed at its actual source instead --
    /// `q_factor_elastic_viscosity_pa_s`'s own conversion formula, not this
    /// stress application -- see that function's own doc for why.
    fn corotated_lame_params(&self) -> Option<(f32, f32)> {
        if self.elastic_viscosity == 0.0 {
            Some((self.lambda, self.mu))
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let elastic =
            corotated_elastic_stress(particles.deformation_gradient[i], self.lambda, self.mu);
        if self.elastic_viscosity == 0.0 {
            return elastic;
        }
        let c = particles.velocity_gradient[i];
        let sym = c + c.transpose();
        let d = sym * 0.5;
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));
        elastic + self.elastic_viscosity * d_dev
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let (f_trial, _) = advance_deformation_gradient(
            *ctx.deformation_gradient,
            dt * *ctx.velocity_gradient,
            carried_volume_ratio(*ctx.volume, ctx.initial_volume),
        );
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);

        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );

        let damage = *ctx.friction_hardening;
        let t_eff = self.tensile_strength_eff(damage);

        let (tau_proj, yielded) = self.project_stress(tau, t_eff);

        let sigma_new = if yielded {
            let eps_proj = stress_to_hencky(tau_proj, self.lambda, self.mu);
            let eps_trial = stress_to_hencky(tau, self.lambda, self.mu);
            *ctx.friction_hardening = (damage + (eps_trial - eps_proj).length())
                .min(rankine_damage_saturation_point(self.softening_rate));
            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        *ctx.deformation_gradient = reconstruct_f(u, sigma_new, vt);
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Rankine as u32,
            lambda: self.lambda,
            mu: self.mu,
            // tensile_strength → hardening_exponent slot (union layout, CPU-only plasticity)
            hardening_exponent: self.tensile_strength,
            // softening_rate → hardening_modulus slot
            hardening_modulus: self.softening_rate,
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
            MIN_J,
            cell_width,
            material_cfl,
        );
        // Same explicit-viscous-diffusion stability bound
        // `DruckerPragerMaterial::timestep_bound` already uses for its own
        // Kelvin-Voigt term -- without this, `elastic_viscosity` adds real
        // stiffness the substep selector never sees (measured directly for
        // sand, 2026-08-25: an unbounded viscous term made peak speed jump
        // instead of damping).
        let viscous_dt = if self.elastic_viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic = self.elastic_viscosity / density;
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

    fn needs_cpu_update(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::Particle;

    /// Isolates whether `update_particle`'s return mapping matches this
    /// material's OWN documented tensile-cutoff criterion (`max(tau1,tau2) <=
    /// t_eff`) exactly -- same discipline as `sand.rs`/`von_mises.rs`'s own
    /// `marginal_yield_tests`. `RankineMaterial` had zero test comparing its
    /// return mapping to an exact analytical prediction before this (only
    /// stability + softening-direction checks existed, confirmed via the
    /// 2026-07-07 citation audit).
    fn run_one_step(mat: &RankineMaterial, sigma: Vec2, damage: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = damage;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    fn run_rate_step(
        mat: &RankineMaterial,
        f: Mat2,
        velocity_gradient: Mat2,
        dt: f32,
        damage: f32,
    ) -> (Mat2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.velocity_gradient = velocity_gradient;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = damage;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), dt);
        (
            particles.deformation_gradient[0],
            particles.friction_hardening[0],
        )
    }

    /// Given eps.y=0, tau.x = a*eps.x (a=2*mu+lambda), tau.y = lambda*eps.x --
    /// solving for the eps.x that puts tau.x EXACTLY at a target tensile stress.
    fn eps_x_for_target_tau_x(mat: &RankineMaterial, target_tau_x: f32) -> f32 {
        let a = 2.0 * mat.mu + mat.lambda;
        target_tau_x / a
    }

    #[test]
    fn marginal_state_at_tensile_strength_does_not_yield() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 0.99 * mat.tensile_strength);
        let sigma = Vec2::new(eps_x.exp(), 1.0); // eps.y = ln(1.0) = 0

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);
        assert!(
            (sigma_after - sigma).length() < 1.0e-4,
            "state inside the tensile-cutoff surface should stay elastic (no change): \
             sigma={sigma:?} sigma_after={sigma_after:?}"
        );
        assert_eq!(
            damage_after, 0.0,
            "damage must not accumulate on an elastic step"
        );
    }

    #[test]
    fn marginal_state_beyond_tensile_strength_projects_exactly_to_the_yield_surface() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength);
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);

        // The projected principal stress must land EXACTLY at tensile_strength
        // (damage=0, so t_eff=tensile_strength exactly -- no floor/saturation
        // complication from the 2026-07-07 ratchet fix).
        let a = 2.0 * mat.mu + mat.lambda;
        let eps_after_x = sigma_after.x.ln();
        let eps_after_y = sigma_after.y.ln();
        let tau_x_after = a * eps_after_x + mat.lambda * eps_after_y;
        let tau_y_after = mat.lambda * eps_after_x + a * eps_after_y;
        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3,
            "projected principal stress should land EXACTLY on the tensile-cutoff \
             surface (t_eff={}), got {tau_x_after:.6}",
            mat.tensile_strength
        );

        // The real invariant for the UNAFFECTED principal direction is the
        // STRESS tau.y (not the strain sigma.y/eps.y) staying fixed -- this is
        // a single-component projection in STRESS space (project_stress's
        // (true,false) branch only rewrites tau.x). Because stress and strain
        // are coupled through lambda, inverting back to strain space changes
        // BOTH eps.x and eps.y even though only tau.x was projected -- eps.y
        // changing is real, correct coupled elasticity, not a bug (confirmed:
        // an earlier version of this test wrongly asserted sigma.y itself must
        // stay fixed, and failed -- the fix is checking the right invariant).
        let original_tau_y = mat.lambda * eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength)
            + a * sigma.y.ln();
        assert!(
            (tau_y_after - original_tau_y).abs() < 1.0e-3,
            "the non-yielding principal STRESS (tau.y) must be untouched: \
             expected {original_tau_y}, got {tau_y_after}"
        );

        assert!(
            damage_after > 0.0,
            "damage must accumulate on a plastic (yielding) step"
        );
    }

    #[test]
    fn biaxial_tension_projects_both_components_to_the_corner() {
        // Both principal stresses exceed t_eff simultaneously -- the documented
        // "corner return" case (project_stress's (true,true) branch).
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let a = 2.0 * mat.mu + mat.lambda;
        // Symmetric biaxial tension: eps.x = eps.y = e, giving tau.x=tau.y=(a+lambda)*e.
        let e = (1.5 * mat.tensile_strength) / (a + mat.lambda);
        let sigma = Vec2::new(e.exp(), e.exp());

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, 0.0);
        let eps_after_x = sigma_after.x.ln();
        let eps_after_y = sigma_after.y.ln();
        let tau_x_after = a * eps_after_x + mat.lambda * eps_after_y;
        let tau_y_after = mat.lambda * eps_after_x + a * eps_after_y;

        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3
                && (tau_y_after - mat.tensile_strength).abs() < 1.0e-3,
            "biaxial tension must project BOTH components exactly to t_eff={}: \
             got tau_x={tau_x_after:.6} tau_y={tau_y_after:.6}",
            mat.tensile_strength
        );
        assert!(
            damage_after > 0.0,
            "damage must accumulate on the corner-return case"
        );
    }

    /// The defining, category-specific behavior of a BRITTLE material (vs.
    /// VonMises's non-softening ductile plasticity): damage makes the body
    /// genuinely WEAKER, not just permanently deformed. A stress level
    /// comfortably under the VIRGIN tensile strength must still fail once the
    /// particle already carries real damage -- softening = real loss of
    /// load-bearing capacity, not bookkeeping. No prior test exercised
    /// `run_one_step` with `damage > 0` at all (confirmed via a read of every
    /// call site in this module before writing this one).
    #[test]
    fn accumulated_damage_lowers_the_yield_threshold_real_softening() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 2.0);
        let damage = 0.5; // well under this softening_rate's saturation point (~1.5)
        let t_eff_damaged = mat.tensile_strength * (-mat.softening_rate * damage).exp();

        // 70% of virgin strength: comfortably elastic at damage=0 (this
        // module's own `marginal_state_at_tensile_strength_does_not_yield`
        // treats 99% as still-elastic), but above the damaged threshold.
        let target_tau_x = 0.7 * mat.tensile_strength;
        assert!(
            target_tau_x > t_eff_damaged,
            "test setup sanity: target stress must exceed the damaged threshold"
        );
        let eps_x = eps_x_for_target_tau_x(&mat, target_tau_x);
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let (sigma_after, damage_after) = run_one_step(&mat, sigma, damage);

        assert!(
            (sigma_after - sigma).length() > 1.0e-4,
            "a damaged particle must yield at a stress the VIRGIN material \
             would have carried elastically: target_tau_x={target_tau_x} \
             t_eff_damaged={t_eff_damaged}"
        );
        let a = 2.0 * mat.mu + mat.lambda;
        let tau_x_after = a * sigma_after.x.ln() + mat.lambda * sigma_after.y.ln();
        assert!(
            (tau_x_after - t_eff_damaged).abs() < 1.0e-3,
            "projected stress should land exactly at the DAMAGED threshold \
             ({t_eff_damaged:.4}), not the virgin one ({}): got {tau_x_after:.6}",
            mat.tensile_strength
        );
        assert!(
            damage_after > damage,
            "damage must keep accumulating on repeated yielding"
        );
    }

    #[test]
    fn exponential_trial_preserves_rigid_rotation_without_false_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let dt = 0.25;
        let omega = 0.8;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
        let (f_after, damage_after) = run_rate_step(&mat, Mat2::IDENTITY, spin, dt, 0.0);
        let expected = Mat2::from_angle(omega * dt);
        let error = (f_after.x_axis - expected.x_axis)
            .abs()
            .max_element()
            .max((f_after.y_axis - expected.y_axis).abs().max_element());
        assert!(
            error < 1.0e-6,
            "rigid spin must integrate to a rotation: {f_after:?}"
        );
        assert!((f_after.determinant() - 1.0).abs() < 1.0e-6);
        assert_eq!(
            damage_after, 0.0,
            "rigid rotation must not create Rankine damage"
        );
    }

    #[test]
    fn opposite_subthreshold_rates_are_reversible_without_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 0.5 * mat.tensile_strength);
        let rate = Mat2::from_diagonal(Vec2::new(eps_x, 0.0));
        let (f_loaded, damage_loaded) = run_rate_step(&mat, Mat2::IDENTITY, rate, 1.0, 0.0);
        assert_eq!(damage_loaded, 0.0);
        let (f_unloaded, damage_unloaded) =
            run_rate_step(&mat, f_loaded, -rate, 1.0, damage_loaded);
        let error = (f_unloaded.x_axis - Mat2::IDENTITY.x_axis)
            .abs()
            .max_element()
            .max(
                (f_unloaded.y_axis - Mat2::IDENTITY.y_axis)
                    .abs()
                    .max_element(),
            );
        assert!(
            error < 1.0e-6,
            "opposite rates must restore F exactly: {f_unloaded:?}"
        );
        assert_eq!(damage_unloaded, 0.0);
    }

    #[test]
    fn nonzero_rate_trial_projects_to_rankine_surface_and_accumulates_damage() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let eps_x = eps_x_for_target_tau_x(&mat, 1.5 * mat.tensile_strength);
        let rate = Mat2::from_diagonal(Vec2::new(eps_x, 0.0));
        let (f_after, damage_after) = run_rate_step(&mat, Mat2::IDENTITY, rate, 1.0, 0.0);
        let (_, sigma_after, _) = svd2(f_after);
        let eps_after = hencky_strains(sigma_after);
        let a = 2.0 * mat.mu + mat.lambda;
        let tau_x_after = a * eps_after.x + mat.lambda * eps_after.y;
        assert!(
            (tau_x_after - mat.tensile_strength).abs() < 1.0e-3,
            "rate-generated trial state must return to the tensile surface: {tau_x_after}"
        );
        assert!(damage_after > 0.0);
    }
}

#[cfg(test)]
mod damping_tests {
    use super::*;
    use crate::Particle;

    /// `q_factor_elastic_viscosity_pa_s` must return a real, finite,
    /// positive SI viscosity for the cited real ice Q-factor -- basic
    /// sanity floor before trusting the constant in a live scene.
    #[test]
    fn ice_quality_factor_viscosity_is_finite_and_positive() {
        let shear_modulus_pa = 9.0e9 / (2.0 * (1.0 + 0.3));
        let eta = q_factor_elastic_viscosity_pa_s(
            shear_modulus_pa,
            ICE_QUALITY_FACTOR_Q,
            ICE_Q_REFERENCE_FREQUENCY_HZ,
        );
        assert!(
            eta.is_finite() && eta > 0.0,
            "real ice Q-factor must produce a finite, positive Pa.s viscosity, got {eta}"
        );
    }

    /// A higher Q (less damping, real physical meaning: colder/purer ice)
    /// must produce a LOWER viscosity -- the real inverse relationship
    /// `eta = G/(Q*omega)`, not an accidental monotonic-the-wrong-way bug.
    #[test]
    fn higher_quality_factor_means_less_damping() {
        let g = 4.0e9;
        let low_q_eta = q_factor_elastic_viscosity_pa_s(g, 100.0, 136.0);
        let high_q_eta = q_factor_elastic_viscosity_pa_s(g, 1700.0, 136.0);
        assert!(
            high_q_eta < low_q_eta,
            "higher Q (less real damping) must give lower viscosity: \
             Q=100 -> {low_q_eta}, Q=1700 -> {high_q_eta}"
        );
    }

    /// Real, direct empirical check of `elastic_viscosity`'s actual damping
    /// against its CITED target Q -- not a sanity floor like the two tests
    /// above, a genuine measurement. Forces one particle through a full
    /// cycle of pure-shear oscillation (`F(t) = I + A*sin(wt)*E`, E
    /// symmetric and traceless so the corotated rotation R=I EXACTLY --
    /// same construction `CorotatedMaterial`'s own
    /// `small_shear_strain_matches_hookes_law` test uses to stay in the
    /// exact small-strain linear-elasticity limit), reads `kirchhoff_stress`
    /// at many samples, and numerically integrates the real dissipated
    /// energy per cycle via `tau:D` (the elastic term's own contribution to
    /// this integral is exactly zero over a full cycle -- a standard
    /// property of any conservative term, not assumed away). Standard
    /// viscoelastic definition: `Q = 2*pi*E_max/delta_E_cycle`.
    ///
    /// Real regression guard (2026-08-29) against reintroducing the bug
    /// this same test originally caught: `q_factor_elastic_viscosity_pa_s`'s
    /// naive `G/(Q*omega)` (matching the textbook `eta=2*zeta*G/omega`
    /// relation for a fully-matched `sigma=2G*eps+2*eta*D` tensor form) was
    /// silently wrong for THIS engine's actual `eta*D_dev` (no factor of 2)
    /// stress convention -- measured, via this exact test, to give a Q 2x
    /// the cited target (half the intended damping). Found via
    /// independent deep research, confirmed by hand
    /// derivation, then fixed at the conversion function itself (now
    /// `2*G/(Q*omega)`) rather than changing the stress formula, since
    /// `ViscoelasticMaterial`'s own `eta*D_dev` convention is deliberate
    /// and already locked by its own test
    /// (`viscous_term_matches_eta_times_deviatoric_strain_rate_exactly`).
    #[test]
    fn measured_q_factor_matches_target_after_the_conversion_fix() {
        let mu = 4.0e9_f32;
        let lambda = 4.0e9_f32;
        let target_q = 65.0_f32;
        let reference_freq_hz = 136.0_f32;
        let omega = 2.0 * std::f32::consts::PI * reference_freq_hz;
        let eta = q_factor_elastic_viscosity_pa_s(mu, target_q, reference_freq_hz);

        let mat = RankineMaterial {
            elastic_viscosity: eta,
            ..RankineMaterial::new(lambda, mu, f32::MAX, 0.0)
        };

        let amplitude = 1.0e-4_f32; // small-strain regime
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)); // pure shear, traceless, R=I exactly

        let n_samples = 2000_u32;
        let period = 2.0 * std::f32::consts::PI / omega;
        let dt = period / n_samples as f32;

        let mut dissipated = 0.0_f64;
        for k in 0..n_samples {
            let t = k as f32 * dt;
            let delta = amplitude * (omega * t).sin();
            let delta_dot = amplitude * omega * (omega * t).cos();

            let mut p = Particle::zeroed();
            p.deformation_gradient = Mat2::IDENTITY + delta * e;
            p.velocity_gradient = delta_dot * e;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            let particles = Particles::from(vec![p]);

            let tau = mat.kirchhoff_stress(&particles, 0);
            let d = delta_dot * e; // already symmetric (E is symmetric)
            let power = (tau.x_axis.x * d.x_axis.x
                + tau.x_axis.y * d.x_axis.y
                + tau.y_axis.x * d.y_axis.x
                + tau.y_axis.y * d.y_axis.y) as f64;
            dissipated += power * dt as f64;
        }

        let e_max = 2.0 * (mu as f64) * (amplitude as f64).powi(2); // peak elastic energy density, pure shear
        let measured_q = 2.0 * std::f64::consts::PI * e_max / dissipated;

        assert!(
            (measured_q - target_q as f64).abs() / (target_q as f64) < 0.05,
            "measured Q from actual kirchhoff_stress output must match the \
             cited target Q to within numerical error: target_q={target_q}, \
             got measured_q={measured_q:.3}"
        );
    }

    /// The actual mechanism this was added for: `elastic_viscosity > 0.0`
    /// must make `kirchhoff_stress` respond to the particle's velocity
    /// gradient (viscous stress), not just its deformation gradient
    /// (elastic stress) -- confirms the Kelvin-Voigt term actually engages,
    /// not just that the field exists.
    #[test]
    fn nonzero_elastic_viscosity_adds_a_real_viscous_stress_term() {
        let elastic_only = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        let mut damped = elastic_only;
        damped.elastic_viscosity = 50.0;

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        // A pure shearing velocity gradient -- nonzero deviatoric strain
        // rate, the exact quantity the Kelvin-Voigt term reacts to.
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let particles = Particles::from(vec![p]);

        let tau_elastic = elastic_only.kirchhoff_stress(&particles, 0);
        let tau_damped = damped.kirchhoff_stress(&particles, 0);

        let diff = tau_damped - tau_elastic;
        let max_abs = diff
            .x_axis
            .abs()
            .max_element()
            .max(diff.y_axis.abs().max_element());
        assert!(
            max_abs > 1.0e-6,
            "nonzero elastic_viscosity under a real velocity gradient must \
             change the Kirchhoff stress: elastic={tau_elastic:?} damped={tau_damped:?}"
        );
    }

    /// `elastic_viscosity == 0.0` (every preset's default) must reproduce
    /// the exact pre-2026-08-28 pure-elastic stress -- a real regression
    /// guard that adding the damping mechanism did not change default
    /// behavior for any existing preset/scene.
    #[test]
    fn zero_elastic_viscosity_is_bit_identical_to_pure_elastic_stress() {
        let mat = RankineMaterial::new(2000.0, 3000.0, 100.0, 1.0);
        assert_eq!(mat.elastic_viscosity, 0.0);

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(0.01, 0.97));
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4));
        let particles = Particles::from(vec![p]);

        let tau = mat.kirchhoff_stress(&particles, 0);
        let expected =
            corotated_elastic_stress(particles.deformation_gradient[0], mat.lambda, mat.mu);
        assert_eq!(
            tau, expected,
            "elastic_viscosity=0.0 must be bit-identical to the pure elastic path"
        );
    }
}

#[cfg(test)]
mod rock_preset_tests {
    use super::*;
    use crate::Particle;

    /// Real, cited rock presets must be genuinely DIFFERENT materials, not the
    /// same numbers under different names -- checks the real distinguishing
    /// feature each preset's own doc claims: limestone is softer (lower E) than
    /// sandstone AND relatively stronger in tension per unit stiffness (higher
    /// tensile-to-modulus ratio), a real geotechnical distinction (Xu 2016), not
    /// an assumption.
    #[test]
    fn sandstone_and_limestone_presets_are_genuinely_distinct() {
        let sandstone = RankineMaterial::sandstone(20.0e9, 0.25);
        let limestone = RankineMaterial::limestone(8.0e9, 0.25);

        assert!(
            limestone.tensile_strength / limestone.lambda.max(1.0)
                != sandstone.tensile_strength / sandstone.lambda.max(1.0),
            "sandstone and limestone presets must not collapse to the same ratio"
        );
        let sandstone_ratio = sandstone.tensile_strength / 20.0e9;
        let limestone_ratio = limestone.tensile_strength / 8.0e9;
        assert!(
            limestone_ratio > sandstone_ratio,
            "limestone's real tensile-to-modulus ratio should be higher than \
             sandstone's (Xu 2016): limestone={limestone_ratio:.2e} sandstone={sandstone_ratio:.2e}"
        );
    }

    /// All 6 Rankine presets (bone/rock/ice family) must produce finite, positive
    /// tensile strengths at a real representative modulus -- a basic sanity floor
    /// before trusting any of them in a live scene.
    #[test]
    fn all_rock_and_bone_presets_produce_finite_positive_tensile_strength() {
        let e = 30.0e9;
        let nu = 0.25;
        for (name, mat) in [
            ("stiff_brittle", RankineMaterial::stiff_brittle(e, nu)),
            ("high_tensile", RankineMaterial::high_tensile(e, nu)),
            ("sandstone", RankineMaterial::sandstone(e, nu)),
            ("limestone", RankineMaterial::limestone(e, nu)),
            ("shale", RankineMaterial::shale(e, nu)),
            ("ice", RankineMaterial::ice(e, nu)),
        ] {
            assert!(
                mat.tensile_strength.is_finite() && mat.tensile_strength > 0.0,
                "{name}: tensile_strength must be finite and positive, got {}",
                mat.tensile_strength
            );
        }
    }

    /// Real Tier-0 closure (2026-09-02): the two tests above only compare
    /// preset PARAMETERS (`tensile_strength` values/ratios) -- real, but
    /// weaker evidence than watching the actual EMERGENT fracture behavior
    /// `rock_fracture.rs`'s own live demo shows (repeated strikes, weaker
    /// rock visibly damages, stiffer rock doesn't). This closes that gap
    /// with a real, dynamic, automated check through `update_particle`'s
    /// own return mapping, same technique `marginal_yield_tests::run_one_
    /// step` (this file, sibling module) already uses for a single preset.
    ///
    /// `sandstone`/`limestone` both funnel through the SAME
    /// `from_young_modulus` at the SAME (E, nu) here, so they share
    /// IDENTICAL lambda/mu -- the only real difference is `tensile_strength`
    /// (sandstone ratio 1.0e-3 vs limestone's real, cited, stronger-per-
    /// modulus 3.1e-3, Xu 2016): at E=20 GPa, sandstone=20 MPa,
    /// limestone=62 MPa. A single, IDENTICAL real tensile stress state
    /// (35 MPa, strictly between the two) must therefore fracture
    /// (damage-accumulate) sandstone while leaving limestone elastic --
    /// the real, live comparative claim the example demonstrates
    /// visually, now checked automatically.
    #[test]
    fn weaker_rock_fractures_under_a_load_stiffer_rock_survives() {
        let e = 20.0e9;
        let nu = 0.25;
        let sandstone = RankineMaterial::sandstone(e, nu);
        let limestone = RankineMaterial::limestone(e, nu);
        assert!(
            sandstone.tensile_strength < limestone.tensile_strength,
            "test setup sanity: sandstone must be the weaker preset here"
        );

        let target_tau_x = 0.5 * (sandstone.tensile_strength + limestone.tensile_strength);
        assert!(
            target_tau_x > sandstone.tensile_strength && target_tau_x < limestone.tensile_strength,
            "test setup sanity: target stress must sit strictly between the \
             two real tensile strengths"
        );

        // Same real construction `run_one_step`/`eps_x_for_target_tau_x`
        // (marginal_yield_tests, this file) use: eps.y=0, tau.x=a*eps.x
        // with a=2*mu+lambda (IDENTICAL for both presets here).
        let a = 2.0 * sandstone.mu + sandstone.lambda;
        assert_eq!(
            a,
            2.0 * limestone.mu + limestone.lambda,
            "sandstone and limestone must share identical lambda/mu at the \
             same (E, nu) -- only tensile_strength should differ"
        );
        let eps_x = target_tau_x / a;
        let sigma = Vec2::new(eps_x.exp(), 1.0);

        let one_step = |mat: &RankineMaterial| -> (Vec2, f32) {
            let mut p = Particle::zeroed();
            p.deformation_gradient =
                Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
            p.mass = 1.0;
            p.initial_volume = 1.0;
            let mut particles = Particles::from(vec![p]);
            mat.update_particle(&mut particles.update_ctx(0), 1.0);
            let f = particles.deformation_gradient[0];
            (
                Vec2::new(f.x_axis.x, f.y_axis.y),
                particles.friction_hardening[0],
            )
        };

        let (sigma_sandstone, damage_sandstone) = one_step(&sandstone);
        let (sigma_limestone, damage_limestone) = one_step(&limestone);

        assert!(
            (sigma_sandstone - sigma).length() > 1.0e-4 && damage_sandstone > 0.0,
            "sandstone (weaker, tensile_strength={:.3e}) must fracture under \
             {target_tau_x:.3e} Pa: sigma_after={sigma_sandstone:?} damage={damage_sandstone}",
            sandstone.tensile_strength
        );
        assert!(
            (sigma_limestone - sigma).length() < 1.0e-6 && damage_limestone == 0.0,
            "limestone (stronger, tensile_strength={:.3e}) must stay ELASTIC \
             under the SAME {target_tau_x:.3e} Pa: sigma_after={sigma_limestone:?} \
             damage={damage_limestone}",
            limestone.tensile_strength
        );
    }
}

use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, NaccProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{MIN_J, deformation_increment_exp, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Non-Associated Cam-Clay (NACC) elastoplastic solid.
///
/// Elastic energy: Neo-Hookean (κ bulk, µ shear).
/// Yield surface: ellipse in (p, q) space -- q² + M²·(p + β·p₀)·(p − p₀) ≤ 0
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
/// Unlike Drucker-Prager (cone), NACC has a *cap* -- it limits compression too.
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
    /// Bulk modulus κ. Note: λ (Lamé) = κ − µ (2D plane-strain relation, not
    /// the 3D κ=λ+2µ/3 relation; see `timestep_bound` and `params()` below,
    /// which invert as λ=κ−µ).
    pub kappa: f32,
    /// Friction slope M -- controls yield surface width in q direction.
    /// Related to friction angle φ (sparkl's `NaccPlasticity::new`, general-d form:
    /// M = √(2/3)·2·sin φ/(3−sin φ)·d/√(2/(6−d))): in 2D (d=2) this reduces to
    /// M = (8/√3)·sin φ/(3−sin φ) ≈ 4.619·sin φ/(3−sin φ). Not called by any
    /// constructor here (presets pass M directly) -- informational only. Do
    /// not use the 3D-style `6 sin φ/(3−sin φ)·√((6−d)/2)` form -- it is
    /// ~1.84x too large at d=2.
    /// Typical: 1.0–2.0.
    pub friction: f32,
    /// Cohesion (beta β) -- shifts yield surface min tip.
    /// 0.0 = no tensile strength (standard). 1.0 = symmetric around p=0.
    pub cohesion: f32,
    /// Hardening factor ξ. Controls how fast p₀ grows: p₀ = κ·(1e-5 + sinh(ξ·max(−α,0))).
    /// 0.0 = no hardening (perfect plasticity cap). Typical: 1.0–5.0.
    pub hardening_factor: f32,
    /// Enable volumetric hardening. If false, p₀ stays fixed (perfect plasticity cap).
    pub hardening_enabled: bool,
    pub min_density: f32,
    /// Real Kelvin-Voigt viscous damping on the deviatoric elastic strain
    /// rate (SI Pa.s, converted with the SAME convention `lambda`/`mu` used
    /// -- see `rankine::q_factor_elastic_viscosity_pa_s`'s own doc for the
    /// pairing rule and the real regression it documents) -- same
    /// mechanism, same formula, as
    /// `RankineMaterial::elastic_viscosity` / `DruckerPragerMaterial::elastic_viscosity`.
    /// Zero cost, zero behavior change at `0.0` (default, matching every
    /// other material using this same mechanism).
    ///
    /// Only damps the ELASTIC response below yield -- real soils/clays are
    /// never purely elastic even before plastic flow begins (internal
    /// friction measurably dissipates energy, reported for soils/clays as a
    /// small-strain damping ratio -- Seed & Idriss 1970, "Soil Moduli and
    /// Damping Factors for Dynamic Response Analyses"; Darendeli 2001 PhD
    /// dissertation modulus-reduction/damping curves, same sources already
    /// used for `DruckerPragerMaterial::elastic_viscosity`'s own sand
    /// default, directly applicable here since both papers cover clay/soil
    /// damping, not only sand). Confirmed live 2026-08-29: this is a real,
    /// structural gap -- `NaccMaterial` had NO damping mechanism of any kind
    /// before this field existed (found while root-causing sustained
    /// post-impact bouncing on `RankineMaterial::ice()`, same class of
    /// missing dissipation, different material).
    pub elastic_viscosity: f32,
    /// Real apparent cohesion (Pa) from soil suction at LOW saturation, via
    /// the SAME generic `MaterialModel::cohesion_bonus_pa` engine-level hook
    /// `DruckerPragerMaterial` (sand) already uses -- see that method's own
    /// doc for the hook's own contract. This is the SECOND real material to
    /// wire it (2026-09-02, closing the "extend to a 2nd material" item of
    /// the 2026-08-23 dual-phase-coupling plan) -- deliberately a DIFFERENT
    /// real mechanism from sand's, not a copy-paste: sand's capillary
    /// bridging PEAKS at low-but-nonzero saturation and requires discrete
    /// grain-scale menisci (Hornbaker et al. 1997), a coarse-granular
    /// phenomenon. Fine-grained soil/clay's own real apparent cohesion comes
    /// from MATRIC SUCTION instead (Fredlund & Rahardjo 1993, "Soil
    /// Mechanics for Unsaturated Soils," extended Mohr-Coulomb: tau_f =
    /// c' + (sigma-u_a)*tan(phi') + (u_a-u_w)*tan(phi^b) -- the suction term
    /// (u_a-u_w)*tan(phi^b) IS an apparent-cohesion contribution), which is
    /// HIGHEST at low saturation (dry clay: high suction, real, well-known
    /// behavior -- e.g. desiccation-cracked clay holding together as hard
    /// clods) and vanishes toward full saturation (zero suction, real
    /// unconfined-strength-vs-water-content relations, Terzaghi & Peck) --
    /// the OPPOSITE saturation trend from sand's own peak, a real,
    /// mechanistically distinct effect, not the same formula reused.
    ///
    /// Real, disclosed simplification, same honesty standard as sand's own
    /// `pendular_regime_ceiling` doc: a plain linear decrease from this
    /// coefficient's own full value at `Sr=0` to `0.0` at `Sr=1`, not a
    /// literal transcription of any cited paper's own suction/saturation
    /// curve (the real soil-water characteristic curve, e.g. van Genuchten
    /// 1980, is nonlinear) -- captures "drier clay is apparently stronger,"
    /// not the full non-monotonic real curve. 0.0 (default) = byte-identical
    /// to every existing preset/scene that doesn't opt in.
    pub saturation_cohesion_coeff: f32,
}

/// Named-field alternative to [`NaccMaterial::new`]'s 5 positional `f32`
/// arguments -- same real struct-bundling fix already used elsewhere in
/// this codebase (`PhysicalRenderContractParams`, `ContactKinematics`,
/// `SubstepScene`/`SubstepBounds`) for a constructor where several
/// same-typed adjacent parameters make transposition a real, silent risk
/// (swapping `friction`/`cohesion` compiles without a hint). Additive only,
/// same "new sibling, don't rename in place" precedent as `lame_from_si`
/// vs `lame_from_si_physical` -- `new` stays exactly as-is for every
/// already-tuned call site.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NaccMaterialParams {
    pub mu: f32,
    pub kappa: f32,
    pub friction: f32,
    pub cohesion: f32,
    pub hardening_factor: f32,
}

impl NaccMaterial {
    /// Construct directly from grid-native shear/bulk moduli and Cam-Clay
    /// parameters -- NOT SI Pascals (see [`Self::from_young_modulus`] for
    /// the common gotcha and the real SI conversion path). Prefer
    /// [`Self::from_params`] for new code -- same values, named fields, no
    /// risk of transposing the two same-typed adjacent parameters.
    pub fn new(mu: f32, kappa: f32, friction: f32, cohesion: f32, hardening_factor: f32) -> Self {
        Self {
            mu,
            kappa,
            friction,
            cohesion,
            hardening_factor,
            hardening_enabled: hardening_factor > 0.0,
            min_density: 1.0e-6,
            elastic_viscosity: 0.0,
            saturation_cohesion_coeff: 0.0,
        }
    }

    /// Same as [`Self::new`], named fields instead of 5 positional `f32`s --
    /// see [`NaccMaterialParams`]'s own doc for why.
    pub fn from_params(params: NaccMaterialParams) -> Self {
        Self::new(
            params.mu,
            params.kappa,
            params.friction,
            params.cohesion,
            params.hardening_factor,
        )
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    /// Friction slope M and cohesion β set separately.
    ///
    /// **Grid units, NOT real Pascals** (real disclosure added 2026-09-05,
    /// same finding as `NeoHookeanMaterial::from_young_modulus`'s own doc):
    /// calls [`lame_from_young`] directly, never touches `dx_meters`/
    /// density. Real, correctly SI-to-grid-converted construction is now
    /// possible via [`Self::from_physical`] (`FromSI<NaccProps>`, added the
    /// same night this gap was found) -- this raw constructor stays
    /// grid-unit-only.
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

    /// Peat / organic soil (USDA Histosol order): M=1.2, β=0, ξ=0.5 -- the LOWEST
    /// hardening_factor in this family, real and deliberate: peat's single most
    /// defining geotechnical trait is extreme compressibility, its real compression
    /// index (Cc) runs roughly an order of magnitude beyond mineral clays (Mesri &
    /// Ajlouni 2007, "Engineering Properties of Fibrous Peats", ASCE J. Geotech.
    /// Geoenviron. Eng.) -- meaning a peat needs far more real plastic volumetric
    /// strain than any mineral soil here before building up meaningful
    /// preconsolidation resistance (p0 growth, this material's own hardening
    /// mechanism, scales with xi -- see `project`'s own doc). Friction slope M kept
    /// near `wet_soil`'s (fibrous peat's real shear resistance from fiber
    /// interlocking is a real, separate, well-documented effect, but less
    /// distinctive than its compressibility -- not this preset's point of
    /// differentiation). HONEST DISCLOSURE, same standard as every other preset in
    /// this file: the constitutive LAW (Cam-Clay) and the qualitative direction
    /// (very low hardening_factor) are real and cited; the exact numeric value 0.5
    /// is illustrative, not fitted to a specific measured peat dataset.
    pub fn peat(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        let kappa = lambda + mu;
        Self::new(mu, kappa, 1.2, 0.0, 0.5)
    }

    /// NACC yield surface projection. Returns updated (F, alpha).
    ///
    /// Three cases from sparkl canonical:
    ///   A -- p_trial > p₀:          compress past preconsolidation → project to max cap
    ///   B -- p_trial < −β·p₀:       pull past tensile limit → project to min tip
    ///   C -- yield surface exceeded: project onto ellipse
    ///   elastic: inside yield surface → no projection
    ///
    /// `cohesion_bonus_pa` (real, Pa, from `Self::cohesion_bonus_pa` -- see
    /// that method's own doc) adds directly onto the ellipse's own `β·p₀`
    /// shift term everywhere it appears below: both are real, additive,
    /// Pa-valued contributions to the SAME "how far the ellipse's own
    /// tension tip sits below p=0" quantity, so `cohesive_shift_pa = β·p₀ +
    /// cohesion_bonus_pa` is the dimensionally exact generalization, not an
    /// approximation -- 0.0 reproduces the original `β·p₀` bit-for-bit.
    fn project(&self, f: Mat2, mut alpha: f32, cohesion_bonus_pa: f32) -> (Mat2, f32) {
        let xi = self.hardening_factor;
        let beta = self.cohesion;
        let m = self.friction;

        let (u, sigma, vt) = svd2(f);

        let sv = Vec2::new(sigma.x, sigma.y);
        let sv_sq = sv * sv;
        let sv_sq_trace = sv_sq.x + sv_sq.y;

        // Current preconsolidation pressure.
        let p0 = self.kappa * (1.0e-5 + (xi * (-alpha).max(0.0)).sinh());
        // See this function's own doc: generalizes every `beta*p0` below to
        // include the real, separate saturation-cohesion contribution.
        let cohesive_shift_pa = beta * p0 + cohesion_bonus_pa;

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
            let j_old_cap = (-2.0 * p0 / self.kappa + 1.0).max(1.0e-8_f32).sqrt();
            let j_n1 = if self.hardening_enabled {
                let (j_n1, compaction) = self.cap_return(j_e_tr, j_old_cap, alpha);
                alpha -= compaction;
                j_n1
            } else {
                j_old_cap
            };
            let sv_new = j_n1.powf(0.5); // J^(1/2) since d=2
            let sigma_new = Vec2::splat(sv_new);
            return (reconstruct(u, sigma_new, vt), alpha);
        }

        // Case B: past min tip (tensile failure).
        if p_tr < -cohesive_shift_pa {
            let j_n1 = (2.0 * cohesive_shift_pa / self.kappa + 1.0)
                .max(1.0e-8_f32)
                .sqrt();
            let sv_new = j_n1.powf(0.5);
            let sigma_new = Vec2::splat(sv_new);
            if self.hardening_enabled {
                alpha += (j_e_tr / j_n1).ln();
            }
            return (reconstruct(u, sigma_new, vt), alpha);
        }

        // Yield function: y = (1+2β)·(6−2)/2·‖s_tr‖² + M²·(p_tr+β·p₀)·(p_tr−p₀)
        // In 2D: d=2, factor = (6−d)/2 = 2. `β·p₀` generalized to
        // `cohesive_shift_pa` -- see this function's own doc.
        let y0 = (1.0 + 2.0 * beta) * 2.0_f32; // (6-d)/2 with d=2
        let y1 = m * m * (p_tr + cohesive_shift_pa) * (p_tr - p0);
        let s_norm_sq = s_tr.x * s_tr.x + s_tr.y * s_tr.y;
        let y = y0 * s_norm_sq + y1;

        if y < 1.0e-4 {
            // Inside yield surface -- elastic, no projection.
            return (f, alpha);
        }

        // Hardening: move p₀ to reduce y to zero. `β·p₀` generalized to
        // `cohesive_shift_pa` throughout -- see this function's own doc.
        let mut y1 = y1;
        if self.hardening_enabled
            && p0 > 1.0e-4
            && p_tr < p0 - 1.0e-4
            && p_tr > -cohesive_shift_pa + 1.0e-4
        {
            let q_tr = (2.0_f32).sqrt() * s_tr.length();
            // The return direction is the ray from the start-of-step centre
            // through the trial state; only the hardening is taken at the end.
            let centre = f64::from((p0 - cohesive_shift_pa) * 0.5);
            let increment = |alpha_end: f64| {
                self.ray_increment(alpha_end, centre, p_tr, q_tr, j_e_tr, cohesion_bonus_pa)
            };
            let start = increment(f64::from(alpha));
            if start < 0.0 {
                // Wet side: the soil hardens. Evaluated at the end of the step,
                // as on the cap, so the surface the stress is projected onto is
                // the hardened one (see `ray_increment`).
                let alpha_end = self.hardened_alpha(f64::from(alpha), start, increment);
                alpha = alpha_end as f32;
                let p0_end = self.kappa * (1.0e-5 + (xi * (-alpha).max(0.0)).sinh());
                let shift_end = beta * p0_end + cohesion_bonus_pa;
                y1 = m * m * (p_tr + shift_end) * (p_tr - p0_end);
            } else if start.is_finite() {
                // Dry side: softening, still evaluated at the start of the step.
                alpha += start as f32;
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

    /// Plastic volume change the shear-and-compression branch gives one step,
    /// `ln(j_tr / J_x)`: `J_x` is where the ray from `centre` (on the p axis)
    /// through the trial state `(p_tr, q_tr)` meets the ellipse whose
    /// preconsolidation pressure comes from `alpha_end`. Negative on the wet
    /// side of the ellipse (hardening), positive on the dry side. Infinite
    /// when the ray does not reach a usable volume ratio, in which case alpha
    /// is left unchanged.
    fn ray_increment(
        &self,
        alpha_end: f64,
        centre: f64,
        p_tr: f32,
        q_tr: f32,
        j_tr: f32,
        cohesion_bonus_pa: f32,
    ) -> f64 {
        let kappa = f64::from(self.kappa);
        let m_sq = f64::from(self.friction) * f64::from(self.friction);
        let beta = f64::from(self.cohesion);
        let p0 =
            kappa * (1.0e-5 + (f64::from(self.hardening_factor) * (-alpha_end).max(0.0)).sinh());
        let shift = beta * p0 + f64::from(cohesion_bonus_pa);
        let (p_tr, q_tr) = (f64::from(p_tr), f64::from(q_tr));
        let p_c = centre;
        let (dx, dy) = (p_c - p_tr, -q_tr);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 {
            return f64::INFINITY;
        }
        let (dx, dy) = (dx / len, dy / len);
        let c = m_sq * (p_c + shift) * (p_c - p0);
        let b = m_sq * dx * (2.0 * p_c + shift - p0);
        let a = m_sq * dx * dx + (1.0 + 2.0 * beta) * dy * dy;
        let discr = (b * b - 4.0 * a * c).max(0.0).sqrt();
        let p1 = p_c + (-b + discr) / (2.0 * a) * dx;
        let p2 = p_c + (-b - discr) / (2.0 * a) * dx;
        let p_x = if (p_tr - p_c) * (p1 - p_c) > 0.0 {
            p1
        } else {
            p2
        };
        let j_x = (1.0 - 2.0 * p_x / kappa).abs().max(1.0e-16).sqrt();
        if j_x > 1.0e-4 {
            (f64::from(j_tr) / j_x).ln()
        } else {
            f64::INFINITY
        }
    }

    /// End-of-step alpha on the wet side: the root of
    /// `alpha_end = alpha + ray_increment(alpha_end)` along a fixed ray.
    /// Hardened ellipses are nested, so along that ray the increment grows
    /// monotonically with hardening: the residual decreases strictly, the
    /// start-of-step answer `alpha + start` lies beyond the root, and the
    /// root is bracketed between it and `alpha`. Solved in f64 by the Illinois
    /// variant of false position, which keeps the bracket and converges in a
    /// few iterations.
    fn hardened_alpha(&self, alpha: f64, start: f64, increment: impl Fn(f64) -> f64) -> f64 {
        let residual = |a: f64| alpha + increment(a) - a;
        let (mut lo, mut hi) = (alpha + start, alpha);
        let (mut r_lo, mut r_hi) = (residual(lo), residual(hi));
        if !(r_lo > 0.0 && r_hi < 0.0) {
            return lo;
        }
        let mut kept_lo_last = None;
        for _ in 0..40 {
            let x = hi - r_hi * (hi - lo) / (r_hi - r_lo);
            let r = residual(x);
            if r > 0.0 {
                lo = x;
                r_lo = r;
                if kept_lo_last == Some(false) {
                    r_hi *= 0.5;
                }
                kept_lo_last = Some(false);
            } else {
                hi = x;
                r_hi = r;
                if kept_lo_last == Some(true) {
                    r_lo *= 0.5;
                }
                kept_lo_last = Some(true);
            }
            if r == 0.0 || hi - lo <= 1.0e-12 * alpha.abs().max(1.0e-6) {
                break;
            }
        }
        0.5 * (lo + hi)
    }

    /// Volume ratio J at which a trial state beyond the cap comes to rest,
    /// with the hardening evaluated at the end of the step (backward Euler,
    /// Simo & Hughes 1998, ch. 3): the carried pressure `kappa/2 (1 - J^2)`
    /// must equal `p0` hardened by this same step's plastic compaction
    /// `u = ln(J / j_tr)`. Hardening evaluated at the start of the step
    /// instead leaves p0 above the carried pressure by `(xi - 1)` times the
    /// overshoot, so the soil would remember a load it never carried.
    ///
    /// The residual decreases and is concave in `u`, and it is <= 0 at the
    /// old cap (`j_old_cap`, no hardening), so Newton started there converges
    /// monotonically to the root. Returns J and `u`; `u` comes straight from
    /// the f64 solve because `ln(J / j_tr)` of a ratio this close to 1 loses
    /// most of its digits in f32.
    fn cap_return(&self, j_tr: f32, j_old_cap: f32, alpha: f32) -> (f32, f32) {
        let kappa = f64::from(self.kappa);
        let xi = f64::from(self.hardening_factor);
        let (j_tr, alpha) = (f64::from(j_tr), f64::from(alpha));
        let mut u = (f64::from(j_old_cap) / j_tr).ln();
        if u <= 0.0 {
            return (j_old_cap, 0.0);
        }
        for _ in 0..30 {
            let j_sq = j_tr * j_tr * (2.0 * u).exp();
            let compaction = u - alpha;
            let (p0, dp0_du) = if compaction > 0.0 {
                (
                    kappa * (1.0e-5 + (xi * compaction).sinh()),
                    kappa * xi * (xi * compaction).cosh(),
                )
            } else {
                (kappa * 1.0e-5, 0.0)
            };
            let residual = 0.5 * kappa * (1.0 - j_sq) - p0;
            let step = residual / (-kappa * j_sq - dp0_du);
            u = (u - step).max(0.0);
            if step.abs() < 1.0e-12 {
                break;
            }
        }
        ((j_tr * u.exp()) as f32, u as f32)
    }
}

/// Real fix (2026-09-05): `NaccMaterial` previously had no dimensionally-
/// correct SI-conversion path at all (see `from_young_modulus`'s own doc,
/// which disclosed exactly this gap). `friction`/`cohesion`/
/// `hardening_factor` are yield-surface shape parameters, not stress-like
/// quantities -- passed through unconverted, matching `from_young_modulus`'s
/// own existing convention for the same three fields.
impl FromSI<NaccProps> for NaccMaterial {
    fn from_physical(props: &NaccProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        // Same 2D plane-strain bulk modulus relation as every other
        // constructor in this file (kappa = lambda + mu, NOT the 3D
        // lambda + 2*mu/3) -- applied here to the now-correctly-SI-scaled
        // lambda/mu, not the raw grid-unit ones `from_young_modulus` uses.
        let kappa = lambda + mu;
        Self::new(
            mu,
            kappa,
            props.friction,
            props.cohesion,
            props.hardening_factor,
        )
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

    /// Real apparent cohesion from soil suction -- see `saturation_cohesion_
    /// coeff`'s own doc for the mechanism, citation, and why this is the
    /// opposite saturation trend from `DruckerPragerMaterial`'s own capillary-
    /// bridging override, not a copy of it.
    fn cohesion_bonus_pa(&self, scalar_field: f32) -> f32 {
        if self.saturation_cohesion_coeff == 0.0 {
            return 0.0;
        }
        let saturation = scalar_field.clamp(0.0, 1.0);
        self.saturation_cohesion_coeff * (1.0 - saturation)
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
        let elastic = dev_stress + vol_stress;

        if self.elastic_viscosity == 0.0 {
            return elastic;
        }
        // Same Kelvin-Voigt dashpot formula as `RankineMaterial`/
        // `DruckerPragerMaterial`/`ViscoelasticMaterial`: tau_v = eta*D_dev
        // (NOT 2*eta*D_dev -- see `RankineMaterial::kirchhoff_stress`'s own
        // doc for why), D the symmetric part of the APIC velocity gradient.
        let c = particles.velocity_gradient[i];
        let sym = c + c.transpose();
        let d = sym * 0.5;
        let d_trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(d_trace * 0.5));
        elastic + self.elastic_viscosity * d_dev
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Elastic predictor -- same pattern as every other CPU plastic material
        // (`sand.rs`, `snow.rs`, `sand_mui.rs`): F must pick up this substep's
        // strain from the velocity gradient BEFORE plastic projection, or F never
        // advances and the material exerts a frozen, non-evolving stress -- e.g.
        // a falling body collapsing to zero height under gravity.
        // Exact constant-C integration prevents forward Euler's O(dt^2)
        // volume drift from being mistaken for permanent Cam-Clay cap
        // plasticity and accumulated preconsolidation history.
        let f_trial =
            deformation_increment_exp(dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        let alpha = *ctx.log_volume_strain;
        let (new_f, new_alpha) =
            self.project(f_trial, alpha, self.cohesion_bonus_pa(ctx.scalar_field));
        *ctx.deformation_gradient = new_f;
        *ctx.log_volume_strain = new_alpha;

        let j = new_f.determinant().max(MIN_J);
        let vol = ctx.initial_volume * j;
        *ctx.volume = vol;
        *ctx.density = if vol > MIN_J {
            ctx.mass / vol
        } else {
            *ctx.density
        };
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.log_volume_strain = 0.0; // nacc_alpha starts at 0 (unstressed)
    }

    fn needs_cpu_update(&self) -> bool {
        true
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "NaccMaterial has no GPU stress path: it uploads as NeoHookean, so the GPU runs kappa ln(J) instead of Cam-Clay's kappa/2 (J^2 - 1); use GranularFluidMaterial for a GPU granular-fluid scene",
        )
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        // Inverse of the 2D plane-strain relation kappa = lambda + mu.
        let lambda = self.kappa - self.mu;
        let elastic_dt = elastic_wave_dt(
            lambda,
            self.mu,
            hardening_scale,
            density,
            self.min_density,
            cell_width,
            material_cfl,
        );
        // Same explicit-viscous-diffusion stability bound `RankineMaterial`/
        // `DruckerPragerMaterial` already use for their own Kelvin-Voigt term --
        // without this, `elastic_viscosity` adds real stiffness the substep
        // selector never sees.
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

    fn rate_particle(f: Mat2, alpha: f32) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.log_volume_strain = alpha;
        Particles::from(vec![p])
    }

    fn run_rate_step(mat: &NaccMaterial, particles: &mut Particles, rate: Mat2, dt: f32) {
        let mut ctx = particles.update_ctx(0);
        *ctx.velocity_gradient = rate;
        mat.update_particle(&mut ctx, dt);
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

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

        let (f_after, _alpha_after) = mat.project(f, alpha, 0.0);
        let j_after = f_after.determinant();
        let p_after = pressure_from_j(mat.kappa, j_after);

        assert!(
            (p_after - p0).abs() < 1.0e-2,
            "compression-cap projection should land EXACTLY at p=p0={p0:.6}, got {p_after:.6}"
        );
    }

    #[test]
    fn rigid_rotation_preserves_volume_and_cam_clay_history() {
        let mat = NaccMaterial::soft_clay(5.0e4, 0.3);
        let mut particles = rate_particle(Mat2::IDENTITY, 0.0);
        let omega = 1.7;
        let dt = 0.2;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));

        run_rate_step(&mat, &mut particles, spin, dt);

        let expected = Mat2::from_angle(omega * dt);
        assert!(matrix_error(particles.deformation_gradient[0], expected) < 3.0e-6);
        assert!((particles.deformation_gradient[0].determinant() - 1.0).abs() < 2.0e-6);
        assert!(particles.log_volume_strain[0].abs() < 1.0e-6);
    }

    #[test]
    fn opposite_subcap_rates_are_reversible_without_alpha_ratchet() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 2.0);
        let alpha = -0.1;
        let baseline = Mat2::from_diagonal(Vec2::splat(0.95));
        let mut particles = rate_particle(baseline, alpha);
        let rate = Mat2::from_diagonal(Vec2::new(0.001, -0.001));

        run_rate_step(&mat, &mut particles, rate, 1.0);
        assert!((particles.log_volume_strain[0] - alpha).abs() < 1.0e-6);
        run_rate_step(&mat, &mut particles, -rate, 1.0);

        assert!(matrix_error(particles.deformation_gradient[0], baseline) < 3.0e-6);
        assert!((particles.log_volume_strain[0] - alpha).abs() < 1.0e-6);
    }

    #[test]
    fn exponential_trial_respects_both_sides_of_compression_cap() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 2.0);
        let alpha = -0.1;
        let p0 = mat.kappa * (1.0e-5 + (mat.hardening_factor * -alpha).sinh());
        let j_cap = (1.0 - 2.0 * p0 / mat.kappa).sqrt();
        let dt = 0.1;

        let j_inside = j_cap + 1.0e-4;
        let sigma_inside = j_inside.sqrt();
        let mut inside = rate_particle(Mat2::IDENTITY, alpha);
        let inside_rate = Mat2::from_diagonal(Vec2::splat(sigma_inside.ln() / dt));
        run_rate_step(&mat, &mut inside, inside_rate, dt);
        assert!((inside.deformation_gradient[0].determinant() - j_inside).abs() < 3.0e-6);
        assert!((inside.log_volume_strain[0] - alpha).abs() < 1.0e-6);

        let j_outside = j_cap - 1.0e-4;
        let sigma_outside = j_outside.sqrt();
        let mut outside = rate_particle(Mat2::IDENTITY, alpha);
        let outside_rate = Mat2::from_diagonal(Vec2::splat(sigma_outside.ln() / dt));
        run_rate_step(&mat, &mut outside, outside_rate, dt);
        // The cap hardens during the step, so the state comes to rest between
        // the old cap and the trial, exactly on the hardened cap.
        let j_after = outside.deformation_gradient[0].determinant();
        let alpha_after = outside.log_volume_strain[0];
        let p0_after = mat.kappa * (1.0e-5 + (mat.hardening_factor * -alpha_after).sinh());
        let p_after = pressure_from_j(mat.kappa, j_after);
        assert!(
            j_after < j_cap && j_after > j_outside - 3.0e-6,
            "trial beyond the cap must stop between the trial J={j_outside} and the old cap \
             J={j_cap}, got {j_after}"
        );
        assert!(
            (p_after - p0_after).abs() <= 1.0e-3 * p0_after,
            "the state must sit on the hardened cap: p={p_after} p0={p0_after}"
        );
        assert!(alpha_after < alpha);
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
        let (f_after, alpha_after) = mat.project(f, alpha, 0.0);
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

    /// On virgin isotropic loading the soil sits on its cap, so after every
    /// step its preconsolidation pressure must equal the pressure it carries:
    /// p0 remembers the largest load, never more.
    #[test]
    fn virgin_loading_keeps_p0_equal_to_the_carried_pressure() {
        for xi in [0.5_f32, 2.0, 27.8] {
            let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, xi);
            let (mut f, mut alpha) = (Mat2::IDENTITY, 0.0_f32);
            for step in 0..40 {
                f = Mat2::from_diagonal(Vec2::splat(0.9995)) * f;
                (f, alpha) = mat.project(f, alpha, 0.0);
                let p = pressure_from_j(mat.kappa, f.determinant());
                let p0 = mat.kappa * (1.0e-5 + (xi * (-alpha).max(0.0)).sinh());
                assert!(
                    (p0 - p).abs() <= 1.0e-3 * p,
                    "xi={xi} step {step}: p0={p0} must equal the carried pressure p={p}"
                );
            }
        }
    }

    /// Principal-stress summary `(p, |s|)` of a deformation gradient, with the
    /// same formulas `project` uses.
    fn p_and_shear(mat: &NaccMaterial, f: Mat2) -> (f32, f32) {
        let (_, sigma, _) = svd2(f);
        let sv_sq = Vec2::new(sigma.x * sigma.x, sigma.y * sigma.y);
        let j = sigma.x * sigma.y;
        let s = mat.mu / j * (sv_sq - Vec2::splat((sv_sq.x + sv_sq.y) * 0.5));
        (pressure_from_j(mat.kappa, j), s.length())
    }

    /// Preconsolidation pressure of the (beta = 0) ellipse that passes through
    /// the stress state `(p, |s|)`: solves `2 |s|^2 + M^2 p (p - p0) = 0`.
    fn p0_through(mat: &NaccMaterial, p: f32, shear: f32) -> f32 {
        p + 2.0 * shear * shear / (mat.friction * mat.friction * p)
    }

    /// Shear plus compression on the wet side of the ellipse: after the soil
    /// hardens, the stress must sit on the hardened surface. The projection
    /// keeps tr(B) rather than J, so even a fixed surface is missed slightly;
    /// hardening must add nothing beyond that.
    #[test]
    fn wet_side_shear_lands_on_the_hardened_surface() {
        for xi in [2.0_f32, 27.8] {
            let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, xi);
            let mut fixed = mat;
            fixed.hardening_enabled = false;
            for (a, b) in [(0.985_f32, 0.975_f32), (0.99, 0.965), (0.995, 0.96)] {
                let f = Mat2::from_diagonal(Vec2::new(a, b));
                let (p_tr, _) = p_and_shear(&mat, f);
                let p0_start = 1.3 * p_tr;
                let alpha = -((p0_start / mat.kappa - 1.0e-5).asinh()) / xi;

                let (f_after, alpha_after) = mat.project(f, alpha, 0.0);
                let (p, shear) = p_and_shear(&mat, f_after);
                let p0 = mat.kappa * (1.0e-5 + (xi * (-alpha_after).max(0.0)).sinh());
                let hardened_miss = (p0 / p0_through(&mat, p, shear) - 1.0).abs();

                let (f_fixed, _) = fixed.project(f, alpha, 0.0);
                let (pf, sf) = p_and_shear(&mat, f_fixed);
                let projection_miss = (p0_start / p0_through(&mat, pf, sf) - 1.0).abs();

                assert!(alpha_after < alpha, "the wet side must harden");
                assert!(
                    hardened_miss <= projection_miss + 1.0e-4,
                    "xi={xi} F=diag({a}, {b}): hardened surface missed by {hardened_miss}, \
                     the projection alone misses by {projection_miss}"
                );
            }
        }
    }

    /// Preconsolidation is a memory: unloading and reloading back to the
    /// previous maximum stays elastic, and the soil yields only beyond it.
    #[test]
    fn reloading_is_elastic_until_the_previous_maximum() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 2.0);
        let squeeze = Mat2::from_diagonal(Vec2::splat(0.9995));
        let release = Mat2::from_diagonal(Vec2::splat(0.9995_f32.recip()));
        let (mut f, mut alpha) = (Mat2::IDENTITY, 0.0_f32);
        for _ in 0..40 {
            (f, alpha) = mat.project(squeeze * f, alpha, 0.0);
        }
        let alpha_max_load = alpha;
        for _ in 0..20 {
            (f, alpha) = mat.project(release * f, alpha, 0.0);
        }
        for _ in 0..20 {
            (f, alpha) = mat.project(squeeze * f, alpha, 0.0);
        }
        assert!(
            (alpha - alpha_max_load).abs() < 1.0e-6,
            "unload-reload to the previous maximum must stay elastic: \
             alpha {alpha_max_load} -> {alpha}"
        );
        (f, alpha) = mat.project(squeeze * squeeze * f, alpha, 0.0);
        let p = pressure_from_j(mat.kappa, f.determinant());
        let p0 = mat.kappa * (1.0e-5 + (mat.hardening_factor * (-alpha).max(0.0)).sinh());
        assert!(
            alpha < alpha_max_load,
            "loading past the maximum must yield"
        );
        assert!((p0 - p).abs() <= 1.0e-3 * p, "p0={p0} p={p}");
    }

    /// Real behavioral distinctness, not just a different field value: after the
    /// SAME prior compaction history (same alpha, representing identical past
    /// loading), `peat` (hardening_factor=0.5, the lowest in this family) must
    /// have built up LESS compression-cap resistance (p0) than `wet_soil`
    /// (hardening_factor=3.0) -- p0's own formula (`kappa*(1e-5+sinh(xi*max(
    /// -alpha,0)))`) grows with xi at any fixed nonzero alpha, so a lower
    /// hardening_factor means less real preconsolidation resistance builds up per
    /// unit of past compaction, matching peat's own real, cited defining trait
    /// (extreme compressibility, Mesri & Ajlouni 2007). Note: at the neutral
    /// alpha=0 start state every preset's p0 is identical regardless of
    /// hardening_factor (sinh(xi*0)=0 for any xi) -- this only differentiates
    /// once real prior compaction (alpha != 0) has happened, so this test starts
    /// from alpha=-1.0, the same "real pre-consolidation" convention
    /// `small_elastic_strain_is_not_projected` above already uses.
    #[test]
    fn peat_hardens_slower_than_wet_soil_after_the_same_prior_compaction() {
        let peat = NaccMaterial::peat(3000.0, 0.3);
        let wet_soil = NaccMaterial::wet_soil(3000.0, 0.3);
        assert!(
            peat.hardening_factor < wet_soil.hardening_factor,
            "peat must have the lowest hardening_factor in this family: \
             peat={} wet_soil={}",
            peat.hardening_factor,
            wet_soil.hardening_factor
        );

        let alpha: f32 = -1.0; // same real prior compaction for both
        let p0 = |mat: &NaccMaterial| {
            mat.kappa * (1.0e-5 + (mat.hardening_factor * (-alpha).max(0.0)).sinh())
        };
        let peat_p0 = p0(&peat);
        let wet_soil_p0 = p0(&wet_soil);
        assert!(
            peat_p0 < wet_soil_p0,
            "peat should have built up LESS compression-cap resistance than \
             wet_soil after identical prior compaction: peat_p0={peat_p0} \
             wet_soil_p0={wet_soil_p0}"
        );
    }
}

#[cfg(test)]
mod elastic_viscosity_tests {
    use super::*;
    use crate::Particle;

    /// Same audit-closing test `RankineMaterial`/`CorotatedMaterial`/
    /// `VonMisesMaterial` all carry for their own copy of this identical
    /// mechanism (see `elastic_viscosity`'s own doc): `kirchhoff_stress`
    /// must actually respond to the particle's velocity gradient when
    /// `elastic_viscosity > 0.0`, not just carry the field.
    #[test]
    fn nonzero_elastic_viscosity_adds_a_real_viscous_stress_term() {
        let elastic_only = NaccMaterial::new(3000.0, 5000.0, 0.5, 0.0, 0.0);
        let mut damped = elastic_only;
        damped.elastic_viscosity = 50.0;

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
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

    /// `elastic_viscosity == 0.0` (every preset's default) must leave
    /// `kirchhoff_stress` completely blind to the velocity gradient -- a
    /// real regression guard against the early-return branch getting
    /// "simplified" away into an unconditional `elastic + 0.0*d_dev`.
    #[test]
    fn zero_elastic_viscosity_ignores_the_velocity_gradient() {
        let mat = NaccMaterial::new(3000.0, 5000.0, 0.5, 0.0, 0.0);
        assert_eq!(mat.elastic_viscosity, 0.0);

        let mut p_rest = Particle::zeroed();
        p_rest.deformation_gradient = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(0.01, 0.97));
        let mut p_shearing = p_rest;
        p_shearing.velocity_gradient = Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4));
        let particles = Particles::from(vec![p_rest, p_shearing]);

        let tau_rest = mat.kirchhoff_stress(&particles, 0);
        let tau_shearing = mat.kirchhoff_stress(&particles, 1);
        assert_eq!(
            tau_rest, tau_shearing,
            "elastic_viscosity=0.0 must be bit-identical regardless of velocity_gradient"
        );
    }
}

#[cfg(test)]
mod saturation_cohesion_tests {
    use super::*;

    /// `saturation_cohesion_coeff == 0.0` (every existing preset/scene) must
    /// keep `cohesion_bonus_pa` at exactly 0.0 for any saturation -- real
    /// regression guard that this second real material (see
    /// `saturation_cohesion_coeff`'s own doc) is genuinely inert by default,
    /// same convention `DruckerPragerMaterial`'s own hook uses.
    #[test]
    fn zero_coefficient_is_inert_at_any_saturation() {
        let mat = NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 0.0);
        assert_eq!(mat.saturation_cohesion_coeff, 0.0);
        for sr in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(mat.cohesion_bonus_pa(sr), 0.0);
        }
    }

    /// Real shape check, and the real point of difference from
    /// `DruckerPragerMaterial`'s own override (see `saturation_cohesion_
    /// coeff`'s own doc for why): apparent cohesion here is MAXIMAL at
    /// Sr=0 (dry, real suction) and EXACTLY zero at Sr=1 (saturated, zero
    /// suction) -- the opposite saturation trend from sand's own pendular-
    /// regime peak at low-but-nonzero saturation.
    #[test]
    fn cohesion_bonus_decreases_monotonically_with_saturation() {
        let mat = NaccMaterial {
            saturation_cohesion_coeff: 500.0,
            ..NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 0.0)
        };
        assert_eq!(
            mat.cohesion_bonus_pa(0.0),
            500.0,
            "must be maximal at Sr=0 (dry)"
        );
        assert_eq!(
            mat.cohesion_bonus_pa(1.0),
            0.0,
            "must be exactly zero at Sr=1 (saturated)"
        );
        let mid = mat.cohesion_bonus_pa(0.5);
        assert!(
            mid > 0.0 && mid < 500.0,
            "Sr=0.5 must land strictly between the two endpoints, got {mid}"
        );
        assert!(
            mat.cohesion_bonus_pa(0.25) > mat.cohesion_bonus_pa(0.75),
            "must decrease monotonically with saturation"
        );
    }

    /// The real, load-bearing proof this mechanism was added for -- not just
    /// that the formula looks plausible in isolation, but that it changes
    /// actual yield-surface behavior end to end through `project()`, same
    /// discipline `DruckerPragerMaterial`'s own wet/dry sand test uses. A
    /// small isotropic TENSION state (F=s*I, s slightly >1, real negative
    /// p_tr, zero shear) is right past the cohesionless (beta=0) material's
    /// own tensile limit (`p_tr < -beta*p0 = 0` whenever beta=0 -- a
    /// cohesionless material has genuinely zero tensile strength, real
    /// critical-state soil mechanics) -- WITHOUT real apparent cohesion
    /// (Sr=1, saturated) this must fail and get projected; WITH it (Sr=0,
    /// dry) the exact same trial state must hold elastically, unprojected.
    #[test]
    fn dry_apparent_cohesion_holds_a_tension_state_wet_cannot() {
        let mat = NaccMaterial {
            saturation_cohesion_coeff: 500.0,
            ..NaccMaterial::new(3000.0, 2000.0, 1.2, 0.0, 0.0)
        };
        let alpha = 0.0;
        let f = Mat2::from_diagonal(Vec2::splat(1.001)); // small isotropic tension

        let wet_bonus = mat.cohesion_bonus_pa(1.0);
        let dry_bonus = mat.cohesion_bonus_pa(0.0);
        assert_eq!(wet_bonus, 0.0);
        assert_eq!(dry_bonus, 500.0);

        let (f_wet, _) = mat.project(f, alpha, wet_bonus);
        let (f_dry, _) = mat.project(f, alpha, dry_bonus);

        assert_ne!(
            f_wet, f,
            "wet (no real apparent cohesion) must fail this tension state and get projected"
        );
        assert_eq!(
            f_dry, f,
            "dry (real apparent cohesion from suction) must hold this same tension state elastically"
        );
    }
}

#[cfg(test)]
mod from_si_tests {
    use super::*;
    use crate::matter::materials::physical_props::Elastic;
    use crate::solver::config::SimConfig;

    /// Real, dt-independent SI conversion -- the same closed-form check
    /// `lame_from_si_physical`'s own tests use: identical physical setup at
    /// two different `dt_seconds` must produce byte-identical grid mu/kappa,
    /// since a converted stiffness must never depend on the timestep.
    #[test]
    fn from_physical_is_dt_independent() {
        let props = NaccProps {
            elastic: Elastic {
                e_pa: 5.0e6,
                nu: 0.3,
                rho_kg_m3: 1600.0,
            },
            friction: 1.0,
            cohesion: 0.0,
            hardening_factor: 2.0,
        };
        let config_a = SimConfig::earth(64, 0.01, 0.1);
        let config_b = SimConfig::earth(64, 0.01, 0.001);
        let mat_a = NaccMaterial::from_physical(&props, &config_a);
        let mat_b = NaccMaterial::from_physical(&props, &config_b);
        assert_eq!(mat_a.mu, mat_b.mu, "mu must not depend on dt_seconds");
        assert_eq!(
            mat_a.kappa, mat_b.kappa,
            "kappa must not depend on dt_seconds"
        );
        assert_eq!(mat_a.friction, props.friction);
        assert_eq!(mat_a.cohesion, props.cohesion);
        assert_eq!(mat_a.hardening_factor, props.hardening_factor);
    }

    /// Real, direct check against the same `scale_lame` + plane-strain
    /// `kappa=lambda+mu` relation this impl documents using -- not just an
    /// opaque "it doesn't crash" test.
    #[test]
    fn from_physical_matches_scale_lame_plus_plane_strain_kappa() {
        let props = NaccProps {
            elastic: Elastic {
                e_pa: 1.0e6,
                nu: 0.25,
                rho_kg_m3: 1000.0,
            },
            friction: 1.2,
            cohesion: 0.0,
            hardening_factor: 3.0,
        };
        let config = SimConfig::earth(64, 0.01, 0.05);
        let mat = NaccMaterial::from_physical(&props, &config);
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            &config,
        );
        assert_eq!(mat.mu, mu);
        assert_eq!(mat.kappa, lambda + mu);
    }
}

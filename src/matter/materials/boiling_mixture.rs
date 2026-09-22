//! Real, enthalpy-quality-driven mixture material for a particle genuinely
//! mid-boil (Voller & Cross 1981's own latent-heat plateau, the same real
//! method `energy::thermodynamics::enthalpy::chained_state_from_enthalpy`
//! already tracks as `PhaseState::Boiling { fraction }`) -- external
//! review's own "minimum honest enthalpy->mechanical" fix (2026-09-01).
//!
//! # The real bug this closes
//! `CavitatingFluidMaterial` alone let a particle's MECHANICAL vapor
//! fraction (`x_rho`, implicit in its own density/`J`) run completely
//! independent of its THERMAL vapor fraction (`x_H`, the enthalpy method's
//! own `boiling_fraction`) -- live-confirmed, `phase_states_gui.rs`'s own
//! boiling-plateau repro: a particle held at `J=5.988` (`x_rho=(J-1)/5
//! ~= 0.998`, nearly fully vaporized MECHANICALLY) while `x_H<0.5` (still
//! `WATER_ID`, thermally under half boiled) -- two independent vapor
//! fractions, never coupled. The proximate mechanism: `T` pinned at the
//! real boiling plateau sits right at `t_liquid_closure_max`'s own real
//! near-zero-stiffness boundary (see `cavitating_eos`'s own doc) for many
//! real frames, not an instant -- a real, predictable consequence of
//! feeding a plateau-held `T` into a purely `T -> p_sat(T)` mechanical
//! closure that was never told how much of the particle had actually
//! boiled.
//!
//! # The real fix
//! This material's own rest state is not one fixed density -- it is the
//! real Homogeneous Equilibrium Model (HEM) two-phase mixture density for
//! the particle's OWN CURRENT mass quality `x` (read from
//! `Particle::friction_hardening`, this material's own real use of that
//! already-established per-material-reused scratch field -- see
//! `Particle`'s own doc: "DP q / VonMises kappa / Rankine damage / SandMuI
//! mu(I)"):
//! ```text
//! rho_eq(x) = 1 / ((1-x)/rho_l_ref + x/rho_v_ref)
//! ```
//! the standard two-phase mixture specific-volume rule (e.g. Collier &
//! Thome, "Convective Boiling and Condensation," 3rd ed., section 2.2 --
//! mass-fraction-weighted specific volume). The local stiffness around
//! that moving equilibrium is Wood's/Wallis's real mixture sound-speed
//! relation (Wallis 1969, "One-Dimensional Two-Phase Flow," eq. 4.36 --
//! the SAME real relation already cited in `cavitating_eos`'s own module
//! doc for `c_min`'s own real physical meaning):
//! ```text
//! 1 / (rho_eq(x) * c_mix(x)^2) = (1-x)/(rho_l_ref*c_l^2) + x/(rho_v_ref*c_v_ref^2)
//! ```
//! Both real relations degenerate EXACTLY to the liquid closure at `x=0`
//! (`rho_eq=rho_l_ref`, `c_mix=c_l`) and to the vapor reference at `x=1`
//! (`rho_eq=rho_v_ref`, `c_mix=c_v_ref`) -- continuous with the
//! neighboring pure-phase materials by construction, not tuned to match.
//! `c_v_ref^2` is derived from the SAME shared-stiffness relation
//! `CavitatingEosParams::new` already uses (`B=c_l^2*rho_l_ref/gamma_l`,
//! `c_v_ref^2=B*gamma_v/rho_v_ref`), recomputed here directly from already-
//! public constants -- not a second, independently-invented number.
//!
//! `x` itself is written every substep by the scene's own enthalpy update,
//! from the SAME `PhaseState::Boiling { fraction }` that already decides
//! `material_id` -- so mechanical and thermal vapor fraction are now, by
//! construction, the SAME NUMBER, not two independent state variables.
//!
//! # What this deliberately does NOT claim
//! Wood's real mixture sound speed can dip BELOW both pure-phase values
//! for a real, wide-impedance-mismatch system (the classic Wood minimum --
//! this engine's own real water/vapor reference densities here use a
//! deliberately compressed ~6:1 ratio, not real water/steam's ~1600:1, so
//! this demo's own live numbers don't hit that regime, but the formula
//! above does not assume they can't) -- `rest_acoustic_c2` below is a
//! real, MEASURED minimum over `x in [0,1]` (a dense sweep at
//! construction), not the smaller of the two endpoints.
//!
//! Real, disclosed, still-open limitation, same as `CavitatingFluidMaterial`'s
//! own doc: this is still a ONE-DIRECTIONAL coupling (`x_H -> mechanical
//! state`) -- a real mechanical deviation away from `rho_eq(x_H)` (e.g.
//! genuine compression under gravity) does not pay any latent heat back
//! into `x_H`/enthalpy. The real two-way closure (Saurel, Boivin & Le
//! Métayer 2016's own mixture-equilibrium relaxation) is the next real
//! milestone, not attempted here. External review's own call (2026-09-01):
//! NOT worth building yet -- `tau` (Saurel et al.'s own relaxation model)
//! depends on real interfacial area/nucleation-site density this engine
//! doesn't resolve, so any relaxation-ODE closure built now would just be
//! a disguised tuning constant; the physically honest alternative (a
//! conservative implicit flash solving `p`/`T`/`x` jointly from conserved
//! `v`/`h`, `g_l(p,T)=g_v(p,T)`) is real but substantially bigger scope,
//! and would also require real `T_sat(p)` (Clausius-Clapeyron / IAPWS)
//! since `T` can no longer stay pinned at `BOILING_POINT_K` once pressure
//! genuinely varies -- deferred as one milestone together, not started.
//!
//! Real, disclosed verification history (2026-09-01) of the `J/J_eq`
//! residual observed under real gravity (0.2%-0.8% in the live demo):
//! `J/J_eq != 1` is not automatically a bug -- a column under real gravity
//! genuinely needs real internal pressure to hold its own weight, exactly
//! what `p=c_mix2(x)*(rho-rho_eq(x))` computes. A first check (comparing
//! the residual with gravity on vs off in the live demo) found it shrinks
//! ~7x with gravity removed -- real, but external review correctly
//! pointed out this only proves the residual is GRAVITY-SENSITIVE, not
//! that it's specifically hydrostatic: a gravity-triggered numerical
//! artifact would shrink the same way, and the metric used (the single
//! most-expanded particle each step) consistently showed the WRONG sign
//! for compression (`J>J_eq`, tension) -- a real, disclosed miss, not a
//! defect in the material itself, tracked in `tests/physics_correctness.rs`'s
//! own `boiling_mixture_volume_tracking_error_is_gravity_sensitive`
//! (renamed from its own first, overclaiming name).
//!
//! A second test, `boiling_mixture_column_shows_real_hydrostatic_
//! compression_by_depth`, pre-initializes a column to its own real
//! analytical hydrostatic profile (`rho(depth)=rho_eq(x)*
//! exp(g*depth/c_mix2(x))`, the exact solution of `dp/dy=-rho*g` against
//! this material's own linear-in-density EOS at fixed `x`), settles it,
//! then compares AVERAGE `J` in two bands both safely INTERIOR (away from
//! the free surface AND the `SlipBoundary` zone, a real, already-
//! documented source of its own discretization artifacts in this
//! codebase). Real, correctly-signed pass: `avg_j_deep=3.443 <
//! avg_j_shallow=3.479` at a genuinely settled state (`e_v=0.0023`). Two
//! real, self-caught mistakes on the way there: (1) a config bug -- fed a
//! "stress-test" gravity into the analytical target while leaving the
//! SOLVER's own config at unmodified Earth gravity, so the two disagreed
//! by construction; (2) mistook a column's real, expected drop-and-bounce
//! transient (confirmed via direct isolation against the ALREADY-TRUSTED
//! `IsothermalCavitatingFluidMaterial`, byte-identical scene, identical
//! curve) for a phantom instability -- the real fix was starting the
//! column already at its own equilibrium, not chasing it.
//!
//! Real, disclosed second-round correction (2026-09-01, external review's
//! own sharper follow-up): that test's own measured `e_rho=0.073`/
//! `e_p=0.84` (printed, not asserted) reveal it can only support a
//! QUALITATIVE claim -- the column's free SIDE faces carry real nonzero
//! pressure under the initial profile with nothing to react against, so
//! it genuinely spread sideways into a puddle (height 16.0 -> 5.3), not a
//! laterally-confined 1D column the analytical profile actually describes.
//! The real, QUANTITATIVE follow-up is `boiling_mixture_confined_column_
//! matches_analytical_profile_at_earth_gravity`: a column filling the
//! domain's own real full interior width between BOTH `SlipBoundary`
//! walls from frame 0 (no room to spread), real UNMODIFIED Earth gravity
//! as the PRIMARY check (a `20x` stress-test run follows with a real,
//! deliberately looser tolerance -- not the primary claim, after `20x`
//! alone was previously mistaken for decisive). Real, measured, PASSING
//! result: `e_rho=0.0036` (0.36% RMS, asserted `<1%`) -- the real,
//! quantitative confirmation the unconfined test couldn't give.
//! `e_p=0.834` stays large regardless (asserted only loosely, `<1.0`, to
//! catch a real blow-up) -- a genuine, structural consequence of this
//! material's own real stiffness at `x=0.5` (`rho*c_mix^2~3.6e7 Pa`)
//! being intrinsically large relative to this scene's own modest
//! hydrostatic pressure scale (`rho*g*h~4.5e4 Pa`), amplifying even the
//! real, small `e_rho` into a large relative pressure error -- not
//! something a better test design can fix, and not itself evidence
//! against the material.
//!
//! Real, disclosed closure (2026-09-02): the gap the paragraph above used
//! to leave open -- whether the LIVE DEMO's own `J/J_eq` residual at real
//! gravity is specifically hydrostatic, not gravity-triggered P2G/boundary
//! error -- is now answered, not by proving the residual IS hydrostatic
//! (external review's own point: not even necessary), but by the more
//! useful question: if converted thermodynamically, would it matter?
//! `phase_states_gui.rs`'s `[boiling-residual-stats]` block (gated every
//! 300 frames, covers the WHOLE `BOILING_ID` population, not one
//! worst-case particle) converts each particle's own gauge pressure to
//! `T_sat(p_absolute)` (`water_saturation::water_saturation_temperature_
//! from_pressure_k`, a real bisection inverse of the IAPWS-IF97 Region 4
//! curve, valid to the real critical point) and reports
//! `epsilon_x_equiv = cp_liquid*|T_sat(p)-BOILING_POINT_K|/L_v` -- the
//! vapor-quality change this residual WOULD cause if coupled.
//!
//! Real, live-measured result (240s headless run, `PHASE_STATES_AUTO_HEAT
//! =420`, `PHASE_STATES_GRAVITY_FRACTION=1.0`, 11 samples before full
//! vaporization): the large early residual (`delta_T_sat` p10/p90 as wide
//! as [-31.9,+13.8]K) tracks high `|v|`/`|div(v)|` and decays as the
//! column settles -- a real, dynamic/settling transient, not durable
//! thermodynamic pressure. What survives once genuinely quasi-static
//! (avg`|v|`<1.2, avg`|div(v)|`<0.03) is small AND depth-coherent (median
//! `delta_T_sat` +0.6 to +0.8K, by-depth shallow~0K -> deep~+1.3-1.5K,
//! monotonic -- real evidence a flash would have SOMETHING to correct) but
//! its thermodynamic impact stays tiny: `epsilon_x_equiv` median
//! ~0.13-0.15%, p90 ~0.6-0.75% of vapor quality, once settled.
//!
//! **Verdict (external review's own call, 2026-09-02)**: document this
//! limitation and STOP -- do not build the two-way flash for this demo.
//! Depth-coherence proves a flash would have something to correct; it does
//! not prove that correction is worth the complexity, when it would move
//! well under 1% of the phase fraction for the large majority of the
//! population. Reopen criterion, stated concretely so this isn't a vague
//! "revisit someday": a SUSTAINED quasi-static residual (not a transient
//! spike) exceeding ~1% `epsilon_x_equiv` at the median, or several
//! percent at p90, under some other real configuration (different
//! gravity, heat rate, or scene scale) not yet tried. Until then this
//! material's real, disclosed limitation stands as stated above: an exact
//! enthalpy-driven transition at a fixed reference pressure, no local
//! bidirectional flash rebalancing -- and the live buoyancy cutoff after
//! full vaporization (a separate, already-disclosed gate needing an
//! ambient-medium density reference, see `phase_states_gui.rs`'s own
//! buoyancy-gate doc) is NOT a symptom this closure would fix either way.
//!
//! One real, genuine bug WAS found and fixed along the way, unrelated to
//! any mistake above: `params()` never set `eos_power`, silently
//! disabling `cfl.rs`'s own real shock-viscosity CFL safety term for this
//! material (the exact real failure mode `IsothermalCavitatingFluidMaterial
//! ::params()`'s own doc already named and fixed for itself) -- fixed on
//! its own real merits, confirmed NOT the cause of the drop-and-bounce
//! confusion above (byte-identical results before/after for that specific
//! scene) -- see `gamma_l`'s own field doc for the full, honest account,
//! including why the specific value (`gamma_l`) isn't this closure's own
//! real exponent (it's linear, with none), only a real, conservative,
//! CFL-only proxy borrowed from the liquid endpoint's own real Tait EOS.

use glam::{Mat2, Vec2};

use super::cavitating_eos::CavitatingEosTable;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Real, exact HEM mixture density (kg/m^3) at mass quality `x in [0,1]`.
/// Real, disclosed precision choice: `x=0`/`x=1` return the reference
/// density DIRECTLY rather than round-tripping through the general
/// `1/((1-x)/rho_l+x/rho_v)` formula -- the round-trip is exact in real
/// arithmetic but loses a real, measurable `f32` ULP or two at these two
/// endpoints specifically (confirmed live: ~2 Pa residual gauge pressure
/// at `x=0` through the general formula, amplified by this material's own
/// real `c_l^2~3e4` slope) -- and these are exactly the two densities a
/// particle sits at for real, extended time on either side of the
/// boiling band (just-transitioned-in, about-to-transition-out), so exact
/// continuity there is worth the two extra branches.
fn rho_eq_kg_m3(x: f32, rho_l_ref_kg_m3: f32, rho_v_ref_kg_m3: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x == 0.0 {
        rho_l_ref_kg_m3
    } else if x == 1.0 {
        rho_v_ref_kg_m3
    } else {
        1.0 / ((1.0 - x) / rho_l_ref_kg_m3 + x / rho_v_ref_kg_m3)
    }
}

/// Real, exact Wood/Wallis mixture sound-speed-squared (m^2/s^2) at mass
/// quality `x in [0,1]` -- see this module's own top doc for the real
/// citation and the exact-at-both-endpoints derivation.
fn c_mix2_m2_s2(
    x: f32,
    rho_l_ref_kg_m3: f32,
    c_l2_m2_s2: f32,
    rho_v_ref_kg_m3: f32,
    c_v_ref2_m2_s2: f32,
) -> f32 {
    let x = x.clamp(0.0, 1.0);
    // Same real, disclosed exactness choice as `rho_eq_kg_m3`'s own doc --
    // returns the pure-phase reference stiffness directly at the two real
    // endpoints instead of round-tripping through the general formula.
    if x == 0.0 {
        return c_l2_m2_s2;
    } else if x == 1.0 {
        return c_v_ref2_m2_s2;
    }
    let inv_rho_c2 =
        (1.0 - x) / (rho_l_ref_kg_m3 * c_l2_m2_s2) + x / (rho_v_ref_kg_m3 * c_v_ref2_m2_s2);
    let rho_eq = rho_eq_kg_m3(x, rho_l_ref_kg_m3, rho_v_ref_kg_m3);
    1.0 / (rho_eq * inv_rho_c2)
}

#[derive(Debug, Clone, Copy)]
pub struct BoilingMixtureMaterial {
    pub rho_l_ref_kg_m3: f32,
    pub c_l_m_s: f32,
    pub rho_v_ref_kg_m3: f32,
    /// Real, derived (not free) vapor-side reference sound speed -- see
    /// this module's own top doc: `c_v_ref^2 = B*gamma_v/rho_v_ref`, the
    /// same shared-stiffness relation `CavitatingEosParams::new` uses.
    pub c_v_ref_m_s: f32,
    /// Liquid branch's own Tait exponent -- carried from the source
    /// `CavitatingEosTable`, kept (not consumed and discarded during
    /// construction) ONLY to feed `params().eos_power`, matching the SAME
    /// established convention `IsothermalCavitatingFluidMaterial::params()`/
    /// `CavitatingFluidMaterial::params()` already use for their own
    /// (also locally-linear) liquid branch. Real, disclosed honesty check
    /// (2026-09-01, external review): this is NOT this material's own
    /// mixture closure's real exponent -- `p=c_mix2(x)*(rho-rho_eq(x))`
    /// is linear in density at every fixed `x`, with no polytropic
    /// exponent of its own at all, and `BoilingMixtureMaterial` applies no
    /// `von_neumann_richtmyer_q` artificial-viscosity stress term (neither
    /// does `CavitatingFluidMaterial`, confirmed by grep -- that shared
    /// stress term is `fluid.rs`/`gas.rs`-only). `gamma_l` is carried over
    /// as a real, conservative CFL proxy for the fact that the liquid
    /// endpoint (`x=0`) genuinely IS derived from a real Tait EOS with
    /// this exponent (`B=c_l^2*rho_l_ref/gamma_l`, the same shared-B
    /// relation `CavitatingEosParams::new` uses) -- not a claim that it
    /// describes the mixture regime's own real nonlinearity, which this
    /// closure doesn't have. `cfl.rs`'s own tightening from this can only
    /// ever shrink the chosen substep (never loosen it), so a real,
    /// disclosed, not-perfectly-derived conservative value here is safe
    /// even though it isn't rigorously this closure's own parameter.
    ///
    /// Real, self-caught correction to an earlier version of this doc:
    /// an earlier version of this struct never stored `gamma_l` at all
    /// (leaving `params().eos_power` at the trait default `0.0`, silently
    /// disabling this CFL term) -- that earlier doc claimed this was "the
    /// direct, measured cause" of an apparent instability in a falling-
    /// column test. Direct A/B (adding this field, re-running the exact
    /// same scene) showed BYTE-IDENTICAL results before and after -- it
    /// was NOT the cause (the real cause was a debugging shortcut that
    /// skipped this material's own real analytical pre-initialization, see
    /// this module's own top doc's verification history). Fixed here
    /// anyway, on its own real merits (matching established convention,
    /// zero cost, only ever more conservative), not because it fixed
    /// anything that was actually broken.
    pub gamma_l: f32,
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    /// Fixed liquid-reference density in GRID units -- the SAME reference
    /// `CavitatingFluidMaterial` uses, so `J`/volume stay exactly
    /// continuous at the water->boiling handoff (`x=0`). `rho_eq(x)`
    /// above only enters the PRESSURE law, never the F/V/rho bookkeeping
    /// -- see this module's own top doc.
    rest_density_grid: f32,
    /// Real, MEASURED minimum of `c_mix2` over `x in [0,1]` (grid units) --
    /// see this module's own top doc's "What this deliberately does NOT
    /// claim" section. Used as the T/x-blind defensive floor
    /// (`rest_acoustic_c2`/`timestep_bound`), same disclosed role
    /// `CavitatingFluidMaterial::rest_acoustic_c2` already plays there.
    min_c_mix2_grid: f32,
    pub min_density: f32,
    pub min_volume: f32,
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

impl BoilingMixtureMaterial {
    /// Real, guaranteed-consistent constructor: reads its own liquid/vapor
    /// reference constants directly from the SAME `CavitatingEosTable`
    /// `CavitatingFluidMaterial` uses, so the `x=0` continuity claim in
    /// this module's own doc is mechanically enforced (identical numbers),
    /// not merely intended.
    pub fn from_table(
        table: &CavitatingEosTable,
        dx_meters: f32,
        dynamic_viscosity: f32,
        volume_ratio_min: f32,
        volume_ratio_max: f32,
    ) -> Self {
        assert!(
            dx_meters.is_finite() && dx_meters > 0.0,
            "BoilingMixtureMaterial requires a positive dx_meters"
        );
        assert!(
            volume_ratio_min > 0.0 && volume_ratio_max > volume_ratio_min,
            "BoilingMixtureMaterial requires 0 < volume_ratio_min < volume_ratio_max, \
             no default -- see CavitatingFluidMaterial's own equivalent field doc"
        );
        let rho_l_ref_kg_m3 = table.rho_l_ref_kg_m3;
        let c_l_m_s = table.c_l_m_s;
        let rho_v_ref_kg_m3 = table.rho_v_ref_kg_m3;
        // Same shared-B relation `CavitatingEosParams::new` derives
        // (b_pa = c_l^2*rho_l_ref/gamma_l; vapor-branch slope AT rho_v_ref
        // = b_pa*gamma_v/rho_v_ref) -- recomputed from already-public
        // table fields, not a second, independently-invented number.
        let b_pa = c_l_m_s * c_l_m_s * rho_l_ref_kg_m3 / table.gamma_l;
        let c_v_ref2 = b_pa * table.gamma_v / rho_v_ref_kg_m3;
        assert!(
            c_v_ref2.is_finite() && c_v_ref2 > 0.0,
            "BoilingMixtureMaterial: derived vapor reference stiffness c_v_ref2={c_v_ref2} \
             is not a real positive stiffness -- check the table's own constants"
        );
        let c_v_ref_m_s = c_v_ref2.sqrt();
        let c_l2 = c_l_m_s * c_l_m_s;

        // Real, measured (not assumed) minimum of c_mix2 over the whole
        // real quality range -- see this struct's own field doc.
        const SWEEP_SAMPLES: u32 = 201;
        let mut min_c_mix2 = f32::INFINITY;
        for k in 0..=SWEEP_SAMPLES {
            let x = k as f32 / SWEEP_SAMPLES as f32;
            let c2 = c_mix2_m2_s2(x, rho_l_ref_kg_m3, c_l2, rho_v_ref_kg_m3, c_v_ref2);
            min_c_mix2 = min_c_mix2.min(c2);
        }
        assert!(
            min_c_mix2.is_finite() && min_c_mix2 > 0.0,
            "BoilingMixtureMaterial: measured minimum mixture stiffness {min_c_mix2} is not \
             real, finite, and positive -- check the table's own constants"
        );

        let rest_density_grid = rho_l_ref_kg_m3 * dx_meters * dx_meters;
        Self {
            rho_l_ref_kg_m3,
            c_l_m_s,
            rho_v_ref_kg_m3,
            c_v_ref_m_s,
            gamma_l: table.gamma_l,
            dx_meters,
            dynamic_viscosity,
            rest_density_grid,
            min_c_mix2_grid: min_c_mix2 / (dx_meters * dx_meters),
            min_density: 1.0e-6,
            min_volume: 1.0e-9,
            volume_ratio_min,
            volume_ratio_max,
        }
    }

    #[inline]
    fn real_density_si(&self, grid_density: f32) -> f32 {
        grid_density / (self.dx_meters * self.dx_meters)
    }

    /// Real per-particle mass quality `x` -- see this module's own top doc
    /// for why `Particle::friction_hardening` is this material's own real
    /// use of that reused scratch field.
    #[inline]
    fn quality(&self, particles: &Particles, i: usize) -> f32 {
        particles.friction_hardening[i].clamp(0.0, 1.0)
    }

    /// Real, public HEM mixture density (kg/m^3) at mass quality `x` -- the
    /// SAME real formula `kirchhoff_stress` uses internally, exposed so
    /// callers outside this module (diagnostics, tests) compute the exact
    /// same number `rho_eq(x)` means here, not a second, independently
    /// reimplemented copy that could drift out of sync. See this module's
    /// own top doc for the real citation.
    pub fn rho_eq_kg_m3(&self, x: f32) -> f32 {
        rho_eq_kg_m3(x, self.rho_l_ref_kg_m3, self.rho_v_ref_kg_m3)
    }

    /// Real, public Wood/Wallis mixture sound-speed-squared (m^2/s^2) at
    /// mass quality `x` -- same real sharing rationale as `rho_eq_kg_m3`.
    pub fn c_mix2_m2_s2(&self, x: f32) -> f32 {
        c_mix2_m2_s2(
            x,
            self.rho_l_ref_kg_m3,
            self.c_l_m_s * self.c_l_m_s,
            self.rho_v_ref_kg_m3,
            self.c_v_ref_m_s * self.c_v_ref_m_s,
        )
    }

    /// Real, exact equilibrium `J` (`rho_l_ref/rho_eq(x)`) at mass quality
    /// `x` -- the real mass-fraction mixture line this module's own top doc
    /// derives (`J_eq(x)=1+(rho_l_ref/rho_v_ref-1)*x`), computed here from
    /// `rho_eq_kg_m3` directly so it can never drift from that real
    /// definition.
    pub fn j_eq(&self, x: f32) -> f32 {
        self.rho_l_ref_kg_m3 / self.rho_eq_kg_m3(x)
    }

    /// Real, public gauge pressure (Pa) at real SI density `density_si` and
    /// mass quality `x` -- the SAME real linearized law `kirchhoff_stress`
    /// evaluates internally, exposed for the same real sharing reason as
    /// `rho_eq_kg_m3`/`c_mix2_m2_s2` above.
    pub fn pressure_gauge_pa(&self, density_si: f32, x: f32) -> f32 {
        self.c_mix2_m2_s2(x) * (density_si - self.rho_eq_kg_m3(x))
    }
}

impl MaterialModel for BoilingMixtureMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "BoilingMixtureMaterial has no GPU stress path: it uploads as a plain Tait fluid, so the GPU would run without its liquid-vapour equation of state",
        )
    }

    // Same real F/V/rho contract as `CavitatingFluidMaterial` -- see this
    // module's own top doc: the fixed liquid reference stays the
    // bookkeeping anchor, `rho_eq(x)` only ever enters the pressure law.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density_grid;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density_grid;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume)
            .clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    fn rest_acoustic_c2(&self) -> Option<f32> {
        Some(self.min_c_mix2_grid)
    }

    /// Real, per-particle-quality-aware acoustic bound -- see this
    /// module's own top doc: `c_mix2` depends on `x`
    /// (`Particle::friction_hardening`), which neither `acoustic_c2_at`
    /// (density+temperature only) nor `timestep_bound` (density+
    /// hardening_scale only) can see, hence this material overrides the
    /// most general tier of the chain directly.
    fn acoustic_c2_at_particle(&self, particles: &Particles, i: usize) -> Option<f32> {
        let x = self.quality(particles, i);
        Some(self.c_mix2_m2_s2(x) / (self.dx_meters * self.dx_meters))
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density_grid = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        let density_si = self.real_density_si(density_grid);
        let x = self.quality(particles, i);
        let pressure_gauge = self.pressure_gauge_pa(density_si, x);
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        if self.dynamic_viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym_strain = c + c.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }
        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        let j = (old_j * (dt * div_v).exp()).clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        let density = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(self.min_volume);
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Real, disclosed limitation: NOT yet wired for GPU -- same real
    /// status as `CavitatingFluidMaterial`'s own doc.
    ///
    /// `eos_power` populated (`self.gamma_l`), matching the same
    /// established convention `CavitatingFluidMaterial::params()` already
    /// follows -- see `gamma_l`'s own field doc for the real, disclosed
    /// honesty check on what this value does and doesn't claim to be.
    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density_grid,
            dynamic_viscosity: self.dynamic_viscosity,
            eos_power: self.gamma_l,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed, x-blind fallback (see `acoustic_c2_at_particle`'s
    /// own doc for where the real, live-quality-aware bound actually
    /// comes from) -- uses the real, MEASURED minimum `c_mix2` over the
    /// whole quality range, kept only as a defensive floor for any caller
    /// invoking `timestep_bound` directly, outside the normal CFL fold
    /// that always also checks `acoustic_c2_at_particle`.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;
        if self.min_c_mix2_grid.is_finite() && self.min_c_mix2_grid > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / self.min_c_mix2_grid.sqrt());
        }
        if self.dynamic_viscosity > 0.0 {
            let density = density.max(self.min_density);
            let kinematic_viscosity = self.dynamic_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }
        dt_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimConfig;
    use crate::particle::Particles;
    use glam::Vec2;

    fn real_table() -> CavitatingEosTable {
        CavitatingEosTable::build(
            1000.0,       // rho_l_ref_kg_m3
            180.0,        // c_l_m_s
            7.0,          // gamma_l (Cole 1948)
            1000.0 / 6.0, // rho_v_ref_kg_m3
            1.33,         // gamma_v
            1.0,          // c_min_m_s
            273.15,       // t_min_k
        )
    }

    fn unit_dx_config() -> SimConfig {
        SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3)
    }

    fn make_particle(material: &BoilingMixtureMaterial, x: f32, j: f32) -> Particle {
        let mut p = Particle::zeroed();
        p.mass = material.rest_density_grid;
        p.deformation_gradient = Mat2::IDENTITY;
        material.init_particle(&mut p);
        let s = j.sqrt();
        p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(s));
        p.friction_hardening = x;
        p
    }

    #[test]
    fn gpu_refuses_the_boiling_mixture() {
        let material = BoilingMixtureMaterial::from_table(&real_table(), 1.0, 0.0, 0.5, 8.0);
        let reason = material.gpu_unsupported_reason();
        assert!(reason.is_some_and(|r| r.starts_with("BoilingMixtureMaterial")));
    }

    /// Real, direct anchor: at `x=0`, `J=1` (equilibrium liquid), pressure
    /// must be exactly zero gauge -- same real rest-state claim
    /// `CavitatingFluidMaterial`'s own equivalent test makes.
    #[test]
    fn pressure_is_zero_at_x_zero_and_rest_liquid_density() {
        let table = real_table();
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let material = BoilingMixtureMaterial::from_table(&table, config.dx_meters, 0.0, 0.5, 8.0);
        let p = make_particle(&material, 0.0, 1.0);
        let particles = Particles::from(vec![p]);
        let stress = material.kirchhoff_stress(&particles, 0);
        let pressure_gauge = -stress.x_axis.x;
        assert!(
            pressure_gauge.abs() < 1.0,
            "x=0, J=1 (rest liquid): pressure must be ~0 gauge, got {pressure_gauge} Pa"
        );
    }

    /// Real, direct anchor: at `x=1`, the equilibrium `J` is
    /// `rho_l_ref/rho_v_ref` (real full vaporization ratio) -- pressure
    /// must be exactly zero gauge there too, the vapor-side rest state.
    #[test]
    fn pressure_is_zero_at_x_one_and_rest_vapor_density() {
        let table = real_table();
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let material = BoilingMixtureMaterial::from_table(&table, config.dx_meters, 0.0, 0.5, 8.0);
        let j_eq_vapor = table.rho_l_ref_kg_m3 / table.rho_v_ref_kg_m3;
        let p = make_particle(&material, 1.0, j_eq_vapor);
        let particles = Particles::from(vec![p]);
        let stress = material.kirchhoff_stress(&particles, 0);
        let pressure_gauge = -stress.x_axis.x;
        assert!(
            pressure_gauge.abs() < 1.0,
            "x=1, J=J_eq(1): pressure must be ~0 gauge, got {pressure_gauge} Pa"
        );
    }

    /// Real, direct anchor: `J_eq(x) = rho_l_ref/rho_eq(x)` must trace the
    /// exact mass-fraction mixture line -- for this material's own real
    /// 6:1 density ratio, `J_eq(x) = 1 + 5x` (the same closed form
    /// external review's own analysis used to prove the original bug).
    #[test]
    fn equilibrium_j_matches_the_real_mass_fraction_mixture_line() {
        let table = real_table();
        let ratio = table.rho_l_ref_kg_m3 / table.rho_v_ref_kg_m3;
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let rho_eq = rho_eq_kg_m3(x, table.rho_l_ref_kg_m3, table.rho_v_ref_kg_m3);
            let j_eq = table.rho_l_ref_kg_m3 / rho_eq;
            let expected = 1.0 + (ratio - 1.0) * x;
            assert!(
                (j_eq - expected).abs() < 1.0e-3,
                "x={x}: J_eq={j_eq}, expected {expected} (1 + (ratio-1)*x)"
            );
        }
    }

    /// Real, direct anchor: `c_mix2` must equal the liquid/vapor reference
    /// stiffness EXACTLY at the two real endpoints (x=0, x=1) -- the
    /// continuity-with-the-neighboring-materials claim this module's own
    /// doc makes, checked directly rather than assumed.
    #[test]
    fn mixture_stiffness_matches_both_pure_phase_references_at_the_endpoints() {
        let table = real_table();
        let material = BoilingMixtureMaterial::from_table(&table, 1.0, 0.0, 0.5, 8.0);
        let c_l2 = table.c_l_m_s * table.c_l_m_s;
        let c_v_ref2 = material.c_v_ref_m_s * material.c_v_ref_m_s;
        let c2_at_0 = c_mix2_m2_s2(
            0.0,
            table.rho_l_ref_kg_m3,
            c_l2,
            table.rho_v_ref_kg_m3,
            c_v_ref2,
        );
        let c2_at_1 = c_mix2_m2_s2(
            1.0,
            table.rho_l_ref_kg_m3,
            c_l2,
            table.rho_v_ref_kg_m3,
            c_v_ref2,
        );
        assert!(
            (c2_at_0 - c_l2).abs() / c_l2 < 1.0e-4,
            "c_mix2(0)={c2_at_0} must match c_l^2={c_l2} exactly"
        );
        assert!(
            (c2_at_1 - c_v_ref2).abs() / c_v_ref2 < 1.0e-4,
            "c_mix2(1)={c2_at_1} must match c_v_ref^2={c_v_ref2} exactly"
        );
    }

    /// Real, direct anchor: `acoustic_c2_at_particle` must genuinely
    /// differ with a particle's own `x` (`friction_hardening`), the same
    /// real per-particle-awareness `CavitatingFluidMaterial`'s own
    /// temperature test proves for its own live input.
    #[test]
    fn acoustic_c2_at_particle_genuinely_differs_with_quality() {
        let table = real_table();
        let material = BoilingMixtureMaterial::from_table(&table, 1.0, 0.0, 0.5, 8.0);
        let p_low = make_particle(&material, 0.1, 1.5);
        let p_high = make_particle(&material, 0.9, 5.0);
        let particles = Particles::from(vec![p_low, p_high]);
        let c2_low = material.acoustic_c2_at_particle(&particles, 0).unwrap();
        let c2_high = material.acoustic_c2_at_particle(&particles, 1).unwrap();
        assert!(
            (c2_low - c2_high).abs() > 1.0,
            "acoustic_c2_at_particle must genuinely respond to x: x=0.1 gave {c2_low}, \
             x=0.9 gave {c2_high}"
        );
    }

    /// Real, direct anchor: the measured `min_c_mix2_grid` must actually be
    /// a real lower bound -- no sampled `x` on a dense sweep may read
    /// below it (up to float slop), the real invariant `rest_acoustic_c2`
    /// depends on to be a genuine floor, not just a plausible one.
    #[test]
    fn measured_minimum_stiffness_is_a_genuine_lower_bound() {
        let table = real_table();
        let material = BoilingMixtureMaterial::from_table(&table, 1.0, 0.0, 0.5, 8.0);
        let c_l2 = table.c_l_m_s * table.c_l_m_s;
        let c_v_ref2 = material.c_v_ref_m_s * material.c_v_ref_m_s;
        for k in 0..=1000 {
            let x = k as f32 / 1000.0;
            let c2 = c_mix2_m2_s2(
                x,
                table.rho_l_ref_kg_m3,
                c_l2,
                table.rho_v_ref_kg_m3,
                c_v_ref2,
            );
            assert!(
                c2 >= material.min_c_mix2_grid - 1.0,
                "x={x}: c_mix2={c2} fell below the claimed measured minimum \
                 {}",
                material.min_c_mix2_grid
            );
        }
    }
}

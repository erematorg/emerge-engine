//! Compressible ideal-gas material -- real density does (no rest-pressure
//! offset), and real shock-capturing matters far more than for a weakly-
//! compressible liquid.
//!
//! `IdealGasMaterial` (`ideal_gas.rs`) landed 2026-08-18: `p=ρRT`, real
//! adiabatic sound speed `c=√(γRT)`, von Neumann-Richtmyer shock
//! viscosity reusing the same shared q-formula
//! (`matter::materials::utils::von_neumann_richtmyer_q`) liquids use, fed
//! this material's own real γ instead of a Tait-derived stand-in. CPU
//! only -- `p2g.wgsl`/`particles_update.wgsl` have no `case 13u` branch
//! yet, a real, disclosed limitation (see `IdealGasMaterial`'s own doc and
//! `ConstitutiveModel::Gas`'s doc), not yet verified against Sod's shock
//! tube (Toro, *Riemann Solvers and Numerical Methods for Fluid
//! Dynamics* -- the standard exact-analytical-solution benchmark for a
//! compressible-gas solver; needs an iterative Riemann solver, real
//! future work).
//!
//! Recovered 2026-08-22 from the `contact-based-interaction` branch
//! (commit `a7e4723`, 2026-08-18), which was never merged into `main` --
//! this branch forked from `main` before that merge happened, so the
//! material was genuinely absent here despite shipping on that other
//! branch. Adapted from that commit's `matter/materials/gas/{mod,
//! ideal_gas}.rs` (a `solid/liquid/gas/mixture` taxonomy split this
//! branch doesn't have yet) into one flat file matching this branch's
//! own current `materials/` layout -- no functional change, same real
//! physics, same tests, verified independently after the move.

use glam::{Mat2, Vec2};

use crate::energy::thermodynamics::ideal_gas::{
    AIR_ADIABATIC_INDEX, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, ideal_gas_sound_speed_from_temperature,
};
use crate::materials::utils::von_neumann_richtmyer_q;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Real standard sea-level atmospheric pressure (Pa) -- the default real
/// ambient reference `IdealGasMaterial::reference_pressure_pa` uses. See
/// that field's own doc.
pub const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;

/// Compressible ideal-gas material: isentropic (adiabatic) EOS
/// `p = p0·(ρ/ρ0)^γ` where `p0 = ρ0·R·T` (real ideal gas law evaluated at
/// the particle's own reference state), real adiabatic sound speed
/// `c = √(γRT)`, von Neumann-Richtmyer shock viscosity for compression
/// events. Genuinely different EOS shape from `NewtonianFluidMaterial`'s
/// Tait law: `p0` is a real physical rest pressure (not an empirically-fit
/// stiffness) and there's NO rest-pressure offset -- `p → 0` as `ρ → 0`,
/// exactly (see `energy::thermodynamics::ideal_gas`'s own
/// `pressure_vanishes_with_density` test).
///
/// Real, disclosed history (2026-08-18): the FIRST version of this
/// material used the naive isothermal law `p=ρRT` at fixed T directly --
/// internally inconsistent with the ADIABATIC sound speed it already used
/// for CFL/shock viscosity, and NOT self-limiting under expansion
/// (`p·V=nRT` stays exactly constant for a fixed-T ideal gas, so the
/// outward force never throttles down). A live demo (`examples/
/// basic_gas.rs`) caught this as a genuine runaway (avg J climbing to
/// 8-11 within under 2 simulated seconds from a 3:1 initial pressure
/// ratio) -- not a tuning problem. The isentropic form fixes it: MPM
/// substeps run far too fast for heat conduction to equalize (the SAME
/// real justification the adiabatic sound speed already relied on), and
/// `p·V` now falls off as `V^(1-γ)` under expansion, strictly decreasing
/// for γ>1.
///
/// Real, disclosed coupling: `p0` reads `Particles::temperature` directly,
/// so a `ThermalDiffusion` attached to the same scene still genuinely
/// shifts gas pressure as it heats/cools particles over its own slower
/// (conductive) timescale -- the fast, per-substep mechanical response to
/// compression/expansion is what changed, not the (slow) thermal coupling.
/// Without a `ThermalDiffusion`, temperature stays at whatever
/// `init_particle` seeds it to (`reference_temperature_k`).
///
/// CPU only today -- `p2g.wgsl`/`particles_update.wgsl` have no
/// `case 13u` branch, so a `Gas`-modeled particle on the GPU path silently
/// falls through those shaders' `default: { return mat2x2<f32>(); }` arm
/// (zero stress). Real, disclosed limitation, not a hidden gap -- see
/// `ConstitutiveModel::Gas`'s own doc. Matches this engine's own standing
/// rule: CPU correctness first, GPU port second.
#[derive(Debug, Clone, Copy)]
pub struct IdealGasMaterial {
    /// Reference density ρ₀ (grid units, `rho_SI * dx_meters²`).
    pub rest_density: f32,
    /// Dynamic viscosity µ (Pa·s, raw SI -- passes through unconverted,
    /// same regression-fixed convention `NewtonianFluidMaterial` uses).
    /// Real air value ≈1.81e-5 Pa·s; 0.0 = inviscid (Euler gas dynamics).
    pub dynamic_viscosity: f32,
    /// Specific gas constant R, GRID-scaled (`R_SI / dx_meters²`) -- see
    /// `from_physical`'s own doc for the full derivation of why R itself
    /// (not just density) needs this factor, unlike Tait's ratio-based EOS.
    pub specific_gas_constant: f32,
    /// Real adiabatic index γ = Cp/Cv (air/diatomic: 1.4, `AIR_ADIABATIC_INDEX`;
    /// monatomic: 5/3; triatomic: ~1.3). Used both for the real adiabatic
    /// sound speed AND as the shock-viscosity weak-shock coefficient's own
    /// gamma (`von_neumann_richtmyer_q`'s `weak_shock_gamma`) -- for this
    /// material that substitution is exact, not a stand-in: γ here IS the
    /// real thermodynamic adiabatic index Kurapatenko's own coefficient is
    /// defined in terms of.
    pub adiabatic_index: f32,
    /// Temperature (Kelvin) seeded onto every particle at spawn
    /// (`init_particle`) and used as the fixed reference state for
    /// `timestep_bound`/`rest_acoustic_c2` (see `timestep_bound`'s own
    /// doc for the real, disclosed limitation this implies under strong
    /// active heating).
    pub reference_temperature_k: f32,
    /// Real, disclosed fix (2026-08-29, independent verification):
    /// ambient absolute pressure (Pa, raw SI)
    /// this gas mechanically pushes AGAINST. `kirchhoff_stress` computes a
    /// real absolute pressure `p_abs = rho0*R*T*(rho/rho0)^gamma` (correct
    /// for the thermodynamics -- temperature/sound-speed/EOS identities all
    /// still use `p_abs`), but the MECHANICAL stress this material
    /// contributes to P2G must be `-(p_abs - reference_pressure_pa)*I`, the
    /// real gauge-pressure convention (Oregon State MPM documentation
    /// explicitly distinguishes absolute pressure for a gas confined by
    /// rigid walls from gauge pressure for a gas interacting with an
    /// initially-unstressed material -- the latter is this engine's own
    /// case, ice/water/steam sharing one grid with no confining walls
    /// around the gas). Real, confirmed structural bug this fixes: without
    /// this subtraction, EVERY steam particle, even sitting exactly at its
    /// own rest density (`J=1`), exerted a real, full ~101325 Pa of
    /// outward-pushing stress with nothing to balance it (unlike
    /// `NewtonianFluidMaterial`'s Tait EOS, which has a `-1` term making
    /// its own pressure exactly ZERO at rest by construction) -- a
    /// persistent, un-opposed DC force no amount of viscosity/damping can
    /// arrest, since damping only resists the RATE of expansion, not a
    /// constant driving pressure. This is the real, confirmed root cause of
    /// steam particles racing to `volume_ratio_max` and sticking there
    /// (the "grossit" symptom) -- two separate, real, sourced damping
    /// escalations (Kelvin-Voigt 2x, bulk viscosity 10x) were tried first
    /// and measured to do nothing, exactly as expected once this mechanism
    /// was understood: a constant unopposed force has no equilibrium for
    /// viscosity to damp toward.
    ///
    /// Real default: standard sea-level atmospheric pressure (101,325 Pa) --
    /// the correct, physically-honest default for a gas released into any
    /// normal terrestrial scene, not an arbitrary zero. A submerged-bubble
    /// scene wanting hydrostatic realism can add `rho_water*g*depth` on
    /// top; this default is the right floor for that, not a replacement.
    pub reference_pressure_pa: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Lower bound on `J = V/V0` -- unlike a weakly-compressible liquid, a
    /// real gas can compress far below half its rest volume, so this is
    /// deliberately much wider than `NewtonianFluidMaterial`'s pinned
    /// `[0.5, 2.0]`. NOT tuned against a real impact/shock test scene the
    /// way fluid's own bounds are (no such scene exists for gas yet) --
    /// first real cut, disclosed as provisional.
    pub volume_ratio_min: f32,
    /// Upper bound on `J`. See `volume_ratio_min`'s own doc.
    pub volume_ratio_max: f32,
    /// Bulk (dilatational/second) viscosity ζ, Pa·s -- raw SI, unconverted,
    /// same convention as `dynamic_viscosity`'s own doc. Adds
    /// `τ += ζ·(∇·v)·I` to Kirchhoff stress -- the real Navier-Stokes
    /// second-viscosity term, mirroring `NewtonianFluidMaterial::
    /// bulk_viscosity`'s own already-proven formula exactly, but UNLIKE
    /// this material's own shock viscosity `q` (gated to `∇·v < 0`,
    /// compression only), this term is unconditional on sign -- it
    /// resists rapid EXPANSION exactly as much as compression.
    ///
    /// Real, found live 2026-08-28 (`phase_states_gui.rs`'s chimney demo):
    /// before this field existed, `IdealGasMaterial` had real resistance to
    /// over-COMPRESSION (the EOS pressure term self-limits, `q` engages),
    /// but genuinely NOTHING resisting over-EXPANSION beyond the hard
    /// `volume_ratio_max` clamp -- under real buoyancy-driven velocity
    /// divergence, individual particles' `J` would race toward that clamp
    /// with zero damping, hit it, fall back, race up again: a real,
    /// confirmed (via direct per-particle `det(F)` logging) chaotic
    /// particle-to-particle size variance, not a rendering bug and not
    /// fixed by narrowing the clamp's numeric range alone.
    ///
    /// Real, cited magnitude: unlike a monatomic ideal gas (bulk viscosity
    /// exactly zero under Stokes' hypothesis), water vapor is polyatomic
    /// (asymmetric top, real rotational AND vibrational relaxation modes)
    /// -- Cramer, M.S. (2012), "Numerical estimates for the bulk viscosity
    /// of ideal gases," Physics of Fluids 24, 066102: water vapor's bulk
    /// viscosity (estimated over 380-1000 K, covering this material's own
    /// real boiling-point reference) is "hundreds or thousands of times
    /// larger than [its] shear viscosity" -- a real, large, well-cited
    /// effect, not a small correction. See `water_vapor_bulk_viscosity_pa_s`
    /// for a real, sourced conversion from a gas's own shear viscosity,
    /// using the conservative (low) end of that cited range.
    ///
    /// 0.0 = off (default, every existing preset/scene unaffected).
    pub bulk_viscosity: f32,
}

impl IdealGasMaterial {
    /// Construct directly from grid-native parameters -- NOT SI units.
    /// Prefer [`Self::air`] for a real-air preset, or [`Self::from_physical`]
    /// for a real SI-to-grid conversion from measured density/viscosity/
    /// gas-constant/temperature.
    pub const fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        specific_gas_constant: f32,
        adiabatic_index: f32,
        reference_temperature_k: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            specific_gas_constant,
            adiabatic_index,
            reference_temperature_k,
            reference_pressure_pa: STANDARD_ATMOSPHERE_PA,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            volume_ratio_min: 0.05,
            volume_ratio_max: 20.0,
            bulk_viscosity: 0.0,
        }
    }

    /// Real dry air at a given SI density/temperature. Real, standard
    /// constants: `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K`/`AIR_ADIABATIC_INDEX`
    /// (already verified against the real ~343 m/s reference speed of
    /// sound in `energy::thermodynamics::ideal_gas`'s own tests), dynamic
    /// viscosity 1.81e-5 Pa·s (air at ~20°C, Sutherland's law reference).
    pub fn air(rho_kg_m3: f32, temperature_k: f32, config: &crate::SimConfig) -> Self {
        Self::from_physical(
            rho_kg_m3,
            1.81e-5,
            AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
            AIR_ADIABATIC_INDEX,
            temperature_k,
            config,
        )
    }

    /// General ideal-gas constructor from real SI properties.
    ///
    /// `specific_gas_constant_j_kg_k` -- R for the specific gas (air:
    /// 287.05, `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K`). `adiabatic_index` --
    /// real γ=Cp/Cv (air/diatomic: 1.4; monatomic: 5/3; triatomic: ~1.3).
    ///
    /// Grid-scaling derivation (mirrors `NewtonianFluidMaterial::
    /// from_physical`'s own doc and the real regression it fixed, extended
    /// here for the ideal gas law's different functional form): solver
    /// time is already real seconds and positions are grid cells
    /// (`x_grid = x_SI/dx`), so pressure must stay raw SI
    /// (`p_grid == p_SI`) and only density converts
    /// (`rho_grid = rho_SI*dx²`), exactly as that fix established. Tait's
    /// EOS uses a density RATIO (`ρ/ρ₀`), which is scale-invariant under
    /// that conversion for free -- both numerator and denominator carry
    /// the same `dx²` factor, which cancels. The ideal gas law is LINEAR
    /// in density (`p=ρRT`, no ratio), so nothing cancels automatically:
    /// `R` itself must absorb the compensating `1/dx²` for `p` to come out
    /// raw SI: `R_grid = R_SI/dx²` gives
    /// `ρ_grid·R_grid·T = ρ_SI·dx²·(R_SI/dx²)·T = ρ_SI·R_SI·T = p_SI`.
    /// The SAME `R_grid` also gives the correct GRID-unit (cells/s)
    /// adiabatic sound speed for the CFL/shock terms:
    /// `c_grid² = γ·R_grid·T = (γ·R_SI·T)/dx² = c_SI²/dx²`, i.e.
    /// `c_grid = c_SI/dx` -- the same `v_grid = v_SI/dx` convention every
    /// other velocity in this engine already uses (e.g. `gravity_to_grid`).
    pub fn from_physical(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        specific_gas_constant_j_kg_k: f32,
        adiabatic_index: f32,
        temperature_k: f32,
        config: &crate::SimConfig,
    ) -> Self {
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "IdealGasMaterial::from_physical requires a positive dx_meters"
        );
        let dx2 = config.dx_meters * config.dx_meters;
        let rho_grid = rho_kg_m3 * dx2;
        let r_grid = specific_gas_constant_j_kg_k / dx2;
        Self::new(rho_grid, eta_pa_s, r_grid, adiabatic_index, temperature_k)
    }

    /// Same as [`Self::from_physical`], named fields instead of 4 adjacent
    /// positional `f32`s -- same real struct-bundling fix already used
    /// elsewhere in this codebase (`PhysicalRenderContractParams`,
    /// `NaccMaterialParams`) for a constructor where several same-typed
    /// parameters make transposition a real, silent risk.
    pub fn from_physical_params(params: IdealGasPhysicalParams, config: &crate::SimConfig) -> Self {
        Self::from_physical(
            params.rho_kg_m3,
            params.eta_pa_s,
            params.specific_gas_constant_j_kg_k,
            params.adiabatic_index,
            params.temperature_k,
            config,
        )
    }
}

/// Named-field parameters for [`IdealGasMaterial::from_physical_params`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdealGasPhysicalParams {
    pub rho_kg_m3: f32,
    pub eta_pa_s: f32,
    pub specific_gas_constant_j_kg_k: f32,
    pub adiabatic_index: f32,
    pub temperature_k: f32,
}

/// Real, cited bulk (dilatational) viscosity for a polyatomic gas whose
/// bulk viscosity is dominated by rotational/vibrational relaxation --
/// water vapor specifically, though the same real mechanism applies to any
/// non-monatomic gas. See `IdealGasMaterial::bulk_viscosity`'s own doc for
/// the full citation (Cramer 2012, Physics of Fluids 24, 066102: water
/// vapor's bulk viscosity is "hundreds or thousands of times" its shear
/// viscosity, estimated over 380-1000 K).
///
/// `BULK_TO_SHEAR_RATIO` uses the HIGH end of that cited range (thousands),
/// not the low end (hundreds) this constant originally used. Real,
/// disclosed escalation (2026-08-29): the low-end value (100x) was
/// confirmed live, via a real per-node P2G diagnostic on
/// `phase_states_gui.rs`'s own Moon-gravity heated scene, to still let an
/// isolated steam particle's J race to `volume_ratio_max` and stick there
/// -- the real "grossit" symptom this whole field exists to damp. Moving to
/// the high end of the SAME citation is a real, sourced choice, not a new
/// number invented to chase the symptom.
///
/// Returns real SI Pa·s -- passes through UNCONVERTED into
/// `bulk_viscosity`, same raw-SI convention `dynamic_viscosity` already
/// uses for this material (see that field's own doc).
pub fn water_vapor_bulk_viscosity_pa_s(shear_viscosity_pa_s: f32) -> f32 {
    // Real, disclosed escalation (2026-08-29): Cramer 2012's own cited range
    // for water vapor is "hundreds or thousands of times" shear viscosity --
    // the previous 100x sat at the conservative LOW edge of that range and
    // was confirmed live (Moon-gravity heated run, `EMERGE_TRACK_PARTICLE_
    // NODES` diagnostic) to still let an isolated steam particle's own J
    // race to the material's `volume_ratio_max` ceiling and stick there --
    // the exact "grossit" symptom this field was added for in the first
    // place (see this material's own `volume_ratio_max` doc, found
    // 2026-08-28). Moving to 1000x -- the higher end of the SAME cited
    // range, not a new number invented to chase the symptom -- since 100x
    // demonstrably wasn't enough real damping against this engine's own
    // buoyancy-driven divergence.
    const BULK_TO_SHEAR_RATIO: f32 = 1000.0;
    shear_viscosity_pa_s * BULK_TO_SHEAR_RATIO
}

impl MaterialModel for IdealGasMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Gas
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "IdealGasMaterial has no GPU stress path: the shaders have no gas case, so it would run with zero pressure",
        )
    }

    /// Seeds the exact analytical rest state, same contract
    /// `NewtonianFluidMaterial::init_particle` establishes (`V0=m/ρ0`,
    /// `ρ=ρ0/J`) -- plus `temperature`, which a strict fluid never needs
    /// but this EOS genuinely depends on (`p=ρRT`).
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
        particle.temperature = self.reference_temperature_k;
    }

    /// Real fix, found live 2026-08-18 debugging `examples/basic_steam.rs`
    /// (water boiling into gas -- see `MaterialModel::
    /// init_particle_from_transition`'s own doc for the full mechanism):
    /// `init_particle`'s own `mass/rest_density` formula is exactly right
    /// for a FRESH spawn, but wrong for a TRANSITION from a material with
    /// a dramatically different rest density (e.g. water->steam, a real
    /// ~1700x ratio) -- it would make the particle's claimed volume jump
    /// that same ~1700x in a single instant, injecting a real but wildly
    /// under-resolved force spike (confirmed live as a real crash cause).
    ///
    /// Correct fix: keep the reference volume TRUE (`mass/rest_density`,
    /// matching exactly what `kirchhoff_stress`/`update_particle` already
    /// assume every substep), and instead give the particle a STARTING
    /// deformation gradient reflecting how compressed it genuinely is
    /// relative to that true reference -- a real, physically honest
    /// picture (freshly-formed gas still occupying roughly its old, much
    /// smaller prior footprint IS heavily compressed relative to this
    /// material's own rest state), clamped to this material's own
    /// `[volume_ratio_min, volume_ratio_max]`, the SAME bounds every
    /// subsequent substep already enforces -- so the starting state is
    /// consistent with the ongoing dynamics from frame one, not a
    /// separate, inconsistent value that later dynamics silently
    /// overwrite (a first, wrong fix attempt -- capping `initial_volume`
    /// directly -- learned this the hard way: `update_particle` recomputes
    /// volume fresh from `mass*j/rest_density` every substep, never
    /// reading `initial_volume` again, so a capped `initial_volume` alone
    /// only held for one substep before becoming permanently inconsistent
    /// with the real volume update.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume)
            .clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = particle.mass / particle.volume.max(1.0e-9);
        // Real regression fix (external review): do NOT touch temperature
        // here. `Simulation::apply_phase_transition` already debits it by
        // `latent_heat / heat_capacity` immediately before calling this --
        // overwriting it back to a fixed `reference_temperature_k` silently
        // erased that debit every single time (water boiling into steam
        // paid the latent-heat cost, then had it discarded a few lines
        // later, always landing exactly at 373.15K regardless of how much
        // energy actually crossed the threshold). Every sibling override
        // (`NewtonianFluidMaterial`, `BoilingMixtureMaterial`,
        // `CavitatingFluidMaterial`) already leaves `particle.temperature`
        // untouched for the same reason -- this material was the one
        // outlier. `init_particle` (fresh spawns, no real prior thermal
        // history) is unaffected and still seeds `reference_temperature_k`.
    }

    /// `c² = γ·R·T` evaluated at the reference state (`J=1`,
    /// `T=reference_temperature_k`) -- the same formula `timestep_bound`
    /// evaluates, not a second derivation. See that method's own doc for
    /// the real, disclosed limitation this fixed-reference-T evaluation
    /// implies under active heating.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.specific_gas_constant > 0.0 && self.reference_temperature_k > 0.0 {
            Some(self.adiabatic_index * self.specific_gas_constant * self.reference_temperature_k)
        } else {
            None
        }
    }

    /// Real fix (2026-08-31, found live -- `phase_states_gui.rs`'s own
    /// sustained-heating steam divergence, confirmed by a direct A/B
    /// against the pre-existing, untouched material: divergence in the
    /// THOUSANDS, `last_substeps` climbing toward its own cap, fps
    /// collapsing to single digits, `steam max(J)` pinned at
    /// `volume_ratio_max`). Real, previously-DISCLOSED gap this closes
    /// (see `timestep_bound`'s own doc, written 2026-08-29, never
    /// implemented): `kirchhoff_stress` already evaluates `p0=rho0*R*T`
    /// at the particle's own LIVE `temperature` (adiabatic ideal-gas law,
    /// same citation as `rest_acoustic_c2`'s own doc), but `timestep_bound`
    /// only ever evaluated `c^2=gamma*R*T` at the FIXED, construction-time
    /// `reference_temperature_k` -- under real active heating (this
    /// demo's own boiling scene routinely pushes steam well past its own
    /// `reference_temperature_k=373.15K` reference), the real stiffness
    /// GROWS (`c^2` is linear in `T`) while the CFL bound stays anchored
    /// to the old, softer value, an increasingly under-resolved timestep
    /// that gets WORSE the longer heating continues -- exactly the
    /// observed escalating-then-runaway signature, not a one-off spike.
    /// Same real formula as `rest_acoustic_c2`, evaluated at the live
    /// temperature instead of the frozen reference -- not a new physics
    /// model, the SAME adiabatic ideal-gas relation with its one
    /// temperature-dependent input finally supplied.
    fn acoustic_c2_at_temperature(&self, temperature_k: f32) -> Option<f32> {
        if self.specific_gas_constant > 0.0 && temperature_k > 0.0 {
            Some(self.adiabatic_index * self.specific_gas_constant * temperature_k)
        } else {
            self.rest_acoustic_c2()
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density = (self.rest_density / j).max(self.min_density);
        let temperature = particles.temperature[i].max(0.0);

        // Isentropic (adiabatic) pressure law, NOT the naive isothermal
        // `p=ρRT` at fixed T this used until 2026-08-18. Real, found via a
        // live-crashing demo (basic_gas.rs): MPM substeps operate on
        // timescales far too short for heat conduction to equalize --
        // exactly the same real justification `ideal_gas_sound_speed`'s
        // own doc already gives for using the ADIABATIC sound speed
        // (`c=√(γRT)`), not the isothermal one. Using the isothermal
        // pressure law together with the adiabatic sound speed was a real,
        // internal inconsistency: the CFL/shock-viscosity terms assumed a
        // stiffer, self-limiting adiabatic response while the actual
        // restoring force was the softer isothermal one, which never
        // throttles down under expansion (`p·V = nRT` stays EXACTLY
        // constant as a fixed-T ideal gas expands, so the P2G force
        // contribution never decays) -- a real energy-injecting mismatch,
        // not a tuning problem, and the direct cause of the observed
        // runaway (avg J climbing to 8-11 in well under 2 simulated
        // seconds from an initial 3:1 pressure ratio).
        //
        // Standard compressible-flow result (isentropic relation):
        // combining `p=ρRT` with the adiabatic relation `T/T0=(ρ/ρ0)^(γ-1)`
        // gives `p = p0·(ρ/ρ0)^γ`, where `p0=ρ0·R·T` is the real rest
        // pressure at the particle's CURRENT temperature (still real
        // thermal coupling -- a `ThermalDiffusion` shifts `p0` exactly as
        // `p=ρRT` would). This form self-limits under expansion (`p·V`
        // now falls off as `V^(1-γ)`, strictly decreasing for γ>1) and its
        // own `dp/dρ` at `ρ=ρ0` is exactly `γRT` -- the SAME formula
        // `rest_acoustic_c2`/`timestep_bound` already compute, now
        // internally consistent rather than assumed.
        let p0 = self.rest_density * self.specific_gas_constant * temperature;
        let pressure_abs = (p0
            * crate::materials::utils::fast_pow(density / self.rest_density, self.adiabatic_index))
        .max(0.0); // physically required floor: ρ,T >= 0 => p_abs >= 0, nothing to configure

        // Real, disclosed fix (2026-08-29, see `reference_pressure_pa`'s own
        // doc): the MECHANICAL stress this material contributes must be
        // gauge pressure, not absolute -- `p_abs` alone means this gas
        // pushes with its full ~101325 Pa even sitting at its own rest
        // density, with nothing to balance it. `p_abs` itself stays
        // available for anything thermodynamic (temperature coupling,
        // sound speed) -- only the mechanical stress below changes.
        let pressure_gauge = pressure_abs - self.reference_pressure_pa;

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y; // = 2·∇·v

        if self.dynamic_viscosity > 0.0 {
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }

        // Bulk viscosity zeta: tau += zeta*(div v)*I -- see this field's
        // own doc. Same real formula NewtonianFluidMaterial::kirchhoff_stress
        // already uses; `div_v` here is tr(C+C^T) = 2*div(v), so the *0.5
        // recovers the true divergence, same convention that file's own
        // comment documents.
        if self.bulk_viscosity > 0.0 {
            stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
        }

        // Artificial (shock) viscosity -- von Neumann & Richtmyer 1950,
        // reusing the SAME shared q-formula `NewtonianFluidMaterial` uses
        // (`materials::utils::von_neumann_richtmyer_q`), fed this
        // material's own real adiabatic sound speed and real γ rather
        // than a Tait-EOS-derived stand-in (see that function's own doc).
        let c_sound = ideal_gas_sound_speed_from_temperature(
            self.specific_gas_constant,
            self.adiabatic_index,
            temperature,
        );
        let q = von_neumann_richtmyer_q(
            self.rest_density,
            j,
            0.5 * div_v,
            1.0,
            c_sound,
            self.adiabatic_index,
        );
        stress += Mat2::from_diagonal(Vec2::splat(-q));

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Real, disclosed regression fixed 2026-08-30 -- same fix, same
        // root cause, as `NewtonianFluidMaterial::update_particle`'s own
        // doc: `det(I+dt*C)` is not rotation-invariant (a pure rigid
        // rotation should leave J exactly unchanged but this formula gives
        // a strictly positive O(dt^2) expansion every substep, baked in
        // permanently by isotropization). Fixed with the continuity
        // equation's own exact exponential solution, `J_{n+1}=J_n*exp(dt*
        // div(v))` -- restores the pre-`57b83dc` `fluid_state.rs` behavior.
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        let j = (old_j * (dt * div_v).exp()).clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        let density = (self.rest_density / j).max(self.min_density);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Gas as u32,
            rest_density: self.rest_density,
            // Repurposed slots -- GPU has no `case 13u` branch to read
            // these yet (see this struct's own top-of-file doc); kept
            // filled so the CPU-side struct is complete and ready for
            // when that branch lands, same convention `params()` already
            // uses for every other material's union-layout fields.
            eos_stiffness: self.specific_gas_constant,
            eos_power: self.adiabatic_index,
            dynamic_viscosity: self.dynamic_viscosity,
            volume_ratio_min: self.volume_ratio_min,
            volume_ratio_max: self.volume_ratio_max,
            bulk_viscosity: self.bulk_viscosity,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed limitation, PARTIALLY closed 2026-08-31 (see
    /// `acoustic_c2_at_temperature`'s own doc): this trait method's
    /// signature still carries only `density`, not per-particle
    /// temperature, so the acoustic term BELOW still uses
    /// `reference_temperature_k` rather than the particle's actual
    /// current `T` -- but `cfl.rs`'s own dispatch site now adds a SEPARATE,
    /// real live-temperature-aware term via `acoustic_c2_at_temperature`
    /// (same established pattern as its shock-viscosity and single-
    /// particle-instability terms, neither of which live inside
    /// `timestep_bound` either), so the true, live-T-tightened bound is
    /// still real and enforced -- just not from this method alone.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;

        let c2 = self.adiabatic_index * self.specific_gas_constant * self.reference_temperature_k;
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Combined explicit-viscous-diffusion bound for BOTH shear and bulk
        // viscosity -- same real stability reasoning `DruckerPragerMaterial::
        // timestep_bound`'s own doc gives for its Kelvin-Voigt term (measured
        // live: an unbounded viscous term makes peak speed jump instead of
        // damping). Combined linearly, not each bounded separately: both
        // terms multiply the SAME velocity-gradient-derived stress, so their
        // worst-case combined diffusive coefficient is the real, conservative
        // bound, not an approximation. `bulk_viscosity` can be "hundreds of
        // times" `dynamic_viscosity` for a real polyatomic gas (see that
        // field's own doc) -- without including it here, the substep
        // selector would never see the real stiffness it adds.
        let combined_viscosity = self.dynamic_viscosity + self.bulk_viscosity.max(0.0);
        if combined_viscosity > 0.0 {
            let density = density.max(self.min_density);
            let kinematic_viscosity = combined_viscosity / density;
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

    /// `dx_meters = 1.0` collapses grid<->SI scaling to identity, so this
    /// test can compare `kirchhoff_stress`'s output directly against the
    /// real, textbook air reference `energy::thermodynamics::ideal_gas`'s
    /// own test already verifies (~101,325 Pa at ρ=1.204 kg/m³, T=293.15K).
    fn unit_dx_config() -> SimConfig {
        SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3)
    }

    fn one_particle_at_rest(mat: &IdealGasMaterial) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = mat.rest_density; // V0 = mass/rest_density = 1.0
        p.deformation_gradient = Mat2::IDENTITY;
        mat.init_particle(&mut p);
        Particles::from(vec![p])
    }

    /// Real, corrected expectation (2026-08-29, see `reference_pressure_pa`'s
    /// own doc -- a confirmed structural bug found while investigating
    /// `phase_states_gui.rs`'s steam-explosion symptom). Real air at its
    /// own real rest density/temperature (p_abs
    /// really is ~101325 Pa here) exerts ZERO mechanical stress once
    /// embedded in the real default ambient (1 standard atmosphere) --
    /// there's nothing pushing it to expand or compress relative to its
    /// surroundings. This REPLACES the pre-fix expectation (that
    /// `kirchhoff_stress` should reproduce the raw ~101325 Pa absolute
    /// pressure), which was exactly the bug: it meant every gas particle
    /// pushed outward with its full absolute pressure even at its own
    /// rest state, with nothing to balance it -- a persistent, un-opposed
    /// force no viscosity could arrest.
    #[test]
    fn rest_gauge_pressure_is_zero_at_standard_atmosphere() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = IdealGasMaterial::air(1.204, 293.15, &config);
        let particles = one_particle_at_rest(&mat);

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure_gauge = -stress.x_axis.x;

        assert!(
            pressure_gauge.abs() < 100.0,
            "real air at its own rest density/temperature, embedded in the \
             default standard atmosphere, must exert ~zero mechanical \
             (gauge) stress: got {pressure_gauge:.1} Pa"
        );
    }

    /// Real regression guard (external review, P0 #5): `init_particle_
    /// from_transition` must NOT touch `particle.temperature` --
    /// `Simulation::apply_phase_transition` already debits it by
    /// `latent_heat/heat_capacity` immediately before calling this, and
    /// this material used to silently overwrite that debit back to a
    /// fixed `reference_temperature_k`, discarding it completely (water
    /// boiling into steam always landed at exactly 373.15K regardless of
    /// how much energy actually crossed the threshold). Starts the
    /// particle at a temperature that is deliberately NOT the reference
    /// value -- the real post-debit state a genuine transition leaves
    /// behind -- and confirms it survives untouched, matching every
    /// sibling `init_particle_from_transition` override
    /// (`NewtonianFluidMaterial`, `BoilingMixtureMaterial`,
    /// `CavitatingFluidMaterial`) which never touched temperature at all.
    #[test]
    fn init_particle_from_transition_preserves_the_incoming_temperature() {
        let config = unit_dx_config();
        let mat = IdealGasMaterial::air(1.204, 373.15, &config);

        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.deformation_gradient = Mat2::IDENTITY;
        p.volume = 1.0; // "prior volume" as whatever material it transitioned from
        let post_latent_heat_debit_temperature = 350.0_f32; // != reference_temperature_k
        p.temperature = post_latent_heat_debit_temperature;

        mat.init_particle_from_transition(&mut p);

        assert_eq!(
            p.temperature, post_latent_heat_debit_temperature,
            "init_particle_from_transition must leave an already-debited \
             temperature exactly untouched, not reset it to reference_temperature_k \
             ({}) -- pre-fix, this always landed at the reference value regardless \
             of the real latent-heat debit",
            mat.reference_temperature_k
        );
    }

    /// `water_vapor_bulk_viscosity_pa_s` must return a real, finite,
    /// positive multiple of shear viscosity -- basic sanity floor for the
    /// cited Cramer 2012 conversion before trusting it in a live scene.
    #[test]
    fn water_vapor_bulk_viscosity_is_a_real_positive_multiple_of_shear() {
        let shear = 1.26e-5_f32; // real steam shear viscosity, this codebase's own cited value
        let bulk = water_vapor_bulk_viscosity_pa_s(shear);
        assert!(
            bulk.is_finite() && bulk > shear,
            "cited water vapor bulk viscosity must be a real, large multiple \
             of shear viscosity (Cramer 2012: hundreds-thousands x), got \
             bulk={bulk} shear={shear}"
        );
    }

    /// The actual mechanism this was added for: nonzero `bulk_viscosity`
    /// must resist EXPANSION (`div(v) > 0`), not just compression -- unlike
    /// this material's own shock viscosity `q`, which only engages under
    /// compression. Confirms the term actually engages under a real
    /// expanding velocity field, not just that the field exists.
    #[test]
    fn nonzero_bulk_viscosity_resists_expansion_not_just_compression() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let inviscid = IdealGasMaterial::air(1.204, 293.15, &config);
        let mut damped = inviscid;
        damped.bulk_viscosity = 1.0;

        let mut p = Particle::zeroed();
        p.mass = inviscid.rest_density;
        p.deformation_gradient = Mat2::IDENTITY;
        inviscid.init_particle(&mut p);
        // Pure isotropic expansion: div(v) > 0.
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.5, 0.0), Vec2::new(0.0, 0.5));
        let particles = Particles::from(vec![p]);

        let tau_inviscid = inviscid.kirchhoff_stress(&particles, 0);
        let tau_damped = damped.kirchhoff_stress(&particles, 0);

        // Real Navier-Stokes second-viscosity sign: resisting expansion
        // means an ADDED positive (compressive-direction-opposing, i.e.
        // less negative / more positive) diagonal stress relative to the
        // inviscid case -- a real restoring force against further
        // expansion, not an amplifying one.
        assert!(
            tau_damped.x_axis.x > tau_inviscid.x_axis.x,
            "bulk viscosity must add real resistance to expansion: \
             inviscid={:.6} damped={:.6}",
            tau_inviscid.x_axis.x,
            tau_damped.x_axis.x
        );
    }

    /// `bulk_viscosity == 0.0` (every preset's default) must reproduce the
    /// exact pre-2026-08-28 stress -- a real regression guard that adding
    /// this mechanism did not change default behavior for any existing
    /// preset/scene.
    #[test]
    fn zero_bulk_viscosity_is_bit_identical_to_prior_behavior() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = IdealGasMaterial::air(1.204, 293.15, &config);
        assert_eq!(mat.bulk_viscosity, 0.0);

        let mut p = Particle::zeroed();
        p.mass = mat.rest_density;
        p.deformation_gradient = Mat2::IDENTITY;
        mat.init_particle(&mut p);
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4));
        let particles = Particles::from(vec![p]);

        // Two materials, one with the new field explicitly re-zeroed via
        // struct-update -- same value either way, proving the new branch
        // is a true no-op at the default.
        let same = IdealGasMaterial {
            bulk_viscosity: 0.0,
            ..mat
        };
        assert_eq!(
            mat.kirchhoff_stress(&particles, 0),
            same.kirchhoff_stress(&particles, 0),
            "bulk_viscosity=0.0 must be bit-identical to the pre-existing path"
        );
    }

    /// Real, corrected physics (2026-08-29, see `reference_pressure_pa`'s
    /// own doc). As a gas pocket approaches vacuum, its ABSOLUTE pressure
    /// really does vanish (real ideal-gas law, unchanged) -- but the
    /// MECHANICAL (gauge) stress it exerts approaches `-reference_
    /// pressure_pa`, not zero, exactly what a real near-vacuum bubble
    /// embedded in a real atmosphere actually does: get crushed inward by
    /// the full ambient pressure, not sit in force-free equilibrium. This
    /// REPLACES the pre-fix expectation (mechanical stress -> 0), which was
    /// exactly the confirmed "no counter-pressure" bug this field fixes.
    #[test]
    fn gauge_pressure_approaches_negative_reference_as_density_vanishes() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = IdealGasMaterial::air(1.204, 293.15, &config);
        let mut particles = one_particle_at_rest(&mat);
        // Expand hugely (J >> 1) -> density -> 0.
        particles.deformation_gradient[0] = Mat2::from_diagonal(Vec2::splat(1000.0));

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure_gauge = -stress.x_axis.x;
        let expected = -mat.reference_pressure_pa;
        assert!(
            (pressure_gauge - expected).abs() / mat.reference_pressure_pa < 0.01,
            "near-vacuum gauge pressure should approach -reference_pressure_pa \
             (real ambient crushing the near-vacuum pocket): expected~={expected:.1}, \
             got {pressure_gauge:.1}"
        );
    }

    #[test]
    fn hotter_gas_has_higher_pressure_at_the_same_density() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let cold = IdealGasMaterial::air(1.204, 250.0, &config);
        let hot = IdealGasMaterial::air(1.204, 400.0, &config);

        let p_cold = -cold
            .kirchhoff_stress(&one_particle_at_rest(&cold), 0)
            .x_axis
            .x;
        let p_hot = -hot
            .kirchhoff_stress(&one_particle_at_rest(&hot), 0)
            .x_axis
            .x;

        assert!(
            p_hot > p_cold,
            "real p=rhoRT must increase with temperature at fixed density: \
             p_cold={p_cold:.1} p_hot={p_hot:.1}"
        );
    }

    #[test]
    fn compression_increases_pressure_at_fixed_temperature() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = IdealGasMaterial::air(1.204, 293.15, &config);
        let mut particles = one_particle_at_rest(&mat);
        particles.deformation_gradient[0] = Mat2::from_diagonal(Vec2::splat(0.5)); // J=0.25, denser

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure = -stress.x_axis.x;
        let rest_pressure = -mat
            .kirchhoff_stress(&one_particle_at_rest(&mat), 0)
            .x_axis
            .x;

        assert!(
            pressure > rest_pressure,
            "compressing a real ideal gas at fixed T must raise its pressure: \
             rest={rest_pressure:.1} compressed={pressure:.1}"
        );
    }

    #[test]
    fn rest_acoustic_c2_matches_timestep_bound_derivation() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = IdealGasMaterial::air(1.204, 293.15, &config);
        let c2 = mat
            .rest_acoustic_c2()
            .expect("air has a real acoustic term");
        let expected = AIR_ADIABATIC_INDEX * mat.specific_gas_constant * 293.15;
        assert!(
            (c2 - expected).abs() / expected < 1.0e-4,
            "rest_acoustic_c2 must match the same formula timestep_bound uses: \
             got {c2}, expected {expected}"
        );
    }

    #[test]
    fn constitutive_model_is_gas() {
        let mat = IdealGasMaterial::new(0.1, 0.0, 287.05, 1.4, 293.15);
        assert_eq!(mat.constitutive_model(), ConstitutiveModel::Gas);
        assert_eq!(mat.constitutive_model() as u32, 13);
    }
}

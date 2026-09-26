use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, NewtonianFluid, scale_stress, scale_visc};
use crate::materials::utils::von_neumann_richtmyer_q;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Current volume ratio `J = V/V0` from the material's own conserved volume
/// state (not a kernel-density gather, which is free-surface biased).
#[inline]
pub(crate) fn volume_j(initial_volume: f32, volume: f32, material_name: &str) -> f32 {
    assert!(
        initial_volume.is_finite() && initial_volume > 0.0 && volume.is_finite() && volume > 0.0,
        "{material_name}: fluid reference/current volume must be finite and positive"
    );
    volume / initial_volume
}

/// Artificial (shock) viscosity `q`, added to pressure as `-q*I`.
///
/// von Neumann & Richtmyer 1950 (LA-671) quadratic term + Landshoff linear
/// term -- the standard shock-capturing pair for Lagrangian hydrocodes, and
/// the same combined form used for MPM specifically (Wang et al., "Portable,
/// Massively Parallel Implementation of a Material Point Method for
/// Compressible Flows", arXiv:2404.17057, eq. 4). Gated to compression
/// (`div(v) < 0`) so it vanishes identically wherever the flow is smooth --
/// real shocks only form under compression, so that gate is textbook, not a
/// convenience.
///
/// `c0` (quadratic) is Kurapatenko 1967-derived rather than a flat constant:
/// `(gamma+1)/4` is the WEAK-shock limit of the fundamental derivative
/// Kurapatenko ties `c0` to. Deliberately weak-shock, NOT strong-shock
/// `(gamma+1)/2`: real, measured 2026-08-08, the strong-shock value made a
/// then-live crash WORSE, because the quadratic term's own contribution must
/// feed back into the CFL bound (Bate et al. 1995's combined
/// `c_eff = c_sound + 2*c0*h*|div(v)|`) before a stronger coefficient is safe.
/// That CFL extension needs a `timestep_bound` signature change and is real,
/// disclosed, unimplemented future work.
///
/// `c1 = 1.0` (Landshoff) is the standard theoretical value.
pub(crate) fn artificial_bulk_viscosity(
    eos_stiffness: f32,
    eos_power: f32,
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
) -> f32 {
    // REAL BUG FIXED 2026-08-13: an earlier form used rho^2 in the
    // quadratic term and NO rho at all in the linear one -- neither
    // matches the cited sources (Wang et al. arXiv:2404.17057 eq. 4;
    // `tmp/GeoTaichi`'s `MaterialModel.py::artifical_viscosity`), both of
    // which multiply BOTH terms by rho exactly once, the same real form
    // `von_neumann_richtmyer_q` (shared, EOS-agnostic) implements below.
    // Dimensionally the old form was inconsistent, and at this engine's
    // grid-unit rho ~ 0.1 it inflated q by ~8x, swamping the EOS --
    // live-measured: max_speed 11 -> 130, J pinned at the upper clamp 2.0,
    // fps 45 -> 12. The corrected form lands q at ~63 against an EOS
    // pressure scale of ~94, the intended same-order balance.
    //
    // Tait EOS's own real `c_sound` (this material's own `dp/drho`) --
    // `von_neumann_richtmyer_q` owns only the shared shock-viscosity form,
    // not any one EOS's sound speed, so it's computed here and passed in.
    // `eos_power` doubles as Kurapatenko's weak-shock gamma (a real,
    // disclosed stand-in, see `von_neumann_richtmyer_q`'s own doc).
    let density_ratio = 1.0 / j;
    let c2 = eos_stiffness
        * eos_power
        * crate::materials::utils::fast_pow(density_ratio, eos_power - 1.0)
        / rest_density;
    let c_sound = c2.max(0.0).sqrt();
    von_neumann_richtmyer_q(rest_density, j, div_v, grid_cell_size, c_sound, eos_power)
}

/// Weakly-compressible Newtonian fluid (Tait EOS + deviatoric viscosity).
/// Refs: Becker & Teschner 2007 (WCSPH), Hu et al. 2018 (MLS-MPM).
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub eos_stiffness: f32,
    pub eos_power: f32,
    /// Floor on the Tait EOS pressure -- prevents unbounded negative
    /// (tensile) pressure at a free surface, where the raw EOS formula has
    /// no restoring force in real fluids (cavitation, not sustained
    /// tension). Real, precedented value: `tmp/sparkl`'s
    /// `MonaghanSphEos::max_neg_pressure` uses the same `-0.1` clamp on
    /// `Ktait0*((rho/rho0)^gamma - 1)`, and `tmp/incremental_mpm`'s MLS-MPM
    /// fluid solver (Unity/C#) uses the identical constant with an honest
    /// author's-own comment ("i clamped it as a bit of a hack") -- this
    /// engine's default traces to that same real, working, if pragmatic,
    /// precedent, not an arbitrary guess. Bounds pressure SIGN/magnitude
    /// only -- does not bound `density`/`volume`'s own kinematic drift (see
    /// GPU/CPU parity work in `basic_fluids_gpu_blank_render_unconfirmed`
    /// memory, 2026-08-15, for the separate mechanism that still needs).
    pub pressure_floor: f32,
    /// Specific heat capacity `c_p`, J/(kg*K). 0 (the default) means
    /// undeclared -- see `MaterialModel::specific_heat_j_kg_k`. Liquid water
    /// is 4182 at 25 C (CRC Handbook); set it the same way `bulk_viscosity`
    /// and `pressure_floor` are set, after construction.
    pub specific_heat_j_kg_k: f32,
    /// Measured optical coefficients for this material, or `None` when the
    /// caller has not supplied any.
    ///
    /// Deliberately NOT a substance flag. A material model never decides
    /// that it "is water" because its numbers happen to look like water's;
    /// it carries whatever measured absorption and scattering the caller
    /// declares, and a name for the result is a label applied on top, never
    /// a branch inside the physics. `matter::materials::optical` holds the
    /// measured datasets to fill this with (`pure_water`, `dry_quartz_sand`,
    /// ...); anything else measured is equally valid here.
    ///
    /// `None` is the honest default: no spectrum was measured, so the
    /// renderer is told nothing rather than being handed an invented one.
    pub optics: Option<crate::energy::radiation::OpticalCoefficientsSi>,
    pub min_density: f32,
    pub min_volume: f32,
    /// Thermal thinning: µ_eff = dynamic_viscosity · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = isothermal. Positive values make the fluid flow easier when hot.
    pub thermal_viscosity_coeff: f32,
    /// Bulk viscosity ζ (second viscosity, Pa·s in physical units).
    ///
    /// Adds τ += ζ·(∇·v)·I to Kirchhoff stress -- damps compression waves (acoustic damping).
    /// Physical: Navier-Stokes second viscosity, distinct from shear viscosity µ.
    /// Stokes assumption (ζ=0) holds for dilute ideal gases; real liquids have ζ > 0.
    /// For water: ζ ≈ 3e-3 Pa·s (Dukhin & Goetz 2009). In simulation units set to
    /// ~0.5–5× dynamic_viscosity. 0.0 = no acoustic damping.
    pub bulk_viscosity: f32,
    /// Surface tension coefficient γ (N/m in physical units).
    ///
    /// Adds isotropic Kirchhoff stress τ += γ·J·I -- continuum surface energy ψ = γ·J.
    /// Reference: Ziran 2020, `SurfaceTension.h` (Chenfanfu Jiang group).
    ///
    /// **Limitation**: curvature-free. Young-Laplace gives Δp = γ·κ (interface curvature κ),
    /// but MPM particles carry no interface normal. This term resists volumetric compression
    /// isotropically -- sufficient for cohesion/droplet stability, not for curvature-driven
    /// flow (e.g. Rayleigh-Plateau instability). 0.0 = disabled.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    ///
    /// Damps residual sloshing and slow plastic creep without affecting fast flow.
    /// 0.0 = off (default). 0.05–0.2 for water, 0.1–0.5 for mud/viscous fluids.
    /// Implemented in the GPU shader via the `dp_h0` slot (unused for fluids).
    pub settling_damping: f32,
}

impl NewtonianFluidMaterial {
    /// Construct directly from grid-native parameters -- NOT SI units.
    /// Prefer [`Self::low_viscosity`] for a real-water preset, or the
    /// [`FromSI`] impl on this type (via `Fluid` properties) for a real
    /// SI-to-grid conversion from measured density/viscosity/stiffness.
    pub const fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            pressure_floor: -0.1,
            specific_heat_j_kg_k: 0.0,
            optics: None,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            thermal_viscosity_coeff: 0.0,
            bulk_viscosity: 0.0,
            surface_tension_coeff: 0.0,
            settling_damping: 0.0,
        }
    }

    /// Low-viscosity preset: γ=7, µ=1e-3 Pa·s. Corresponds to water at 20°C.
    ///
    /// `eos_stiffness` controls incompressibility -- higher = stiffer; 1e4 works
    /// well at emerge's default grid scale. Reference: Becker & Teschner 2007 §4.
    pub fn low_viscosity(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 1.0e-3, eos_stiffness, 7.0)
    }

    /// Weakly-compressible variant: caps sound speed at `c_ref_m_s`.
    ///
    /// Use `c_ref_m_s = 10 * v_max_m_s` (WCSPH rule) to limit compressibility to ~1%.
    /// `rho_kg_m3` and `eta_pa_s` are the fluid's SI density and viscosity.
    pub fn weakly_compressible(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        c_ref_m_s: f32,
        config: &crate::SimConfig,
    ) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions"
        // (the original real-fluid measurement this exponent is drawn from); used
        // identically in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994;
        // Becker & Teschner 2007, already cited elsewhere in this project).
        //
        // Real, confirmed regression fix (2026-08-29, independent
        // git-history verification): this exact
        // real-SI-direct form (no `scale_stress`/`scale_visc`, matching
        // `IdealGasMaterial::from_physical`'s own already-correct convention
        // -- solver time is already real seconds and positions are grid
        // cells, so stress/viscosity stay raw SI, only density converts via
        // `dx^2`) existed in this exact file as of commit `cac544b`
        // (2026-08-11), then was SILENTLY LOST one day later by `57b83dc`
        // ("restore pre-cac544b material state"), a wholesale revert that
        // was only meant to restore the J clamp/pressure floor/settling
        // damping but reverted the whole file to an even older state,
        // sweeping this separate, correct fix away with it -- that same
        // revert's own commit message disclosed "very little motion...
        // likely over-damped" as a known, unresolved side effect, unknowingly
        // describing this exact bug three weeks before it was root-caused.
        // The stale `scale_stress`/`dt_seconds`-based form silently made
        // this scene's water EOS stiffness ~4.4 million times too soft and
        // its viscosity ~296 million times too large (verified live,
        // `project_gas_bulk_viscosity_shipped_steam_lag_unresolved` memory) --
        // not a tuning gap, a real, confirmed regression, restored here.
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "weakly_compressible requires a positive dx_meters"
        );
        let rho_grid = rho_kg_m3 * config.dx_meters * config.dx_meters;
        let tait_b_pa = rho_kg_m3 * c_ref_m_s * c_ref_m_s / GAMMA;
        let mut material = Self::new(rho_grid, eta_pa_s, tait_b_pa, GAMMA);
        // REAL FIX (2026-09-17): same root bug as `FromSI::from_physical`'s
        // own fix just above -- `Self::new`'s `pressure_floor: -0.1` default
        // is a bare, unconverted grid-unit constant. This constructor's own
        // convention keeps stress/viscosity RAW SI (see this function's own
        // doc above -- no `scale_stress` here, unlike `from_physical`), so
        // the fix must match: the real cavitation pressure stays raw SI Pa
        // too, not run through `scale_stress` (which would double-convert
        // and be wrong for this specific constructor's units).
        material.pressure_floor = -100_000.0; // real dissolved-gas cavitation onset, Pa gauge
        material
    }
}

impl FromSI<NewtonianFluid> for NewtonianFluidMaterial {
    /// `rest_density` defaults to `props.rho_kg_m3`. Caller should adjust if
    /// particle mass/volume don't match the SI density.
    fn from_physical(props: &NewtonianFluid, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions";
        // standard in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(props.eta_pa_s, props.rho_kg_m3, config);
        let eos = scale_stress(props.bulk_modulus_pa / GAMMA, props.rho_kg_m3, config);
        // rest_density must be in the SAME units `particles.density[i]` actually
        // comes out in. A particle spawned at `spacing` carries
        // `grid_density * spacing^2` of mass in a `spacing^2` cell area, so the
        // density the solver measures is a RATIO against the scene's reference
        // density -- exactly 1.0 for a fluid at that reference. Getting this
        // wrong is not a small error: it makes the fluid believe it is spawned
        // pre-compressed, and the Tait EOS answers a `rho/rho_0` of 4 with a
        // pressure spike no CFL substep can bound.
        //
        // Do not reintroduce a `dx_meters^2` (or `/dt_seconds^2`) factor here.
        // Those pin a real fluid's EOS pressure at its floor regardless of
        // actual depth or compression -- see `SimConfig::grid_density`.
        let rho_grid = props.rho_kg_m3 / config.reference_density_kg_m3;
        let mut material = Self::new(rho_grid, visc, eos, GAMMA);
        // REAL FIX (2026-09-17): `Self::new`'s own `pressure_floor: -0.1`
        // default is a bare grid-unit constant, never SI-converted -- the
        // exact bug already found and patched per-demo in
        // `basic_fluids_gpu.rs`/`basic_fluids.rs` this week
        // (`HANDOFF_fluid_gpu_thin_layer_bug.md`, Tenth pass). Fixing it
        // only in those two call sites left this, the actually-documented
        // "real SI" construction path, still silently broken for any other
        // caller. Real cavitation onset for water in practice
        // (dissolved-gas nucleation, the standard engineering figure, not
        // the much higher pure-degassed lab value) is ~-100,000 Pa gauge --
        // converted through the SAME `scale_stress`/`stress_from_si_physical`
        // pipeline `eos` itself just used above, at THIS material's own
        // real `props.rho_kg_m3`, not assumed water.
        const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
        material.pressure_floor =
            scale_stress(REAL_CAVITATION_PRESSURE_PA, props.rho_kg_m3, config);
        material
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    // Restored 2026-08-13 after a wholesale revert silently dropped it;
    // PERMANENT and required (no longer "temporary" -- the uncertainty that
    // word carried is resolved, the override is proven necessary and is
    // covered by this file's own tests). Historical detail kept because it
    // explains WHY the override is needed at all: this
    // material never overrode `init_particle` even in the true pre-
    // `cac544b` file (confirmed: `git show 6234d06:...` has no override
    // either) -- but the CURRENT (non-reverted) engine's spawn contract
    // relies on materials that own their volume/density state to set them
    // exactly here, overriding `SpawnRegion::precompute_initial_volumes`'s
    // own kernel-density estimate (a real, legitimate default for materials
    // that DON'T have an exact analytical initial state, but wrong for a
    // strict fluid, which does: V0 = mass/rest_density exactly). Without
    // this override, that kernel estimate was the only thing setting
    // `volume`/`density` at spawn, while `deformation_gradient` stayed at
    // Identity (J=1) -- an internally INCONSISTENT state
    // (`assert_owned_deformation_state` caught it live: "det(F)=1,
    // V/V0=0.528"). This restores the exact contract this demo's own spawn
    // comment already describes ("The strict fluid initializer then sets
    // V0=m/rho0 and rho=rho0").
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// Real fix, same mechanism as `IdealGasMaterial::init_particle_from_transition`'s
    /// own doc (found live 2026-08-18, `examples/basic_steam.rs`, water
    /// boiling into steam) -- but for the REVERSE direction, root-caused
    /// live 2026-08-28 from a real "gas cooling down explodes the
    /// particles" bug report. Without this override, condensing (e.g.
    /// steam -> water) fell back to the default `init_particle_from_transition`
    /// (delegates straight to `init_particle` above), which throws away
    /// `Simulation::apply_phase_transition`'s own real, continuous
    /// rebaseline (`initial_volume` = the particle's actual current volume
    /// as steam, `density` = mass over that real volume) and replaces it
    /// with water's rest-state formula at `deformation_gradient=IDENTITY`
    /// (`j=1`, zero pressure) -- making the particle's claimed volume jump
    /// instantly to `mass/rest_density(water)`, several times smaller than
    /// its real physical footprint a substep earlier, while that real
    /// footprint (position, spacing from still-gaseous neighbors) hasn't
    /// changed at all in the same instant. A diffuse condensation front
    /// then has freshly-condensed particles falsely claiming water's small
    /// rest volume sitting right next to neighbors still occupying steam's
    /// real, larger volume -- the Tait EOS reacts to that fabricated
    /// overcompression with a violent repulsive pressure spike (particles
    /// "exploding" apart), confirmed as the real cause, not assumed.
    ///
    /// Same fix as `IdealGasMaterial`'s: keep the reference volume TRUE
    /// (`mass/rest_density`, matching what `kirchhoff_stress`/`update_particle`
    /// already assume every substep), and instead set a STARTING
    /// deformation gradient reflecting the real compression ratio between
    /// the particle's actual prior volume (as whatever it transitioned
    /// FROM) and this material's true rest volume -- clamped to the SAME
    /// `[0.5, 2.0]` bound `update_particle` already enforces every
    /// subsequent substep (see that method's own comment: empirically
    /// verified load-bearing against a real drop test, min_j/max_j hit
    /// exactly, not vestigial), so the starting state is consistent with
    /// the ongoing dynamics from frame one, not a separate, inconsistent
    /// value later dynamics silently overwrite.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume).clamp(0.5, 2.0);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// Rest-state acoustic speed squared, `c^2 = B*gamma/rho0` (Tait EOS
    /// evaluated at `J = 1`).
    ///
    /// Restored 2026-08-13; PERMANENT and required -- without it the
    /// near-wall CFL gate silently degrades (see below). Kept documented
    /// because the failure it prevents is invisible, not because it is
    /// provisional. It was the THIRD
    /// trait method the wholesale pre-`cac544b` revert silently dropped
    /// (after `owns_deformation_volume_state` and `init_particle`) -- it
    /// postdates this file's restored form. Without it the near-wall CFL
    /// gate (`cfl.rs`) can't compute a Mach number, falls back to its fixed
    /// absolute threshold, and therefore fires identically at every flow
    /// speed -- exactly what
    /// `near_wall_gate_relaxes_when_measured_speed_predicts_this_much_compression`
    /// caught (dt_low_speed == dt_high_speed, no relaxation).
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.eos_stiffness > 0.0 && self.rest_density > 0.0 {
            Some(self.eos_stiffness * self.eos_power / self.rest_density)
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Density from F's own determinant (rho = rest_density / J), NOT the
        // grid-mass-gathered `particles.density[i]` this used before -- a
        // real CPU/GPU parity fix: the GPU fluid path (`p2g.wgsl`'s case 1u)
        // already uses this exact formula ("sparkl canonical, no grid-lag"
        // per its own comment), and `GranularFluidMaterial`'s CPU code
        // (the engine's other EOS-pressure material) already does too --
        // plain `NewtonianFluidMaterial` was the one inconsistent holdout.
        // Grid-mass density carries a real one-substep lag (P2G scatter ->
        // grid -> G2P gather, vs J which is already current this same
        // substep) and is blind to how it's actually used elsewhere in this
        // engine (GranularFluid, GPU) -- switching removes a real, disclosed
        // inconsistency, not just a style choice.
        //
        // Real, honest disclosure: the OLD grid-mass approach is exactly
        // what `hydrostatic_pressure_matches_rho_g_h`'s own doc measured
        // settling at ~1.3x rest_density (not the correct ~1.003x) --
        // whether J-based density changes that specific overshoot is NOT
        // yet re-measured (that test stays `#[ignore]`d); this fix is
        // motivated by real consistency across the engine, not a confirmed
        // fix for that specific still-open gap.
        //
        // Clamp density both ways: min prevents div-by-zero, max (2x rho0)
        // limits how far the EOS pressure response saturates under impact
        // overcompression. Keep this at 2x, not looser --
        // `fluid_spreads_more_than_elastic_under_gravity` (tests/accuracy.rs)
        // needs it (a looser clamp stops the fluid spreading at all).
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        let eff_viscosity = if self.thermal_viscosity_coeff > 0.0 {
            self.dynamic_viscosity
                * (-self.thermal_viscosity_coeff * particles.temperature[i]).exp()
        } else {
            self.dynamic_viscosity
        };
        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y; // = 2·tr(D) = 2·∇·v
        let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
        stress += eff_viscosity * strain_dev;

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I -- damps longitudinal/acoustic waves.
        // ∇·v ≈ div_v/2 (div_v here is trace of sym_strain = C+Cᵀ = 2D, so ∇·v = div_v/2).
        if self.bulk_viscosity > 0.0 {
            stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
        }

        if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            stress += Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j));
        }

        // Artificial (shock) viscosity -- a REAL PDE term, not a bandaid:
        // von Neumann & Richtmyer 1950 (LA-671) quadratic + Landshoff linear,
        // the standard shock-capturing pair for Lagrangian hydrocodes, gated
        // to compression only (`div(v) < 0`) so it vanishes identically in
        // smooth flow. Independently corroborated in `tmp/GeoTaichi`
        // (`MaterialModel.py::artifical_viscosity`, same von Neumann citation,
        // same compression gate, same linear+quadratic form) -- and its own
        // shipped Newtonian dam-break example enables it (`cL: 1.0, cQ: 2`).
        //
        // Restored here 2026-08-13: it had existed in `fluid_state.rs`, which
        // a wholesale revert to this file's pre-`cac544b` form deleted, so the
        // CPU fluid path silently lost it while `p2g.wgsl` kept its own copy.
        //
        // CORRECTION (2026-09-15, audit-found): the claim above ("now
        // closed") stopped being true on 2026-08-15 -- commit `7c7991f`
        // ("consolidate GPU/render backlog") silently deleted THIS SAME term
        // from `p2g.wgsl`'s fluid branch (and the granular-fluid branch's own
        // Kelvin-Voigt damping) alongside ~480 unrelated line changes, with
        // no disclosure that it had gone. Every GPU-run fluid/mud scene had
        // zero shock-capturing viscosity for a month with this doc still
        // claiming parity. Both restored 2026-09-15, ported from THIS
        // function's current form (not the stale pre-deletion GPU snapshot,
        // which still used the strong-shock coefficient this function has
        // since moved away from) -- see `p2g.wgsl`'s own comment at the
        // restoration site for the full account. Real, disclosed lesson: a
        // "closed" cross-language parity claim needs its own regression
        // test, not just a comment, to actually stay closed -- none existed
        // here, which is exactly how a large consolidation commit could
        // delete it without any test failing.
        // Reuses `j` (this function's own `det(F)`, computed above) rather
        // than recomputing `volume/initial_volume`. They are the SAME
        // quantity for a strict fluid -- `assert_owned_deformation_state`
        // exists precisely to enforce that they agree (to 2e-4) -- so this is
        // bit-equivalent, not an approximation. Saves one call with four
        // asserts and a division per particle per substep (~52k/frame at this
        // demo's 2912 particles x 18 substeps).
        let q = artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j,
            // `div_v` above is tr(C + C^T) = 2*div(v); this term wants the
            // true divergence.
            0.5 * div_v,
            // `kirchhoff_stress`'s trait signature carries no grid_cell_size;
            // every scene in this engine uses 1.0 (exact today, not an
            // approximation) -- same disclosed limitation the GPU copy has.
            1.0,
        );
        stress += Mat2::from_diagonal(Vec2::splat(-q));

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Real bug fixed 2026-08-06: this used to re-isotropize FROM THE OLD F's
        // own determinant, never actually applying the velocity-gradient update
        // every other material's `update_particle` does -- J was permanently
        // frozen at its spawn value (1.0) for the material's entire lifetime,
        // regardless of any real compression/expansion. Silent before 2026-08-04
        // (density came from the separate grid-mass estimate then), but became a
        // real, previously-undetected regression once `kirchhoff_stress` switched
        // to `rest_density/J` -- the EOS pressure term went completely inert
        // (density permanently == rest_density, pressure permanently ~0). Found
        // via `fluid_impact_shows_real_free_surface_splash_separation`
        // (`tests/physics_correctness.rs`): a hard floor impact showed max_j_seen
        // EXACTLY 1.0000 across 250 steps, not just close to it.
        // Real, disclosed regression fixed 2026-08-30: `det(I + dt*C)` (the
        // old `f_trial` this line used to build) is NOT rotation-invariant --
        // for a pure rigid rotation `C=[[0,-w],[w,0]]` (div(v)=tr(C)=0, no
        // real volume change should occur), `det(I+dt*C) = 1 + dt^2*w^2`, a
        // strictly POSITIVE expansion every single substep from pure O(dt^2)
        // discretization error, not real physics. Immediately baked in
        // permanently by the very next line's isotropization (nothing ever
        // reverses it), this is a real, measured, monotonic `detF`-max drift
        // confirmed live over thousands of frames, dynamics-independent
        // (kept climbing at an unchanged rate even after real convection/
        // vorticity had fully died to near-zero). The exact fix is the
        // continuity equation's own exponential solution for constant `C`
        // over a substep, `J_{n+1} = J_n * exp(dt*div(v))` -- exactly 1.0
        // for any rotation (div(v)=0 always, regardless of vorticity), and
        // matching `det(I+dt*C)` to first order for genuine compression/
        // expansion, so this is a strict correctness fix, not a behavior
        // change for real divergent flow. This exact formula previously
        // existed in a since-deleted `fluid_state.rs` module (added
        // `cac544b` 2026-08-11, silently lost the very next day by the SAME
        // wholesale revert, `57b83dc`, that also caused the
        // `weakly_compressible` units regression found earlier this
        // session) -- restored here, not reinvented.
        // `old_j` recovers the fluid's own scalar state from its ALREADY-
        // isotropic F (`s*I`, `det=s^2`) -- exactly this material's `j`
        // from the previous substep, the same equivalence
        // `assert_owned_deformation_state` already enforces elsewhere in
        // this file (bit-equivalent to `volume/initial_volume` to 2e-4).
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        // Real, measured 2026-08-14: this clamp is NOT dead weight from a
        // stiffness-derivation era this engine has since outgrown -- earlier
        // memory recorded it as "dormant" after the real EOS-stiffness fix
        // raised the observed floor to ~0.964 on a calm interactive scene,
        // but that was never checked against a genuinely violent one. It
        // is: `fluid_impact_shows_real_free_surface_splash_separation`
        // (`tests/physics_correctness.rs`, a 6x6 block dropped 20 units onto
        // a rigid floor at eos_stiffness=50) hits BOTH bounds EXACTLY --
        // min_j_seen=0.5000, max_j_seen=2.0000 -- over 250 real steps. Under
        // a hard impact this clamp is load-bearing, not vestigial; keep it.
        let j = (old_j * (dt * div_v).exp()).clamp(0.5, 2.0);
        let s = j.sqrt();
        *ctx.deformation_gradient =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        if self.settling_damping > 0.0 {
            *ctx.v *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
        // Real, disclosed 2026-08-04 fix: `stress_volume`/`timestep_bound` both
        // read `particles.volume`/`density` -- but until now this material never
        // wrote either. Both were left entirely to `estimate_particle_volumes`'s
        // grid-mass-kernel estimate (`spacetime::solver::density`), which is
        // real but has NO ceiling on compaction (`clamp_rarefied_volume` only
        // bounds rarefaction, i.e. it's a floor on density, not a ceiling) --
        // and unlike every plastic solid material (DP/Snow/NACC/etc, see
        // `sand.rs`'s own `*ctx.density = ctx.mass / v` at the end of every
        // substep), nothing here ever pulled a drifting estimate back to a
        // physically-bounded value. In a settling/compacting scene the
        // per-substep noise can only ratchet UP (nothing corrects it back
        // down at near-zero divergence), which silently stiffens this
        // material's own CFL bound over a long horizon -- the real, root
        // mechanism behind `mixture_sand_water.rs`'s `dropped`/min_dt-clamp
        // issue (see `mixture_sand_water_explosion_investigation` memory).
        // Fix: self-correct every substep from the SAME already-bounded
        // formula `stress()` already uses for pressure (`(rest_density/j)
        // .max(min_density).min(2*rest_density)`) -- real, symmetric,
        // nothing new invented, just applied where it was missing.
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    // Restored 2026-08-13; PERMANENT and required -- removing it froze GPU
    // water completely from frame 1 (live-confirmed, see below). Not
    // provisional. This trait
    // method did not exist before `cac544b` -- this file predates it, so
    // reverting the file wholesale silently dropped the override, leaving
    // the default `false`. REAL, LIVE-CONFIRMED bug this caused on GPU:
    // without this returning `true`, `basic_fluids_gpu.rs`'s water froze
    // completely from frame 1 (v~0, J=1.000 exactly, forever) -- most
    // likely because a kernel-density recompute this method exists to
    // suppress (see this file's own top-of-file doc: "biased at a free
    // surface... not a conservative thermodynamic state update") started
    // overwriting volume/density in a way that failed
    // `strict_fluid_state_is_admissible`'s internal consistency check in
    // `particles_update.wgsl` every single substep, silently freezing the
    // fluid branch's own state update via its early return. `true` restores
    // the correct, current-codebase-wide convention: this material owns its
    // own volume/density (see `update_particle` above), don't let a kernel
    // gather overwrite it.
    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn specific_heat_j_kg_k(&self) -> f32 {
        self.specific_heat_j_kg_k
    }

    /// Tait inverted. The state equation is
    /// `p = k * ((rho/rho_0)^gamma - 1)`, so the density in equilibrium at
    /// pressure `p` is `rho = rho_0 * (1 + p/k)^(1/gamma)`, and since MPM
    /// carries density as `rho = rho_0 / J`, the volume ratio is
    /// `J = (1 + p/k)^(-1/gamma)`.
    ///
    /// Returns `None` for a non-positive pressure: a free surface is
    /// already at `J = 1` and needs no correction, and a negative pressure
    /// here would mean tension, which this state equation does not model.
    fn hydrostatic_volume_ratio(&self, pressure: f32) -> Option<f32> {
        if !pressure.is_finite() || pressure <= 0.0 || self.eos_stiffness <= 0.0 {
            return None;
        }
        Some((1.0 + pressure / self.eos_stiffness).powf(-1.0 / self.eos_power))
    }

    /// Whatever the caller measured, verbatim. See the `optics` field: this
    /// model reports coefficients, it does not identify a substance.
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            thermal_viscosity_coeff: self.thermal_viscosity_coeff,
            // Free-surface J cap: GPU clamps det(F) to [J_MIN, volume_ratio_max].
            // 2.0 = realistic free-surface density (half rest_density with no restoring EOS force).
            volume_ratio_max: 2.0,
            pressure_floor: self.pressure_floor,
            specific_heat_j_kg_k: self.specific_heat_j_kg_k,
            bulk_viscosity: self.bulk_viscosity,
            dp_h0: self.settling_damping, // fluid repurposes dp_h0 for settling damping (DP unused)
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
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
        const MIN_DENSITY_RATIO: f32 = 1.0e-6;
        let density = density.max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(MIN_DENSITY_RATIO);

        let mut dt_bound = f32::INFINITY;

        // Acoustic timestep bound from EOS derivative dp/drho.
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Viscous diffusion bound for explicit integration -- combines
        // dynamic_viscosity AND bulk_viscosity (real regression fix,
        // external review: this used to bound only shear viscosity,
        // leaving bulk viscosity's own explicit-damping term with no
        // matching CFL bound, same real instability mechanism
        // `GranularFluidMaterial::timestep_bound`'s own doc already fixed
        // this for -- an explicit damping term whose dt*viscosity/mass
        // ratio is too large INJECTS energy instead of removing it. Both
        // terms multiply the same velocity-gradient-derived stress, so
        // combining them linearly is the real, conservative bound, not an
        // approximation.
        let combined_viscosity = self.dynamic_viscosity + self.bulk_viscosity.max(0.0);
        if combined_viscosity > 0.0 {
            let kinematic_viscosity = combined_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    /// `false`, not the `true` this file carried (2026-08-13).
    ///
    /// This override is a leftover from the pre-`cac544b` design, where this
    /// material really did read `particles.density[i]` straight out of the
    /// kernel-density gather. It no longer does: `init_particle` seeds
    /// `rho = rho0/J` analytically and `update_particle` maintains it every
    /// substep, which is exactly what `owns_deformation_volume_state -> true`
    /// declares. The trait's own doc states the rule directly -- "Strict
    /// WC-MPM liquids do *not* [consume a kernel-density measurement]: their
    /// EOS state is rho = rho0 / J, owned together with V = V0 J" -- so
    /// returning `true` here contradicted this same impl's other two methods.
    ///
    /// Keeping it `true` was also pure waste, not merely redundant:
    /// `estimate_particle_volumes` runs a full `grid.clear()` + mass scatter
    /// over EVERY particle, and only then skips each one that owns its own
    /// volume state (`density.rs`'s own `continue`) -- so the entire pass was
    /// computed and thrown away. Live-measured on `basic_fluids_gui.rs`:
    /// `density_us` was 12000-13700 us of a ~54000 us step, ~24%, the
    /// second-largest cost in the whole solver, for zero effect on state.
    fn needs_density_recompute(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod si_construction_tests {
    use super::*;

    /// Real regression guard (2026-08-29): `weakly_compressible` must keep
    /// stress/viscosity in raw SI units (matching `IdealGasMaterial::
    /// from_physical`'s own already-correct convention) and convert ONLY
    /// density via `dx^2` -- NOT route through `scale_stress`/`scale_visc`
    /// (a stale, `dt_seconds`-based convention gravity itself no longer
    /// uses). This exact test existed in this exact file as of commit
    /// `cac544b` (2026-08-11), was silently lost the next day by a
    /// wholesale revert (`57b83dc`) that only meant to restore unrelated
    /// fields (J clamp/pressure floor/settling damping), and stayed lost
    /// for weeks -- a real, confirmed regression (found live 2026-08-29 while
    /// investigating a still-open water-compression symptom, then verified
    /// independently against this file's own git
    /// history), not a hypothetical. Restored here as the real regression
    /// guard it always should have stayed.
    #[test]
    fn si_constructor_preserves_pressure_and_viscosity_units() {
        let cfg = crate::SimConfig::earth(32, 0.01, 0.1);
        let material = NewtonianFluidMaterial::weakly_compressible(1000.0, 1.0e-3, 20.0, &cfg);
        assert!((material.rest_density - 0.1).abs() < 1.0e-7);
        assert!((material.dynamic_viscosity - 1.0e-3).abs() < 1.0e-9);
        assert!((material.eos_stiffness - (1000.0 * 20.0 * 20.0 / 7.0)).abs() < 1.0e-3);
    }

    /// Real regression guard (2026-09-17): `Self::new`'s own `pressure_floor:
    /// -0.1` default is a bare, unconverted grid-unit constant -- the exact
    /// bug already found and manually patched per-demo in
    /// `basic_fluids_gpu.rs`/`basic_fluids.rs` (`HANDOFF_fluid_gpu_thin_
    /// layer_bug.md`, Tenth pass). Both real SI construction paths must
    /// carry a properly-converted, real cavitation pressure (~-100,000 Pa
    /// gauge, dissolved-gas nucleation onset) by default, not leave it to
    /// every caller to remember to override manually -- each in ITS OWN
    /// constructor's real unit convention (`weakly_compressible` keeps
    /// stress raw SI; `from_physical` routes through `scale_stress`, same
    /// as its own `eos_stiffness`).
    #[test]
    fn si_constructors_convert_pressure_floor_not_just_stiffness() {
        let cfg = crate::SimConfig::earth(32, 0.01, 0.1);

        let wc = NewtonianFluidMaterial::weakly_compressible(1000.0, 1.0e-3, 20.0, &cfg);
        assert!(
            (wc.pressure_floor - (-100_000.0)).abs() < 1.0e-3,
            "weakly_compressible: pressure_floor={} -- expected raw SI -100000.0 Pa, \
             not the unconverted default -0.1",
            wc.pressure_floor
        );

        let props = NewtonianFluid {
            rho_kg_m3: 1000.0,
            eta_pa_s: 1.0e-3,
            bulk_modulus_pa: 1000.0 * 20.0 * 20.0,
        };
        let si = NewtonianFluidMaterial::from_physical(&props, &cfg);
        let expected = cfg.stress_from_si_physical(-100_000.0, props.rho_kg_m3);
        assert!(
            (si.pressure_floor - expected).abs() < 1.0e-6,
            "from_physical: pressure_floor={} -- expected {expected} (real cavitation \
             pressure through the same scale_stress conversion eos_stiffness uses), not \
             the unconverted default -0.1",
            si.pressure_floor
        );
        assert_ne!(
            si.pressure_floor, -0.1,
            "from_physical must not silently keep Self::new's raw grid-unit default"
        );
    }
}

#[cfg(test)]
mod volume_integration_tests {
    use super::*;
    use crate::particle::{Particle, Particles};

    fn particle_with_f(f: Mat2) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = f.determinant();
        p.density = 1.0;
        Particles::from(vec![p])
    }

    /// Real regression guard (2026-08-30): a pure rigid rotation carries
    /// zero divergence (`tr(C)=0`) and must leave `J=det(F)` EXACTLY
    /// unchanged -- rotation alone never compresses or expands anything.
    /// The bug this guards against: `update_particle` used to build
    /// `f_trial=(I+dt*C)*F_old` and take `det(f_trial)` directly, which for
    /// this exact `C` gives `det(f_trial) = 1 + dt^2*omega^2`, a strictly
    /// POSITIVE expansion every substep from pure O(dt^2) discretization
    /// error -- confirmed live over thousands of frames as a real, slow,
    /// monotonic `detF`-max drift, dynamics-independent (kept climbing at
    /// an unchanged rate even once real vorticity had fully died to near-
    /// zero). Fixed with the exact exponential solution
    /// `J_new=J_old*exp(dt*div(v))`, which is identically 1.0 for any
    /// rotation regardless of `omega`. Same root cause, same fix, as the
    /// `weakly_compressible` units regression -- both lost the same day
    /// (`57b83dc`) from the same `fluid_state.rs` module `cac544b`
    /// introduced (`log_j = old_j.ln() + dt*div_v; j = log_j.exp()`).
    #[test]
    fn rigid_rotation_leaves_j_exactly_unchanged() {
        let mat = NewtonianFluidMaterial::new(1.0, 0.0, 100.0, 7.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.01;
        let omega = 5.0_f32; // deliberately large -- the old bug scales as omega^2
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
            for _ in 0..500 {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-5,
            "500 substeps of pure rotation (omega={omega}) must leave J exactly at 1.0, \
             got {j} -- the old det(I+dt*C) bug would give a real, measurable expansion here"
        );
    }

    /// Real regression guard (2026-08-30): a constant, uniform dilation
    /// (`C = k*I`, `div(v) = 2k` in 2D) must integrate to EXACTLY the
    /// continuity equation's own exponential solution,
    /// `J_new = J_old * exp(N*dt*2k)`, for constant `C` held over `N`
    /// substeps -- not the old `det(I+dt*C)`-per-step approximation, which
    /// only agrees with this to first order and drifts at higher `dt*k`.
    #[test]
    fn constant_dilation_matches_exact_exponential_solution() {
        let mat = NewtonianFluidMaterial::new(1.0, 0.0, 100.0, 7.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.001;
        let k = 2.0_f32;
        const N: i32 = 50;
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_diagonal(Vec2::splat(k));
            for _ in 0..N {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        let expected = (N as f32 * dt * 2.0 * k).exp();
        assert!(
            (j - expected).abs() / expected < 1.0e-4,
            "constant dilation over {N} substeps must match J_old*exp(N*dt*2k) exactly \
             (continuity equation's own solution for constant C) -- got {j}, expected {expected}"
        );
    }
}

#[cfg(test)]
mod transition_continuity_tests {
    use super::*;

    /// The real bug this override fixes: condensing from a much-LESS-dense
    /// prior material (e.g. steam, same mass spread over a much larger real
    /// volume) must NOT reset straight to `deformation_gradient=IDENTITY`
    /// (`j=1`, the old default-`init_particle`-fallback behavior) -- it must
    /// clamp to the SAME real `[0.5, 2.0]` bound `update_particle` already
    /// enforces every substep, landing at the upper bound here since the
    /// real ratio (6.0) is far outside it.
    #[test]
    fn condensing_from_a_much_larger_prior_volume_clamps_to_the_upper_compression_bound() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 6.0; // real prior (steam) volume: 6x this material's true rest volume
        water.init_particle_from_transition(&mut p);

        let true_initial_volume = p.mass / water.rest_density;
        assert_eq!(p.initial_volume, true_initial_volume);
        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 2.0).abs() < 1.0e-4,
            "ratio 6.0 is far outside [0.5, 2.0], must clamp to the upper bound 2.0, got {j}"
        );
        assert!(
            (p.volume - true_initial_volume * 2.0).abs() < 1.0e-4,
            "volume must be true_initial_volume * clamped j, not an instant jump to \
             true_initial_volume alone: got {}",
            p.volume
        );
    }

    /// Same mechanism, opposite direction: transitioning from a much-MORE-
    /// dense prior material clamps to the lower compression bound instead
    /// of silently allowing an unbounded compression spike.
    #[test]
    fn transitioning_from_a_much_smaller_prior_volume_clamps_to_the_lower_compression_bound() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 0.1; // real prior volume: far denser than water's own rest state
        water.init_particle_from_transition(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 0.5).abs() < 1.0e-4,
            "ratio 0.1 is far outside [0.5, 2.0], must clamp to the lower bound 0.5, got {j}"
        );
    }

    /// Regression parity: a prior material with the SAME real rest density
    /// (prior volume already equal to this material's true rest volume)
    /// must land at j=1 exactly -- no artificial jump introduced where none
    /// is physically warranted.
    #[test]
    fn transitioning_from_an_already_matching_volume_introduces_no_artificial_jump() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 1.0; // already equal to mass/rest_density
        water.init_particle_from_transition(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-5,
            "no real density mismatch should mean no artificial jump: got j={j}"
        );
    }
}

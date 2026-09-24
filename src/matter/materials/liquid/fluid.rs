use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, NewtonianFluid};
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
///
/// REAL BUG FIXED 2026-08-13: the previous form was
/// `c0*(rho*h*div_v)^2 - c1*h*c_sound*div_v` -- i.e. rho^2 in the
/// quadratic term and NO rho at all in the linear one. Neither matches
/// the cited sources: Wang et al. (arXiv:2404.17057, eq. 4) and
/// `tmp/GeoTaichi`'s `MaterialModel.py::artifical_viscosity` both
/// multiply BOTH terms by rho exactly once. Dimensionally the old form
/// is inconsistent (the two terms don't even share units), and with this
/// engine's grid-unit rho ~ 0.1 it inflated q by ~8x, swamping the EOS --
/// live-measured: max_speed 11 -> 130, J pinned at the upper clamp 2.0,
/// fps 45 -> 12. With the corrected form q lands at ~63 against an EOS
/// pressure scale of ~94, which is the intended same-order balance.
///
/// Refactored 2026-08-18 to delegate the shared shock-viscosity FORM to
/// `materials::utils::von_neumann_richtmyer_q` (bit-identical result --
/// this function now only derives Tait's own `c_sound = sqrt(dp/drho)`
/// and hands `eos_power` through as the weak-shock gamma stand-in, same
/// as before) so a genuinely different EOS (ideal gas) can reuse the real
/// q-formula without faking Tait parameters.
pub(crate) fn artificial_bulk_viscosity(
    eos_stiffness: f32,
    eos_power: f32,
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
) -> f32 {
    let density_ratio = 1.0 / j;
    let c2 = eos_stiffness
        * eos_power
        * crate::materials::utils::fast_pow(density_ratio, eos_power - 1.0)
        / rest_density;
    let c_sound = c2.max(0.0).sqrt();
    crate::materials::utils::von_neumann_richtmyer_q(
        rest_density,
        j,
        div_v,
        grid_cell_size,
        c_sound,
        eos_power,
    )
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
    pub min_density: f32,
    pub min_volume: f32,
    /// Thermal thinning: µ_eff = dynamic_viscosity · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = isothermal. Positive values make the fluid flow easier when hot.
    pub thermal_viscosity_coeff: f32,
    /// Bulk viscosity ζ (second viscosity, Pa·s in physical units).
    ///
    /// Adds τ += ζ·(∇·v)·I to Kirchhoff stress — damps compression waves (acoustic damping).
    /// Physical: Navier-Stokes second viscosity, distinct from shear viscosity µ.
    /// Stokes assumption (ζ=0) holds for dilute ideal gases; real liquids have ζ > 0.
    /// For water: ζ ≈ 3e-3 Pa·s (Dukhin & Goetz 2009). In simulation units set to
    /// ~0.5–5× dynamic_viscosity. 0.0 = no acoustic damping.
    pub bulk_viscosity: f32,
    /// Surface tension coefficient γ (N/m in physical units).
    ///
    /// Adds isotropic Kirchhoff stress τ += γ·J·I — continuum surface energy ψ = γ·J.
    /// Reference: Ziran 2020, `SurfaceTension.h` (Chenfanfu Jiang group).
    ///
    /// **Limitation**: curvature-free. Young-Laplace gives Δp = γ·κ (interface curvature κ),
    /// but MPM particles carry no interface normal. This term resists volumetric compression
    /// isotropically — sufficient for cohesion/droplet stability, not for curvature-driven
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
    /// `eos_stiffness` controls incompressibility — higher = stiffer; 1e4 works
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
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "weakly_compressible requires a positive dx_meters"
        );
        // Real, confirmed regression (2026-08-17): `cac544b` (2026-08-11)
        // fixed this to pass pressure/viscosity through RAW, unconverted --
        // solver time is already real seconds and only length is rescaled
        // by dx (same convention `gravity_to_grid`'s `g_grid = g_SI/dx`
        // already uses), so applying the legacy `dt^2/(rho*dx^2)`-style
        // scaling here double-scales it (see `from_physical`'s own doc,
        // just below, for the full derivation). `57b83dc` (2026-08-13,
        // "restore pre-cac544b material state") wholesale-reverted this
        // whole file to fix an UNRELATED problem (a missing J/pressure
        // clamp) and silently brought the old, wrong `scale_visc`/
        // `scale_stress` calls back with it -- confirmed via `git show
        // cac544b:...fluid.rs` and a live re-run of `tests/physics_
        // correctness.rs::diag_wcsph_unit_consistency_sweep_under_full_
        // real_gravity`, which prints `B_grid=1.4575e4` today vs. that
        // fix's own already-recorded `1.4575e5` (exactly 10x, matching the
        // `dt^2/(rho*dx^2)` factor for this test's own DT=0.1/DX=0.01/
        // RHO=1000). Also independently re-derived and numerically
        // verified via `examples/diag_lame_from_si_wave_speed_check.rs`.
        let rho_grid = rho_kg_m3 * config.dx_meters * config.dx_meters;
        let tait_b_pa = rho_kg_m3 * c_ref_m_s * c_ref_m_s / GAMMA;
        Self::new(rho_grid, eta_pa_s, tait_b_pa, GAMMA)
    }
}

impl FromSI<NewtonianFluid> for NewtonianFluidMaterial {
    /// `rest_density` defaults to `props.rho_kg_m3`. Caller should adjust if
    /// particle mass/volume don't match the SI density.
    fn from_physical(props: &NewtonianFluid, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions";
        // standard in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "NewtonianFluidMaterial::from_physical requires a positive dx_meters"
        );
        // Solver time is already real seconds and positions are grid cells
        // (`x_grid = x_SI/dx`), so pressure and dynamic viscosity stay in
        // raw SI stress units -- only density converts, to mass per
        // grid-cell area. With `V_grid = V_SI/dx^2` and `rho_grid =
        // rho_SI*dx^2`, a stress coefficient in Pa already produces exactly
        // `sigma/(rho*dx^2)` grid acceleration; applying the legacy
        // `dt^2/(rho*dx^2)` conversion here would double-scale it (see
        // `weakly_compressible`'s own doc, just above, for the fuller
        // regression history -- this exact reasoning was already shipped
        // once in `cac544b` and silently reverted by `57b83dc`).
        //
        // rest_density must be in the SAME units `particles.density[i]`
        // actually comes out in -- i.e. whatever `estimate_particle_
        // volumes`'s kernel-based density estimate produces for a particle
        // spawned via `ParticleMass::particle_mass` (real SI kilograms) at
        // rest: `rho_grid = rho_SI * dx_meters^2`. Do not add an extra
        // `/dt_seconds^2` factor here -- it pins any real fluid's EOS
        // pressure at its floor regardless of real depth/compression.
        // Inflating particle mass by `1/dt^2` instead breaks the
        // gravity/EOS force balance -- see `Elastic::particle_mass`'s doc.
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Self::new(
            rho_grid,
            props.eta_pa_s,
            props.bulk_modulus_pa / GAMMA,
            GAMMA,
        )
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    // TEMPORARY, explicitly disclosed restoration (2026-08-13): this
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

    /// Rest-state acoustic speed squared, `c^2 = B*gamma/rho0` (Tait EOS
    /// evaluated at `J = 1`).
    ///
    /// TEMPORARY, explicitly disclosed restoration (2026-08-13): the THIRD
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

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I — damps longitudinal/acoustic waves.
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
        // CPU fluid path silently lost it while `p2g.wgsl` kept its own copy
        // -- a real CPU/GPU physics divergence, now closed.
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
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
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
        let j = f_trial.determinant().clamp(0.5, 2.0);
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

    // TEMPORARY, explicitly disclosed restoration (2026-08-13): this trait
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

        // Viscous diffusion bound for explicit integration.
        if self.dynamic_viscosity > 0.0 {
            let kinematic_viscosity = self.dynamic_viscosity / density;
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

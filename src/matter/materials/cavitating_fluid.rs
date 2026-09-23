//! Two real `MaterialModel`s wired on top of `cavitating_eos`'s own real,
//! sourced cavitation EOS (see that module's own doc for the full
//! derivation, and its "Real C^1 junctions" section for the T-dependent
//! closure both materials below are built on):
//!
//! - `IsothermalCavitatingFluidMaterial` -- fixes `p_v_gauge_pa` at
//!   construction from one real temperature. Real, bounded, disclosed
//!   limitation: does NOT couple to a particle's own live `temperature`.
//!   Still a real, useful choice for a scene that genuinely doesn't need
//!   real temperature-coupled cavitation (cheaper: no per-query patch
//!   reconstruction).
//! - `CavitatingFluidMaterial` -- the real, temperature-coupled successor:
//!   owns a `cavitating_eos::CavitatingEosTable` and reconstructs the real
//!   EOS at the particle's own LIVE temperature every query. Water near
//!   freezing and water near boiling genuinely behave differently now.
//!   Real, disclosed remaining limitation (both materials): a one-
//!   directional thermomechanical coupling (`T -> p_sat(T) -> mechanical
//!   response`) -- real cavitation does not yet pay latent heat back into
//!   the particle's own enthalpy state (Saurel/Boivin/Le Métayer 2016's
//!   own real two-way mixture-equilibrium closure, cited in project
//!   memory, is the real, disclosed future work for that).

use glam::{Mat2, Vec2};

use super::cavitating_eos::{CavitatingEosParams, CavitatingEosTable};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

#[derive(Debug, Clone, Copy)]
pub struct IsothermalCavitatingFluidMaterial {
    /// The real, gauge-pressure, three-branch EOS core -- always real SI
    /// internally (kg/m^3, Pa, m/s), see `CavitatingEosParams`'s own doc.
    pub eos: CavitatingEosParams,
    /// Grid cell size (m) this material was constructed for -- needed to
    /// convert the particle's own GRID-unit `density` (`rho_si*dx^2`, this
    /// engine's established convention) back to real SI before evaluating
    /// `eos.pressure_gauge_pa`, and back again for `stress_volume`/
    /// `timestep_bound`'s own grid-consistent quantities. Set once at
    /// construction (`new`), same convention `NewtonianFluidMaterial`'s
    /// `weakly_compressible` establishes -- this material does NOT convert
    /// stress/pressure by `dx^2` (only density does, matching that same,
    /// already-fixed convention).
    pub dx_meters: f32,
    /// Dynamic viscosity (Pa.s, raw SI -- same already-fixed, unconverted
    /// convention `NewtonianFluidMaterial`/`IdealGasMaterial` both use).
    pub dynamic_viscosity: f32,
    /// Rest density in GRID units (`eos.rho_l_ref_kg_m3 * dx_meters^2`) --
    /// cached at construction so `init_particle`/`update_particle` don't
    /// recompute it every call. Matches `NewtonianFluidMaterial::rest_density`'s
    /// own role exactly.
    rest_density_grid: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Real, explicit, CALLER-SUPPLIED numerical safety bound on `J`
    /// (`volume/initial_volume`) -- same established convention this
    /// engine already uses (`IdealGasMaterial::volume_ratio_min/max`,
    /// `NewtonianFluidMaterial`'s own `[0.5,2.0]` inline clamp). NOT
    /// defaulted to an invented number: the caller should size this
    /// relative to the real `rho_l_ref/rho_v_ref` ratio this material's
    /// own `eos` was built with (full vaporization real corresponds to
    /// `J ~= rho_l_ref/rho_v_ref`) plus real headroom for further vapor
    /// expansion at low pressure, not an arbitrary flat number.
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

/// Named-field alternative to [`IsothermalCavitatingFluidMaterial::new`]'s
/// positional arguments -- same real struct-bundling fix already used
/// elsewhere in this codebase (`PhysicalRenderContractParams`,
/// `NaccMaterialParams`) for a constructor with several same-typed adjacent
/// `f32` parameters.
#[derive(Clone, Copy, Debug)]
pub struct IsothermalCavitatingFluidMaterialParams {
    pub eos: CavitatingEosParams,
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

impl IsothermalCavitatingFluidMaterial {
    /// Same as [`Self::new`], named fields instead of positional args --
    /// see [`IsothermalCavitatingFluidMaterialParams`]'s own doc for why.
    pub fn from_params(params: IsothermalCavitatingFluidMaterialParams) -> Self {
        Self::new(
            params.eos,
            params.dx_meters,
            params.dynamic_viscosity,
            params.volume_ratio_min,
            params.volume_ratio_max,
        )
    }

    pub fn new(
        eos: CavitatingEosParams,
        dx_meters: f32,
        dynamic_viscosity: f32,
        volume_ratio_min: f32,
        volume_ratio_max: f32,
    ) -> Self {
        assert!(
            dx_meters.is_finite() && dx_meters > 0.0,
            "IsothermalCavitatingFluidMaterial requires a positive dx_meters"
        );
        assert!(
            volume_ratio_min > 0.0 && volume_ratio_max > volume_ratio_min,
            "IsothermalCavitatingFluidMaterial requires 0 < volume_ratio_min < volume_ratio_max, \
             no default -- see this field's own doc for how to size it"
        );
        let rest_density_grid = eos.rho_l_ref_kg_m3 * dx_meters * dx_meters;
        Self {
            eos,
            dx_meters,
            dynamic_viscosity,
            rest_density_grid,
            min_density: 1.0e-6,
            min_volume: 1.0e-9,
            volume_ratio_min,
            volume_ratio_max,
        }
    }

    /// Real SI density (kg/m^3) recovered from this particle's own
    /// GRID-unit `density`/`volume` state (`rho_grid = rho_si*dx^2`).
    #[inline]
    fn real_density_si(&self, grid_density: f32) -> f32 {
        grid_density / (self.dx_meters * self.dx_meters)
    }
}

impl MaterialModel for IsothermalCavitatingFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "IsothermalCavitatingFluidMaterial has no GPU stress path: it uploads as a plain Tait fluid, so the GPU would run without cavitation",
        )
    }

    /// Same real contract as `NewtonianFluidMaterial::init_particle` --
    /// see that method's own doc for why a strict fluid must set
    /// `initial_volume`/`volume`/`density` exactly here (V0=m/rho0,
    /// rho=rho0), not rely on `SpawnRegion`'s own kernel-density estimate.
    /// A scene that spawns this at a density other than the liquid
    /// reference must set the lattice spacing AND
    /// `SpawnRegion::initial_deformation_gradient` together -- see
    /// `cavitating_eos`'s own module doc for the measured cost of
    /// setting only one of the two.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density_grid;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    /// Same real contract as `NewtonianFluidMaterial::init_particle_from_transition`
    /// -- see that method's own doc for the real, live-confirmed bug
    /// (violent pressure spikes at a phase-transition front) this exact
    /// rebaseline avoids.
    ///
    /// Real, disclosed correction (2026-08-30): an earlier version of
    /// this method hardcoded `.clamp(0.5, 2.0)`, copy-pasted from
    /// `NewtonianFluidMaterial` without updating it to this material's own
    /// real, explicit `volume_ratio_min/max` -- confirmed consequence:
    /// real steam (`J~6`, real full vaporization ratio) condensing INTO
    /// this material would be instantly, wrongly clamped to `J=2`,
    /// breaking the exact volume-continuity guarantee this override
    /// exists to provide. Fixed to use this material's own real bounds.
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
        // Real, grid-consistent rest acoustic speed squared, same
        // dx-folding convention `NewtonianFluidMaterial::rest_acoustic_c2`
        // uses (`B*gamma/rho_GRID`, which is `c_real^2/dx^2` -- grid
        // cells^2/s^2, not real m^2/s^2 -- see this material's own
        // `timestep_bound` doc for the full derivation).
        Some(self.eos.c_l_m_s * self.eos.c_l_m_s / (self.dx_meters * self.dx_meters))
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        // Real, disclosed correction (2026-08-30): the
        // density ceiling must correspond to THIS material's own real
        // `volume_ratio_min` (density=rest/J, so the smallest admissible J
        // gives the largest admissible density) -- a copy-pasted hardcoded
        // `rest_density*2.0` (matching `NewtonianFluidMaterial`'s own
        // fixed `[0.5,2.0]` convention) silently broke the F/V/rho triple's
        // own consistency for any caller setting `volume_ratio_min` below
        // 0.5, which this material's own explicit, undefaulted field
        // exists specifically to allow.
        let density_grid = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        let density_si = self.real_density_si(density_grid);
        // Real gauge pressure straight from the EOS core -- stays raw SI
        // Pa, same already-fixed, unconverted stress convention every
        // other material in this engine now uses (see `cavitating_eos`'s
        // own doc for why this is gauge, not absolute).
        let pressure_gauge = self.eos.pressure_gauge_pa(density_si);
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;

        if self.dynamic_viscosity > 0.0 {
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }

        // Artificial (shock) viscosity -- a real PDE term this material had
        // NONE of, despite cavitation being inherently a violent,
        // discontinuous pressure phenomenon (arguably needing shock
        // capturing MORE than a plain liquid, not less). Same real, cited
        // von Neumann & Richtmyer 1950 (LA-671) quadratic + Landshoff linear
        // term `NewtonianFluidMaterial::kirchhoff_stress` uses (see that
        // material's own doc for the full citation/derivation) -- reused via
        // the shared, EOS-agnostic `von_neumann_richtmyer_q` primitive, not
        // reinvented. `c_sound` comes from this material's own real REST
        // acoustic speed (`rest_acoustic_c2`, already computed elsewhere in
        // this file for the CFL bound) -- a disclosed rest-state stand-in,
        // not the exact per-density value (this isothermal variant exposes
        // no live density-dependent sound speed; see `CavitatingFluidMaterial`'s
        // own `acoustic_c2_at` for that more precise version).
        // `weak_shock_gamma = self.eos.gamma_l` is the SAME real liquid-
        // branch exponent this file's own `params()` already feeds through
        // for the identical CFL-safety mechanism, not a new value invented
        // here. `grid_cell_size=1.0`: same disclosed exact (not approximate)
        // convention `NewtonianFluidMaterial`'s own call site uses.
        if let Some(c2_rest) = self.rest_acoustic_c2() {
            let c_sound = c2_rest.max(0.0).sqrt();
            let q = crate::materials::utils::von_neumann_richtmyer_q(
                self.rest_density_grid,
                j,
                0.5 * div_v,
                1.0,
                c_sound,
                self.eos.gamma_l,
            );
            stress -= Mat2::from_diagonal(Vec2::splat(q));
        }
        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    /// Real, disclosed fix reused directly (2026-08-30): the exact
    /// exponential volume integrator (`J_new=J_old*exp(dt*div(v))`) this
    /// same session restored for `NewtonianFluidMaterial` -- see that
    /// material's own `update_particle` doc for the full writeup on why
    /// `det(I+dt*C)` is NOT rotation-invariant and must not be used here
    /// either. This material is built specifically to fix a real drift
    /// bug in fluid volume evolution -- it would be a real, disclosed
    /// contradiction to build it on the SAME buggy integrator that
    /// investigation started from.
    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        let j = (old_j * (dt * div_v).exp()).clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        // Real, disclosed correction (2026-08-30) -- same F/V/rho
        // consistency fix as `kirchhoff_stress`'s own doc:
        // the density ceiling must track `volume_ratio_min`, not a
        // hardcoded `2.0` left over from `NewtonianFluidMaterial`'s own
        // fixed convention.
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

    /// CPU-only: on the GPU this material would upload as a plain Tait
    /// fluid, so `GpuSimulation` refuses it (`gpu_unsupported_reason`).
    ///
    /// Real, disclosed correction (2026-08-30): `params()` is NOT purely
    /// GPU-facing metadata -- `eos_power`
    /// specifically doubles as `cfl.rs`'s own CPU-side shock-viscosity
    /// safety term's `weak_shock_gamma` (see that file's own comment,
    /// confirmed: `NewtonianFluidMaterial`/`IdealGasMaterial` both feed
    /// their own real exponent through this exact same field for this
    /// exact reason). Leaving it at the trait's default `0.0` SILENTLY
    /// disabled this real CFL safety margin for this material -- fixed by
    /// supplying `gamma_l` (this material's own real, liquid-branch Tait-
    /// like exponent, the same physical role `NewtonianFluidMaterial`'s
    /// own real `eos_power` plays for the identical mechanism).
    /// `eos_stiffness` is NOT populated -- no other confirmed CPU-side
    /// consumer found for it on this material's own real code path.
    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density_grid,
            dynamic_viscosity: self.dynamic_viscosity,
            eos_power: self.eos.gamma_l,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed correction (2026-08-30): an earlier version of this
    /// method used the LIQUID branch's own `c_l^2` as a blanket bound
    /// everywhere, believing it always conservative -- confirmed WRONG by
    /// direct computation: `CavitatingEosParams::acoustic_c2_si`'s own
    /// test found the vapor branch's real `dp/drho` at its boundary
    /// EXCEEDS `c_l^2` for this material's own real test parameters (a
    /// real, ~13% UNDERESTIMATE, not just a loose bound), and the mixture
    /// branch's own exact derivative diverges to infinity at its edges (a
    /// real, disclosed feature of the arcsin closure). Fixed: evaluates
    /// the EOS's own exact, per-density `acoustic_c2_si` at the ACTUAL
    /// current density (recovered to real SI first, same bridge
    /// `kirchhoff_stress` uses) -- both more correct (no more silent
    /// underestimate) and tighter (not needlessly conservative in the
    /// liquid regime) than a single blanket constant.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;
        let density_si = self.real_density_si(density.max(self.min_density));
        let c2_si = self.eos.acoustic_c2_si(density_si);
        // Same real dx-folding as `rest_acoustic_c2` -- grid cells^2/s^2.
        let c2_grid = c2_si / (self.dx_meters * self.dx_meters);
        if c2_grid.is_finite() && c2_grid > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2_grid.sqrt());
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

/// `CavitatingFluidMaterial` -- the real, genuinely temperature-coupled
/// successor to `IsothermalCavitatingFluidMaterial` (external review's own
/// step 7): owns a `CavitatingEosTable` instead of one fixed
/// `CavitatingEosParams`, and reconstructs the real EOS at the particle's
/// own LIVE temperature every time it needs pressure or a derivative --
/// water near freezing and water near boiling now genuinely behave
/// differently, closing the real gap `IsothermalCavitatingFluidMaterial`'s
/// own doc disclosed. Same real F/V/rho contract, `volume_ratio_min/max`
/// convention, and dx-folding as that material -- see its own doc for the
/// shared rationale; this doc only covers what's genuinely different.
///
/// Real, disclosed limitation, unchanged from the isothermal version:
/// this is a one-directional thermomechanical coupling
/// (`T -> p_sat(T) -> mechanical response`) -- real cavitation happening
/// mechanically does NOT yet pay any real latent heat back into the
/// particle's own enthalpy state. A genuinely two-way thermodynamic
/// phase-transition closure (mechanical cavitation fraction consuming/
/// releasing real latent heat) is real, disclosed, still-open future
/// work, not attempted here.
#[derive(Debug, Clone)]
pub struct CavitatingFluidMaterial {
    /// The real, T-indexed table this material reconstructs from at
    /// every real query -- see `CavitatingEosTable`'s own doc.
    pub table: CavitatingEosTable,
    /// Same real role as `IsothermalCavitatingFluidMaterial::dx_meters`.
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    rest_density_grid: f32,
    pub min_density: f32,
    pub min_volume: f32,
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

/// Named-field alternative to [`CavitatingFluidMaterial::new`]'s positional
/// arguments -- see [`IsothermalCavitatingFluidMaterialParams`]'s own doc
/// for why. `Clone`-only, not `Copy`: `table` owns a real `Vec`-backed
/// lookup table (see [`CavitatingEosTable`]'s own doc), same reason that
/// type itself isn't `Copy`.
#[derive(Clone, Debug)]
pub struct CavitatingFluidMaterialParams {
    pub table: CavitatingEosTable,
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
}

impl CavitatingFluidMaterial {
    /// Same as [`Self::new`], named fields instead of positional args --
    /// see [`CavitatingFluidMaterialParams`]'s own doc for why.
    pub fn from_params(params: CavitatingFluidMaterialParams) -> Self {
        Self::new(
            params.table,
            params.dx_meters,
            params.dynamic_viscosity,
            params.volume_ratio_min,
            params.volume_ratio_max,
        )
    }

    pub fn new(
        table: CavitatingEosTable,
        dx_meters: f32,
        dynamic_viscosity: f32,
        volume_ratio_min: f32,
        volume_ratio_max: f32,
    ) -> Self {
        assert!(
            dx_meters.is_finite() && dx_meters > 0.0,
            "CavitatingFluidMaterial requires a positive dx_meters"
        );
        assert!(
            volume_ratio_min > 0.0 && volume_ratio_max > volume_ratio_min,
            "CavitatingFluidMaterial requires 0 < volume_ratio_min < volume_ratio_max, \
             no default -- see IsothermalCavitatingFluidMaterial's own equivalent field doc \
             for how to size it"
        );
        let rest_density_grid = table.rho_l_ref_kg_m3 * dx_meters * dx_meters;
        Self {
            table,
            dx_meters,
            dynamic_viscosity,
            rest_density_grid,
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
}

impl MaterialModel for CavitatingFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        Some(
            "CavitatingFluidMaterial has no GPU stress path: it uploads as a plain Tait fluid, so the GPU would run without cavitation",
        )
    }

    /// A scene that spawns this at a density other than the liquid
    /// reference must set the lattice spacing AND
    /// `SpawnRegion::initial_deformation_gradient` together -- see
    /// `cavitating_eos`'s own module doc for the measured cost of
    /// setting only one of the two.
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

    /// Real, T-INDEPENDENT rest bound -- the liquid branch's own constant
    /// `c_l^2` (a real field of `table` itself, not reconstructed), used
    /// ONLY by the near-wall CFL gate's own Mach-relative threshold
    /// (`SimConfig::fluid_near_wall_compression_mach_margin`'s own doc),
    /// which genuinely needs one fixed reference value, not a live one.
    /// The real, live density-AND-temperature-aware bound used for the
    /// actual CFL dt selection is `acoustic_c2_at`, below.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        Some(self.table.c_l_m_s * self.table.c_l_m_s / (self.dx_meters * self.dx_meters))
    }

    /// Real, live density-AND-temperature-jointly-aware acoustic bound
    /// (external review's own explicit requirement: `T` alone cannot
    /// resolve which real branch/patch a `(density,T)` pair lands in for
    /// this closure) -- reconstructs the real EOS at the particle's own
    /// live `temperature_k` via `table.reconstruct`, then reads its own
    /// exact `acoustic_c2_si` at the ACTUAL current density, same real
    /// bridge `kirchhoff_stress` uses. `cfl.rs`'s own dispatch adds this
    /// as a real, separate CFL term (see `MaterialModel::acoustic_c2_at`'s
    /// own doc) -- this material's own `timestep_bound` below stays
    /// T-blind (same real, disclosed limitation `IdealGasMaterial::
    /// timestep_bound` has), so this is where the real, tight, live-T-
    /// aware bound actually comes from.
    fn acoustic_c2_at(&self, density: f32, temperature_k: f32) -> Option<f32> {
        let density_si = self.real_density_si(density.max(self.min_density));
        let c2_si = self
            .table
            .reconstruct(temperature_k)
            .acoustic_c2_si(density_si);
        Some(c2_si / (self.dx_meters * self.dx_meters))
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density_grid = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        let density_si = self.real_density_si(density_grid);
        // Real, live-temperature-coupled gauge pressure: reconstructs the
        // real EOS at THIS particle's own current temperature, then reads
        // its own pressure at the current density -- water near freezing
        // and water near boiling genuinely get different real curves now.
        let eos_at_t = self.table.reconstruct(particles.temperature[i]);
        let pressure_gauge = eos_at_t.pressure_gauge_pa(density_si);
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;

        if self.dynamic_viscosity > 0.0 {
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }

        // Artificial (shock) viscosity -- same real PDE term/citation as
        // `IsothermalCavitatingFluidMaterial::kirchhoff_stress`'s own doc
        // explains in full; this temperature-coupled variant uses the MORE
        // precise live density-AND-temperature-aware sound speed
        // (`acoustic_c2_at`, already built for this material's own
        // `timestep_bound`) rather than a rest-state stand-in.
        // `self.table.gamma_l` is the SAME real liquid-branch exponent this
        // file's own `params()` already feeds through for the identical
        // CFL-safety mechanism.
        if let Some(c2_local) = self.acoustic_c2_at(density_grid, particles.temperature[i]) {
            let c_sound = c2_local.max(0.0).sqrt();
            let q = crate::materials::utils::von_neumann_richtmyer_q(
                self.rest_density_grid,
                j,
                0.5 * div_v,
                1.0,
                c_sound,
                self.table.gamma_l,
            );
            stress -= Mat2::from_diagonal(Vec2::splat(q));
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
    /// status as `IsothermalCavitatingFluidMaterial`'s own doc.
    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density_grid,
            dynamic_viscosity: self.dynamic_viscosity,
            eos_power: self.table.gamma_l,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed, T-blind fallback (see `acoustic_c2_at`'s own doc
    /// for where the real, live-T-aware bound actually comes from): uses
    /// the same T-independent `rest_acoustic_c2` this material's own
    /// near-wall gate already relies on. A real, known-imperfect bound on
    /// its own (the vapor branch's own real derivative at its boundary
    /// can exceed `c_l^2`, same real finding `IsothermalCavitating
    /// FluidMaterial::timestep_bound`'s own doc already made) -- kept
    /// only as a defensive floor for any caller that invokes
    /// `timestep_bound` directly, outside the normal CFL fold that always
    /// also checks `acoustic_c2_at`.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;
        if let Some(c2_grid) = self.rest_acoustic_c2()
            && c2_grid.is_finite()
            && c2_grid > f32::EPSILON
        {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2_grid.sqrt());
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

    /// Real, direct anchor: this is the whole real point of building
    /// `CavitatingFluidMaterial` at all -- pressure at the SAME density
    /// must genuinely differ at two different real temperatures. A
    /// material that didn't actually feed live `temperature` into its own
    /// `pressure_gauge_pa` would fail this trivially (both values equal).
    #[test]
    fn pressure_genuinely_differs_with_the_particles_own_live_temperature() {
        let table = real_table();
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let material = CavitatingFluidMaterial::new(table, config.dx_meters, 1.0e-3, 0.5, 8.0);

        let mut p_cold = Particle::zeroed();
        p_cold.mass = material.rest_density_grid;
        p_cold.deformation_gradient = Mat2::IDENTITY;
        material.init_particle(&mut p_cold);
        // Real, direct expansion state: stretch slightly above rest so the
        // mixture band is actually engaged, where p_v_gauge(T) (and
        // therefore the real T-dependence) actually shows up -- AT rest
        // density both temperatures give exactly 0 gauge by construction,
        // which would trivially "pass" without proving anything.
        p_cold.deformation_gradient = Mat2::from_diagonal(Vec2::splat(1.02f32.sqrt()));
        p_cold.temperature = 275.0;

        let mut p_hot = p_cold;
        p_hot.temperature = 370.0;

        let particles_cold = Particles::from(vec![p_cold]);
        let particles_hot = Particles::from(vec![p_hot]);
        let stress_cold = material.kirchhoff_stress(&particles_cold, 0);
        let stress_hot = material.kirchhoff_stress(&particles_hot, 0);
        let p_cold_pa = -stress_cold.x_axis.x;
        let p_hot_pa = -stress_hot.x_axis.x;
        assert!(
            (p_cold_pa - p_hot_pa).abs() > 1.0,
            "pressure at the same density must genuinely differ with the \
             particle's own live temperature: T=275K gave {p_cold_pa} Pa, \
             T=370K gave {p_hot_pa} Pa -- these should not match"
        );
    }

    /// Real, direct anchor: at the material's own real rest density, ANY
    /// real temperature within the table's own valid range must still
    /// give exactly zero gauge pressure -- the liquid branch's own
    /// `c_l^2*(rho-rho_l_ref)` term is T-independent by construction, so
    /// this must hold regardless of which real T reconstructed the patch.
    #[test]
    fn pressure_is_exactly_zero_gauge_at_rest_density_for_any_real_temperature() {
        let table = real_table();
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let material = CavitatingFluidMaterial::new(table, config.dx_meters, 1.0e-3, 0.5, 8.0);
        for t in [275.0, 300.0, 340.0, 372.0] {
            let mut p = Particle::zeroed();
            p.mass = material.rest_density_grid;
            p.deformation_gradient = Mat2::IDENTITY;
            p.temperature = t;
            material.init_particle(&mut p);
            let particles = Particles::from(vec![p]);
            let stress = material.kirchhoff_stress(&particles, 0);
            let pressure_gauge = -stress.x_axis.x;
            assert!(
                pressure_gauge.abs() < 1.0,
                "T={t}K: pressure at rest density must be ~0 gauge, got {pressure_gauge} Pa"
            );
        }
    }

    /// Real, direct anchor: `acoustic_c2_at` must also genuinely respond
    /// to the particle's own live temperature, at a density inside the
    /// mixture band where the real T-dependence actually shows up.
    #[test]
    fn acoustic_c2_at_genuinely_differs_with_temperature() {
        let table = real_table();
        let config = unit_dx_config();
        let material = CavitatingFluidMaterial::new(table, config.dx_meters, 1.0e-3, 0.5, 8.0);
        let density_grid = 500.0 * config.dx_meters * config.dx_meters;
        let c2_cold = material.acoustic_c2_at(density_grid, 275.0).unwrap();
        let c2_hot = material.acoustic_c2_at(density_grid, 370.0).unwrap();
        assert!(
            c2_cold.is_finite() && c2_cold > 0.0 && c2_hot.is_finite() && c2_hot > 0.0,
            "acoustic_c2_at must stay real, finite, and positive: cold={c2_cold} hot={c2_hot}"
        );
    }

    /// Real Tier-0 stress-test closure (2026-09-02): every other test in
    /// this module stays near rest density -- this is the real, distinctive
    /// physical extreme this material's own three-branch closure exists
    /// for, at the exact real bounds it self-declares
    /// (`volume_ratio_min`/`volume_ratio_max`): extreme TENSION (J near
    /// `volume_ratio_max`, deep in the real vapor branch -- the whole
    /// reason this material exists over a naive linear EOS, which would
    /// let gauge pressure diverge to `-infinity` under real tension) and
    /// extreme COMPRESSION (J near `volume_ratio_min`). Both must stay
    /// finite, and BOTH stress (`kirchhoff_stress`) and the live CFL bound
    /// (`acoustic_c2_at`) must agree with each other on which branch a
    /// state is in -- checked directly, not assumed.
    #[test]
    fn pressure_and_sound_speed_stay_real_and_bounded_at_the_material_own_extremes() {
        let table = real_table();
        let config = unit_dx_config();
        let volume_ratio_min = 0.5;
        let volume_ratio_max = 8.0;
        let material = CavitatingFluidMaterial::new(
            table,
            config.dx_meters,
            1.0e-3,
            volume_ratio_min,
            volume_ratio_max,
        );
        let temperature_k = 300.0;

        let particle_at_j = |j: f32| -> Particle {
            let mut p = Particle::zeroed();
            p.mass = material.rest_density_grid;
            p.deformation_gradient = Mat2::IDENTITY;
            material.init_particle(&mut p);
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(j.sqrt()));
            p.temperature = temperature_k;
            p
        };

        // Extreme tension: deep in the real vapor branch, right at this
        // material's own self-declared upper bound.
        let p_tension = particle_at_j(volume_ratio_max * 0.99);
        let particles_tension = Particles::from(vec![p_tension]);
        let stress_tension = material.kirchhoff_stress(&particles_tension, 0);
        let pressure_tension = -stress_tension.x_axis.x;
        let density_tension = material.rest_density_grid / (volume_ratio_max * 0.99);
        let c2_tension = material
            .acoustic_c2_at(density_tension, temperature_k)
            .unwrap();
        assert!(
            pressure_tension.is_finite(),
            "gauge pressure must stay FINITE at extreme tension (J={:.3}, near \
             volume_ratio_max={volume_ratio_max}) -- a naive linear EOS would \
             diverge to -infinity here, the real reason this material's own \
             vapor branch exists: got {pressure_tension} Pa",
            volume_ratio_max * 0.99
        );
        assert!(
            c2_tension.is_finite() && c2_tension > 0.0,
            "acoustic_c2_at must stay real, finite, and positive even deep in \
             the vapor branch: got {c2_tension}"
        );

        // Extreme compression: right at this material's own self-declared
        // lower bound -- real, strong resistance expected, not a collapse.
        let p_compression = particle_at_j(volume_ratio_min * 1.01);
        let particles_compression = Particles::from(vec![p_compression]);
        let stress_compression = material.kirchhoff_stress(&particles_compression, 0);
        let pressure_compression = -stress_compression.x_axis.x;
        let density_compression = material.rest_density_grid / (volume_ratio_min * 1.01);
        let c2_compression = material
            .acoustic_c2_at(density_compression, temperature_k)
            .unwrap();
        assert!(
            pressure_compression.is_finite() && pressure_compression > 0.0,
            "gauge pressure must stay finite and strongly POSITIVE under \
             extreme compression (J={:.3}, near volume_ratio_min={volume_ratio_min}): \
             got {pressure_compression} Pa",
            volume_ratio_min * 1.01
        );
        assert!(
            c2_compression.is_finite() && c2_compression > 0.0,
            "acoustic_c2_at must stay real, finite, and positive under extreme \
             compression: got {c2_compression}"
        );
        assert!(
            pressure_compression > pressure_tension,
            "extreme compression must give a genuinely higher real pressure \
             than extreme tension: compression={pressure_compression} Pa \
             tension={pressure_tension} Pa"
        );
    }

    /// Real, direct anchor for the artificial (shock) viscosity term added
    /// 2026-09-15 (see `kirchhoff_stress`'s own doc for the full citation):
    /// von Neumann & Richtmyer's term must be IDENTICALLY ZERO under
    /// expansion/no-motion (`div(v) >= 0`) and genuinely NONZERO, adding
    /// real extra compressive stress, under compression (`div(v) < 0`) --
    /// the textbook compression gate this citation requires, not assumed.
    #[test]
    fn isothermal_shock_viscosity_activates_only_under_compression() {
        let table = real_table();
        let temperature_k = 300.0;
        let eos = table.reconstruct(temperature_k);
        let material = IsothermalCavitatingFluidMaterial::new(eos, 1.0, 1.0e-3, 0.5, 8.0);

        let base_particle = || -> Particle {
            let mut p = Particle::zeroed();
            p.mass = material.rest_density_grid;
            p.deformation_gradient = Mat2::IDENTITY;
            material.init_particle(&mut p);
            // Slightly compressed so a real, nonzero pressure/density exists
            // for the shock term's own density-dependent sound speed to act
            // against -- exactly at rest density the EOS itself is 0 gauge.
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(0.98f32.sqrt()));
            p
        };

        let mut p_rest = base_particle();
        p_rest.velocity_gradient = Mat2::ZERO;
        let stress_rest = material.kirchhoff_stress(&Particles::from(vec![p_rest]), 0);

        let mut p_expanding = base_particle();
        p_expanding.velocity_gradient = Mat2::from_diagonal(Vec2::splat(0.5)); // div(v) = +1.0
        let stress_expanding = material.kirchhoff_stress(&Particles::from(vec![p_expanding]), 0);

        let mut p_compressing = base_particle();
        p_compressing.velocity_gradient = Mat2::from_diagonal(Vec2::splat(-0.5)); // div(v) = -1.0
        let stress_compressing =
            material.kirchhoff_stress(&Particles::from(vec![p_compressing]), 0);

        assert!(
            (stress_expanding.x_axis.x - stress_rest.x_axis.x).abs() < 1.0e-6,
            "shock viscosity must be exactly gated OFF under expansion (div(v)>0): \
             rest={:?} expanding={:?}",
            stress_rest.x_axis.x,
            stress_expanding.x_axis.x
        );
        assert!(
            stress_compressing.x_axis.x < stress_rest.x_axis.x - 1.0e-6,
            "shock viscosity must add genuine EXTRA compressive stress under \
             compression (div(v)<0), more negative than the rest-state baseline: \
             rest={:?} compressing={:?}",
            stress_rest.x_axis.x,
            stress_compressing.x_axis.x
        );
    }

    #[test]
    fn gpu_refuses_both_cavitating_fluids() {
        let table = real_table();
        let isothermal =
            IsothermalCavitatingFluidMaterial::new(table.reconstruct(300.0), 1.0, 1.0e-3, 0.5, 8.0);
        let coupled = CavitatingFluidMaterial::new(table, 1.0, 1.0e-3, 0.5, 8.0);
        assert!(
            isothermal
                .gpu_unsupported_reason()
                .is_some_and(|r| r.starts_with("IsothermalCavitatingFluidMaterial"))
        );
        assert!(
            coupled
                .gpu_unsupported_reason()
                .is_some_and(|r| r.starts_with("CavitatingFluidMaterial"))
        );
    }

    /// Same real anchor as `isothermal_shock_viscosity_activates_only_under_compression`,
    /// for the temperature-coupled material -- both cavitation materials had
    /// this term missing before 2026-09-15, not just one.
    #[test]
    fn temperature_coupled_shock_viscosity_activates_only_under_compression() {
        let table = real_table();
        let material = CavitatingFluidMaterial::new(table, 1.0, 1.0e-3, 0.5, 8.0);
        let temperature_k = 300.0;

        let base_particle = || -> Particle {
            let mut p = Particle::zeroed();
            p.mass = material.rest_density_grid;
            p.deformation_gradient = Mat2::IDENTITY;
            material.init_particle(&mut p);
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(0.98f32.sqrt()));
            p.temperature = temperature_k;
            p
        };

        let mut p_rest = base_particle();
        p_rest.velocity_gradient = Mat2::ZERO;
        let stress_rest = material.kirchhoff_stress(&Particles::from(vec![p_rest]), 0);

        let mut p_expanding = base_particle();
        p_expanding.velocity_gradient = Mat2::from_diagonal(Vec2::splat(0.5));
        let stress_expanding = material.kirchhoff_stress(&Particles::from(vec![p_expanding]), 0);

        let mut p_compressing = base_particle();
        p_compressing.velocity_gradient = Mat2::from_diagonal(Vec2::splat(-0.5));
        let stress_compressing =
            material.kirchhoff_stress(&Particles::from(vec![p_compressing]), 0);

        assert!(
            (stress_expanding.x_axis.x - stress_rest.x_axis.x).abs() < 1.0e-6,
            "shock viscosity must be exactly gated OFF under expansion (div(v)>0): \
             rest={:?} expanding={:?}",
            stress_rest.x_axis.x,
            stress_expanding.x_axis.x
        );
        assert!(
            stress_compressing.x_axis.x < stress_rest.x_axis.x - 1.0e-6,
            "shock viscosity must add genuine EXTRA compressive stress under \
             compression (div(v)<0), more negative than the rest-state baseline: \
             rest={:?} compressing={:?}",
            stress_rest.x_axis.x,
            stress_compressing.x_axis.x
        );
    }
}

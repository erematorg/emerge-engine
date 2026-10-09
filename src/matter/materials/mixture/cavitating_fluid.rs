//! Two `MaterialModel`s on `cavitating_eos`'s cavitation EOS (see that
//! module, and its "C^1 junctions" section for the temperature-dependent
//! closure):
//!
//! - `IsothermalCavitatingFluidMaterial` fixes `p_v_gauge_pa` at
//!   construction from one temperature and ignores a particle's live
//!   `temperature`; cheaper, for scenes that do not need temperature-coupled
//!   cavitation.
//! - `CavitatingFluidMaterial` owns a `cavitating_eos::CavitatingEosTable`
//!   and reconstructs the EOS at the particle's live temperature on every
//!   query, so water near freezing and near boiling behave differently.
//!
//! Both couple one way (`T -> p_sat(T) -> mechanical response`):
//! cavitation does not pay latent heat back into the particle's enthalpy.
//! A two-way mixture-equilibrium closure (Saurel, Boivin & Le Métayer 2016)
//! is future work.

use glam::{Mat2, Vec2};

use super::cavitating_eos::{CavitatingEosParams, CavitatingEosTable};
use crate::materials::utils::advance_log_volume_ratio;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

#[derive(Debug, Clone, Copy)]
pub struct IsothermalCavitatingFluidMaterial {
    /// The gauge-pressure, three-branch EOS core -- always real SI
    /// internally (kg/m^3, Pa, m/s), see `CavitatingEosParams`'s doc.
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
    /// Caller-supplied numerical bound on `J` (`volume/initial_volume`), the
    /// convention of `IdealGasMaterial::volume_ratio_min/max` and
    /// `NewtonianFluidMaterial`'s inline `[0.5, 2.0]`. No default: size it from
    /// the `rho_l_ref/rho_v_ref` ratio the `eos` was built with (full
    /// vaporization is `J ~= rho_l_ref/rho_v_ref`) plus headroom for vapour
    /// expansion at low pressure.
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
    /// see [`IsothermalCavitatingFluidMaterialParams`]'s doc for why.
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

    /// SI density (kg/m^3) recovered from this particle's grid-unit
    /// `density`/`volume` state (`rho_grid = rho_si*dx^2`).
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
    /// see that method's doc for why a strict fluid must set
    /// `initial_volume`/`volume`/`density` exactly here (V0=m/rho0,
    /// rho=rho0), not rely on `SpawnRegion`'s own kernel-density estimate.
    /// A scene that spawns this at a density other than the liquid
    /// reference must set the lattice spacing AND
    /// `SpawnRegion::initial_deformation_gradient` together -- see
    /// `cavitating_eos`'s module doc for the measured cost of
    /// setting only one of the two.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density_grid;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density_grid / j;
    }

    /// Same contract as `NewtonianFluidMaterial::init_particle_from_transition`
    /// (a rebaseline that avoids pressure spikes at a phase-transition
    /// front), clamped to this material's own `volume_ratio_min/max`, not
    /// `[0.5, 2.0]`: steam (`J ~ 6` at full vaporization) condensing into this
    /// material must keep its volume.
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
        // Grid-consistent rest acoustic speed squared, the dx folding of
        // `NewtonianFluidMaterial::rest_acoustic_c2` (`B*gamma/rho_GRID` =
        // `c_real^2/dx^2`, cells^2/s^2), see `timestep_bound`.
        Some(self.eos.c_l_m_s * self.eos.c_l_m_s / (self.dx_meters * self.dx_meters))
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        // The density ceiling follows `volume_ratio_min` (density = rest/J,
        // so the smallest admissible J gives the largest density), not a fixed
        // `rest_density*2.0`, which breaks the F/V/rho consistency for any
        // `volume_ratio_min` below 0.5.
        let density_grid = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        let density_si = self.real_density_si(density_grid);
        // Gauge pressure straight from the EOS, raw SI Pa like every other
        // material's stress (see `cavitating_eos` for why gauge).
        let pressure_gauge = self.eos.pressure_gauge_pa(density_si);
        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure_gauge));

        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;

        if self.dynamic_viscosity > 0.0 {
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }

        // Artificial (shock) viscosity -- a PDE term this material had
        // NONE of, despite cavitation being inherently a violent,
        // discontinuous pressure phenomenon (arguably needing shock
        // capturing MORE than a plain liquid, not less). Same cited
        // von Neumann & Richtmyer 1950 (LA-671) quadratic + Landshoff linear
        // term `NewtonianFluidMaterial::kirchhoff_stress` uses (see that
        // material's doc for the full citation/derivation) -- reused via
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

    /// Exact exponential volume integrator, `J_new = J_old*exp(dt*div(v))`,
    /// as `NewtonianFluidMaterial::update_particle`: `det(I+dt*C)` is not
    /// rotation-invariant and drifts the volume.
    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        // The carried logarithm is the state; reading J back from F
        // and multiplying loses a fraction of every small increment (see
        // `advance_log_volume_ratio`'s doc for the measurement).
        let carried = if *ctx.log_volume_strain != 0.0 || old_j == 1.0 {
            *ctx.log_volume_strain
        } else {
            old_j.max(1.0e-9).ln()
        };
        let (log_j, j) = advance_log_volume_ratio(
            carried,
            dt * div_v,
            self.volume_ratio_min,
            self.volume_ratio_max,
        );
        *ctx.log_volume_strain = log_j;
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        // Density ceiling tracks `volume_ratio_min` (see `kirchhoff_stress`).
        let density = (self.rest_density_grid / j)
            .max(self.min_density)
            .min(self.rest_density_grid / self.volume_ratio_min);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(self.min_volume);
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    /// CPU-only: on the GPU this material would upload as a plain Tait
    /// fluid, so `GpuSimulation` refuses it (`gpu_unsupported_reason`).
    ///
    /// `params()` is not only GPU metadata: `eos_power` is also the
    /// `weak_shock_gamma` of `cfl.rs`'s CPU shock-viscosity term (as for
    /// `NewtonianFluidMaterial`/`IdealGasMaterial`), and the trait default
    /// `0.0` disabled that term. It carries `gamma_l`, the liquid branch's
    /// Tait-like exponent. `eos_stiffness` has no CPU consumer here and is
    /// left unset.
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

    /// Evaluates the EOS's exact `acoustic_c2_si` at the current density
    /// (recovered to SI as in `kirchhoff_stress`). The liquid `c_l^2` is not a
    /// safe blanket bound: the vapour branch's `dp/drho` at its boundary
    /// exceeds it (by ~13% for the test parameters), and the mixture branch's
    /// derivative diverges at its edges.
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

/// Temperature-coupled successor to `IsothermalCavitatingFluidMaterial`:
/// owns a `CavitatingEosTable` instead of one fixed `CavitatingEosParams`
/// and reconstructs the EOS at the particle's live temperature whenever it
/// needs pressure or a derivative. Same F/V/rho contract,
/// `volume_ratio_min/max` convention and dx folding as that material (see
/// its doc); this doc only covers what differs.
///
/// Still one-way (`T -> p_sat(T) -> mechanical response`): cavitation does
/// not pay latent heat back into the particle's enthalpy.
#[derive(Debug, Clone)]
pub struct CavitatingFluidMaterial {
    /// The real, T-indexed table this material reconstructs from at
    /// every query -- see `CavitatingEosTable`'s doc.
    pub table: CavitatingEosTable,
    /// Same real role as `IsothermalCavitatingFluidMaterial::dx_meters`.
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    rest_density_grid: f32,
    pub min_density: f32,
    pub min_volume: f32,
    pub volume_ratio_min: f32,
    pub volume_ratio_max: f32,
    /// Measured optical coefficients (absorption and reduced scattering,
    /// `m^-1`) the caller declares for what this material represents, or
    /// `None`. Filled from `matter::materials::optical`'s datasets
    /// (`pure_water`, ...), never chosen by the model: see
    /// `NewtonianFluidMaterial::optics` for why this is not a substance flag.
    pub optics: Option<crate::energy::radiation::OpticalCoefficientsSi>,
}

/// Named-field alternative to [`CavitatingFluidMaterial::new`]'s positional
/// arguments -- see [`IsothermalCavitatingFluidMaterialParams`]'s doc
/// for why. `Clone`-only, not `Copy`: `table` owns a `Vec`-backed
/// lookup table (see [`CavitatingEosTable`]'s doc), same reason that
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
    /// see [`CavitatingFluidMaterialParams`]'s doc for why.
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
            optics: None,
        }
    }

    #[inline]
    fn real_density_si(&self, grid_density: f32) -> f32 {
        grid_density / (self.dx_meters * self.dx_meters)
    }
}

impl MaterialModel for CavitatingFluidMaterial {
    /// Whatever the caller measured, verbatim (see the `optics` field).
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
    }

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
    /// `cavitating_eos`'s module doc for the measured cost of
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

    /// Temperature-independent rest bound: the liquid branch's constant
    /// `c_l^2` (a field of `table`), used only by the near-wall CFL gate's
    /// Mach-relative threshold (`SimConfig::fluid_near_wall_compression_mach_
    /// margin`), which needs one fixed reference. The CFL itself uses the
    /// live `acoustic_c2_at` below.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        Some(self.table.c_l_m_s * self.table.c_l_m_s / (self.dx_meters * self.dx_meters))
    }

    /// Acoustic bound aware of density and temperature together (`T` alone
    /// does not decide which branch or patch a `(density, T)` pair lands in):
    /// reconstructs the EOS at the particle's live `temperature_k` through
    /// `table.reconstruct`, then reads `acoustic_c2_si` at the current
    /// density, as `kirchhoff_stress` does. `cfl.rs` adds it as a separate
    /// CFL term (see `MaterialModel::acoustic_c2_at`); `timestep_bound` below
    /// stays temperature-blind, like `IdealGasMaterial::timestep_bound`.
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
        // Gauge pressure from the EOS reconstructed at this particle's
        // current temperature, read at the current density.
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
        // `IsothermalCavitatingFluidMaterial::kirchhoff_stress`'s doc
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
        // The carried logarithm is the state; reading J back from F
        // and multiplying loses a fraction of every small increment (see
        // `advance_log_volume_ratio`'s doc for the measurement).
        let carried = if *ctx.log_volume_strain != 0.0 || old_j == 1.0 {
            *ctx.log_volume_strain
        } else {
            old_j.max(1.0e-9).ln()
        };
        let (log_j, j) = advance_log_volume_ratio(
            carried,
            dt * div_v,
            self.volume_ratio_min,
            self.volume_ratio_max,
        );
        *ctx.log_volume_strain = log_j;
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

    /// Not wired for the GPU yet, as `IsothermalCavitatingFluidMaterial`.
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

    /// Temperature-blind fallback (the live bound is `acoustic_c2_at`): the
    /// `rest_acoustic_c2` the near-wall gate uses. Not a safe bound on its own
    /// (the vapour branch's slope can exceed `c_l^2`, see
    /// `IsothermalCavitatingFluidMaterial::timestep_bound`); it only serves a
    /// caller calling `timestep_bound` directly, outside the CFL fold that
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

    /// Pressure at one density differs between two temperatures; a material
    /// not feeding the live `temperature` into `pressure_gauge_pa` would give
    /// equal values.
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
        // Stretched slightly above rest so the mixture band is engaged, where
        // p_v_gauge(T) shows; at rest density both temperatures give 0 gauge
        // by construction and would prove nothing.
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

    /// At rest density every temperature in the table's range gives exactly
    /// zero gauge pressure: the liquid branch `c_l^2*(rho-rho_l_ref)` does
    /// not depend on T.
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

    /// `acoustic_c2_at` responds to the particle's live temperature at a
    /// density inside the mixture band.
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

    /// The material's extremes at its own declared bounds: strong tension (J
    /// near `volume_ratio_max`, deep in the vapour branch, where a linear EOS
    /// would send gauge pressure to minus infinity) and strong compression (J
    /// near `volume_ratio_min`). Both stay finite, and `kirchhoff_stress` and
    /// the live CFL bound `acoustic_c2_at` agree on which branch a state is
    /// in.
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

        // Extreme tension: deep in the vapor branch, right at this
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
        // lower bound -- strong resistance expected, not a collapse.
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

    /// The von Neumann-Richtmyer shock viscosity (see `kirchhoff_stress`) is
    /// exactly zero under expansion or no motion (`div(v) >= 0`) and adds
    /// compressive stress under compression (`div(v) < 0`).
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
            // Slightly compressed so a nonzero pressure/density exists
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

    /// Same check as `isothermal_shock_viscosity_activates_only_under_compression`,
    /// for the temperature-coupled material.
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

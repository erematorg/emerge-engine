//! Mixture material for a particle mid-boil, driven by the enthalpy
//! quality: Voller & Cross 1981's latent-heat plateau, tracked by
//! `energy::thermodynamics::enthalpy::chained_state_from_enthalpy` as
//! `PhaseState::Boiling { fraction }`.
//!
//! # Why
//! With `CavitatingFluidMaterial` alone a particle's mechanical vapour
//! fraction (implicit in its density/`J`) runs independently of its thermal
//! one (the enthalpy `boiling_fraction`): in `phase_states_gui.rs` a
//! particle sat at `J = 5.988` (`x_rho ~ 0.998`) while `x_H < 0.5`. With `T`
//! pinned at the boiling plateau, next to `t_liquid_closure_max`'s
//! near-zero-stiffness boundary (see `cavitating_eos`), a pure `T -> p_sat(T)`
//! closure is never told how much of the particle has boiled.
//!
//! # Model
//! The rest state is the Homogeneous Equilibrium Model mixture density at
//! the particle's current mass quality `x` (stored in
//! `Particle::friction_hardening`, the per-material scratch field):
//! ```text
//! rho_eq(x) = 1 / ((1-x)/rho_l_ref + x/rho_v_ref)
//! ```
//! the mass-weighted specific volume (Collier & Thome, "Convective Boiling
//! and Condensation", 3rd ed., section 2.2). The stiffness around it is
//! Wood's/Wallis's mixture sound speed (Wallis 1969, "One-Dimensional
//! Two-Phase Flow", eq. 4.36):
//! ```text
//! 1 / (rho_eq(x) * c_mix(x)^2) = (1-x)/(rho_l_ref*c_l^2) + x/(rho_v_ref*c_v_ref^2)
//! ```
//! Both reduce exactly to the liquid closure at `x = 0` and the vapour
//! reference at `x = 1`, so they join the neighbouring pure-phase materials.
//! `c_v_ref^2 = B*gamma_v/rho_v_ref` with `B = c_l^2*rho_l_ref/gamma_l`, the
//! shared-stiffness relation of `CavitatingEosParams::new`.
//!
//! The scene's enthalpy update writes `x` every substep from the same
//! `PhaseState::Boiling { fraction }` that sets `material_id`, so the
//! mechanical and thermal vapour fractions are one number.
//!
//! Wood's speed can dip below both pure-phase values (the Wood minimum);
//! `rest_acoustic_c2` is the minimum measured over a dense sweep of `x`, not
//! the smaller endpoint. (The demo's ~6:1 water/vapour density ratio, not
//! steam's ~1600:1, does not reach that regime.)
//!
//! # Limits
//! One-way (`x_H -> mechanical state`), like `CavitatingFluidMaterial`: a
//! deviation from `rho_eq(x_H)` (compression under gravity) pays no latent
//! heat back. A relaxation closure (Saurel, Boivin & Le Métayer 2016) needs
//! an interfacial area and nucleation density this engine does not resolve,
//! so its `tau` would be a tuning constant; the alternative, an implicit
//! flash solving `p`/`T`/`x` from conserved `v`/`h` with `g_l(p,T) =
//! g_v(p,T)`, also needs `T_sat(p)`. Deferred as one milestone.
//!
//! Checked by `boiling_mixture_confined_column_matches_analytical_profile_
//! at_earth_gravity`: a column filling the width between both
//! `SlipBoundary` walls, pre-set to its analytical hydrostatic profile
//! `rho(depth) = rho_eq(x)*exp(g*depth/c_mix2(x))` (the exact solution of
//! `dp/dy = -rho*g` for this linear-in-density EOS at fixed `x`), at Earth
//! gravity: `e_rho = 0.0036` (asserted < 1%). `e_p = 0.834` stays large by
//! structure: at `x = 0.5` the stiffness `rho*c_mix^2 ~ 3.6e7 Pa` dwarfs the
//! hydrostatic scale `rho*g*h ~ 4.5e4 Pa`, amplifying the small `e_rho`.
//!
//! In the live demo, the pressure residual converted to a vapour-quality
//! error through `T_sat(p)` (`water_saturation_temperature_from_pressure_k`,
//! `epsilon_x_equiv = cp_liquid*|T_sat(p)-BOILING_POINT_K|/L_v`, printed as
//! `[boiling-residual-stats]` in `phase_states_gui.rs`) is large only while
//! the column settles; once quasi-static it is depth-coherent but small,
//! median ~0.13-0.15% and p90 ~0.6-0.75% of vapour quality (240 s headless,
//! Earth gravity). Not worth a two-way flash. Reopen if a sustained
//! quasi-static median passes ~1% or p90 several percent in another
//! configuration.

use glam::{Mat2, Vec2};

use super::cavitating_eos::CavitatingEosTable;
use crate::materials::utils::advance_log_volume_ratio;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// HEM mixture density (kg/m^3) at mass quality `x in [0,1]`. `x = 0` and
/// `x = 1` return the reference densities directly: the general formula
/// loses an f32 ULP or two there (~2 Pa of gauge pressure at `x = 0`
/// through the `c_l^2 ~ 3e4` slope), and particles sit at those two
/// densities for long stretches on either side of the boiling band.
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

/// Wood/Wallis mixture sound speed squared (m^2/s^2) at mass quality
/// `x in [0,1]`, exact at both endpoints (see the module doc).
fn c_mix2_m2_s2(
    x: f32,
    rho_l_ref_kg_m3: f32,
    c_l2_m2_s2: f32,
    rho_v_ref_kg_m3: f32,
    c_v_ref2_m2_s2: f32,
) -> f32 {
    let x = x.clamp(0.0, 1.0);
    // Pure-phase reference stiffness returned directly at the endpoints, as
    // in `rho_eq_kg_m3`.
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
    /// Derived vapour-side reference sound speed: `c_v_ref^2 =
    /// B*gamma_v/rho_v_ref`, the shared-stiffness relation of
    /// `CavitatingEosParams::new`.
    pub c_v_ref_m_s: f32,
    /// The liquid branch's Tait exponent, carried from the source
    /// `CavitatingEosTable` only to feed `params().eos_power`, as in
    /// `IsothermalCavitatingFluidMaterial::params()`/
    /// `CavitatingFluidMaterial::params()`. It is not this closure's own
    /// exponent: `p = c_mix2(x)*(rho - rho_eq(x))` is linear in density at
    /// each `x`. It stands in, for the CFL only, because the `x = 0`
    /// endpoint is derived from a Tait EOS with this exponent (`B =
    /// c_l^2*rho_l_ref/gamma_l`), and `cfl.rs` can only shrink the substep
    /// with it. Without it `eos_power` defaulted to `0.0` and disabled that
    /// CFL term.
    pub gamma_l: f32,
    pub dx_meters: f32,
    pub dynamic_viscosity: f32,
    /// Fixed liquid-reference density in GRID units -- the SAME reference
    /// `CavitatingFluidMaterial` uses, so `J`/volume stay exactly
    /// continuous at the water->boiling handoff (`x=0`). `rho_eq(x)`
    /// above only enters the PRESSURE law, never the F/V/rho bookkeeping
    /// -- see this module's own top doc.
    rest_density_grid: f32,
    /// Measured minimum of `c_mix2` over `x in [0,1]` (grid units), the
    /// quality-blind floor for `rest_acoustic_c2`/`timestep_bound`, as
    /// `CavitatingFluidMaterial::rest_acoustic_c2`.
    min_c_mix2_grid: f32,
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

impl BoilingMixtureMaterial {
    /// Reads the liquid/vapour reference constants from the same
    /// `CavitatingEosTable` `CavitatingFluidMaterial` uses, so continuity at
    /// `x = 0` holds with identical numbers.
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

        // Measured minimum of c_mix2 over the quality range (see the field).
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
            optics: None,
        }
    }

    #[inline]
    fn real_density_si(&self, grid_density: f32) -> f32 {
        grid_density / (self.dx_meters * self.dx_meters)
    }

    /// Per-particle mass quality `x`, stored in `Particle::friction_hardening`
    /// (see the module doc).
    #[inline]
    fn quality(&self, particles: &Particles, i: usize) -> f32 {
        particles.friction_hardening[i].clamp(0.0, 1.0)
    }

    /// HEM mixture density (kg/m^3) at mass quality `x`, the formula
    /// `kirchhoff_stress` uses, exposed so diagnostics and tests compute the
    /// same `rho_eq(x)`.
    pub fn rho_eq_kg_m3(&self, x: f32) -> f32 {
        rho_eq_kg_m3(x, self.rho_l_ref_kg_m3, self.rho_v_ref_kg_m3)
    }

    /// Wood/Wallis mixture sound speed squared (m^2/s^2) at mass quality `x`,
    /// exposed like `rho_eq_kg_m3`.
    pub fn c_mix2_m2_s2(&self, x: f32) -> f32 {
        c_mix2_m2_s2(
            x,
            self.rho_l_ref_kg_m3,
            self.c_l_m_s * self.c_l_m_s,
            self.rho_v_ref_kg_m3,
            self.c_v_ref_m_s * self.c_v_ref_m_s,
        )
    }

    /// Equilibrium `J` (`rho_l_ref/rho_eq(x)`) at mass quality `x`, the
    /// mixture line `J_eq(x) = 1 + (rho_l_ref/rho_v_ref - 1)*x`, computed from
    /// `rho_eq_kg_m3`.
    pub fn j_eq(&self, x: f32) -> f32 {
        self.rho_l_ref_kg_m3 / self.rho_eq_kg_m3(x)
    }

    /// Gauge pressure (Pa) at SI density `density_si` and mass quality `x`,
    /// the linearized law `kirchhoff_stress` evaluates, exposed like
    /// `rho_eq_kg_m3`/`c_mix2_m2_s2`.
    pub fn pressure_gauge_pa(&self, density_si: f32, x: f32) -> f32 {
        self.c_mix2_m2_s2(x) * (density_si - self.rho_eq_kg_m3(x))
    }
}

impl MaterialModel for BoilingMixtureMaterial {
    /// Whatever the caller measured, verbatim (see the `optics` field).
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
    }

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

    fn rest_acoustic_c2(&self) -> Option<f32> {
        Some(self.min_c_mix2_grid)
    }

    /// Acoustic bound aware of the particle's quality: `c_mix2` depends on
    /// `x` (`Particle::friction_hardening`), which neither `acoustic_c2_at`
    /// (density and temperature) nor `timestep_bound` (density and
    /// hardening_scale) sees, so this material overrides the most general
    /// tier.
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

    /// Not wired for the GPU yet, as `CavitatingFluidMaterial`.
    ///
    /// `eos_power` is `self.gamma_l`, as in `CavitatingFluidMaterial::params()`
    /// (see `gamma_l` for what it does and does not claim).
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

    /// Quality-blind fallback (the live bound is `acoustic_c2_at_particle`):
    /// the measured minimum `c_mix2`, for a caller invoking `timestep_bound`
    /// directly, outside the CFL fold that also checks
    /// `acoustic_c2_at_particle`.
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

    /// At `x = 0`, `J = 1` (equilibrium liquid) the gauge pressure is exactly
    /// zero, as in `CavitatingFluidMaterial`'s equivalent test.
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

    /// At `x = 1` the equilibrium `J` is `rho_l_ref/rho_v_ref` (full
    /// vaporization), and the gauge pressure there is exactly zero too.
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

    /// `J_eq(x) = rho_l_ref/rho_eq(x)` traces the mass-fraction mixture line:
    /// for this 6:1 density ratio, `J_eq(x) = 1 + 5x`.
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

    /// `c_mix2` equals the liquid/vapour reference stiffness exactly at x = 0
    /// and x = 1, the continuity with the neighbouring materials.
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

    /// `acoustic_c2_at_particle` changes with a particle's `x`
    /// (`friction_hardening`).
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

    /// `min_c_mix2_grid` is a lower bound: no `x` on a dense sweep reads
    /// below it (up to float slop).
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

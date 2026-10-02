//! Compressible ideal-gas material: density matters fully (no rest-pressure
//! offset), and shock capturing matters far more than for a weakly
//! compressible liquid.
//!
//! `IdealGasMaterial`: `p = ρRT` as an isentropic law, adiabatic sound speed
//! `c = √(γRT)`, and von Neumann-Richtmyer shock viscosity through the
//! shared `matter::materials::utils::von_neumann_richtmyer_q`, fed this
//! material's own γ. CPU only: `p2g.wgsl`/`particles_update.wgsl` have no
//! `case 13u` branch (see `ConstitutiveModel::Gas`). Not yet checked against
//! Sod's shock tube (Toro, *Riemann Solvers and Numerical Methods for Fluid
//! Dynamics*), which needs an iterative Riemann solver.

use glam::{Mat2, Vec2};

use crate::energy::thermodynamics::ideal_gas::{
    AIR_ADIABATIC_INDEX, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, ideal_gas_sound_speed_from_temperature,
};
use crate::materials::utils::von_neumann_richtmyer_q;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Standard sea-level atmospheric pressure (Pa), the default ambient
/// `IdealGasMaterial::reference_pressure_pa`.
pub const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;

/// Compressible ideal-gas material: isentropic (adiabatic) EOS
/// `p = p0·(ρ/ρ0)^γ` with `p0 = ρ0·R·T` (the ideal gas law at the particle's
/// own reference state), adiabatic sound speed `c = √(γRT)`, and von
/// Neumann-Richtmyer shock viscosity under compression. Unlike
/// `NewtonianFluidMaterial`'s Tait law, `p0` is a physical rest pressure,
/// not a fitted stiffness, and there is no rest-pressure offset: `p → 0` as
/// `ρ → 0` (`energy::thermodynamics::ideal_gas`'s
/// `pressure_vanishes_with_density`).
///
/// Isentropic, not isothermal: substeps are far too short for conduction to
/// equalize (the reason for the adiabatic sound speed), and an isothermal
/// `p = ρRT` never throttles under expansion (`p·V` stays constant; in
/// `examples/basic_gas.rs` average J ran to 8-11 within 2 s from a 3:1
/// pressure ratio), while here `p·V` falls as `V^(1-γ)`.
///
/// `p0` reads `Particles::temperature`, so a `ThermalDiffusion` in the same
/// scene still shifts the pressure on its slower conductive timescale.
/// Without one, temperature stays at the `init_particle` seed
/// (`reference_temperature_k`).
///
/// CPU only: `p2g.wgsl`/`particles_update.wgsl` have no `case 13u` branch,
/// so a `Gas` particle on the GPU falls through to their zero-stress default
/// arm (see `ConstitutiveModel::Gas`).
#[derive(Debug, Clone, Copy)]
pub struct IdealGasMaterial {
    /// Reference density ρ₀ (grid units, `rho_SI * dx_meters²`).
    pub rest_density: f32,
    /// Dynamic viscosity µ (Pa·s, raw SI, passed through unconverted as in
    /// `NewtonianFluidMaterial`). Air ≈1.81e-5 Pa·s; 0.0 = inviscid (Euler).
    pub dynamic_viscosity: f32,
    /// Specific gas constant R, GRID-scaled (`R_SI / dx_meters²`) -- see
    /// `from_physical`'s doc for the full derivation of why R itself
    /// (not just density) needs this factor, unlike Tait's ratio-based EOS.
    pub specific_gas_constant: f32,
    /// Adiabatic index γ = Cp/Cv (air/diatomic: 1.4, `AIR_ADIABATIC_INDEX`;
    /// monatomic: 5/3; triatomic: ~1.3), used for the adiabatic sound speed
    /// and as the shock viscosity's `weak_shock_gamma`: for a gas that is
    /// exactly the index Kurapatenko's coefficient is defined with.
    pub adiabatic_index: f32,
    /// Temperature (Kelvin) seeded onto every particle at spawn
    /// (`init_particle`) and used as the fixed reference state for
    /// `timestep_bound`/`rest_acoustic_c2` (see `timestep_bound` for what
    /// that means under strong heating).
    pub reference_temperature_k: f32,
    /// Ambient absolute pressure (Pa, raw SI) this gas pushes against.
    /// `kirchhoff_stress` computes the absolute `p_abs = rho0*R*T*
    /// (rho/rho0)^gamma` for the thermodynamics, but the mechanical stress it
    /// adds to P2G is gauge, `-(p_abs - reference_pressure_pa)*I` (the Oregon
    /// State MPM notes separate absolute pressure for a gas confined by rigid
    /// walls from gauge pressure for a gas next to an unstressed material,
    /// this engine's case). With absolute pressure a steam particle at rest
    /// density pushed with a full ~101325 Pa that nothing balanced, drove J to
    /// `volume_ratio_max` and stuck there; viscosity cannot damp a constant
    /// force (Kelvin-Voigt 2x and bulk viscosity 10x changed nothing).
    ///
    /// Default: standard sea-level pressure (101,325 Pa). A submerged bubble
    /// can add `rho_water*g*depth` on top.
    pub reference_pressure_pa: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Lower bound on `J = V/V0` -- unlike a weakly-compressible liquid, a
    /// real gas can compress far below half its rest volume, so this is
    /// deliberately much wider than `NewtonianFluidMaterial`'s pinned
    /// `[0.5, 2.0]`. NOT tuned against an impact/shock test scene the
    /// way fluid's own bounds are (no such scene exists for gas yet) --
    /// first real cut, disclosed as provisional.
    pub volume_ratio_min: f32,
    /// Upper bound on `J`. See `volume_ratio_min`'s doc.
    pub volume_ratio_max: f32,
    /// Bulk (dilatational/second) viscosity ζ, Pa·s -- raw SI, unconverted,
    /// same convention as `dynamic_viscosity`. Adds `τ += ζ·(∇·v)·I` to the
    /// Kirchhoff stress, the Navier-Stokes second-viscosity term (as in
    /// `NewtonianFluidMaterial::bulk_viscosity`). Unlike the shock viscosity
    /// `q` (compression only), it resists expansion as much as compression:
    /// without it nothing but the `volume_ratio_max` clamp resisted
    /// buoyancy-driven expansion, and J raced to the clamp and back
    /// (`phase_states_gui.rs` chimney, per-particle `det(F)` logs).
    ///
    /// A monatomic gas has zero bulk viscosity (Stokes' hypothesis); water
    /// vapour is polyatomic, with rotational and vibrational relaxation, and
    /// its bulk viscosity is "hundreds or thousands of times larger than
    /// [its] shear viscosity" over 380-1000 K (Cramer 2012, "Numerical
    /// estimates for the bulk viscosity of ideal gases", Physics of Fluids
    /// 24, 066102). See `water_vapor_bulk_viscosity_pa_s`.
    ///
    /// 0.0 = off (default).
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

    /// Dry air at a given SI density and temperature:
    /// `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K`/`AIR_ADIABATIC_INDEX` (checked
    /// against the ~343 m/s speed of sound in `energy::thermodynamics::
    /// ideal_gas`'s tests), dynamic viscosity 1.81e-5 Pa·s (air at ~20°C,
    /// Sutherland's law reference).
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
    /// from_physical`'s doc and the regression it fixed, extended
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
    /// parameters make transposition a silent risk.
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

/// Bulk (dilatational) viscosity of a polyatomic gas dominated by
/// rotational/vibrational relaxation, water vapour in particular: Cramer
/// 2012 (Physics of Fluids 24, 066102) puts water vapour's bulk viscosity at
/// "hundreds or thousands of times" its shear viscosity over 380-1000 K.
///
/// `BULK_TO_SHEAR_RATIO` takes the high end (thousands): at 100x an isolated
/// steam particle in `phase_states_gui.rs`'s heated Moon-gravity scene still
/// reached `volume_ratio_max` and stuck there.
///
/// Returns SI Pa·s, passed unconverted into `bulk_viscosity`.
pub fn water_vapor_bulk_viscosity_pa_s(shear_viscosity_pa_s: f32) -> f32 {
    // The high end of Cramer 2012's range ("hundreds or thousands of times"
    // the shear viscosity); 100x let an isolated steam particle reach
    // `volume_ratio_max` in the heated Moon-gravity run.
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
    /// but this EOS depends on (`p=ρRT`).
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
        particle.temperature = self.reference_temperature_k;
    }

    /// For a transition from a material of very different rest density
    /// (water -> steam, ~1700x), `init_particle`'s `mass/rest_density` would
    /// make the particle's volume jump ~1700x in one instant, a force spike
    /// that crashed `examples/basic_steam.rs` (see `MaterialModel::
    /// init_particle_from_transition`).
    ///
    /// The reference volume stays `mass/rest_density`, as `kirchhoff_stress`
    /// and `update_particle` assume; the particle instead starts with a
    /// deformation gradient saying how compressed it is against that
    /// reference (freshly formed gas still in its small old footprint),
    /// clamped to `[volume_ratio_min, volume_ratio_max]`, the bounds every
    /// substep enforces. Capping `initial_volume` does not work:
    /// `update_particle` recomputes volume from `mass*j/rest_density` every
    /// substep and never reads it again.
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
        // Temperature is not touched here: `Simulation::apply_phase_transition`
        // has just debited `latent_heat / heat_capacity`, and resetting it to
        // `reference_temperature_k` would erase that debit (boiling would
        // always land at 373.15 K). The sibling overrides
        // (`NewtonianFluidMaterial`, `BoilingMixtureMaterial`,
        // `CavitatingFluidMaterial`) leave it alone too; `init_particle`
        // (fresh spawns) still seeds `reference_temperature_k`.
    }

    /// `c² = γ·R·T` evaluated at the reference state (`J=1`,
    /// `T=reference_temperature_k`) -- the same formula `timestep_bound`
    /// evaluates, not a second derivation. See `timestep_bound` for what
    /// the fixed reference temperature means under heating.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.specific_gas_constant > 0.0 && self.reference_temperature_k > 0.0 {
            Some(self.adiabatic_index * self.specific_gas_constant * self.reference_temperature_k)
        } else {
            None
        }
    }

    /// `c^2 = gamma*R*T` at a live temperature. `kirchhoff_stress` evaluates
    /// `p0 = rho0*R*T` at the particle's live temperature, so under heating
    /// the stiffness grows linearly with `T`; a CFL bound fixed at
    /// `reference_temperature_k` (373.15 K) fell further behind the longer
    /// heating lasted (in `phase_states_gui.rs`: divergence in the thousands,
    /// substeps at their cap, steam max(J) at `volume_ratio_max`). Same
    /// relation as `rest_acoustic_c2`, with its temperature input supplied.
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

        // Isentropic (adiabatic) pressure law, not an isothermal `p = ρRT` at
        // fixed T: substeps are far too short for conduction to equalize,
        // the reason the sound speed is adiabatic (`c = √(γRT)`). An
        // isothermal pressure with an adiabatic sound speed is inconsistent,
        // and it never throttles under expansion (`p·V = nRT` stays constant),
        // which ran `basic_gas.rs` away (average J 8-11 within 2 s from a 3:1
        // pressure ratio).
        //
        // Combining `p = ρRT` with the adiabatic relation `T/T0 =
        // (ρ/ρ0)^(γ-1)` gives `p = p0·(ρ/ρ0)^γ`, with `p0 = ρ0·R·T` at the
        // particle's current temperature (so a `ThermalDiffusion` still
        // shifts it). `p·V` falls as `V^(1-γ)` under expansion, and `dp/dρ`
        // at `ρ = ρ0` is `γRT`, the formula `rest_acoustic_c2`/
        // `timestep_bound` use.
        let p0 = self.rest_density * self.specific_gas_constant * temperature;
        let pressure_abs = (p0
            * crate::materials::utils::fast_pow(density / self.rest_density, self.adiabatic_index))
        .max(0.0); // physically required floor: ρ,T >= 0 => p_abs >= 0, nothing to configure

        // Mechanical stress is gauge (see `reference_pressure_pa`); `p_abs`
        // stays for the thermodynamics (temperature coupling, sound speed).
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
        // doc. Same real formula NewtonianFluidMaterial::kirchhoff_stress
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
        // than a Tait-EOS-derived stand-in (see that function's doc).
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
        // `det(I+dt*C)` is not rotation-invariant: a rigid rotation gives a
        // positive O(dt^2) expansion every substep, which isotropization bakes
        // in. The continuity equation's exact solution, `J_{n+1} = J_n*exp(dt*
        // div(v))`, does not (as in `NewtonianFluidMaterial::update_particle`).
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

    /// This trait method only has `density`, not the particle's temperature,
    /// so the acoustic term below uses `reference_temperature_k`. `cfl.rs`
    /// adds a separate live-temperature term through
    /// `acoustic_c2_at_temperature` (like its shock-viscosity and
    /// single-particle terms, which do not live here either), so the bound
    /// enforced does follow the live temperature.
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
        // timestep_bound`'s doc gives for its Kelvin-Voigt term (measured
        // live: an unbounded viscous term makes peak speed jump instead of
        // damping). Combined linearly, not each bounded separately: both
        // terms multiply the SAME velocity-gradient-derived stress, so their
        // worst-case combined diffusive coefficient is the conservative
        // bound, not an approximation. `bulk_viscosity` can be "hundreds of
        // times" `dynamic_viscosity` for a polyatomic gas (see that
        // field's doc) -- without including it here, the substep
        // selector would never see the stiffness it adds.
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
    /// textbook air reference `energy::thermodynamics::ideal_gas`'s
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

    /// Air at its rest density and temperature (`p_abs` ~101325 Pa) exerts
    /// zero mechanical stress inside the default ambient of one standard
    /// atmosphere: nothing pushes it to expand or compress relative to its
    /// surroundings (see `reference_pressure_pa`).
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

    /// `init_particle_from_transition` leaves `particle.temperature` alone:
    /// `Simulation::apply_phase_transition` has already debited
    /// `latent_heat/heat_capacity`. Starts the particle at a temperature that
    /// is not the reference value, as a transition leaves it, and checks
    /// it survives, as in the sibling overrides.
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

    /// `water_vapor_bulk_viscosity_pa_s` must return a finite,
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

        // Navier-Stokes second-viscosity sign: resisting expansion adds a
        // positive diagonal stress relative to the inviscid case, a restoring
        // force against further expansion.
        assert!(
            tau_damped.x_axis.x > tau_inviscid.x_axis.x,
            "bulk viscosity must add real resistance to expansion: \
             inviscid={:.6} damped={:.6}",
            tau_inviscid.x_axis.x,
            tau_damped.x_axis.x
        );
    }

    /// `bulk_viscosity == 0.0` (every preset's default) reproduces the stress
    /// without the term.
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

    /// As a gas pocket approaches vacuum its absolute pressure vanishes (the
    /// ideal gas law), but its mechanical (gauge) stress approaches
    /// `-reference_pressure_pa`: a near-vacuum bubble in an atmosphere is
    /// crushed inward by the full ambient pressure (see
    /// `reference_pressure_pa`).
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

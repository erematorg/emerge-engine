pub mod bingham;
pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod granular_fluid;
pub mod nacc;
pub mod params;
pub mod physical_props;
pub mod rankine;
pub mod registry;
pub mod sand;
pub mod sand_mui;
pub mod snow;
pub(crate) mod svd;
pub mod utils;
pub mod viscoelastic;
pub mod von_mises;

pub use physical_props::{
    BrittleProps, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI, ParticleMass,
    PlasticityModel, Viscoelastic,
};

pub use bingham::BinghamFluidMaterial;
pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use granular_fluid::GranularFluidMaterial;
pub use nacc::NaccMaterial;
pub use params::MaterialParams;
pub use rankine::RankineMaterial;
pub use registry::{MAX_MATERIAL_SLOTS, MaterialRegistry};
pub use sand::DruckerPragerMaterial;
pub use sand_mui::MuIRheologyMaterial;
pub use snow::StomakhinMaterial;
pub use utils::{
    elastic_wave_dt, gravity_to_grid, lame_from_si, lame_from_young, polar_decomposition_2d,
    rankine_damage_estimate,
};
pub use viscoelastic::ViscoelasticMaterial;
pub use von_mises::VonMisesMaterial;

use glam::Mat2;

use crate::particle::{Particle, Particles};

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles — do not change them.
#[non_exhaustive]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstitutiveModel {
    Fallback = 0,
    Fluid = 1,            // Weakly-compressible Newtonian fluid, Tait EOS
    NeoHookean = 2,       // Neo-Hookean hyperelastic (jelly, soft solids)
    Corotated = 3,        // Corotated linear elastic (stiffer baseline)
    Snow = 4,             // Corotated + SVD plasticity (Stomakhin 2013)
    DruckerPrager = 5,    // Corotated elastic + DP yield surface (sand, soil, rock)
    VonMises = 6,         // J2 perfect plasticity — ductile flow, no hardening (lava, metal, clay)
    Rankine = 7,          // Tensile cutoff + exponential softening — brittle rock, bone, ice
    DruckerPragerMuI = 8, // Rate-dependent DP — µ(I) rheology, granular flow
    Viscoelastic = 9,     // Kelvin-Voigt: NeoHookean elastic + viscous dashpot in parallel
    Nacc = 10,            // Non-Associated Cam-Clay — wet soil, clay, bio tissue under compression
    GranularFluid = 11, // Granular-fluid mixture — Tait EOS + corotated deviatoric + SVD plasticity
}

// WGSL shaders (p2g.wgsl, particles_update.wgsl) index material branches by the
// ConstitutiveModel discriminant cast to u32. These assertions catch any enum reordering
// that would silently run the wrong GPU stress branch on a material.
const _: () = {
    use ConstitutiveModel as C;
    assert!(C::Fallback as u32 == 0);
    assert!(C::Fluid as u32 == 1);
    assert!(C::NeoHookean as u32 == 2);
    assert!(C::Corotated as u32 == 3);
    assert!(C::Snow as u32 == 4);
    assert!(C::DruckerPrager as u32 == 5);
    assert!(C::VonMises as u32 == 6);
    assert!(C::Rankine as u32 == 7);
    assert!(C::DruckerPragerMuI as u32 == 8);
    assert!(C::Viscoelastic as u32 == 9);
    assert!(C::Nacc as u32 == 10);
    assert!(C::GranularFluid as u32 == 11);
};

/// Which role a material plays in two-phase mixture coupling (Tampubolon et al.
/// 2017, "Multi-species simulation of porous sand and water mixtures" --
/// interpenetrating granular-fluid Darcy drag, e.g. water soaking into sand).
/// This is a MATERIAL-level classification (via `MaterialModel::mixture_phase`),
/// not a per-particle field -- every particle of a given material shares the
/// same phase, matching how `constitutive_model` already works. `None` (the
/// default for every existing material) opts a scene entirely out of mixture
/// coupling at zero cost -- see `Grid::has_mixture_activity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixturePhase {
    /// The porous solid (e.g. sand/soil) -- keeps its own full elastic/plastic
    /// deformation, unaffected by mixture coupling beyond the drag force itself.
    Solid,
    /// The interpenetrating fluid (e.g. water) -- exchanges momentum with the
    /// solid phase via Darcy-style drag at every node both phases touch.
    Fluid,
}

pub trait MaterialModel: Send + Sync + core::fmt::Debug {
    /// Which constitutive law this material implements.
    /// Used by the GPU shader to select the correct stress branch per particle.
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fallback
    }
    // Returns the Kirchhoff-like stress used by the transfer kernel.
    // The kernel applies geometry/time factors (dt, kernel_d_inverse, cell_dist, weight).
    fn kirchhoff_stress(&self, _particles: &Particles, _i: usize) -> Mat2 {
        Mat2::ZERO
    }

    // Returns the particle volume used in the stress contribution.
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    /// CFL timestep bound for one particle. Takes `density`/`hardening_scale` as plain
    /// scalars rather than `&Particles, i: usize` — every implementation only ever reads
    /// these two fields, both of which exist directly on `Particle` (AoS) too, so the CPU
    /// (SoA) and GPU (AoS) CFL scans can both call this without either one needing the
    /// other's storage representation.
    fn timestep_bound(
        &self,
        _density: f32,
        _hardening_scale: f32,
        _cell_width: f32,
        _material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        f32::INFINITY
    }

    fn update_particle(&self, _particles: &mut Particles, _i: usize, _dt: f32) {}

    /// Seed per-particle plastic state at spawn time.
    ///
    /// Called once per particle immediately after position/volume assignment.
    /// Default: no-op (elastic materials need no initial plastic state).
    /// Override for materials that have a non-zero neutral accumulator (e.g. sand).
    fn init_particle(&self, _particle: &mut Particle) {}

    /// Whether `update_particle` does real work on the CPU.
    ///
    /// Return `false` if plasticity is fully handled on GPU (default).
    /// Return `true` for CPU-only plasticity paths — the GPU solver uses this to
    /// decide whether to download particles and run the CPU pass each frame.
    fn needs_cpu_update(&self) -> bool {
        false
    }

    /// Whether particles of this material require a per-substep density recompute.
    ///
    /// Fluid EOS materials (Newtonian, Bingham) need up-to-date density each substep
    /// because their pressure is a function of current ρ. Elastic/plastic materials
    /// do not — density is derived from J at the end of update_particle.
    /// Default: false. Override in fluid models.
    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Which two-phase mixture role this material plays, if any -- see
    /// `MixturePhase`'s own doc. `None` (default) means this material never
    /// participates in mixture coupling, the zero-cost-when-unused case that
    /// covers every material/scene that doesn't need this feature.
    fn mixture_phase(&self) -> Option<MixturePhase> {
        None
    }

    /// Scaling coefficient for activation-driven deviatoric stress.
    ///
    /// When non-zero, the per-particle `activation` field (0.0–1.0) modulates the
    /// deviatoric component of the Kirchhoff stress. This is the engine-level hook for
    /// active matter: muscles, motile cells, contractile tissue.
    ///
    /// Physics: τ_total = τ_elastic + activation × coeff × I  (contractile active pressure)
    /// Default: 0.0 — activation has no effect on passive materials.
    fn activation_scale(&self) -> f32 {
        0.0
    }

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }

    /// Energy cost (J/kg, in whatever temperature unit `Particle::temperature` uses)
    /// of transitioning INTO this material via `Simulation::phase_transition` /
    /// `add_phase_rule`. Positive = endothermic (e.g. melting into a liquid — absorbs
    /// energy, cooling the particle). Negative = exothermic (e.g. freezing into a
    /// solid — releases energy, warming the particle). Default 0.0 = no energy cost
    /// (existing behavior for every material, unchanged).
    ///
    /// Applied in `Simulation::phase_transition`/`add_phase_rule` (CPU) against
    /// `ThermalDiffusion::heat_capacity` when a thermal model is configured, and in
    /// `GpuSimulation::phase_transition` against the `heat_capacity` passed to
    /// `attach_thermal_gpu` -- same debit, same formula, on both. GPU has no automatic
    /// `add_phase_rule` counterpart yet (only the manual, one-shot `phase_transition`);
    /// that gap is real and separate from this energy accounting.
    fn latent_heat(&self) -> f32 {
        0.0
    }
}

/// Wraps any `MaterialModel` to give it a non-zero `latent_heat()` without writing a full
/// delegating impl by hand — none of the 12 built-in materials expose a settable
/// `latent_heat` field directly, since most users never need one.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NewtonianFluidMaterial, WithLatentHeat};
/// // Water absorbs 334 (sim-unit) energy per unit mass when transitioning into this material.
/// let water = WithLatentHeat::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1.0e5), 334.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithLatentHeat<M> {
    pub inner: M,
    pub latent_heat: f32,
}

impl<M> WithLatentHeat<M> {
    pub fn new(inner: M, latent_heat: f32) -> Self {
        Self { inner, latent_heat }
    }
}

impl<M: MaterialModel> MaterialModel for WithLatentHeat<M> {
    fn constitutive_model(&self) -> ConstitutiveModel {
        self.inner.constitutive_model()
    }
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        self.inner.kirchhoff_stress(particles, i)
    }
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        self.inner.stress_volume(particles, i)
    }
    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        self.inner.timestep_bound(
            density,
            hardening_scale,
            cell_width,
            material_cfl,
            viscous_cfl,
        )
    }
    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        self.inner.update_particle(particles, i, dt)
    }
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn needs_cpu_update(&self) -> bool {
        self.inner.needs_cpu_update()
    }
    fn needs_density_recompute(&self) -> bool {
        self.inner.needs_density_recompute()
    }
    fn activation_scale(&self) -> f32 {
        self.inner.activation_scale()
    }
    fn params(&self) -> MaterialParams {
        self.inner.params()
    }
    fn latent_heat(&self) -> f32 {
        self.latent_heat
    }
}

/// Wraps any `MaterialModel` to opt it into two-phase mixture coupling as either
/// the `Solid` or `Fluid` phase -- see `MixturePhase`'s own doc. Same pattern as
/// `WithLatentHeat`: existing materials (`DruckerPragerMaterial`,
/// `NewtonianFluidMaterial`, etc.) never opt in by themselves, so combining sand
/// and water in a scene that DOESN'T wrap either one behaves exactly as before
/// (two ordinary single-phase materials sharing a grid, no drag coupling) --
/// mixture physics is explicit, per-scene, not a silent side effect of which
/// materials happen to coexist.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{DruckerPragerMaterial, MixturePhase, WithMixturePhase};
/// let sand = WithMixturePhase::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.2), MixturePhase::Solid);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithMixturePhase<M> {
    pub inner: M,
    pub phase: MixturePhase,
}

impl<M> WithMixturePhase<M> {
    pub fn new(inner: M, phase: MixturePhase) -> Self {
        Self { inner, phase }
    }
}

impl<M: MaterialModel> MaterialModel for WithMixturePhase<M> {
    fn constitutive_model(&self) -> ConstitutiveModel {
        self.inner.constitutive_model()
    }
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        self.inner.kirchhoff_stress(particles, i)
    }
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        self.inner.stress_volume(particles, i)
    }
    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        self.inner.timestep_bound(
            density,
            hardening_scale,
            cell_width,
            material_cfl,
            viscous_cfl,
        )
    }
    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        self.inner.update_particle(particles, i, dt)
    }
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn needs_cpu_update(&self) -> bool {
        self.inner.needs_cpu_update()
    }
    fn needs_density_recompute(&self) -> bool {
        self.inner.needs_density_recompute()
    }
    fn activation_scale(&self) -> f32 {
        self.inner.activation_scale()
    }
    fn params(&self) -> MaterialParams {
        self.inner.params()
    }
    fn latent_heat(&self) -> f32 {
        self.inner.latent_heat()
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        Some(self.phase)
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

// ── props.material(&config) — property-first material construction ─────────────

use physical_props::{BinghamProps, DuctileProps, GranularProps, NewtonianFluid, SnowProps};

impl Elastic {
    /// Canonical model: `NeoHookeanMaterial` (Simo-Pister vol-dev split).
    /// For corotated linear elasticity: `CorotatedMaterial::from_physical(self, config)`.
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(NeoHookeanMaterial::from_physical(self, config))
    }

    /// Particle mass (real SI kilograms -- `rho_kg_m3 * (spacing*dx_meters)^2`, an areal
    /// mass for this 2D solver) for a `SpawnRegion` spawning this material at `spacing`.
    /// Pass to `SpawnRegion { mass_override: Some(props.particle_mass(spacing, &config)),
    /// .. }` — without this, every material in a multi-material scene gets the same
    /// inertia regardless of `rho_kg_m3` (only `SimConfig::particle_mass`, one global
    /// value, is used).
    ///
    /// INVESTIGATED 2026-07-07: briefly "fixed" by adding a `1/dt_seconds^2` factor here,
    /// then REVERTED -- that was the wrong side of the bug. Confirmed by reading
    /// `transfer.rs::scatter_particles_to_grid`: gravity's momentum contribution
    /// (`mass_i * v_i`) and the grid mass accumulator both scale with `mass_i`, but the
    /// STRESS-based momentum contribution does not depend on particle mass at all (pure
    /// `stress * geometry`). Both terms get divided by the SAME grid-node mass during
    /// grid update, so inflating `mass_i` by `1/dt_seconds^2` (often a huge factor, e.g.
    /// 10000x at dt=0.01) dilutes the EOS's restoring force relative to gravity by that
    /// same factor -- confirmed empirically: a water column settled into a stable
    /// equilibrium requiring ~1000x more compression than real hydrostatic physics
    /// needs, not a numerics/CFL issue (resolution-independent, reproduced identically
    /// via both a dropped column and a gentle layer-by-layer pour). The REAL bug was in
    /// `FromSI<NewtonianFluid>`'s (and Bingham/GranularFluid's) `rest_density` conversion
    /// -- see their fix docs. This formula was correct all along for every material
    /// (elastic/plastic solids never referenced `rest_density`, so force-balance was
    /// never in question for them; fluids needed the OTHER side of the ratio fixed).
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.rho_kg_m3 * (spacing * config.dx_meters).powi(2)
    }
}

impl ParticleMass for Elastic {
    fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.particle_mass(spacing, config)
    }
}

impl Elastoplastic {
    /// Dispatches to the correct constitutive model based on `self.model`:
    /// - `Snow`                  → `StomakhinMaterial`
    /// - `Granular`              → `DruckerPragerMaterial`
    /// - `GranularRateDependent` → `MuIRheologyMaterial`
    /// - `Ductile`               → `VonMisesMaterial`
    /// - `Brittle`               → `RankineMaterial`
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        use PlasticityModel::*;
        match self.model {
            Snow => Box::new(StomakhinMaterial::from_physical(
                &SnowProps {
                    elastic: self.elastic,
                },
                config,
            )),
            Granular {
                friction_angle_deg,
                dilatancy_angle_deg,
            } => Box::new(DruckerPragerMaterial::from_physical(
                &GranularProps {
                    elastic: self.elastic,
                    friction_angle_deg,
                    dilatancy_angle_deg,
                },
                config,
            )),
            GranularRateDependent {
                friction_angle_deg,
                dilatancy_angle_deg,
            } => Box::new(MuIRheologyMaterial::from_physical(
                &GranularProps {
                    elastic: self.elastic,
                    friction_angle_deg,
                    dilatancy_angle_deg,
                },
                config,
            )),
            Ductile { yield_stress_pa } => Box::new(VonMisesMaterial::from_physical(
                &DuctileProps {
                    elastic: self.elastic,
                    yield_stress_pa,
                },
                config,
            )),
            Brittle {
                tensile_strength_pa,
                softening_rate,
            } => Box::new(RankineMaterial::from_physical(
                &BrittleProps {
                    elastic: self.elastic,
                    tensile_strength_pa,
                    softening_rate,
                },
                config,
            )),
        }
    }

    /// See `Elastic::particle_mass` — density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl ParticleMass for Elastoplastic {
    fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.particle_mass(spacing, config)
    }
}

impl Viscoelastic {
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(ViscoelasticMaterial::from_physical(self, config))
    }

    /// See `Elastic::particle_mass` — density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl ParticleMass for Viscoelastic {
    fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.particle_mass(spacing, config)
    }
}

impl FluidGranular {
    /// Dispatches to `GranularFluidMaterial` — Tait EOS pressure + corotated deviatoric + SVD plasticity.
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        use physical_props::{scale_lame, scale_stress};
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let (lambda, mu) = scale_lame(self.e_pa, self.nu, self.rho_kg_m3, config);
        let eos = scale_stress(self.bulk_modulus_pa / GAMMA, self.rho_kg_m3, config);
        // See `NewtonianFluidMaterial::from_physical`'s fix doc (2026-07-07) -- rest_density
        // must match `particles.density[i]`'s real units, not an extra `/dt_seconds^2`.
        let rho_grid = self.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Box::new(GranularFluidMaterial {
            mu,
            lambda,
            rest_density: rho_grid,
            eos_stiffness: eos,
            eos_power: GAMMA,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
        })
    }

    /// See `Elastic::particle_mass`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.rho_kg_m3 * (spacing * config.dx_meters).powi(2)
    }
}

impl ParticleMass for FluidGranular {
    fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.particle_mass(spacing, config)
    }
}

impl Fluid {
    /// `yield_stress_pa = None`  → `NewtonianFluidMaterial`
    /// `yield_stress_pa = Some(τ₀)` → `BinghamFluidMaterial`
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        match self.yield_stress_pa {
            None => Box::new(NewtonianFluidMaterial::from_physical(
                &NewtonianFluid {
                    rho_kg_m3: self.rho_kg_m3,
                    eta_pa_s: self.eta_pa_s,
                    bulk_modulus_pa: self.bulk_modulus_pa,
                },
                config,
            )),
            Some(tau0) => Box::new(BinghamFluidMaterial::from_physical(
                &BinghamProps {
                    rho_kg_m3: self.rho_kg_m3,
                    eta_pa_s: self.eta_pa_s,
                    bulk_modulus_pa: self.bulk_modulus_pa,
                    yield_stress_pa: tau0,
                },
                config,
            )),
        }
    }

    /// See `Elastic::particle_mass`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.rho_kg_m3 * (spacing * config.dx_meters).powi(2)
    }
}

impl ParticleMass for Fluid {
    fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.particle_mass(spacing, config)
    }
}

#[cfg(test)]
mod particle_mass_tests {
    use super::*;
    use crate::{SimConfig, SpawnRegion};

    fn earth_config() -> SimConfig {
        SimConfig::earth(64, 0.01, 0.05)
    }

    /// mass_from(&props) == props.particle_mass(spacing) called directly — no duplication risk.
    #[test]
    fn mass_from_matches_direct_call() {
        let config = earth_config();
        let props = Elastic {
            e_pa: 500.0,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        };
        let spacing = 0.5_f32;
        let region = SpawnRegion::for_sim(&config)
            .spacing(spacing)
            .mass_from(&props, &config);
        let expected = props.particle_mass(spacing, &config);
        assert!(
            (region.mass_override.unwrap() - expected).abs() < 1e-9,
            "mass_from result {:.6e} != direct call {:.6e}",
            region.mass_override.unwrap(),
            expected
        );
    }

    /// All 5 property families implement ParticleMass identically.
    #[test]
    fn all_families_implement_particle_mass() {
        let config = earth_config();
        let spacing = 0.6_f32;
        let expected_elastic = Elastic {
            e_pa: 500.0,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        }
        .particle_mass(spacing, &config);
        let from_ep = Elastoplastic {
            elastic: Elastic {
                e_pa: 500.0,
                nu: 0.45,
                rho_kg_m3: 1000.0,
            },
            model: PlasticityModel::Snow,
        }
        .particle_mass(spacing, &config);
        let from_ve = Viscoelastic {
            elastic: Elastic {
                e_pa: 500.0,
                nu: 0.45,
                rho_kg_m3: 1000.0,
            },
            eta_pa_s: 1.0,
        }
        .particle_mass(spacing, &config);
        let from_fluid = Fluid {
            rho_kg_m3: 1000.0,
            eta_pa_s: 0.001,
            bulk_modulus_pa: 2.2e9,
            yield_stress_pa: None,
        }
        .particle_mass(spacing, &config);
        assert!((from_ep - expected_elastic).abs() < 1e-9);
        assert!((from_ve - expected_elastic).abs() < 1e-9);
        assert!((from_fluid - expected_elastic).abs() < 1e-9);
    }
}

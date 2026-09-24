//! Property-first material construction: `props.material(&config)` /
//! `props.particle_mass(spacing, &config)` for each of the 7 property
//! families (`Elastic`, `Elastoplastic`, `Viscoelastic`, `Pressurized`,
//! `NoCompression`, `FluidGranular`, `Fluid`, defined in `physical_props.rs`)
//! — dispatching each preset to its concrete `MaterialModel` constructor.
//! Split out of `mod.rs` purely for LOC — no behavior change.

use super::physical_props::{BinghamProps, DuctileProps, GranularProps, NewtonianFluid, SnowProps};
use super::{
    BinghamFluidMaterial, BrittleProps, DruckerPragerMaterial, Elastic, Elastoplastic, Fluid,
    FluidGranular, FromSI, GranularFluidMaterial, MaterialModel, MuIRheologyMaterial,
    NeoHookeanMaterial, NewtonianFluidMaterial, NoCompression, NoCompressionMaterial, ParticleMass,
    PlasticityModel, Pressurized, RankineMaterial, StomakhinMaterial, Viscoelastic,
    ViscoelasticMaterial, VonMisesMaterial, WithPreStress,
};

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
    /// Do not add a `1/dt_seconds^2` factor here to fix fluid force balance --
    /// this formula is correct as-is for every material. The scaling that
    /// matters for fluids lives in `FromSI<NewtonianFluid>` (and
    /// Bingham/GranularFluid)'s `rest_density` conversion; see their docs.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.rho_kg_m3 * (spacing * config.dx_meters).powi(2)
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

impl Viscoelastic {
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(ViscoelasticMaterial::from_physical(self, config))
    }

    /// See `Elastic::particle_mass` — density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl Pressurized {
    /// `NeoHookeanMaterial` (the elastic base) wrapped in `WithPreStress`, carrying the
    /// real SI-converted internal pressure. Reuses the same `scale_stress` Pa→grid-unit
    /// conversion already used for `e_pa`/`bulk_modulus_pa` elsewhere -- pressure is the
    /// same physical quantity (stress), so no new conversion formula is needed.
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        use super::physical_props::scale_stress;
        let base = NeoHookeanMaterial::from_physical(&self.elastic, config);
        let pressure_grid = scale_stress(self.internal_pressure_pa, self.elastic.rho_kg_m3, config);
        Box::new(WithPreStress::new(base, pressure_grid))
    }

    /// See `Elastic::particle_mass` — density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl NoCompression {
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(NoCompressionMaterial::from_physical(&self.elastic, config))
    }

    /// See `Elastic::particle_mass` — density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl FluidGranular {
    /// Dispatches to `GranularFluidMaterial` — Tait EOS pressure + corotated deviatoric + SVD plasticity.
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        use super::physical_props::scale_lame;
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let (lambda, mu) = scale_lame(self.e_pa, self.nu, self.rho_kg_m3, config);
        // Real fix 2026-08-11: this EOS/bulk-pressure term is the SAME
        // density-ratio Tait pressure `NewtonianFluidMaterial`/
        // `BinghamFluidMaterial` use (`p = k*((rho/rho0)^gamma - 1)`, see
        // `GranularFluidMaterial::kirchhoff_stress`) -- their own
        // `from_physical` doc says applying `scale_stress`'s legacy
        // dt^2/(rho*dx^2) conversion here "would double-scale it". This
        // used to call `scale_stress(self.bulk_modulus_pa / GAMMA, ...)`,
        // the exact same abandoned path the WCSPH diagnostic test was
        // caught using (see that test's own doc) -- real, live bug, not
        // just a test issue: it made every `FluidGranular`-dispatched mud/
        // wet-terrain material's bulk pressure orders of magnitude too
        // soft to resist compression. `lambda`/`mu` above stay on
        // `scale_lame` correctly -- that term is added to the SAME
        // F-based corotated elastic stress space every other solid
        // material uses, a genuinely different (and correctly scaled)
        // pipeline from the density-ratio EOS pressure below.
        let eos = self.bulk_modulus_pa / GAMMA;
        // See `NewtonianFluidMaterial::from_physical`'s doc -- rest_density
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
            // Same real, disclosed damping convention as `GranularFluidMaterial::
            // saturated_loam` (see that field's own doc on the struct) --
            // `FluidGranular` itself doesn't yet expose a distinct viscosity
            // input, so this generic SI-driven dispatch path uses the same
            // 0.3*mu default rather than silently shipping zero damping here too.
            dynamic_viscosity: 0.3 * mu,
            // Real correction -- see `GranularFluidMaterial::saturated_loam`'s
            // own note: scales with THIS scene's own real (SI-derived)
            // eos_stiffness, not mu.
            bulk_viscosity: 0.5 * eos,
        })
    }

    /// See `Elastic::particle_mass`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.rho_kg_m3 * (spacing * config.dx_meters).powi(2)
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

/// Every property family above already has an inherent `particle_mass` method (used
/// directly by its own `material()`); this just forwards the `ParticleMass` trait to
/// that same inherent method, identically for all 7 families -- see `SpawnRegion::
/// mass_from`, the only real caller, which needs the trait (not each type's own
/// inherent method) to stay generic over property-family type.
macro_rules! forward_particle_mass {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ParticleMass for $ty {
                fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
                    self.particle_mass(spacing, config)
                }
            }
        )+
    };
}

forward_particle_mass!(
    Elastic,
    Elastoplastic,
    Viscoelastic,
    Pressurized,
    NoCompression,
    FluidGranular,
    Fluid,
);

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

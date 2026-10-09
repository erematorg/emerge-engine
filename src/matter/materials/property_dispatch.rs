//! Property-first material construction: `props.material(&config)` /
//! `props.particle_mass(spacing, &config)` for each of the 7 property
//! families (`Elastic`, `Elastoplastic`, `Viscoelastic`, `Pressurized`,
//! `NoCompression`, `FluidGranular`, `Fluid`, defined in `physical_props.rs`)
//! -- dispatching each preset to its concrete `MaterialModel` constructor.
//! Split out of `mod.rs` purely for LOC -- no behavior change.

use super::physical_props::{
    BinghamProps, DuctileProps, GranularProps, NaccProps, NewtonianFluid, SnowProps,
};
use super::{
    BinghamFluidMaterial, BrittleProps, DruckerPragerMaterial, Elastic, Elastoplastic, Fluid,
    FluidGranular, FromSI, GranularFluidMaterial, MaterialModel, MuIRheologyMaterial, NaccMaterial,
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

    /// Particle mass in real SI kilograms -- `rho_kg_m3 * (spacing*dx_meters)^2`,
    /// an areal mass for this 2D solver.
    ///
    /// Do NOT assign this to `SpawnRegion::mass_override`: that field is in GRID
    /// units, and the two differ by `rho * dx_meters^2`. Use
    /// `SpawnRegion::mass_from(&props, &config)`, which applies the conversion.
    /// A multi-material scene needs it so regions differ in inertia and not only
    /// in stiffness; a single-material scene does not, since
    /// `SimConfig::grid_density` already puts it at grid density 1.
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
    /// - `CamClay`               → `NaccMaterial` (CPU-only, see that
    ///   material's doc -- GPU construction rejects it)
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
            CamClay {
                friction,
                cohesion,
                compression_index,
                swelling_index,
                void_ratio,
            } => Box::new(NaccMaterial::from_physical(
                &NaccProps {
                    elastic: self.elastic,
                    friction,
                    cohesion,
                    compression_index,
                    swelling_index,
                    void_ratio,
                    preconsolidation_pa: 0.0,
                },
                config,
            )),
        }
    }

    /// See `Elastic::particle_mass` -- density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl NaccProps {
    /// See `Elastic::particle_mass` -- density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl Viscoelastic {
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(ViscoelasticMaterial::from_physical(self, config))
    }

    /// See `Elastic::particle_mass` -- density lives in `self.elastic.rho_kg_m3`.
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

    /// See `Elastic::particle_mass` -- density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl NoCompression {
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(NoCompressionMaterial::from_physical(&self.elastic, config))
    }

    /// See `Elastic::particle_mass` -- density lives in `self.elastic.rho_kg_m3`.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl FluidGranular {
    /// Dispatches to `GranularFluidMaterial` -- Tait EOS pressure + corotated deviatoric + SVD plasticity.
    pub fn material(&self, config: &crate::SimConfig) -> Box<dyn MaterialModel> {
        Box::new(self.granular_fluid_material(config))
    }

    fn granular_fluid_material(&self, config: &crate::SimConfig) -> GranularFluidMaterial {
        use super::physical_props::{scale_lame, scale_stress};
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let (lambda, mu) = scale_lame(self.e_pa, self.nu, self.rho_kg_m3, config);
        // The EOS pressure (`p = k*((rho/rho0)^gamma - 1)`, see
        // `GranularFluidMaterial::kirchhoff_stress`) converts exactly as
        // `NewtonianFluidMaterial::from_physical` does: the stiffness through
        // `scale_stress` at this material's density, the rest density as a
        // ratio to the scene's reference density, the unit
        // `particles.density[i]` comes out in.
        let eos = scale_stress(self.bulk_modulus_pa / GAMMA, self.rho_kg_m3, config);
        let rho_grid = self.rho_kg_m3 / config.reference_density_kg_m3;
        GranularFluidMaterial {
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
            // Same damping convention as `GranularFluidMaterial::saturated_loam`
            // (see that field's doc on the struct): `FluidGranular` has no
            // viscosity input yet, so this uses the same 0.3*mu default rather
            // than zero damping.
            dynamic_viscosity: 0.3 * mu,
            // Scales with this material's eos_stiffness, not mu (see
            // `GranularFluidMaterial::saturated_loam`).
            bulk_viscosity: 0.5 * eos,
        }
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
                    // `Fluid` describes a liquid, and a liquid has no
                    // storage modulus. Reaching the elastoviscoplastic
                    // branch is a deliberate act via `BinghamProps`, not
                    // something the liquid route turns on behind the caller.
                    shear_modulus_pa: 0.0,
                    // A liquid carrying a yield stress is a mixed suspension, so it
                    // carries mixed-in gas. Leaving this at 0.0 would contradict the
                    // Newtonian branch just above, which already states its own
                    // cavitation figure, and would contradict it in the one direction
                    // that lets volume only ever grow.
                    cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
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

impl GranularProps {
    /// See `Elastic::particle_mass`. Delegates to the shared elastic
    /// density -- `DruckerPragerMaterial`/`MuIRheologyMaterial` add no
    /// separate mass concept of their own on top of it.
    pub fn particle_mass(&self, spacing: f32, config: &crate::SimConfig) -> f32 {
        self.elastic.particle_mass(spacing, config)
    }
}

impl BinghamProps {
    /// See `Elastic::particle_mass`. Present for the same reason as every
    /// other family's: a caller building this material directly (rather
    /// than through `Fluid::material`) still needs its particle mass.
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
    BinghamProps,
    GranularProps,
    NaccProps,
);

#[cfg(test)]
mod particle_mass_tests {
    use super::*;
    use crate::{SimConfig, SpawnRegion};

    fn earth_config() -> SimConfig {
        SimConfig::earth(64, 0.01, 0.05)
    }

    /// mass_from(&props) == props.particle_mass(spacing) converted through the
    /// same SI-kg -> grid-unit factor `mass_from` itself applies -- no
    /// duplication risk between the two `particle_mass` call sites.
    ///
    /// `expected` goes through the same SI-kg -> grid-unit conversion as
    /// `mass_from` (src/spacetime/solver/config/spawn.rs), not
    /// `particle_mass`'s raw SI-kg return.
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
        let si_kg = props.particle_mass(spacing, &config);
        let to_grid = 1.0 / (config.reference_density_kg_m3 * config.dx_meters * config.dx_meters);
        let expected = si_kg * to_grid;
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

    /// `PlasticityModel::CamClay` dispatch must produce the same
    /// `NaccMaterial` as calling `NaccMaterial::from_physical` directly: no
    /// double conversion, no dropped fields, as the `Brittle`/`Ductile`/
    /// `Granular` arms check.
    #[test]
    fn camclay_dispatch_matches_direct_nacc_from_physical() {
        use crate::materials::NaccMaterial;

        let config = earth_config();
        let elastic = Elastic {
            e_pa: 2.0e6,
            nu: 0.3,
            rho_kg_m3: 1800.0,
        };
        let (friction, cohesion) = (1.2, 0.1);
        let (compression_index, swelling_index, void_ratio) = (0.12, 0.023, 1.7);

        let via_dispatch = Elastoplastic {
            elastic,
            model: PlasticityModel::CamClay {
                friction,
                cohesion,
                compression_index,
                swelling_index,
                void_ratio,
            },
        }
        .material(&config);

        let direct = NaccMaterial::from_physical(
            &NaccProps {
                elastic,
                friction,
                cohesion,
                compression_index,
                swelling_index,
                void_ratio,
                preconsolidation_pa: 0.0,
            },
            &config,
        );

        // `.material()` returns `Box<dyn MaterialModel>`; `MaterialModel:
        // Debug` is a supertrait bound, so comparing the trait objects'
        // Debug output directly (not just a `MaterialParams` projection)
        // catches any field the dispatch path might drop or double-convert.
        assert_eq!(
            format!("{via_dispatch:?}"),
            format!("{direct:?}"),
            "CamClay dispatch produced a different NaccMaterial than the direct FromSI path"
        );

        // particle_mass for Elastoplastic must still route through the same
        // elastic density, unaffected by which PlasticityModel variant is chosen.
        let spacing = 0.5_f32;
        let expected_mass = elastic.particle_mass(spacing, &config);
        let ep = Elastoplastic {
            elastic,
            model: PlasticityModel::CamClay {
                friction,
                cohesion,
                compression_index,
                swelling_index,
                void_ratio,
            },
        };
        assert!((ep.particle_mass(spacing, &config) - expected_mass).abs() < 1e-9);
    }

    /// `FluidGranular`'s EOS converts like the Newtonian fluid's: the same bulk
    /// modulus and density give the same grid stiffness and rest density, so
    /// mud and water in one scene read `particles.density` in one unit.
    #[test]
    fn fluid_granular_eos_converts_like_newtonian_fluid() {
        let config = earth_config();
        let mud = FluidGranular::saturated_loam_preset();
        let granular = mud.granular_fluid_material(&config);
        let fluid = NewtonianFluidMaterial::from_physical(
            &NewtonianFluid {
                rho_kg_m3: mud.rho_kg_m3,
                eta_pa_s: 1.0e-3,
                bulk_modulus_pa: mud.bulk_modulus_pa,
            },
            &config,
        );
        assert_eq!(granular.eos_stiffness, fluid.eos_stiffness);
        assert_eq!(granular.rest_density, fluid.rest_density);
    }
}

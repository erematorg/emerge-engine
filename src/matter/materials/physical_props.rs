//! Physical property families — the entry point for all material construction.
//!
//! Five families cover all continuum matter:
//! - [`Elastic`]        — pure elastic solid (NeoHookean / Corotated)
//! - [`Elastoplastic`]  — elastic + plastic yield (snow, granular, ductile, brittle)
//! - [`Viscoelastic`]   — elastic + viscous damping (Kelvin-Voigt)
//! - [`Fluid`]          — viscous fluid (Newtonian if no yield, Bingham if yield set)
//! - [`FluidGranular`]  — fluid-granular blend (EOS pressure + corotated deviatoric + SVD plasticity = mud)
//!
//! # Usage
//! ```rust,no_run
//! # extern crate emerge_engine as emerge;
//! use emerge::{Elastic, Elastoplastic, Fluid, PlasticityModel,
//!              SimConfig, Viscoelastic};
//!
//! let config = SimConfig::earth(64, 0.01, 0.05);
//!
//! // Soft elastic solid (E=500 Pa, ν=0.45, ρ=1000 kg/m³)
//! let mat = Elastic { e_pa: 500.0, nu: 0.45, rho_kg_m3: 1000.0 }.material(&config);
//!
//! // Cohesionless granular (E=50 MPa, φ=35°)
//! let mat = Elastoplastic {
//!     elastic: Elastic { e_pa: 50e6, nu: 0.3, rho_kg_m3: 1600.0 },
//!     model: PlasticityModel::Granular { friction_angle_deg: 35.0, dilatancy_angle_deg: 0.0 },
//! }.material(&config);
//!
//! // Snow (E=2 MPa, ρ=200 kg/m³, Stomakhin 2013 plasticity)
//! let mat = Elastoplastic {
//!     elastic: Elastic { e_pa: 2e6, nu: 0.2, rho_kg_m3: 200.0 },
//!     model: PlasticityModel::Snow,
//! }.material(&config);
//!
//! // Viscoplastic fluid (ρ=1500, τ₀=100 Pa → Bingham)
//! let mat = Fluid {
//!     rho_kg_m3: 1500.0, eta_pa_s: 0.5, bulk_modulus_pa: 1.5e9,
//!     yield_stress_pa: Some(100.0),
//! }.material(&config);
//! ```

use crate::SimConfig;

// ── Public property families ──────────────────────────────────────────────────

/// Pure elastic solid.
///
/// Default constitutive model: `NeoHookeanMaterial`.
/// For corotated linear elasticity use `CorotatedMaterial::from_physical`.
#[derive(Debug, Clone, Copy)]
pub struct Elastic {
    /// Young's modulus `[Pa]`
    pub e_pa: f32,
    /// Poisson's ratio (dimensionless, −1 < ν < 0.5)
    pub nu: f32,
    /// Rest density `[kg/m³]`
    pub rho_kg_m3: f32,
}

/// Elastoplastic solid: elastic skeleton + a plastic yield rule.
///
/// Pick the yield criterion via [`PlasticityModel`].
/// Default constitutive model dispatched from `model` field.
#[derive(Debug, Clone, Copy)]
pub struct Elastoplastic {
    pub elastic: Elastic,
    pub model: PlasticityModel,
}

/// Plastic yield criterion for [`Elastoplastic`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum PlasticityModel {
    /// Volumetric snow plasticity (Stomakhin 2013).
    /// Hardening ξ=10, critical compression θ_c=0.025, critical stretch θ_s=0.0075.
    /// No extra parameters — determined by MPM snow physics.
    Snow,

    /// Drucker-Prager cohesionless granular (rate-independent).
    /// → `DruckerPragerMaterial`
    Granular {
        /// Peak internal friction angle `[degrees]`. Dry sand ≈ 30–38°.
        friction_angle_deg: f32,
        /// Reynolds dilatancy angle `[degrees]`. 0 = non-dilatant.
        dilatancy_angle_deg: f32,
    },

    /// µ(I)-rheology rate-dependent granular flow (Cicoira et al. DPMui, see
    /// `MuIRheologyMaterial`'s own doc for the full, corrected citation).
    /// Better for dense granular at high shear rates. CPU-only.
    /// → `MuIRheologyMaterial`
    GranularRateDependent {
        /// Static friction angle `[degrees]`.
        friction_angle_deg: f32,
        /// Dilatancy angle `[degrees]`.
        dilatancy_angle_deg: f32,
    },

    /// J2 ductile plastic flow (von Mises), linear isotropic hardening.
    /// → `VonMisesMaterial`
    Ductile {
        /// Yield stress `[Pa]`. Flow begins above this deviatoric stress.
        yield_stress_pa: f32,
    },

    /// Tensile cutoff + exponential softening (Rankine criterion).
    /// Models brittle fracture under tension. CPU-only.
    /// → `RankineMaterial`
    Brittle {
        /// Tensile strength `[Pa]`. Fracture initiates above this.
        tensile_strength_pa: f32,
        /// Exponential softening rate. Higher = faster strength loss post-fracture.
        softening_rate: f32,
    },
}

/// Viscoelastic solid (Kelvin-Voigt): elastic spring + viscous dashpot in parallel.
///
/// The material deforms elastically AND dissipates energy simultaneously.
/// Creep under constant stress eventually stops (spring limits deformation).
/// → `ViscoelasticMaterial`
#[derive(Debug, Clone, Copy)]
pub struct Viscoelastic {
    pub elastic: Elastic,
    /// Dynamic viscosity η `[Pa·s]`
    pub eta_pa_s: f32,
}

/// Fluid-granular blend: EOS pressure + corotated elastic deviatoric + SVD plasticity.
///
/// → `GranularFluidMaterial`
///
/// Use for wet terrain substrates, saturated granular flows, biological cell matrices.
/// Distinct from `Fluid` (no elastic restoring force) and `Elastoplastic` (no EOS bulk pressure).
#[derive(Debug, Clone, Copy)]
pub struct FluidGranular {
    /// Rest density `[kg/m³]`
    pub rho_kg_m3: f32,
    /// Bulk modulus K `[Pa]` — EOS stiffness. Controls compressibility.
    pub bulk_modulus_pa: f32,
    /// Young's modulus E `[Pa]` — elastic shear stiffness. Controls shape-restoring force.
    pub e_pa: f32,
    /// Poisson's ratio ν
    pub nu: f32,
    /// Max elastic compression before plastic yield (fraction): singular values clamped at (1−θ_c).
    /// Larger = more elastic range before mud flows. 0.2–0.6 for wet mud.
    pub compression_limit: f32,
    /// Max elastic stretch before plastic yield (fraction). Small (0.01–0.05) keeps mud cohesion low.
    pub stretch_limit: f32,
    /// Hardening exponent ξ. h = exp(ξ·(1−Jp)). 0 = no hardening, 3–8 for compacting mud.
    pub hardening_exponent: f32,
}

impl FluidGranular {
    /// Saturated loam — yields easily, flows slowly under sustained load.
    ///
    /// HONEST DISCLOSURE (audit 2026-07-17, same finding as `GranularFluidMaterial`'s
    /// own presets in `granular_fluid.rs`): `scale_lame`/`scale_stress` below DO
    /// perform a real, dimensionally-consistent SI-to-grid-unit conversion (same
    /// pattern already verified for `NewtonianFluidMaterial`), so the MECHANISM here is
    /// sound. But these specific SI values (`rho_kg_m3`, `bulk_modulus_pa`, `e_pa`,
    /// `compression_limit` etc.) are not tied to any specific real measurement/paper —
    /// `rho_kg_m3=1800`/`e_pa=5e3` are plausible ballpark figures for real wet loam, not
    /// verified against one. Presenting them in real SI units carries a stronger
    /// implicit "this is measured" claim than a dimensionless test parameter would, so
    /// this needs the same honest flag: real conversion math, unverified specific
    /// numbers, not yet a literature-sourced material.
    pub fn saturated_loam() -> Self {
        Self {
            rho_kg_m3: 1800.0,
            bulk_modulus_pa: 2.0e5,
            e_pa: 5.0e3,
            nu: 0.3,
            compression_limit: 0.4,
            stretch_limit: 0.01,
            hardening_exponent: 5.0,
        }
    }

    /// Consolidated clay — stiffer shear, slow plastic creep.
    ///
    /// Same honest disclosure as `saturated_loam` above: real conversion mechanism,
    /// unverified specific SI values.
    pub fn consolidated_clay() -> Self {
        Self {
            rho_kg_m3: 2000.0,
            bulk_modulus_pa: 8.0e5,
            e_pa: 2.0e4,
            nu: 0.35,
            compression_limit: 0.3,
            stretch_limit: 0.01,
            hardening_exponent: 3.0,
        }
    }

    /// Cytoplasmic matrix — very soft elastic, near-fluid, large yield surface.
    ///
    /// Same honest disclosure as `saturated_loam` above: real conversion mechanism,
    /// unverified specific SI values (though `e_pa=500` is at least in the right real
    /// ballpark per AFM cytoplasm-stiffness literature -- not yet tied to a specific
    /// paper).
    pub fn cytoplasmic() -> Self {
        Self {
            rho_kg_m3: 1050.0,
            bulk_modulus_pa: 2.0e4,
            e_pa: 500.0,
            nu: 0.45,
            compression_limit: 0.6,
            stretch_limit: 0.05,
            hardening_exponent: 1.0,
        }
    }
}

/// Viscous fluid (Tait EOS + shear viscosity).
///
/// - `yield_stress_pa = None`  → Newtonian (flow at any stress) → `NewtonianFluidMaterial`
/// - `yield_stress_pa = Some(τ₀)` → Bingham viscoplastic (rigid plug below τ₀) → `BinghamFluidMaterial`
#[derive(Debug, Clone, Copy)]
pub struct Fluid {
    pub rho_kg_m3: f32,
    /// Dynamic (shear) viscosity η `[Pa·s]`
    pub eta_pa_s: f32,
    /// Bulk modulus K `[Pa]`. Sets EOS stiffness (compressibility).
    /// Real water K ≈ 2.2 GPa. Use K = ρ·c_ref²/γ with c_ref = 10·v_max for weakly-compressible.
    pub bulk_modulus_pa: f32,
    /// `None` = Newtonian. `Some(τ₀)` = Bingham: plug flow below τ₀ `[Pa]`.
    pub yield_stress_pa: Option<f32>,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Convert a physical property description to a grid-unit material model.
///
/// `P` is the property family. `config` supplies `dx_meters` and `dt_seconds`
/// for non-dimensionalization.
///
/// Used internally by `.material()` and available for advanced overrides
/// (e.g. `CorotatedMaterial::from_physical(&elastic_props, &config)`).
pub trait FromSI<P> {
    fn from_physical(props: &P, config: &SimConfig) -> Self;
}

/// Particle mass (grid units) for a `SpawnRegion` spawning this material at a
/// given spacing — `rho_kg_m3 * (spacing * dx_meters)^2` for a 2D areal-density
/// particle. Implemented identically by every physical-property family so
/// `SpawnRegion::mass_from` can stay generic over which material is being spawned.
pub trait ParticleMass {
    fn particle_mass(&self, spacing: f32, config: &SimConfig) -> f32;
}

// ── Internal bridging structs (pub(super) — not part of LP API) ──────────────
//
// These carry the exact parameters that each material impl's `from_physical` needs.
// They are constructed inside `.material()` dispatch — callers never see them.

#[derive(Debug, Clone, Copy)]
pub(super) struct GranularProps {
    pub elastic: Elastic,
    pub friction_angle_deg: f32,
    pub dilatancy_angle_deg: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DuctileProps {
    pub elastic: Elastic,
    pub yield_stress_pa: f32,
}

/// Real-unit properties for brittle/tensile-failure materials (`RankineMaterial`),
/// and the source of correctly-scaled Lame/tensile-strength parameters for
/// `rankine_damage_estimate` when a non-Rankine material still needs a real
/// damage signal (construct a `RankineMaterial::from_physical(&props, &config)`
/// and read its already-scaled `.lambda`/`.mu`/`.tensile_strength`/
/// `.softening_rate` fields -- don't hand-scale a Pa value yourself).
#[derive(Debug, Clone, Copy)]
pub struct BrittleProps {
    pub elastic: Elastic,
    pub tensile_strength_pa: f32,
    pub softening_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SnowProps {
    pub elastic: Elastic,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NewtonianFluid {
    pub rho_kg_m3: f32,
    pub eta_pa_s: f32,
    pub bulk_modulus_pa: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BinghamProps {
    pub rho_kg_m3: f32,
    pub eta_pa_s: f32,
    pub bulk_modulus_pa: f32,
    pub yield_stress_pa: f32,
}

// ── Scaling helpers (pub(super) — used by material impls) ─────────────────────

/// Scale SI stress (Pa) to grid units: `p_grid = p_SI · dt² / (ρ · dx²)`.
#[inline]
pub(super) fn scale_stress(pa: f32, rho: f32, config: &SimConfig) -> f32 {
    config.stress_from_si(pa, rho)
}

/// Scale SI viscosity (Pa·s) to grid units: `η_grid = η_SI · ρ · dx² / dt³`.
#[inline]
pub(super) fn scale_visc(eta: f32, rho: f32, config: &SimConfig) -> f32 {
    config.visc_from_si(eta, rho)
}

/// Scale SI Young's modulus to grid Lamé parameters.
#[inline]
pub(super) fn scale_lame(e_pa: f32, nu: f32, rho: f32, config: &SimConfig) -> (f32, f32) {
    config.lame_from_si_cfg(e_pa, nu, rho)
}

// ── Reference SI values used in unit tests below ─────────────────────────────
#[cfg(test)]
mod _ref {
    use super::*;

    // Elastic — E [Pa], ν, ρ [kg/m³]
    pub const SOFT_ELASTIC: Elastic = Elastic {
        e_pa: 500.0,
        nu: 0.45,
        rho_kg_m3: 1000.0,
    };

    // Viscoelastic — η [Pa·s]
    pub const SOFT_VISCOELASTIC: Viscoelastic = Viscoelastic {
        elastic: Elastic {
            e_pa: 50_000.0,
            nu: 0.45,
            rho_kg_m3: 1100.0,
        },
        eta_pa_s: 10.0,
    };

    // Granular — φ=35°
    pub const COHESIONLESS_GRANULAR: Elastoplastic = Elastoplastic {
        elastic: Elastic {
            e_pa: 50.0e6,
            nu: 0.3,
            rho_kg_m3: 1600.0,
        },
        model: super::PlasticityModel::Granular {
            friction_angle_deg: 35.0,
            dilatancy_angle_deg: 0.0,
        },
    };

    // Snow
    pub const LOW_DENSITY_GRANULAR: Elastoplastic = Elastoplastic {
        elastic: Elastic {
            e_pa: 2.0e6,
            nu: 0.20,
            rho_kg_m3: 200.0,
        },
        model: super::PlasticityModel::Snow,
    };

    // Ductile — σ_Y=30 kPa
    pub const SOFT_DUCTILE: Elastoplastic = Elastoplastic {
        elastic: Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1800.0,
        },
        model: super::PlasticityModel::Ductile {
            yield_stress_pa: 30_000.0,
        },
    };

    // Brittle — σ_t=10 MPa
    pub const STIFF_BRITTLE: Elastoplastic = Elastoplastic {
        elastic: Elastic {
            e_pa: 70.0e9,
            nu: 0.25,
            rho_kg_m3: 2700.0,
        },
        model: super::PlasticityModel::Brittle {
            tensile_strength_pa: 10.0e6,
            softening_rate: 3.0,
        },
    };

    // Fluid — Newtonian (no yield)
    pub const LOW_VISCOSITY_FLUID: Fluid = Fluid {
        rho_kg_m3: 1000.0,
        eta_pa_s: 0.001,
        bulk_modulus_pa: 2.2e9,
        yield_stress_pa: None,
    };

    // Fluid — Bingham (yield=100 Pa)
    pub const VISCOPLASTIC_FLUID: Fluid = Fluid {
        rho_kg_m3: 1500.0,
        eta_pa_s: 0.5,
        bulk_modulus_pa: 1.5e9,
        yield_stress_pa: Some(100.0),
    };

    /// Verify all reference presets construct successfully — catches API breakage.
    #[test]
    fn all_presets_build() {
        use crate::solver::config::SimConfig;
        use glam::Vec2;
        let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
        let _ = SOFT_ELASTIC.material(&config);
        let _ = SOFT_VISCOELASTIC.material(&config);
        let _ = COHESIONLESS_GRANULAR.material(&config);
        let _ = LOW_DENSITY_GRANULAR.material(&config);
        let _ = SOFT_DUCTILE.material(&config);
        let _ = STIFF_BRITTLE.material(&config);
        let _ = LOW_VISCOSITY_FLUID.material(&config);
        let _ = VISCOPLASTIC_FLUID.material(&config);
        let _ = FluidGranular::saturated_loam().material(&config);
    }
}

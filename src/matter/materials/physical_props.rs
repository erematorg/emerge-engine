//! Physical property families -- the entry point for all material construction.
//!
//! Five families cover all continuum matter:
//! - [`Elastic`]        -- pure elastic solid (NeoHookean / Corotated)
//! - [`Elastoplastic`]  -- elastic + plastic yield (snow, granular, ductile, brittle)
//! - [`Viscoelastic`]   -- elastic + viscous damping (Kelvin-Voigt)
//! - [`Fluid`]          -- viscous fluid (Newtonian if no yield, Bingham if yield set)
//! - [`FluidGranular`]  -- fluid-granular blend (EOS pressure + corotated deviatoric + SVD plasticity = mud)
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
    /// No extra parameters -- determined by MPM snow physics.
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
    /// Better for dense granular at high shear rates. CPU + GPU.
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
    /// Models brittle fracture under tension. CPU + GPU.
    /// → `RankineMaterial`
    Brittle {
        /// Tensile strength `[Pa]`. Fracture initiates above this.
        tensile_strength_pa: f32,
        /// Exponential softening rate. Higher = faster strength loss post-fracture.
        softening_rate: f32,
    },

    /// Non-Associated Cam-Clay: elliptical yield surface with a compression
    /// cap (preconsolidation), for wet soil/clay/soft tissue (Klar et al.
    /// 2016; see `NaccMaterial`'s own doc for the full citation and natural
    /// phenomena). CPU-only (GPU construction rejects it -- see
    /// `GpuSimulation`'s own NACC guard). Wired in 2026-09-08, closing the
    /// gap `NaccProps`'s own doc used to disclose.
    /// → `NaccMaterial`
    CamClay {
        /// Friction slope M (tan-like, not a raw angle). Typical 0.8-1.8 --
        /// see `NaccMaterial::friction`'s own doc for the exact relation.
        friction: f32,
        /// Cohesion β. 0.0 = no tensile strength (standard soil).
        cohesion: f32,
        /// Compression index λ of the soil's oedometer curve.
        compression_index: f32,
        /// Swelling index κ of the same curve.
        swelling_index: f32,
        /// Void ratio e at the reference state.
        void_ratio: f32,
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

/// Elastic solid under real internal pre-stress pressure -- a "prestressed structure"
/// (Kirchhoff stress gets an added isotropic `-P·I` term; see `Particle::internal_pressure`
/// doc for the full mechanism). Real motivating case: turgor pressure, the internal
/// hydrostatic pressure that does real structural work in plant cells, genuinely
/// distinct from cell-wall elastic stiffness (Niklas 1992's "hydro-skeleton" theory) --
/// but generic, not plant-specific: any internally-pressurized body.
///
/// → `NeoHookeanMaterial` wrapped in `WithPreStress`
#[derive(Debug, Clone, Copy)]
pub struct Pressurized {
    pub elastic: Elastic,
    /// Internal pre-stress pressure `[Pa]`. Real, measured range for healthy plant
    /// cells: 0.2–2.0 MPa (root cells ~0.6 MPa, leaf epidermal cells 1.5–2.0 MPa --
    /// Niklas 1992; Wikipedia "Turgor pressure", sourced from real measurements).
    pub internal_pressure_pa: f32,
}

/// Tension-only (no-compression) elastic solid -- real, established continuum theory
/// for cables, membranes, tendons, spider silk (see `NoCompressionMaterial`'s own doc
/// for the full citation). Fully reversible, distinct from `Elastoplastic` -- this is
/// an asymmetric nonlinear ELASTIC law (goes slack under compression, regains full
/// stiffness under tension with no memory), not an irreversible yield criterion.
///
/// → `NoCompressionMaterial`
#[derive(Debug, Clone, Copy)]
pub struct NoCompression {
    pub elastic: Elastic,
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
    /// Bulk modulus K `[Pa]` -- EOS stiffness. Controls compressibility.
    pub bulk_modulus_pa: f32,
    /// Young's modulus E `[Pa]` -- elastic shear stiffness. Controls shape-restoring force.
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
    // Suffixed `_preset` to disambiguate from `GranularFluidMaterial`'s own,
    // differently-parameterized presets of the same name
    // (`saturated_loam`/`consolidated_clay`/`cytoplasmic`) in
    // `granular_fluid.rs` (zero-arg fixed-SI-literature-value here vs.
    // parameterized `(young_modulus, poisson_ratio)` there). Prefer the
    // unsuffixed, parameterized versions unless you specifically want this
    // property family's fixed literature-style defaults.

    /// Saturated loam -- yields easily, flows slowly under sustained load.
    ///
    /// UPDATED (2026-08-15): the conversion mechanism (`scale_lame`/
    /// `scale_stress`) was already real and verified (2026-07-17 audit).
    /// `rho_kg_m3=1800` is not tied to one specific paper -- real soil bulk
    /// density is inherently composition/moisture-dependent, not a
    /// universal constant like Kleiber's law -- but it IS now verified to
    /// sit inside the real, published range for compacted/saturated loam-
    /// family soils: dry bulk density 1150-1820 kg/m3 across tested
    /// densities (Xu et al., triaxial compression on sandy loam), saturated
    /// remolded loess tested at 1500-1700 kg/m3 dry-basis (PMC9282495).
    /// 1800 sits at the dense end of that real range, appropriate for
    /// "saturated" (pore-filled, denser than dry) rather than an arbitrary
    /// guess. `e_pa`/`bulk_modulus_pa`/`nu` remain honestly undocumented
    /// against one specific measurement (the same literature shows Young's
    /// modulus varying strongly with moisture/density, no single citable
    /// number) -- left as disclosed, mechanism-sound, range-plausible
    /// engineering defaults, same convention as `FORAGING_RECOVERY_RATE`'s
    /// own documented precedent elsewhere in this engine.
    /// Sources: [Effects of Bulk Density and Moisture Content on Selected Mechanical Properties of Sandy Loam Soil](https://www.sciencedirect.com/science/article/abs/pii/S1537511002901030),
    /// [Experimental study on shear strength of saturated remolded loess](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9282495/).
    pub const fn saturated_loam_preset() -> Self {
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

    /// Consolidated clay -- stiffer shear, slow plastic creep.
    ///
    /// CONFIRMED (2026-08-15): `rho_kg_m3=2000` is above the general loose
    /// clay/fine-silt range, but is realistic for the stiff overconsolidated
    /// clay implied by this preset's name. Rouainia et al. describe London
    /// Clay explicitly as "very stiff and heavily overconsolidated" and use
    /// a measured/calculated bulk unit weight of 20 kN/m3 at Denmark Place;
    /// dividing by standard gravity gives ~2040 kg/m3. The British Geological
    /// Survey independently compiles London Clay bulk densities of
    /// 1.83-2.35 Mg/m3. Thus 2000 kg/m3 is directly inside a published range
    /// for the intended dense-clay regime, not an extrapolation from ordinary
    /// loose clay. `e_pa`/`bulk_modulus_pa`/`nu` remain undocumented against a
    /// specific measurement, same as `saturated_loam`.
    /// Sources: Rouainia et al., "A pressuremeter-based evaluation of structure
    /// in London Clay using a kinematic hardening constitutive model", Acta
    /// Geotechnica 15 (2020), doi:10.1007/s11440-020-00940-w; British Geological
    /// Survey, "Geology of London", Table 23d.
    pub const fn consolidated_clay_preset() -> Self {
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

    /// Cytoplasmic matrix -- very soft elastic, near-fluid, large yield surface.
    ///
    /// CONFIRMED (2026-08-15): real AFM (atomic force microscopy) cell-
    /// mechanics literature reports cell elastic modulus spanning ~100 Pa
    /// to 100 kPa, with the ~100 Pa end specifically attributed to the
    /// actin cortex at small deformations (PMC5377332, "On the
    /// determination of elastic moduli of cells by AFM based indentation").
    /// `e_pa=500` sits inside this real measured range, toward the soft
    /// end -- appropriate for this preset's own "near-fluid, large yield
    /// surface" framing (cytoplasm proper, not the stiffer cortex/membrane).
    /// `rho_kg_m3=1050` also matches real cytoplasm density (close to
    /// water's 1000 kg/m3, real cell biology convention).
    /// `bulk_modulus_pa`/`nu`/plasticity params remain undocumented against
    /// a specific measurement.
    /// Source: [On the determination of elastic moduli of cells by AFM based indentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5377332/).
    pub const fn cytoplasmic_preset() -> Self {
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
/// given spacing -- `rho_kg_m3 * (spacing * dx_meters)^2` for a 2D areal-density
/// particle. Implemented identically by every physical-property family so
/// `SpawnRegion::mass_from` can stay generic over which material is being spawned.
pub trait ParticleMass {
    fn particle_mass(&self, spacing: f32, config: &SimConfig) -> f32;
}

// ── Internal bridging structs (pub(super) -- not part of LP API) ──────────────
//
// These carry the exact parameters that each material impl's `from_physical` needs.
// They are constructed inside `.material()` dispatch -- callers never see them.

/// Real-unit properties for `DruckerPragerMaterial`/`MuIRheologyMaterial`.
///
/// `pub` for the same reason as `BrittleProps`/`BinghamProps`: the dispatch
/// enum (`Elastoplastic::material`) returns a type-erased `Box<dyn
/// MaterialModel>`, so a caller that needs the concrete type back --
/// `MuIRheologyMaterial`'s own `inertial_q` has no dispatch field yet, see
/// that struct's own doc -- has to build it directly.
#[derive(Debug, Clone, Copy)]
pub struct GranularProps {
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

/// Real-unit properties for `NaccMaterial` (Non-Associated Cam-Clay).
///
/// Real fix (2026-09-05): `NaccMaterial` was the one real material family
/// with NO `from_physical`/SI-conversion constructor at all -- its own
/// `from_young_modulus` doc disclosed this as a genuine open gap. `pub`
/// (not `pub(super)`) because this type is also constructed directly by
/// callers of `Elastoplastic::material` via `PlasticityModel::CamClay` (see
/// that variant's own doc) -- same visibility reason as `BrittleProps`/
/// `GranularProps`/`DuctileProps`, all `pub` for the same match-arm-internal
/// reason. Wired into the dispatch enum 2026-09-08 (was real, disclosed
/// follow-up work before that -- see this doc's own prior revision in git
/// history if the old rationale is needed).
#[derive(Debug, Clone, Copy)]
pub struct NaccProps {
    pub elastic: Elastic,
    /// Friction slope M -- see `NaccMaterial::friction`'s own doc for the
    /// real friction-angle relation. NOT an SI quantity, passed through
    /// unconverted (matches `NaccMaterial::from_young_modulus`'s own
    /// convention).
    pub friction: f32,
    /// Cohesion β (0.0 = no tensile strength). NOT an SI quantity, passed
    /// through unconverted.
    pub cohesion: f32,
    /// Compression index λ of the soil's own oedometer curve (slope of the
    /// normal compression line in e against ln p'). Dimensionless.
    pub compression_index: f32,
    /// Swelling index κ of the same curve (unload-reload slope), usually a
    /// third to a fifth of the compression index. Dimensionless.
    pub swelling_index: f32,
    /// Void ratio e at the reference state, so the hardening exponent is
    /// `(1 + e) / (compression_index - swelling_index)`.
    pub void_ratio: f32,
    /// Preconsolidation pressure in Pa, the largest mean effective stress
    /// the soil has carried before (oedometer test, Casagrande 1936). 0.0 =
    /// a soil that has never been loaded.
    pub preconsolidation_pa: f32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NewtonianFluid {
    pub rho_kg_m3: f32,
    pub eta_pa_s: f32,
    pub bulk_modulus_pa: f32,
}

/// Real-unit properties for `BinghamFluidMaterial`.
///
/// `pub` for the same reason as `BrittleProps`: `Fluid::material` returns a
/// `Box<dyn MaterialModel>`, so a caller that needs to set a field the SI
/// family does not carry (`optics`, `specific_heat_j_kg_k`, surface
/// tension) has no way to reach the concrete material through that route.
/// Building it directly -- `BinghamFluidMaterial::from_physical(&props,
/// &config)` -- keeps every rheological parameter in real pascals and still
/// hands back the concrete type.
#[derive(Debug, Clone, Copy)]
pub struct BinghamProps {
    pub rho_kg_m3: f32,
    pub eta_pa_s: f32,
    pub bulk_modulus_pa: f32,
    pub yield_stress_pa: f32,
    /// Storage modulus G' `[Pa]` below the yield point. `0.0` keeps the
    /// classical purely-viscous Bingham fluid, which cannot hold a shape at
    /// rest; a positive value selects the elastoviscoplastic form that can.
    /// See `BinghamFluidMaterial::shear_modulus`.
    pub shear_modulus_pa: f32,
    /// Gauge pressure `[Pa]` at which this fluid cavitates and stops
    /// carrying tension. Negative. The Tait law this material uses is a
    /// gauge law, zero at rest density, so a particle above rest volume
    /// asks for a negative pressure; this is how far down that request is
    /// honoured before the fluid is taken to have opened a cavity instead.
    ///
    /// It is a coefficient of THIS fluid, not a number borrowed from a pure
    /// liquid: `BinghamProps::cavitation_pressure_from_nucleus` builds it
    /// from two independent published figures rather than stating it. One of
    /// those, the nucleus radius, is a class boundary rather than a
    /// measurement, and that is said where it is defined.
    ///
    /// Leaving it at 0.0 is a real physical statement and not a neutral
    /// default. It says the fluid carries no tension at all, so expansion
    /// meets no restoring force while compression meets the full one, and
    /// any symmetric noise in the divergence ratchets volume upward. That
    /// was measured: every expanded particle in every slab of
    /// `tests/scratch_thin_layer_volume_drift.rs` had its pressure deleted.
    pub cavitation_pressure_pa: f32,
}

impl BinghamProps {
    /// Surface tension `[N/m]` measured ON yield-stress fluids, not assumed
    /// from the liquid they are built in: Mohammadigoushki and Shoele,
    /// "Cavitation Rheology of Model Yield Stress Fluids Based on Carbopol",
    /// Langmuir 39(22), 7672-7683, 2023 (arXiv 2304.03187) measure
    /// 70 +/- 3 mN/m by needle-induced cavitation across Carbopol gels, and
    /// find it
    /// INDEPENDENT of the rheology over yield stresses of 0.5 to 120 Pa.
    /// That independence is why one value can serve a whole family of
    /// yield-stress fluids, and why a demo may share it across columns that
    /// differ only in yield stress.
    pub const YIELD_STRESS_FLUID_SURFACE_TENSION_N_M: f32 = 0.070;

    /// Radius `[m]` of the largest gas nucleus an ordinarily mixed paste
    /// carries. The Federal Highway Administration's petrographic manual
    /// (FHWA-HRT-04-150, "Petrographic Methods of Examining Hardened
    /// Concrete", July 2006, chapter 6) describes entrained voids as
    /// "spherical voids larger than the capillaries, but less than 1 mm on
    /// the lapped surface", anything above 1 mm being classed as entrapped
    /// instead.
    ///
    /// Read that for what it is: 1 mm is the boundary of a CLASS in a
    /// petrographic manual for HARDENED concrete, not the largest bubble
    /// anyone measured in a fresh paste. Entrapped voids above 1 mm exist
    /// too and would set a shallower threshold still. So the radius here is
    /// a declared modelling choice standing on a published class boundary,
    /// not a measurement, and it is the least defensible number in this
    /// file.
    ///
    /// What makes it usable anyway is that it barely matters:
    /// `tests/scratch_thin_layer_volume_drift.rs` sweeps this floor and
    /// finds the slab behaves much the same anywhere from -280 to -2800 Pa,
    /// a factor of ten.
    pub const ENTRAINED_AIR_NUCLEUS_RADIUS_M: f32 = 500.0e-6;

    /// Cavitation pressure `[Pa gauge, negative]` from the nucleus term of
    /// the needle-induced-cavitation relation `P_c = 5E/6 + 2*gamma/R`
    /// (Mohammadigoushki and Shoele, above, their Eq. 3): a cavity of
    /// radius `R` runs away once the pressure difference across its own
    /// surface tension is exceeded.
    ///
    /// One extrapolation, stated rather than hidden. That paper measures a
    /// cavity INJECTED at a needle tip, so its `R` is the needle's own
    /// inner radius, swept from 76 to 850 micrometres. Applying the same
    /// relation to a gas nucleus already sitting in the fluid is the same
    /// Laplace physics with a different cavity, and it is not what the
    /// experiment tested. The radius used here does at least sit inside the
    /// window they measured over.
    ///
    /// The elastic term `5E/6` is deliberately NOT included. It would make
    /// the floor depend on the fluid's own stiffness, so two fluids that
    /// differ only in yield stress would no longer share a value, and it
    /// deepens the floor, which is the permissive direction. Leaving it out
    /// keeps the shallower, more conservative threshold and keeps the
    /// coefficient a property of the nucleus rather than of the rheology.
    /// Adding it back is the natural refinement, and it is measurable.
    ///
    /// This relation also reproduces the figure the Newtonian twin already
    /// ships: `NewtonianFluidMaterial::from_physical` uses -100,000 Pa for
    /// practical dissolved-gas cavitation, and `2*gamma/R` returns that at
    /// a nucleus radius of 1.4 micrometres. The two are the same
    /// physics at two nucleus sizes, not two conventions: a clean liquid
    /// carries only sub-micron nuclei, an ordinarily mixed paste carries
    /// bubbles two hundred times larger, and cavitates that much sooner.
    pub fn cavitation_pressure_from_nucleus(
        surface_tension_n_m: f32,
        nucleus_radius_m: f32,
    ) -> f32 {
        -2.0 * surface_tension_n_m / nucleus_radius_m.max(f32::MIN_POSITIVE)
    }

    /// The cavitation pressure of an ordinarily mixed, air-entrained paste,
    /// from this type's own two constants. About -280 Pa. Half derived and
    /// half declared: the surface tension is measured on this fluid family,
    /// the nucleus radius is a class boundary read off a petrographic
    /// manual. See `ENTRAINED_AIR_NUCLEUS_RADIUS_M`.
    pub fn air_entrained_cavitation_pressure() -> f32 {
        Self::cavitation_pressure_from_nucleus(
            Self::YIELD_STRESS_FLUID_SURFACE_TENSION_N_M,
            Self::ENTRAINED_AIR_NUCLEUS_RADIUS_M,
        )
    }
}

// ── Scaling helpers (pub(super) -- used by material impls) ─────────────────────
//
// Real fix (2026-09-05): this module's own doc calls itself "the entry point
// for all material construction," but these three helpers were still routed
// through the `dt^2`-polluted `lame_from_si_cfg`/`stress_from_si`/
// `visc_from_si` family (see `lame_from_si_physical`'s own doc for the
// measured 200x dt-dependence this causes) -- confirmed live in LP's own
// `materials.rs` comment (`CREATURE_ACTIVE_STRESS_FRACTION_OF_MU`'s doc):
// "scaled through `lame_from_si`) overpowered the actual elastic stiffness
// by orders of magnitude and blew up the simulation ... within ~15 steps."
// Every one of `Elastic`/`Elastoplastic`/`Viscoelastic`/`Fluid`/
// `FluidGranular`'s real material families went through this bug via
// `.material(&config)`, not just the raw `lame_from_si_cfg` call sites
// found and migrated one scene at a time elsewhere. Fixed at the actual
// entry point instead: all three now route through the dt-independent
// `_physical` conversions, which MUST move together, never mixed with the
// old family -- `stress_from_si_physical`/`visc_from_si_physical`'s own doc
// name the exact bug (RankineMaterial::ice, this session) that mixing them
// causes.

/// Scale SI stress (Pa) to grid units: `p_grid = p_SI / (ρ · dx²)`, the
/// dt-independent conversion (see this module's own migration note above).
#[inline]
pub(super) fn scale_stress(pa: f32, rho: f32, config: &SimConfig) -> f32 {
    config.stress_from_si_physical(pa, rho)
}

/// Scale SI viscosity (Pa·s) to grid units: `η_grid = η_SI / (ρ · dx²)`, the
/// dt-independent conversion (see this module's own migration note above).
#[inline]
pub(super) fn scale_visc(eta: f32, rho: f32, config: &SimConfig) -> f32 {
    config.visc_from_si_physical(eta, rho)
}

/// Scale SI Young's modulus to grid Lamé parameters, the dt-independent
/// conversion (see this module's own migration note above).
#[inline]
pub(super) fn scale_lame(e_pa: f32, nu: f32, rho: f32, config: &SimConfig) -> (f32, f32) {
    config.lame_from_si_physical_cfg(e_pa, nu, rho)
}

// ── Reference SI values used in unit tests below ─────────────────────────────
#[cfg(test)]
mod _ref {
    use super::*;

    // Elastic -- E [Pa], ν, ρ [kg/m³]
    pub const SOFT_ELASTIC: Elastic = Elastic {
        e_pa: 500.0,
        nu: 0.45,
        rho_kg_m3: 1000.0,
    };

    // Viscoelastic -- η [Pa·s]
    pub const SOFT_VISCOELASTIC: Viscoelastic = Viscoelastic {
        elastic: Elastic {
            e_pa: 50_000.0,
            nu: 0.45,
            rho_kg_m3: 1100.0,
        },
        eta_pa_s: 10.0,
    };

    // Granular -- φ=35°
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

    // Ductile -- σ_Y=30 kPa
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

    // Brittle -- σ_t=10 MPa
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

    // Fluid -- Newtonian (no yield)
    pub const LOW_VISCOSITY_FLUID: Fluid = Fluid {
        rho_kg_m3: 1000.0,
        eta_pa_s: 0.001,
        bulk_modulus_pa: 2.2e9,
        yield_stress_pa: None,
    };

    // Fluid -- Bingham (yield=100 Pa)
    pub const VISCOPLASTIC_FLUID: Fluid = Fluid {
        rho_kg_m3: 1500.0,
        eta_pa_s: 0.5,
        bulk_modulus_pa: 1.5e9,
        yield_stress_pa: Some(100.0),
    };

    /// Verify all reference presets construct successfully -- catches API breakage.
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
        let _ = FluidGranular::saturated_loam_preset().material(&config);
    }
}

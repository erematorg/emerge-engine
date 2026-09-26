//! Common imports for LP / game consumers.
//!
//! ```rust,no_run
//! # extern crate emerge_engine as emerge;
//! use emerge::prelude::*;
//! ```
//!
//! Covers: solver, materials, force fields, boundary conditions, thermodynamics,
//! diagnostics, and runtime. Does not include experimental or GPU-backend types.

pub use crate::{
    AabbConfinementField,
    ActivationStatsPlugin,
    // Materials -- all seventeen (*Material types only)
    BinghamFluidMaterial,
    BinghamProps,
    // Queries + density field export
    BodyState,
    BoilingMixtureMaterial,
    // Boundary conditions
    BoundaryCondition,
    BrittleProps,
    BuoyancyField,
    CavitatingFluidMaterial,
    ChemotaxisField,
    CorotatedMaterial,
    CoulombField,
    DiagnosticsFrame,
    DiagnosticsPlugin,
    // Diagnostics
    DiagnosticsRegistry,
    DruckerPragerMaterial,
    // Physical property families + trait
    Elastic,
    Elastoplastic,
    // Force fields
    Field,
    // Runtime
    FixedStepConfig,
    FixedStepController,

    Fluid,
    FluidGranular,
    // Per-material stats + logging
    FrameLogger,
    FrictionBoundary,
    FromSI,
    GranularFluidMaterial,
    GranularProps,
    GravityWellField,
    // Directional/phase-gated grip boundaries (shipped with the ratchet
    // locomotion work).
    GripFrictionBoundary,
    HeightmapBoundary,

    IdealGasMaterial,
    IsothermalCavitatingFluidMaterial,
    // Real, kinematically-driven moving obstacle -- see its own doc for
    // the "no rigid bodies" scope-compliant design and the real two-way
    // momentum-exchange mechanism.
    KinematicCircleBoundary,
    // Creature locomotion controller
    Lnn,
    MaterialCountPlugin,
    MaterialHandle,
    MaterialModel,
    MaterialParams,
    MaterialStats,
    MuIRheologyMaterial,
    NBodyGravityField,
    NaccMaterial,
    NeoHookeanMaterial,
    NewtonianFluidMaterial,
    NoCompression,
    NoCompressionMaterial,
    Particle,
    ParticleGroup,
    ParticleMass,
    Particles,

    PlasticityModel,
    Pressurized,
    RadialConfinementField,
    RankineMaterial,
    RatchetFrictionBoundary,
    RollingPlugin,
    ScalarDiffusionConfig,
    ScalarDiffusionField,

    SimConfig,
    SimSnapshot,

    // Solver
    Simulation,
    SlipBoundary,
    SpawnRegion,
    SpawnShape,
    StabilityStatus,
    StabilityThresholds,
    StepTiming,
    StomakhinMaterial,
    // Thermodynamics
    ThermalConfig,
    ThermalDiffusion,
    ThermalStatsPlugin,
    UniformElectricField,

    Viscoelastic,
    ViscoelasticMaterial,
    VonMisesMaterial,
    WithLatentHeat,
    WithLatentHeatTable,
    WithPreStress,
    // Particle construction helpers
    build_particles,
    collect_snapshot,
    collect_snapshot_particles_only,
    evaluate_stability,
    gravity_to_grid,

    lame_from_si,
    // Real, dt-independent SI->grid conversion -- prefer this over
    // `lame_from_si` for any new scene (see its own doc for the measured
    // dt^2 bug in the older sibling above, kept only for scenes already
    // tuned against it).
    lame_from_si_physical,
    lame_from_young,
    log_frame_full,
    log_frame_gpu,
    per_material_stats,
    per_material_stats_of,
    rankine_damage_estimate,
};

// Math types -- re-exported so consumers don't need a separate glam dependency.
pub use glam::{IVec2, Mat2, Vec2};

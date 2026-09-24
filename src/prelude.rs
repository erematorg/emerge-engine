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
    // Materials — all fourteen (*Material types only)
    BinghamFluidMaterial,
    // Queries + density field export
    BodyState,
    // Boundary conditions
    BoundaryCondition,
    BrittleProps,
    BuoyancyField,
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
    GasMaterial,
    GranularFluidMaterial,
    GravityWellField,
    // Directional/phase-gated grip boundaries (shipped with the ratchet
    // locomotion work).
    GripFrictionBoundary,
    HeightmapBoundary,

    // Kinematically-driven moving obstacle (position/velocity set live, no
    // rigid-body dynamics of its own -- see feedback_engine_scope memory)
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
    NoSlipBoundary,

    Particle,
    ParticleGroup,
    ParticleMass,
    Particles,

    PlasticityModel,
    PredictiveBoundary,
    Pressurized,
    RadialConfinementField,
    RadianceField,
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
    WithPreStress,
    // Particle construction helpers
    build_particles,
    collect_snapshot,
    collect_snapshot_particles_only,
    evaluate_stability,
    gravity_to_grid,

    lame_from_si,
    lame_from_young,
    log_frame,
    log_frame_full,
    log_frame_gpu,
    per_material_stats,
    per_material_stats_of,
    rankine_damage_estimate,
};

// Math types — re-exported so consumers don't need a separate glam dependency.
pub use glam::{IVec2, Mat2, Vec2};

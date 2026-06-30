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
    // Materials — all twelve (*Material types only)
    BinghamFluidMaterial,
    // Queries + density field export
    BodyState,
    // Boundary conditions
    BoundaryCondition,
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
    FrictionBoundary,
    FromSI,
    GranularFluidMaterial,
    GravityWellField,
    HeightmapBoundary,

    // Creature locomotion controller
    Lnn,
    MaterialCountPlugin,
    MaterialHandle,
    MaterialModel,
    MaterialParams,
    MuIRheologyMaterial,
    NBodyGravityField,
    NaccMaterial,
    NeoHookeanMaterial,
    NewtonianFluidMaterial,
    Particle,
    ParticleGroup,
    ParticleMass,
    Particles,

    PlasticityModel,
    PredictiveBoundary,
    RadialConfinementField,
    RankineMaterial,
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
    // Particle construction helpers
    build_particles,
    collect_snapshot,
    gravity_to_grid,

    lame_from_si,
    lame_from_young,
};

// Math types — re-exported so consumers don't need a separate glam dependency.
pub use glam::{IVec2, Mat2, Vec2};

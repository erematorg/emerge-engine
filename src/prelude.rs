//! Common imports for LP / game consumers.
//!
//! ```rust,no_run
//! use emerge::prelude::*;
//! ```
//!
//! Covers: solver, materials, force fields, boundary conditions, thermodynamics,
//! diagnostics, and runtime. Does not include experimental or GPU-backend types.

pub use crate::{
    AabbConfinementField,
    ActivationStatsPlugin,
    // Materials — all eleven
    BinghamFluidMaterial,
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
    // Runtime
    FixedStepConfig,
    FixedStepController,

    // Force fields
    ForceField,
    FrictionBoundary,

    GravityWellField,
    // Creature locomotion controller
    Lnn,
    MaterialCountPlugin,
    MaterialHandle,
    MaterialModel,
    MaterialParams,
    // Queries + density field export
    MaterialState,
    MpmSnapshot,

    // Solver
    MpmSolver,
    NBodyGravityField,
    NaccMaterial,
    NeoHookeanMaterial,
    NewtonianFluidMaterial,
    Particle,
    ParticleGroup,
    Particles,

    PredictiveBoundary,
    RadialConfinementField,
    RankineMaterial,
    RollingPlugin,
    SandMaterial,
    SandMuIMaterial,
    ScalarDiffusionConfig,
    ScalarDiffusionField,

    SlipBoundary,
    SnowMaterial,
    SolverConfig,
    SpawnConfig,
    SpawnShape,
    // Thermodynamics
    ThermalConfig,
    ThermalDiffusion,
    ThermalStatsPlugin,
    UniformElectricField,

    ViscoelasticMaterial,
    VonMisesMaterial,
    // Particle construction helpers
    build_particles,
    collect_mpm_snapshot,
    compute_density_grid,

    estimate_particle_volumes,

    gravity_to_grid,

    lame_from_si,
    lame_from_young,
};

// Math types — re-exported so consumers don't need a separate glam dependency.
pub use glam::{IVec2, Mat2, Vec2};

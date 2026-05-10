//! Common imports for LP / game consumers.
//!
//! ```rust,no_run
//! use emerge::prelude::*;
//! ```
//!
//! Covers: solver, materials, force fields, boundary conditions, thermodynamics,
//! diagnostics, and runtime. Does not include experimental or GPU-backend types.

pub use crate::{
    // Solver
    MpmSolver, SolverConfig, SpawnConfig, SpawnShape,
    MaterialHandle, ParticleGroup,
    Particle, Particles,

    // Materials — all eleven
    BinghamFluidMaterial, CorotatedMaterial, NaccMaterial, NeoHookeanMaterial, NewtonianFluidMaterial,
    RankineMaterial, SandMaterial, SandMuIMaterial, SnowMaterial, ViscoelasticMaterial,
    VonMisesMaterial, MaterialModel, MaterialParams, lame_from_young, lame_from_si, gravity_to_grid,

    // Boundary conditions
    BoundaryCondition, SlipBoundary, PredictiveBoundary, FrictionBoundary,

    // Force fields
    ForceField,
    AabbConfinementField, BuoyancyField, ChemotaxisField, CoulombField, GravityWellField,
    NBodyGravityField, RadialConfinementField, UniformElectricField,

    // Thermodynamics
    ThermalConfig, ThermalDiffusion,
    ScalarDiffusionConfig, ScalarDiffusionField,

    // Diagnostics
    DiagnosticsRegistry, DiagnosticsFrame, DiagnosticsPlugin,
    ActivationStatsPlugin, ThermalStatsPlugin, MaterialCountPlugin, RollingPlugin,
    collect_mpm_snapshot, MpmSnapshot,

    // Runtime
    FixedStepConfig, FixedStepController,

    // Queries + density field export
    MaterialState, compute_density_grid,

    // Particle construction helpers
    build_particles, estimate_particle_volumes,

    // Creature locomotion controller
    Lnn,
};

// Math types — re-exported so consumers don't need a separate glam dependency.
pub use glam::{IVec2, Mat2, Vec2};

// ─────────────────────────────────────────────────────────────────────────────
// emerge — MPM physics engine for Life's Progress
//
// Module layout:
//
//   Core physics (always compiled, stable API)
//   ├── solver/          MpmSolver, SolverConfig, SpawnConfig, query, density, cutoff
//   ├── particle         Particle struct
//   ├── grid/            Grid, Cell, kernel (quadratic weights)
//   ├── materials/       MaterialModel trait, constitutive models, MaterialRegistry
//   ├── boundary         BoundaryCondition + impls
//   ├── transfer         P2G / G2P transfer kernels
//   ├── fields/          ForceField trait + impls (gravity, Coulomb, EM, confinement)
//   ├── control/         Lnn (Liquid Time-constant Network locomotion controller)
//   ├── thermodynamics/  ThermalDiffusion · ScalarDiffusionField
//   ├── diagnostics/     health monitoring · plugin-based stats collection
//   └── runtime/         FixedStepController
//
//   Compute backends (feature-gated)
//   ├── gpu/             GpuSolver + WGSL shaders        [feature = "gpu"]
//   └── render/          Instanced particle debug draw    [feature = "render"]
//
//   Extended physics (experimental, not part of LP-stable API)
//   ├── acoustics/       WaveEquation2D                  [feature = "experimental"]
//   └── electromagnetics/ EM field math                  [feature = "experimental"]
// ─────────────────────────────────────────────────────────────────────────────

// ── Core ─────────────────────────────────────────────────────────────────────
pub mod boundary;
pub mod control;
pub mod diagnostics;
pub mod fields;
pub mod grid;
pub mod materials;
pub mod particle;
pub mod runtime;
pub mod solver;
pub mod thermodynamics;
pub mod transfer;

// ── Compute backends ─────────────────────────────────────────────────────────
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "render")]
pub mod render;

// ── Extended physics ─────────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod acoustics;
#[cfg(feature = "experimental")]
pub mod electromagnetics;

// ── Prelude — common imports for LP/game consumers ───────────────────────────
pub mod prelude;

// ── Flat re-exports ───────────────────────────────────────────────────────────
// `use emerge::MpmSolver` instead of `use emerge::solver::MpmSolver`.

// Solver core
pub use solver::MpmSolver;
pub use solver::config::{SolverConfig, SpawnConfig, SpawnShape};
pub use solver::handle::{MaterialHandle, ParticleGroup};
pub use particle::{Particle, Particles};
pub use grid::{Cell, Grid};

// Materials
pub use materials::{
    BinghamFluidMaterial, ConstitutiveModel, MaterialModel, MaterialParams, MaterialRegistry,
    MAX_MATERIAL_SLOTS, lame_from_young, lame_from_si, gravity_to_grid,
    CorotatedMaterial, NaccMaterial, NeoHookeanMaterial, NewtonianFluidMaterial,
    RankineMaterial, SandMaterial, SandMuIMaterial, SnowMaterial,
    ViscoelasticMaterial, VonMisesMaterial,
};

// Boundary conditions
pub use boundary::{BoundaryCondition, FrictionBoundary, PredictiveBoundary, SlipBoundary};

// Force fields
pub use fields::ForceField;
pub use fields::{
    AabbConfinementField, BuoyancyField, ChemotaxisField, CoulombField, GravityWellField,
    NBodyGravityField, RadialConfinementField, UniformElectricField,
};

// State queries + density export for rendering
pub use solver::query::MaterialState;
pub use solver::density::compute_density_grid;
pub use control::Lnn;

/// Build a `Vec<Particle>` from a `SpawnConfig` — the primary way to construct
/// initial particle regions for `GpuSolver::new` or to merge multiple regions.
///
/// Respects `SpawnConfig::shape` (box or disk), jitter, and material assignment.
/// For physically accurate initial volumes call with `spawn.precompute_volumes()`
/// or follow up with `estimate_particle_volumes`.
///
/// LP pattern:
/// ```rust,no_run
/// # use emerge::{SolverConfig, SpawnConfig, build_particles, NewtonianFluidMaterial};
/// # use glam::Vec2;
/// # let config = SolverConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
/// let mut particles = build_particles(&config,
///     SpawnConfig::for_solver(&config).at(Vec2::new(20.0, 32.0)).disk(10.0).spacing(0.5).material(0));
/// particles.extend(build_particles(&config,
///     SpawnConfig::for_solver(&config).at(Vec2::new(44.0, 32.0)).disk(10.0).spacing(0.5).material(1)));
/// ```
pub fn build_particles(config: &SolverConfig, spawn: SpawnConfig) -> Vec<Particle> {
    use crate::solver::LcgRng;
    let mut rng = LcgRng::new(spawn.rng_seed);
    let mut particles = crate::solver::initialize_particles(config, spawn, &mut rng);
    if spawn.precompute_initial_volumes {
        estimate_particle_volumes(&mut particles, config.grid_res);
    }
    particles
}

/// Estimate initial particle volumes from P2G density.
///
/// Use when building particles manually for `GpuSolver::new` and you need the same
/// physically accurate density that `SpawnConfig::precompute_volumes()` gives you
/// inside `MpmSolver::spawn_region`. Without it, initial particle density is geometric
/// (`mass / spacing²`) which can cause a pressure spike on the first substep.
pub fn estimate_particle_volumes(particles: &mut Vec<Particle>, grid_res: usize) {
    use crate::solver::density::estimate_initial_particle_volumes;
    let mut soa = Particles::from(particles.clone());
    let mut grid = Grid::new(grid_res);
    estimate_initial_particle_volumes(&mut soa, &mut grid);
    for (i, p) in particles.iter_mut().enumerate() {
        p.density = soa.density[i];
        p.volume = soa.volume[i];
        p.initial_volume = soa.initial_volume[i];
    }
}

// Thermodynamics
pub use thermodynamics::{ThermalConfig, ThermalDiffusion, ScalarDiffusionConfig, ScalarDiffusionField};

// Diagnostics + plugin system
pub use diagnostics::{
    // Snapshot + health
    collect_mpm_snapshot, MpmSnapshot,
    MpmHealthThresholds, MpmHealthStatus, evaluate_mpm_health,
    // Per-material stats + logging
    MaterialStats, per_material_stats, per_material_stats_of, log_frame, log_frame_full, log_frame_gpu,
    // Plugin infrastructure
    DiagnosticsPlugin, DiagnosticsRegistry, DiagnosticsFrame,
    ActivationStatsPlugin, ThermalStatsPlugin, MaterialCountPlugin, RollingPlugin,
};

// Runtime
pub use runtime::{FixedStepConfig, FixedStepController};

// GPU backend
#[cfg(feature = "gpu")]
pub use gpu::{GpuSolver, GpuForceFieldEntry, GpuForceFieldsParams, MAX_FORCE_FIELDS, field_type};

// Render backend
#[cfg(feature = "render")]
pub use render::{ColorMode, DebugRenderer};

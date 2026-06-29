// ─────────────────────────────────────────────────────────────────────────────
// emerge — MPM physics engine for Life's Progress
//
// Module layout:
//
//   Core physics (always compiled, stable API)
//   ├── solver/          Simulation, SimConfig, SpawnRegion, query, density, cutoff
//   ├── particle         Particle struct
//   ├── grid/            Grid, Cell, kernel (quadratic weights)
//   ├── materials/       MaterialModel trait, constitutive models, MaterialRegistry
//   ├── boundary         BoundaryCondition + impls
//   ├── transfer         P2G / G2P transfer kernels
//   ├── fields/          Field trait + impls (gravity, Coulomb, EM, confinement)
//   ├── control/         Lnn (Liquid Time-constant Network locomotion controller)
//   ├── thermodynamics/  ThermalDiffusion · ScalarDiffusionField
//   ├── diagnostics/     health monitoring · plugin-based stats collection
//   └── runtime/         FixedStepController
//
//   Compute backends (feature-gated)
//   ├── gpu/             GpuSimulation + WGSL shaders        [feature = "gpu"]
//   └── render/          Instanced particle debug draw    [feature = "render"]
//
//   Extended physics (experimental, not part of LP-stable API)
//   ├── acoustics/       WaveEquation2D                  [feature = "experimental"]
//   ├── electromagnetics/ EM field math                  [feature = "experimental"]
//   └── measures/        O(N) entropy (spatial · kinetic · phase) · local MI · KL divergence
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
#[cfg(feature = "experimental")]
pub mod measures;

// ── Prelude — common imports for LP/game consumers ───────────────────────────
pub mod prelude;

// ── Flat re-exports ───────────────────────────────────────────────────────────
// `use emerge::Simulation` instead of `use emerge::solver::Simulation`.

// Solver core
pub use grid::{Cell, Grid};
pub use particle::{Particle, Particles};
pub use solver::Simulation;
pub use solver::config::{SimConfig, SpawnRegion, SpawnShape};
pub use solver::handle::{MaterialHandle, ParticleGroup};

// Materials
pub use materials::{
    BinghamFluidMaterial, ConstitutiveModel, CorotatedMaterial, DruckerPragerMaterial, Elastic,
    Elastoplastic, Fluid, FluidGranular, FromSI, GranularFluidMaterial, MAX_MATERIAL_SLOTS,
    MaterialModel, MaterialParams, MaterialRegistry, MuIRheologyMaterial, NaccMaterial,
    NeoHookeanMaterial, NewtonianFluidMaterial, ParticleMass, PlasticityModel, RankineMaterial,
    StomakhinMaterial, Viscoelastic, ViscoelasticMaterial, VonMisesMaterial, gravity_to_grid,
    lame_from_si, lame_from_young,
};

// Boundary conditions
pub use boundary::{
    BoundaryCondition, FrictionBoundary, HeightmapBoundary, PredictiveBoundary, SlipBoundary,
};

// Force fields
pub use fields::Field;
pub use fields::{
    AabbConfinementField, BuoyancyField, ChemotaxisField, CoulombField, GravityWellField,
    NBodyGravityField, RadialConfinementField, UniformElectricField,
};

// State queries + density export for rendering
pub use control::Lnn;
pub use solver::density::compute_density_grid;
pub use solver::query::BodyState;

/// Build a `Vec<Particle>` from a `SpawnRegion` — the primary way to construct
/// initial particle regions for `GpuSimulation::new` or to merge multiple regions.
///
/// Respects `SpawnRegion::shape` (box or disk), jitter, and material assignment.
/// For physically accurate initial volumes call with `spawn.precompute_volumes()`
/// or follow up with `estimate_particle_volumes`.
///
/// LP pattern:
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{SimConfig, SpawnRegion, build_particles, NewtonianFluidMaterial};
/// # use glam::Vec2;
/// # let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
/// let mut particles = build_particles(&config,
///     SpawnRegion::for_sim(&config).at(Vec2::new(20.0, 32.0)).disk(10.0).spacing(0.5).material(0));
/// particles.extend(build_particles(&config,
///     SpawnRegion::for_sim(&config).at(Vec2::new(44.0, 32.0)).disk(10.0).spacing(0.5).material(1)));
/// ```
pub fn build_particles(config: &SimConfig, spawn: SpawnRegion) -> Vec<Particle> {
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
/// Use when building particles manually for `GpuSimulation::new` and you need the same
/// physically accurate density that `SpawnRegion::precompute_volumes()` gives you
/// inside `Simulation::spawn_region`. Without it, initial particle density is geometric
/// (`mass / spacing²`) which can cause a pressure spike on the first substep.
pub fn estimate_particle_volumes(particles: &mut Vec<Particle>, grid_res: usize) {
    use crate::solver::density::estimate_particle_volumes as density_estimate;
    let mut soa = Particles::from(std::mem::take(particles));
    let mut grid = Grid::new(grid_res);
    let n = soa.len();
    density_estimate(&mut soa, &mut grid, n, true);
    *particles = soa.to_vec();
}

// Thermodynamics
pub use thermodynamics::{
    ScalarDiffusionConfig, ScalarDiffusionField, ThermalConfig, ThermalDiffusion,
};

// Diagnostics + plugin system
pub use diagnostics::{
    ActivationStatsPlugin,
    DiagnosticsFrame,
    // Plugin infrastructure
    DiagnosticsPlugin,
    DiagnosticsRegistry,
    // Snapshot + health
    FrameLogger,
    MaterialCountPlugin,
    // Per-material stats + logging
    MaterialStats,
    RollingPlugin,
    SimSnapshot,
    StabilityStatus,
    StabilityThresholds,
    StepTiming,
    ThermalStatsPlugin,
    collect_snapshot,
    collect_snapshot_particles_only,
    evaluate_stability,
    log_frame,
    log_frame_full,
    log_frame_gpu,
    per_material_stats,
    per_material_stats_of,
};

// Runtime
pub use runtime::{FixedStepConfig, FixedStepController};

// GPU backend
#[cfg(feature = "gpu")]
pub use gpu::{GpuFieldEntry, GpuFieldsParams, GpuSimulation, MAX_FORCE_FIELDS, field_type};

// Render backend
#[cfg(feature = "render")]
pub use render::{ColorMode, Renderer};

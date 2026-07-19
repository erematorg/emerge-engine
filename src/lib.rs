// ─────────────────────────────────────────────────────────────────────────────
// emerge — MPM physics engine for Life's Progress
//
// Module layout:
//
//   Core physics (always compiled, stable API)
//   ├── spacetime/       Spacetime domain: solver (Simulation, SimConfig,
//   │                    SpawnRegion, query, density, cutoff), grid (Grid, Cell,
//   │                    kernel), transfer (P2G/G2P transfer kernels)
//   ├── matter/          Matter domain: particle (Particle struct), materials/
//   │                    (MaterialModel trait, constitutive models, MaterialRegistry)
//   ├── forces/          Forces domain: boundary (BoundaryCondition + impls),
//   │                    fields (Field trait + impls: gravity, Coulomb, EM, confinement)
//   ├── information/     Information domain: control (Lnn), measures (entropy/MI) [experimental]
//   ├── energy/          Energy domain: thermodynamics (ThermalDiffusion,
//   │                    ScalarDiffusionField), acoustics (WaveEquation2D) [experimental]
//   └── runtime/         FixedStepController
//
//   Systems domain -- pure orchestration, no IRL counterpart (feature-gated where relevant)
//   ├── systems::diagnostics  health monitoring, plugin-based stats collection
//   ├── systems::gpu          GpuSimulation + WGSL shaders        [feature = "gpu"]
//   └── systems::render       Instanced particle debug draw    [feature = "render"]
//
//   Extended physics (experimental, not part of LP-stable API)
//   ├── forces::electromagnetics  E/B field-query math       [feature = "experimental"]
//   └── energy::electromagnetics  EM waves + optical MaterialProperties [feature = "experimental"]
// ─────────────────────────────────────────────────────────────────────────────

// ── Core ─────────────────────────────────────────────────────────────────────
pub mod energy;
pub mod forces;
pub mod information;
pub mod matter;
pub mod runtime;
pub mod spacetime;
pub mod systems;

// Domain folders re-export their contents at the old crate-root paths --
// every existing internal `crate::x::` path and every LP `emerge::x::` path
// keeps resolving unchanged. See each domain's `mod.rs` doc for why.
#[cfg(feature = "experimental")]
pub use energy::acoustics;
pub use energy::thermodynamics;
pub use forces::boundary;
pub use forces::fields;
pub use information::control;
#[cfg(feature = "experimental")]
pub use information::measures;
pub use matter::materials;
pub use matter::particle;
pub use spacetime::diff;
pub use spacetime::grid;
pub use spacetime::solver;
pub use spacetime::transfer;
pub use systems::diagnostics;
#[cfg(feature = "gpu")]
pub use systems::gpu;
#[cfg(feature = "render")]
pub use systems::render;

// ── Prelude — common imports for LP/game consumers ───────────────────────────
pub mod prelude;

// ── Flat re-exports ───────────────────────────────────────────────────────────
// `use emerge::Simulation` instead of `use emerge::solver::Simulation`.

// Solver core
pub use grid::{Cell, DirectionalContactGrip, Grid};
pub use particle::{Particle, Particles};
pub use solver::Simulation;
pub use solver::config::{SimConfig, SpawnRegion, SpawnShape};
pub use solver::handle::{MaterialHandle, ParticleGroup};

// Materials
pub use materials::{
    BinghamFluidMaterial, BrittleProps, ConstitutiveModel, CorotatedMaterial,
    DruckerPragerMaterial, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI,
    GranularFluidMaterial, MAX_MATERIAL_SLOTS, MaterialModel, MaterialParams, MaterialRegistry,
    MixturePhase, MuIRheologyMaterial, NaccMaterial, NeoHookeanMaterial, NewtonianFluidMaterial,
    ParticleMass, PlasticityModel, RankineMaterial, StomakhinMaterial, Viscoelastic,
    ViscoelasticMaterial, VonMisesMaterial, WithLatentHeat, WithMixturePhase, gravity_to_grid,
    lame_from_si, lame_from_young, rankine_damage_estimate,
};

// Boundary conditions
pub use boundary::{
    BoundaryCondition, FrictionBoundary, GripFrictionBoundary, HeightmapBoundary,
    PredictiveBoundary, RatchetFrictionBoundary, SlipBoundary,
};

// Force fields
pub use fields::Field;
pub use fields::{
    AabbConfinementField, BuoyancyField, ChemotaxisField, CoulombField, GravityWellField,
    LinearDragField, NBodyGravityField, RadialConfinementField, SpatialDragField,
    UniformElectricField,
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
    ScalarDiffusionConfig, ScalarDiffusionField, ThermalConfig, ThermalDiffusion, saturating_uptake,
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
pub use render::{ColorMode, GridVolumeSource, Renderer};

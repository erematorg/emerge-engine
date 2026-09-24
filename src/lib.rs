// ─────────────────────────────────────────────────────────────────────────────
// emerge — MPM physics engine for Life's Progress
//
// Module layout:
//
//   Core physics (always compiled, stable API)
//   ├── spacetime/       Spacetime domain: solver (Simulation, SimConfig,
//   │                    SpawnRegion, body_state, density), grid (Grid, Cell,
//   │                    kernel), transfer (P2G/G2P transfer kernels), rod
//   │                    (1D discrete elastic rod sub-solver, grid-coupled),
//   │                    grains (DEM grain dynamics: population/coupling/oracle)
//   ├── matter/          Matter domain: particle (Particle, Grain, RodPoints),
//   │                    materials/ (MaterialModel trait, 13 constitutive models,
//   │                    MaterialRegistry; granular/ groups the sand research thread)
//   ├── forces/          Forces domain: boundary (BoundaryCondition + impls,
//   │                    friction/ groups the Coulomb-friction family), fields
//   │                    (Field trait + impls: gravity, Coulomb, EM, confinement, cutoff)
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
#[cfg(feature = "experimental")]
pub use energy::orbital;
pub use energy::thermodynamics;
pub use forces::boundary;
pub use forces::fields;
pub use information::control;
#[cfg(feature = "experimental")]
pub use information::measures;
pub use matter::materials;
pub use matter::particle;
pub use spacetime::diff;
pub use spacetime::grains;
pub use spacetime::grid;
pub use spacetime::rod;
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
    DruckerPragerMaterial, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI, GasMaterial,
    GranularFluidMaterial, MAX_MATERIAL_SLOTS, MaterialModel, MaterialParams, MaterialRegistry,
    MixturePhase, MuIRheologyMaterial, NaccMaterial, NeoHookeanMaterial, NewtonianFluidMaterial,
    NoCompression, NoCompressionMaterial, ParticleMass, PlasticityModel, Pressurized,
    RankineMaterial, StomakhinMaterial, Viscoelastic, ViscoelasticMaterial, VonMisesMaterial,
    WithLatentHeat, WithMixturePhase, WithPreStress, gravity_to_grid, lame_from_si,
    lame_from_young, rankine_damage_estimate,
};

// Boundary conditions
pub use boundary::{
    BoundaryCondition, FrictionBoundary, GripFrictionBoundary, HeightmapBoundary,
    KinematicCircleBoundary, NoSlipBoundary, PredictiveBoundary, RatchetFrictionBoundary,
    SlipBoundary,
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
pub use solver::body_state::BodyState;
pub use solver::density::compute_density_grid;

/// Build a `Vec<Particle>` from a `SpawnRegion` — the primary way to construct
/// initial particle regions for `GpuSimulation::new` or to merge multiple regions.
///
/// Respects `SpawnRegion::shape` (box or disk), jitter, and material assignment.
/// For solid/plastic materials, call with `spawn.precompute_volumes()` or
/// follow up with `estimate_particle_volumes` when a kernel-measured initial
/// volume is required. `GpuSimulation::new` subsequently runs each registered
/// material's initializer; strict WC-MPM liquids establish `V0=m/rho0` there
/// and must not use a kernel-density estimate as EOS state.
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

/// Estimate initial particle volumes from a P2G density measurement.
///
/// Use for solid/plastic particle sets whose material model consumes that
/// measurement. This low-level helper has no material registry and therefore
/// must not be applied to strict WC-MPM liquid particles: their EOS state is
/// initialized from conserved mass and rest density by `GpuSimulation::new`.
pub fn estimate_particle_volumes(particles: &mut Vec<Particle>, grid_res: usize) {
    use crate::solver::density::estimate_particle_volumes as density_estimate;
    let mut soa = Particles::from(std::mem::take(particles));
    let mut grid = Grid::new(grid_res);
    let n = soa.len();
    density_estimate(&mut soa, &mut grid, None, n, true);
    *particles = soa.to_vec();
}

// Thermodynamics
pub use thermodynamics::{
    GranularFluidityConfig, GranularFluidityField, RadianceField, ScalarDiffusionConfig,
    ScalarDiffusionField, ThermalConfig, ThermalDiffusion, irradiance_at_distance,
    saturating_uptake, stellar_luminosity_w,
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
    SiSnapshot,
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

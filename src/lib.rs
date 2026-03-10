pub mod diagnostics;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod runtime;
pub mod solver;
pub mod state;

// Flat re-exports — LP and other consumers can write `use emerge::MpmSolver`
// rather than digging into internal module paths.
pub use diagnostics::{collect_mpm_snapshot, MpmSnapshot};
pub use runtime::FixedStepController;
pub use solver::{
    BoundaryCondition, ConstitutiveModel, MaterialModel, MaterialParams, MaterialRegistry,
    MpmSolver, NeoHookeanMaterial, NewtonianFluidMaterial, PredictiveBoundary, SandMaterial, SlipBoundary,
    SolverConfig, SpawnConfig,
};
pub use state::grid::{Cell, Grid};
pub use state::particle::Particle;

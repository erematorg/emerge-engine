pub mod boundary;
pub mod config;
pub mod density;
pub mod material_registry;
pub mod materials;
mod solver;
pub(crate) mod svd;
pub mod transfer;

pub use boundary::{BoundaryCondition, PredictiveBoundary, SlipBoundary};
pub use config::{SolverConfig, SpawnConfig};
pub use material_registry::MaterialRegistry;
pub use materials::{ConstitutiveModel, CorotatedMaterial, MaterialModel, MaterialParams, NeoHookeanMaterial, NewtonianFluidMaterial, SnowMaterial};
pub use solver::MpmSolver;

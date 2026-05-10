use glam::Vec2;

use crate::particle::{Particle, Particles};

/// A body force that produces a per-particle acceleration.
///
/// Implement this trait to add custom external forces to the CPU solver:
/// non-uniform gravity, magnetic confinement, electric fields, player attractors, etc.
///
/// # Notes
/// - Applied after G2P gather, before state projection — i.e., as a velocity correction
///   `v += dt × acceleration(p)` per substep.
/// - Only available on `MpmSolver` (CPU). For `GpuSolver`, uniform body forces go into
///   `SolverConfig::gravity`; non-uniform GPU force fields are future work.
/// - Keep implementations cheap — called once per particle per substep.
pub trait ForceField: Send + Sync {
    /// Called once per substep before any `acceleration()` calls.
    ///
    /// Use to rebuild internal state that depends on the full particle set —
    /// e.g., a Barnes-Hut quadtree for N-body gravity.
    /// Default implementation does nothing (stateless fields need not override).
    #[allow(unused_variables)]
    fn prepare(&mut self, particles: &Particles) {}

    /// Return the acceleration (in grid-units/s²) applied to `particle` this substep.
    fn acceleration(&self, particle: &Particle) -> Vec2;
}

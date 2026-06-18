use glam::Vec2;

use crate::particle::Particles;

/// A body force that produces a per-particle acceleration.
///
/// Implement this trait to add custom external forces to the CPU solver:
/// non-uniform gravity, magnetic confinement, electric fields, player attractors, etc.
///
/// # Notes
/// - Applied after G2P gather, before state projection — i.e., as a velocity correction
///   `v += dt × acceleration(p)` per substep.
/// - Only available on `Simulation` (CPU). For `GpuSimulation`, uniform body forces go into
///   `SimConfig::gravity`; non-uniform GPU force fields are future work.
/// - Keep implementations cheap — called once per particle per substep.
pub trait Field: Send + Sync {
    /// Called once per substep before any `acceleration()` calls.
    ///
    /// Use to rebuild internal state that depends on the full particle set —
    /// e.g., a Barnes-Hut quadtree for N-body gravity.
    /// Default implementation does nothing (stateless fields need not override).
    fn prepare(&mut self, _particles: &Particles) {}

    /// Return the acceleration (in grid-units/s²) applied to particle `i` this substep.
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2;
}

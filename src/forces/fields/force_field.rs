use glam::Vec2;

use crate::particle::Particles;

/// A body force that produces a per-particle acceleration.
///
/// Implement this trait to add external forces to the CPU solver:
/// non-uniform gravity, electric and magnetic fields, a rotating frame, etc.
///
/// # Notes
/// - Applied after G2P gather, before state projection, per substep. With no
///   gyroscopic part this is `v += dt × acceleration(p)`; with one, the
///   solver applies the Boris rotation (`fields::gyroscopic`).
/// - Only available on `Simulation` (CPU). For `GpuSimulation`, uniform body forces go into
///   `SimConfig::gravity`; non-uniform GPU force fields are future work.
/// - Keep implementations cheap -- called once per particle per substep.
pub trait Field: Send + Sync {
    /// Called once per substep before any `acceleration()` calls.
    ///
    /// Use to rebuild internal state that depends on the full particle set --
    /// e.g., a Barnes-Hut quadtree for N-body gravity.
    /// Default implementation does nothing (stateless fields need not override).
    fn prepare(&mut self, _particles: &Particles) {}

    /// Return the acceleration (in grid-units/s²) applied to particle `i` this substep.
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2;

    /// Rate `omega` (rad/s) of a force that turns particle `i`'s velocity
    /// without doing work, `dv/dt = omega J v` with `J v = (-v_y, v_x)`: the
    /// magnetic Lorentz force, the Coriolis acceleration (see
    /// `fields::gyroscopic`). Default 0: the field acts only through
    /// `acceleration`.
    fn gyration_rate(&self, _particles: &Particles, _i: usize) -> f32 {
        0.0
    }
}

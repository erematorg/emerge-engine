use glam::Vec2;

use crate::particle::Grain;

/// A body force that produces a per-grain acceleration -- the `GrainField`
/// counterpart to `Field`, for a standalone `GrainPopulation` (`Vec<Grain>`,
/// see `spacetime::grains::population`) instead of the MPM solver's own
/// `Particles` SoA.
///
/// Kept as a SEPARATE trait rather than extending `Field` itself: `Grain`
/// and `Particles` are genuinely different storage shapes (an AoS
/// `Vec<Grain>` vs a columnar SoA), so one trait signature can't serve both
/// without an artificial adapter. A field that makes physical sense for
/// both bodies (e.g. `LinearDragField`, which only ever reads a body's own
/// velocity) implements BOTH traits against the SAME config struct -- one
/// set of tuned numbers, one citation, no duplicated formula.
///
/// # Notes
/// - Applied inside `GrainPopulation::step`, added on top of gravity and
///   contact forces -- same "velocity correction per substep" contract as
///   `Field`.
/// - No `prepare()` hook (unlike `Field`) and no material-mask convention
///   (unlike most `Field` impls): a `GrainPopulation` is a single, already-
///   homogeneous population. Add masking here only once a real scene needs
///   mixed-material grains -- don't build it speculatively.
pub trait GrainField: Send + Sync {
    /// Return the acceleration (in grid-units/s²) applied to this grain this substep.
    fn acceleration(&self, grain: &Grain) -> Vec2;
}

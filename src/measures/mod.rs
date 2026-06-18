/// Information-theoretic measures for particle systems — all O(N), real-time safe.
///
/// All estimators use the MPM grid or a single pass over particles.
/// No pairwise distance matrix. No O(N²).
///
/// References:
/// - Shannon 1948
/// - Gibbs entropy S = -k Σ pᵢ ln pᵢ (statistical mechanics)
/// - Maxwell-Boltzmann kinetic temperature
pub mod divergence;
pub mod spatial;

pub use divergence::KLDivergence;
pub use spatial::{kinetic_entropy, local_phase_mi, phase_entropy, spatial_entropy};

//! Discrete-element grain DYNAMICS -- real, cited rolling-resistance physics
//! (Cundall & Strack 1979 / Luding 2008 / Ai, Chen, Rotter & Ooi 2011) that
//! every rate-dependent mechanism already tried for sand's repose-angle
//! problem this session structurally cannot provide. Deliberately mirrors
//! how the main MPM system splits state from dynamics across domains, not
//! just `spacetime::rod`'s general precedent: `Grain` (pure per-point
//! kinematic state, same role `Particle` plays) lives in
//! `matter::particle::grain`; the contact force law (`GrainContactState`,
//! `ContactLawConfig`, `resolve_contact_pair`) lives in
//! `matter::materials::granular::grain_contact_law` alongside the other constitutive
//! models. Everything in THIS module -- `population` (the `GrainPopulation`
//! container plus its own integration step), `coupling` (grid
//! scatter/gather), `oracle` (packing-fraction-driven WHERE-are-grains-
//! needed decision, real precedent: Yue, Smith, Chen, Chantharayukhonthorn,
//! Kamrin & Grinspun 2018, "Hybrid Grains," ACM TOG 37(6)) -- is dynamics:
//! how grains move, couple to the shared grid, and where they're spawned,
//! not what a grain IS or how it collides. All real and implemented, not
//! scaffolding.

pub mod coupling;
pub mod implicit;
pub mod micro_rotation;
pub mod oracle;
pub mod population;
pub mod terrain_contact;

// Flattened re-export of the types a caller actually needs to construct and
// attach a grain population (`Simulation::with_grain_population` takes a
// `GrainPopulation` by value) -- same convention `spacetime::rod` already
// uses to make its own construction types reachable one level up instead of
// requiring the full `emerge::grains::population::GrainPopulation` path.
pub use population::{ContactModel, GrainPopulation};

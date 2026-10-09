//! Discrete-element grain dynamics with rolling resistance (Cundall & Strack
//! 1979 / Luding 2008 / Ai, Chen, Rotter & Ooi 2011), which the rate-dependent
//! continuum mechanisms cannot provide for sand's repose angle. State and
//! dynamics are split as for MPM particles: `Grain` (per-point kinematic
//! state, the role `Particle` plays) lives in `matter::particle::grain`, and
//! the contact force law (`GrainContactState`, `ContactLawConfig`,
//! `resolve_contact_pair`) in `matter::materials::granular::grain_contact_law`
//! with the other constitutive models. This module is the dynamics:
//! `population` (the `GrainPopulation` container and its integration step),
//! `coupling` (grid scatter/gather) and `oracle` (the packing-fraction
//! signal for where grains are needed, after Yue, Smith, Chen,
//! Chantharayukhonthorn, Kamrin & Grinspun 2018, "Hybrid Grains", ACM TOG
//! 37(6)).

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

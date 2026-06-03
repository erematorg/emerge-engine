pub mod core;

use bevy::prelude::*;
pub use core::newton_laws::NewtonLawsPlugin;

// FROZEN: no new features. Migrate to emerge's ForceField trait when emerge stabilizes.
//
// Portability split (DONE):
//   core::gravity_math — pure Rust: pair_force_vector, Barnes-Hut quadtree (usize keys),
//                        orbital/escape velocity formulas. Zero Bevy ECS coupling.
//   core::gravity      — ECS wrapper: GravityPlugin, systems, Components, Resources
//   core::newton_laws  — integration math (Velocity Verlet) + ECS systems
//
// NOTE: Velocity Verlet (2nd order) implemented — ~0.01% energy drift over 830+ seconds.

/// System sets for physics execution order.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum PhysicsSet {
    /// Force accumulation (Coulomb, gravity, etc.)
    AccumulateForces,
    /// Apply accumulated forces to velocities
    ApplyForces,
    /// Integrate velocities to update positions
    Integrate,
}

/// Interface for applying forces to entities
pub trait ForceApplicator: Send + Sync {
    /// Apply a force to an entity
    fn apply_force(&self, entity: Entity, force: Vec3);
    /// Get the force magnitude
    fn get_magnitude(&self) -> f32;
    /// Get the force direction
    fn get_direction(&self) -> Vec3;
}

/// Forces domain plugin
#[derive(Default)]
pub struct ForcesPlugin;

impl Plugin for ForcesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((NewtonLawsPlugin, core::gravity::GravityPlugin::new()))
            .register_type::<core::newton_laws::Mass>()
            .register_type::<core::newton_laws::Velocity>()
            .register_type::<core::newton_laws::AppliedForce>()
            .register_type::<core::gravity::GravityAffected>()
            .register_type::<core::gravity::GravitySource>()
            .register_type::<core::gravity::MassiveBody>();
    }
}

/// Common forces types and functions
pub mod prelude {
    // Core interfaces from crate root
    pub use crate::{ForceApplicator, ForcesPlugin, PhysicsSet};

    // Re-export core module prelude
    pub use crate::core::prelude::*;
}

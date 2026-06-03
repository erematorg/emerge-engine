use super::newton_laws::{AppliedForce, Mass};
use crate::PhysicsSet;
use bevy::prelude::*;

use super::gravity_math;
pub use super::gravity_math::{
    DEFAULT_GRAVITATIONAL_CONSTANT, MUTUAL_REALTIME_BODY_LIMIT, elliptical_orbit_velocity,
    escape_velocity, orbital_velocity, pair_force_vector, plummer_orbital_velocity,
};

// ---------------------------------------------------------------------------
// Legacy re-exports (keep public API stable for examples / tests)
// ---------------------------------------------------------------------------
pub use super::gravity_math::spatial::{Aabb, MassProperties, Quadtree, QuadtreeNode};

/// Deprecated name — use `orbital_velocity`.
#[inline]
pub fn calculate_orbital_velocity(central_mass: f32, orbit_radius: f32) -> f32 {
    orbital_velocity(central_mass, orbit_radius)
}
/// Deprecated name — use `plummer_orbital_velocity`.
#[inline]
pub fn calculate_plummer_orbital_velocity(
    central_mass: f32,
    orbit_radius: f32,
    softening: f32,
) -> f32 {
    plummer_orbital_velocity(central_mass, orbit_radius, softening)
}
/// Deprecated name — use `elliptical_orbit_velocity`.
#[inline]
pub fn calculate_elliptical_orbit_velocity(
    central_mass: f32,
    distance: f32,
    eccentricity: f32,
    is_periapsis: bool,
) -> f32 {
    elliptical_orbit_velocity(central_mass, distance, eccentricity, is_periapsis)
}
/// Deprecated name — use `escape_velocity`.
#[inline]
pub fn calculate_escape_velocity(central_mass: f32, distance: f32) -> f32 {
    escape_velocity(central_mass, distance)
}

// ---------------------------------------------------------------------------
// ECS components / resources
// ---------------------------------------------------------------------------

/// Configuration for the gravity simulation.
#[derive(Resource, Clone, Debug)]
pub struct GravityParams {
    pub softening: f32,
    pub gravitational_constant: f32,
    pub barnes_hut_max_depth: usize,
    pub barnes_hut_max_bodies_per_node: usize,
}

impl Default for GravityParams {
    fn default() -> Self {
        Self {
            softening: 5.0,
            gravitational_constant: DEFAULT_GRAVITATIONAL_CONSTANT,
            barnes_hut_max_depth: 8,
            barnes_hut_max_bodies_per_node: 8,
        }
    }
}

impl GravityParams {
    pub fn with_softening(mut self, softening: f32) -> Self {
        self.softening = softening;
        self
    }

    pub fn with_gravitational_constant(mut self, g: f32) -> Self {
        self.gravitational_constant = g;
        self
    }

    pub fn with_barnes_hut_params(mut self, max_depth: usize, max_bodies_per_node: usize) -> Self {
        self.barnes_hut_max_depth = max_depth.max(1);
        self.barnes_hut_max_bodies_per_node = max_bodies_per_node.max(1);
        self
    }
}

/// Selects how gravitational forces are applied.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GravityForceMode {
    /// Apply forces only to affected bodies (one-way).
    OneWay,
    /// Apply equal and opposite forces between participating bodies (Newton's 3rd law).
    #[default]
    Mutual,
}

/// Uniform gravitational field (e.g. Earth's surface).
#[derive(Resource, Debug, Clone, Copy)]
pub struct UniformGravity {
    pub acceleration: Vec3,
}

impl Default for UniformGravity {
    fn default() -> Self {
        Self {
            acceleration: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

/// Marks an entity as subject to gravitational forces.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct GravityAffected;

/// Marks measurement points for the gravitational field.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct GravityFieldMarker;

/// Marks an entity as a gravitational attractor.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct GravitySource;

/// Marks bodies with significant mass (tagged once in PreUpdate).
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct MassiveBody;

// ---------------------------------------------------------------------------
// Internal staging buffers (reused per frame via Local<>)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct GravityBody {
    entity: Entity,
    position: Vec3,
    mass: f32,
}

#[derive(Default)]
struct StagedGravitySources {
    entities: Vec<Entity>,
    positions: Vec<Vec3>,
    masses: Vec<f32>,
}

impl StagedGravitySources {
    fn stage(&mut self, query: &Query<(Entity, &Transform, &Mass), With<GravitySource>>) {
        self.entities.clear();
        self.positions.clear();
        self.masses.clear();
        for (entity, transform, mass) in query.iter() {
            if mass.value <= f32::EPSILON {
                continue;
            }
            self.entities.push(entity);
            self.positions.push(transform.translation);
            self.masses.push(mass.value);
        }
    }
}

#[derive(Default)]
struct MutualGravityBuffers {
    bodies: Vec<GravityBody>,
    forces: Vec<Vec3>,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn apply_uniform_gravity(
    gravity: Res<UniformGravity>,
    mut query: Query<(&Mass, &mut AppliedForce), With<GravityAffected>>,
) {
    for (mass, mut force) in &mut query {
        force.force += mass.value * gravity.acceleration;
    }
}

/// Tags qualifying bodies as MassiveBody once, not every frame.
/// Runs in PreUpdate so gravity systems see the component immediately.
pub fn tag_massive_bodies(
    mut commands: Commands,
    query: Query<(Entity, &Mass), (With<GravityAffected>, Without<MassiveBody>)>,
) {
    for (entity, mass) in &query {
        if mass.value > 1000.0 {
            commands.entity(entity).insert(MassiveBody);
        }
    }
}

#[allow(private_interfaces)]
pub fn calculate_mutual_gravitational_attraction(
    gravity_params: Res<GravityParams>,
    mut query: Query<(Entity, &Transform, &Mass, &mut AppliedForce), With<GravitySource>>,
    mut ctx: Local<MutualGravityBuffers>,
) {
    let softening_squared = gravity_params.softening * gravity_params.softening;
    let g = gravity_params.gravitational_constant;

    ctx.bodies.clear();
    ctx.bodies.extend(
        query
            .iter()
            .map(|(entity, transform, mass, _)| GravityBody {
                entity,
                position: transform.translation,
                mass: mass.value,
            }),
    );
    ctx.bodies.sort_by_key(|b| b.entity.to_bits());

    if ctx.bodies.len() > MUTUAL_REALTIME_BODY_LIMIT {
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            warn!(
                "Mutual gravity with {} bodies exceeds LP-0 realtime guideline ({}). \
                 Consider one-way gravity or Barnes-Hut.",
                ctx.bodies.len(),
                MUTUAL_REALTIME_BODY_LIMIT
            );
        }
    }

    let n = ctx.bodies.len();
    ctx.forces.clear();
    ctx.forces.resize(n, Vec3::ZERO);

    for i in 0..n {
        let a = ctx.bodies[i];
        for j in (i + 1)..n {
            let b = ctx.bodies[j];
            let Some(f_on_a) =
                pair_force_vector(b.position, b.mass, a.position, a.mass, g, softening_squared)
            else {
                continue;
            };
            ctx.forces[i] += f_on_a;
            ctx.forces[j] -= f_on_a; // Newton's 3rd law
        }
    }

    for (idx, body) in ctx.bodies.iter().enumerate() {
        if let Ok((_, _, _, mut applied)) = query.get_mut(body.entity) {
            applied.force += ctx.forces[idx];
        }
    }
}

pub fn calculate_gravitational_attraction(
    gravity_params: Res<GravityParams>,
    query: Query<(Entity, &Transform, &Mass), With<GravitySource>>,
    mut affected_query: Query<
        (Entity, &Transform, &Mass, &mut AppliedForce),
        With<GravityAffected>,
    >,
) {
    let softening_squared = gravity_params.softening * gravity_params.softening;
    let g = gravity_params.gravitational_constant;

    let mut sources = StagedGravitySources::default();
    sources.stage(&query);

    one_way_gravity_inner(&sources, &mut affected_query, g, softening_squared);
}

fn one_way_gravity_inner(
    sources: &StagedGravitySources,
    affected_query: &mut Query<
        (Entity, &Transform, &Mass, &mut AppliedForce),
        With<GravityAffected>,
    >,
    g: f32,
    softening_squared: f32,
) {
    affected_query.par_iter_mut().for_each(
        |(affected_entity, affected_transform, affected_mass, mut force)| {
            let affected_pos = affected_transform.translation;
            for i in 0..sources.entities.len() {
                if sources.entities[i] == affected_entity {
                    continue;
                }
                let Some(fv) = pair_force_vector(
                    sources.positions[i],
                    sources.masses[i],
                    affected_pos,
                    affected_mass.value,
                    g,
                    softening_squared,
                ) else {
                    continue;
                };
                force.force += fv;
            }
        },
    );
}

pub fn calculate_barnes_hut_attraction(
    gravity_params: Res<GravityParams>,
    query: Query<(Entity, &Transform, &Mass), With<GravitySource>>,
    mut affected_query: Query<
        (Entity, &Transform, &Mass, &mut AppliedForce),
        With<GravityAffected>,
    >,
    theta: f32,
) {
    let softening = gravity_params.softening;
    let softening_squared = softening * softening;
    let g = gravity_params.gravitational_constant;

    if query.iter().count() < 20 {
        let mut sources = StagedGravitySources::default();
        sources.stage(&query);
        one_way_gravity_inner(&sources, &mut affected_query, g, softening_squared);
        return;
    }

    // Single-pass staging: build both the quadtree body list and entity→idx map
    // from the same iteration to guarantee consistent ordering.
    let mut body_data: Vec<(usize, Vec3, f32)> = Vec::new();
    let mut entity_to_idx: std::collections::HashMap<Entity, usize> =
        std::collections::HashMap::new();
    for (entity, transform, mass) in query.iter() {
        let idx = body_data.len();
        body_data.push((idx, transform.translation, mass.value));
        entity_to_idx.insert(entity, idx);
    }

    let quadtree = Quadtree::from_indexed_bodies(
        &body_data,
        gravity_params.barnes_hut_max_depth,
        gravity_params.barnes_hut_max_bodies_per_node,
    );

    affected_query
        .par_iter_mut()
        .for_each(|(entity, transform, mass, mut force)| {
            let position = transform.translation;
            // Use the entity's source idx (usize::MAX means not a source → no self-skip needed).
            let self_idx = entity_to_idx.get(&entity).copied().unwrap_or(usize::MAX);

            let fv = gravity_math::barnes_hut_force(
                self_idx,
                position,
                mass.value,
                &quadtree.root,
                theta,
                softening,
                g,
            );
            force.force += fv;
        });
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct GravityPlugin {
    pub use_barnes_hut: bool,
    pub barnes_hut_theta: f32,
}

impl GravityPlugin {
    pub fn new() -> Self {
        Self {
            use_barnes_hut: true,
            barnes_hut_theta: 0.5,
        }
    }

    pub fn with_barnes_hut(mut self, enabled: bool) -> Self {
        self.use_barnes_hut = enabled;
        self
    }

    pub fn with_theta(mut self, theta: f32) -> Self {
        self.barnes_hut_theta = theta.clamp(0.1, 1.0);
        self
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GravitySet {
    UniformGravity,
    NBodyGravity,
}

fn use_mutual(mode: Res<GravityForceMode>) -> bool {
    *mode == GravityForceMode::Mutual
}

fn use_one_way(mode: Res<GravityForceMode>) -> bool {
    *mode == GravityForceMode::OneWay
}

fn has_many_sources(query: Query<(Entity, &Transform, &Mass), With<GravitySource>>) -> bool {
    query.iter().count() >= 20
}

fn has_few_sources(query: Query<(Entity, &Transform, &Mass), With<GravitySource>>) -> bool {
    query.iter().count() < 20
}

impl Plugin for GravityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GravityParams>()
            .init_resource::<UniformGravity>()
            .init_resource::<GravityForceMode>()
            .configure_sets(
                FixedUpdate,
                (GravitySet::UniformGravity, GravitySet::NBodyGravity)
                    .chain()
                    .in_set(PhysicsSet::AccumulateForces),
            )
            .add_systems(PreUpdate, tag_massive_bodies)
            .add_systems(
                FixedUpdate,
                apply_uniform_gravity.in_set(GravitySet::UniformGravity),
            )
            .add_systems(
                FixedUpdate,
                calculate_mutual_gravitational_attraction
                    .in_set(GravitySet::NBodyGravity)
                    .run_if(use_mutual),
            );

        if self.use_barnes_hut {
            let theta = self.barnes_hut_theta;
            app.add_systems(
                FixedUpdate,
                (move |gravity_params: Res<GravityParams>,
                       query: Query<(Entity, &Transform, &Mass), With<GravitySource>>,
                       affected_query: Query<
                    (Entity, &Transform, &Mass, &mut AppliedForce),
                    With<GravityAffected>,
                >| {
                    calculate_barnes_hut_attraction(gravity_params, query, affected_query, theta);
                })
                .in_set(GravitySet::NBodyGravity)
                .run_if(use_one_way)
                .run_if(has_many_sources),
            );
            app.add_systems(
                FixedUpdate,
                calculate_gravitational_attraction
                    .in_set(GravitySet::NBodyGravity)
                    .run_if(use_one_way)
                    .run_if(has_few_sources),
            );
        } else {
            app.add_systems(
                FixedUpdate,
                calculate_gravitational_attraction
                    .in_set(GravitySet::NBodyGravity)
                    .run_if(use_one_way),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_body_action_reaction() {
        let params = GravityParams::default();
        let g = params.gravitational_constant;
        let softening_sq = params.softening * params.softening;

        let pos1 = Vec3::new(0.0, 0.0, 0.0);
        let pos2 = Vec3::new(10.0, 0.0, 0.0);
        let mass1 = 5.0;
        let mass2 = 3.0;

        let direction = pos2 - pos1;
        let dist_sq = direction.length_squared();
        let norm_s = dist_sq + softening_sq;
        let force_scalar = g * mass1 * mass2 / (norm_s * norm_s.sqrt());
        let force_on_2 = direction * force_scalar;
        let force_on_1 = -force_on_2;

        assert!((force_on_1.length() - force_on_2.length()).abs() < 1e-5);
        assert!(
            (force_on_1 + force_on_2).length() < 1e-5,
            "Action-reaction forces do not cancel"
        );
    }

    #[test]
    fn test_barnes_hut_vs_brute_force_small_n() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(GravityParams::default());

        let bodies = vec![
            (Vec3::new(0.0, 0.0, 0.0), 100.0),
            (Vec3::new(50.0, 0.0, 0.0), 50.0),
            (Vec3::new(0.0, 50.0, 0.0), 50.0),
        ];

        let entities: Vec<_> = bodies
            .iter()
            .map(|(pos, mass)| {
                app.world_mut()
                    .spawn((
                        Transform::from_translation(*pos),
                        Mass::new(*mass),
                        AppliedForce::new(Vec3::ZERO),
                        GravitySource,
                        GravityAffected,
                    ))
                    .id()
            })
            .collect();

        let mut brute_forces = vec![Vec3::ZERO; entities.len()];

        let params = app.world().resource::<GravityParams>();
        let g = params.gravitational_constant;
        let soft_sq = params.softening * params.softening;

        for i in 0..entities.len() {
            for j in 0..entities.len() {
                if i == j {
                    continue;
                }
                let dir = bodies[j].0 - bodies[i].0;
                let dist_sq = dir.length_squared();
                let norm_s = dist_sq + soft_sq;
                let force_scalar = g * bodies[j].1 * bodies[i].1 / (norm_s * norm_s.sqrt());
                brute_forces[i] += dir * force_scalar;
            }
        }

        let body_data: Vec<(usize, Vec3, f32)> = entities
            .iter()
            .enumerate()
            .map(|(i, _)| (i, bodies[i].0, bodies[i].1))
            .collect();

        let params = app.world().resource::<GravityParams>();
        let quadtree = Quadtree::from_indexed_bodies(
            &body_data,
            params.barnes_hut_max_depth,
            params.barnes_hut_max_bodies_per_node,
        );

        for (i, _) in entities.iter().enumerate() {
            let bh_force = gravity_math::barnes_hut_force(
                i,
                bodies[i].0,
                bodies[i].1,
                &quadtree.root,
                0.5,
                params.softening,
                g,
            );
            let diff = (bh_force - brute_forces[i]).length();
            assert!(diff < 1.0, "BH force differs from brute force by {}", diff);
        }
    }
}

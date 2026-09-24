extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real two-way coupling test: the obstacle is NOT scripted this time. It
/// starts with a real initial velocity heading into settled water, and its
/// OWN velocity evolves from the real, mass-weighted reaction impulse
/// `KinematicCircleBoundary::take_reaction_impulse()` accumulates each step
/// (Newton's third law, fed by the exact same grid correction already
/// verified safe against strict-fluid water). If this is real physics, a
/// light-enough obstacle plowing into water should visibly decelerate --
/// water pushing back, not just the obstacle pushing water.
use emerge::{KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SETTLE_STEPS: usize = 150;
const RUN_STEPS: usize = 200;
const MAX_SUBSTEPS: usize = 400;
const OBSTACLE_RADIUS: f32 = 2.0;
const OBSTACLE_MASS: f32 = 8.0;
const INITIAL_SPEED: f32 = 2.0;

fn bounding_box(sim: &Simulation) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for p in sim.particles().x.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: MAX_SUBSTEPS,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    const SPACING: f32 = 0.9;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 10),
        box_center: Vec2::new(30.0, 8.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let obstacle = Arc::new(KinematicCircleBoundary::new(
        Vec2::new(1.0, 1.0),
        OBSTACLE_RADIUS,
        0.3,
    ));
    let mut sim = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(obstacle.clone()));

    let total_before = sim.particles().x.len();
    for _ in 0..SETTLE_STEPS {
        sim.step();
    }
    let (bb_min, bb_max) = bounding_box(&sim);
    println!(
        "settle done: {} particles, bbox=({:.2},{:.2})-({:.2},{:.2})",
        total_before, bb_min.x, bb_min.y, bb_max.x, bb_max.y
    );

    let mut obstacle_pos = Vec2::new(
        bb_min.x - OBSTACLE_RADIUS - 1.0,
        (bb_min.y + bb_max.y) * 0.5,
    );
    let mut obstacle_vel = Vec2::new(INITIAL_SPEED, 0.0);
    println!(
        "obstacle: start=({:.2},{:.2}) initial_v=({:.2},{:.2}) mass={}",
        obstacle_pos.x, obstacle_pos.y, obstacle_vel.x, obstacle_vel.y, OBSTACLE_MASS
    );

    for step in 0..RUN_STEPS {
        obstacle.set_position_velocity(obstacle_pos, obstacle_vel);
        sim.step();
        let reaction = obstacle.take_reaction_impulse();
        obstacle_vel += reaction / OBSTACLE_MASS;
        obstacle_pos += obstacle_vel * DT;

        if step % 20 == 0 {
            println!(
                "step={:>3} obstacle_pos=({:.2},{:.2}) obstacle_v=({:.3},{:.3}) speed={:.3} reaction_this_step=({:.4},{:.4})",
                step,
                obstacle_pos.x,
                obstacle_pos.y,
                obstacle_vel.x,
                obstacle_vel.y,
                obstacle_vel.length(),
                reaction.x,
                reaction.y
            );
        }
    }

    let total_after = sim.particles().x.len();
    let max_speed = sim
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    println!(
        "final: count {}->{}  obstacle_v=({:.3},{:.3}) speed={:.3} (started at {:.2})  water max_speed={:.2}",
        total_before,
        total_after,
        obstacle_vel.x,
        obstacle_vel.y,
        obstacle_vel.length(),
        INITIAL_SPEED,
        max_speed
    );
}

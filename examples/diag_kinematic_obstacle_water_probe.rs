extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Second pass: the first pass's obstacle likely never touched the settled
/// puddle (placed by guessing the post-settle geometry, which turned out
/// wrong -- min_x=1.57 vs an assumed ~12). This pass MEASURES the real
/// settled water bounding box first, places the obstacle just outside it,
/// and prints live obstacle-to-nearest-water-particle distance every
/// checkpoint so real contact is CONFIRMED, not assumed, before trusting
/// any push-phase measurement.
use emerge::{KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SETTLE_STEPS: usize = 150;
const PUSH_STEPS: usize = 200;
const PUSH_SPEED: f32 = 0.3;
const MAX_SUBSTEPS: usize = 400;
const OBSTACLE_RADIUS: f32 = 2.0;

fn bounding_box(sim: &Simulation) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for p in sim.particles().x.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

fn nearest_water_dist(sim: &Simulation, obstacle_center: Vec2) -> f32 {
    sim.particles()
        .x
        .iter()
        .map(|p| (*p - obstacle_center).length())
        .fold(f32::MAX, f32::min)
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

    // Placeholder far away during settle -- real placement decided AFTER
    // measuring where the puddle actually ends up, not guessed in advance.
    let obstacle = Arc::new(KinematicCircleBoundary::new(
        Vec2::new(1.0, 1.0),
        OBSTACLE_RADIUS,
        0.3,
    ));
    let mut sim = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(obstacle.clone()));

    let total_before = sim.particles().x.len();
    println!(
        "water + kinematic circle obstacle, {} particles",
        total_before
    );

    for _ in 0..SETTLE_STEPS {
        sim.step();
    }
    let settle_speed = sim
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    let (bb_min, bb_max) = bounding_box(&sim);
    println!(
        "settle: max_speed={:.2} bbox=({:.2},{:.2})-({:.2},{:.2})",
        settle_speed, bb_min.x, bb_min.y, bb_max.x, bb_max.y
    );

    // Place the obstacle just outside the settled puddle's left edge, at
    // mid-height of the puddle -- guaranteed close enough to reach real
    // contact quickly once pushed rightward, verified live below, not assumed.
    let start_x = bb_min.x - OBSTACLE_RADIUS - 1.0;
    let mid_y = (bb_min.y + bb_max.y) * 0.5;
    let start = Vec2::new(start_x, mid_y);
    println!(
        "obstacle start=({:.2},{:.2}) radius={}",
        start.x, start.y, OBSTACLE_RADIUS
    );

    let mut max_speed = settle_speed;
    let mut min_dist_ever = f32::MAX;
    for step in 0..PUSH_STEPS {
        let center = start + Vec2::new(PUSH_SPEED * step as f32 * DT, 0.0);
        obstacle.set_position_velocity(center, Vec2::new(PUSH_SPEED, 0.0));
        sim.step();
        for v in sim.particles().v.iter() {
            max_speed = max_speed.max(v.length());
        }
        let d = nearest_water_dist(&sim, center);
        min_dist_ever = min_dist_ever.min(d);
        if step % 25 == 0 {
            println!(
                "  step={:>3} max_speed={:.2} obstacle_x={:.2} nearest_water_dist={:.2} (radius={})",
                step, max_speed, center.x, d, OBSTACLE_RADIUS
            );
        }
    }

    let total_after = sim.particles().x.len();
    let (bb_min_after, bb_max_after) = bounding_box(&sim);
    println!(
        "count {}->{}  max_speed={:.2}  min_dist_ever={:.2} (< radius={} means REAL overlap occurred)  bbox_after=({:.2},{:.2})-({:.2},{:.2})",
        total_before,
        total_after,
        max_speed,
        min_dist_ever,
        OBSTACLE_RADIUS,
        bb_min_after.x,
        bb_min_after.y,
        bb_max_after.x,
        bb_max_after.y
    );
}

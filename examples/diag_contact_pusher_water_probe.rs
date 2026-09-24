extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Refined pass: let the water column settle under gravity FIRST (no
/// pusher motion), THEN engage the kinematic contact pusher and measure
/// displacement from the settled baseline -- isolates the real push effect
/// from the column's own gravity-collapse dynamics (confound found in the
/// first pass). Also narrows the mass-ratio search between the last known
/// stable point (5.0) and the last known unstable point (20.0), same
/// bisection-by-measurement methodology as the 2026-07-23 sand-shovel sweep.
use emerge::{
    NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const MAT_PUSHER: u32 = 1;
const SETTLE_STEPS: usize = 150;
const PUSH_STEPS: usize = 150;
const PUSH_SPEED: f32 = 0.3;
const MAX_SUBSTEPS: usize = 400;

fn make_sim(mass_multiplier: f32) -> (Simulation, u32) {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: MAX_SUBSTEPS,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    let pusher = NeoHookeanMaterial::from_young_modulus(1.0e4, 0.3);

    const SPACING: f32 = 0.9;
    // Shallower, wider puddle instead of a tall column -- less of its own
    // gravity-collapse energy to confound the push measurement with.
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 10),
        box_center: Vec2::new(24.0, 8.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_pusher = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(5.0, 8.0),
        material_id: MAT_PUSHER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_PUSHER, Box::new(pusher))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let pusher_tag = solver.add_body(spawn_pusher);
    solver.set_group_contact_group(pusher_tag, 1);
    solver.scale_group_mass(pusher_tag, mass_multiplier);
    (solver, pusher_tag)
}

fn mean_x_in_box(sim: &Simulation, box_min: Vec2, box_max: Vec2) -> f32 {
    let xs: Vec<f32> = sim
        .particles()
        .x
        .iter()
        .filter(|p| p.x >= box_min.x && p.x <= box_max.x && p.y >= box_min.y && p.y <= box_max.y)
        .map(|p| p.x)
        .collect();
    xs.iter().sum::<f32>() / xs.len().max(1) as f32
}

fn run_probe(mass_multiplier: f32) {
    let (mut sim, pusher_tag) = make_sim(mass_multiplier);
    let total_before = sim.particles().x.len();

    // Phase 1: settle. Pusher held at zero velocity, far from the puddle.
    for _ in 0..SETTLE_STEPS {
        sim.set_group_velocity(pusher_tag, Vec2::ZERO);
        sim.step();
    }
    let settle_speed = sim
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    let box_min = Vec2::new(10.0, 2.0);
    let box_max = Vec2::new(24.0, 12.0);
    let mean_x_settled = mean_x_in_box(&sim, box_min, box_max);

    // Phase 2: push. Same settled scene, now drive the contact body.
    let push_velocity = Vec2::new(PUSH_SPEED, 0.0);
    let mut max_speed = settle_speed;
    for step in 0..PUSH_STEPS {
        sim.set_group_velocity(pusher_tag, push_velocity);
        sim.step();
        for v in sim.particles().v.iter() {
            max_speed = max_speed.max(v.length());
        }
        if step % 50 == 0 {
            println!(
                "  mult={:>5.1} push_step={:>3} max_speed={:.2}",
                mass_multiplier, step, max_speed
            );
        }
    }

    let total_after = sim.particles().x.len();
    let mean_x_pushed = mean_x_in_box(&sim, box_min, box_max);

    println!(
        "mult={:>5.1}  count {}->{}  settle_speed={:.2}  max_speed={:.2}  mean_x settled={:.2} pushed={:.2} (delta={:+.2})",
        mass_multiplier,
        total_before,
        total_after,
        settle_speed,
        max_speed,
        mean_x_settled,
        mean_x_pushed,
        mean_x_pushed - mean_x_settled
    );
}

fn main() {
    println!(
        "settled-baseline contact push probe -- max_substeps={}",
        MAX_SUBSTEPS
    );
    for mult in [5.0, 8.0, 12.0, 16.0, 20.0] {
        run_probe(mult);
    }
}

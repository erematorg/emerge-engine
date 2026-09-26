//! Real, direct re-verification: does `KinematicCircleBoundary`'s real
//! two-way reaction-impulse coupling (ported back into the current
//! codebase 2026-09-14 from a stale, unmerged branch -- see
//! `[[project_fluid_solid_coupling_real_root_cause_and_path_2026-08-15]]`,
//! project memory) still behave correctly against the CURRENT engine, 29
//! real days after it was last measured? The port itself only touched
//! wiring (a new default trait method + one call site); this test re-runs
//! the ORIGINAL real probe's own scenario (`diag_reactive_obstacle_water_
//! probe.rs` on the `contact-based-interaction` branch, byte-for-byte
//! adapted from a `fn main()` example into a `#[test]`) to confirm the
//! actual PHYSICS -- not just that it compiles -- still holds after
//! everything else this engine has changed since 2026-08-16.
//!
//! Real design, unchanged from the original probe: the obstacle is NOT
//! scripted -- it starts with a real initial velocity heading into
//! settled water, and its OWN velocity evolves purely from the real,
//! mass-weighted reaction impulse `take_reaction_impulse()` accumulates
//! each step (Newton's third law). If this is real physics, a light-
//! enough obstacle plowing into water should visibly decelerate on
//! contact -- water pushing back, not just the obstacle pushing water.

extern crate emerge_engine as emerge;
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

#[test]
#[ignore = "real re-verification of the ported KinematicCircleBoundary two-way coupling -- run \
            explicitly with --release --ignored --nocapture"]
fn reactive_obstacle_genuinely_decelerates_and_water_pushes_back() {
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
        "settle done: {total_before} particles, bbox=({:.2},{:.2})-({:.2},{:.2})",
        bb_min.x, bb_min.y, bb_max.x, bb_max.y
    );

    let mut obstacle_pos = Vec2::new(
        bb_min.x - OBSTACLE_RADIUS - 1.0,
        (bb_min.y + bb_max.y) * 0.5,
    );
    let mut obstacle_vel = Vec2::new(INITIAL_SPEED, 0.0);
    println!(
        "obstacle: start=({:.2},{:.2}) initial_v=({:.2},{:.2}) mass={OBSTACLE_MASS}",
        obstacle_pos.x, obstacle_pos.y, obstacle_vel.x, obstacle_vel.y
    );

    let mut min_speed_after_contact: f32 = f32::MAX;
    let mut saw_contact = false;
    let mut saw_noncontact_stretch_after_contact = false;
    for step in 0..RUN_STEPS {
        obstacle.set_position_velocity(obstacle_pos, obstacle_vel);
        sim.step();
        let reaction = obstacle.take_reaction_impulse();
        if reaction != Vec2::ZERO {
            saw_contact = true;
        } else if saw_contact {
            saw_noncontact_stretch_after_contact = true;
        }
        obstacle_vel += reaction / OBSTACLE_MASS;
        obstacle_pos += obstacle_vel * DT;
        if saw_contact {
            min_speed_after_contact = min_speed_after_contact.min(obstacle_vel.length());
        }

        if step % 20 == 0 {
            println!(
                "step={step:>3} obstacle_pos=({:.2},{:.2}) obstacle_v=({:.3},{:.3}) speed={:.3} \
                 reaction_this_step=({:.4},{:.4})",
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
        "final: count {total_before}->{total_after}  obstacle_v=({:.3},{:.3}) speed={:.3} \
         (started at {INITIAL_SPEED:.2})  water max_speed={max_speed:.2}",
        obstacle_vel.x,
        obstacle_vel.y,
        obstacle_vel.length()
    );

    assert_eq!(
        total_before, total_after,
        "mass must be exactly conserved -- no particle should appear/disappear from a \
         boundary-correction-only interaction"
    );
    assert!(
        max_speed.is_finite() && max_speed < 100.0,
        "water must stay bounded, no blowup -- max_speed={max_speed}"
    );
    assert!(
        saw_contact,
        "the obstacle never registered a single real contact -- the whole test is \
         inconclusive if geometry never overlapped"
    );
    assert!(
        min_speed_after_contact < INITIAL_SPEED * 0.9,
        "a real two-way reaction must genuinely decelerate the obstacle on contact -- \
         started at {INITIAL_SPEED}, min speed after contact was {min_speed_after_contact}"
    );
    assert!(
        saw_noncontact_stretch_after_contact,
        "must see at least one later step with exactly zero reaction impulse -- proves \
         the coupling is contact-driven, not a constant drag hack"
    );
}

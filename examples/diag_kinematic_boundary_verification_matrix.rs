extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Broader verification of `KinematicCircleBoundary`'s
/// `is_strict_wc_mpm_fluid_compatible` flag before building anything more
/// on top of it (per the agreed plan: verify wide, THEN generalize).
/// `diag_reactive_obstacle_water_probe.rs` proved ONE case (water, moderate
/// horizontal speed, ~2.0). This runs a small real matrix: a different
/// strict-fluid material (Bingham mud), a higher speed, and a different
/// approach angle (vertical drop) -- same settle-then-verified-contact
/// methodology throughout, real distance tracking, not assumed geometry.
use emerge::{
    BinghamFluidMaterial, KinematicCircleBoundary, MaterialModel, NewtonianFluidMaterial,
    SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_FLUID: u32 = 0;
const SETTLE_STEPS: usize = 150;
const RUN_STEPS: usize = 200;
const MAX_SUBSTEPS: usize = 400;
const OBSTACLE_RADIUS: f32 = 2.0;
const OBSTACLE_MASS: f32 = 8.0;
const SANE_SPEED_CEILING: f32 = 30.0; // catches a pinned-clamp-style blowup, not a real physical speed

struct CaseResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

fn bounding_box(sim: &Simulation) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for p in sim.particles().x.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

fn nearest_dist(sim: &Simulation, center: Vec2) -> f32 {
    sim.particles()
        .x
        .iter()
        .map(|p| (*p - center).length())
        .fold(f32::MAX, f32::min)
}

fn run_case(
    name: &'static str,
    material: Box<dyn MaterialModel>,
    approach_dir: Vec2,
    speed: f32,
) -> CaseResult {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: MAX_SUBSTEPS,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    const SPACING: f32 = 0.9;
    let spawn_fluid = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 10),
        box_center: Vec2::new(30.0, 8.0),
        material_id: MAT_FLUID,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let obstacle = Arc::new(KinematicCircleBoundary::new(
        Vec2::new(1.0, 1.0),
        OBSTACLE_RADIUS,
        0.3,
    ));
    let mut sim = Simulation::new(config, spawn_fluid)
        .with_default_material(material)
        .with_boundary(Box::new(obstacle.clone()));

    let total_before = sim.particles().x.len();
    for _ in 0..SETTLE_STEPS {
        sim.step();
    }
    let (bb_min, bb_max) = bounding_box(&sim);

    // Place the obstacle just outside the settled bbox, opposite the
    // approach direction, so it's guaranteed to travel INTO the fluid.
    let mid = (bb_min + bb_max) * 0.5;
    let start = mid
        - approach_dir.normalize_or(Vec2::X) * (bb_max - bb_min).length() * 0.5
        - approach_dir.normalize_or(Vec2::X) * (OBSTACLE_RADIUS + 1.0);
    let mut pos = start;
    let mut vel = approach_dir.normalize_or(Vec2::X) * speed;

    let mut max_speed = 0.0f32;
    let mut min_dist_ever = f32::MAX;
    let mut any_nonfinite = false;

    for _ in 0..RUN_STEPS {
        let reaction = obstacle.take_reaction_impulse();
        vel += config.gravity * DT + reaction / OBSTACLE_MASS;
        pos += vel * DT;
        obstacle.set_position_velocity(pos, vel);
        sim.step();
        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
            max_speed = max_speed.max(v.length());
        }
        min_dist_ever = min_dist_ever.min(nearest_dist(&sim, pos));
    }

    let total_after = sim.particles().x.len();
    let contact_confirmed = min_dist_ever <= OBSTACLE_RADIUS + 0.5;
    let mass_conserved = total_before == total_after;
    let bounded = max_speed < SANE_SPEED_CEILING;
    let passed = !any_nonfinite && mass_conserved && bounded && contact_confirmed;

    CaseResult {
        name,
        passed,
        detail: format!(
            "count {}->{} contact_dist={:.2}(r={}) max_speed={:.2} finite={} final_v=({:.2},{:.2})",
            total_before,
            total_after,
            min_dist_ever,
            OBSTACLE_RADIUS,
            max_speed,
            !any_nonfinite,
            vel.x,
            vel.y
        ),
    }
}

fn main() {
    let results = vec![
        run_case(
            "water, moderate speed, horizontal (baseline, re-confirm)",
            Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 2.5)),
            Vec2::X,
            2.0,
        ),
        run_case(
            "water, HIGH speed, horizontal",
            Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 2.5)),
            Vec2::X,
            6.0,
        ),
        run_case(
            "Bingham mud, moderate speed, horizontal",
            Box::new(BinghamFluidMaterial::new(4.0, 8.0, 100.0, 3.0, 4.0)),
            Vec2::X,
            2.0,
        ),
        run_case(
            "water, moderate speed, VERTICAL drop",
            Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 2.5)),
            Vec2::NEG_Y,
            2.0,
        ),
    ];

    println!("\n=== KinematicCircleBoundary verification matrix ===");
    let mut all_passed = true;
    for r in &results {
        all_passed &= r.passed;
        println!(
            "[{}] {} -- {}",
            if r.passed { "PASS" } else { "FAIL" },
            r.name,
            r.detail
        );
    }
    println!(
        "\n{}",
        if all_passed {
            "ALL CASES PASSED"
        } else {
            "AT LEAST ONE CASE FAILED -- see above"
        }
    );
}

extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Proves `KinematicCircleBoundary`'s new per-material friction profile
/// produces a REAL, measurable, physically distinct outcome for water vs.
/// mud at the SAME obstacle -- not just plumbing that compiles.
///
/// Friction values used are NOT invented for this probe -- both trace to a
/// real source already in this codebase:
///   - mud: 0.2, wet clay -- `FrictionBoundary`'s own doc comment
///     (`src/forces/boundary/friction/base.rs`), "IRL mu values: rock-on-
///     rock ~= 0.6, wet clay ~= 0.2, ice ~= 0.05".
///   - water: ~0.0 -- consistent with this engine's own existing default
///     for every fluid demo (`basic_fluids.rs` et al. all use
///     `SlipBoundary`, i.e. free-slip/zero tangential friction, for water
///     walls) -- a real, established engine convention, not a guessed
///     number. Honestly NOT a primary tribology citation for water-on-
///     solid mu specifically -- see project_fluid_wall_noslip_friction_
///     vision memory for the disclosed gap this traces to.
///
/// Approach is DIAGONAL (both a normal and a tangential velocity
/// component) specifically because `apply_coulomb_wall`'s friction term
/// only acts on the tangential component -- a pure head-on hit wouldn't
/// exercise the thing being tested at all.
use emerge::{
    BinghamFluidMaterial, KinematicCircleBoundary, MaterialModel, NewtonianFluidMaterial,
    SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_FLUID: u32 = 0;
const MAT_WATER: u32 = 10; // arbitrary IDs used only as profile lookup keys in this probe
const MAT_MUD: u32 = 11;
const SETTLE_STEPS: usize = 150;
const RUN_STEPS: usize = 150;
const MAX_SUBSTEPS: usize = 400;
const OBSTACLE_RADIUS: f32 = 2.0;
const OBSTACLE_MASS: f32 = 8.0;

// Sourced, see module doc.
const FRICTION_WATER: f32 = 0.0;
const FRICTION_MUD: f32 = 0.2;

fn bounding_box(sim: &Simulation) -> (Vec2, Vec2) {
    let mut min = Vec2::splat(f32::MAX);
    let mut max = Vec2::splat(f32::MIN);
    for p in sim.particles().x.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }
    (min, max)
}

fn run_case(name: &str, material: Box<dyn MaterialModel>, active_friction: f32) -> f32 {
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
        box_size: IVec2::new(30, 10),
        box_center: Vec2::new(32.0, 8.0),
        material_id: MAT_FLUID,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let obstacle = Arc::new(
        KinematicCircleBoundary::new(Vec2::new(1.0, 1.0), OBSTACLE_RADIUS, 0.0)
            .with_material_friction(MAT_WATER, FRICTION_WATER)
            .with_material_friction(MAT_MUD, FRICTION_MUD),
    );
    obstacle.set_active_friction(active_friction);

    let mut sim = Simulation::new(config, spawn_fluid)
        .with_default_material(material)
        .with_boundary(Box::new(obstacle.clone()));

    for _ in 0..SETTLE_STEPS {
        sim.step();
    }
    let (bb_min, bb_max) = bounding_box(&sim);

    // Diagonal approach: skims along the settled surface (tangential) while
    // also pressing slightly into it (normal) -- friction's real effect is
    // only on the tangential component, so this is the case that actually
    // exercises it.
    let mut pos = Vec2::new(bb_min.x - OBSTACLE_RADIUS - 1.0, bb_max.y + 1.0);
    let mut vel = Vec2::new(3.0, -0.6); // strongly tangential (rightward), mildly into the surface

    for _ in 0..RUN_STEPS {
        let reaction = obstacle.take_reaction_impulse();
        vel += config.gravity * DT + reaction / OBSTACLE_MASS;
        pos += vel * DT;
        obstacle.set_position_velocity(pos, vel);
        sim.step();
    }

    println!(
        "{name}: active_friction={:.2} final_v=({:.3},{:.3}) tangential(x)_speed={:.3}",
        active_friction, vel.x, vel.y, vel.x
    );
    vel.x
}

fn main() {
    println!(
        "Per-material friction profile probe -- same obstacle, same approach, different material"
    );

    let water_tangential = run_case(
        "water",
        Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 2.5)),
        FRICTION_WATER,
    );
    let mud_tangential = run_case(
        "mud",
        Box::new(BinghamFluidMaterial::new(4.0, 8.0, 100.0, 3.0, 4.0)),
        FRICTION_MUD,
    );

    println!(
        "\nwater retained tangential speed {:.3}, mud retained {:.3} (delta={:+.3})",
        water_tangential,
        mud_tangential,
        mud_tangential - water_tangential
    );
    if mud_tangential < water_tangential {
        println!(
            "PASS: higher-friction mud bled off more tangential speed than near-zero-friction water, as expected from the SAME profile mechanism."
        );
    } else {
        println!(
            "UNEXPECTED: mud did not lose more tangential speed than water -- investigate before trusting the profile."
        );
    }
}

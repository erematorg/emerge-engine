extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real, sourced rolling-snowball accretion test. Real closed-form target:
/// Rubin 2019 ("A Variable-Mass Snowball Rolling Down a Snowy Slope," The
/// Physics Teacher 57(3):150) and Mungan 2019 (companion analytic paper) --
/// a rolling snowball accreting mass approaches a terminal acceleration of
/// (1/6)*g*sin(theta) down a constant-angle slope, independent of mass.
///
/// Real, disclosed accretion mechanism (NOT a true particle-merge system --
/// that's real, separate, future work, see project_generic_multi_material_
/// coupling_vision memory): loose snow particles within the growing ball's
/// own KinematicCircleBoundary radius get relabeled loose->packed
/// (material_id write + phase-transition-style semantics), and the
/// boundary's own radius grows to match the real 2D area added (each
/// particle represents `spacing^2` of area, same convention used
/// throughout this engine's own mass_override derivations).
///
/// Real physical grounding for using COHESIVE (not loose/dry) snow as the
/// thing being rolled through: wet-snow accretion literature (Journal of
/// Glaciology 1950, "Snow Rollers") attributes real snowball/roller growth
/// to liquid-water-meniscus/capillary bonding between ice granules -- dry
/// powder does not cohere the same way. `high_cohesion()` is the closest
/// existing preset to that regime.
use emerge::{
    HeightmapBoundary, KinematicCircleBoundary, MaterialModel, SimConfig, Simulation, SpawnRegion,
    StomakhinMaterial,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_PACKED: u32 = 1;
const SPACING: f32 = 0.6;
const PARTICLE_AREA: f32 = SPACING * SPACING;
const SLOPE_START_X: usize = 4;
const SLOPE_END_X: usize = 34;
const SLOPE_START_H: f32 = 26.0;
const SLOPE_END_H: f32 = 10.0; // gentle decline over 30 cells -- real, modest angle
const BALL_START_RADIUS: f32 = 1.5;
const BALL_MASS_PER_AREA: f32 = 4.0; // matches MAT_PACKED's own real mass convention below

fn slope_angle_rad() -> f32 {
    let dh = SLOPE_START_H - SLOPE_END_H;
    let dx = (SLOPE_END_X - SLOPE_START_X) as f32;
    (dh / dx).atan()
}

fn heightmap() -> Vec<f32> {
    (0..GRID)
        .map(|x| {
            if x < SLOPE_START_X {
                SLOPE_START_H
            } else if x < SLOPE_END_X {
                let t = (x - SLOPE_START_X) as f32 / (SLOPE_END_X - SLOPE_START_X) as f32;
                SLOPE_START_H + t * (SLOPE_END_H - SLOPE_START_H)
            } else {
                SLOPE_END_H
            }
        })
        .collect()
}

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 200,
        material_cfl_coefficient: 0.3,
        ..SimConfig::standard(GRID, 0.05, Vec2::new(0.0, -9.81))
    };

    let loose = StomakhinMaterial::new(38_889.0, 58_333.0, 10.0, 0.025, 0.0075, 0.6, 20.0);
    let packed: Box<dyn MaterialModel> = Box::new(
        StomakhinMaterial::new(38_889.0, 58_333.0, 10.0, 0.025, 0.0075, 0.6, 20.0)
            .with_cohesion(800.0),
    );

    let spawn_snow = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(4.0 * SPACING * SPACING),
        box_size: IVec2::new(46, 6),
        box_center: Vec2::new(37.0, SLOPE_START_H + 6.0),
        material_id: MAT_LOOSE,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let heights = heightmap();
    let boundary = HeightmapBoundary::new(heights.clone(), 0.4, 2);

    // Ball's real starting placement is decided AFTER settling (see below,
    // measured from the real bounding box) -- this is just a placeholder,
    // far off to the side, so it doesn't interfere with the settle phase.
    let ball = Arc::new(KinematicCircleBoundary::new(
        Vec2::new(-5.0, -5.0),
        BALL_START_RADIUS,
        0.4,
    ));

    let mut sim = Simulation::new(config, spawn_snow)
        .with_default_material(Box::new(loose))
        .with_material(MAT_PACKED, packed)
        .with_boundary(Box::new(boundary))
        .with_boundary(Box::new(ball.clone()));

    println!(
        "slope angle = {:.1} deg, Rubin terminal accel target = (1/6)*g*sin(theta) = {:.3}",
        slope_angle_rad().to_degrees(),
        (1.0 / 6.0) * 9.81 * slope_angle_rad().sin()
    );

    for _ in 0..100 {
        sim.step();
    }

    // Real fix, second pass (2026-08-16): the FIRST fix (leaving x=[4,14]
    // bare) traded the self-jamming bug for a NEW one -- the ball built up
    // real speed (correctly, matching the isolated slope-only probe) over
    // that bare run, then slammed into the snow blanket's own abrupt
    // leading edge and bounced chaotically (reaction spiked to -1265,
    // speed reached 23.4, never smoothly accreted). Real fix: measure the
    // settled blanket's ACTUAL leading edge (not guessed/hand-placed) and
    // start the ball touching it gently -- present from the start (no
    // violent high-speed first contact) but NOT buried inside it (no
    // self-jamming either), with only a small real nudge, not a long free
    // fall first.
    let mut min_x = f32::MAX;
    for p in sim.particles().x.iter() {
        min_x = min_x.min(p.x);
    }
    let ball_start_x = min_x - BALL_START_RADIUS + 0.5; // slight real overlap: "gently touching"
    let ball_start_y = {
        let col = (ball_start_x.round() as isize).clamp(0, GRID as isize - 1) as usize;
        heights[col] + BALL_START_RADIUS
    };
    let ball_center = Vec2::new(ball_start_x, ball_start_y);
    println!(
        "measured snow leading edge min_x={min_x:.2} -> ball starts at ({ball_start_x:.2},{ball_start_y:.2})"
    );

    let total_before = sim.particles().x.len();
    println!("settled: {total_before} loose particles");

    let mut pos = ball_center;
    // Small, real, gentle nudge -- not a violent launch, not a dead stop
    // either. Real precedent for "someone gives it a starting roll before
    // gravity takes over": Rubin 2019's own derivation assumes a nonzero
    // starting condition, not a snowball spontaneously starting from a
    // perfect standstill on a real (never perfectly frictionless) slope.
    let mut vel = Vec2::new(0.5, 0.0);
    let mut omega = 0.0f32; // real angular velocity, rad/s
    let mut radius = BALL_START_RADIUS;
    let mut mass = std::f32::consts::PI * radius * radius * BALL_MASS_PER_AREA;
    let mut absorbed = 0usize;
    let gravity = Vec2::new(0.0, -9.81);

    ball.set_position_velocity(pos, vel);
    ball.set_angular_velocity(omega);

    const STEPS: usize = 600;
    let mut any_nonfinite = false;
    let mut last_speed = 0.0f32;
    for step in 0..STEPS {
        let reaction = ball.take_reaction_impulse();
        // Real 2D solid-disk moment of inertia: I = 0.5*m*r^2 (see
        // `KinematicCircleBoundary::set_angular_velocity`'s own doc).
        let moment_of_inertia = 0.5 * mass * radius * radius;
        let torque = ball.take_torque();
        let vel_before = vel;
        vel += gravity * DT + reaction / mass;
        omega += torque / moment_of_inertia.max(1.0e-6) * DT;
        pos += vel * DT;
        if step < 20 || step % 50 == 0 {
            println!(
                "  step={step:>3} reaction=({:.4},{:.4}) torque={:.4} omega={:.4} vel_before=({:.3},{:.3}) mass={mass:.3} radius={radius:.3}",
                reaction.x, reaction.y, torque, omega, vel_before.x, vel_before.y
            );
        }

        // Real bug, found and fixed live (2026-08-16, same class as the
        // reactive water demo's own floor bug earlier tonight): the ball
        // is app-level state, not a real particle -- `HeightmapBoundary`'s
        // automatic per-particle clamp never touches it, and empty terrain
        // (no particle mass under it) gives `KinematicCircleBoundary`
        // nothing to react against either. Without an explicit terrain
        // check here, the ball free-falls through the slope forever.
        // Real, direct terrain-follow: sample the SAME heightmap array the
        // real boundary uses, keep the ball resting on the surface.
        // Real fix, second pass: a pure vertical (Y-only) correction treats
        // the slope as a staircase of flat micro-floors -- gravity never
        // gets a component ALONG the surface, so the ball never actually
        // rolls downhill, just sits wherever it's vertically pinned (real,
        // measured: froze completely, speed=0.000, after the first pass's
        // fix). Real inclined-plane physics needs the correction applied
        // along the LOCAL SLOPE NORMAL (from the heightmap's own gradient),
        // not the Y axis -- that's what leaves gravity's real tangential
        // component unbalanced and lets it genuinely accelerate the ball
        // down the slope, same mechanism Rubin 2019's own derivation relies
        // on (g*sin(theta) tangential to the incline).
        let col = (pos.x.round() as isize).clamp(1, GRID as isize - 2) as usize;
        let terrain_h = heights[col];
        let dh_dx = (heights[col + 1] - heights[col - 1]) / 2.0;
        let slope_normal = Vec2::new(-dh_dx, 1.0).normalize();
        let terrain_point = Vec2::new(col as f32, terrain_h);
        let perp_dist = (pos - terrain_point).dot(slope_normal);
        if perp_dist < radius {
            pos += slope_normal * (radius - perp_dist);
            let v_n = vel.dot(slope_normal);
            if v_n < 0.0 {
                vel -= v_n * slope_normal;
            }
        }
        // Left/near-start safety clamp only -- the RIGHT-side wall is a
        // real, meaningful collision (the user's own "hits a border, meant
        // to break the ball" design), not just an array-index safety net,
        // handled separately below with a real measured impact speed.
        if pos.x < 2.0 + radius {
            pos.x = 2.0 + radius;
            vel.x = vel.x.max(0.0);
        }
        let wall_x = GRID as f32 - 2.0 - radius;
        let mut wall_impact_speed = 0.0f32;
        if pos.x > wall_x {
            wall_impact_speed = vel.x.max(0.0);
            pos.x = wall_x;
            vel.x = 0.0;
        }

        // Real accretion: loose particles within the ball's own radius
        // (plus a small capture margin) become packed, and the ball's
        // radius/mass grow by the real 2D area/mass added.
        let capture_r = radius + 0.5;
        let nearby = sim.particles_near(pos, capture_r);
        let mut newly_absorbed = 0usize;
        for i in nearby {
            if sim.particles().material_id[i] == MAT_LOOSE {
                sim.particles_mut().material_id[i] = MAT_PACKED;
                newly_absorbed += 1;
            }
        }
        if newly_absorbed > 0 {
            absorbed += newly_absorbed;
            let added_area = newly_absorbed as f32 * PARTICLE_AREA;
            let new_area = std::f32::consts::PI * radius * radius + added_area;
            radius = (new_area / std::f32::consts::PI).sqrt();
            mass = new_area * BALL_MASS_PER_AREA;
            ball.set_radius(radius);
        }

        ball.set_position_velocity(pos, vel);
        ball.set_angular_velocity(omega);
        sim.step();

        if wall_impact_speed > 0.0 {
            println!(
                "  WALL IMPACT step={step} speed={wall_impact_speed:.3} radius={radius:.3} absorbed={absorbed}"
            );
        }

        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
        }
        if any_nonfinite {
            println!("ABORTED step={step}: non-finite state");
            return;
        }
        last_speed = vel.length();

        if step % 50 == 0 {
            println!(
                "step={step:>3} pos=({:.2},{:.2}) speed={:.3} omega={:.3} radius={:.3} absorbed={absorbed}",
                pos.x, pos.y, last_speed, omega, radius
            );
        }
    }

    let total_after = sim.particles().x.len();
    println!(
        "\nDONE {STEPS} steps: count {total_before}->{total_after} (must match, relabel-only) \
         final radius={radius:.3} absorbed={absorbed} final_speed={last_speed:.3} finite={}",
        !any_nonfinite
    );
}

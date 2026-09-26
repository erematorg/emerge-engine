//! The rod <-> MPM grid coupling in `Simulation::do_substep`: genuine two-way momentum
//! exchange through the shared grid, not parallel plumbing that happens to compile.

extern crate emerge_engine as emerge;

#[path = "common/mod.rs"]
mod common;

use emerge::rod::{RodMaterial, build_straight_rod};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

fn zero_gravity_config(grid_res: usize) -> SimConfig {
    common::zero_gravity_config(grid_res, 0.02)
}

/// Total linear momentum: particles + rods summed together.
fn total_momentum(solver: &Simulation) -> Vec2 {
    let mut p = Vec2::ZERO;
    for particle in solver.particles().iter() {
        p += particle.mass * particle.v;
    }
    for rod in solver.rods() {
        for i in 0..rod.points.len() {
            p += rod.points.mass[i] * rod.points.v[i];
        }
    }
    p
}

/// Zero gravity, zero pinning, no wind: a small MPM block and an unpinned straight rod
/// each get initial velocity and are left alone. Total momentum (particles + rod) must
/// stay near its initial value, same weak-conservation contract ordinary MPM particles
/// already have.
#[test]
fn rod_and_particles_momentum_conserved_zero_gravity() {
    let config = zero_gravity_config(48);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(16.0, 24.0),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));
    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            particles.v[i] = Vec2::new(0.3, 0.0);
        }
    }

    // Unpinned rod, far from the particle block, straight (zero rest
    // curvature) so it starts with zero internal strain -- translates as a
    // free rigid body under its own initial velocity, no gravity, no wind.
    let mut rod_points =
        build_straight_rod(Vec2::new(30.0, 24.0), Vec2::new(30.0, 34.0), 8, 0.2, 1.0);
    for v in rod_points.v.iter_mut() {
        *v = Vec2::new(-0.2, 0.1);
    }
    let material = RodMaterial::from_young_modulus_rectangular(1.0e6, 0.05, 0.02, 50.0, 5.0);
    solver.add_rod(emerge::rod::Rod::new(rod_points, material));

    let p0 = total_momentum(&solver);
    solver.step_n(80);
    let p1 = total_momentum(&solver);

    let drift = (p1 - p0).length();
    assert!(
        drift < 0.05 * p0.length().max(1.0),
        "combined particle+rod momentum drifted too much: p0={p0:?} p1={p1:?} drift={drift:.4}"
    );
}

/// Builds a cantilever rod (clamped at points 0-1, sticking out horizontally)
/// inside a real gravity scene, optionally spawning an MPM particle block
/// above the tip so it falls onto the rod. Returns (final average particle
/// height, final rod tip y).
fn cantilever_with_optional_particles(with_particles: bool, steps: usize) -> (Option<f32>, f32) {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 200,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    // span must stay small: self-weight tip deflection ~ L^4/EI, so a much longer span at
    // this stiffness folds nearly flat (not a coupling bug, just genuine overload). 2.0
    // hand-checks to ~8% of span, a visible sag without collapse.
    let span = 2.0;
    let n_points = 12usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(8.0, 24.0),
        Vec2::new(8.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 3.0e6 * 0.06 * 0.02;
    let ei = 3.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        3.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    solver.add_rod(emerge::rod::Rod::new(rod_points, material));

    if with_particles {
        let particle_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(1, 1),
            box_center: Vec2::new(9.0, 25.5),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(particle_spawn);
    }

    solver.step_n(steps);

    let rod_tip_y = solver.rods()[0].points.x.last().unwrap().y;
    let avg_particle_y = if with_particles {
        let particles = solver.particles();
        let n = particles.len();
        if n == 0 {
            None
        } else {
            Some(particles.iter().map(|p| p.x.y).sum::<f32>() / n as f32)
        }
    } else {
        None
    };
    (avg_particle_y, rod_tip_y)
}

#[test]
fn rod_deflects_and_mpm_particles_feel_reaction() {
    // Baseline: rod alone, no particles -- self-weight-only sag.
    let (_, tip_y_alone) = cantilever_with_optional_particles(false, 4000);
    // Same rod, with an MPM particle block dropped onto its tip.
    let (avg_particle_y, tip_y_loaded) = cantilever_with_optional_particles(true, 4000);

    let avg_particle_y = avg_particle_y.expect("particles must still exist");
    // Particles spawned above the rod (y=25.5, rod at y=24) should rest near that height,
    // not fall through -- proves the rod exerts a real reaction force.
    assert!(
        avg_particle_y > 20.0,
        "particles fell through the rod instead of resting on it: avg_y={avg_particle_y:.3}"
    );
    // Rod should sag more with particles resting on it than under self-weight alone.
    assert!(
        tip_y_loaded < tip_y_alone - 0.05,
        "rod tip didn't sag further under particle load: alone={tip_y_alone:.4} loaded={tip_y_loaded:.4}"
    );
}

/// `rod_cfl_dt` depends only on stiffness/mass/geometry, not velocity, so a
/// settled-but-awake rod keeps clamping every substep to its tiny CFL bound forever --
/// the per-rod cost `Rod::sleeping`/`rod_sleep_threshold` removes. Critically damped so
/// it settles fast.
fn settled_cantilever(rod_sleep_threshold: f32, steps: usize) -> Simulation {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 400,
        min_dt: 0.0005,
        rod_sleep_threshold,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));
    let span = 2.0;
    let n_points = 12usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(8.0, 24.0),
        Vec2::new(8.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 3.0e6 * 0.06 * 0.02;
    let ei = 3.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        3.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    solver.add_rod(emerge::rod::Rod::new(rod_points, material));
    solver.step_n(steps);
    solver
}

#[test]
fn sleeping_rod_stops_dominating_the_cfl_bound() {
    let mut awake = settled_cantilever(0.0, 3000); // sleep disabled -- baseline
    let mut asleep = settled_cantilever(0.02, 3000); // same settle, sleep enabled

    assert!(
        asleep.rods()[0].sleeping,
        "critically-damped rod should have settled below rod_sleep_threshold by step 3000"
    );

    awake.step();
    asleep.step();

    assert!(
        asleep.last_substeps() < awake.last_substeps(),
        "sleeping rod should stop dominating the CFL bound: awake={} asleep={}",
        awake.last_substeps(),
        asleep.last_substeps()
    );
    assert_eq!(
        asleep.last_substeps(),
        1,
        "with the only stiff body asleep, dt should reach the full frame dt in one substep"
    );
}

#[test]
fn sleeping_rod_wakes_on_new_contact_and_still_reacts() {
    // Settle with sleep enabled so it's genuinely asleep before contact.
    let mut solver = settled_cantilever(0.02, 3000);
    assert!(
        solver.rods()[0].sleeping,
        "rod should be asleep before the drop"
    );
    let tip_y_before = solver.rods()[0].points.x.last().unwrap().y;

    // Drop a real MPM particle block onto the sleeping rod's tip.
    let particle_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(1, 1),
        box_center: Vec2::new(9.0, 25.5),
        ..SpawnRegion::for_sim(solver.config())
    };
    let _ = solver.add_body(particle_spawn);

    solver.step_n(1500);

    assert!(
        !solver.rods()[0].sleeping,
        "rod should have woken on contact with the falling particle block"
    );
    let tip_y_after = solver.rods()[0].points.x.last().unwrap().y;
    assert!(
        tip_y_after < tip_y_before - 0.02,
        "woken rod didn't sag further under the new particle load: before={tip_y_before:.4} after={tip_y_after:.4}"
    );
}

/// Grass-field scaling: many non-interacting rods should all sleep independently once
/// settled, so ongoing cost is bounded by how many are currently interacted with, not
/// field size. Also checks waking one doesn't cascade to the rest.
#[test]
fn many_rods_all_settle_and_sleep_independently() {
    const N_RODS: usize = 6;
    let config = SimConfig {
        grid_res: 64,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 400,
        min_dt: 0.0005,
        rod_sleep_threshold: 0.02,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let span = 2.0;
    let n_points = 12usize;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 3.0e6 * 0.06 * 0.02;
    let ei = 3.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        3.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );

    // Spaced 8 grid-units apart -- clear of each other's 3x3 kernel support, so they
    // never exchange momentum through the shared grid.
    for k in 0..N_RODS {
        let x0 = 8.0 + 8.0 * k as f32;
        let mut rod_points = build_straight_rod(
            Vec2::new(x0, 24.0),
            Vec2::new(x0 + span, 24.0),
            n_points,
            0.3,
            1.0,
        );
        rod_points.pinned[0] = 1;
        rod_points.pinned[1] = 1;
        solver.add_rod(emerge::rod::Rod::new(rod_points, material));
    }

    solver.step_n(3000);

    let asleep_count = solver.rods().iter().filter(|r| r.sleeping).count();
    assert_eq!(
        asleep_count, N_RODS,
        "every non-interacting rod should independently settle and sleep, got {asleep_count}/{N_RODS}"
    );

    // Wake exactly one via an active push -- the rest must stay asleep (no cross-rod
    // wake cascade).
    solver.rods_mut()[0].push_center = Some(Vec2::new(9.0, 24.0));
    solver.rods_mut()[0].push_strength = 500.0;
    solver.rods_mut()[0].push_radius = 2.0;
    solver.step();

    assert!(!solver.rods()[0].sleeping, "pushed rod should have woken");
    let still_asleep = solver.rods()[1..].iter().filter(|r| r.sleeping).count();
    assert_eq!(
        still_asleep,
        N_RODS - 1,
        "waking one rod must not wake the rest of an unrelated field"
    );
}

/// `scatter_rod_to_grid`'s coverage-gap fix (Guo et al. 2018 SIGGRAPH quadrature-point
/// pattern): a 3-point rod with 6.0-grid-unit segments (past the kernel's 1.5-cell
/// support radius) is pinned rigid, and a particle is dropped at the segment midpoint --
/// the worst-case point farthest from both rod points -- to check it still gets
/// deposited into and rests on the rod instead of free-falling through the gap.
fn drop_particle_through_gap(with_rod: bool, steps: usize) -> f32 {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 200,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    if with_rod {
        // base=(10,20), middle=(16,20), tip=(22,20) -- 6.0-unit segments,
        // 4x past COVERAGE_SPACING (1.5). All three pinned: a static,
        // unmoving "beam" isolates the coverage-gap question from rod
        // dynamics entirely.
        let mut rod_points =
            build_straight_rod(Vec2::new(10.0, 20.0), Vec2::new(22.0, 20.0), 3, 0.3, 1.0);
        for p in rod_points.pinned.iter_mut() {
            *p = 1;
        }
        let material = RodMaterial::from_young_modulus_rectangular(3.0e6, 0.06, 0.02, 0.0, 0.0);
        solver.add_rod(emerge::rod::Rod::new(rod_points, material));
    }

    // Spawn directly above (13.0, 20.0) -- the exact midpoint of the
    // base-middle segment, the single farthest point from both real rod
    // control points.
    let particle_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(1, 1),
        box_center: Vec2::new(13.0, 30.0),
        ..SpawnRegion::for_sim(&config)
    };
    let _ = solver.add_body(particle_spawn);

    solver.step_n(steps);

    let particles = solver.particles();
    particles.iter().map(|p| p.x.y).sum::<f32>() / particles.len() as f32
}

#[test]
fn coverage_gap_fix_catches_particle_falling_through_sparse_rod_midpoint() {
    let free_fall_y = drop_particle_through_gap(false, 600);
    let with_rod_y = drop_particle_through_gap(true, 600);

    // Negative control: without the rod the particle should fall well past y=20,
    // confirming this is a real fall and not a scene that stops there anyway.
    assert!(
        free_fall_y < 10.0,
        "negative control didn't actually fall far -- test geometry is wrong: free_fall_y={free_fall_y:.3}"
    );
    // With the rod, the particle dropped at the worst-case gap midpoint should still
    // rest near y=20 instead of falling through.
    assert!(
        with_rod_y > 15.0,
        "particle fell through the gap between sparse rod points instead of resting on it: \
         with_rod_y={with_rod_y:.3} (free-fall baseline was {free_fall_y:.3})"
    );
}

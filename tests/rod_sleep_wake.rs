//! Real regression coverage (2026-07-29) for a genuine bug found live: a
//! sleeping rod given only `wind_velocity` (no push) never woke up, in
//! either integration path -- confirmed directly from the running
//! `rod_blade_of_grass_gui` demo's own NDJSON log (`wind_on:1` sustained for
//! 5800+ frames, zero movement) before the fix, and by the user re-toggling
//! wind live after the fix (tip speed climbing frame over frame). This test
//! exercises the same real `Simulation::step()` code path the demo does,
//! not a hand-rolled repro, for both `use_implicit_integration` states since
//! the bug existed independently in both wake-check sites
//! (`src/spacetime/solver/step.rs`).

extern crate emerge_engine as emerge;
use emerge::rod::{Rod, RodMaterial, build_straight_rod};
use emerge::{SimConfig, Simulation};
use glam::Vec2;

fn settled_sleeping_rod(use_implicit: bool) -> Simulation {
    let config = SimConfig::standard(32, 0.02, Vec2::new(0.0, -0.05));
    let material = RodMaterial::from_young_modulus_rectangular(1.0e6, 0.02, 0.01, 50.0, 5.0);
    let points = build_straight_rod(Vec2::new(16.0, 16.0), Vec2::new(16.0, 20.0), 5, 0.1, 1.0);
    let mut rod = Rod::new(points, material);
    rod.use_implicit_integration = use_implicit;
    // Force the rod directly into the sleeping state a genuinely settled
    // blade would reach on its own -- skips the real (but here irrelevant)
    // settle time, same shortcut this file's own sibling tests use for
    // sleep-adjacent behavior.
    rod.sleeping = true;

    let mut sim = Simulation::empty(config).with_rod(rod);
    sim.step();
    assert!(
        sim.rods()[0].sleeping,
        "test setup invalid: rod should still be asleep with zero wind/push"
    );
    sim
}

#[test]
fn sleeping_rod_wakes_from_wind_alone_implicit_path() {
    let mut sim = settled_sleeping_rod(true);
    sim.rods_mut()[0].wind_velocity = Vec2::new(5.0, 0.0);
    sim.rods_mut()[0].wind_drag_coeff = 1.0;

    for _ in 0..5 {
        sim.step();
    }

    assert!(
        !sim.rods()[0].sleeping,
        "a rod given nonzero wind_velocity alone must wake up (implicit integration path)"
    );
    let max_v = sim.rods()[0]
        .points
        .v
        .iter()
        .fold(0.0f32, |m, v| m.max(v.length()));
    assert!(
        max_v > 1.0e-4,
        "a woken rod under real wind drag must show real nonzero velocity, got {max_v}"
    );
}

#[test]
fn sleeping_rod_wakes_from_wind_alone_explicit_path() {
    let mut sim = settled_sleeping_rod(false);
    sim.rods_mut()[0].wind_velocity = Vec2::new(5.0, 0.0);
    sim.rods_mut()[0].wind_drag_coeff = 1.0;

    for _ in 0..5 {
        sim.step();
    }

    assert!(
        !sim.rods()[0].sleeping,
        "a rod given nonzero wind_velocity alone must wake up (explicit integration path)"
    );
    let max_v = sim.rods()[0]
        .points
        .v
        .iter()
        .fold(0.0f32, |m, v| m.max(v.length()));
    assert!(
        max_v > 1.0e-4,
        "a woken rod under real wind drag must show real nonzero velocity, got {max_v}"
    );
}

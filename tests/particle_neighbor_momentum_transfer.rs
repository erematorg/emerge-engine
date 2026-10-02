//! When two ordinary MPM blocks meet, does the one at rest react? For DEM
//! grains this is checked in `tests/grains_grid_coupling.rs`
//! (`spinning_grain_...`/`falling_grain_impact_...`); this file checks the
//! continuum MPM path, which has no pairwise contact springs: momentum moves
//! between particles only through the shared grid (P2G -> grid stress/update
//! -> G2P), through the material's stress response to local compression.

extern crate emerge_engine as emerge;
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

/// Two small NeoHookean blocks, zero gravity (isolates the collision effect
/// from settling): block A starts moving toward block B, which starts at
/// rest. The blocks start apart, not touching (starting already overlapped
/// injects an unrelated initial-condition violation); A closes the gap under
/// its given velocity and the two make contact through the shared grid.
#[test]
fn moving_block_makes_a_resting_neighbor_block_react_through_the_grid() {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        boundary_thickness: 2,
        ..SimConfig::default()
    };

    // Block A: 4x4 particles, centered at x=14, moving right at v=2.0.
    let spawn_a = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(14.0, 24.0),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_a)
        .with_default_material(Box::new(NeoHookeanMaterial::new(2.0e4, 4.0e4)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            particles.v[i] = Vec2::new(2.0, 0.0);
        }
    }

    // Block B: 4x4 particles, centered at x=24 -- a real 8-unit gap from A's
    // own center (A's own half-width is 2 cells at spacing=0.5*4/2=1, so the
    // real initial edge-to-edge gap is comfortably nonzero, not overlapping).
    let spawn_b = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(24.0, 24.0),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let tag_b = solver.add_body(spawn_b);

    let mut max_speed_b = 0.0f32;
    for step in 0..400u32 {
        solver.step();
        let avg_speed = solver.group_state(tag_b).avg_speed;
        max_speed_b = max_speed_b.max(avg_speed);
        if step % 40 == 0 {
            println!(
                "step={step} block_b avg_speed={avg_speed:.4} max_speed_b_so_far={max_speed_b:.4}"
            );
        }
    }

    println!("FINAL: block B max centroid speed reached over the run = {max_speed_b:.4}");
    assert!(
        max_speed_b > 0.05,
        "block B started completely at rest and block A (v=2.0) closed a real gap onto it \
         through {} steps of the real grid-coupled Simulation::step() pipeline, but B's \
         centroid speed never rose meaningfully above zero (max={max_speed_b:.6}) -- real, \
         direct evidence that stress/momentum is not propagating from a moving block to a \
         resting neighbor through the shared MPM grid",
        400
    );
}

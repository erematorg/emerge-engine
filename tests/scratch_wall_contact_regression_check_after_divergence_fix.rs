//! Regression check written for the reverted divergence-RHS change in
//! `Grid::project_fluid_incompressibility` (`velocity_at_or_extrapolated`
//! instead of a hard-zero `velocity_at`, see `pressure.rs`): a water column
//! starting ~2 cells from a wall, the scene of
//! `examples/cpu/fluid_pressure_projection.rs`.
//!
//! It asserts only that no particle state goes non-finite or out of bounds.
//! The current code passes, but J sits at the [0.5, 2.0] safety clamp from
//! about frame 20: "stable" here means no NaN, not a valid incompressible
//! run. See the pressure projection entry in `KNOWN_LIMITATIONS.md`.

extern crate emerge_engine as emerge;

use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn wall_touching_column_stays_stable_after_divergence_fix() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        // The demo this mirrors (`examples/cpu/fluid_pressure_projection.rs`)
        // starts at `gravity_fraction: 0.003` of `SimConfig::earth`'s
        // gravity; its 120-frame runs were never done at full gravity.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // Exact real geometry from `fluid_pressure_projection.rs`'s own proven
    // scene: water starting ~2 cells from the left wall, spanning nearly
    // the full grid height.
    const BOX: IVec2 = IVec2::new(14, 52);
    let center = Vec2::new(11.0, 30.0);

    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    const SPACING: f32 = 0.6;
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: BOX,
        box_center: center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let n = 120; // matches the real validated run length
    let wall_start = std::time::Instant::now();
    for step in 1..=n {
        solver.step();
        let snap = solver.diagnostics_snapshot();
        if step % 20 == 0 || step == 1 || step == n {
            println!(
                "step {step:3}: non_finite={} oob={} max_speed={:8.3} J=[{:.4},{:.4}]",
                snap.non_finite_particle_values,
                snap.out_of_bounds_particles,
                snap.max_particle_speed,
                snap.min_deformation_j,
                snap.max_deformation_j,
            );
        }
        assert_eq!(
            snap.non_finite_particle_values, 0,
            "step {step}: non-finite -- the divergence fix broke this real, previously-stable scene"
        );
        assert_eq!(
            snap.out_of_bounds_particles, 0,
            "step {step}: particle left the grid"
        );
    }
    let elapsed = wall_start.elapsed();
    println!(
        "wall_time={:.2}s  ({:.2}fps over {n} frames)",
        elapsed.as_secs_f64(),
        n as f64 / elapsed.as_secs_f64()
    );
}

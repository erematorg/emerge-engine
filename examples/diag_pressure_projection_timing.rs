//! Temporary diagnostic (regional-substepping plan, Part B "Step 0"):
//! real, current per-phase timing breakdown for the CPU pressure-
//! projection fluid path, using the engine's own existing `StepTiming`
//! instrumentation (`pressure_us`, `retry_snapshot_us` -- both already
//! exist, see `src/systems/diagnostics/snapshot.rs`). SAME hard scene as
//! `fluid_pressure_projection_gui.rs`'s own `make_sim()` (water column
//! ~2 cells from the left wall, mud block, GRID=64) -- the exact scene
//! the original "63% in pressure solve" number came from, now re-measured
//! post the retry/backstop fixes that number predates. Run with
//! --release; debug-mode timing is noise.
extern crate emerge_engine as emerge;

#[path = "diag_common/mod.rs"]
mod diag_common;

use emerge::{
    BinghamFluidMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400, // verify the value now shipped in fluid_pressure_projection_gui.rs
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 0.0, 3.0, 4.0);
    const WATER_MASS: f32 = 0.1 * 0.6 * 0.6;
    const MUD_MASS: f32 = 4.0 * 0.6 * 0.6;
    let spawn_water = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(11.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mud = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(16, 18),
        box_center: Vec2::new(50.0, 38.0),
        material_id: MAT_MUD,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(MUD_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_MUD, Box::new(mud))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn_mud);
    solver
}

fn main() {
    let mut solver = make_sim();
    // n=120 matches fluid_pressure_projection_gui's own verified 120-frame run.
    diag_common::run_and_report_timing(&mut solver, 120, Some(20));
}

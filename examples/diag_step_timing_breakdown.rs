//! Temporary diagnostic: real per-phase CPU step() timing breakdown at a
//! realistic particle count, using the engine's own existing `StepTiming`
//! instrumentation (no new profiler needed). Run with --release; debug-mode
//! timing is noise (see perf_opportunities_survey memory's own false-alarm
//! entries).
extern crate emerge_engine as emerge;

#[path = "diag_common/mod.rs"]
mod diag_common;

use emerge::prelude::*;
use glam::Vec2;

fn main() {
    let config = SimConfig::standard(96, 0.001, Vec2::NEG_Y * 9.8);
    let spawn = SpawnRegion::for_sim(&config)
        .at(Vec2::new(48.0, 48.0))
        .box_of(glam::IVec2::new(70, 70))
        .spacing(0.5)
        .material(0);
    let mut solver = Simulation::new(config, spawn)
        .with_material(0, Box::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.3)));

    // Warm up (settle past initial transients).
    for _ in 0..30 {
        solver.step();
    }

    diag_common::run_and_report_timing(&mut solver, 60, None);
}

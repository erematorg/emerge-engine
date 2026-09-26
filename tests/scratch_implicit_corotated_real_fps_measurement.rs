//! Real wall-clock measurement (2026-09-11), NOT the standalone synthetic
//! benchmark from the night before (`stage3_dp_multi_particle_real_wall_
//! clock_speedup_vs_real_explicit`, 10.3x) -- that number predates every
//! correctness fix landed since (Kirchhoff-vs-Piola, mass-normalized
//! tolerance, and critically the Gershgorin/PSD Hessian construction, which
//! adds 4 extra JVP evaluations per particle per CG iteration). This
//! measures the REAL production code path (`Simulation::step` with
//! `SimConfig::implicit_corotated_elastic`) at basic_sand's own real scale
//! (~2016 particles, per project memory), for the regime that's actually
//! verified correct: an already-settled pile, not a violent drop.
//!
//! `cargo test --release --test scratch_implicit_corotated_real_fps_measurement -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::DruckerPragerMaterial;
use emerge::{SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::time::Instant;

const GRID: usize = 64;
const MAT_LOOSE: u32 = 1;

fn make_sand(phi_deg: f32) -> DruckerPragerMaterial {
    // Same real SI stiffness as basic_sand.rs itself (E=15MPa, nu=0.3),
    // not a toy value -- the whole point is measuring the REAL scene's cost.
    let config = SimConfig::earth(GRID, 0.01, 0.016);
    let (lambda, mu) = config.lame_from_si_physical_cfg(15.0e6, 0.3, 1600.0);
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim(implicit: bool) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 3000,
        material_cfl_coefficient: 0.7,
        implicit_corotated_elastic: implicit,
        ..SimConfig::earth(GRID, 0.01, 0.016)
    };
    let mass_grid = (1600.0 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14), // same real box as basic_sand.rs -> ~1008 particles
        box_center: Vec2::new(32.0, 20.0),
        material_id: MAT_LOOSE,
        position_jitter: 0.5,
        rng_seed: 11,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(make_sand(30.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn real_fps_settled_pile_explicit_vs_implicit() {
    // Settle purely via explicit first -- the real, verified-correct
    // regime. 60 frames is enough for a compact box-spawned pile at this
    // scale to stop actively flowing under its own weight.
    let mut settling = make_sim(false);
    for _ in 0..60 {
        settling.step();
    }
    let n = settling.particles().len();
    let max_v = settling
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    println!("n_particles={n} post-settle max_speed={max_v:.4}");

    // Fork into two fresh sims from the SAME construction, settle both
    // identically via explicit (so the comparison starts from the same
    // real, physically-settled state), then measure real wall-clock
    // Simulation::step() cost for each path from there.
    let mut explicit = make_sim(false);
    let mut implicit = make_sim(false);
    for _ in 0..60 {
        explicit.step();
        implicit.step();
    }
    implicit.config_mut().implicit_corotated_elastic = true;

    const MEASURE_FRAMES: usize = 30;
    let t0 = Instant::now();
    for _ in 0..MEASURE_FRAMES {
        explicit.step();
    }
    let explicit_ms = t0.elapsed().as_secs_f64() * 1000.0 / MEASURE_FRAMES as f64;

    let t1 = Instant::now();
    let mut implicit_engaged_frames = 0usize;
    for _ in 0..MEASURE_FRAMES {
        let x_before = implicit.particles().x[0];
        implicit.step();
        if implicit.particles().x[0] != x_before || implicit.last_substeps() == 1 {
            implicit_engaged_frames += 1;
        }
    }
    let implicit_ms = t1.elapsed().as_secs_f64() * 1000.0 / MEASURE_FRAMES as f64;

    let speedup = explicit_ms / implicit_ms;
    println!(
        "REAL PRODUCTION MEASUREMENT ({n} particles, real E=15MPa DruckerPrager, {MEASURE_FRAMES} frames): \
         explicit={explicit_ms:.4}ms/frame ({:.1} fps)  implicit={implicit_ms:.4}ms/frame ({:.1} fps)  \
         speedup={speedup:.2}x  implicit_engaged_frames={implicit_engaged_frames}/{MEASURE_FRAMES}",
        1000.0 / explicit_ms,
        1000.0 / implicit_ms,
    );

    let drift = (0..n)
        .map(|i| (explicit.particles().x[i] - implicit.particles().x[i]).length())
        .fold(0.0f32, f32::max);
    println!("max position drift after {MEASURE_FRAMES} frames: {drift:.4} grid cells");
}

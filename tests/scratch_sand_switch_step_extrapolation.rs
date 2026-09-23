//! Real, direct extrapolation of `tests/accuracy.rs::post_event_relax_
//! switch_step_long_horizon_convergence_comparison`'s own established
//! trend (switch_step=800/1500/3000 -> 65.6/58.6/50.4deg, monotonically
//! DECREASING with larger switch_step) -- does pushing switch_step further
//! continue toward the real 30-35deg dry-sand target (GH issue #28), or
//! does the trend plateau/reverse before getting there? Same scene, same
//! material, same `measure_pile_shape`/`DT`/`FLOOR` convention as that
//! test (copied exactly, not re-derived), for direct comparability.
//!
//! `cargo test --release --test scratch_sand_switch_step_extrapolation -- --ignored --nocapture`

extern crate emerge_engine as emerge;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const LOCAL_GRID: usize = 128;
const DT: f32 = 0.1;
const FLOOR: f32 = 2.0;

struct PileShape {
    height: f32,
    base_half_width: f32,
    angle_deg: f32,
}

/// Copied verbatim from `tests/accuracy.rs::measure_pile_shape` -- must stay
/// bit-identical so this file's numbers are directly comparable to the
/// switch_step=800/1500/3000 trend it extrapolates.
fn measure_pile_shape(xs: &[Vec2], floor: f32) -> PileShape {
    let n = xs.len() as f32;
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    let height = xs
        .iter()
        .filter(|p| (p.x - center_x).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::MIN, f32::max)
        - floor;
    let base_half_width = xs
        .iter()
        .filter(|p| p.y < floor + 1.5)
        .map(|p| (p.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle_deg = (height / base_half_width.max(0.1)).atan().to_degrees();
    PileShape {
        height,
        base_half_width,
        angle_deg,
    }
}

fn run(switch_step: usize) -> Vec<(usize, f32)> {
    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.6,
        ..SimConfig::standard(LOCAL_GRID, DT, Vec2::new(0.0, -0.3))
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(LOCAL_GRID as f32 * 0.5, FLOOR + 8.0),
        material_id: 0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    sand.post_event_relax_threshold = 0.001;
    let mut solver = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

    solver.step_n(switch_step);
    solver.set_apic_blend(0.05);
    solver.set_cundall_damping(1.0);

    let checkpoints: &[usize] = &[6000, 12000, 25000, 50000, 100000];
    let mut cumulative = 0usize;
    let mut trajectory = Vec::new();
    for &target in checkpoints {
        solver.step_n(target - cumulative);
        cumulative = target;
        let shape = measure_pile_shape(&solver.particles().x.clone(), FLOOR);
        trajectory.push((switch_step + cumulative, shape.angle_deg));
        println!(
            "    switch_step={switch_step:6} total_step={:7} height={:.2} half-w={:.2} angle={:.1} deg",
            switch_step + cumulative,
            shape.height,
            shape.base_half_width,
            shape.angle_deg,
        );
    }
    trajectory
}

#[test]
#[ignore = "real, long-running extrapolation sweep -- run explicitly with --ignored"]
fn switch_step_extrapolation_toward_real_repose_angle() {
    println!(
        "── EXTRAPOLATING THE REAL SWITCH-STEP TREND (800/1500/3000 -> 65.6/58.6/50.4deg) \
         PAST THE ORIGINAL SWEEP'S RANGE ──"
    );
    for &switch_step in &[6000usize, 10000, 20000] {
        println!("  switch_step={switch_step}:");
        run(switch_step);
    }
}

/// Same scene, extended checkpoint schedule (out to 300,000 total steps
/// instead of 100,000) -- real, necessary follow-up: `switch_step=20000`'s
/// own trajectory did NOT settle within the original 100k-step horizon
/// (36.6 -> 35.7 -> 30.3 -> 32.8 -> 38.7deg, still moving at the last
/// checkpoint), which means `switch_step=10000`'s apparent clean plateau
/// (36.7 -> 36.2 -> 36.2deg) could just as easily be a temporary lull that
/// resumes drifting given more time, not a genuine stable equilibrium.
/// Zero randomness in this scene (`position_jitter=0.0` by construction),
/// so this is a real extension of the SAME deterministic trajectory, not a
/// repeat of an already-known result.
fn run_extended(switch_step: usize, checkpoints: &[usize]) -> Vec<(usize, f32)> {
    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.6,
        ..SimConfig::standard(LOCAL_GRID, DT, Vec2::new(0.0, -0.3))
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(LOCAL_GRID as f32 * 0.5, FLOOR + 8.0),
        material_id: 0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    sand.post_event_relax_threshold = 0.001;
    let mut solver = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

    solver.step_n(switch_step);
    solver.set_apic_blend(0.05);
    solver.set_cundall_damping(1.0);

    let mut cumulative = 0usize;
    let mut trajectory = Vec::new();
    for &target in checkpoints {
        solver.step_n(target - cumulative);
        cumulative = target;
        let shape = measure_pile_shape(&solver.particles().x.clone(), FLOOR);
        trajectory.push((switch_step + cumulative, shape.angle_deg));
        println!(
            "    switch_step={switch_step:6} total_step={:7} height={:.2} half-w={:.2} angle={:.1} deg",
            switch_step + cumulative,
            shape.height,
            shape.base_half_width,
            shape.angle_deg,
        );
    }
    trajectory
}

#[test]
#[ignore = "real, long-running verification -- run explicitly with --ignored"]
fn switch_step_10000_holds_at_a_genuinely_long_horizon() {
    println!("── DOES switch_step=10000's PLATEAU (36.2deg) SURVIVE PAST 100,000 STEPS? ──");
    run_extended(10000, &[6000, 12000, 25000, 50000, 100000, 200000, 300000]);
}

/// Real robustness check, not just a single lucky point: if ONLY exactly
/// switch_step=10000 lands near the real target while its immediate
/// neighbors look wildly different, that is a sign of fragility/coincidence
/// (matching this same investigation's own switch_step=20000 finding --
/// non-monotonic, noisy behavior nearby), not a genuine usable region.
#[test]
#[ignore = "real, long-running neighborhood-robustness sweep -- run explicitly with --ignored"]
fn switch_step_neighborhood_around_10000_is_checked_for_robustness() {
    println!("── NEIGHBORHOOD ROBUSTNESS AROUND switch_step=10000 ──");
    for &switch_step in &[8000usize, 9000, 11000, 12000] {
        println!("  switch_step={switch_step}:");
        run(switch_step);
    }
}

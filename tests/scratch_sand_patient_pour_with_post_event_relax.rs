//! Real, direct test of a combination never tried before: the ALREADY-
//! DOCUMENTED "patient pour" recipe (`tests/accuracy.rs::sand_pile_built_
//! by_patient_pour_matching_real_creep_timescale`, real creep timescale
//! between pours, STEPS_BETWEEN_POURS=600 not 15) -- REOPENED 2026-08-20,
//! its own real 30.8deg result found to be a false positive from an
//! unrelated `SlipBoundary`-defeats-`FrictionBoundary` bug; once fixed, the
//! SAME recipe gives 69.1deg, and `apic_blend`/boundary `mu` sweeps were
//! both real, measured, ruled out -- COMBINED with `post_event_relax_
//! threshold` (this session's own separately-validated real fix for
//! column-drop's own "pile never topples" problem), which the patient-pour
//! test never tried. Direct diagnostic (`diag_lateral_motion_during_real_
//! pour`) this session found particles DO yield during a short pour, but
//! only by infinitesimal amounts (q barely moves), consistent with the
//! twentieth finding's own real "no time to develop real creep" diagnosis
//! -- `post_event_relax_threshold`'s own real mechanism (reset elastic
//! strain on the falling edge of straining) could plausibly interact
//! differently with a LONG, patient pour than a short one. Real, honest,
//! not-yet-known outcome -- measuring, not assuming.
//!
//! Identical geometry/config to the real, existing patient-pour test
//! (POUR_GRID=256, STEPS_BETWEEN_POURS=600, N_POURS=70,
//! SETTLE_STEPS_AFTER=4000, DROP_GAP_CELLS=2.0) -- only
//! `post_event_relax_threshold` changes.

extern crate emerge_engine as emerge;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

struct PileShape {
    height: f32,
    base_half_width: f32,
    angle_deg: f32,
}

fn measure_pile_shape(xs: &[Vec2], floor: f32) -> PileShape {
    let n = xs.len() as f32;
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    let height = xs
        .iter()
        .filter(|p| (p.x - center_x).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max)
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

#[test]
#[ignore = "real, long-running combination test -- run explicitly with --release --ignored --nocapture"]
fn patient_pour_with_post_event_relax_threshold() {
    const POUR_GRID: usize = 256;
    const POUR_DT: f32 = 0.016;
    const POUR_FLOOR: f32 = 2.0;
    const N_POURS: usize = 70;
    const STEPS_BETWEEN_POURS: usize = 600;
    const SETTLE_STEPS_AFTER: usize = 4000;
    const DROP_GAP_CELLS: f32 = 2.0;

    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.05,
        cundall_damping: 0.0,
        ..SimConfig::standard(POUR_GRID, POUR_DT, Vec2::new(0.0, -0.3))
    };
    let cx = POUR_GRID as f32 * 0.5;
    let sand = DruckerPragerMaterial {
        post_event_relax_threshold: 0.001,
        ..DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2)
    };

    let seed = SpawnRegion {
        spacing: 0.25,
        box_size: IVec2::new(4, 1),
        box_center: Vec2::new(cx, POUR_FLOOR + 0.5),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, seed)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

    for i in 0..N_POURS {
        let xs_now = &solver.particles().x;
        let surface_y = xs_now
            .iter()
            .filter(|p| (p.x - cx).abs() < 4.0)
            .map(|p| p.y)
            .fold(POUR_FLOOR, f32::max);
        let center_x_now = xs_now.iter().map(|p| p.x).sum::<f32>() / xs_now.len() as f32;
        let spread_now = xs_now
            .iter()
            .map(|p| (p.x - center_x_now).abs())
            .fold(0.0f32, f32::max);
        println!(
            "pour {i:3}: surface_y={surface_y:.2} n={} spread={spread_now:.2}",
            xs_now.len()
        );
        let batch = SpawnRegion {
            spacing: 0.25,
            box_size: IVec2::new(3, 1),
            box_center: Vec2::new(cx, surface_y + DROP_GAP_CELLS),
            material_id: 0,
            precompute_initial_volumes: true,
            rng_seed: 400 + i as u32,
            position_jitter: 0.15,
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(batch);
        solver.step_n(STEPS_BETWEEN_POURS);
    }
    solver.set_cundall_damping(1.0);
    solver.step_n(SETTLE_STEPS_AFTER);

    let xs: Vec<Vec2> = solver.particles().x.clone();
    let shape = measure_pile_shape(&xs, POUR_FLOOR);
    println!("── PATIENT POUR + post_event_relax_threshold ──");
    println!("  {N_POURS} pours, {} particles total", xs.len());
    println!("  final height      = {:.2} cells", shape.height);
    println!("  final base half-w = {:.2} cells", shape.base_half_width);
    println!(
        "  -> final angle     = {:.1} deg  (real target: 30-35deg; without this fix: 69.1deg)",
        shape.angle_deg
    );
}

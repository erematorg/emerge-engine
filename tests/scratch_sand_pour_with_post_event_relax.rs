//! Real, direct test: does `post_event_relax_threshold` (the actual,
//! already-validated real fix for column-drop's own toppling/repose-angle
//! problem, per `project_sand_angle_of_repose_and_mibf_2026-09-13` project
//! memory) ALSO fix the SEPARATE, still-open poured-pile problem
//! (`tests/accuracy.rs::sand_pile_built_by_slow_pour_with_pradhana_
//! correction`, real measured result: 88.7deg non-toppling tower, ~2.25
//! cells/pour growth, UNCHANGED by two separate Pradhana volume-gain fix
//! attempts)? The existing real pour test never sets this field at all
//! (defaults to 0.0, disabled) -- this is the one, real, most direct,
//! not-yet-tried lever for the ACTUAL observed symptom (a pile that never
//! topples sideways), as opposed to the volume-gain mechanism (a real,
//! separate, additive effect, not necessarily the DOMINANT one for why
//! this specific pile builds a vertical tower).
//!
//! Identical geometry/config to the real
//! `sand_pile_built_by_slow_pour_with_pradhana_correction` test (same
//! POUR_GRID/DT/FLOOR/N_POURS/STEPS_BETWEEN_POURS/SETTLE_STEPS_AFTER/
//! DROP_GAP_CELLS) -- only the material's own `post_event_relax_threshold`
//! (and, in the combined case, `use_pradhana`) change, so any real
//! difference is directly attributable.

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

fn run(post_event_relax_threshold: f32, use_pradhana: bool, use_material_friction: bool) {
    const POUR_GRID: usize = 128;
    const POUR_DT: f32 = 0.016;
    const POUR_FLOOR: f32 = 2.0;
    const N_POURS: usize = 45;
    const STEPS_BETWEEN_POURS: usize = 15;
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
        post_event_relax_threshold,
        use_pradhana,
        ..DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2)
    };
    let mut boundary = FrictionBoundary::new(2, 0.7);
    boundary.use_material_friction = use_material_friction;

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
        .with_boundary(Box::new(boundary));

    for i in 0..N_POURS {
        let xs_now = &solver.particles().x;
        let surface_y = xs_now
            .iter()
            .filter(|p| (p.x - cx).abs() < 4.0)
            .map(|p| p.y)
            .fold(POUR_FLOOR, f32::max);
        let batch = SpawnRegion {
            spacing: 0.25,
            box_size: IVec2::new(3, 1),
            box_center: Vec2::new(cx, surface_y + DROP_GAP_CELLS),
            material_id: 0,
            precompute_initial_volumes: true,
            rng_seed: 200 + i as u32,
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
    println!(
        "post_event_relax_threshold={post_event_relax_threshold} use_pradhana={use_pradhana} \
         use_material_friction={use_material_friction}: n={} height={:.2} half-w={:.2} angle={:.1}deg \
         (real target: 30-35deg)",
        xs.len(),
        shape.height,
        shape.base_half_width,
        shape.angle_deg,
    );
}

#[test]
#[ignore = "real, long-running combination sweep -- run explicitly with --release --ignored --nocapture"]
fn pour_with_post_event_relax_and_combinations() {
    println!("── baseline (no fixes, matches already-measured 88.7deg) ──");
    run(0.0, false, false);
    println!("── post_event_relax_threshold alone ──");
    run(0.001, false, false);
    println!("── post_event_relax_threshold + use_pradhana ──");
    run(0.001, true, false);
    println!("── post_event_relax_threshold + MIBF ──");
    run(0.001, false, true);
    println!("── all three combined ──");
    run(0.001, true, true);
}

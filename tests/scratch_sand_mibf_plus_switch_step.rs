//! Real test of MIBF COMBINED with the already-validated `switch_step`
//! Cundall-damping recipe (verified tonight: switch_step=9000/11000 hold
//! stable 32.3-32.4deg plateaus through 100k+ steps). Does adding real,
//! material-derived wall friction on top of that already-working recipe
//! help further, hurt, or leave it unchanged? Same scene as `tests/
//! scratch_sand_switch_step_extrapolation.rs` (copied, not re-derived).
//!
//! `cargo test --release --test scratch_sand_mibf_plus_switch_step -- --nocapture`

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

fn run(switch_step: usize, use_material_friction: bool) -> Vec<(usize, f32)> {
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
    let mut boundary = FrictionBoundary::new(2, 0.7);
    boundary.use_material_friction = use_material_friction;
    let mut solver = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(boundary));

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
            "    switch_step={switch_step:6} mibf={use_material_friction:5} total_step={:7} height={:.2} half-w={:.2} angle={:.1} deg",
            switch_step + cumulative,
            shape.height,
            shape.base_half_width,
            shape.angle_deg,
        );
    }
    trajectory
}

#[test]
#[ignore = "real, long-running combination test -- run explicitly with --ignored"]
fn mibf_combined_with_validated_switch_step_recipe() {
    println!("── MIBF + switch_step COMBINATION (validated switch_step=9000/11000 region) ──");
    for &switch_step in &[9000usize, 11000] {
        println!("  switch_step={switch_step}, mibf=false (baseline):");
        run(switch_step, false);
        println!("  switch_step={switch_step}, mibf=true:");
        run(switch_step, true);
    }
}

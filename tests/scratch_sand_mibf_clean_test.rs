//! Real, clean test of MIBF's own claim (Blatny & Gaume 2025, `tmp/
//! ref_matter.md` sec.19): does making wall friction come from the sand's
//! own real internal state -- instead of `FrictionBoundary`'s fixed 0.7 --
//! move the dynamic-collapse angle of repose toward the real 30-35deg
//! target, WITHOUT any Cundall-damping numerical trick? Same exact scene
//! as `tests/accuracy.rs::sand_angle_of_repose_is_physical` (copied, not
//! re-derived, for direct comparability): GRID=64, DT=0.1, FLOOR=2.0, an
//! 8x16 column, `DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2)`,
//! 1500 steps, no `apic_blend`/`cundall_damping` at all -- that baseline
//! settles at ~12deg.
//!
//! `cargo test --release --test scratch_sand_mibf_clean_test -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const FLOOR: f32 = 2.0;

struct PileShape {
    height: f32,
    base_half_width: f32,
    angle_deg: f32,
}

/// Copied verbatim from `tests/accuracy.rs::measure_pile_shape`.
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

fn run(use_material_friction: bool) -> PileShape {
    let config = SimConfig {
        max_substeps_per_step: 64,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + 8.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    let mut boundary = FrictionBoundary::new(2, 0.7);
    boundary.use_material_friction = use_material_friction;
    let mut solver = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(boundary));

    solver.step_n(1500);
    measure_pile_shape(&solver.particles().x.clone(), FLOOR)
}

#[test]
fn mibf_alone_vs_fixed_friction_baseline_no_cundall_trick() {
    let baseline = run(false);
    let mibf = run(true);

    println!("── MIBF CLEAN TEST (no Cundall/post_event_relax trick at all) ──");
    println!(
        "  baseline (fixed mu=0.7):  height={:.2} half-w={:.2} angle={:.1} deg",
        baseline.height, baseline.base_half_width, baseline.angle_deg
    );
    println!(
        "  MIBF (material friction): height={:.2} half-w={:.2} angle={:.1} deg",
        mibf.height, mibf.base_half_width, mibf.angle_deg
    );
    println!("  real dry sand target: 30-35 deg");
}

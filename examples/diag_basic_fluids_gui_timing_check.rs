extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Honest perf check on `basic_fluids_gui.rs`'s REAL current scene (water-
/// only, mud currently disabled per that file's own 2026-08-13 comment) --
/// before assuming there's a quick win to find, actually measure where the
/// time goes right now, via the engine's own existing `StepTiming`
/// instrumentation (`diagnostics_snapshot().timing`), same tool the
/// perf_opportunities_survey memory's own real wins were found with.
use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 150,
        phase_rules_once_per_step: true,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    const WATER_EOS_POWER: f32 = 3.0;
    const COLUMN_HEIGHT_CELLS: f32 = 52.0 * SPACING;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water = NewtonianFluidMaterial::new(0.1, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

fn main() {
    let mut sim = make_sim();
    println!("{} particles", sim.particles().x.len());

    // Warm up (skip transient spawn-settle spike).
    for _ in 0..30 {
        sim.step();
    }

    let n = 60;
    let mut sums = [0u64; 8];
    for _ in 0..n {
        sim.step();
        let t = sim.diagnostics_snapshot().timing;
        sums[0] += t.p2g_us;
        sums[1] += t.g2p_us;
        sums[2] += t.cfl_us;
        sums[3] += t.grid_update_us;
        sums[4] += t.project_us;
        sums[5] += t.pressure_us;
        sums[6] += t.spatial_hash_us;
        sums[7] += t.total_us;
    }
    let names = [
        "p2g",
        "g2p",
        "cfl",
        "grid_update",
        "project",
        "pressure",
        "spatial_hash",
        "TOTAL",
    ];
    println!("avg over {n} steps (post warmup):");
    for (name, sum) in names.iter().zip(sums.iter()) {
        let avg = *sum as f64 / n as f64;
        let pct = 100.0 * avg / (sums[7] as f64 / n as f64);
        println!("  {name:>14}: {avg:>8.1}us  ({pct:>5.1}%)");
    }
    println!(
        "implied fps (single-threaded step cost only): {:.1}",
        1_000_000.0 / (sums[7] as f64 / n as f64)
    );
}

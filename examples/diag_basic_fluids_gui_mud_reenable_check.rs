extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// `basic_fluids_gui.rs` disabled its mud spawn 2026-08-13 ("isolating to
/// water-only" while a real water/mud interaction bug was investigated).
/// Many real fluid fixes have landed since (this session alone: the
/// contact_group/strict-fluid G2P bug, plus earlier EOS/pressure work).
/// Real check, not assumed: does re-enabling mud now run stably for a
/// real, long-ish horizon, with real conservation, before touching the
/// actual demo file?
use emerge::{
    BinghamFluidMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
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
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 100.0, 3.0, 4.0);

    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    const MUD_MASS: f32 = 4.0 * SPACING * SPACING;
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
    let spawn_mud = SpawnRegion {
        spacing: SPACING,
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
    let mut sim = make_sim();
    let total_before = sim.particles().x.len();
    println!("water+mud re-enabled: {total_before} particles");

    let mut max_speed_ever = 0.0f32;
    let mut any_nonfinite = false;
    const STEPS: usize = 400;
    for step in 0..STEPS {
        sim.step();
        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
            max_speed_ever = max_speed_ever.max(v.length());
        }
        if any_nonfinite {
            println!("ABORTED step={step}: non-finite state");
            return;
        }
        if step % 50 == 0 {
            println!(
                "step={step:>3} max_speed_ever={max_speed_ever:.2} count={}",
                sim.particles().x.len()
            );
        }
    }
    let total_after = sim.particles().x.len();
    println!(
        "DONE {STEPS} steps: count {total_before}->{total_after}  max_speed_ever={max_speed_ever:.2}  finite={}",
        !any_nonfinite
    );
}

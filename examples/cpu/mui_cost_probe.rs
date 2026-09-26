//! Real per-frame cost for the mu(I) rheology scene, headless.
//!
//!   cargo run --release --example mui_cost_probe
extern crate emerge_engine as emerge;

use emerge::{
    Elastic, FromSI, GranularProps, MuIRheologyMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.0003;
const YOUNG_MODULUS_PA: f32 = 15.0e6;
const POISSON_RATIO: f32 = 0.3;
const DENSITY_KG_M3: f32 = 1600.0;
const FRICTION_ANGLE_DEG: f32 = 30.0;
const INERTIAL_Q: [f32; 3] = [5.58, 3.00, 1.12];
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
const COLUMN_CELLS: IVec2 = IVec2::new(10, 22);

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 128,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let elastic = Elastic {
        e_pa: YOUNG_MODULUS_PA,
        nu: POISSON_RATIO,
        rho_kg_m3: DENSITY_KG_M3,
    };
    let props = GranularProps {
        elastic,
        friction_angle_deg: FRICTION_ANGLE_DEG,
        dilatancy_angle_deg: 0.0,
    };
    let build = |q: f32| {
        let mut m = MuIRheologyMaterial::from_physical(&props, &config);
        m.inertial_q = q;
        m
    };
    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(COLUMN_X[slot], 4.0 + COLUMN_CELLS.y as f32 * 0.5),
            material_id: mat,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };

    let mut sim = Simulation::new(config, spawn(0, 0))
        .with_default_material(Box::new(build(INERTIAL_Q[0])))
        .with_material(1, Box::new(build(INERTIAL_Q[1])))
        .with_material(2, Box::new(build(INERTIAL_Q[2])))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1, 1));
    let _ = sim.add_body(spawn(2, 2));

    println!("particles={}", sim.particles().len());
    const FRAMES: usize = 120;
    let mut substeps = 0usize;
    let wall = std::time::Instant::now();
    for _ in 0..FRAMES {
        sim.step();
        substeps += sim.diagnostics_snapshot().substeps_last_step;
    }
    let elapsed_ms = wall.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{FRAMES} frames in {elapsed_ms:.1} ms -> {:.2} ms/frame, {:.1} fps, {:.1} substeps/frame",
        elapsed_ms / FRAMES as f64,
        1000.0 * FRAMES as f64 / elapsed_ms,
        substeps as f64 / FRAMES as f64
    );
}

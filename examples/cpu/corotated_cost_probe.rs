//! Real per-frame cost for the stiff-elastic bounce scene, headless.
extern crate emerge_engine as emerge;

use emerge::{
    CorotatedMaterial, Elastic, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.0005;
const E_PA: [f32; 3] = [5.0e5, 2.0e6, 1.0e7];
const RHO: f32 = 1000.0;
const NU: f32 = 0.3;
const BLOCK_X: [f32; 3] = [14.0, 32.0, 50.0];
const BLOCK_CELLS: IVec2 = IVec2::new(10, 10);

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 128,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let props = |e: f32| Elastic {
        e_pa: e,
        nu: NU,
        rho_kg_m3: RHO,
    };
    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: BLOCK_CELLS,
            box_center: Vec2::new(BLOCK_X[slot], 40.0),
            material_id: mat,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(E_PA[slot]), &config)
    };

    let mut sim = Simulation::new(config, spawn(0, 0))
        .with_default_material(Box::new(CorotatedMaterial::from_physical(
            &props(E_PA[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(CorotatedMaterial::from_physical(&props(E_PA[1]), &config)),
        )
        .with_material(
            2,
            Box::new(CorotatedMaterial::from_physical(&props(E_PA[2]), &config)),
        )
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

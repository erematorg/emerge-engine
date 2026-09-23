//! Real per-frame cost for the viscoelastic damping-comparison scene, headless.
extern crate emerge_engine as emerge;

use emerge::{
    FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion, Viscoelastic, ViscoelasticMaterial,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.0006;
const E_PA: f32 = 2.0e6;
const NU: f32 = 0.45;
const RHO: f32 = 1000.0;
const ETA_PA_S: [f32; 3] = [0.0, 100.0, 1000.0];
const BLOCK_X: [f32; 3] = [14.0, 32.0, 50.0];
const BLOCK_CELLS: IVec2 = IVec2::new(10, 10);

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 128,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let props = |eta: f32| Viscoelastic {
        elastic: emerge::Elastic {
            e_pa: E_PA,
            nu: NU,
            rho_kg_m3: RHO,
        },
        eta_pa_s: eta,
    };
    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: BLOCK_CELLS,
            box_center: Vec2::new(BLOCK_X[slot], 20.0),
            material_id: mat,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(ETA_PA_S[slot]), &config)
    };

    let mut sim = Simulation::new(config, spawn(0, 0))
        .with_default_material(Box::new(ViscoelasticMaterial::from_physical(
            &props(ETA_PA_S[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(ViscoelasticMaterial::from_physical(
                &props(ETA_PA_S[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(ViscoelasticMaterial::from_physical(
                &props(ETA_PA_S[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1, 1));
    let _ = sim.add_body(spawn(2, 2));

    println!("particles={}", sim.particles().len());
    const FRAMES: usize = 300;
    let mut substeps = 0usize;
    let wall = std::time::Instant::now();
    for f in 0..FRAMES {
        sim.step();
        substeps += sim.diagnostics_snapshot().substeps_last_step;
        if f % 30 == 0 {
            let mut line = format!("frame={f} ");
            for (slot, eta) in ETA_PA_S.iter().enumerate() {
                let s = sim.material_state(slot as u32);
                line += &format!("eta={eta:.0}[vavg={:.3}] ", s.avg_speed);
            }
            println!("{line}");
        }
    }
    let elapsed_ms = wall.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{FRAMES} frames in {elapsed_ms:.1} ms -> {:.2} ms/frame, {:.1} fps, {:.1} substeps/frame",
        elapsed_ms / FRAMES as f64,
        1000.0 * FRAMES as f64 / elapsed_ms,
        substeps as f64 / FRAMES as f64
    );
}

// Fine-grained trace: shrink dt so each step_frame() call advances a tiny time
// window, watching WATER particles' own velocity/J at fine resolution to catch
// the exact moment instability enters, not just downstream creature corruption.
//   cargo run --release --example debug_gpu_water_trace --features gpu
extern crate emerge_engine as emerge;

use emerge::{
    Elastic, Elastoplastic, Fluid, FromSI, GpuSimulation, MaterialRegistry, NeoHookeanMaterial,
    PlasticityModel, SimConfig, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID_RES: usize = 128;
const TERRAIN_ID: u32 = 0;
const WATER_ID: u32 = 1;
const CREATURE_ID: u32 = 2;
const TERRAIN_PROPS: Elastoplastic = Elastoplastic {
    elastic: Elastic {
        e_pa: 10.0e6,
        nu: 0.3,
        rho_kg_m3: 1600.0,
    },
    model: PlasticityModel::Granular {
        friction_angle_deg: 35.0,
        dilatancy_angle_deg: 0.0,
    },
};
const WATER_PROPS: Fluid = Fluid {
    rho_kg_m3: 1000.0,
    eta_pa_s: 0.001,
    bulk_modulus_pa: 2.2e9,
    yield_stress_pa: None,
};
const CREATURE_PROPS: Elastic = Elastic {
    e_pa: 500.0,
    nu: 0.45,
    rho_kg_m3: 1000.0,
};

fn main() {
    // Small dt so each step_frame() call only needs a handful of substeps --
    // fine time resolution to catch exactly when/where instability enters.
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 200,
        ..SimConfig::earth(GRID_RES, 0.01, 0.001)
    };
    let creature_mat = NeoHookeanMaterial::from_physical(&CREATURE_PROPS, &config);
    let mut registry = MaterialRegistry::with_default(TERRAIN_PROPS.material(&config));
    registry.insert(TERRAIN_ID, TERRAIN_PROPS.material(&config));
    registry.insert(WATER_ID, WATER_PROPS.material(&config));
    registry.insert(CREATURE_ID, Box::new(creature_mat));

    let mut solver = pollster::block_on(GpuSimulation::new(config, Vec::new(), registry));
    let _ = solver.spawn_region(SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(GRID_RES as i32 - 8, 30),
        box_center: Vec2::new(GRID_RES as f32 * 0.5, 18.0),
        material_id: TERRAIN_ID,
        precompute_initial_volumes: true,
        mass_override: Some(TERRAIN_PROPS.particle_mass(0.6, &config)),
        ..SpawnRegion::for_sim(&config)
    });
    let water_range = solver.spawn_region(SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(28, 14),
        box_center: Vec2::new(GRID_RES as f32 * 0.22, 42.0),
        material_id: WATER_ID,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_PROPS.particle_mass(0.5, &config)),
        ..SpawnRegion::for_sim(&config)
    });
    let creature_range = solver.spawn_region(SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 6),
        box_center: Vec2::new(GRID_RES as f32 * 0.5, 60.0),
        material_id: CREATURE_ID,
        precompute_initial_volumes: true,
        mass_override: Some(CREATURE_PROPS.particle_mass(0.5, &config)),
        ..SpawnRegion::for_sim(&config)
    });

    println!("Fine trace: dt=0.001s per call, watching water AND creature");
    for i in 0..40 {
        solver.step_frame();
        solver.sync_particles_blocking();
        let particles = solver.particles();

        let water_max_v = particles[water_range.clone()]
            .iter()
            .map(|p| p.v.length())
            .fold(0.0f32, f32::max);
        let water_max_j = particles[water_range.clone()]
            .iter()
            .map(|p| p.deformation_gradient.determinant())
            .fold(0.0f32, f32::max);
        let water_has_nan = particles[water_range.clone()]
            .iter()
            .any(|p| p.v.x.is_nan() || p.v.y.is_nan());

        let creature_max_j = particles[creature_range.clone()]
            .iter()
            .map(|p| p.deformation_gradient.determinant())
            .fold(0.0f32, f32::max);
        let creature_min_j = particles[creature_range.clone()]
            .iter()
            .map(|p| p.deformation_gradient.determinant())
            .fold(f32::INFINITY, f32::min);

        println!(
            "call {i}: substeps={} water_max_v={:.2} water_max_J={:.4} water_NaN={} creature_J=[{:.4},{:.4}]",
            solver.last_substeps(),
            water_max_v,
            water_max_j,
            water_has_nan,
            creature_min_j,
            creature_max_j
        );

        if water_has_nan || water_max_j > 10.0 || creature_max_j > 10.0 {
            println!(">>> instability detected at call {i}, stopping trace");
            break;
        }
    }
}

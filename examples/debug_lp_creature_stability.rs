// Throwaway diagnostic: reproduce LP's exact terrain/creature scene on CPU
// Simulation to check whether the severe J-instability found 2026-07-01
// (median J -> 0, some particles J > 500, within first few steps, muscle-
// independent) is CPU+GPU (a real physics bug) or GPU-only (a migration bug).
//
//   cargo run --example debug_lp_creature_stability
extern crate emerge_engine as emerge;

use emerge::{
    Elastic, Elastoplastic, Fluid, FrictionBoundary, FromSI, NeoHookeanMaterial, PlasticityModel,
    SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID_RES: usize = 128;
const DX_METERS: f32 = 0.01;
const DT_SECONDS: f32 = 0.05;
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
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 6000,
        ..SimConfig::earth(GRID_RES, DX_METERS, DT_SECONDS)
    };

    let creature_mat = NeoHookeanMaterial::from_physical(&CREATURE_PROPS, &config);

    let mut solver = Simulation::empty(config)
        .with_material(TERRAIN_ID, TERRAIN_PROPS.material(&config))
        .with_material(WATER_ID, WATER_PROPS.material(&config))
        .with_material(CREATURE_ID, Box::new(creature_mat))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.4)));

    let _ = solver.add_body(SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(GRID_RES as i32 - 8, 30),
        box_center: Vec2::new(GRID_RES as f32 * 0.5, 18.0),
        material_id: TERRAIN_ID,
        precompute_initial_volumes: true,
        mass_override: Some(TERRAIN_PROPS.particle_mass(0.6, &config)),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(24, 16),
        box_center: Vec2::new(GRID_RES as f32 * 0.68, 41.0),
        material_id: TERRAIN_ID,
        precompute_initial_volumes: true,
        mass_override: Some(TERRAIN_PROPS.particle_mass(0.6, &config)),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(28, 14),
        box_center: Vec2::new(GRID_RES as f32 * 0.22, 42.0),
        material_id: WATER_ID,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_PROPS.particle_mass(0.5, &config)),
        ..SpawnRegion::for_sim(&config)
    });

    let creature_tag = solver.add_body(SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 6),
        box_center: Vec2::new(GRID_RES as f32 * 0.5, 60.0),
        material_id: CREATURE_ID,
        precompute_initial_volumes: true,
        mass_override: Some(CREATURE_PROPS.particle_mass(0.5, &config)),
        ..SpawnRegion::for_sim(&config)
    });

    println!("CPU Simulation -- no muscle activation, gravity + contact only");
    for i in 0..6 {
        solver.step();
        let particles = solver.particles();
        let mut js: Vec<f32> = solver
            .particles_with_tag(creature_tag)
            .map(|idx| particles.deformation_gradient[idx].determinant())
            .collect();
        js.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = js.len();
        println!(
            "step {i}: substeps={} min_J={:.4} median_J={:.4} max_J={:.4}",
            solver.last_substeps(),
            js[0],
            js[n / 2],
            js[n - 1]
        );
    }
}

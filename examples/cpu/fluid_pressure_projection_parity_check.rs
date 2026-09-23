extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- direct CPU/GPU parity check for the new
/// GPU port of `Grid::project_fluid_incompressibility`
/// (`src/systems/gpu/shaders/fluid_pressure.wgsl`). Reuses the EXACT scene
/// geometry, config, and checkpoint schedule as
/// `examples/gpu/fluid_pressure_projection_gpu_check.rs`, headless (no
/// window), so the two runs' printed numbers can be compared line for line.
///
/// GPU's run showed J permanently pinned at the hard [0.5,2.0] safety clamp
/// from step ~20 onward. This answers: does CPU's own proven solver, on the
/// IDENTICAL scene, also hit that clamp (meaning the GPU port is faithfully
/// reproducing an already-known CPU limitation), or does CPU stay bounded
/// without pinning (meaning the GPU port has a real, separate bug)?
///
///   cargo run --example fluid_pressure_projection_parity_check
use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const MAT_WATER: u32 = 0;

fn main() {
    let dt = 0.1;
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, dt)
    };

    const SPACING: f32 = 0.6;
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;

    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(11.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn_water).with_default_material(Box::new(water));
    let n = sim.particles().x.len();

    println!("n={n}  wall-contact scene, CPU pressure-projection ENABLED (iterations=1)");

    let wall_start = std::time::Instant::now();
    const N_STEPS: u64 = 120;
    let mut max_speed_ever = 0.0f32;
    let mut min_j_ever = f32::MAX;
    let mut max_j_ever = f32::MIN;
    let mut any_non_finite = 0u32;
    let mut any_out_of_bounds = 0u32;

    for step in 1..=N_STEPS {
        sim.step();
        let particles = sim.particles();

        let mut max_speed = 0.0f32;
        let (mut jmin, mut jmax) = (f32::MAX, f32::MIN);
        let mut non_finite = 0u32;
        let mut out_of_bounds = 0u32;
        for i in 0..particles.x.len() {
            let j = particles.deformation_gradient[i].determinant();
            let v = particles.v[i];
            let x = particles.x[i];
            max_speed = max_speed.max(v.length());
            if !j.is_finite() || !v.is_finite() || !x.is_finite() {
                non_finite += 1;
            }
            if x.x < 0.0 || x.y < 0.0 || x.x > GRID as f32 || x.y > GRID as f32 {
                out_of_bounds += 1;
            }
            jmin = jmin.min(j);
            jmax = jmax.max(j);
        }
        max_speed_ever = max_speed_ever.max(max_speed);
        min_j_ever = min_j_ever.min(jmin);
        max_j_ever = max_j_ever.max(jmax);
        any_non_finite += non_finite;
        any_out_of_bounds += out_of_bounds;

        if step % 10 == 0 || step == 1 || step == N_STEPS {
            println!(
                "step={step:3}  max_speed={max_speed:8.3}  J=[{jmin:.3},{jmax:.3}]  non_finite={non_finite}  oob={out_of_bounds}"
            );
        }
        if non_finite > 0 {
            println!("STOPPING EARLY: non-finite state at step {step}");
            break;
        }
    }
    let elapsed = wall_start.elapsed();
    println!(
        "wall_time={:.2}s  ({:.2}fps over up to {N_STEPS} frames)",
        elapsed.as_secs_f64(),
        N_STEPS as f64 / elapsed.as_secs_f64()
    );
    println!(
        "max_speed_ever={max_speed_ever:.3}  min_j_ever={min_j_ever:.3}  max_j_ever={max_j_ever:.3}  total_non_finite={any_non_finite}  total_out_of_bounds={any_out_of_bounds}"
    );
}

extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- does CPU's pressure-projection fluid
/// solver survive a REPRESENTATIVE contact event (a falling droplet landing
/// on a resting pool), as opposed to
/// `fluid_pressure_projection_parity_check.rs`'s scene, which is
/// `SimConfig::fluid_pressure_iterations`'s own documented WORST case (a
/// column already jammed against a wall at spawn, maximal compression from
/// frame 0 -- the known "Round 9" limitation).
///
/// Reuses `examples/gpu/basic_fluids_gpu.rs`'s exact `DropletImpact`
/// geometry (pool `IVec2::new(50,12)` at `(32.0,9.0)`, blob `IVec2::new(7,7)`
/// at `(32.0,42.0)`, same water density/spacing), routed through the
/// pressure-projection path instead of that demo's stiff-EOS path. This is
/// the actual target scenario for the hybrid explicit/projection
/// contact-switching plan -- a falling body making first contact with a
/// floor/pool, not a pre-jammed column.
///
///   cargo run --release --example fluid_pressure_projection_droplet_landing_check
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

    const SPACING: f32 = 0.5;
    const WATER_RHO_GRID: f32 = 0.1;
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    // Pool first (becomes the base spawn), blob added as a second body --
    // same two-region split basic_fluids_gpu.rs uses.
    let pool_region = water_region(IVec2::new(50, 12), Vec2::new(32.0, 9.0));
    let blob_region = water_region(IVec2::new(7, 7), Vec2::new(32.0, 42.0));

    let mut sim = Simulation::new(config, pool_region).with_default_material(Box::new(water));
    let _ = sim.add_body(blob_region);
    let n = sim.particles().x.len();

    println!(
        "n={n}  droplet-landing scene (pool+falling blob), CPU pressure-projection ENABLED (iterations=1)"
    );

    let wall_start = std::time::Instant::now();
    const N_STEPS: u64 = 150;
    let mut max_speed_ever = 0.0f32;
    let mut min_j_ever = f32::MAX;
    let mut max_j_ever = f32::MIN;
    let mut any_non_finite = 0u32;
    let mut any_out_of_bounds = 0u32;
    let mut clamp_hit_step: Option<u64> = None;

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
        if clamp_hit_step.is_none() && (jmin <= 0.501 || jmax >= 1.999) {
            clamp_hit_step = Some(step);
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
    match clamp_hit_step {
        Some(s) => println!("J HIT THE [0.5,2.0] SAFETY CLAMP at step {s}"),
        None => println!("J NEVER hit the [0.5,2.0] safety clamp across all {N_STEPS} steps"),
    }
}

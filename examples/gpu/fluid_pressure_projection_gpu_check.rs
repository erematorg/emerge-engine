extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- first real, live test of the new GPU port
/// of the CPU-proven Chorin-style fluid incompressibility pressure
/// projection (`src/systems/gpu/shaders/fluid_pressure.wgsl`). Reuses the
/// EXACT wall-contact geometry `examples/cpu/fluid_pressure_projection.rs`
/// already proved stable on CPU (~90fps at the real, correct
/// `gravity_fraction: 0.003`, confirmed earlier the same day) -- the
/// hardest scene this technique has real, positive evidence for. This is
/// the real go/no-go check for whether the GPU port reproduces that.
///
///   cargo run --example fluid_pressure_projection_gpu_check --features gpu
use emerge::gpu::GpuSimulation;
use emerge::{MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;

fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(&InstanceDescriptor {
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

fn main() {
    let instance = create_instance();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    let dt = 0.1;
    // EXACT config from examples/cpu/fluid_pressure_projection.rs's own proven make_sim.
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        // Real, validated gravity for this exact scene -- its own CPU GUI
        // starts at gravity_fraction: 0.003, NOT full earth gravity.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, dt)
    };

    const SPACING: f32 = 0.6;
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    let reference_cell_mass = water.rest_density;
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;

    // EXACT wall-contact geometry (water starting ~2 cells from the left wall).
    let particles = build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(14, 52),
            box_center: Vec2::new(11.0, 30.0),
            material_id: MAT_WATER,
            initial_velocity_scale: 0.0,
            precompute_initial_volumes: true,
            mass_override: Some(WATER_MASS),
            ..SpawnRegion::for_sim(&config)
        },
    );

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    sim.set_fluid_pressure_reference_mass(reference_cell_mass);

    println!(
        "n={}  wall-contact scene, GPU pressure-projection ENABLED (iterations=1)",
        sim.particle_count()
    );

    let wall_start = std::time::Instant::now();
    const N_STEPS: u64 = 120; // matches CPU's own validated run length
    let mut max_speed_ever = 0.0f32;
    let mut min_j_ever = f32::MAX;
    let mut max_j_ever = f32::MIN;
    let mut any_non_finite = 0u32;
    let mut any_out_of_bounds = 0u32;

    for step in 1..=N_STEPS {
        sim.step_frame();
        sim.sync_particles_blocking();
        let particles = sim.particles();

        let max_speed = particles
            .iter()
            .map(|p| p.v.length())
            .fold(0.0f32, f32::max);
        let (mut jmin, mut jmax) = (f32::MAX, f32::MIN);
        let mut non_finite = 0u32;
        let mut out_of_bounds = 0u32;
        for p in particles.iter() {
            let j = p.deformation_gradient.determinant();
            if !j.is_finite() || !p.v.is_finite() || !p.x.is_finite() {
                non_finite += 1;
            }
            if p.x.x < 0.0 || p.x.y < 0.0 || p.x.x > GRID as f32 || p.x.y > GRID as f32 {
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

extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- cost AND benefit of refreshing the GPU's
/// CFL timestep more often, under the REAL demo's async readback.
///
/// GPU computes its CFL `sub_dt` once per `step_frame` from a CPU particle
/// mirror, then reuses it for every substep of that frame; CPU instead
/// re-runs `choose_substep_dt` before every substep. A shorter frame `dt`
/// re-runs the GPU scan more often per simulated second. Claim under test:
/// this costs almost nothing, because `sub_dt` is set by the CFL bound, not
/// by the frame length -- 10x shorter frames should mean ~10x fewer
/// substeps per frame, so the same TOTAL substeps per simulated second.
///
/// Deliberately does NOT call `sync_particles_blocking` every frame (unlike
/// `fragmentation_check_gpu.rs`, which did, making its CPU mirror
/// artificially fresh): the real demo relies on the async readback
/// (`readback_stride = 1`, lands 1-2 frames late), so that is what is
/// reproduced here. One blocking sync at the very end, to measure.
///
///   cargo run --release --example cfl_refresh_rate_check_gpu --features gpu
use emerge::gpu::GpuSimulation;
use emerge::{
    GpuFieldEntry, MaterialRegistry, NewtonianFluidMaterial, Particle, SimConfig, SpawnRegion,
    build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use std::collections::HashMap;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;
const WATER_RHO_GRID: f32 = 0.1;
const SIM_SECONDS: f32 = 4.0;

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

fn isolated_count(particles: &[Particle]) -> usize {
    const CELL: f32 = SPACING * 3.0;
    const ISOLATION_THRESHOLD: f32 = SPACING * 4.0;
    let mut buckets: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, p) in particles.iter().enumerate() {
        let key = ((p.x.x / CELL).floor() as i32, (p.x.y / CELL).floor() as i32);
        buckets.entry(key).or_default().push(i);
    }
    let mut count = 0;
    for (i, p) in particles.iter().enumerate() {
        let (bx, by) = ((p.x.x / CELL).floor() as i32, (p.x.y / CELL).floor() as i32);
        let mut best = f32::MAX;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(bucket) = buckets.get(&(bx + dx, by + dy)) {
                    for &j in bucket {
                        if j != i {
                            best = best.min((particles[j].x - p.x).length());
                        }
                    }
                }
            }
        }
        if best > ISOLATION_THRESHOLD {
            count += 1;
        }
    }
    count
}

fn run(dt: f32) {
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

    // EXACT config and material from fragmentation_check_gpu.rs / basic_fluids_gpu.rs.
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 1000,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: 0.3,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        fluid_near_wall_cfl_scale: 20.0,
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    let particles = build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(14, 52),
            box_center: Vec2::new(20.0, 30.0),
            material_id: MAT_WATER,
            precompute_initial_volumes: true,
            mass_override: Some(WATER_RHO_GRID * SPACING * SPACING),
            ..SpawnRegion::for_sim(&config)
        },
    );
    let v_max_grid = (2.0 * 0.3 * 52.0f32).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_viscosity = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_viscosity,
        1000.0 * c_ref_m_s * c_ref_m_s / 3.0,
        3.0,
    );
    water.bulk_viscosity = 3.0 * water_viscosity;
    water.pressure_floor = config.stress_from_si_physical(-100_000.0, 1000.0);

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim = GpuSimulation::with_device(
        std::sync::Arc::new(device),
        std::sync::Arc::new(queue),
        config,
        particles,
        registry,
    );
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));

    let frames = (SIM_SECONDS / dt).round() as u64;
    let mut total_substeps = 0usize;
    let mut max_substeps_frame = 0usize;
    let start = std::time::Instant::now();
    for _ in 0..frames {
        sim.step_frame();
        total_substeps += sim.last_substeps();
        max_substeps_frame = max_substeps_frame.max(sim.last_substeps());
    }
    sim.sync_particles_blocking();
    let wall = start.elapsed().as_secs_f64();

    let particles = sim.particles();
    let max_speed = particles
        .iter()
        .map(|p| p.v.length())
        .fold(0.0f32, f32::max);
    let (mut jmin, mut jmax) = (f32::MAX, f32::MIN);
    for p in particles {
        let j = p.deformation_gradient.determinant();
        jmin = jmin.min(j);
        jmax = jmax.max(j);
    }
    println!(
        "dt={dt:<5} frames={frames:<4} total_substeps={total_substeps:<6} \
         substeps/frame avg={:<7.1} max={max_substeps_frame:<5} wall={wall:6.2}s \
         ({:.2}x realtime)  END: max_speed={max_speed:7.3} J=[{jmin:.3},{jmax:.3}] isolated={}",
        total_substeps as f64 / frames as f64,
        SIM_SECONDS as f64 / wall,
        isolated_count(particles),
    );
}

fn main() {
    println!("DamBreak, {SIM_SECONDS}s simulated each, async readback (no per-frame sync)");
    run(0.1);
    run(0.01);
}

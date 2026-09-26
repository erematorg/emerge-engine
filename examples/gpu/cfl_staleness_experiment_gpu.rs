extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-17) -- delete after use.
///
/// Tests one specific hypothesis about the splash-disintegration bug
/// (`HANDOFF_fluid_gpu_thin_layer_bug.md`): the GPU CFL scan (`systems/gpu/
/// solver/step.rs`) computes ALL of its stability terms -- including the
/// real, cited Sun/Shinar/Schroeder 2020 single-particle-instability bound
/// (`single_particle_instability_dt_bound`) -- exactly ONCE per rendered
/// frame, from the particle state BEFORE any of that frame's substeps run,
/// then reuses that one dt for up to `max_substeps_per_step` (1000)
/// substeps in a row. CPU's own `choose_substep_dt` (spacetime/solver/
/// step.rs) re-runs the SAME formulas EVERY substep, using freshly-updated
/// state. If a particle's danger (isolation / feedback growth) develops
/// mid-frame -- exactly what a violent splash does -- GPU's one-shot scan
/// cannot react until the NEXT frame, a real, mechanistic "blind window"
/// CPU does not have, structurally distinct from anything about the
/// formulas themselves being wrong.
///
/// This is NOT a claim that this is the only real gap -- it is one
/// concrete, falsifiable hypothesis, tested in isolation.
///
/// Method: run the EXACT `basic_fluids_gpu.rs` DamBreak scene (config,
/// material, spawn geometry copied verbatim, not re-derived), with the
/// shader-side deviatoric-relaxation fix TEMPORARILY DISABLED (g2p.wgsl's
/// `SHEAR_RELAX_BASELINE`/`C_DEV_DANGER_CEILING` set to 0.0/1e9 for this
/// experiment only -- see that file's own temp comment), so nothing masks
/// the raw CFL-scan behavior. Compare two runs covering the SAME total
/// simulated time (`BASE_STEPS * DT_BASE` = 4.0s):
///   - DT_DIVISOR=1: current production frame granularity (DT=0.1, one CFL
///     scan per 0.1s of sim time).
///   - DT_DIVISOR=20: SAME total sim time, but each "frame" is 20x shorter
///     (DT=0.005), so the identical CFL formulas re-scan 20x more often
///     across the same physical impact window.
///
/// If DIVISOR=20 stays bounded (max_particle_speed never crosses the
/// established 20.0 OUTLIER threshold used throughout this investigation)
/// while DIVISOR=1 blows up, that is direct, falsifiable evidence for the
/// staleness hypothesis -- not proof of a complete fix (this does not by
/// itself decide HOW to close the gap in production), but real evidence
/// for WHERE the gap lives.
///
///   cargo run --example cfl_staleness_experiment_gpu --features gpu --release
use emerge::gpu::GpuSimulation;
use emerge::{
    GpuFieldEntry, MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion,
    build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;
const WATER_RHO_GRID: f32 = 0.1;

// Flip this, rebuild, re-run -- see this file's own top-level doc.
const DT_DIVISOR: f32 = 1.0;

const DT_BASE: f32 = 6.0 / 60.0; // verbatim basic_fluids_gpu.rs's own DT derivation
const BASE_STEPS: u64 = 40; // 4.0s of simulated time at DT_BASE -- covers the known danger window (original frames 1-30) with margin
const OUTLIER_SPEED_THRESHOLD: f32 = 20.0; // same threshold basic_fluids_gpu.rs's own OUTLIER trace uses

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
    let dt = DT_BASE / DT_DIVISOR;
    let n_steps = (BASE_STEPS as f32 * DT_DIVISOR).round() as u64;

    let instance = create_instance();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let info = adapter.get_info();
    println!(
        "GPU adapter: {} ({:?}, backend={:?})  DT_DIVISOR={}  dt={:.6}  n_steps={}  total_sim_time={:.3}s",
        info.name,
        info.device_type,
        info.backend,
        DT_DIVISOR,
        dt,
        n_steps,
        n_steps as f32 * dt,
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    // Verbatim from `basic_fluids_gpu.rs`'s DamBreak path (production config,
    // as of the pressure_floor + max_substeps_per_step=1000 fixes).
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 1000,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: 0.3,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        fluid_near_wall_cfl_scale: 20.0,
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    const SPACING: f32 = 0.5;
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn);

    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    const WATER_EOS_POWER: f32 = 3.0;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    // Matches the real viscosity-unit fix now in basic_fluids_gpu.rs.
    let water_dynamic_viscosity = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_dynamic_viscosity,
        water_tait_b_pa,
        WATER_EOS_POWER,
    );
    water.bulk_viscosity = 3.0 * water_dynamic_viscosity;
    const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
    const WATER_RHO_SI_KG_M3: f32 = 1000.0;
    water.pressure_floor =
        config.stress_from_si_physical(REAL_CAVITATION_PRESSURE_PA, WATER_RHO_SI_KG_M3);

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));

    println!(
        "n={}  (DamBreak, box 14x52 grid-units)",
        sim.particle_count()
    );

    let mut running_max_speed = 0.0f32;
    let mut first_outlier_step: Option<u64> = None;
    let log_every = DT_DIVISOR.max(1.0).round() as u64; // one log line per 0.1s-equivalent, comparable across divisors

    for step in 1..=n_steps {
        sim.step_frame();
        let snap = sim.diagnostics_snapshot();
        running_max_speed = running_max_speed.max(snap.max_particle_speed);
        if first_outlier_step.is_none() && snap.max_particle_speed > OUTLIER_SPEED_THRESHOLD {
            first_outlier_step = Some(step);
            println!(
                "  >>> FIRST OUTLIER at step={step} (sim_time={:.4}s): max_particle_speed={:.3}",
                step as f32 * dt,
                snap.max_particle_speed
            );
        }
        if step % log_every == 0 || step == n_steps {
            println!(
                "step={:5}  sim_time={:.4}s  max_particle_speed={:.4}  running_max={:.4}  non_finite={}  oob={}  sub={}",
                step,
                step as f32 * dt,
                snap.max_particle_speed,
                running_max_speed,
                snap.non_finite_particle_values,
                snap.out_of_bounds_particles,
                sim.last_substeps(),
            );
        }
    }

    println!(
        "\n=== RESULT (DT_DIVISOR={DT_DIVISOR}): running_max_speed={:.3}  first_outlier_step={:?}  ===",
        running_max_speed, first_outlier_step
    );
}

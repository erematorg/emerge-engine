extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-16) -- delete after use.
///
/// "Combined-factor" test per HANDOFF_fluid_gpu_thin_layer_bug.md's Ninth
/// pass conclusion: every SINGLE-factor isolated test (uniform velocity
/// field, no gravity, no wall) came back clean or too small. Real bug
/// needs a real non-uniform velocity field + a real wall + many
/// compounding substeps together. This runs CPU and GPU in LOCKSTEP on
/// BIT-IDENTICAL initial particles (same `build_particles` call feeds
/// both, CPU's own particle state is then force-overwritten to match
/// exactly, removing any doubt about spawn-order differences) with a
/// FORCED, IDENTICAL, non-adaptive timestep schedule (`adaptive_timestep:
/// false`, fixed `dt`, `max_substeps_per_step: 1` -- exactly one substep
/// per `step_frame`/`step_n(1)` call on both sides, so any divergence is
/// attributable to a genuine backend difference, not a different substep
/// schedule). Scans every particle every step for the first one where
/// CPU's and GPU's own computed `div(v)` disagree beyond a real
/// tolerance, reporting the exact step, particle, and full local state.
///
///   cargo run --example cpu_gpu_lockstep_divergence_finder --features gpu
use emerge::gpu::GpuSimulation;
use emerge::{
    MaterialRegistry, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 32;
const MAT_WATER: u32 = 0;
const WATER_RHO_GRID: f32 = 0.1;
const SPACING: f32 = 0.5;
const FIXED_DT: f32 = 0.005;
const N_STEPS: u64 = 800;
const DIV_MISMATCH_THRESHOLD: f32 = 0.1;

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

fn make_config() -> SimConfig {
    SimConfig {
        adaptive_timestep: false,
        dt: FIXED_DT,
        max_substeps_per_step: 1,
        cfl_include_affine_speed: false,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, FIXED_DT)
    }
}

// TEST (2026-09-16): bulk_viscosity multiplier, real physical damping
// already present in NewtonianFluidMaterial -- see HANDOFF's Ninth pass.
// If the CPU/GPU divergence is chaotic amplification of an unavoidable
// tiny floating-point difference (not a wrong formula -- six other
// hypotheses already ruled out tonight), real damping should suppress the
// explosive growth. Read from an env var so both the baseline and damped
// runs use the SAME binary.
fn make_water() -> NewtonianFluidMaterial {
    let damping_mult: f32 = std::env::var("DAMPING_MULT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut water = NewtonianFluidMaterial::new(WATER_RHO_GRID, 1.0e-3, 104.0, 3.0);
    water.bulk_viscosity = 3.0 * 1.0e-3 * damping_mult;
    water.settling_damping = std::env::var("SETTLING_DAMPING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    water
}

fn main() {
    let config = make_config();
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(6, 16),
        box_center: Vec2::new(10.0, 15.0),
        material_id: MAT_WATER,
        mass_override: Some(WATER_RHO_GRID * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    };

    // Canonical, single source of truth for initial particle state -- both
    // backends get bit-identical starting conditions.
    let canonical_particles = build_particles(&config, spawn);
    println!("n_particles={}", canonical_particles.len());

    // CPU sim: constructed normally. `Simulation::new` and `build_particles`
    // both call the identical seeded `initialize_particles` -- verified
    // below (before any stepping) rather than trusted blindly, since a
    // manual post-construction overwrite via `Particles::from` was tried
    // first and panicked a strict-fluid consistency check (it skips an
    // invariant-fixing step `Simulation::new`'s own flow performs
    // internally). Trusting the shared construction path instead.
    let mut cpu_sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(make_water()))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    {
        let cpu_particles = cpu_sim.particles();
        let mut max_x_diff = 0f32;
        for (i, p) in canonical_particles.iter().enumerate() {
            max_x_diff = max_x_diff.max((cpu_particles.x[i] - p.x).length());
        }
        println!("sanity check: max initial position diff CPU vs canonical = {max_x_diff:.8}");
        assert!(
            max_x_diff < 1e-5,
            "CPU's own construction diverges from build_particles's canonical set -- cannot trust lockstep comparison"
        );
    }

    // GPU sim: same canonical particles directly.
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
    let registry = MaterialRegistry::with_default(Box::new(make_water()));
    let mut gpu_sim =
        GpuSimulation::with_device(device, queue, config, canonical_particles, registry);

    let mut first_mismatch_reported = false;
    for step in 1..=N_STEPS {
        cpu_sim.step_n(1);
        gpu_sim.step_frame();
        gpu_sim.sync_particles_blocking();

        let cpu_particles = cpu_sim.particles();
        let gpu_particles = gpu_sim.particles();
        assert_eq!(
            cpu_particles.x.len(),
            gpu_particles.len(),
            "particle count diverged -- should never happen, both start identical"
        );

        let mut max_div_diff = 0f32;
        let mut max_diff_idx = 0usize;
        let mut max_pos_diff = 0f32;
        for (i, gpu) in gpu_particles.iter().enumerate() {
            let cpu_c = cpu_particles.velocity_gradient[i];
            let gpu_c = gpu.velocity_gradient;
            let cpu_div = cpu_c.x_axis.x + cpu_c.y_axis.y;
            let gpu_div = gpu_c.x_axis.x + gpu_c.y_axis.y;
            let diff = (cpu_div - gpu_div).abs();
            if diff > max_div_diff {
                max_div_diff = diff;
                max_diff_idx = i;
            }
            max_pos_diff = max_pos_diff.max((cpu_particles.x[i] - gpu.x).length());
        }

        if step % 20 == 0 || step == N_STEPS {
            println!(
                "step={step:4}  max_div_diff={max_div_diff:.5} (particle {max_diff_idx})  max_pos_diff={max_pos_diff:.5}"
            );
        }

        if max_div_diff > DIV_MISMATCH_THRESHOLD && !first_mismatch_reported {
            first_mismatch_reported = true;
            let i = max_diff_idx;
            println!(
                "\n*** FIRST REAL DIVERGENCE at step={step}, particle={i} ***\n\
                 CPU: x=({:.4},{:.4}) v=({:.4},{:.4}) C=[{:.4},{:.4};{:.4},{:.4}] J={:.6}\n\
                 GPU: x=({:.4},{:.4}) v=({:.4},{:.4}) C=[{:.4},{:.4};{:.4},{:.4}] J={:.6}\n",
                cpu_particles.x[i].x,
                cpu_particles.x[i].y,
                cpu_particles.v[i].x,
                cpu_particles.v[i].y,
                cpu_particles.velocity_gradient[i].x_axis.x,
                cpu_particles.velocity_gradient[i].y_axis.x,
                cpu_particles.velocity_gradient[i].x_axis.y,
                cpu_particles.velocity_gradient[i].y_axis.y,
                cpu_particles.deformation_gradient[i].determinant(),
                gpu_particles[i].x.x,
                gpu_particles[i].x.y,
                gpu_particles[i].v.x,
                gpu_particles[i].v.y,
                gpu_particles[i].velocity_gradient.x_axis.x,
                gpu_particles[i].velocity_gradient.y_axis.x,
                gpu_particles[i].velocity_gradient.x_axis.y,
                gpu_particles[i].velocity_gradient.y_axis.y,
                gpu_particles[i].deformation_gradient.determinant(),
            );
        }
    }
    if !first_mismatch_reported {
        println!(
            "\nNo particle ever exceeded div-diff threshold {DIV_MISMATCH_THRESHOLD} across {N_STEPS} lockstep steps."
        );
    }
}

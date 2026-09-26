extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-16) -- delete after use.
///
/// Direct test of the one remaining live hypothesis in
/// `HANDOFF_fluid_gpu_thin_layer_bug.md`'s "Eighth pass": does GPU's
/// fixed-point atomic scatter (P2G) compute the wrong result specifically
/// when MANY particles concurrently contribute to the SAME grid node
/// (heavy thread contention), even though it's already been verified
/// correct in the calm, low-contention case?
///
/// Design: N particles clustered at (nearly) the same position, all with
/// the IDENTICAL velocity v0, no gravity, no boundary interference, a
/// single fixed-size substep. Because every contributing particle shares
/// the same velocity, the mass-weighted average velocity at any grid node
/// they scatter into MUST equal v0 EXACTLY (to fixed-point-quantization
/// precision, ~1e-5), regardless of how many particles contribute or how
/// many GPU threads race to atomically add into that same node. If the
/// atomic scatter has a real correctness bug under heavy contention, the
/// computed velocity should drift away from v0 as N grows; if it stays
/// correct at every N, the atomic-scatter-under-load hypothesis is ruled
/// out too.
///
///   cargo run --example atomic_scatter_contention_test --features gpu
use emerge::gpu::GpuSimulation;
use emerge::{MaterialRegistry, NewtonianFluidMaterial, Particle, SimConfig};
use glam::Vec2;
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const CENTER: Vec2 = Vec2::new(32.0, 32.0);
const V0: Vec2 = Vec2::new(3.0, 0.0);
const WATER_MASS: f32 = 0.025;

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

/// Builds N particles clustered within a tiny jitter radius of `CENTER`,
/// all sharing velocity `V0` -- deliberately NOT using `SpawnRegion`
/// (which spaces particles a full `spacing` apart) since the whole point
/// is to force many particles into the SAME handful of grid cells.
fn make_clustered_particles(n: usize) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic pseudo-jitter (no external RNG dependency) --
        // spreads particles within +/-0.05 grid-units of CENTER, small
        // enough that all land in the same 3x3 kernel stencil, large
        // enough to avoid every particle sitting at the EXACT same f32
        // bit pattern (which could hide a real per-cell-index bug).
        let jitter = Vec2::new(
            ((i * 37) % 101) as f32 / 101.0 - 0.5,
            ((i * 53) % 97) as f32 / 97.0 - 0.5,
        ) * 0.1;
        let mut p = Particle::zeroed();
        p.x = CENTER + jitter;
        p.v = V0;
        p.mass = WATER_MASS;
        p.initial_volume = WATER_MASS / 0.1;
        p.volume = p.initial_volume;
        p.density = 0.1;
        p.material_id = 0;
        p.deformation_gradient = glam::Mat2::IDENTITY;
        p.plastic_volume_ratio = 1.0;
        p.hardening_scale = 1.0;
        particles.push(p);
    }
    particles
}

struct StepResult {
    center_grid_v: Vec2,
    max_particle_div: f32,
    mean_particle_div: f64,
}

fn run_one_substep_and_read_center_velocity(
    n: usize,
    device: &std::sync::Arc<wgpu::Device>,
    queue: &std::sync::Arc<wgpu::Queue>,
) -> StepResult {
    let config = SimConfig {
        adaptive_timestep: false,
        dt: 0.001,
        max_substeps_per_step: 1,
        gravity: Vec2::ZERO,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, 0.001)
    };
    let water = NewtonianFluidMaterial::new(0.1, 1.0e-3, 1.0, 3.0);
    let registry = MaterialRegistry::with_default(Box::new(water));
    let particles = make_clustered_particles(n);

    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    sim.step_frame();

    // Raw grid buffer readback -- 4 f32 per cell (momentum.x, momentum.y,
    // mass, pad), matching the WGSL `Cell` struct layout exactly (see
    // `GpuBuffers::readback_f32_blocking`'s own doc, which this inlines
    // since that method isn't part of GpuSimulation's public surface).
    let cell_count = GRID * GRID;
    let byte_count = (cell_count * 4 * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_grid_readback_staging"),
        size: byte_count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_grid_readback"),
    });
    encoder.copy_buffer_to_buffer(sim.grid_buffer(), 0, &staging, 0, byte_count);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..byte_count);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    // Cell index for CENTER=(32,32): after grid_update, momentum slot holds velocity.
    let cx = CENTER.x as usize;
    let cy = CENTER.y as usize;
    let idx = (cy * GRID + cx) * 4;
    let center_grid_v = Vec2::new(values[idx], values[idx + 1]);

    // Real physics check: every particle shares the SAME velocity, so the
    // velocity FIELD is exactly uniform near the cluster -- a uniform
    // field has zero divergence and zero gradient everywhere, by
    // definition (already the kernel's own cited "zero first moment"
    // property this codebase's extrapolation-fix comments rely on
    // elsewhere). If G2P's gather computes a nonzero velocity_gradient
    // here, that's a direct, isolated bug in the GATHER step specifically
    // (not the scatter, already proven correct above).
    sim.sync_particles_blocking();
    let particles = sim.particles();
    let mut max_div = 0f32;
    let mut div_sum = 0f64;
    for p in particles {
        let div = p.velocity_gradient.x_axis.x + p.velocity_gradient.y_axis.y;
        max_div = max_div.max(div.abs());
        div_sum += div as f64;
    }
    StepResult {
        center_grid_v,
        max_particle_div: max_div,
        mean_particle_div: div_sum / particles.len() as f64,
    }
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

    println!(
        "Expected velocity at center cell: {:?} (every particle shares this exact velocity)  Expected div(v): 0.0 (uniform field)",
        V0
    );
    for &n in &[2usize, 10, 50, 200, 1000, 2912] {
        let r = run_one_substep_and_read_center_velocity(n, &device, &queue);
        let err = (r.center_grid_v - V0).length();
        println!(
            "n={n:5}  measured_v=({:.6},{:.6})  v_error={err:.6}  {}  |  max_particle_div={:.6}  mean_particle_div={:.6}  {}",
            r.center_grid_v.x,
            r.center_grid_v.y,
            if err > 1e-3 {
                "*** V DRIFT ***"
            } else {
                "v ok"
            },
            r.max_particle_div,
            r.mean_particle_div,
            if r.max_particle_div.abs() > 1e-3 {
                "*** SPURIOUS DIVERGENCE ***"
            } else {
                "div ok"
            }
        );
    }
}

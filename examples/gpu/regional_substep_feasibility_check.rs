extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-17) -- delete after use.
///
/// Cheap, read-only, non-invasive feasibility check for regional/adaptive
/// substepping (Fang, Hu, Hu & Jiang, "A Temporally Adaptive Material Point
/// Method with Regional Time Stepping," SCA 2018): before investing in the
/// real block-based scheduler (buffer blocks, power-of-two multipliers --
/// several days of real GPU work), measure whether the paper's own disclosed
/// caveat applies to OUR scene: *"it is not always the preferred choice
/// especially for cases where stiff materials occupy the main portion of a
/// scene."* If most of the domain's blocks are classified "needs fine dt"
/// during the active splash, regional substepping cannot help much even if
/// built perfectly -- there would be no calm region left to skip.
///
/// Method: run the exact, current, already-stabilized `basic_fluids_gpu.rs`
/// DamBreak scene (config copied verbatim). Every frame, partition the
/// domain into 4x4-cell blocks (256 blocks on this GRID=64 scene, matching
/// this engine's own existing "256-block partition" the config doc comments
/// already reference for classification). For each populated block, compute
/// a real advection-CFL bound from the block's own max particle speed
/// (`dt_block = cfl_coefficient * grid_cell_size / max_speed_in_block`,
/// the same shape this engine's own CFL scan already uses, and the term the
/// paper itself identifies as most often binding for fluids). Classify a
/// block Fine if `dt_block <= current_frame_sub_dt * margin`, else Coarse.
/// Report the FRACTION of populated blocks that are Fine, every frame,
/// through the actual violent splash window.
///
/// This does NOT change simulation behavior at all -- pure post-hoc
/// analysis of the real particle state each frame already produces.
///
/// REAL RESULT (2026-09-17), on BOTH scenes tested: DamBreak (whole-column
/// free fall) and DropletImpact (small blob into an ostensibly calm pool)
/// -- 0% of populated blocks classify Fine through the entire violent
/// window, including the exact frames where substep count spikes to
/// 380-440/frame. Confirms this engine's weakly-compressible EOS
/// propagates velocity/pressure through the whole connected fluid body
/// fast enough that no real spatial locality exists to exploit on these
/// scenes -- the paper's own disclosed caveat applies here. See
/// `KNOWN_LIMITATIONS.md` entry 2 (kept as the record of this finding;
/// this file itself may be deleted once ported into a permanent test).
///
///   cargo run --example regional_substep_feasibility_check --features gpu
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
const BLOCK_CELLS: usize = 4; // matches the "256-block partition" the config doc comments reference
const N_BLOCKS_PER_AXIS: usize = GRID / BLOCK_CELLS;
const FINE_TIER_MARGIN: f32 = 8.0; // matches SimConfig::fluid_regional_substepping_fine_tier_margin's own former default

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

    let dt = 6.0 / 60.0; // verbatim basic_fluids_gpu.rs's own DT
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
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    // DropletImpact pattern (basic_fluids_gpu.rs verbatim): a shallow, wide,
    // ALREADY-CALM pool plus a small falling blob -- tests whether regional
    // substepping fares better here than on DamBreak's whole-column-falls
    // case, where the entire domain moved in near-unison and showed 0% Fine.
    let mut particles = build_particles(
        &config,
        water_region(IVec2::new(50, 12), Vec2::new(32.0, 9.0)),
    );
    particles.extend(build_particles(
        &config,
        water_region(IVec2::new(7, 7), Vec2::new(32.0, 42.0)),
    ));

    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    const WATER_EOS_POWER: f32 = 3.0;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water_dynamic_viscosity = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_dynamic_viscosity,
        water_tait_b_pa,
        WATER_EOS_POWER,
    );
    water.bulk_viscosity = 3.0 * water_dynamic_viscosity;
    water.pressure_floor = config.stress_from_si_physical(-100_000.0, 1000.0);

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));

    println!(
        "n={}  DropletImpact, {}x{} blocks of {}x{} cells",
        sim.particle_count(),
        N_BLOCKS_PER_AXIS,
        N_BLOCKS_PER_AXIS,
        BLOCK_CELLS,
        BLOCK_CELLS
    );

    const N_STEPS: u64 = 40; // 4.0s sim time, covers the known violent window with margin

    for step in 1..=N_STEPS {
        sim.step_frame();
        sim.sync_particles_blocking();
        let particles = sim.particles();
        let substeps = sim.last_substeps().max(1);
        let current_sub_dt = dt / substeps as f32;

        let mut block_max_speed = vec![0.0f32; N_BLOCKS_PER_AXIS * N_BLOCKS_PER_AXIS];
        let mut block_populated = vec![false; N_BLOCKS_PER_AXIS * N_BLOCKS_PER_AXIS];
        for p in particles.iter().filter(|p| p.material_id == MAT_WATER) {
            let bx = ((p.x.x as i32 / BLOCK_CELLS as i32).clamp(0, N_BLOCKS_PER_AXIS as i32 - 1))
                as usize;
            let by = ((p.x.y as i32 / BLOCK_CELLS as i32).clamp(0, N_BLOCKS_PER_AXIS as i32 - 1))
                as usize;
            let idx = by * N_BLOCKS_PER_AXIS + bx;
            block_populated[idx] = true;
            block_max_speed[idx] = block_max_speed[idx].max(p.v.length());
        }

        let mut n_populated = 0usize;
        let mut n_fine = 0usize;
        for i in 0..block_populated.len() {
            if !block_populated[i] {
                continue;
            }
            n_populated += 1;
            let speed = block_max_speed[i].max(1.0e-6);
            let dt_block = config.material_cfl_coefficient * config.grid_cell_size / speed;
            if dt_block <= current_sub_dt * FINE_TIER_MARGIN {
                n_fine += 1;
            }
        }
        let fine_frac = if n_populated > 0 {
            n_fine as f32 / n_populated as f32
        } else {
            0.0
        };

        println!(
            "step={:3}  sim_time={:.2}s  sub={:4}  populated_blocks={:3}  FINE={:3} ({:.0}%)  COARSE={:3} ({:.0}%)  max_speed={:.3}",
            step,
            step as f32 * dt,
            substeps,
            n_populated,
            n_fine,
            fine_frac * 100.0,
            n_populated - n_fine,
            (1.0 - fine_frac) * 100.0,
            particles
                .iter()
                .filter(|p| p.material_id == MAT_WATER)
                .map(|p| p.v.length())
                .fold(0.0f32, f32::max),
        );
    }
}

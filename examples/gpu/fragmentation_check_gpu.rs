extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- real answer to a live, direct user report
/// ("ça se casse en l'air", it breaks apart while still airborne) that
/// contradicted this session's own bounding-box-based ext/J readings.
///
/// Real, honest admission this diagnostic exists to correct: a bounding
/// box's width/height growing is EXACTLY what both a healthy puddle spread
/// AND a violent scatter into separate droplets look like in that metric --
/// it cannot tell the two apart. This measures something that CAN: for
/// every particle, the distance to its own nearest neighbor. A coherent
/// fluid body has every particle within roughly 1-2x its own spacing of a
/// neighbor; real fragmentation shows up as particles whose nearest
/// neighbor is suddenly, genuinely far away.
///
/// Also the per-stage GPU profiler this session's perf work was driven by
/// (`PROFILE=1`): per-stage begin/end timestamps of the frame's last substep,
/// plus the whole frame's GPU span, so idle gaps between stages are visible.
///
///   cargo run --release --example fragmentation_check_gpu --features gpu
///
/// Environment knobs (all optional, defaults = `basic_fluids_gpu.rs`'s own scene):
///   PATTERN=dam|drop|vortex   which of the demo's three geometries to run
///   N_STEPS=150               frames to simulate
///   PROFILE=1                 per-stage GPU timings + frame GPU span
///   GRAV_SIZING=2.943         gravity the Tait sound speed is sized from (WCSPH 10*v_max)
///   MAT_CFL=0.3               material CFL coefficient
///   NEAR_WALL_SCALE=1.0       `SimConfig::fluid_near_wall_cfl_scale`
///   PRESSURE_FLOOR_PA=-100000 cavitation floor, in Pa
///   EOS_POWER=3               Tait exponent (also rescales stiffness)
///   DUMP_FRAMES=20,40 DUMP_DIR=. DUMP_TAG=run   dump x,y,J per particle as CSV
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
const SPACING: f32 = 0.5;
const WATER_RHO_GRID: f32 = 0.1;

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
        required_features: adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    let dt = 0.1;
    // EXACT config from basic_fluids_gpu.rs's own current make_sim_data.
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 1000,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: std::env::var("MAT_CFL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3),
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        fluid_near_wall_cfl_scale: std::env::var("NEAR_WALL_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0),
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    // EXACT basic_fluids_gpu.rs geometry for the pattern picked by PATTERN
    // (env: dam | drop | vortex, default dam).
    let pattern = std::env::var("PATTERN").unwrap_or_else(|_| "dam".into());
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut vortex_fields = Vec::new();
    let particles = match pattern.as_str() {
        "dam" => build_particles(
            &config,
            water_region(IVec2::new(14, 52), Vec2::new(20.0, 30.0)),
        ),
        "drop" => {
            let mut p = build_particles(
                &config,
                water_region(IVec2::new(50, 12), Vec2::new(32.0, 9.0)),
            );
            p.extend(build_particles(
                &config,
                water_region(IVec2::new(7, 7), Vec2::new(32.0, 42.0)),
            ));
            p
        }
        "vortex" => {
            let center = Vec2::new(32.0, 26.0);
            let edge_r = 17.0f32;
            let gm = 0.02 * config.gravity.length() * edge_r * edge_r;
            vortex_fields.push(GpuFieldEntry::gravity_well(center, gm, 4.0, 0.0, 0.0));
            let mut p = build_particles(&config, water_region(IVec2::new(54, 48), center));
            let seed_l = 1.0 * edge_r;
            for particle in p.iter_mut() {
                let r = particle.x - center;
                if r.length() < 22.0 {
                    let d = r.length().max(2.0);
                    particle.v = (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
                }
            }
            p
        }
        other => panic!("unknown PATTERN={other}"),
    };

    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    let derated_gravity_for_acoustic_sizing: f32 = std::env::var("GRAV_SIZING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    let v_max_grid = (2.0 * derated_gravity_for_acoustic_sizing * COLUMN_HEIGHT_CELLS).sqrt();
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
    let pressure_floor_pa: f32 = std::env::var("PRESSURE_FLOOR_PA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-100_000.0);
    water.pressure_floor = config.stress_from_si_physical(pressure_floor_pa, 1000.0);
    let eos_power_override: Option<f32> =
        std::env::var("EOS_POWER").ok().and_then(|s| s.parse().ok());
    if let Some(power) = eos_power_override {
        water.eos_power = power;
        water.eos_stiffness = 1000.0 * c_ref_m_s * c_ref_m_s / power;
    }

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim = GpuSimulation::with_device(device.clone(), queue, config, particles, registry);
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
    for field in vortex_fields {
        sim.add_force_field_gpu(field);
    }
    let profile = std::env::var("PROFILE").is_ok();
    if profile {
        println!("adapter features: {:?}", adapter.features());
    }
    if profile {
        println!("profiling enabled: {}", sim.enable_profiling());
    }

    println!(
        "n={}  PATTERN={pattern}, fragmentation check",
        sim.particle_count()
    );

    const CELL: f32 = SPACING * 3.0; // bucket size for the neighbor search
    let n_steps: u64 = std::env::var("N_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    for step in 1..=n_steps {
        let t0 = std::time::Instant::now();
        sim.step_frame();
        let step_ms = t0.elapsed().as_secs_f64() * 1e3;
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let gpu_done_ms = t0.elapsed().as_secs_f64() * 1e3;
        if profile
            && step % 10 == 0
            && let Some(timeline) = sim.last_pass_timeline_ns()
        {
            {
                let busy: f32 = timeline.iter().map(|(_, b, e)| e - b).sum();
                let span = timeline.iter().map(|(_, _, e)| *e).fold(0.0f32, f32::max);
                let parts: Vec<String> = timeline
                    .iter()
                    .filter(|(_, b, e)| e > b)
                    .map(|(label, b, e)| format!("{label}[{:.0}->{:.0}]", b * 1e-3, e * 1e-3))
                    .collect();
                let frame_span = sim.last_frame_gpu_span_ns().unwrap_or(0.0);
                println!(
                    "PASS step={step} frame_gpu={:.1}ms busy={:.0}us span={:.0}us  {}",
                    frame_span * 1e-6,
                    busy * 1e-3,
                    span * 1e-3,
                    parts.join(" ")
                );
            }
        }
        sim.sync_particles_blocking();
        let frame_ms = t0.elapsed().as_secs_f64() * 1e3;
        let (cfl_ns, enc_ns, wait_ns, rb_ns, _) = sim.last_cpu_timings_ns();
        let particles = sim.particles();

        // Simple spatial-hash bucketing, real nearest-neighbor distance per
        // particle via a 3x3 bucket search (bucket size covers the search
        // radius so no real neighbor is missed).
        use std::collections::HashMap;
        let mut buckets: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, p) in particles.iter().enumerate() {
            let bx = (p.x.x / CELL).floor() as i32;
            let by = (p.x.y / CELL).floor() as i32;
            buckets.entry((bx, by)).or_default().push(i);
        }
        let mut max_nn = 0.0f32;
        let mut isolated_count = 0usize;
        const ISOLATION_THRESHOLD: f32 = SPACING * 4.0; // real spacing is 0.5; 2.0 is already 4x
        for (i, p) in particles.iter().enumerate() {
            let bx = (p.x.x / CELL).floor() as i32;
            let by = (p.x.y / CELL).floor() as i32;
            let mut best = f32::MAX;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(bucket) = buckets.get(&(bx + dx, by + dy)) {
                        for &j in bucket {
                            if j == i {
                                continue;
                            }
                            let d = (particles[j].x - p.x).length();
                            if d < best {
                                best = d;
                            }
                        }
                    }
                }
            }
            if best.is_finite() {
                max_nn = max_nn.max(best);
                if best > ISOLATION_THRESHOLD {
                    isolated_count += 1;
                }
            }
        }

        let max_speed = particles
            .iter()
            .map(|p| p.v.length())
            .fold(0.0f32, f32::max);
        let (mut jmin, mut jmax) = (f32::MAX, f32::MIN);
        let (mut jsum, mut vysum) = (0.0f32, 0.0f32);
        for p in particles.iter() {
            let j = p.deformation_gradient.determinant();
            jmin = jmin.min(j);
            jmax = jmax.max(j);
            jsum += j;
            vysum += p.v.y;
        }
        let jmean = jsum / particles.len() as f32;
        let com_vy = vysum / particles.len() as f32;
        if let Ok(frames) = std::env::var("DUMP_FRAMES")
            && frames
                .split(',')
                .any(|f| f.trim().parse::<u64>().ok() == Some(step))
        {
            {
                let dir = std::env::var("DUMP_DIR").unwrap_or_else(|_| ".".into());
                let tag = std::env::var("DUMP_TAG").unwrap_or_else(|_| "run".into());
                let mut out = String::new();
                for p in particles.iter() {
                    out.push_str(&format!(
                        "{:.3},{:.3},{:.4}
",
                        p.x.x,
                        p.x.y,
                        p.deformation_gradient.determinant()
                    ));
                }
                std::fs::write(format!("{dir}/{tag}_f{step:03}.csv"), out).expect("dump");
            }
        }
        // GRID_STATS=1: what `ColorMode::GridVolume` would actually see after the frame
        // (it renders the solver's own grid mass, unlike the particle/surface modes).
        if std::env::var("GRID_STATS").is_ok() && step % 10 == 0 {
            let cells = sim.grid_cells_blocking();
            let res = GRID;
            let mut max_mass = 0.0f32;
            let mut above_floor = 0usize;
            let render_floor = 0.15 * WATER_RHO_GRID; // Renderer::grid_volume mass_floor
            for i in 0..res * res {
                let m = cells[i * 4 + 2];
                max_mass = max_mass.max(m);
                if m > render_floor {
                    above_floor += 1;
                }
            }
            println!(
                "  GRID_STATS max_cell_mass={max_mass:.4} cells_above_render_floor={above_floor} (floor={render_floor:.4})"
            );
        }
        println!(
            "step={step:3}  max_speed={max_speed:7.3}  J=[{jmin:.3},{jmax:.3}] Jmean={jmean:.4} com_vy={com_vy:+.3}  max_nearest_neighbor_dist={max_nn:.3} (spacing={SPACING})  isolated_particles(>{ISOLATION_THRESHOLD:.2})={isolated_count}  sub={}  frame_ms={frame_ms:.1} step_ms={step_ms:.1} gpu_done_ms={gpu_done_ms:.1} cfl={:.2} enc={:.2} wait={:.2} rb={:.2}",
            sim.last_substeps(),
            cfl_ns * 1e-6,
            enc_ns * 1e-6,
            wait_ns * 1e-6,
            rb_ns * 1e-6,
        );
    }
}

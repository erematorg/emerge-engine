//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/gpu/basic_snow_gpu.rs`'s scene (no window) to verify the real
//! SI migration (Stomakhin 2013 canonical snow, E=1.4e5 Pa/nu=0.2/rho=200,
//! same citation as the CPU twin) survives the real snowball collision on
//! the GPU backend.
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod gpu_probe {
    use emerge::gpu::GpuSimulation;
    use emerge::{MaterialRegistry, SimConfig, SpawnRegion, StomakhinMaterial, build_particles};
    use glam::{IVec2, Vec2};
    use pollster::block_on;
    use wgpu::InstanceDescriptor;

    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MAT_SOFT: u32 = 0;
    const MAT_PACKED: u32 = 1;
    const BALL_R: f32 = 9.0;
    const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
    const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
    const SPEED: f32 = 15.0;

    const SNOW_YOUNG_MODULUS_PA: f32 = 1.4e5;
    const SNOW_POISSON_RATIO: f32 = 0.2;
    const SNOW_DENSITY_KG_M3: f32 = 200.0;

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

    fn gpu_available() -> bool {
        let instance = create_instance();
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    fn make_sim(max_substeps_per_step: usize) -> GpuSimulation {
        let config = SimConfig {
            max_substeps_per_step,
            gravity: Vec2::new(0.0, -0.08),
            ..SimConfig::earth(GRID, 0.01, DT)
        };
        let (lambda, mu) = config.lame_from_si_physical_cfg(
            SNOW_YOUNG_MODULUS_PA,
            SNOW_POISSON_RATIO,
            SNOW_DENSITY_KG_M3,
        );
        println!("[snow-gpu-probe] lambda={lambda} mu={mu}");
        let mass_grid = (SNOW_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
        let spawn_ball = |center: Vec2, mat: u32, seed: u32| SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new((BALL_R * 2.0) as i32, (BALL_R * 2.0) as i32),
            box_center: center,
            material_id: mat,
            precompute_initial_volumes: true,
            rng_seed: seed,
            mass_override: Some(mass_grid),
            ..SpawnRegion::for_sim(&config)
        };
        let mut particles = build_particles(&config, spawn_ball(BALL_A, MAT_SOFT, 1));
        for p in particles.iter_mut() {
            p.v.x = SPEED;
        }
        let mut right = build_particles(&config, spawn_ball(BALL_B, MAT_PACKED, 2));
        for p in right.iter_mut() {
            p.v.x = -SPEED;
        }
        particles.extend(right);

        let mut registry = MaterialRegistry::with_default(Box::new(StomakhinMaterial::new(
            lambda, mu, 7.0, 0.025, 0.0075, 0.6, 20.0,
        )));
        registry.insert(
            MAT_PACKED,
            Box::new(
                StomakhinMaterial::new(lambda, mu, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0),
            ),
        );
        block_on(GpuSimulation::new(config, particles, registry))
    }

    pub fn run_probe(label: &str, max_substeps_per_step: usize, steps: u64) {
        if !gpu_available() {
            println!("[{label}] no GPU adapter available, skipping");
            return;
        }
        let mut sim = make_sim(max_substeps_per_step);
        for step in 1..=steps {
            sim.step_frame();
            if step.is_multiple_of(steps / 10) || step == 1 || step == steps {
                let snap = sim.diagnostics_snapshot();
                println!(
                    "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] non_finite={} time_dropped={:.4}",
                    step as f32 * DT,
                    snap.substeps_last_step,
                    snap.min_deformation_j,
                    snap.max_deformation_j,
                    snap.non_finite_particle_values,
                    snap.sim_time_dropped,
                );
                assert_eq!(
                    snap.non_finite_particle_values, 0,
                    "[{label}] NaN/Inf at step {step}"
                );
            }
        }
        let snap = sim.diagnostics_snapshot();
        assert!(
            snap.sim_time_dropped < 1.0e-6,
            "[{label}] sim_time_dropped={} -- max_substeps_per_step too low",
            snap.sim_time_dropped
        );
    }
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
#[cfg(feature = "gpu")]
fn basic_snow_gpu_real_si_stiffness_collision_survives() {
    // BALL_A/BALL_B start 32 grid units apart, closing at SPEED*2=30
    // units/s -- collision happens within ~1s, well inside a 60-step (6s)
    // window that also covers post-impact settling.
    gpu_probe::run_probe("substeps=8000", 8000, 60);
}

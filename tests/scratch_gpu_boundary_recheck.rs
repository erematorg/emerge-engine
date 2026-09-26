//! TEMPORARY -- re-verifying a claim made earlier tonight ("basic_jellies_gpu.rs
//! has no boundary condition at all") against direct evidence that
//! grid_update.wgsl DOES have an always-on slip boundary using
//! boundary_thickness (default 2, not 0). Tracks real particle position over
//! time to see what actually happens, instead of trusting either claim.
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod gpu_check {
    use emerge::gpu::GpuSimulation;
    use emerge::{
        CorotatedMaterial, MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion,
        ViscoelasticMaterial, build_particles,
    };
    use glam::{IVec2, Vec2};
    use pollster::block_on;
    use wgpu::InstanceDescriptor;

    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const MAT_NEO: u32 = 0;
    const MAT_COR: u32 = 1;
    const MAT_VIS: u32 = 2;

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

    fn make_sim() -> GpuSimulation {
        let config = SimConfig {
            max_substeps_per_step: 20_000,
            gravity: Vec2::new(0.0, -0.3),
            ..SimConfig::earth(GRID, 0.01, DT)
        };
        println!(
            "[boundary-recheck] boundary_thickness={}",
            config.boundary_thickness
        );
        let (lambda, mu) = config.lame_from_si_physical_cfg(500.0, 0.45, 1000.0);
        let visc = config.visc_from_si_physical(1.0, 1000.0);
        let mass_grid = (1000.0 / config.reference_density_kg_m3) * 0.5 * 0.5;
        let blob = |cx: f32, mat: u32, seed: u32| SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(16, 16),
            box_center: Vec2::new(cx, 48.0),
            material_id: mat,
            precompute_initial_volumes: true,
            rng_seed: seed,
            mass_override: Some(mass_grid),
            ..SpawnRegion::for_sim(&config)
        };
        let mut particles = build_particles(&config, blob(16.0, MAT_NEO, 1));
        particles.extend(build_particles(&config, blob(32.0, MAT_COR, 2)));
        particles.extend(build_particles(&config, blob(48.0, MAT_VIS, 3)));
        let mut registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(lambda, mu)));
        registry.insert(MAT_COR, Box::new(CorotatedMaterial::new(lambda, mu)));
        registry.insert(
            MAT_VIS,
            Box::new(ViscoelasticMaterial::new(lambda, mu, visc)),
        );
        block_on(GpuSimulation::new(config, particles, registry))
    }

    pub fn run() {
        if !gpu_available() {
            println!("[boundary-recheck] no GPU adapter, skipping");
            return;
        }
        let mut sim = make_sim();
        for step in 1..=200u64 {
            sim.step_frame();
            if step.is_multiple_of(20) || step == 1 {
                let particles = sim.particles();
                let mut min_y = f32::MAX;
                let mut max_speed = 0.0f32;
                for p in particles.iter() {
                    min_y = min_y.min(p.x.y);
                    max_speed = max_speed.max(p.v.length());
                }
                println!(
                    "[boundary-recheck] step={step} t={:.1} min_y={min_y:.3} max_speed={max_speed:.3}",
                    step as f32 * DT,
                );
            }
        }
    }
}

#[test]
#[ignore = "one-off recheck of an earlier claim, not a regression test"]
#[cfg(feature = "gpu")]
fn recheck_basic_jellies_gpu_boundary_behavior() {
    gpu_check::run();
}

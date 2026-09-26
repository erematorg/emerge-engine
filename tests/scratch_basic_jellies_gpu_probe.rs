//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/gpu/basic_jellies_gpu.rs`'s scene (no window) to check the
//! real SI migration (soft-tissue E=500 Pa/nu=0.45/rho=1000, same citation
//! as the CPU twin) for early instability on the GPU backend.
//!
//! Real, honest limitation found running this: the scene has NO boundary
//! condition at all (unlike the CPU twin's `SlipBoundary`), so the blob
//! free-falls the entire 30s probe window and never hits anything -- J
//! stays exactly 1.0 throughout. This confirms free-fall itself is stable,
//! it does NOT confirm impact-safety (the actual worst case
//! max_substeps_per_step needs to survive) -- that's why the example
//! itself matches the CPU twin's empirically-impact-tested value (20000)
//! rather than trusting this probe's clean result.
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod gpu_probe {
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

    const JELLY_YOUNG_MODULUS_PA: f32 = 500.0;
    const JELLY_POISSON_RATIO: f32 = 0.45;
    const JELLY_DENSITY_KG_M3: f32 = 1000.0;
    const JELLY_VISCOSITY_PA_S: f32 = 1.0;

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
            gravity: Vec2::new(0.0, -0.3),
            ..SimConfig::earth(GRID, 0.01, DT)
        };
        let (lambda, mu) = config.lame_from_si_physical_cfg(
            JELLY_YOUNG_MODULUS_PA,
            JELLY_POISSON_RATIO,
            JELLY_DENSITY_KG_M3,
        );
        let visc = config.visc_from_si_physical(JELLY_VISCOSITY_PA_S, JELLY_DENSITY_KG_M3);
        println!("[jellies-gpu-probe] lambda={lambda} mu={mu} visc={visc}");
        let mass_grid = (JELLY_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
        let blob = |cx: f32, mat: u32, seed: u32| SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(16, 16),
            box_center: Vec2::new(cx, 48.0),
            material_id: mat,
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
fn basic_jellies_gpu_real_si_stiffness_substep_check() {
    // 120 steps (12s) wasn't enough to even reach the floor at this scene's
    // deliberately-weak gravity (-0.3) -- J stayed flat at 1.0 the entire
    // run, meaning the real impact/settling event (the thing that actually
    // stresses max_substeps_per_step) hadn't happened yet. ~37 grid units
    // of fall at g=0.3 needs ~t=sqrt(2*37/0.3)=~15.7s -- 300 steps (30s)
    // gives real margin past that.
    gpu_probe::run_probe("substeps=12", 12, 300);
}

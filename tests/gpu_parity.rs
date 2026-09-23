//! CPU/GPU parity, law by law: does the GPU compute the same physics?
//!
//! The GPU suite next door is mostly stability checks: they catch a crash
//! or a NaN, not a shader that quietly computes something else. Only two
//! tests in it ever compared the two paths. This file is the matrix that
//! phase 2 of the core plan asks for: every law the GPU will accept, run
//! through the same three scenes on both paths from the same spawn, with
//! the differences printed rather than summarised.
//!
//! The three scenes are chosen so a mismatch points somewhere:
//!
//! - **free fall** carries no stress at all, so the two paths differ only
//!   in their transfer and integration. A gap here is not the law.
//! - **uniaxial compression** starts the body already squeezed with no
//!   gravity, so what moves is the constitutive law answering.
//! - **hydrostatic column** stands a body on a floor under gravity, which
//!   is where a plastic law's own damping and any dropped time show up.
//!
//! Every row also reports the substeps each side actually executed. Phase
//! 1 added that counter for this reason: without it, a difference cannot
//! be told apart from the GPU simply advancing less time.
//!
//! Real hardware only, like the rest of the GPU suite: the software
//! adapter CI runs on does not reproduce a real device's behaviour here.
//!
//! # The process sometimes dies after this test passes
//!
//! About one run in nine ends with Windows exit code 0xc0000409
//! (FAST_FAIL) AFTER the harness has printed its result. It is not this
//! engine's: there is no `unsafe` and no `Drop` anywhere in
//! `systems::gpu`, so nothing of ours runs at teardown, and with
//! `RUST_BACKTRACE=full` and wgpu's own logging on there is no panic
//! message, no backtrace and no validation warning -- a Rust panic would
//! have printed one. Attributed by measurement rather than by argument:
//! 2 aborts in 18 runs on the DX12 backend, 0 in 20 on Vulkan, with the
//! matrix printing identical numbers on both, and the existing 52-test
//! GPU suite never showing it. So it belongs to the DX12 teardown path,
//! and this file must not be made to gate CI on that backend.
//!
//!   cargo test --test gpu_parity --features gpu -- --ignored --nocapture --test-threads=1
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod parity {
    use emerge::gpu::GpuSimulation;
    use emerge::{
        BinghamFluidMaterial, CorotatedMaterial, DruckerPragerMaterial, MaterialModel,
        MaterialRegistry, MuIRheologyMaterial, NeoHookeanMaterial, NewtonianFluidMaterial,
        NoCompressionMaterial, RankineMaterial, SimConfig, SlipBoundary, SpawnRegion,
        StomakhinMaterial, ViscoelasticMaterial, VonMisesMaterial, build_particles,
    };
    use glam::{IVec2, Mat2, Vec2};
    use pollster::block_on;
    use wgpu::InstanceDescriptor;

    /// Same instance setup as `tests/gpu.rs` -- see that file's own comment
    /// for why the DX12 compiler is pinned.
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
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).is_ok()
    }

    /// One law, built twice: the CPU solver and the GPU registry each need
    /// their own box of the same thing.
    struct Law {
        name: &'static str,
        make: fn() -> Box<dyn MaterialModel>,
    }

    /// Every law with a stress path worth comparing. Which of them the
    /// GPU will actually accept is not decided here: the matrix asks the
    /// registry through phase 1's own guard and reports a refusal as a
    /// row, so the list cannot quietly drift out of date the way a
    /// hand-kept exclusion list does. Found this way on the first run:
    /// NoCompression is refused too (issue #29), which a guessed list
    /// had missed.
    fn laws() -> Vec<Law> {
        vec![
            Law {
                name: "NeoHookean",
                make: || Box::new(NeoHookeanMaterial::new(2000.0, 4000.0)),
            },
            Law {
                name: "Corotated",
                make: || Box::new(CorotatedMaterial::new(2000.0, 4000.0)),
            },
            Law {
                name: "Viscoelastic",
                make: || Box::new(ViscoelasticMaterial::new(2000.0, 4000.0, 1.0)),
            },
            Law {
                name: "NoCompression",
                make: || Box::new(NoCompressionMaterial::new(2000.0, 4000.0)),
            },
            Law {
                name: "NewtonianFluid",
                make: || Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)),
            },
            Law {
                name: "BinghamFluid",
                make: || Box::new(BinghamFluidMaterial::low_yield(4.0, 10.0)),
            },
            Law {
                name: "Snow",
                make: || Box::new(StomakhinMaterial::from_young_modulus(10_000.0, 0.3)),
            },
            Law {
                name: "DruckerPrager",
                make: || Box::new(DruckerPragerMaterial::cohesionless(2000.0, 4000.0)),
            },
            Law {
                name: "SandMuI",
                make: || Box::new(MuIRheologyMaterial::small_grain(2000.0, 4000.0)),
            },
            Law {
                name: "VonMises",
                make: || Box::new(VonMisesMaterial::new(2000.0, 4000.0, 50.0)),
            },
            Law {
                name: "Rankine",
                make: || Box::new(RankineMaterial::stiff_brittle(2000.0, 4000.0)),
            },
        ]
    }

    #[derive(Clone, Copy)]
    enum Scene {
        FreeFall,
        UniaxialCompression,
        HydrostaticColumn,
    }

    impl Scene {
        fn name(self) -> &'static str {
            match self {
                Scene::FreeFall => "free fall",
                Scene::UniaxialCompression => "uniaxial compression",
                Scene::HydrostaticColumn => "hydrostatic column",
            }
        }

        fn frames(self) -> usize {
            match self {
                Scene::FreeFall => 20,
                Scene::UniaxialCompression => 20,
                Scene::HydrostaticColumn => 60,
            }
        }

        fn config(self) -> SimConfig {
            let gravity = match self {
                Scene::UniaxialCompression => Vec2::ZERO,
                _ => Vec2::new(0.0, -9.81),
            };
            SimConfig {
                max_substeps_per_step: 64,
                ..SimConfig::standard(32, 0.002, gravity)
            }
        }

        fn spawn(self, config: &SimConfig) -> SpawnRegion {
            let squeezed = Mat2::from_diagonal(Vec2::new(0.9, 1.0));
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(6, 6),
                box_center: match self {
                    Scene::HydrostaticColumn => Vec2::new(16.0, 8.0),
                    _ => Vec2::splat(16.0),
                },
                initial_velocity_scale: 0.0,
                initial_deformation_gradient: match self {
                    Scene::UniaxialCompression => squeezed,
                    _ => Mat2::IDENTITY,
                },
                ..SpawnRegion::for_sim(config)
            }
        }
    }

    /// What the two paths disagree about, in the scene's own units.
    struct Gap {
        position: f32,
        velocity: f32,
        volume_ratio: f32,
        cpu_substeps: usize,
        gpu_substeps: usize,
    }

    /// `None` when the GPU refuses the law, with the engine's own reason.
    fn compare(law: &Law, scene: Scene) -> Result<Gap, &'static str> {
        let config = scene.config();
        let spawn = scene.spawn(&config);

        let mut cpu = emerge::Simulation::new(config, spawn)
            .with_default_material((law.make)())
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let gpu_particles = build_particles(&config, spawn);
        let gpu_registry = MaterialRegistry::with_default((law.make)());
        if let Some((_, reason)) = gpu_registry.first_gpu_unsupported() {
            return Err(reason);
        }
        let mut gpu = block_on(GpuSimulation::new(config, gpu_particles, gpu_registry));

        let (mut cpu_substeps, mut gpu_substeps) = (0usize, 0usize);
        for _ in 0..scene.frames() {
            cpu.step();
            cpu_substeps += cpu.diagnostics_snapshot().substeps_last_step;
            gpu.step_frame();
            // Phase 1's executed-substep counter. It reports the frame
            // whose stats have been read back, so this total lags the
            // CPU's by one frame out of the scene's twenty or sixty.
            gpu_substeps += gpu.last_substeps();
        }
        gpu.sync_particles_blocking();

        let cpu_particles = cpu.particles();
        let gpu_particles = gpu.particles();
        assert_eq!(cpu_particles.len(), gpu_particles.len());
        let mut gap = Gap {
            position: 0.0,
            velocity: 0.0,
            volume_ratio: 0.0,
            cpu_substeps,
            gpu_substeps,
        };
        for (cpu_p, gpu_p) in cpu_particles.iter().zip(gpu_particles) {
            gap.position = gap.position.max((cpu_p.x - gpu_p.x).length());
            gap.velocity = gap.velocity.max((cpu_p.v - gpu_p.v).length());
            let cpu_j = cpu_p.deformation_gradient.determinant();
            let gpu_j = gpu_p.deformation_gradient.determinant();
            gap.volume_ratio = gap.volume_ratio.max((cpu_j - gpu_j).abs());
        }
        Ok(gap)
    }

    /// What a cell is allowed to differ by, and why when it is a lot.
    ///
    /// Every bound here is a measured number times three, not a guess:
    /// the matrix was run first with no bounds at all, and these are what
    /// it read, with headroom for the run-to-run spread the GPU's own
    /// substep count shows. A cell with a `note` is a known gap, kept
    /// bounded rather than silenced: it fails the day it gets worse.
    struct Expect {
        position: f32,
        velocity: f32,
        note: Option<&'static str>,
    }

    /// The GPU damps velocity on every plastic model (`v *= 0.999` in
    /// `particles_update.wgsl`) and the CPU does not. Free fall proves it
    /// is not the law: the scene carries no stress at all, and the same
    /// five laws still separate from the elastic ones by two orders of
    /// magnitude in velocity.
    const GPU_PLASTIC_DAMPING: &str =
        "GPU damps plastic models by 0.999 per substep, the CPU does not (phase 3)";

    fn expected(scene: Scene, law: &str) -> Expect {
        let free = |position, velocity, note| Expect {
            position,
            velocity,
            note,
        };
        let plastic = matches!(
            law,
            "Snow" | "DruckerPrager" | "SandMuI" | "VonMises" | "Rankine"
        );
        match (scene, law) {
            // Snow's own GPU gap, separate from the damping: under
            // compression it both answers differently and runs half the
            // substeps the CPU asks for (39 against 19).
            (Scene::UniaxialCompression, "Snow") => free(
                0.3,
                7.0,
                Some("GPU snow answers a compression differently and runs half the substeps"),
            ),
            // Von Mises drifts under compression beyond what the damping
            // explains, and nothing yet says why.
            (Scene::UniaxialCompression, "VonMises") => free(
                7.0e-3,
                0.6,
                Some("GPU von Mises answers a compression differently, cause unknown"),
            ),
            // Bingham advances far less time than the CPU: 220 substeps
            // against 57 in free fall, 660 against 177 standing.
            (_, "BinghamFluid") => free(
                2.0e-3,
                5.0e-2,
                Some("GPU Bingham executes about a third of the CPU's substeps"),
            ),
            (Scene::FreeFall, _) if plastic => free(2.0e-4, 1.3e-2, Some(GPU_PLASTIC_DAMPING)),
            (Scene::HydrostaticColumn, _) if plastic => {
                free(5.0e-3, 1.2e-1, Some(GPU_PLASTIC_DAMPING))
            }
            (Scene::FreeFall, _) => free(1.0e-5, 1.5e-4, None),
            (Scene::UniaxialCompression, _) => free(1.0e-3, 6.0e-2, None),
            (Scene::HydrostaticColumn, _) => free(3.0e-4, 8.0e-3, None),
        }
    }

    /// The matrix, printed in full and checked against the bounds above.
    #[test]
    #[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
    fn cpu_gpu_parity_matrix() {
        let mut failures: Vec<String> = Vec::new();
        if !gpu_available() {
            println!("no GPU adapter, nothing measured");
            return;
        }
        // Narrowing knobs, kept because the first run needed them: this
        // test builds one wgpu device per cell, and a machine that
        // cannot take thirty of them in one process has to be able to
        // ask for fewer.
        let want = |name: &str, default: usize| -> usize {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let scene_count = want("PARITY_SCENES", 3).clamp(1, 3);
        let law_count = want("PARITY_LAWS", laws().len()).max(1);
        for scene in [
            Scene::FreeFall,
            Scene::UniaxialCompression,
            Scene::HydrostaticColumn,
        ]
        .into_iter()
        .take(scene_count)
        {
            println!(
                "=== {} ({} frames, dt {} s)",
                scene.name(),
                scene.frames(),
                scene.config().dt
            );
            for law in laws().into_iter().take(law_count) {
                match compare(&law, scene) {
                    Ok(gap) => {
                        let bar = expected(scene, law.name);
                        let over = gap.position > bar.position || gap.velocity > bar.velocity;
                        println!(
                            "  {:>16}: dx={:.3e}  dv={:.3e}  dJ={:.3e}   substeps cpu={} gpu={}{}",
                            law.name,
                            gap.position,
                            gap.velocity,
                            gap.volume_ratio,
                            gap.cpu_substeps,
                            gap.gpu_substeps,
                            match bar.note {
                                Some(note) => format!("   KNOWN: {note}"),
                                None => String::new(),
                            }
                        );
                        if over {
                            failures.push(format!(
                                "{} / {}: dx={:.3e} (bar {:.3e}), dv={:.3e} (bar {:.3e})",
                                scene.name(),
                                law.name,
                                gap.position,
                                bar.position,
                                gap.velocity,
                                bar.velocity
                            ));
                        }
                    }
                    Err(reason) => {
                        println!("  {:>16}: refused by the engine -- {reason}", law.name)
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "CPU and GPU moved apart by more than this matrix has ever measured:
  {}",
            failures.join(
                "
  "
            )
        );
    }
}

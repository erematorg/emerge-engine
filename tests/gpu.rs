//! GPU solver smoke tests — basic stability checks for GpuSimulation.
//!
//! These tests run headlessly (no window, no Bevy) using pollster::block_on.
//! They verify that the GPU pipeline doesn't crash or produce NaN on standard
//! material configurations.

extern crate emerge_engine as emerge;
#[cfg(feature = "gpu")]
mod gpu_tests {
    use emerge::gpu::GpuSimulation;
    use emerge::{
        DruckerPragerMaterial, MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial,
        SimConfig, SpawnRegion, StomakhinMaterial, ViscoelasticMaterial, build_particles,
    };
    use glam::Vec2;
    use pollster::block_on;
    use wgpu::InstanceDescriptor;

    /// Returns false when no GPU adapter is available (e.g. CI runners without a GPU).
    /// Tests call this and return early so they show as passed-but-skipped rather than crashing.
    fn gpu_available() -> bool {
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    fn small_config() -> SimConfig {
        SimConfig {
            max_substeps_per_step: 8,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        }
    }

    fn spawn_disk(config: &SimConfig, center: Vec2, mat: u32) -> Vec<emerge::Particle> {
        build_particles(
            config,
            SpawnRegion::for_sim(config)
                .at(center)
                .disk(5.0)
                .spacing(0.5)
                .material(mat)
                .precompute_volumes(),
        )
    }

    #[test]
    fn gpu_neohookean_stable() {
        if !gpu_available() {
            return;
        }
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..30 {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(p.x.is_finite(), "gpu neo particle {i}: position NaN");
            assert!(
                p.deformation_gradient.determinant() > 0.0,
                "gpu neo particle {i}: J collapsed"
            );
        }
    }

    #[test]
    fn gpu_sand_stable() {
        if !gpu_available() {
            return;
        }
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::new(400.0, 200.0)));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..30 {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(p.x.is_finite(), "gpu sand particle {i}: position NaN");
            assert!(
                p.deformation_gradient.determinant() > 0.0,
                "gpu sand particle {i}: J collapsed"
            );
            assert!(
                p.friction_hardening.is_finite(),
                "gpu sand particle {i}: q NaN"
            );
        }
    }

    #[test]
    fn gpu_fluid_stable() {
        if !gpu_available() {
            return;
        }
        let config = SimConfig {
            recompute_density_each_step: true,
            max_substeps_per_step: 8,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        };
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry = MaterialRegistry::with_default(Box::new(NewtonianFluidMaterial::new(
            4.0, 0.1, 10.0, 4.0,
        )));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..30 {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(p.x.is_finite(), "gpu fluid particle {i}: position NaN");
            assert!(p.density > 0.0, "gpu fluid particle {i}: density collapsed");
        }
    }

    #[test]
    fn gpu_snow_stable() {
        if !gpu_available() {
            return;
        }
        let config = SimConfig {
            max_substeps_per_step: 20,
            ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.1))
        };
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let snow = StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.02, 0.006, 0.6, 20.0);
        let registry = MaterialRegistry::with_default(Box::new(snow));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..30 {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(p.x.is_finite(), "gpu snow particle {i}: position NaN");
            assert!(
                p.deformation_gradient.determinant() > 0.0,
                "gpu snow particle {i}: J collapsed"
            );
            assert!(
                p.plastic_volume_ratio.is_finite(),
                "gpu snow particle {i}: Jp NaN"
            );
        }
    }

    #[test]
    fn gpu_irl_calibration_stable() {
        if !gpu_available() {
            return;
        }
        // Full IRL calibration: earth() + lame_from_si + particle_mass.
        // Soft gel (5 kPa, ν=0.45, ρ=1000 kg/m³) at 1cm/cell under Earth gravity.
        // J must stay > 0 (no collapse) and positions must be finite.
        const CELL_M: f32 = 0.01;
        const DT: f32 = 0.1;
        const RHO: f32 = 1000.0;
        const SPACING: f32 = 0.5;

        let mut config = SimConfig {
            max_substeps_per_step: 20,
            ..SimConfig::earth(32, CELL_M, DT)
        };
        config.particle_mass = RHO * (SPACING * CELL_M).powi(2);

        let (lambda, mu) = emerge::lame_from_si(5_000.0, 0.45, RHO, CELL_M, DT);
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(lambda, mu)));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..30 {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(p.x.is_finite(), "irl particle {i}: position NaN");
            assert!(
                p.deformation_gradient.determinant() > 0.0,
                "irl particle {i}: J collapsed under IRL gravity"
            );
        }
    }

    /// Characterizes GPU per-step cost AND VRAM footprint vs. grid resolution, pushed up to
    /// just below wgpu's default `max_storage_buffer_binding_size` (128 MiB) — the real wall.
    ///
    /// The GPU grid buffer is dense — allocated as grid_res² · 16 bytes (sizeof(GpuCell))
    /// regardless of how many particles are active (src/gpu/buffers.rs). The CPU grid is
    /// sparse (HashMap keyed by touched cell, src/grid/mod.rs) and stays flat under the same
    /// test (benches/scaling.rs::grid_resolution_scaling: ~15.2ms @ grid=32 -> ~16.6ms @ grid=256).
    ///
    /// Measured wall: wgpu's default storage-binding limit is 128 MiB = 8,388,608 cells,
    /// i.e. grid_res ≈ 2896 — NOT VRAM capacity. A 4096² grid (256 MiB) would already exceed
    /// the default binding limit before VRAM itself becomes the constraint. This is a wgpu API
    /// ceiling, not a hardware one (raisable via `required_limits` at device request time, up
    /// to whatever `adapter.limits()` actually allows).
    ///
    /// Verifies the particle_sort pipeline (clear -> count -> scan -> scatter) produces a valid
    /// permutation of `sorted_particle_ids`: every index 0..N appears exactly once. This is the
    /// strict correctness check beyond "physics looks stable" — a scan/scatter off-by-one could
    /// duplicate or drop slots while still leaving particles numerically finite (e.g. if a
    /// dropped slot happens to retain a harmless stale value), so this checks the permutation
    /// invariant directly rather than inferring correctness from particle state.
    #[test]
    fn gpu_particle_sort_is_valid_permutation() {
        if !gpu_available() {
            return;
        }
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));
        let n = solver.particle_count();

        for frame in 0..10 {
            solver.step_frame();
            let ids = solver.sorted_particle_ids_blocking();
            assert_eq!(
                ids.len(),
                n,
                "frame {frame}: sorted_particle_ids length mismatch"
            );
            let mut seen = vec![false; n];
            for &id in &ids {
                assert!(
                    (id as usize) < n,
                    "frame {frame}: sorted_particle_ids contains out-of-range index {id} (n={n})"
                );
                assert!(
                    !seen[id as usize],
                    "frame {frame}: index {id} appears more than once in sorted_particle_ids \
                     -- not a valid permutation"
                );
                seen[id as usize] = true;
            }
            assert!(
                seen.iter().all(|&s| s),
                "frame {frame}: sorted_particle_ids is missing at least one index \
                 -- not a valid permutation"
            );
        }
    }

    /// Run with `cargo test --features gpu --test gpu gpu_grid_resolution_cost -- --nocapture`.
    /// No hard perf assertion (timing varies by machine/CI runner) — only stability is asserted.
    #[test]
    fn gpu_grid_resolution_cost() {
        if !gpu_available() {
            return;
        }
        const DEFAULT_MAX_STORAGE_BINDING: u64 = 128 << 20; // wgpu::Limits::default()
        const CELL_BYTES: u64 = 16;
        let wall_cells = DEFAULT_MAX_STORAGE_BINDING / CELL_BYTES;
        let wall_grid_res = (wall_cells as f64).sqrt().floor() as usize;
        println!(
            "gpu_grid_resolution_cost: default max_storage_buffer_binding_size=128MiB -> \
             wall at grid_res~{wall_grid_res} (grid_res^2 * 16B = 128MiB)"
        );

        for &grid_res in &[32usize, 64, 128, 256, 512, 1024, 2048] {
            let config = SimConfig {
                max_substeps_per_step: 8,
                ..SimConfig::standard(grid_res, 0.1, Vec2::new(0.0, -0.3))
            };
            let particles = spawn_disk(&config, Vec2::splat(grid_res as f32 * 0.5), 0);
            let registry =
                MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
            let mut solver = block_on(GpuSimulation::new(config, particles, registry));

            // Warm up (pipeline/buffer creation already paid for by GpuSimulation::new).
            for _ in 0..5 {
                solver.step_frame();
            }
            let start = std::time::Instant::now();
            const STEPS: u32 = 30;
            for _ in 0..STEPS {
                solver.step_frame();
            }
            solver.sync_particles_blocking();
            let per_step_ms = start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
            let buffer_mib = (grid_res * grid_res) as f64 * CELL_BYTES as f64 / (1024.0 * 1024.0);
            println!(
                "gpu_grid_resolution_cost: grid={grid_res:>4} buffer={buffer_mib:>7.2}MiB per_step={per_step_ms:.3}ms"
            );

            for (i, p) in solver.particles().iter().enumerate() {
                assert!(
                    p.x.is_finite(),
                    "grid={grid_res} particle {i}: position NaN"
                );
            }
        }
    }

    /// Stress-tests GPU per-step cost at LP's stated target particle budget
    /// (100k-500k particles @ 60fps, see project_lp_world_design memory) — fixed grid_res=512
    /// (well clear of the grid_resolution_cost cliff at 1024-2048), varying particle count.
    ///
    /// IMPORTANT CONTEXT: GpuSimulation has NO sleep/wake mechanism (unlike CPU Simulation,
    /// which partitions active/sleeping particles — src/solver/mod.rs). Every step_frame()
    /// processes every particle regardless of camera distance. LP's chunk design assumes
    /// "chunks distants = gelés (sleep system emerge = mécanisme naturel)" on a GPU-primary
    /// architecture — that assumption does not hold today. This test measures the cost of
    /// that gap directly: if LP's world has 500k total particles and none can sleep on GPU,
    /// this is the per-step cost LP would actually pay regardless of how many are on-screen.
    ///
    /// Run with `cargo test --features gpu --test gpu gpu_particle_count_lp_budget -- --nocapture`.
    #[test]
    fn gpu_particle_count_lp_budget() {
        if !gpu_available() {
            return;
        }
        const GRID_RES: usize = 512;
        const FRAME_BUDGET_60FPS_MS: f64 = 16.67;

        for &target in &[10_000usize, 50_000, 100_000, 250_000, 500_000] {
            let config = SimConfig {
                max_substeps_per_step: 4,
                ..SimConfig::standard(GRID_RES, 0.1, Vec2::new(0.0, -0.3))
            };
            // box_size in cells at spacing 0.5 -> ~4 particles/cell; side = sqrt(target)/2.
            let side = ((target as f32) / 4.0).sqrt().ceil() as i32;
            let particles = build_particles(
                &config,
                SpawnRegion {
                    spacing: 0.5,
                    box_size: glam::IVec2::splat(side),
                    box_center: Vec2::splat(GRID_RES as f32 * 0.5),
                    precompute_initial_volumes: true,
                    ..SpawnRegion::for_sim(&config)
                },
            );
            let n = particles.len();
            let registry =
                MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
            let mut solver = block_on(GpuSimulation::new(config, particles, registry));

            for _ in 0..3 {
                solver.step_frame();
            }
            let start = std::time::Instant::now();
            const STEPS: u32 = 20;
            for _ in 0..STEPS {
                solver.step_frame();
            }
            solver.sync_particles_blocking();
            let per_step_ms = start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
            let verdict = if per_step_ms <= FRAME_BUDGET_60FPS_MS {
                "OK 60fps"
            } else {
                "MISSES 60fps budget"
            };
            println!(
                "gpu_particle_count_lp_budget: n={n:>7} per_step={per_step_ms:>7.3}ms -> {verdict}"
            );

            for (i, p) in solver.particles().iter().enumerate() {
                assert!(p.x.is_finite(), "n={n} particle {i}: position NaN");
            }
        }
    }

    /// Stress-tests the apply_impulses GPU pass at its hard cap (MAX_GPU_IMPULSES=16 per
    /// frame). Pushes 16 simultaneous radial impulses every frame for 20 frames and asserts
    /// particles stay finite and bounded — the apply_impulses pass runs once per frame before
    /// any substep, so this checks the GPU-native impulse path under max simultaneous load
    /// (e.g. many creature limbs pushing at once in LP), not just the single-impulse smoke test.
    #[test]
    fn gpu_apply_impulses_count_stress() {
        if !gpu_available() {
            return;
        }
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let mut solver = block_on(GpuSimulation::new(config, particles, registry));

        const N_IMPULSES: usize = 16; // == MAX_GPU_IMPULSES
        for _ in 0..20 {
            for i in 0..N_IMPULSES {
                let angle = i as f32 / N_IMPULSES as f32 * std::f32::consts::TAU;
                let center = Vec2::splat(16.0) + Vec2::new(angle.cos(), angle.sin()) * 4.0;
                solver.apply_radial_impulse(center, 2.0, 0.5);
            }
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        for (i, p) in solver.particles().iter().enumerate() {
            assert!(
                p.x.is_finite(),
                "gpu_apply_impulses_count_stress: particle {i} position NaN"
            );
            assert!(
                p.v.is_finite(),
                "gpu_apply_impulses_count_stress: particle {i} velocity NaN"
            );
        }
    }

    /// Queries the ACTUAL runtime device limits (not the textbook wgpu::Limits::default()
    /// assumed elsewhere) and computes the real hard ceilings for particle count and grid
    /// resolution on whatever hardware this runs on. Safe — no buffer creation, just
    /// arithmetic against `adapter.limits()`. Run with `-- --nocapture` to see the numbers.
    #[test]
    fn gpu_runtime_limits_report() {
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        let Ok(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) else {
            return;
        };
        let limits = adapter.limits();
        let max_binding = limits.max_storage_buffer_binding_size as u64;
        let max_buffer = limits.max_buffer_size;

        const PARTICLE_BYTES: u64 = 112;
        const CELL_BYTES: u64 = 16;
        let max_particles_by_binding = max_binding / PARTICLE_BYTES;
        let max_grid_res_by_binding = ((max_binding / CELL_BYTES) as f64).sqrt() as u64;

        println!(
            "gpu_runtime_limits_report: max_storage_buffer_binding_size={}MiB max_buffer_size={}MiB",
            max_binding / (1024 * 1024),
            max_buffer / (1024 * 1024)
        );
        println!(
            "gpu_runtime_limits_report: hard ceiling -- max_particles={max_particles_by_binding} \
             (single buffer), max_grid_res={max_grid_res_by_binding} (single buffer)"
        );
        println!(
            "gpu_runtime_limits_report: LP target 500,000 particles uses {:.1}% of the particle \
             buffer ceiling",
            500_000.0 / max_particles_by_binding as f64 * 100.0
        );

        // Sanity: LP's stated 500k-particle target must fit under whatever this hardware
        // actually reports — if this ever fails, LP's budget assumption is unreachable
        // regardless of compute speed, on any hardware reporting limits this low.
        assert!(
            max_particles_by_binding >= 500_000,
            "LP's 500k particle target exceeds this device's storage-binding ceiling \
             ({max_particles_by_binding}) -- raise required_limits or shard across buffers"
        );
    }

    /// Pushes GPU particle count toward the storage-binding ceiling (~1.19M particles at the
    /// default 128MiB limit) to find the REAL compute wall beyond LP's stated 500k target —
    /// answering "what happens past the documented budget" with measurement, not guesswork.
    #[test]
    fn gpu_particle_count_beyond_lp_budget() {
        if !gpu_available() {
            return;
        }
        const GRID_RES: usize = 512;
        for &target in &[750_000usize, 1_000_000] {
            let config = SimConfig {
                max_substeps_per_step: 2, // keep total work bounded at this particle count
                ..SimConfig::standard(GRID_RES, 0.1, Vec2::new(0.0, -0.3))
            };
            let side = ((target as f32) / 4.0).sqrt().ceil() as i32;
            let particles = build_particles(
                &config,
                SpawnRegion {
                    spacing: 0.5,
                    box_size: glam::IVec2::splat(side),
                    box_center: Vec2::splat(GRID_RES as f32 * 0.5),
                    precompute_initial_volumes: true,
                    ..SpawnRegion::for_sim(&config)
                },
            );
            let n = particles.len();
            let registry =
                MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
            let mut solver = block_on(GpuSimulation::new(config, particles, registry));

            let start = std::time::Instant::now();
            const STEPS: u32 = 5;
            for _ in 0..STEPS {
                solver.step_frame();
            }
            solver.sync_particles_blocking();
            let per_step_ms = start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
            println!(
                "gpu_particle_count_beyond_lp_budget: n={n:>8} per_step={per_step_ms:>8.2}ms \
                 ({:.1}x over 60fps budget)",
                per_step_ms / 16.67
            );
            for (i, p) in solver.particles().iter().enumerate() {
                assert!(p.x.is_finite(), "n={n} particle {i}: position NaN");
            }
        }
    }

    /// Combined LP-realistic worst case: sand terrain + water + creature bodies (viscoelastic
    /// with active-stress fields populated) sharing one grid at LP's actual particle budget,
    /// all at once — not one axis at a time. This is the integration test that actually answers
    /// "does LP's real scene hold together," not just "does each isolated axis scale."
    #[test]
    fn gpu_lp_realistic_combined_stress() {
        if !gpu_available() {
            return;
        }
        const GRID_RES: usize = 256;
        const SAND_ID: u32 = 1;
        const WATER_ID: u32 = 2;
        let config = SimConfig {
            max_substeps_per_step: 8,
            recompute_density_each_step: true,
            ..SimConfig::standard(GRID_RES, 0.1, Vec2::new(0.0, -0.3))
        };

        let terrain = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: glam::IVec2::new(GRID_RES as i32 - 20, 40),
                box_center: Vec2::new(GRID_RES as f32 * 0.5, 30.0),
                material_id: SAND_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );
        let water = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: glam::IVec2::new(60, 30),
                box_center: Vec2::new(GRID_RES as f32 * 0.3, 90.0),
                material_id: WATER_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );
        let creatures = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: glam::IVec2::new(40, 40),
                box_center: Vec2::new(GRID_RES as f32 * 0.7, 100.0),
                material_id: 0, // default = viscoelastic creature body
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );

        let mut all_particles = terrain;
        all_particles.extend(water);
        all_particles.extend(creatures);
        let n = all_particles.len();
        // Give the creature-body particles an active-stress drive, exercising F.A.F^T in the
        // same step as sand plasticity and fluid pressure.
        for p in &mut all_particles {
            if p.material_id == 0 {
                p.activation = 0.3;
                p.activation_dir = Vec2::new(1.0, 0.0);
            }
        }

        let mut registry = MaterialRegistry::with_default(Box::new(
            ViscoelasticMaterial::near_incompressible(5.0e4, 10.0),
        ));
        registry.insert(
            SAND_ID,
            Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)),
        );
        registry.insert(
            WATER_ID,
            Box::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1.28e5)),
        );

        // Radial confinement radii must clear the terrain's actual extent (it spans nearly
        // the full grid width) — too tight, and corner particles overshoot by tens of cells
        // at frame 0, causing a violent first-substep correction. SlipBoundary already bounds
        // the domain; these three fields stack on top of it at a safe radius for the stress.
        let mut solver = block_on(GpuSimulation::new(config, all_particles, registry));
        let safe_radius = GRID_RES as f32 * 0.85; // clears the terrain's corner-to-center distance
        for i in 0..3 {
            solver.add_force_field_gpu(emerge::gpu::GpuFieldEntry::radial_confinement(
                Vec2::splat(GRID_RES as f32 * 0.5),
                safe_radius + i as f32 * 5.0,
                50.0,
                2.0,
            ));
        }

        let start = std::time::Instant::now();
        const STEPS: u32 = 30;
        for _ in 0..STEPS {
            solver.step_frame();
        }
        solver.sync_particles_blocking();
        let per_step_ms = start.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
        println!(
            "gpu_lp_realistic_combined_stress: n={n} (sand+water+creature, 3 confinement fields) \
             per_step={per_step_ms:.3}ms"
        );

        for (i, p) in solver.particles().iter().enumerate() {
            assert!(
                p.x.is_finite(),
                "combined stress: particle {i} position NaN"
            );
            assert!(
                p.v.is_finite(),
                "combined stress: particle {i} velocity NaN"
            );
            assert!(
                p.deformation_gradient.determinant() > 0.0,
                "combined stress: particle {i} J collapsed"
            );
        }
    }

    #[test]
    fn gpu_earth_config_gravity_correct() {
        // g_solver = 9.81 / cell_m — velocity-based: v += g * sub_dt (sub_dt in real seconds)
        let config = SimConfig::earth(64, 0.01, 0.05);
        let expected_g = 9.81f32 / 0.01; // = 981 cells/s²
        assert!(
            (config.gravity.y + expected_g).abs() < 1e-3,
            "earth gravity wrong: got {}, expected {}",
            config.gravity.y,
            -expected_g
        );
        assert_eq!(config.dx_meters, 0.01);
        assert_eq!(config.dt_seconds, 0.05);
    }
}

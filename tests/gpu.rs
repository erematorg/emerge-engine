//! GPU solver smoke tests — basic stability checks for GpuSolver.
//!
//! These tests run headlessly (no window, no Bevy) using pollster::block_on.
//! They verify that the GPU pipeline doesn't crash or produce NaN on standard
//! material configurations.

#[cfg(feature = "gpu")]
mod gpu_tests {
    use emerge::gpu::GpuSolver;
    use emerge::{
        MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial, SandMaterial, SnowMaterial,
        SolverConfig, SpawnConfig, build_particles,
    };
    use glam::Vec2;
    use pollster::block_on;

    fn small_config() -> SolverConfig {
        SolverConfig {
            max_substeps_per_step: 8,
            ..SolverConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        }
    }

    fn spawn_disk(config: &SolverConfig, center: Vec2, mat: u32) -> Vec<emerge::Particle> {
        build_particles(
            config,
            SpawnConfig::for_solver(config)
                .at(center)
                .disk(5.0)
                .spacing(0.5)
                .material(mat)
                .precompute_volumes(),
        )
    }

    #[test]
    fn gpu_neohookean_stable() {
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let mut solver = block_on(GpuSolver::new(config, particles, registry));
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
        let config = small_config();
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(SandMaterial::new(400.0, 200.0)));
        let mut solver = block_on(GpuSolver::new(config, particles, registry));
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
        let config = SolverConfig {
            recompute_density_each_step: true,
            max_substeps_per_step: 8,
            ..SolverConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
        };
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry = MaterialRegistry::with_default(Box::new(
            NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0),
        ));
        let mut solver = block_on(GpuSolver::new(config, particles, registry));
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
        let config = SolverConfig {
            max_substeps_per_step: 20,
            ..SolverConfig::standard(32, 0.1, Vec2::new(0.0, -0.1))
        };
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let snow = SnowMaterial::new(1389.0, 2083.0, 10.0, 0.02, 0.006, 0.6, 20.0);
        let registry = MaterialRegistry::with_default(Box::new(snow));
        let mut solver = block_on(GpuSolver::new(config, particles, registry));
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
        // Full IRL calibration: earth() + lame_from_si + particle_mass.
        // Soft gel (5 kPa, ν=0.45, ρ=1000 kg/m³) at 1cm/cell under Earth gravity.
        // J must stay > 0 (no collapse) and positions must be finite.
        const CELL_M: f32 = 0.01;
        const DT: f32 = 0.1;
        const RHO: f32 = 1000.0;
        const SPACING: f32 = 0.5;

        let mut config = SolverConfig {
            max_substeps_per_step: 20,
            ..SolverConfig::earth(32, CELL_M, DT)
        };
        config.particle_mass = RHO * (SPACING * CELL_M).powi(2);

        let (lambda, mu) = emerge::lame_from_si(5_000.0, 0.45, RHO, CELL_M, DT);
        let particles = spawn_disk(&config, Vec2::splat(16.0), 0);
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(lambda, mu)));
        let mut solver = block_on(GpuSolver::new(config, particles, registry));
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

    #[test]
    fn gpu_earth_config_gravity_correct() {
        // g_solver = 9.81 / cell_m — velocity-based: v += g * sub_dt (sub_dt in real seconds)
        let config = SolverConfig::earth(64, 0.01, 0.05);
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

extern crate emerge_engine as emerge;

/// Real, permanent perf-scaling diagnostic (2026-08-04): direct user request
/// for a grounded "how many particles/materials for a viable 60fps scene"
/// table -- built from ACTUAL measured wall-clock step cost at several
/// particle counts, not guessed/extrapolated from one data point. CPU-only,
/// debug-mode (matches this project's own standing "debug only for dev"
/// rule -- these numbers are the honest baseline, not an inflated release
/// number). `#[ignore]`d (real wall-clock timing, not a correctness test --
/// same convention as every other perf diagnostic in this project) --
/// run manually: `cargo test --test perf_particle_scaling -- --ignored --nocapture`.
use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::time::Instant;

/// Real, single-material (sand alone) scaling curve -- the honest floor,
/// no mixture/multi-material overhead included.
#[test]
#[ignore = "real wall-clock perf diagnostic, run manually"]
fn diag_sand_single_material_particle_scaling() {
    const GRID: usize = 256;
    println!("── Single-material sand (DruckerPrager), CPU, debug-mode wall-clock ──");
    println!("  particle_count  avg_ms_per_step  implied_fps");
    for &side in &[16i32, 32, 50, 71, 100, 141] {
        let config = SimConfig {
            material_cfl_coefficient: 0.7,
            ..SimConfig::earth(GRID, 0.01, 0.1)
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(side, side),
            box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.5),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let sand = DruckerPragerMaterial::new(10_000.0, 15_000.0);
        let mut solver = Simulation::new(config, spawn)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let n = solver.particles().len();

        // Real settle window (not timed) -- avoid measuring the initial
        // free-fall/impact transient's own atypical substep count.
        for _ in 0..20 {
            solver.step();
        }
        const TIMED_STEPS: u32 = 60;
        let start = Instant::now();
        for _ in 0..TIMED_STEPS {
            solver.step();
        }
        let elapsed = start.elapsed();
        let ms_per_step = elapsed.as_secs_f64() * 1000.0 / TIMED_STEPS as f64;
        let implied_fps = 1000.0 / ms_per_step;
        println!("  {n:>14}  {ms_per_step:>15.3}  {implied_fps:>11.1}");
    }
}

/// Real two-material mixture scaling (sand+water, `WithMixturePhase`) at a
/// couple of representative particle counts -- gives a real, measured
/// multi-material overhead FACTOR to apply against the single-material
/// curve above, instead of guessing how much combining materials costs.
#[test]
#[ignore = "real wall-clock perf diagnostic, run manually"]
fn diag_mixture_two_material_particle_scaling() {
    use emerge::{MixturePhase, NewtonianFluidMaterial, WithMixturePhase};

    const GRID: usize = 256;
    println!("── Two-material mixture (sand+water, WithMixturePhase), CPU, debug-mode ──");
    println!("  particle_count  avg_ms_per_step  implied_fps");
    for &side in &[16i32, 32, 50] {
        let config = SimConfig {
            gravity: Vec2::new(0.0, -0.3),
            mixture_drag_coefficient: 30.0,
            mixture_pressure_iterations: 8,
            ..SimConfig::earth(GRID, 0.01, 0.1)
        };
        let spawn_sand = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(side, side),
            box_center: Vec2::new(GRID as f32 * 0.4, GRID as f32 * 0.3),
            material_id: 0,
            precompute_initial_volumes: true,
            mass_override: Some(1.8),
            ..SpawnRegion::for_sim(&config)
        };
        let spawn_water = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(side, side),
            box_center: Vec2::new(GRID as f32 * 0.6, GRID as f32 * 0.6),
            material_id: 1,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let sand = WithMixturePhase::new(
            DruckerPragerMaterial::new(10_000.0, 15_000.0),
            MixturePhase::SOLID,
        );
        let water = WithMixturePhase::new(
            NewtonianFluidMaterial::low_viscosity(4.0, 10.0),
            MixturePhase::FLUID,
        );
        let mut solver = Simulation::new(config, spawn_sand)
            .with_default_material(Box::new(sand))
            .with_material(1, Box::new(water))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let _ = solver.add_body(spawn_water);
        let n = solver.particles().len();

        for _ in 0..20 {
            solver.step();
        }
        const TIMED_STEPS: u32 = 60;
        let start = Instant::now();
        for _ in 0..TIMED_STEPS {
            solver.step();
        }
        let elapsed = start.elapsed();
        let ms_per_step = elapsed.as_secs_f64() * 1000.0 / TIMED_STEPS as f64;
        let implied_fps = 1000.0 / ms_per_step;
        println!("  {n:>14}  {ms_per_step:>15.3}  {implied_fps:>11.1}");
    }
}

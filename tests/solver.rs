use emerge::solver::{
    MpmSolver, NeoHookeanMaterial, NewtonianFluidMaterial, SolverConfig, SpawnConfig,
};
use glam::{IVec2, Vec2};

// --- helpers ---

fn small_solver_config() -> SolverConfig {
    SolverConfig {
        grid_res: 32,
        dt: 0.1,
        adaptive_timestep: true,
        ..SolverConfig::default()
    }
}

fn small_spawn_config(center: f32) -> SpawnConfig {
    SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::new(8, 8),
        box_center: Vec2::splat(center),
        initial_velocity_offset: Vec2::ZERO,
        initial_velocity_scale: 0.0,
        ..SpawnConfig::default()
    }
}

// --- boundary ---

#[test]
fn step_keeps_particles_inside_domain() {
    let mut solver = MpmSolver::new(SolverConfig::default(), SpawnConfig::default());
    solver.step_n(20);
    let min = solver.config().boundary_thickness.saturating_sub(1) as f32;
    let max = solver
        .config()
        .grid_res
        .saturating_sub(solver.config().boundary_thickness) as f32;
    for p in solver.particles() {
        assert!(p.x.x >= min && p.x.x <= max);
        assert!(p.x.y >= min && p.x.y <= max);
    }
}

#[test]
fn precomputed_volumes_are_positive() {
    let spawn = SpawnConfig {
        precompute_initial_volumes: true,
        ..SpawnConfig::default()
    };
    let solver = MpmSolver::new(SolverConfig::default(), spawn);
    for p in solver.particles() {
        assert!(p.initial_volume > 0.0);
    }
}

// --- stability regression ---

#[test]
fn jelly_stable_after_many_steps() {
    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.x.is_finite(),
            "particle {i}: position non-finite after jelly sim"
        );
        assert!(
            p.v.is_finite(),
            "particle {i}: velocity non-finite after jelly sim"
        );
        let j = p.deformation_gradient.determinant();
        assert!(
            j > 0.0,
            "particle {i}: deformation collapsed (J={j}) after jelly sim"
        );
    }
}

#[test]
fn fluid_stable_after_many_steps() {
    let solver_config = SolverConfig {
        recompute_density_each_step: true,
        ..small_solver_config()
    };
    let mut solver = MpmSolver::new(solver_config, small_spawn_config(16.0))
        .with_default_material(Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.x.is_finite(),
            "particle {i}: position non-finite after fluid sim"
        );
        assert!(
            p.v.is_finite(),
            "particle {i}: velocity non-finite after fluid sim"
        );
        assert!(
            p.density > 0.0,
            "particle {i}: density collapsed after fluid sim"
        );
    }
}

#[test]
fn spawn_for_solver_adapts_center_to_grid_resolution() {
    let config = SolverConfig {
        grid_res: 128,
        ..SolverConfig::default()
    };
    let spawn = SpawnConfig::for_solver(&config);
    assert_eq!(spawn.box_center, Vec2::splat(64.0));
}

#[test]
fn small_grid_validation_is_consistent_with_grid_constructor() {
    let config = SolverConfig {
        grid_res: 3,
        ..SolverConfig::default()
    };
    let spawn = SpawnConfig::for_solver(&config);
    let result = std::panic::catch_unwind(|| {
        let _ = MpmSolver::new(config, spawn);
    });
    assert!(result.is_err(), "grid_res=3 should fail validation");
}

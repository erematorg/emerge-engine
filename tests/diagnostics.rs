use emerge::diagnostics::{MpmHealthThresholds, collect_mpm_snapshot, evaluate_mpm_health};
use emerge::solver::SolverConfig;
use emerge::state::{grid::Grid, particle::Particle};
use glam::{IVec2, Mat2, Vec2};

fn make_config(grid_res: usize) -> SolverConfig {
    SolverConfig {
        grid_res,
        grid_cell_size: 1.0,
        dt: 0.1,
        adaptive_timestep: false,
        cfl_include_affine_speed: true,
        cfl_coefficient: 0.9,
        material_cfl_coefficient: 0.5,
        viscous_timestep_coefficient: 0.5,
        min_dt: 1.0e-3,
        project_invalid_state: true,
        projection_min_density: 1.0e-6,
        projection_min_volume: 1.0e-6,
        projection_min_deformation_j: 1.0e-6,
        gravity: Vec2::new(0.0, -0.3),
        boundary_thickness: 2,
        default_initial_volume: 1.0,
        recompute_density_each_step: false,
        particle_mass: 1.0,
        d_inverse: 4.0,
        max_substeps_per_step: 64,
    }
}

#[test]
fn collect_and_evaluate_basic_snapshot() {
    let config = make_config(8);

    let particles = vec![Particle {
        x: Vec2::new(4.0, 4.0),
        v: Vec2::new(1.0, 0.0),
        affine: Mat2::ZERO,
        deformation_gradient: Mat2::IDENTITY,
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        material_id: 0,
        plastic_jacobian: 1.0,
        elastic_hardening: 1.0,
        plastic_hardening: 0.0,
        log_vol_gain: 0.0,
        temperature: 0.0,
        user_tag: 0,
        _pad: 0.0,
    }];

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 1.0, Vec2::new(1.0, 0.0));

    let snapshot = collect_mpm_snapshot(7, &particles, &grid, &config, config.dt, 1);
    assert_eq!(snapshot.frame_index, 7);
    assert_eq!(snapshot.particle_count, 1);
    assert_eq!(snapshot.active_grid_cells, 1);
    assert_eq!(snapshot.out_of_bounds_particles, 0);
    assert_eq!(snapshot.non_finite_particle_values, 0);
    assert_eq!(snapshot.non_finite_grid_values, 0);
    assert!(snapshot.relative_mass_error <= f32::EPSILON);
    assert!(snapshot.mixed_material_cell_ratio <= f32::EPSILON);
    assert!(snapshot.mixed_material_particle_ratio <= f32::EPSILON);

    let status = evaluate_mpm_health(&snapshot, &MpmHealthThresholds::default());
    assert!(status.healthy());
}

#[test]
fn empty_particle_snapshot_is_unhealthy() {
    let config = make_config(8);
    let grid = Grid::new(config.grid_res);

    let snapshot = collect_mpm_snapshot(0, &[], &grid, &config, config.dt, 0);
    let status = evaluate_mpm_health(&snapshot, &MpmHealthThresholds::default());
    assert!(!status.healthy());
    assert!(status.particle_count_violation);
}

#[test]
fn concentrated_particles_trigger_violation() {
    let config = make_config(8);

    let particles = vec![
        Particle {
            x: Vec2::new(4.0, 4.0),
            v: Vec2::ZERO,
            affine: Mat2::ZERO,
            deformation_gradient: Mat2::IDENTITY,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_jacobian: 1.0,
            elastic_hardening: 1.0,
            plastic_hardening: 0.0,
            log_vol_gain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            _pad: 0.0,
        };
        128
    ];

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 128.0, Vec2::ZERO);

    let snapshot = collect_mpm_snapshot(0, &particles, &grid, &config, config.dt, 1);
    let thresholds = MpmHealthThresholds {
        max_particles_per_active_cell: 64.0,
        ..MpmHealthThresholds::default()
    };
    let status = evaluate_mpm_health(&snapshot, &thresholds);

    assert!(!status.healthy());
    assert!(status.cell_concentration_violation);
}

#[test]
fn mixed_material_ratio_detects_cell_level_blending() {
    let config = make_config(8);

    let particles = vec![
        Particle {
            x: Vec2::new(4.1, 4.1),
            v: Vec2::ZERO,
            affine: Mat2::ZERO,
            deformation_gradient: Mat2::IDENTITY,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_jacobian: 1.0,
            elastic_hardening: 1.0,
            plastic_hardening: 0.0,
            log_vol_gain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            _pad: 0.0,
        },
        Particle {
            x: Vec2::new(4.3, 4.2),
            v: Vec2::ZERO,
            affine: Mat2::ZERO,
            deformation_gradient: Mat2::IDENTITY,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 1,
            plastic_jacobian: 1.0,
            elastic_hardening: 1.0,
            plastic_hardening: 0.0,
            log_vol_gain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            _pad: 0.0,
        },
    ];

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 2.0, Vec2::ZERO);

    let snapshot = collect_mpm_snapshot(0, &particles, &grid, &config, config.dt, 1);
    assert!(snapshot.mixed_material_cell_ratio > 0.9);
    assert!(snapshot.mixed_material_particle_ratio > 0.9);

    let strict = MpmHealthThresholds {
        max_mixed_material_cell_ratio: 0.2,
        max_mixed_material_particle_ratio: 0.2,
        ..MpmHealthThresholds::default()
    };
    let strict_status = evaluate_mpm_health(&snapshot, &strict);
    assert!(strict_status.mixed_material_violation);
}

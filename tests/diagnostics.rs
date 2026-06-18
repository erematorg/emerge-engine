extern crate emerge_engine as emerge;

use emerge::SimConfig;
use emerge::diagnostics::{StabilityThresholds, collect_snapshot, evaluate_stability};
use emerge::{
    grid::Grid,
    particle::{Particle, Particles},
};
use glam::{IVec2, Vec2};

fn make_config(grid_res: usize) -> SimConfig {
    SimConfig {
        grid_res,
        dt: 0.1,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::default()
    }
}

#[test]
fn collect_and_evaluate_basic_snapshot() {
    let config = make_config(8);

    let particles = Particles::from(vec![Particle {
        x: Vec2::new(4.0, 4.0),
        v: Vec2::new(1.0, 0.0),
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        ..Particle::zeroed()
    }]);

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 1.0, Vec2::new(1.0, 0.0));

    let snapshot = collect_snapshot(7, &particles, &grid, &config, config.dt, 1);
    assert_eq!(snapshot.frame_index, 7);
    assert_eq!(snapshot.particle_count, 1);
    assert_eq!(snapshot.active_grid_cells, 1);
    assert_eq!(snapshot.out_of_bounds_particles, 0);
    assert_eq!(snapshot.non_finite_particle_values, 0);
    assert_eq!(snapshot.non_finite_grid_values, 0);
    assert!(snapshot.relative_mass_error <= f32::EPSILON);
    assert!(snapshot.mixed_material_cell_ratio <= f32::EPSILON);
    assert!(snapshot.mixed_material_particle_ratio <= f32::EPSILON);

    let status = evaluate_stability(&snapshot, &StabilityThresholds::default());
    assert!(status.healthy());
}

#[test]
fn empty_particle_snapshot_is_unhealthy() {
    let config = make_config(8);
    let grid = Grid::new(config.grid_res);

    let snapshot = collect_snapshot(0, &Particles::new(), &grid, &config, config.dt, 0);
    let status = evaluate_stability(&snapshot, &StabilityThresholds::default());
    assert!(!status.healthy());
    assert!(status.particle_count_violation);
}

#[test]
fn concentrated_particles_trigger_violation() {
    let config = make_config(8);

    let particles = Particles::from(vec![
        Particle {
            x: Vec2::new(4.0, 4.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            ..Particle::zeroed()
        };
        128
    ]);

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 128.0, Vec2::ZERO);

    let snapshot = collect_snapshot(0, &particles, &grid, &config, config.dt, 1);
    let thresholds = StabilityThresholds {
        max_particles_per_active_cell: 64.0,
        ..StabilityThresholds::default()
    };
    let status = evaluate_stability(&snapshot, &thresholds);

    assert!(!status.healthy());
    assert!(status.cell_concentration_violation);
}

#[test]
fn mixed_material_ratio_detects_cell_level_blending() {
    let config = make_config(8);

    let particles = Particles::from(vec![
        Particle {
            x: Vec2::new(4.1, 4.1),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            ..Particle::zeroed()
        },
        Particle {
            x: Vec2::new(4.3, 4.2),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 1,
            ..Particle::zeroed()
        },
    ]);

    let mut grid = Grid::new(config.grid_res);
    grid.add_mass_momentum(IVec2::new(4, 4), 2.0, Vec2::ZERO);

    let snapshot = collect_snapshot(0, &particles, &grid, &config, config.dt, 1);
    assert!(snapshot.mixed_material_cell_ratio > 0.9);
    assert!(snapshot.mixed_material_particle_ratio > 0.9);

    let strict = StabilityThresholds {
        max_mixed_material_cell_ratio: 0.2,
        max_mixed_material_particle_ratio: 0.2,
        ..StabilityThresholds::default()
    };
    let strict_status = evaluate_stability(&snapshot, &strict);
    assert!(strict_status.mixed_material_violation);
}

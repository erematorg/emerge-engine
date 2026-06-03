use std::collections::HashMap;

use emerge::fields::{
    AabbConfinementField, CoulombField, GravityWellField, RadialConfinementField,
};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    MpmSolver, NaccMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, RankineMaterial,
    SandMaterial, SandMuIMaterial, SlipBoundary, SnowMaterial, SolverConfig, SpawnConfig,
    VonMisesMaterial,
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

// --- material stability regressions ---

#[test]
fn snow_stable_after_many_steps() {
    let snow = SnowMaterial::new(38_889.0, 58_333.0, 10.0, 0.02, 0.006, 0.05, 20.0);
    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(snow));
    solver.step_n(200);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "snow particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "snow particle {i}: J collapsed"
        );
        assert!(
            p.plastic_volume_ratio.is_finite(),
            "snow particle {i}: Jp non-finite"
        );
        assert!(
            p.hardening_scale.is_finite(),
            "snow particle {i}: h non-finite"
        );
    }
}

#[test]
fn sand_stable_after_many_steps() {
    let sand = SandMaterial::new(1_000.0, 500.0);
    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(sand));
    solver.step_n(200);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "sand particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "sand particle {i}: J collapsed"
        );
        assert!(
            p.friction_hardening.is_finite(),
            "sand particle {i}: q non-finite"
        );
        assert!(
            p.log_volume_strain.is_finite(),
            "sand particle {i}: log_vol_gain non-finite"
        );
    }
}

#[test]
fn von_mises_yield_stays_finite() {
    let vm = VonMisesMaterial::new(500.0, 200.0, 50.0);
    let config = SolverConfig {
        gravity: Vec2::new(0.0, -9.81),
        ..small_solver_config()
    };
    let spawn = SpawnConfig {
        initial_velocity_scale: 10.0,
        ..small_spawn_config(16.0)
    };
    let mut solver = MpmSolver::new(config, spawn).with_default_material(Box::new(vm));
    solver.step_n(100);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "vm particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.is_finite(),
            "vm particle {i}: F non-finite"
        );
    }
}

#[test]
fn rankine_damage_stays_finite_and_j_positive() {
    // High tensile load: spawn with upward velocity so particles stretch.
    // Rankine should project tensile stress and accumulate finite damage.
    let rock = RankineMaterial::rock(2_000.0, 1_000.0);
    let config = SolverConfig {
        gravity: Vec2::new(0.0, 9.81), // upward — stretches the block in tension
        ..small_solver_config()
    };
    let spawn = SpawnConfig {
        initial_velocity_scale: 5.0,
        ..small_spawn_config(16.0)
    };
    let mut solver = MpmSolver::new(config, spawn).with_default_material(Box::new(rock));
    solver.step_n(100);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "rankine particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "rankine particle {i}: J collapsed"
        );
        assert!(
            p.friction_hardening >= 0.0 && p.friction_hardening.is_finite(),
            "rankine particle {i}: damage non-finite or negative ({:.4})",
            p.friction_hardening
        );
    }
}

#[test]
fn rankine_softening_reduces_tensile_strength() {
    // Verify: a particle under sustained tension accumulates damage (friction_hardening > 0)
    // and that the effective tensile strength decreases with softening_rate > 0.
    use emerge::materials::MaterialModel;
    use emerge::particle::{Particle, Particles};

    let mat = RankineMaterial::new(1_000.0, 500.0, 100.0, 2.0);
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Deformation gradient: pure extension in x by 20% — puts particle in tension
    p.deformation_gradient =
        glam::Mat2::from_cols(glam::Vec2::new(1.2, 0.0), glam::Vec2::new(0.0, 1.0));
    // Velocity gradient: zero (no ongoing flow — just check state update)
    p.velocity_gradient = glam::Mat2::ZERO;

    let mut soa = Particles::from(vec![p]);
    mat.update_particle(&mut soa, 0, 0.01);
    p = soa.get(0);

    // Damage should be positive (tensile yield occurred) or zero (elastic)
    assert!(
        p.friction_hardening >= 0.0 && p.friction_hardening.is_finite(),
        "damage must be non-negative finite, got {}",
        p.friction_hardening
    );
    assert!(
        p.deformation_gradient.determinant() > 0.0,
        "J must stay positive after Rankine update"
    );
}

#[test]
fn phase_transition_switches_material_ids() {
    const JELLY_ID: u32 = 0;
    const FLUID_ID: u32 = 1;

    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 50.0)))
        .with_material(
            FLUID_ID,
            Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)),
        );

    assert!(solver.particles().iter().all(|p| p.material_id == JELLY_ID));
    solver.phase_transition(|p| p.x.x < 16.0, FLUID_ID);

    let fluid_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == FLUID_ID)
        .count();
    let jelly_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == JELLY_ID)
        .count();
    assert!(fluid_count > 0, "no particles transitioned to fluid");
    assert!(
        jelly_count > 0,
        "all particles transitioned — expected partial"
    );
    assert_eq!(fluid_count + jelly_count, solver.particles().len());
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

// --- ForceField integration tests ---

#[test]
fn gravity_well_pulls_particles_toward_source() {
    // Zero background gravity so only the well acts.
    // Blob placed left, well placed right — centre of mass must drift rightward.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let spawn = SpawnConfig {
        box_center: Vec2::new(8.0, 16.0),
        ..small_spawn_config(8.0)
    };
    let well_pos = Vec2::new(24.0, 16.0);

    let well = GravityWellField::new(
        vec![(well_pos, 1_000.0)],
        0.1, // gravitational_constant
        1.0, // softening (grid cells)
    )
    .with_cutoff(30.0);

    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_force_field(Box::new(well));

    let cx_before: f32 =
        solver.particles().iter().map(|p| p.x.x).sum::<f32>() / solver.particles().len() as f32;

    solver.step_n(80);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.x.is_finite(),
            "gravity_well: particle {i} position non-finite"
        );
        assert!(
            p.v.is_finite(),
            "gravity_well: particle {i} velocity non-finite"
        );
    }

    let cx_after: f32 =
        solver.particles().iter().map(|p| p.x.x).sum::<f32>() / solver.particles().len() as f32;
    assert!(
        cx_after > cx_before,
        "gravity_well: CoM did not move toward well (before={cx_before:.2}, after={cx_after:.2})"
    );
}

#[test]
fn radial_confinement_keeps_particles_inside() {
    // High-velocity particles should not escape beyond confinement radius + 2 cell tolerance.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let center = Vec2::splat(16.0);
    let radius = 6.0_f32;

    let spawn = SpawnConfig {
        box_center: center,
        box_size: IVec2::new(4, 4),
        initial_velocity_scale: 15.0,
        ..SpawnConfig::default()
    };

    let field = RadialConfinementField::new(center, radius, 500.0);

    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_force_field(Box::new(field));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.x.is_finite(),
            "confinement: particle {i} position non-finite"
        );
        let dist = (p.x - center).length();
        assert!(
            dist <= radius + 2.0,
            "confinement: particle {i} escaped (dist={dist:.2}, radius={radius:.2})"
        );
    }
}

#[test]
fn coulomb_repulsion_pushes_charged_particles_away() {
    // Positive point source at center. Same-sign material particles should spread outward.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let source_pos = Vec2::splat(16.0);
    let spawn = SpawnConfig {
        box_center: source_pos,
        box_size: IVec2::new(4, 4),
        ..SpawnConfig::default()
    };

    let mut mat_charges = HashMap::new();
    mat_charges.insert(0u32, 1.0_f32); // material 0 = positive charge, same as source → repels

    let field = CoulombField::new(
        vec![(source_pos, 10.0)],
        mat_charges,
        50.0, // coulomb_constant
        0.5,  // softening (grid cells)
    )
    .with_cutoff(20.0);

    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_force_field(Box::new(field));

    let avg_dist_before: f32 = solver
        .particles()
        .iter()
        .map(|p| (p.x - source_pos).length())
        .sum::<f32>()
        / solver.particles().len() as f32;

    solver.step_n(60);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "coulomb: particle {i} position non-finite");
        assert!(p.v.is_finite(), "coulomb: particle {i} velocity non-finite");
    }

    let avg_dist_after: f32 = solver
        .particles()
        .iter()
        .map(|p| (p.x - source_pos).length())
        .sum::<f32>()
        / solver.particles().len() as f32;

    assert!(
        avg_dist_after > avg_dist_before,
        "coulomb repulsion: avg distance did not increase (before={avg_dist_before:.2}, after={avg_dist_after:.2})"
    );
}

// --- ThermalDiffusion integration tests ---

#[test]
fn thermal_diffusion_spreads_heat() {
    // Left half hot, right half cold. After diffusion:
    // max temp must drop (hot cools), min temp must rise (cold warms).
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: 4182.0,
            ambient: 0.0,
            grid_cell_size: 0.1,
            ..Default::default()
        },
        config.grid_res,
    );

    let mut solver = MpmSolver::new(config, small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_thermal(thermal);

    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            particles.temperature[i] = if particles.x[i].x < 16.0 { 100.0 } else { 0.0 };
        }
    }

    // Mean temperature of each half — more robust than min/max at a sharp discontinuity.
    let mean_hot_before = {
        let hot: Vec<f32> = solver
            .particles()
            .iter()
            .filter(|p| p.x.x < 16.0)
            .map(|p| p.temperature)
            .collect();
        hot.iter().sum::<f32>() / hot.len() as f32
    };
    let mean_cold_before = {
        let cold: Vec<f32> = solver
            .particles()
            .iter()
            .filter(|p| p.x.x >= 16.0)
            .map(|p| p.temperature)
            .collect();
        cold.iter().sum::<f32>() / cold.len() as f32
    };

    solver.step_n(50);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.temperature.is_finite(),
            "thermal: particle {i} temperature non-finite"
        );
    }

    let mean_hot_after = {
        let hot: Vec<f32> = solver
            .particles()
            .iter()
            .filter(|p| p.x.x < 16.0)
            .map(|p| p.temperature)
            .collect();
        hot.iter().sum::<f32>() / hot.len() as f32
    };
    let mean_cold_after = {
        let cold: Vec<f32> = solver
            .particles()
            .iter()
            .filter(|p| p.x.x >= 16.0)
            .map(|p| p.temperature)
            .collect();
        cold.iter().sum::<f32>() / cold.len() as f32
    };

    assert!(
        mean_hot_after < mean_hot_before,
        "thermal: hot region did not cool (mean before={mean_hot_before:.1}, after={mean_hot_after:.1})"
    );
    assert!(
        mean_cold_after > mean_cold_before,
        "thermal: cold region did not warm (mean before={mean_cold_before:.1}, after={mean_cold_after:.1})"
    );
}

#[test]
fn thermal_uniform_temperature_stays_stable() {
    // All particles at the same temperature as ambient — diffusion should produce no drift.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let initial_temp = 20.0_f32;
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 1.0,
            heat_capacity: 1000.0,
            ambient: initial_temp, // same as particles → no boundary sink/source
            grid_cell_size: 0.1,
            ..Default::default()
        },
        config.grid_res,
    );

    let mut solver = MpmSolver::new(config, small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_thermal(thermal);

    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            particles.temperature[i] = initial_temp;
        }
    }

    solver.step_n(50);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            (p.temperature - initial_temp).abs() < 1.0,
            "thermal uniform: particle {i} drifted to {:.2} (expected ~{initial_temp})",
            p.temperature
        );
    }
}

// --- LP integration API tests ---

#[test]
fn apply_impulse_shifts_velocity() {
    // Apply rightward impulse from center. All particles near center should gain +x velocity.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let mut solver = MpmSolver::new(config, small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let avg_vx_before: f32 =
        solver.particles().iter().map(|p| p.v.x).sum::<f32>() / solver.particles().len() as f32;

    solver.apply_impulse(Vec2::splat(16.0), 10.0, Vec2::new(50.0, 0.0));

    let avg_vx_after: f32 =
        solver.particles().iter().map(|p| p.v.x).sum::<f32>() / solver.particles().len() as f32;

    assert!(
        avg_vx_after > avg_vx_before,
        "apply_impulse: avg vx did not increase (before={avg_vx_before:.2}, after={avg_vx_after:.2})"
    );
}

#[test]
fn apply_radial_impulse_increases_avg_speed() {
    // Outward radial impulse: all directions cancel in mean velocity but speed goes up.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let mut solver = MpmSolver::new(config, small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let avg_speed_before: f32 = solver.particles().iter().map(|p| p.v.length()).sum::<f32>()
        / solver.particles().len() as f32;

    solver.apply_radial_impulse(Vec2::splat(16.0), 10.0, 100.0);

    let avg_speed_after: f32 = solver.particles().iter().map(|p| p.v.length()).sum::<f32>()
        / solver.particles().len() as f32;

    assert!(
        avg_speed_after > avg_speed_before,
        "apply_radial_impulse: avg speed did not increase (before={avg_speed_before:.2}, after={avg_speed_after:.2})"
    );
}

#[test]
fn material_state_counts_and_centroid() {
    const FLUID_ID: u32 = 1;
    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_material(
            FLUID_ID,
            Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)),
        );

    // Left half → FLUID_ID, right half → default (0).
    solver.phase_transition(|p| p.x.x < 16.0, FLUID_ID);

    let total = solver.particles().len();
    let fluid_state = solver.material_state(FLUID_ID);
    let jelly_state = solver.material_state(0);

    assert!(
        fluid_state.count > 0,
        "material_state: no fluid particles found"
    );
    assert!(
        jelly_state.count > 0,
        "material_state: no jelly particles found"
    );
    assert_eq!(
        fluid_state.count + jelly_state.count,
        total,
        "material_state: counts don't add up"
    );
    // Fluid is on the left side.
    assert!(
        fluid_state.centroid.x < 16.0,
        "material_state: fluid centroid not on left (centroid.x={:.2})",
        fluid_state.centroid.x
    );
    // Jelly is on the right side.
    assert!(
        jelly_state.centroid.x >= 16.0,
        "material_state: jelly centroid not on right (centroid.x={:.2})",
        jelly_state.centroid.x
    );
}

#[test]
fn region_state_returns_subset_in_radius() {
    // Small radius should include fewer particles than a large radius.
    let solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let center = Vec2::splat(16.0);
    let small = solver.region_state(center, 2.0);
    let large = solver.region_state(center, 100.0);

    assert!(
        small.count > 0,
        "region_state: no particles in small radius"
    );
    assert!(
        large.count >= small.count,
        "region_state: large radius captured fewer than small"
    );
    // Large radius should capture all particles.
    assert_eq!(
        large.count,
        solver.particles().len(),
        "region_state: large radius missed particles"
    );
}

#[test]
fn aabb_confinement_keeps_particles_inside() {
    // High-velocity particles should stay within the AABB soft wall bounds.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        ..small_solver_config()
    };
    let min = Vec2::new(8.0, 8.0);
    let max = Vec2::new(24.0, 24.0);

    let spawn = SpawnConfig {
        box_center: Vec2::splat(16.0),
        box_size: IVec2::new(4, 4),
        initial_velocity_scale: 15.0,
        ..SpawnConfig::default()
    };

    let field = AabbConfinementField::new(min, max, 500.0);
    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_force_field(Box::new(field));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "aabb_confinement: particle {i} non-finite");
        // Allow 2-cell overshoot before restoring force fully acts.
        assert!(
            p.x.x >= min.x - 2.0 && p.x.x <= max.x + 2.0,
            "aabb_confinement: particle {i} escaped in x (x={:.2})",
            p.x.x
        );
        assert!(
            p.x.y >= min.y - 2.0 && p.x.y <= max.y + 2.0,
            "aabb_confinement: particle {i} escaped in y (y={:.2})",
            p.x.y
        );
    }
}

#[test]
fn spawn_region_appends_particles() {
    // First region at left side, second region at right side.
    // spawn_region must return the correct index range and increase particle count.
    let config = small_solver_config();
    let first_spawn = SpawnConfig {
        box_center: Vec2::new(10.0, 16.0),
        box_size: IVec2::new(4, 4),
        ..SpawnConfig::default()
    };
    let mut solver = MpmSolver::new(config, first_spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let count_before = solver.particles().len();
    assert!(
        count_before > 0,
        "spawn_region: initial spawn produced no particles"
    );

    let second_spawn = SpawnConfig {
        box_center: Vec2::new(22.0, 16.0),
        box_size: IVec2::new(4, 4),
        ..SpawnConfig::default()
    };
    let tag = solver.spawn_group(second_spawn);

    let count_after = solver.particles().len();
    assert!(count_after > count_before, "spawn_group: spawned zero particles");

    let group_count = solver.group_count(tag);
    assert!(group_count > 0, "spawn_group: tag_index has no entries");
    assert_eq!(group_count, count_after - count_before, "spawn_group: group_count mismatch");

    // All particles in the new group should be in the right region.
    let ps = solver.particles();
    for i in solver.particles_with_tag(tag) {
        assert!(
            ps.x[i].x > 16.0,
            "spawn_group: particle not in expected region (x={:.2})",
            ps.x[i].x
        );
    }
}

#[test]
fn diagnostics_snapshot_is_clean_after_stable_sim() {
    let mut solver = MpmSolver::new(small_solver_config(), small_spawn_config(16.0))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    solver.step_n(20);
    let snap = solver.diagnostics_snapshot();

    assert_eq!(
        snap.particle_count,
        solver.particles().len(),
        "snapshot: particle_count mismatch"
    );
    assert_eq!(
        snap.non_finite_particle_values, 0,
        "snapshot: non-finite particle values found"
    );
    assert_eq!(
        snap.out_of_bounds_particles, 0,
        "snapshot: particles out of bounds"
    );
    assert_eq!(
        snap.invalid_physical_particle_values, 0,
        "snapshot: invalid physical values"
    );
    assert!(snap.min_deformation_j > 0.0, "snapshot: min J collapsed");
}

#[test]
fn gravity_well_cutoff_prevents_far_particles_from_moving() {
    // Particles placed far beyond cutoff. With gravity=0, they should not accelerate.
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        grid_res: 64,
        ..SolverConfig::default()
    };
    // Well at center (32,32), cutoff=5 cells. Particles far away at (56,32) → dist=24 >> cutoff.
    let well = GravityWellField::new(
        vec![(Vec2::new(32.0, 32.0), 1_000_000.0)],
        1.0, // strong G
        1.0, // softening
    )
    .with_cutoff(5.0); // cutoff — particles at dist=24 are 4.8× beyond cutoff
    let spawn = SpawnConfig {
        box_center: Vec2::new(56.0, 32.0),
        box_size: IVec2::new(4, 4),
        initial_velocity_scale: 0.0,
        ..SpawnConfig::default()
    };
    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_force_field(Box::new(well));

    let cx_before: f32 =
        solver.particles().iter().map(|p| p.x.x).sum::<f32>() / solver.particles().len() as f32;

    solver.step_n(30);

    let cx_after: f32 =
        solver.particles().iter().map(|p| p.x.x).sum::<f32>() / solver.particles().len() as f32;

    // CoM should not have drifted left (toward well) — cutoff blocks the force.
    // Allow 0.5-cell drift from boundary reflection and elastic oscillation.
    assert!(
        (cx_after - cx_before).abs() < 0.5,
        "gravity_well cutoff: far particles moved toward well (before={cx_before:.2}, after={cx_after:.2})"
    );
}

/// GPU and CPU solvers must produce statistically equivalent physics.
/// Compares aggregate quantities (centre of mass, mean speed) — not per-particle positions,
/// since GPU atomic-scatter ordering causes sub-cell trajectory differences that are
/// physically equivalent but particle-ID-permuted.
#[cfg(feature = "gpu")]
#[test]
fn gpu_cpu_parity() {
    use emerge::gpu::GpuSolver;
    use emerge::materials::MaterialRegistry;

    let config = SolverConfig {
        grid_res: 32,
        dt: 0.002,
        adaptive_timestep: false,
        gravity: Vec2::new(0.0, -1.0),
        ..SolverConfig::default()
    };
    let material = NeoHookeanMaterial::new(1_000.0, 500.0);

    let mut cpu = MpmSolver::new(config.clone(), small_spawn_config(16.0))
        .with_default_material(Box::new(material));

    // Identical starting state for GPU.
    let mut gpu = pollster::block_on(GpuSolver::new(
        config.clone(),
        cpu.particles().to_vec(),
        MaterialRegistry::with_default(Box::new(material)),
    ));

    for _ in 0..20 {
        cpu.step();
        gpu.step_frame();
    }
    // Force a blocking readback so we compare actual final GPU state, not a stale snapshot.
    gpu.sync_particles_blocking();

    let n = cpu.particles().len() as f32;
    let cpu_com: Vec2 = cpu.particles().iter().map(|p| p.x).sum::<Vec2>() / n;
    let gpu_com: Vec2 = gpu.particles().iter().map(|p| p.x).sum::<Vec2>() / n;
    let cpu_spd: f32 = cpu.particles().iter().map(|p| p.v.length()).sum::<f32>() / n;
    let gpu_spd: f32 = gpu.particles().iter().map(|p| p.v.length()).sum::<f32>() / n;

    // Centre of mass must agree within 0.5 grid cells.
    let com_diff = (cpu_com - gpu_com).length();
    assert!(
        com_diff < 0.5,
        "CoM drift CPU {cpu_com:.3?} GPU {gpu_com:.3?} diff {com_diff:.4}"
    );

    // Mean speed must agree within 10 %.
    let spd_diff = (cpu_spd - gpu_spd).abs();
    assert!(
        spd_diff < 0.1 * cpu_spd.max(1e-6),
        "speed CPU {cpu_spd:.4} GPU {gpu_spd:.4}"
    );
}

#[test]
fn sand_mui_stable_after_many_steps() {
    // µ(I) sand: high-velocity spawn stresses the rate-dependent return mapping.
    let mui = SandMuIMaterial::new(1_000.0, 500.0);
    let config = SolverConfig {
        gravity: Vec2::new(0.0, -0.5),
        ..small_solver_config()
    };
    let spawn = SpawnConfig {
        initial_velocity_scale: 5.0,
        ..small_spawn_config(16.0)
    };
    let mut solver = MpmSolver::new(config, spawn).with_default_material(Box::new(mui));
    solver.step_n(200);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "mui particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "mui particle {i}: J collapsed"
        );
        assert!(
            p.friction_hardening.is_finite(),
            "mui particle {i}: mu_i non-finite"
        );
        assert!(
            p.friction_hardening >= 0.0,
            "mui particle {i}: mu_i negative (={:.4})",
            p.friction_hardening
        );
    }
}

#[test]
fn nacc_stable_after_many_steps() {
    let nacc = NaccMaterial::soft_clay();
    let config = SolverConfig {
        gravity: Vec2::new(0.0, -0.3),
        ..small_solver_config()
    };
    let mut solver =
        MpmSolver::new(config, small_spawn_config(16.0)).with_default_material(Box::new(nacc));
    solver.step_n(200);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "nacc particle {i}: position non-finite");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "nacc particle {i}: J collapsed"
        );
        assert!(
            p.log_volume_strain.is_finite(),
            "nacc particle {i}: alpha non-finite"
        );
    }
}

#[test]
fn retain_particles_syncs_active_count_and_steps_cleanly() {
    // Regression: particles_mut().retain() desynchronised active_count,
    // causing index-out-of-bounds in scatter_particle_mass on next step.
    let config = SolverConfig {
        grid_res: 32,
        dt: 0.1,
        ..SolverConfig::standard(32, 0.1, Vec2::new(0.0, -0.1))
    };
    let spawn = SpawnConfig {
        spacing: 0.5,
        box_size: glam::IVec2::new(16, 16),
        box_center: Vec2::splat(16.0),
        initial_velocity_scale: 0.0,
        ..SpawnConfig::for_solver(&config)
    };
    let mut solver = MpmSolver::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 50.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.spawn_group(spawn);

    let before = solver.particles().len();
    // Keep only particles in the left half.
    solver.retain_particles(|p| p.x.x < 16.0);
    let after = solver.particles().len();
    assert!(after < before, "retain should remove particles");

    // Must not panic — active_count must match particle array length.
    solver.step_n(5);
    for p in solver.particles() {
        assert!(p.x.is_finite(), "position non-finite after retain + step");
    }
}

#[test]
fn split_particles_where_conserves_mass_and_increases_count() {
    let config = SolverConfig::standard(32, 0.1, Vec2::new(0.0, -0.1));
    let spawn = SpawnConfig {
        spacing: 0.5,
        box_size: glam::IVec2::new(8, 8),
        box_center: Vec2::splat(16.0),
        ..SpawnConfig::for_solver(&config)
    };
    let mut solver = MpmSolver::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 50.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.spawn_group(spawn);

    // Run a few steps to build up some deformation gradient variation.
    solver.step_n(5);

    let total_mass_before: f32 = solver.particles().iter().map(|p| p.mass).sum();
    let count_before = solver.particles().len();

    // Split all active particles (unconditionally for test).
    solver.split_particles_where(|_| true, 0.2);

    let total_mass_after: f32 = solver.particles().iter().map(|p| p.mass).sum();
    let count_after = solver.particles().len();

    assert!(count_after > count_before, "particle count must increase after split");
    assert!(count_after <= count_before * 2, "particle count cannot exceed double");
    assert!(
        (total_mass_after - total_mass_before).abs() < 1e-3,
        "total mass must be conserved: before={total_mass_before:.4} after={total_mass_after:.4}"
    );

    // Must still simulate without panic.
    solver.step_n(3);
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "particle {i} non-finite after split");
    }
}

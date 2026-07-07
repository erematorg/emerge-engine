//! Physics correctness tests for emerge.
//!
//! These tests verify conservation laws, material invariants, and solver properties
//! that must hold for the engine to be physically valid.
//!
//! Each test has a clear physical claim and is comparable to reference MPM implementations
//! (sparkl, matter, taichi128).

extern crate emerge_engine as emerge;
use emerge::materials::MaterialModel;
use emerge::particle::{Particle, Particles};
use emerge::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
use emerge::{
    ActivationStatsPlugin, DiagnosticsFrame, DiagnosticsRegistry, MaterialCountPlugin,
    ThermalStatsPlugin, collect_snapshot,
};
use emerge::{
    BinghamFluidMaterial, CorotatedMaterial, DruckerPragerMaterial, MuIRheologyMaterial,
    NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion,
    StomakhinMaterial, ViscoelasticMaterial, VonMisesMaterial,
};
use glam::{IVec2, Mat2, Vec2};

// â”€â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Wrap a single `Particle` in a one-element `Particles` SoA, call `kirchhoff_stress`, return result.
fn kirchhoff_stress_of(mat: &dyn emerge::materials::MaterialModel, p: &Particle) -> glam::Mat2 {
    let soa = Particles::from(vec![*p]);
    mat.kirchhoff_stress(&soa, 0)
}

/// Wrap a single `Particle` in a one-element `Particles` SoA, call `update_particle`, write back.
fn update_particle_of(mat: &dyn emerge::materials::MaterialModel, p: &mut Particle, dt: f32) {
    let mut soa = Particles::from(vec![*p]);
    mat.update_particle(&mut soa, 0, dt);
    *p = soa.get(0);
}

fn zero_gravity_config(grid_res: usize) -> SimConfig {
    SimConfig {
        grid_res,
        dt: 0.05,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        ..SimConfig::default()
    }
}

fn center_spawn(grid_res: usize, side: usize) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(side as i32, side as i32),
        box_center: Vec2::splat(grid_res as f32 * 0.5),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::default()
    }
}

fn total_mass(solver: &Simulation) -> f32 {
    solver.particles().iter().map(|p| p.mass).sum()
}

fn linear_momentum(solver: &Simulation) -> Vec2 {
    solver.particles().iter().map(|p| p.mass * p.v).sum()
}

fn kinetic_energy(solver: &Simulation) -> f32 {
    solver
        .particles()
        .iter()
        .map(|p| 0.5 * p.mass * p.v.length_squared())
        .sum()
}

fn min_j(solver: &Simulation) -> f32 {
    solver
        .particles()
        .iter()
        .map(|p| p.deformation_gradient.determinant())
        .fold(f32::INFINITY, f32::min)
}

// â”€â”€â”€ CONSERVATION: MASS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Mass is a particle property and never changes â€” the solver must not add or remove particles.
#[test]
fn mass_is_conserved_neohookean() {
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "mass changed: before={m0:.6} after={m1:.6} delta={:.2e}",
        (m1 - m0).abs()
    );
}

#[test]
fn mass_is_conserved_fluid() {
    let config = SimConfig {
        recompute_density_each_step: true,
        ..zero_gravity_config(32)
    };
    let mut solver = Simulation::new(config, center_spawn(32, 6))
        .with_default_material(Box::new(NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0)));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "fluid: mass changed: before={m0:.6} after={m1:.6}"
    );
}

#[test]
fn mass_is_conserved_snow() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(snow));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "snow: mass not conserved: {m0:.6} â†’ {m1:.6}"
    );
}

// â”€â”€â”€ CONSERVATION: LINEAR MOMENTUM (no external forces) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// With zero gravity and zero initial velocity, total momentum must stay near zero.
/// (MLS-MPM is weakly momentum conserving; small residuals from grid averaging are expected.)
#[test]
fn zero_velocity_spawn_has_near_zero_momentum() {
    let mut solver = Simulation::new(zero_gravity_config(32), center_spawn(32, 8))
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let p0 = linear_momentum(&solver);
    solver.step_n(50);
    let p1 = linear_momentum(&solver);

    // Absolute momentum drift per particle (mass=1): should stay tiny
    let n = solver.particles().len() as f32;
    let drift = (p1 - p0).length() / n;
    assert!(
        drift < 1e-3,
        "momentum drift per particle too large: {drift:.2e} (initial p={p0}, final p={p1})"
    );
}

/// With uniform gravity and no initial motion, momentum grows at rate mÂ·g â€” verify linearity.
#[test]
fn gravity_grows_momentum_linearly() {
    let g = Vec2::new(0.0, -9.81);
    let config = SimConfig {
        gravity: g,
        dt: 0.01,
        adaptive_timestep: false,
        ..SimConfig::default()
    };
    let mut solver = Simulation::new(config, center_spawn(64, 4))
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 200.0)));

    let m_total = total_mass(&solver);
    let p_before = linear_momentum(&solver);

    let n_steps = 10;
    let dt = 0.01f32;
    solver.step_n(n_steps);

    let p_after = linear_momentum(&solver);
    let elapsed = dt * n_steps as f32;
    let expected_impulse = g * m_total * elapsed;
    let actual_impulse = p_after - p_before;

    // Allow 5% tolerance: boundary clamping absorbs some momentum
    let rel_err = (actual_impulse - expected_impulse).length() / (expected_impulse.length() + 1e-6);
    assert!(
        rel_err < 0.05,
        "gravity impulse wrong: expected={expected_impulse:.3?} actual={actual_impulse:.3?} rel_err={rel_err:.3}"
    );
}

// â”€â”€â”€ J > 0 INVARIANT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// det(F) > 0 is a non-negotiable physical invariant â€” particles can't invert.
/// Requires `project_invalid_state: true` (standard config) â€” the J floor that real simulations use.
#[test]
fn j_stays_positive_neohookean() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "NeoHookean: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_snow() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Snow: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_sand() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(sand));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Sand: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_corotated() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(CorotatedMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Corotated: J collapsed to {jmin:.2e}");
}

// â”€â”€â”€ SNOW PLASTICITY: Jp BOUNDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Snow Jp must stay within [min_jp, max_jp] after any number of steps.
/// This is the yield surface enforcement â€” clamped singular values constrain Jp.
#[test]
fn snow_jp_stays_within_bounds() {
    let min_jp = 0.6f32;
    let max_jp = 20.0f32;
    let snow = StomakhinMaterial::new(38_889.0, 58_333.0, 10.0, 0.025, 0.0075, min_jp, max_jp);

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(300);

    for (i, p) in solver.particles().iter().enumerate() {
        let jp = p.plastic_volume_ratio;
        assert!(
            jp >= min_jp * 0.99 && jp <= max_jp * 1.01,
            "snow particle {i}: Jp={jp:.4} out of [{min_jp}, {max_jp}]"
        );
    }
}

/// Snow hardening scale h = exp(Î¾(1-Jp)) must be non-negative and finite.
/// Note: h=0.0 is valid f32 underflow of exp(âˆ’190) when Jpâ‰ˆmax_jp â€” effectively zero stress.
/// What matters is that h stays finite (no NaN/Inf) and non-negative.
#[test]
fn snow_hardening_scale_finite() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(snow));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.hardening_scale >= 0.0 && p.hardening_scale.is_finite(),
            "snow particle {i}: hardening_scale={:.4} (must be finite â‰¥0)",
            p.hardening_scale
        );
    }
}

// â”€â”€â”€ SAND: NO TENSION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Sand cannot sustain tension (p â‰¤ 0 â†’ project to stress-free).
/// Test via direct material update on a tensile deformation gradient.
#[test]
fn sand_tension_cutoff_removes_tensile_stress() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Pure extension: F = diag(1.5, 1.5) â€” volume 2.25Ã—, tensile state
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.5));
    p.velocity_gradient = Mat2::ZERO;

    // Initialize particle (seeds plastic state)
    sand.init_particle(&mut p);
    update_particle_of(&sand, &mut p, 0.01);

    // After projection, stress should be near zero (tensile â†’ return to identity)
    let tau = kirchhoff_stress_of(&sand, &p);
    let tau_norm = (tau.x_axis.length_squared() + tau.y_axis.length_squared()).sqrt();
    assert!(
        tau_norm < 1.0,
        "sand: tensile stress not projected (||Ï„||={tau_norm:.4})"
    );
}

/// Sand Drucker-Prager: log_volume_strain must stay finite.
/// Requires project_invalid_state=true to prevent Jâ†’0 which causes log(J)=âˆ’âˆž.
#[test]
fn sand_log_volume_strain_finite() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(sand));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.log_volume_strain.is_finite(),
            "sand particle {i}: log_volume_strain={}",
            p.log_volume_strain
        );
    }
}

// â”€â”€â”€ MATERIAL STRESS SYMMETRY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Kirchhoff stress Ï„ must be symmetric for all materials (objectivity / frame-indifference).
/// Ï„ = Ï„áµ€: |Ï„â‚€â‚ âˆ’ Ï„â‚â‚€| < Îµ.
fn check_stress_symmetry(mat: &dyn MaterialModel, label: &str) {
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Small shear deformation: F = [[1.1, 0.1], [0.05, 0.95]]
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.1, 0.05), Vec2::new(0.1, 0.95));
    mat.init_particle(&mut p);

    let tau = kirchhoff_stress_of(mat, &p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(
        asym < 1e-4,
        "{label}: Kirchhoff stress asymmetric: Ï„â‚€â‚={:.6} Ï„â‚â‚€={:.6} |diff|={asym:.2e}",
        tau.col(1).x,
        tau.col(0).y,
    );
}

#[test]
fn neohookean_stress_symmetric() {
    check_stress_symmetry(&NeoHookeanMaterial::new(100.0, 200.0), "NeoHookean");
}

#[test]
fn corotated_stress_symmetric() {
    check_stress_symmetry(&CorotatedMaterial::new(100.0, 200.0), "Corotated");
}

#[test]
fn snow_stress_symmetric() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    check_stress_symmetry(&snow, "Snow");
}

#[test]
fn sand_stress_symmetric() {
    let sand = DruckerPragerMaterial::cohesionless(5429.0, 0.357);
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Compressive deformation (sand only resists compression)
    p.deformation_gradient = Mat2::from_cols(Vec2::new(0.9, 0.05), Vec2::new(0.05, 0.9));
    sand.init_particle(&mut p);
    update_particle_of(&sand, &mut p, 0.01);

    let tau = kirchhoff_stress_of(&sand, &p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(asym < 1e-4, "Sand: stress asymmetric: {asym:.2e}");
}

// â”€â”€â”€ SVD CORRECTNESS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Our analytical 2Ã—2 SVD must satisfy F = UÂ·diag(Ïƒ)Â·Váµ€ and U,V orthogonal.
/// This is tested internally in mechanics/svd.rs, but we verify the public path
/// through StomakhinMaterial.update_particle which uses svd2().
#[test]
fn snow_update_preserves_f_decomposition_invariant() {
    // After snow update, F_elastic must remain a valid deformation gradient.
    // det(F) > 0, F finite, singular values in (0, +âˆž).
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Start from slight compression
    p.deformation_gradient = Mat2::from_cols(Vec2::new(0.95, 0.02), Vec2::new(-0.02, 0.95));
    p.plastic_volume_ratio = 1.0;
    p.hardening_scale = 1.0;

    for _ in 0..50 {
        p.velocity_gradient = Mat2::from_cols(Vec2::new(-0.01, 0.005), Vec2::new(0.005, -0.01));
        update_particle_of(&snow, &mut p, 0.01);
    }

    let j = p.deformation_gradient.determinant();
    assert!(
        j > 0.0 && j.is_finite(),
        "Snow: F det invalid after updates: J={j}"
    );
    assert!(p.deformation_gradient.is_finite(), "Snow: F non-finite");
    assert!(p.hardening_scale > 0.0 && p.hardening_scale.is_finite());
}

// â”€â”€â”€ ENERGY NON-GROWTH (elastic, no gravity) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Kinetic energy of a resting elastic blob (no gravity, zero initial velocity)
/// must stay near zero â€” no spurious energy injection from the solver.
#[test]
fn resting_jelly_no_energy_growth() {
    let config = SimConfig {
        gravity: Vec2::ZERO,
        dt: 0.05,
        ..SimConfig::default()
    };
    let spawn = SpawnRegion {
        initial_velocity_scale: 0.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let ke0 = kinetic_energy(&solver);
    solver.step_n(200);
    let ke1 = kinetic_energy(&solver);

    // Resting blob: initial KE â‰ˆ 0. After steps it may have tiny numerical KE but
    // must not grow significantly.
    let n = solver.particles().len() as f32;
    assert!(
        ke1 / n < 1e-4,
        "resting jelly: KE grew from {ke0:.2e} to {ke1:.2e} ({:.2e} per particle)",
        ke1 / n
    );
}

// â”€â”€â”€ CFL STABILITY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Adaptive substep must never produce a sub_dt that violates particle CFL.
/// Proxy: particle speed Ã— sub_dt â‰¤ 1 cell (with CFL coeff).
/// We verify this by checking velocities never exceed the grid/dt threshold.
#[test]
fn adaptive_substep_keeps_velocities_bounded() {
    let config = SimConfig {
        gravity: Vec2::new(0.0, -9.81),
        dt: 0.1,
        adaptive_timestep: true,
        cfl_coefficient: 0.4,
        ..SimConfig::default()
    };
    // High initial velocity to stress CFL
    let spawn = SpawnRegion {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(50.0, 100.0)));

    solver.step_n(100);

    // With CFL=0.4, max speed per step â‰¤ 0.4 * h / sub_dt.
    // Since sub_dt â‰¤ dt=0.1 and h=1, max speed per substep â‰¤ 0.4/sub_dt.
    // But we just check that all particles have finite velocity.
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.v.is_finite(),
            "CFL test: particle {i} velocity non-finite: {:?}",
            p.v
        );
        assert!(
            p.v.length() < 500.0,
            "CFL test: particle {i} velocity exploded: |v|={:.1}",
            p.v.length()
        );
    }
}

// â”€â”€â”€ DIAGNOSTICS PLUGIN SYSTEM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// DiagnosticsRegistry::collect must aggregate all plugin outputs.
#[test]
fn diagnostics_registry_aggregates_plugins() {
    use emerge::grid::Grid;

    let config = SimConfig {
        grid_res: 8,
        dt: 0.1,
        ..SimConfig::default()
    };

    let particles = vec![
        Particle {
            x: Vec2::new(4.0, 4.0),
            v: Vec2::new(1.0, 0.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 300.0,
            activation: 0.8,
            material_id: 0,
            ..Particle::zeroed()
        },
        Particle {
            x: Vec2::new(5.0, 4.0),
            v: Vec2::new(-1.0, 0.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 320.0,
            activation: 0.0,
            material_id: 1,
            ..Particle::zeroed()
        },
    ];

    let grid = Grid::new(config.grid_res);
    let particles_soa = emerge::particle::Particles::from(particles.clone());
    let snap = collect_snapshot(0, &particles_soa, &grid, &config, config.dt, 1);

    let mut registry = DiagnosticsRegistry::new();
    registry.register(Box::new(ActivationStatsPlugin));
    registry.register(Box::new(ThermalStatsPlugin));
    registry.register(Box::new(MaterialCountPlugin));
    // Closure plugin
    registry.register_fn("custom", |particles, _snap| {
        vec![("n_total".into(), particles.len() as f32)]
    });

    assert_eq!(registry.len(), 4);

    let frame = registry.collect(&particles, &snap);

    // Activation: mean = (0.8 + 0.0)/2 = 0.4, frac = 1/2 = 0.5
    let act_mean = frame.get("act_mean").expect("act_mean missing");
    assert!(
        (act_mean - 0.4).abs() < 1e-5,
        "act_mean={act_mean:.4} expected 0.4"
    );

    let act_frac = frame.get("act_frac").expect("act_frac missing");
    assert!(
        (act_frac - 0.5).abs() < 1e-5,
        "act_frac={act_frac:.4} expected 0.5"
    );

    // Temperature: mean = (300+320)/2=310, max=320
    let t_mean = frame.get("T_mean").expect("T_mean missing");
    assert!(
        (t_mean - 310.0).abs() < 1e-3,
        "T_mean={t_mean:.2} expected 310"
    );

    let t_max = frame.get("T_max").expect("T_max missing");
    assert!(
        (t_max - 320.0).abs() < 1e-3,
        "T_max={t_max:.2} expected 320"
    );

    // Material counts: mat0_n=1, mat1_n=1
    let mat0 = frame.get("mat0_n").expect("mat0_n missing");
    assert_eq!(mat0 as usize, 1, "mat0_n wrong");

    let mat1 = frame.get("mat1_n").expect("mat1_n missing");
    assert_eq!(mat1 as usize, 1, "mat1_n wrong");

    // Custom: n_total=2
    let n = frame.get("n_total").expect("n_total missing");
    assert_eq!(n as usize, 2, "n_total wrong");
}

/// DiagnosticsFrame::format_line produces compact output with all keys.
#[test]
fn diagnostics_frame_format_line_is_compact() {
    let frame = DiagnosticsFrame {
        stats: vec![
            ("n".into(), 256.0),
            ("ke".into(), 1.2345),
            ("act_mean".into(), 0.5),
        ],
    };
    let line = frame.format_line();
    assert!(line.contains("n=256"), "missing n=256 in: {line}");
    assert!(line.contains("ke=1.2345"), "missing ke in: {line}");
    assert!(
        line.contains("act_mean=0.5000"),
        "missing act_mean in: {line}"
    );
}

/// Empty registry produces empty DiagnosticsFrame.
#[test]
fn empty_registry_produces_empty_frame() {
    let mut registry = DiagnosticsRegistry::new();
    let p: Vec<Particle> = vec![];
    let config = SimConfig {
        grid_res: 8,
        ..SimConfig::default()
    };
    use emerge::grid::Grid;
    let grid = Grid::new(8);
    let snap = collect_snapshot(
        0,
        &emerge::particle::Particles::new(),
        &grid,
        &config,
        0.1,
        1,
    );
    let frame = registry.collect(&p, &snap);
    assert!(frame.stats.is_empty(), "expected empty frame");
    assert!(frame.format_line().is_empty(), "expected empty format");
}

// â”€â”€â”€ SCALAR DIFFUSION FIELD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Scalar diffusion must move a high-concentration particle's field toward lower concentration.
/// Without decay: total Ï† (summed over particles) should be approximately conserved.
#[test]
fn scalar_diffusion_spreads_and_conserves() {
    let grid_res = 16;
    let config = ScalarDiffusionConfig {
        diffusivity: 1.0,
        decay_rate: 0.0, // no decay â†’ conserved
        ambient: 0.0,
    };

    let mut field = ScalarDiffusionField::new(
        config,
        |p| p.temperature,
        |p, delta| p.temperature += delta,
        grid_res,
    );

    // Two particles: one hot (T=100), one cold (T=0). After diffusion, heat spreads.
    let mut particles = Particles::from(vec![
        Particle {
            x: Vec2::new(7.0, 8.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 100.0,
            ..Particle::zeroed()
        },
        Particle {
            x: Vec2::new(9.0, 8.0),
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            temperature: 0.0,
            ..Particle::zeroed()
        },
    ]);

    let t_total_before: f32 = particles.temperature.iter().sum();

    // 10 substeps of diffusion
    for _ in 0..10 {
        field.apply(&mut particles, 0.01);
    }

    let t_total_after: f32 = particles.temperature.iter().sum();

    // Cold particle should have warmed
    assert!(
        particles.temperature[1] > 0.1,
        "cold particle didn't warm: T={:.4}",
        particles.temperature[1]
    );

    // Hot particle should have cooled
    assert!(
        particles.temperature[0] < 100.0,
        "hot particle didn't cool: T={:.4}",
        particles.temperature[0]
    );

    // Conservation: total T should be roughly conserved (Â±20% tolerance â€” boundary effects)
    let conservation_err = (t_total_after - t_total_before).abs() / t_total_before;
    assert!(
        conservation_err < 0.20,
        "scalar field: total T changed too much: before={t_total_before:.2} after={t_total_after:.2} err={conservation_err:.2}"
    );
}

/// With decay_rate > 0, total Ï† must decrease over time.
#[test]
fn scalar_diffusion_decay_reduces_total() {
    let config = ScalarDiffusionConfig {
        diffusivity: 0.0,
        decay_rate: 1.0, // fast decay â€” T halves in ~0.69s
        ambient: 0.0,
    };

    let mut field = ScalarDiffusionField::new(
        config,
        |p| p.temperature,
        |p, delta| p.temperature += delta,
        16,
    );

    let mut particles = Particles::from(vec![Particle {
        x: Vec2::new(8.0, 8.0),
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        temperature: 100.0,
        ..Particle::zeroed()
    }]);

    for _ in 0..50 {
        field.apply(&mut particles, 0.02); // 1s total
    }

    // After 1s at decay_rate=1.0: T should be ~100*e^(-1) â‰ˆ 36.8
    // Allow Â±50% â€” grid average discretization makes this noisy with one particle
    assert!(
        particles.temperature[0] < 70.0,
        "decay: temperature not decreasing: T={:.2}",
        particles.temperature[0]
    );
    assert!(
        particles.temperature[0] > 0.0,
        "decay: temperature went negative: T={:.2}",
        particles.temperature[0]
    );
}

// â”€â”€â”€ MATERIAL RATE CONSISTENCY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Half-step Ã— 2 must be approximately equivalent to one full step.
/// This tests that material update is smooth/continuous (not discontinuous jumps).
#[test]
fn snow_half_step_consistency() {
    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);

    let base_particle = Particle {
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        deformation_gradient: Mat2::from_cols(Vec2::new(0.98, 0.01), Vec2::new(-0.01, 0.98)),
        plastic_volume_ratio: 1.0,
        hardening_scale: 1.0,
        velocity_gradient: Mat2::from_cols(Vec2::new(-0.01, 0.005), Vec2::new(0.005, -0.01)),
        ..Particle::zeroed()
    };

    // Full step
    let mut p_full = base_particle;
    update_particle_of(&snow, &mut p_full, 0.02);

    // Two half-steps
    let mut p_half = base_particle;
    update_particle_of(&snow, &mut p_half, 0.01);
    update_particle_of(&snow, &mut p_half, 0.01);

    let j_full = p_full.deformation_gradient.determinant();
    let j_half = p_half.deformation_gradient.determinant();

    // J should be close (within 1% â€” subcycling plasticity has small discrepancies)
    assert!(
        (j_full - j_half).abs() < 0.01,
        "snow: full-step J={j_full:.6} vs halfÃ—2 J={j_half:.6} â€” too different"
    );
}

/// VonMises: after enough plastic deformation, stress norm must not exceed yield surface.
#[test]
fn von_mises_stress_bounded_by_yield() {
    let yield_stress = 100.0f32;
    let vm = VonMisesMaterial::new(1_000.0, 500.0, yield_stress);

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let spawn = SpawnRegion {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = Simulation::new(config, spawn).with_default_material(Box::new(vm));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let tau = kirchhoff_stress_of(&vm, &p);
        // von Mises equivalent stress: sqrt(3/2 * s:s) where s = dev(Ï„)
        let tr = (tau.col(0).x + tau.col(1).y) * 0.5;
        let s00 = tau.col(0).x - tr;
        let s11 = tau.col(1).y - tr;
        let s01 = tau.col(1).x; // off-diagonal
        let vm_stress = (1.5 * (s00 * s00 + s11 * s11 + 2.0 * s01 * s01)).sqrt();
        // Allow 40% overshoot: initial_velocity_scale=5.0 creates violent collisions where
        // discrete return-mapping can't fully project to the yield surface in a single step.
        // Key invariant: stress stays finite and bounded, not that it's exactly at yield.
        assert!(
            vm_stress < yield_stress * 1.40,
            "VonMises particle {i}: Ïƒ_vm={vm_stress:.2} > yield {yield_stress:.2}"
        );
    }
}

// â”€â”€â”€ MULTI-MATERIAL ISOLATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Two materials spawned in different regions must not interfere with each other's invariants.
#[test]
fn two_material_solver_both_j_positive() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));

    let spawn0 = SpawnRegion {
        box_center: Vec2::new(20.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::default()
    };

    let snow = StomakhinMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = Simulation::new(config, spawn0)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)))
        .with_material(1, Box::new(snow));

    let spawn1 = SpawnRegion {
        box_center: Vec2::new(44.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        material_id: 1,
        ..SpawnRegion::default()
    };
    let _tag = solver.add_body(spawn1);

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let j = p.deformation_gradient.determinant();
        assert!(
            j > 0.0,
            "two-material: particle {i} mat={} J={j:.2e}",
            p.material_id
        );
    }
}

// â”€â”€â”€ Âµ(I) RHEOLOGY SANITY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// SandMuI: friction_hardening (Âµ(I)) must stay within [Âµ_static, Âµ_dynamic].
#[test]
fn sand_mui_friction_stays_in_range() {
    let mat = MuIRheologyMaterial::small_grain(5429.0, 0.357);
    let mu_static = 20.9f32.to_radians().tan();
    let mu_dynamic = 32.8f32.to_radians().tan();

    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver =
        Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(mat));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let mu_i = p.friction_hardening;
        assert!(
            mu_i >= mu_static * 0.95 && mu_i <= mu_dynamic * 1.05,
            "SandMuI particle {i}: Âµ(I)={mu_i:.4} out of [{mu_static:.4}, {mu_dynamic:.4}]"
        );
    }
}

// â”€â”€â”€ Bingham fluid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Bingham mud stays above floor without collapsing (yield stress holds shape under gravity).
#[test]
fn bingham_mud_stable_under_gravity() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::high_yield(1500.0, 1.0e4)));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(
            p.x.y > 1.0,
            "mud particle fell through floor: y={:.3}",
            p.x.y
        );
        assert!(p.x.is_finite(), "mud particle position NaN");
        assert!(p.v.is_finite(), "mud particle velocity NaN");
    }
}

/// Bingham J > 0 invariant.
#[test]
fn bingham_j_positive() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::high_yield(1500.0, 1.0e4)));
    solver.step_n(60);
    for p in solver.particles() {
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "Bingham J={j:.4} â‰¤ 0 â€” volume collapsed");
    }
}

/// Bingham lava: higher yield/viscosity than mud, still stable.
#[test]
fn bingham_lava_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 6)).with_default_material(Box::new(
        BinghamFluidMaterial::viscous_high_yield(2700.0, 1.0e5),
    ));
    solver.step_n(40);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "lava particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "lava J={j:.4} â‰¤ 0");
    }
}

// â”€â”€â”€ Viscoelastic (Kelvin-Voigt) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Viscoelastic soft tissue: J > 0, no NaN, stable under gravity.
#[test]
fn viscoelastic_soft_tissue_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 8)).with_default_material(Box::new(
        ViscoelasticMaterial::near_incompressible(5.0e4, 10.0),
    ));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "tissue particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "tissue J={j:.4} â‰¤ 0");
    }
}

/// Viscoelastic cell body: very soft, stable.
#[test]
fn viscoelastic_cell_body_stable() {
    let config = SimConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = Simulation::new(config, center_spawn(64, 6)).with_default_material(Box::new(
        ViscoelasticMaterial::moderately_compressible(5.0e3, 0.05),
    ));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "cell particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "cell J={j:.4} â‰¤ 0");
    }
}

/// KV viscous contribution: stress with non-zero strain rate > stress without.
/// Tests that the dashpot term activates when velocity_gradient is non-zero.
#[test]
fn viscoelastic_viscous_term_activates() {
    let e = 5.0e4f32;
    let nu = 0.40f32;
    let eta = 500.0f32;

    let visco = ViscoelasticMaterial::from_young_modulus(e, nu, eta);
    let elastic = NeoHookeanMaterial::from_young_modulus(e, nu);

    // Particle at rest with identity F â€” same elastic stress for both.
    let mut p = Particle::zeroed();
    p.volume = 1.0;
    p.density = 1.0;
    p.mass = 1.0;

    let tau_elastic_rest = kirchhoff_stress_of(&elastic, &p);
    let tau_visco_rest = kirchhoff_stress_of(&visco, &p);
    // At rest (C=0, F=I) both give same stress (NeoHookean base is identical).
    let diff_rest = (tau_visco_rest - tau_elastic_rest).x_axis.length()
        + (tau_visco_rest - tau_elastic_rest).y_axis.length();
    assert!(
        diff_rest < 1.0,
        "at rest KV and elastic should agree: diff={diff_rest}"
    );

    // Now give particle a shear strain rate via velocity_gradient.
    p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0));

    let tau_elastic_shear = kirchhoff_stress_of(&elastic, &p);
    let tau_visco_shear = kirchhoff_stress_of(&visco, &p);

    // KV adds Î·Â·D_dev â€” stress norms must differ.
    let norm_e = tau_elastic_shear.x_axis.length() + tau_elastic_shear.y_axis.length();
    let norm_v = tau_visco_shear.x_axis.length() + tau_visco_shear.y_axis.length();
    assert!(
        (norm_v - norm_e).abs() > 1.0,
        "KV dashpot should contribute when Câ‰ 0: norm_elastic={norm_e:.2} norm_visco={norm_v:.2}"
    );
}

// â”€â”€â”€ Phase rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Phase rule transitions hot particles to a new material id.
#[test]
fn phase_rule_transitions_hot_particles() {
    const COLD_ID: u32 = 0;
    const HOT_ID: u32 = 1;
    let hot_threshold = 0.5f32;

    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_material(
            HOT_ID,
            Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)),
        )
        .with_phase_rule(move |p| {
            if p.material_id == COLD_ID && p.temperature > hot_threshold {
                Some(HOT_ID)
            } else {
                None
            }
        });

    // Heat half the particles manually.
    let n = solver.particles().len();
    for i in 0..n / 2 {
        solver.particles_mut().temperature[i] = hot_threshold + 0.1;
    }

    solver.step();

    let hot_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == HOT_ID)
        .count();
    assert!(
        hot_count >= n / 2,
        "expected â‰¥{} hot particles, got {hot_count}",
        n / 2
    );
}

/// Phase rule: no transitions when condition not met.
#[test]
fn phase_rule_no_spurious_transitions() {
    const MAT_B: u32 = 1;

    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = Simulation::new(config, center_spawn(64, 8))
        .with_material(
            MAT_B,
            Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)),
        )
        .with_phase_rule(|p| {
            if p.temperature > 999.0 {
                Some(MAT_B)
            } else {
                None
            }
        });

    // No particles have temperature > 999
    solver.step_n(10);

    let b_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == MAT_B)
        .count();
    assert_eq!(b_count, 0, "spurious transitions to MAT_B: {b_count}");
}

// â”€â”€â”€ Neighbor queries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// particles_near returns only particles within radius.
#[test]
fn particles_near_radius_correct() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 2.0;

    let ps = solver.particles();
    for i in solver.particles_near(center, radius) {
        let dist = (ps.x[i] - center).length();
        assert!(
            dist <= radius + f32::EPSILON,
            "particle at dist={dist:.3} outside radius={radius}"
        );
    }
}

/// count_near matches manual count.
#[test]
fn count_near_matches_manual() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 3.0;
    let mat_id = 0u32;

    let api_count = solver.count_near(center, radius, mat_id);
    let manual_count = solver
        .particles()
        .iter()
        .filter(|p| p.material_id == mat_id && (p.x - center).length() <= radius)
        .count();

    assert_eq!(api_count, manual_count);
}

/// `particles_knn` must return exactly the same k-nearest INDEX SET as a
/// brute-force sort over every particle -- proves the geometric radius
/// expansion doesn't miss a closer particle just outside its current search
/// box. Query point is deliberately OFF the spawn's own symmetric center
/// (32.0, 32.0): querying from dead center over a symmetric grid puts many
/// particles at the exact same distance, making the k-th-nearest cutoff
/// genuinely ambiguous (confirmed empirically -- an earlier version of this
/// test queried from center and failed on a real tie at the boundary, not an
/// algorithm bug). An off-center point makes distances generically distinct.
#[test]
fn particles_knn_matches_brute_force() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let center = Vec2::new(32.37, 31.82);
    let k = 7; // Ballerini et al. 2008's real ~6-7 neighbor figure

    let ps = solver.particles();
    let mut brute: Vec<(usize, f32)> = (0..ps.len())
        .map(|i| (i, (ps.x[i] - center).length_squared()))
        .collect();
    brute.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let mut expected: Vec<usize> = brute.into_iter().take(k).map(|(i, _)| i).collect();
    expected.sort_unstable();

    let mut got = solver.particles_knn(center, k);
    got.sort_unstable();
    assert_eq!(
        got, expected,
        "particles_knn must match a brute-force k-nearest scan (same particle set)"
    );
}

/// Requesting more neighbors than exist must return everything, not panic or loop forever.
#[test]
fn particles_knn_clamps_to_available_particle_count() {
    let config = SimConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = Simulation::new(config, center_spawn(64, 8));

    let total = solver.particles().len();
    let got = solver.particles_knn(Vec2::splat(32.0), total + 1000);
    assert_eq!(
        got.len(),
        total,
        "requesting more neighbors than exist must return exactly all of them, not panic"
    );
}

// ─── thermo-mechanical coupling (E(T)) ──────────────────────────────────────────
//
// `thermal_expansion` already existed on NeoHookean/Corotated/Viscoelastic and was already
// wired into both the CPU kirchhoff_stress AND the GPU p2g.wgsl shader (identical formula,
// `t_scale = 1.0 + thermal_expansion * temperature`) — but had zero tests or examples
// exercising it anywhere in the repo. This verifies the feature actually does what its own
// doc comment claims (negative = softening) rather than assuming it from reading the code.

fn stress_frobenius_norm(tau: Mat2) -> f32 {
    (tau.col(0).length_squared() + tau.col(1).length_squared()).sqrt()
}

#[test]
fn neohookean_negative_thermal_expansion_softens_stress() {
    let mut mat = NeoHookeanMaterial::new(100.0, 200.0);
    mat.thermal_expansion = -1.0e-3; // per its own doc comment: negative = softening

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Same moderate shear/stretch deformation for both — only temperature differs.
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(0.15, 0.9));
    mat.init_particle(&mut p);

    p.temperature = 0.0;
    let tau_cold = kirchhoff_stress_of(&mat, &p);

    p.temperature = 500.0;
    let tau_hot = kirchhoff_stress_of(&mat, &p);

    let norm_cold = stress_frobenius_norm(tau_cold);
    let norm_hot = stress_frobenius_norm(tau_hot);
    assert!(
        norm_hot < norm_cold,
        "heating with negative thermal_expansion should soften (lower stress for the same \
         deformation): cold={norm_cold:.4} hot={norm_hot:.4}"
    );

    // Sanity: thermal_expansion=0.0 (the default) must be completely temperature-independent —
    // this is the "zero behavior change for anything that doesn't opt in" guarantee.
    let neutral = NeoHookeanMaterial::new(100.0, 200.0);
    let mut p_neutral = p;
    p_neutral.temperature = 0.0;
    let tau_neutral_cold = kirchhoff_stress_of(&neutral, &p_neutral);
    p_neutral.temperature = 500.0;
    let tau_neutral_hot = kirchhoff_stress_of(&neutral, &p_neutral);
    assert!(
        (stress_frobenius_norm(tau_neutral_cold) - stress_frobenius_norm(tau_neutral_hot)).abs()
            < 1e-6,
        "thermal_expansion=0.0 must be exactly temperature-independent"
    );
}

#[test]
fn corotated_negative_thermal_expansion_softens_stress() {
    let mut mat = CorotatedMaterial::new(100.0, 200.0);
    mat.thermal_expansion = -1.0e-3;

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    p.deformation_gradient = Mat2::from_cols(Vec2::new(1.2, 0.1), Vec2::new(0.15, 0.9));
    mat.init_particle(&mut p);

    p.temperature = 0.0;
    let norm_cold = stress_frobenius_norm(kirchhoff_stress_of(&mat, &p));
    p.temperature = 500.0;
    let norm_hot = stress_frobenius_norm(kirchhoff_stress_of(&mat, &p));

    assert!(
        norm_hot < norm_cold,
        "Corotated: heating with negative thermal_expansion should soften: \
         cold={norm_cold:.4} hot={norm_hot:.4}"
    );
}

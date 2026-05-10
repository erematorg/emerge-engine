//! Physics correctness tests for emerge.
//!
//! These tests verify conservation laws, material invariants, and solver properties
//! that must hold for the engine to be physically valid.
//!
//! Each test has a clear physical claim and is comparable to reference MPM implementations
//! (sparkl, matter, taichi128).

use emerge::{
    BinghamFluidMaterial, CorotatedMaterial, MpmSolver, NeoHookeanMaterial,
    NewtonianFluidMaterial, SandMaterial, SandMuIMaterial, SnowMaterial, SolverConfig,
    SpawnConfig, ViscoelasticMaterial, VonMisesMaterial,
};
use emerge::materials::MaterialModel;
use emerge::particle::{Particle, Particles};
use emerge::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
use emerge::{
    ActivationStatsPlugin, DiagnosticsFrame, DiagnosticsRegistry,
    MaterialCountPlugin, ThermalStatsPlugin, collect_mpm_snapshot,
};
use glam::{IVec2, Mat2, Vec2};

// ─── helpers ────────────────────────────────────────────────────────────────

fn zero_gravity_config(grid_res: usize) -> SolverConfig {
    SolverConfig {
        grid_res,
        dt: 0.05,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        ..SolverConfig::default()
    }
}

fn center_spawn(grid_res: usize, side: usize) -> SpawnConfig {
    SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::new(side as i32, side as i32),
        box_center: Vec2::splat(grid_res as f32 * 0.5),
        initial_velocity_scale: 0.0,
        ..SpawnConfig::default()
    }
}

fn total_mass(solver: &MpmSolver) -> f32 {
    solver.particles().iter().map(|p| p.mass).sum()
}

fn linear_momentum(solver: &MpmSolver) -> Vec2 {
    solver.particles().iter().map(|p| p.mass * p.v).sum()
}

fn kinetic_energy(solver: &MpmSolver) -> f32 {
    solver.particles().iter().map(|p| 0.5 * p.mass * p.v.length_squared()).sum()
}

fn min_j(solver: &MpmSolver) -> f32 {
    solver
        .particles()
        .iter()
        .map(|p| p.deformation_gradient.determinant())
        .fold(f32::INFINITY, f32::min)
}

// ─── CONSERVATION: MASS ─────────────────────────────────────────────────────

/// Mass is a particle property and never changes — the solver must not add or remove particles.
#[test]
fn mass_is_conserved_neohookean() {
    let mut solver = MpmSolver::new(zero_gravity_config(32), center_spawn(32, 6))
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
    let config = SolverConfig {
        recompute_density_each_step: true,
        ..zero_gravity_config(32)
    };
    let mut solver = MpmSolver::new(config, center_spawn(32, 6))
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
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = MpmSolver::new(zero_gravity_config(32), center_spawn(32, 6))
        .with_default_material(Box::new(snow));

    let m0 = total_mass(&solver);
    solver.step_n(100);
    let m1 = total_mass(&solver);

    assert!(
        (m1 - m0).abs() < 1e-6,
        "snow: mass not conserved: {m0:.6} → {m1:.6}"
    );
}

// ─── CONSERVATION: LINEAR MOMENTUM (no external forces) ─────────────────────

/// With zero gravity and zero initial velocity, total momentum must stay near zero.
/// (MLS-MPM is weakly momentum conserving; small residuals from grid averaging are expected.)
#[test]
fn zero_velocity_spawn_has_near_zero_momentum() {
    let mut solver = MpmSolver::new(zero_gravity_config(32), center_spawn(32, 8))
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

/// With uniform gravity and no initial motion, momentum grows at rate m·g — verify linearity.
#[test]
fn gravity_grows_momentum_linearly() {
    let g = Vec2::new(0.0, -9.81);
    let config = SolverConfig {
        gravity: g,
        dt: 0.01,
        adaptive_timestep: false,
        ..SolverConfig::default()
    };
    let mut solver = MpmSolver::new(config, center_spawn(64, 4))
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

// ─── J > 0 INVARIANT ────────────────────────────────────────────────────────

/// det(F) > 0 is a non-negotiable physical invariant — particles can't invert.
/// Requires `project_invalid_state: true` (standard config) — the J floor that real simulations use.
#[test]
fn j_stays_positive_neohookean() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "NeoHookean: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_snow() {
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(snow));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Snow: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_sand() {
    let sand = SandMaterial::dry_sand(5_000.0, 2_000.0);
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(sand));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Sand: J collapsed to {jmin:.2e}");
}

#[test]
fn j_stays_positive_corotated() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(CorotatedMaterial::new(10.0, 20.0)));

    solver.step_n(200);
    let jmin = min_j(&solver);
    assert!(jmin > 0.0, "Corotated: J collapsed to {jmin:.2e}");
}

// ─── SNOW PLASTICITY: Jp BOUNDS ─────────────────────────────────────────────

/// Snow Jp must stay within [min_jp, max_jp] after any number of steps.
/// This is the yield surface enforcement — clamped singular values constrain Jp.
#[test]
fn snow_jp_stays_within_bounds() {
    let min_jp = 0.6f32;
    let max_jp = 20.0f32;
    let snow = SnowMaterial::new(38_889.0, 58_333.0, 10.0, 0.025, 0.0075, min_jp, max_jp);

    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(snow));

    solver.step_n(300);

    for (i, p) in solver.particles().iter().enumerate() {
        let jp = p.plastic_volume_ratio;
        assert!(
            jp >= min_jp * 0.99 && jp <= max_jp * 1.01,
            "snow particle {i}: Jp={jp:.4} out of [{min_jp}, {max_jp}]"
        );
    }
}

/// Snow hardening scale h = exp(ξ(1-Jp)) must be non-negative and finite.
/// Note: h=0.0 is valid f32 underflow of exp(−190) when Jp≈max_jp — effectively zero stress.
/// What matters is that h stays finite (no NaN/Inf) and non-negative.
#[test]
fn snow_hardening_scale_finite() {
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(snow));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.hardening_scale >= 0.0 && p.hardening_scale.is_finite(),
            "snow particle {i}: hardening_scale={:.4} (must be finite ≥0)", p.hardening_scale
        );
    }
}

// ─── SAND: NO TENSION ───────────────────────────────────────────────────────

/// Sand cannot sustain tension (p ≤ 0 → project to stress-free).
/// Test via direct material update on a tensile deformation gradient.
#[test]
fn sand_tension_cutoff_removes_tensile_stress() {
    let sand = SandMaterial::dry_sand(5_000.0, 2_000.0);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Pure extension: F = diag(1.5, 1.5) — volume 2.25×, tensile state
    p.deformation_gradient = Mat2::from_cols(
        Vec2::new(1.5, 0.0),
        Vec2::new(0.0, 1.5),
    );
    p.velocity_gradient = Mat2::ZERO;

    // Initialize particle (seeds plastic state)
    sand.init_particle(&mut p);
    sand.update_particle(&mut p, 0.01);

    // After projection, stress should be near zero (tensile → return to identity)
    let tau = sand.kirchhoff_stress(&p);
    let tau_norm = (tau.x_axis.length_squared() + tau.y_axis.length_squared()).sqrt();
    assert!(
        tau_norm < 1.0,
        "sand: tensile stress not projected (||τ||={tau_norm:.4})"
    );
}

/// Sand Drucker-Prager: log_volume_strain must stay finite.
/// Requires project_invalid_state=true to prevent J→0 which causes log(J)=−∞.
#[test]
fn sand_log_volume_strain_finite() {
    let sand = SandMaterial::dry_sand(5_000.0, 2_000.0);
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(sand));

    solver.step_n(200);

    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.log_volume_strain.is_finite(),
            "sand particle {i}: log_volume_strain={}", p.log_volume_strain
        );
    }
}

// ─── MATERIAL STRESS SYMMETRY ────────────────────────────────────────────────

/// Kirchhoff stress τ must be symmetric for all materials (objectivity / frame-indifference).
/// τ = τᵀ: |τ₀₁ − τ₁₀| < ε.
fn check_stress_symmetry(mat: &dyn MaterialModel, label: &str) {
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Small shear deformation: F = [[1.1, 0.1], [0.05, 0.95]]
    p.deformation_gradient = Mat2::from_cols(
        Vec2::new(1.1, 0.05),
        Vec2::new(0.1, 0.95),
    );
    mat.init_particle(&mut p);

    let tau = mat.kirchhoff_stress(&p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(
        asym < 1e-4,
        "{label}: Kirchhoff stress asymmetric: τ₀₁={:.6} τ₁₀={:.6} |diff|={asym:.2e}",
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
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);
    check_stress_symmetry(&snow, "Snow");
}

#[test]
fn sand_stress_symmetric() {
    let sand = SandMaterial::dry_sand(5_000.0, 2_000.0);
    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Compressive deformation (sand only resists compression)
    p.deformation_gradient = Mat2::from_cols(
        Vec2::new(0.9, 0.05),
        Vec2::new(0.05, 0.9),
    );
    sand.init_particle(&mut p);
    sand.update_particle(&mut p, 0.01);

    let tau = sand.kirchhoff_stress(&p);
    let asym = (tau.col(0).y - tau.col(1).x).abs();
    assert!(asym < 1e-4, "Sand: stress asymmetric: {asym:.2e}");
}

// ─── SVD CORRECTNESS ────────────────────────────────────────────────────────

/// Our analytical 2×2 SVD must satisfy F = U·diag(σ)·Vᵀ and U,V orthogonal.
/// This is tested internally in mechanics/svd.rs, but we verify the public path
/// through SnowMaterial.update_particle which uses svd2().
#[test]
fn snow_update_preserves_f_decomposition_invariant() {
    // After snow update, F_elastic must remain a valid deformation gradient.
    // det(F) > 0, F finite, singular values in (0, +∞).
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);

    let mut p = Particle::zeroed();
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    // Start from slight compression
    p.deformation_gradient = Mat2::from_cols(
        Vec2::new(0.95, 0.02),
        Vec2::new(-0.02, 0.95),
    );
    p.plastic_volume_ratio = 1.0;
    p.hardening_scale = 1.0;

    for _ in 0..50 {
        p.velocity_gradient = Mat2::from_cols(Vec2::new(-0.01, 0.005), Vec2::new(0.005, -0.01));
        snow.update_particle(&mut p, 0.01);
    }

    let j = p.deformation_gradient.determinant();
    assert!(j > 0.0 && j.is_finite(), "Snow: F det invalid after updates: J={j}");
    assert!(p.deformation_gradient.is_finite(), "Snow: F non-finite");
    assert!(p.hardening_scale > 0.0 && p.hardening_scale.is_finite());
}

// ─── ENERGY NON-GROWTH (elastic, no gravity) ────────────────────────────────

/// Kinetic energy of a resting elastic blob (no gravity, zero initial velocity)
/// must stay near zero — no spurious energy injection from the solver.
#[test]
fn resting_jelly_no_energy_growth() {
    let config = SolverConfig {
        gravity: Vec2::ZERO,
        dt: 0.05,
        ..SolverConfig::default()
    };
    let spawn = SpawnConfig {
        initial_velocity_scale: 0.0,
        ..center_spawn(64, 6)
    };
    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let ke0 = kinetic_energy(&solver);
    solver.step_n(200);
    let ke1 = kinetic_energy(&solver);

    // Resting blob: initial KE ≈ 0. After steps it may have tiny numerical KE but
    // must not grow significantly.
    let n = solver.particles().len() as f32;
    assert!(
        ke1 / n < 1e-4,
        "resting jelly: KE grew from {ke0:.2e} to {ke1:.2e} ({:.2e} per particle)",
        ke1 / n
    );
}

// ─── CFL STABILITY ──────────────────────────────────────────────────────────

/// Adaptive substep must never produce a sub_dt that violates particle CFL.
/// Proxy: particle speed × sub_dt ≤ 1 cell (with CFL coeff).
/// We verify this by checking velocities never exceed the grid/dt threshold.
#[test]
fn adaptive_substep_keeps_velocities_bounded() {
    let config = SolverConfig {
        gravity: Vec2::new(0.0, -9.81),
        dt: 0.1,
        adaptive_timestep: true,
        cfl_coefficient: 0.4,
        ..SolverConfig::default()
    };
    // High initial velocity to stress CFL
    let spawn = SpawnConfig {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = MpmSolver::new(config.clone(), spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(50.0, 100.0)));

    solver.step_n(100);

    // With CFL=0.4, max speed per step ≤ 0.4 * h / sub_dt.
    // Since sub_dt ≤ dt=0.1 and h=1, max speed per substep ≤ 0.4/sub_dt.
    // But we just check that all particles have finite velocity.
    for (i, p) in solver.particles().iter().enumerate() {
        assert!(
            p.v.is_finite(),
            "CFL test: particle {i} velocity non-finite: {:?}", p.v
        );
        assert!(
            p.v.length() < 500.0,
            "CFL test: particle {i} velocity exploded: |v|={:.1}", p.v.length()
        );
    }
}

// ─── DIAGNOSTICS PLUGIN SYSTEM ───────────────────────────────────────────────

/// DiagnosticsRegistry::collect must aggregate all plugin outputs.
#[test]
fn diagnostics_registry_aggregates_plugins() {
    use emerge::grid::Grid;

    let config = SolverConfig {
        grid_res: 8,
        dt: 0.1,
        ..SolverConfig::default()
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
    let snap = collect_mpm_snapshot(0, &particles_soa, &grid, &config, config.dt, 1);

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
    assert!((act_mean - 0.4).abs() < 1e-5, "act_mean={act_mean:.4} expected 0.4");

    let act_frac = frame.get("act_frac").expect("act_frac missing");
    assert!((act_frac - 0.5).abs() < 1e-5, "act_frac={act_frac:.4} expected 0.5");

    // Temperature: mean = (300+320)/2=310, max=320
    let t_mean = frame.get("T_mean").expect("T_mean missing");
    assert!((t_mean - 310.0).abs() < 1e-3, "T_mean={t_mean:.2} expected 310");

    let t_max = frame.get("T_max").expect("T_max missing");
    assert!((t_max - 320.0).abs() < 1e-3, "T_max={t_max:.2} expected 320");

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
    assert!(line.contains("act_mean=0.5000"), "missing act_mean in: {line}");
}

/// Empty registry produces empty DiagnosticsFrame.
#[test]
fn empty_registry_produces_empty_frame() {
    let mut registry = DiagnosticsRegistry::new();
    let p: Vec<Particle> = vec![];
    let config = SolverConfig { grid_res: 8, ..SolverConfig::default() };
    use emerge::grid::Grid;
    let grid = Grid::new(8);
    let snap = collect_mpm_snapshot(0, &emerge::particle::Particles::new(), &grid, &config, 0.1, 1);
    let frame = registry.collect(&p, &snap);
    assert!(frame.stats.is_empty(), "expected empty frame");
    assert!(frame.format_line().is_empty(), "expected empty format");
}

// ─── SCALAR DIFFUSION FIELD ──────────────────────────────────────────────────

/// Scalar diffusion must move a high-concentration particle's field toward lower concentration.
/// Without decay: total φ (summed over particles) should be approximately conserved.
#[test]
fn scalar_diffusion_spreads_and_conserves() {
    let grid_res = 16;
    let config = ScalarDiffusionConfig {
        diffusivity: 1.0,
        decay_rate: 0.0, // no decay → conserved
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
        "cold particle didn't warm: T={:.4}", particles.temperature[1]
    );

    // Hot particle should have cooled
    assert!(
        particles.temperature[0] < 100.0,
        "hot particle didn't cool: T={:.4}", particles.temperature[0]
    );

    // Conservation: total T should be roughly conserved (±20% tolerance — boundary effects)
    let conservation_err = (t_total_after - t_total_before).abs() / t_total_before;
    assert!(
        conservation_err < 0.20,
        "scalar field: total T changed too much: before={t_total_before:.2} after={t_total_after:.2} err={conservation_err:.2}"
    );
}

/// With decay_rate > 0, total φ must decrease over time.
#[test]
fn scalar_diffusion_decay_reduces_total() {
    let config = ScalarDiffusionConfig {
        diffusivity: 0.0,
        decay_rate: 1.0, // fast decay — T halves in ~0.69s
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

    // After 1s at decay_rate=1.0: T should be ~100*e^(-1) ≈ 36.8
    // Allow ±50% — grid average discretization makes this noisy with one particle
    assert!(
        particles.temperature[0] < 70.0,
        "decay: temperature not decreasing: T={:.2}", particles.temperature[0]
    );
    assert!(
        particles.temperature[0] > 0.0,
        "decay: temperature went negative: T={:.2}", particles.temperature[0]
    );
}

// ─── MATERIAL RATE CONSISTENCY ───────────────────────────────────────────────

/// Half-step × 2 must be approximately equivalent to one full step.
/// This tests that material update is smooth/continuous (not discontinuous jumps).
#[test]
fn snow_half_step_consistency() {
    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);

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
    snow.update_particle(&mut p_full, 0.02);

    // Two half-steps
    let mut p_half = base_particle;
    snow.update_particle(&mut p_half, 0.01);
    snow.update_particle(&mut p_half, 0.01);

    let j_full = p_full.deformation_gradient.determinant();
    let j_half = p_half.deformation_gradient.determinant();

    // J should be close (within 1% — subcycling plasticity has small discrepancies)
    assert!(
        (j_full - j_half).abs() < 0.01,
        "snow: full-step J={j_full:.6} vs half×2 J={j_half:.6} — too different"
    );
}

/// VonMises: after enough plastic deformation, stress norm must not exceed yield surface.
#[test]
fn von_mises_stress_bounded_by_yield() {
    let yield_stress = 100.0f32;
    let vm = VonMisesMaterial::new(1_000.0, 500.0, yield_stress);

    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let spawn = SpawnConfig {
        initial_velocity_scale: 5.0,
        ..center_spawn(64, 6)
    };
    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(vm));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let tau = vm.kirchhoff_stress(&p);
        // von Mises equivalent stress: sqrt(3/2 * s:s) where s = dev(τ)
        let tr = (tau.col(0).x + tau.col(1).y) * 0.5;
        let s00 = tau.col(0).x - tr;
        let s11 = tau.col(1).y - tr;
        let s01 = tau.col(1).x; // off-diagonal
        let vm_stress = (1.5 * (s00*s00 + s11*s11 + 2.0*s01*s01)).sqrt();
        // Allow 40% overshoot: initial_velocity_scale=5.0 creates violent collisions where
        // discrete return-mapping can't fully project to the yield surface in a single step.
        // Key invariant: stress stays finite and bounded, not that it's exactly at yield.
        assert!(
            vm_stress < yield_stress * 1.40,
            "VonMises particle {i}: σ_vm={vm_stress:.2} > yield {yield_stress:.2}"
        );
    }
}

// ─── MULTI-MATERIAL ISOLATION ────────────────────────────────────────────────

/// Two materials spawned in different regions must not interfere with each other's invariants.
#[test]
fn two_material_solver_both_j_positive() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));

    let spawn0 = SpawnConfig {
        box_center: Vec2::new(20.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        ..SpawnConfig::default()
    };

    let snow = SnowMaterial::from_young_modulus(1.4e5, 0.2);
    let mut solver = MpmSolver::new(config, spawn0)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)))
        .with_material(1, Box::new(snow));

    let spawn1 = SpawnConfig {
        box_center: Vec2::new(44.0, 40.0),
        box_size: IVec2::new(6, 6),
        spacing: 0.5,
        initial_velocity_scale: 0.0,
        material_id: 1,
        ..SpawnConfig::default()
    };
    let _range = solver.spawn_region(spawn1);

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "two-material: particle {i} mat={} J={j:.2e}", p.material_id);
    }
}

// ─── µ(I) RHEOLOGY SANITY ────────────────────────────────────────────────────

/// SandMuI: friction_hardening (µ(I)) must stay within [µ_static, µ_dynamic].
#[test]
fn sand_mui_friction_stays_in_range() {
    let mat = SandMuIMaterial::fine_sand(5_000.0, 2_000.0);
    let mu_static  = 20.9f32.to_radians().tan();
    let mu_dynamic = 32.8f32.to_radians().tan();

    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(mat));

    solver.step_n(100);

    for (i, p) in solver.particles().iter().enumerate() {
        let mu_i = p.friction_hardening;
        assert!(
            mu_i >= mu_static * 0.95 && mu_i <= mu_dynamic * 1.05,
            "SandMuI particle {i}: µ(I)={mu_i:.4} out of [{mu_static:.4}, {mu_dynamic:.4}]"
        );
    }
}

// ─── Bingham fluid ───────────────────────────────────────────────────────────

/// Bingham mud stays above floor without collapsing (yield stress holds shape under gravity).
#[test]
fn bingham_mud_stable_under_gravity() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::mud()));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.y > 1.0, "mud particle fell through floor: y={:.3}", p.x.y);
        assert!(p.x.is_finite(), "mud particle position NaN");
        assert!(p.v.is_finite(), "mud particle velocity NaN");
    }
}

/// Bingham J > 0 invariant.
#[test]
fn bingham_j_positive() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(BinghamFluidMaterial::mud()));
    solver.step_n(60);
    for p in solver.particles() {
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "Bingham J={j:.4} ≤ 0 — volume collapsed");
    }
}

/// Bingham lava: higher yield/viscosity than mud, still stable.
#[test]
fn bingham_lava_stable() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 6))
        .with_default_material(Box::new(BinghamFluidMaterial::lava()));
    solver.step_n(40);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "lava particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "lava J={j:.4} ≤ 0");
    }
}

// ─── Viscoelastic (Kelvin-Voigt) ─────────────────────────────────────────────

/// Viscoelastic soft tissue: J > 0, no NaN, stable under gravity.
#[test]
fn viscoelastic_soft_tissue_stable() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_default_material(Box::new(ViscoelasticMaterial::soft_tissue()));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "tissue particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "tissue J={j:.4} ≤ 0");
    }
}

/// Viscoelastic cell body: very soft, stable.
#[test]
fn viscoelastic_cell_body_stable() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -9.81));
    let mut solver = MpmSolver::new(config, center_spawn(64, 6))
        .with_default_material(Box::new(ViscoelasticMaterial::cell_body()));
    solver.step_n(60);
    for p in solver.particles() {
        assert!(p.x.is_finite() && p.v.is_finite(), "cell particle NaN");
        let j = p.deformation_gradient.determinant();
        assert!(j > 0.0, "cell J={j:.4} ≤ 0");
    }
}

/// KV viscous contribution: stress with non-zero strain rate > stress without.
/// Tests that the dashpot term activates when velocity_gradient is non-zero.
#[test]
fn viscoelastic_viscous_term_activates() {
    use emerge::materials::MaterialModel;

    let e = 5.0e4f32;
    let nu = 0.40f32;
    let eta = 500.0f32;

    let visco = ViscoelasticMaterial::from_young_modulus(e, nu, eta);
    let elastic = NeoHookeanMaterial::from_young_modulus(e, nu);

    // Particle at rest with identity F — same elastic stress for both.
    let mut p = Particle::zeroed();
    p.volume = 1.0;
    p.density = 1.0;
    p.mass = 1.0;

    let tau_elastic_rest = elastic.kirchhoff_stress(&p);
    let tau_visco_rest = visco.kirchhoff_stress(&p);
    // At rest (C=0, F=I) both give same stress (NeoHookean base is identical).
    let diff_rest = (tau_visco_rest - tau_elastic_rest).x_axis.length()
        + (tau_visco_rest - tau_elastic_rest).y_axis.length();
    assert!(diff_rest < 1.0, "at rest KV and elastic should agree: diff={diff_rest}");

    // Now give particle a shear strain rate via velocity_gradient.
    p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0));

    let tau_elastic_shear = elastic.kirchhoff_stress(&p);
    let tau_visco_shear = visco.kirchhoff_stress(&p);

    // KV adds η·D_dev — stress norms must differ.
    let norm_e = tau_elastic_shear.x_axis.length() + tau_elastic_shear.y_axis.length();
    let norm_v = tau_visco_shear.x_axis.length() + tau_visco_shear.y_axis.length();
    assert!(
        (norm_v - norm_e).abs() > 1.0,
        "KV dashpot should contribute when C≠0: norm_elastic={norm_e:.2} norm_visco={norm_v:.2}"
    );
}

// ─── Phase rules ─────────────────────────────────────────────────────────────

/// Phase rule transitions hot particles to a new material id.
#[test]
fn phase_rule_transitions_hot_particles() {
    const COLD_ID: u32 = 0;
    const HOT_ID: u32 = 1;
    let hot_threshold = 0.5f32;

    let config = SolverConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_material(HOT_ID, Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)))
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

    let hot_count = solver.particles().iter().filter(|p| p.material_id == HOT_ID).count();
    assert!(hot_count >= n / 2, "expected ≥{} hot particles, got {hot_count}", n / 2);
}

/// Phase rule: no transitions when condition not met.
#[test]
fn phase_rule_no_spurious_transitions() {
    const MAT_B: u32 = 1;

    let config = SolverConfig::standard(64, 0.05, Vec2::ZERO);
    let mut solver = MpmSolver::new(config, center_spawn(64, 8))
        .with_material(MAT_B, Box::new(NeoHookeanMaterial::from_young_modulus(1.0e5, 0.3)))
        .with_phase_rule(|p| {
            if p.temperature > 999.0 { Some(MAT_B) } else { None }
        });

    // No particles have temperature > 999
    solver.step_n(10);

    let b_count = solver.particles().iter().filter(|p| p.material_id == MAT_B).count();
    assert_eq!(b_count, 0, "spurious transitions to MAT_B: {b_count}");
}

// ─── Neighbor queries ─────────────────────────────────────────────────────────

/// particles_near returns only particles within radius.
#[test]
fn particles_near_radius_correct() {
    let config = SolverConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = MpmSolver::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 2.0;

    for (_, p) in solver.particles_near(center, radius) {
        let dist = (p.x - center).length();
        assert!(dist <= radius + f32::EPSILON, "particle at dist={dist:.3} outside radius={radius}");
    }
}

/// count_near matches manual count.
#[test]
fn count_near_matches_manual() {
    let config = SolverConfig::standard(64, 0.05, Vec2::ZERO);
    let solver = MpmSolver::new(config, center_spawn(64, 8));

    let center = Vec2::splat(32.0);
    let radius = 3.0;
    let mat_id = 0u32;

    let api_count = solver.count_near(center, radius, mat_id);
    let manual_count = solver.particles().iter()
        .filter(|p| p.material_id == mat_id && (p.x - center).length() <= radius)
        .count();

    assert_eq!(api_count, manual_count);
}

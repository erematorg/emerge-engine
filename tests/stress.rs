//! Stress tests — correctness under load, not just speed.
//!
//! Each test pushes one core-system axis (phase rules, boundaries, diagnostics plugins,
//! materials, force fields) to a higher count than any other test exercises, and asserts
//! real invariants (finite positions, J > 0, expected transition/metric counts) rather than
//! just timing. Complements benches/scaling.rs (perf) and tests/gpu.rs (GPU perf + stability).

extern crate emerge_engine as emerge;

use emerge::diagnostics::DiagnosticsRegistry;
use emerge::fields::RadialConfinementField;
use emerge::{
    DruckerPragerMaterial, FrictionBoundary, HeightmapBoundary, NeoHookeanMaterial, SimConfig,
    Simulation, SlipBoundary, SpawnRegion, lame_from_young,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;

fn base_config() -> SimConfig {
    SimConfig::standard(GRID, 0.1, Vec2::new(0.0, -0.3))
}

fn box_spawn(config: &SimConfig, side: i32, center: Vec2, material_id: u32) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: center,
        material_id,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(config)
    }
}

fn assert_all_finite_and_stable(sim: &Simulation, label: &str) {
    for (i, p) in sim.particles().iter().enumerate() {
        assert!(p.x.is_finite(), "{label}: particle {i} position NaN");
        assert!(p.v.is_finite(), "{label}: particle {i} velocity NaN");
        assert!(
            p.deformation_gradient.determinant() > 0.0,
            "{label}: particle {i} J collapsed (<=0)"
        );
    }
}

// ── phase_rule_count_stress ───────────────────────────────────────────────────
//
// Registers up to 32 simultaneous phase rules (each checking a different temperature
// threshold against a distinct target material) and steps long enough for rules whose
// threshold is below the actual particle temperature to fire. Asserts every particle stays
// finite/stable AND that at least one rule actually transitioned particles (rules aren't dead
// code — the per-substep evaluation in `add_phase_rule` is genuinely exercised).

#[test]
fn phase_rule_count_stress() {
    for &n_rules in &[1usize, 8, 32] {
        let config = base_config();
        let spawn = box_spawn(&config, 16, Vec2::splat(GRID as f32 * 0.5), 0);
        let (l, u) = lame_from_young(5.0e4, 0.3);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
            .with_material(1, Box::new(NeoHookeanMaterial::new(l, u)))
            .with_material(2, Box::new(NeoHookeanMaterial::new(l, u)))
            .with_material(3, Box::new(NeoHookeanMaterial::new(l, u)));

        // Give particles a uniform starting temperature above every rule's threshold so at
        // least the loosest rules fire deterministically.
        let n = sim.particles().len();
        for i in 0..n {
            sim.particles_mut().temperature[i] = 500.0;
        }

        for i in 0..n_rules {
            let threshold = 100.0 + i as f32 * 10.0; // all thresholds < 500.0 -> all fire
            let target_material = (i % 3 + 1) as u32; // cycle 1,2,3 — never 0 (the source id)
            sim.add_phase_rule(move |p| {
                if p.material_id == 0 && p.temperature > threshold {
                    Some(target_material)
                } else {
                    None
                }
            });
        }

        sim.step_n(10);
        assert_all_finite_and_stable(&sim, &format!("phase_rule_count_stress n={n_rules}"));

        // At least one particle should have transitioned away from material 0, proving the
        // rules were actually evaluated (not just registered and ignored).
        if n_rules > 0 {
            let transitioned = sim.particles().iter().any(|p| p.material_id != 0);
            assert!(
                transitioned,
                "phase_rule_count_stress n={n_rules}: no particle transitioned — rules not firing"
            );
        }
    }
}

// ── boundary_count_stress ────────────────────────────────────────────────────
//
// Stacks Slip + Friction + Heightmap boundaries simultaneously (each contributes its own
// apply_to_grid_velocity / clamp_particle_position pass). Asserts particles stay inside the
// domain and finite under three active boundary implementations at once — a case no other
// test exercises (existing tests use exactly one boundary).

#[test]
fn boundary_count_stress() {
    let config = base_config();
    let spawn = box_spawn(&config, 16, Vec2::splat(GRID as f32 * 0.5), 0);
    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.4)))
        .with_boundary(Box::new(HeightmapBoundary::flat_floor(GRID, 4.0, 0.4)));

    sim.step_n(30);
    assert_all_finite_and_stable(&sim, "boundary_count_stress");

    let min = 0.0;
    let max = GRID as f32;
    for (i, p) in sim.particles().iter().enumerate() {
        assert!(
            p.x.x >= min && p.x.x <= max && p.x.y >= min && p.x.y <= max,
            "boundary_count_stress: particle {i} escaped domain at {:?}",
            p.x
        );
    }
}

// ── diagnostics_plugin_count_stress ──────────────────────────────────────────
//
// Registers up to 32 closure-based diagnostics plugins and collects every frame for 20 steps.
// Asserts the collected frame always reports exactly one metric set per plugin (no silent
// drops) and every reported value is finite (no plugin produces NaN/garbage under load).

#[test]
fn diagnostics_plugin_count_stress() {
    for &n_plugins in &[1usize, 8, 32] {
        let config = base_config();
        let spawn = box_spawn(&config, 12, Vec2::splat(GRID as f32 * 0.5), 0);
        let (l, u) = lame_from_young(5.0e4, 0.3);
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)));

        let mut registry = DiagnosticsRegistry::new();
        for i in 0..n_plugins {
            let name: &'static str = Box::leak(format!("plugin_{i}").into_boxed_str());
            registry.register_fn(name, move |particles, _snap| {
                vec![(format!("metric_{i}"), particles.len() as f32)]
            });
        }

        for _ in 0..20 {
            sim.step();
            let snap = sim.diagnostics_snapshot();
            let particles_aos = sim.particles().to_vec();
            let frame = registry.collect(&particles_aos, &snap);
            assert_eq!(
                frame.stats.len(),
                n_plugins,
                "diagnostics_plugin_count_stress n={n_plugins}: expected {n_plugins} stat \
                 entries (one per plugin), got {}",
                frame.stats.len()
            );
            for (metric_name, value) in &frame.stats {
                assert!(
                    value.is_finite(),
                    "diagnostics_plugin_count_stress n={n_plugins}: metric {metric_name} \
                     is non-finite"
                );
            }
        }
        assert_all_finite_and_stable(
            &sim,
            &format!("diagnostics_plugin_count_stress n={n_plugins}"),
        );
    }
}

// ── material_count_stress (correctness companion to benches::material_count_scaling) ────────
//
// Up to MAX_MATERIAL_SLOTS materials simultaneously active, each in its own spawn region.
// Asserts every particle across every material stays finite and elastically stable (J > 0).

#[test]
fn material_count_stress() {
    for &k in &[1usize, 8, 32, emerge::MAX_MATERIAL_SLOTS] {
        let config = base_config();
        let side = 2i32;
        let grid_dim = (k as f32).sqrt().ceil() as usize;
        let spacing_cells = GRID as f32 / (grid_dim as f32 + 1.0);

        let spawns: Vec<SpawnRegion> = (0..k)
            .map(|i| {
                let col = i % grid_dim;
                let row = i / grid_dim;
                let center = Vec2::new(
                    (col as f32 + 1.0) * spacing_cells,
                    (row as f32 + 1.0) * spacing_cells,
                );
                box_spawn(&config, side, center, i as u32)
            })
            .collect();

        let (l, u) = lame_from_young(5.0e4, 0.3);
        let mut sim = Simulation::empty(config).with_boundary(Box::new(SlipBoundary::new(2)));
        for i in 0..k {
            sim = sim.with_material(i as u32, Box::new(NeoHookeanMaterial::new(l, u)));
        }
        for spawn in spawns {
            let _ = sim.add_body(spawn);
        }

        sim.step_n(10);
        assert_all_finite_and_stable(&sim, &format!("material_count_stress k={k}"));
    }
}

// ── force_field_count_stress (correctness companion to benches::force_field_scaling) ────────
//
// Up to 16 simultaneous RadialConfinementField instances (mirrors GPU MAX_FORCE_FIELDS).
// Asserts particles stay finite and confined to within each field's strictest radius.

#[test]
fn force_field_count_stress() {
    let config = base_config();
    let center = Vec2::splat(GRID as f32 * 0.5);
    let spawn = box_spawn(&config, 10, center, 0);
    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)));

    const N_FIELDS: usize = 16;
    const RADIUS: f32 = 12.0;
    for _ in 0..N_FIELDS {
        sim.add_force_field(Box::new(RadialConfinementField::new(center, RADIUS, 200.0)));
    }

    sim.step_n(30);
    assert_all_finite_and_stable(&sim, "force_field_count_stress");

    // All 16 fields confine to the same radius+margin — particles should not have escaped
    // far beyond it despite 16x the confinement evaluations per substep.
    let max_allowed = RADIUS * 1.5;
    for (i, p) in sim.particles().iter().enumerate() {
        let dist = (p.x - center).length();
        assert!(
            dist <= max_allowed,
            "force_field_count_stress: particle {i} escaped confinement, dist={dist}"
        );
    }
}

// ── sand_count_stress (granular plasticity under multi-material load) ───────────────────────
//
// Sand (Drucker-Prager) mixed with elastic NeoHookean across several materials simultaneously —
// checks that yield-surface return-mapping (the most numerically active plasticity path)
// stays stable when sharing a step with other constitutive models, not just in isolation.

#[test]
fn sand_count_stress() {
    let config = base_config();
    let (l, u) = lame_from_young(5.0e4, 0.3);

    const SAND_A: u32 = 1;
    const SAND_B: u32 = 2;
    const SAND_C: u32 = 3;

    let spawns = [
        box_spawn(&config, 10, Vec2::new(16.0, 16.0), SAND_A),
        box_spawn(&config, 10, Vec2::new(32.0, 16.0), SAND_B),
        box_spawn(&config, 10, Vec2::new(48.0, 16.0), SAND_C),
        box_spawn(&config, 10, Vec2::new(32.0, 48.0), 0),
    ];

    let mut sim = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_material(
            SAND_A,
            Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)),
        )
        .with_material(
            SAND_B,
            Box::new(DruckerPragerMaterial::low_friction(133.3, 0.333)),
        )
        .with_material(
            SAND_C,
            Box::new(DruckerPragerMaterial::dilatant(133.3, 0.333)),
        )
        .with_boundary(Box::new(SlipBoundary::new(2)));

    for spawn in spawns {
        let _ = sim.add_body(spawn);
    }

    sim.step_n(50);
    assert_all_finite_and_stable(&sim, "sand_count_stress");
}

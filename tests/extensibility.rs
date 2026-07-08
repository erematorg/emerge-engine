//! Modularity proof: every extension seam accepts EXTERNAL implementations.
//!
//! This file deliberately acts as a third-party consumer: everything below is
//! implemented against `emerge::prelude::*` alone -- no engine internals, no
//! private modules, no engine-side changes. If a seam ever regresses into
//! requiring internal access (a private type in a trait signature, a
//! registration path not exported, a prelude gap), this file stops compiling
//! or its assertions fail.
//!
//! Each custom implementation carries an `AtomicU32` call counter, so the tests
//! assert the engine genuinely *invoked* the external code through the seam --
//! compiling against a trait proves the interface exists; the counters prove
//! the substep loop actually calls it.
//!
//! Seams covered (the extension table in ARCHITECTURE.md §7):
//! - `MaterialModel`      (custom constitutive response)
//! - `Field`              (custom external body force)
//! - `BoundaryCondition`  (custom grid boundary)
//! - `DiagnosticsPlugin`  (custom observation)
//! - phase rules          (closure-based matter state change)

extern crate emerge_engine as emerge;

use std::sync::atomic::{AtomicU32, Ordering};

use emerge::prelude::*;

// ─── Shared scene ────────────────────────────────────────────────────────────

const GRID: usize = 48;
const DT: f32 = 0.1;

fn config() -> SimConfig {
    SimConfig {
        max_substeps_per_step: 16,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.4))
    }
}

fn spawn(config: &SimConfig) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 8),
        box_center: Vec2::new(24.0, 20.0),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(config)
    }
}

// Counters live for the whole test process ('static) so the boxed trait objects
// handed to the solver can share them with the asserting test body. Leaking one
// AtomicU32 per test is deliberate and harmless.
fn counter() -> &'static AtomicU32 {
    Box::leak(Box::new(AtomicU32::new(0)))
}

// ─── 1. Custom material through the MaterialModel seam ──────────────────────

/// Small-strain linear elastic solid: τ = λ·tr(ε)·I + 2µ·ε with ε = (F+Fᵀ)/2 − I.
/// Deliberately NOT one of emerge's shipped models -- the point is that a
/// consumer can bring their own constitutive law.
#[derive(Debug)]
struct ExternalLinearElastic {
    lambda: f32,
    mu: f32,
    stress_calls: &'static AtomicU32,
}

impl MaterialModel for ExternalLinearElastic {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        self.stress_calls.fetch_add(1, Ordering::Relaxed);
        let f = particles.deformation_gradient[i];
        let strain = (f + f.transpose()) * 0.5 - Mat2::IDENTITY;
        let trace = strain.col(0).x + strain.col(1).y;
        Mat2::from_diagonal(Vec2::splat(self.lambda * trace)) + strain * (2.0 * self.mu)
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        // Standard elastic wave-speed bound: dt <= CFL·dx / c, c = sqrt((λ+2µ)/ρ).
        let c = ((self.lambda + 2.0 * self.mu) / density.max(1.0e-6)).sqrt();
        material_cfl * cell_width / c.max(1.0e-6)
    }
}

#[test]
fn external_material_is_invoked_and_simulates_stably() {
    let stress_calls = counter();
    let cfg = config();
    let sp = spawn(&cfg);
    let mut solver = Simulation::new(cfg, sp)
        .with_default_material(Box::new(ExternalLinearElastic {
            lambda: 40.0,
            mu: 25.0,
            stress_calls,
        }))
        .with_boundary(Box::new(SlipBoundary::new(2)));

    solver.step_n(60);

    assert!(
        stress_calls.load(Ordering::Relaxed) > 0,
        "engine never called the external material's kirchhoff_stress -- \
         the MaterialModel seam did not dispatch to consumer code"
    );
    let particles = solver.particles();
    for i in 0..particles.len() {
        assert!(
            particles.x[i].is_finite(),
            "external material produced non-finite position at particle {i}"
        );
        assert!(
            particles.deformation_gradient[i].determinant() > 0.0,
            "external material collapsed J at particle {i}"
        );
    }
}

// ─── 2. Custom force field through the Field seam ────────────────────────────

/// Constant horizontal wind. `prepare` and `acceleration` both counted.
struct ExternalWind {
    accel: Vec2,
    prepare_calls: &'static AtomicU32,
    accel_calls: &'static AtomicU32,
}

impl Field for ExternalWind {
    fn prepare(&mut self, _particles: &Particles) {
        self.prepare_calls.fetch_add(1, Ordering::Relaxed);
    }
    fn acceleration(&self, _particles: &Particles, _i: usize) -> Vec2 {
        self.accel_calls.fetch_add(1, Ordering::Relaxed);
        self.accel
    }
}

#[test]
fn external_field_is_invoked_and_measurably_pushes_the_body() {
    let run = |wind: Option<(Vec2, &'static AtomicU32, &'static AtomicU32)>| -> f32 {
        let cfg = config();
        let sp = spawn(&cfg);
        let mut solver = Simulation::new(cfg, sp)
            .with_default_material(Box::new(NeoHookeanMaterial::new(30.0, 20.0)))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        if let Some((accel, prepare_calls, accel_calls)) = wind {
            solver.add_force_field(Box::new(ExternalWind {
                accel,
                prepare_calls,
                accel_calls,
            }));
        }
        solver.step_n(50);
        let particles = solver.particles();
        let n = particles.len() as f32;
        (0..particles.len()).map(|i| particles.x[i].x).sum::<f32>() / n
    };

    let baseline_x = run(None);
    let prepares = counter();
    let accels = counter();
    let windy_x = run(Some((Vec2::new(0.6, 0.0), prepares, accels)));

    assert!(
        prepares.load(Ordering::Relaxed) > 0,
        "Field::prepare never called on external field"
    );
    assert!(
        accels.load(Ordering::Relaxed) > 0,
        "Field::acceleration never called on external field"
    );
    assert!(
        windy_x > baseline_x + 0.5,
        "external wind field had no measurable effect: baseline com_x={baseline_x:.3}, \
         windy com_x={windy_x:.3} -- the Field seam is not actually feeding forces \
         into the substep loop"
    );
}

// ─── 3. Custom boundary through the BoundaryCondition seam ──────────────────

/// Box walls + call counters: verifies the engine consults an external boundary
/// for BOTH grid-velocity application and particle-position clamping.
#[derive(Debug)]
struct ExternalWalls {
    thickness: usize,
    grid_calls: &'static AtomicU32,
    clamp_calls: &'static AtomicU32,
}

impl BoundaryCondition for ExternalWalls {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        self.grid_calls.fetch_add(1, Ordering::Relaxed);
        let t = self.thickness;
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let hi = grid_res.saturating_sub(t + 1);
        if (x < t && velocity.x < 0.0) || (x > hi && velocity.x > 0.0) {
            velocity.x = 0.0;
        }
        if (y < t && velocity.y < 0.0) || (y > hi && velocity.y > 0.0) {
            velocity.y = 0.0;
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        self.clamp_calls.fetch_add(1, Ordering::Relaxed);
        let min = self.thickness as f32;
        let max = grid_res as f32 - self.thickness as f32;
        position.clamp(Vec2::splat(min), Vec2::splat(max))
    }
}

#[test]
fn external_boundary_is_invoked_and_contains_the_body() {
    let grid_calls = counter();
    let clamp_calls = counter();
    let cfg = config();
    let sp = spawn(&cfg);

    let mut solver = Simulation::new(cfg, sp)
        .with_default_material(Box::new(NeoHookeanMaterial::new(30.0, 20.0)))
        .with_boundary(Box::new(ExternalWalls {
            thickness: 2,
            grid_calls,
            clamp_calls,
        }));

    solver.step_n(80);

    assert!(
        grid_calls.load(Ordering::Relaxed) > 0,
        "engine never called the external boundary's apply_to_grid_velocity"
    );
    assert!(
        clamp_calls.load(Ordering::Relaxed) > 0,
        "engine never called the external boundary's clamp_particle_position"
    );
    let particles = solver.particles();
    for i in 0..particles.len() {
        let p = particles.x[i];
        assert!(
            p.x >= 1.9 && p.x <= GRID as f32 - 1.9 && p.y >= 1.9 && p.y <= GRID as f32 - 1.9,
            "particle {i} escaped the external boundary: {p:?} -- the \
             BoundaryCondition seam is not actually enforcing consumer walls"
        );
    }
}

// ─── 4. Custom diagnostics through the DiagnosticsPlugin seam ────────────────

struct ExternalMaxHeight;

impl DiagnosticsPlugin for ExternalMaxHeight {
    fn name(&self) -> &'static str {
        "external_max_height"
    }
    fn collect(&mut self, particles: &[Particle], _snapshot: &SimSnapshot) -> Vec<(String, f32)> {
        let max_y = particles
            .iter()
            .map(|p| p.x.y)
            .fold(f32::NEG_INFINITY, f32::max);
        vec![("max_height".to_string(), max_y)]
    }
}

#[test]
fn external_diagnostics_plugin_reports_through_the_registry() {
    let cfg = config();
    let sp = spawn(&cfg);
    let mut solver = Simulation::new(cfg, sp)
        .with_default_material(Box::new(NeoHookeanMaterial::new(30.0, 20.0)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    solver.step_n(10);

    let mut registry = DiagnosticsRegistry::new();
    registry.register(Box::new(ExternalMaxHeight));
    let particles_aos = solver.collect_particles();
    let snapshot = solver.diagnostics_snapshot();
    let frame: DiagnosticsFrame = registry.collect(&particles_aos, &snapshot);

    let value = frame
        .stats
        .iter()
        .find(|(k, _)| k == "max_height")
        .map(|(_, v)| *v);
    assert!(
        value.is_some(),
        "external plugin's stat never surfaced through DiagnosticsRegistry"
    );
    let v = value.unwrap();
    assert!(
        v.is_finite() && v > 0.0 && v < GRID as f32,
        "external plugin's max_height value implausible: {v}"
    );
}

// ─── 5. Phase rules (closure seam) ───────────────────────────────────────────

#[test]
fn external_phase_rule_transitions_matter() {
    let cfg = config();
    let sp = spawn(&cfg);
    let mut solver = Simulation::new(cfg, sp)
        .with_default_material(Box::new(NeoHookeanMaterial::new(30.0, 20.0)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    // Second material to transition into.
    let handle: MaterialHandle = solver.register_material(Box::new(NewtonianFluidMaterial::new(
        1.0, 1.0e-3, 50.0, 7.0,
    )));

    let threshold = 18.0_f32;
    let target = handle.id();
    solver.add_phase_rule(move |p| {
        if p.x.y < threshold && p.material_id != target {
            Some(target)
        } else {
            None
        }
    });

    solver.step_n(60); // long enough for the blob to fall below the threshold

    let particles = solver.particles();
    let transitioned = (0..particles.len())
        .filter(|&i| particles.material_id[i] == target)
        .count();
    assert!(
        transitioned > 0,
        "external phase rule never fired -- closure seam not evaluated during stepping"
    );
}

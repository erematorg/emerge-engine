//! Particle-count scaling benchmark.
//!
//! Measures one full `solver.step()` (P2G → grid → G2P, all substeps) as the particle
//! count grows. The curve reveals how emerge scales and catches any O(N²) regression.
//!
//!   cargo bench --bench scaling
//!
//! Read the report: target/criterion/step_scaling/report/index.html

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use emerge::{
    MpmSolver, NeoHookeanMaterial, SlipBoundary, SolverConfig, SpawnConfig, lame_from_young,
};
use glam::{IVec2, Vec2};

/// Build a settled, actively-moving NeoHookean block of roughly `target` particles.
fn build_solver(target: usize) -> MpmSolver {
    const GRID: usize = 64;
    let config = SolverConfig {
        max_substeps_per_step: 64,
        ..SolverConfig::standard(GRID, 0.1, Vec2::new(0.0, -0.3))
    };

    // spacing 0.5 → 4 particles per cell; side ≈ sqrt(target)·0.5 cells.
    let side = ((target as f32).sqrt() * 0.5).ceil() as i32;
    let spawn = SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: Vec2::splat(GRID as f32 * 0.5),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };

    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut solver = MpmSolver::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)));

    // Warm to a representative active state (NeoHookean keeps bouncing — work stays steady).
    solver.step_n(20);
    solver
}

fn step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_scaling");
    for &target in &[100usize, 500, 1000, 2500, 5000] {
        let mut solver = build_solver(target);
        let actual = solver.particles().len();
        group.bench_with_input(BenchmarkId::from_parameter(actual), &actual, |b, _| {
            b.iter(|| solver.step());
        });
    }
    group.finish();
}

criterion_group!(benches, step_scaling);
criterion_main!(benches);

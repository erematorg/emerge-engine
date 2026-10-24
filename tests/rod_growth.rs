//! Checks for `rod::growth`: the tip segment's `rest_edge_length` must match
//! the closed-form logistic solution (Verhulst 1838), as
//! `resource_regrowth_matches_logistic_curve` does in `tests/accuracy.rs`, so
//! the dynamics are sigmoidal, not just increasing. The second test checks the
//! mechanical consequence: the rod tip moves further out as growth pulls the
//! elastic equilibrium outward, through the internal-force integrator, not a
//! direct position change.

extern crate emerge_engine as emerge;
use emerge::grid::Grid;
use emerge::rod::{Growth, RodMaterial, apply_growth, build_straight_rod, rod_cfl_dt, step_rod};
use glam::Vec2;

#[test]
fn tip_segment_growth_matches_logistic_curve() {
    let rate = 0.5_f32;
    let l0 = 0.05_f32; // starts short, well below the carrying capacity
    let k = 1.0_f32;
    let dt = 0.01_f32;
    let n_steps = 500u32;
    let t_total = dt * n_steps as f32;

    let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(l0, 0.0), 2, 0.3, 1.0);
    rod.rest_edge_length[0] = l0;

    // Empty grid -- ungated growth (no `resistance` configured) never reads
    // it, this is just satisfying the signature.
    let grid = Grid::new(16);
    let mut growth = Growth::new(rate, k);
    for _ in 0..n_steps {
        apply_growth(&mut rod, &mut growth, &grid, Vec2::Y, 1.0, dt);
    }

    let l_final = rod.rest_edge_length[0];
    // Closed-form logistic solution (Verhulst 1838): L(t) = K/(1+((K-L0)/L0)*e^(-r*t))
    let l_expected = k / (1.0 + ((k - l0) / l0) * (-rate * t_total).exp());
    let tolerance = l_expected * 0.02;

    assert!(
        (l_final - l_expected).abs() < tolerance,
        "tip growth doesn't match the real closed-form logistic solution: \
         expected={l_expected:.5} measured={l_final:.5} tolerance={tolerance:.5}"
    );
}

#[test]
fn growth_pulls_actual_rod_tip_further_away_through_real_elastic_dynamics() {
    // Checkable mechanical consequence: growth changes the TARGET
    // (rest_edge_length), then the already-proven internal-force
    // integrator (step_rod) does the work of stretching the actual
    // geometry to follow it -- not a position hack.
    let mut grown = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.1, 0.0), 2, 0.3, 1.0);
    let mut baseline = grown.clone();

    let material = RodMaterial::from_young_modulus_rectangular(1.0e6, 0.02, 0.01, 50.0, 5.0);
    let mut growth = Growth::new(2.0, 1.0); // fast growth rate for a short test

    // CFL-safe dt for this material and geometry, recomputed each step as
    // rest_edge_length grows: step_rod is a standalone integrator with no
    // stability enforcement of its own (see `rod_cfl_dt`: the caller picks a
    // stable dt). Growth and mechanics advance by the same dt each iteration.
    let grid = Grid::new(16);
    let mut elapsed = 0.0_f32;
    while elapsed < 3.0 {
        let dt = rod_cfl_dt(&grown, &material, 0.5).min(0.001);
        apply_growth(&mut grown, &mut growth, &grid, Vec2::Y, 1.0, dt);
        step_rod(&mut grown, &material, Vec2::ZERO, Vec2::ZERO, 0.0, 1.0, dt);
        step_rod(
            &mut baseline,
            &material,
            Vec2::ZERO,
            Vec2::ZERO,
            0.0,
            1.0,
            dt,
        );
        elapsed += dt;
    }

    // Fast, sustained growth matures the tip edge and triggers point insertion
    // (see `growth.rs`'s "cell division" doc), so `grown` may have more points
    // than it started with: measure the actual tip (`.last()`), not index 1.
    let grown_length = (*grown.x.last().unwrap() - grown.x[0]).length();
    let baseline_length = (baseline.x[1] - baseline.x[0]).length();

    assert!(
        grown_length > baseline_length * 2.0,
        "growth should have measurably stretched the real rod geometry via the \
         existing elastic dynamics: grown={grown_length:.4} baseline={baseline_length:.4}"
    );
}

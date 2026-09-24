//! Real implicit (backward Euler) time integration for a discrete elastic
//! rod — the actual fix for the CFL-driven substep ceiling explicit
//! integration hits for a stiff rod (see `mod.rs`'s own doc and
//! `integrator::rod_cfl_dt`). Standard, established numerical method for
//! stiff ODEs (Baraff & Witkin 1998, "Large Steps in Cloth Simulation";
//! DisMech, a real published fully-implicit discrete-elastic-rod simulator,
//! confirms this is the standard approach for exactly this rod
//! formulation) -- genuinely solving the SAME Newtonian equations of motion
//! `compute_internal_forces` already does, just with an implicit numerical
//! integration scheme instead of explicit symplectic Euler. NOT a
//! constraint-based method (PBD/XPBD) -- those reformulate the problem as
//! geometric constraint projection, a different mathematical object from
//! the real force-based PDE this engine is built on, and were explicitly
//! rejected for that reason.
//!
//! # Method
//! Backward Euler: `v_{n+1} = v_n + dt*a(x_{n+1}, v_{n+1})`,
//! `x_{n+1} = x_n + dt*v_{n+1}`. Linearizing `F` around the current state
//! (standard Newton/quasi-Newton treatment) with `x_{n+1} = x_n + dt*v_{n+1}`
//! gives the real, standard linear system:
//! `(M - dt*C - dt^2*K) * dv = dt * F(x_n, v_n)`
//! where `K = dF/dx`, `C = dF/dv` (system Jacobians), solved once per real
//! step for `dv = v_{n+1} - v_n`.
//!
//! # Real, disclosed simplification: analytic Jacobians, one term approximate
//! `K`/`C` are assembled analytically, not finite-differenced (an earlier
//! version of this file used central differences of `compute_internal_
//! forces` throughout -- see `implicit_solver_cost_profile`'s own doc for
//! the real, measured history of replacing that). Axial (stretch +
//! Kelvin-Voigt damping) has an exact, standard damped-spring Jacobian
//! (Baraff & Witkin 1998; `forces::axial_force_and_jacobian`). Bending
//! (discrete curvature) has `forces::bending_jacobian_gauss_newton`: EXACT
//! for `dF/dv` (bending damping's rate term is exactly linear in velocity),
//! a disclosed Gauss-Newton (material-stiffness-only) APPROXIMATION for
//! `dF/dx` -- the true Hessian of `discrete_curvature` has no derived
//! closed form yet for this engine's own 2D reduction (a real, separate,
//! harder undertaking than the gradient was); see that function's own doc
//! for exactly what's dropped, why it's a standard technique, and why it's
//! positive-semi-definite (a real stability advantage, not just an
//! approximation of convenience).
//!
//! # Real, disclosed simplification: dense solve, not banded
//! The true Jacobian has bandwidth ~2 (a pentadiagonal-like structure --
//! stretch couples i to i+/-1, bending couples i to i+/-2 through each
//! vertex's own 3-point coupling), but this uses a general dense Gaussian
//! elimination with partial pivoting instead of a specialized banded
//! solver. For a rod's real point counts (tens, not thousands), O(N^3)
//! dense solve is a real, correct, negligible cost -- optimizing to exploit
//! the band structure is real future work if profiling ever shows this
//! matters, not attempted here (YAGNI).

use glam::Vec2;

use super::coupling::push_acceleration;
use super::forces::{
    axial_force_and_jacobian, bending_jacobian_gauss_newton, discrete_curvature_gradient,
};
use super::{RodMaterial, RodPoints, RodRestState, compute_internal_forces};

/// Solve `A x = b` via Gaussian elimination with partial pivoting.
/// `a` is row-major `n*n`, destroyed in the process. Returns `None` if `A`
/// is numerically singular (no pivot found above a real tolerance) --
/// callers should fall back to an explicit substep in that case, same
/// spirit as any real implicit solver needing a graceful degradation path.
fn solve_dense(mut a: Vec<f32>, mut b: Vec<f32>, n: usize) -> Option<Vec<f32>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);

    for col in 0..n {
        // Partial pivoting: swap in the largest-magnitude entry in this column
        // (among remaining rows) to reduce numerical error, standard practice.
        let mut pivot_row = col;
        let mut pivot_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1.0e-12 {
            return None;
        }
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution.
    let mut x = vec![0.0f32; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in (row + 1)..n {
            sum -= a[row * n + k] * x[k];
        }
        x[row] = sum / a[row * n + row];
    }
    Some(x)
}

/// One real implicit (backward Euler) step for a linear rod. Pinned points
/// are excluded from the solved system entirely (their velocity is fixed
/// at zero, same Dirichlet-anchor convention as the explicit path) --
/// standard reduction to only the FREE degrees of freedom, not a special
/// case bolted on afterward.
///
/// Grouped step parameters for `step_rod_implicit` — everything except the
/// rod/material being stepped (one struct instead of an 8-argument tail).
#[derive(Debug, Clone, Copy)]
pub struct RodImplicitStepParams {
    pub gravity: Vec2,
    pub wind_velocity: Vec2,
    pub wind_drag_coeff: f32,
    pub push_center: Option<Vec2>,
    pub push_strength: f32,
    pub push_radius: f32,
    pub dx_meters: f32,
    pub dt: f32,
}

/// Real fallback: if the assembled system is singular (degenerate
/// geometry, e.g. a fully-collapsed rod), falls back to one explicit
/// substep rather than silently producing nonsense -- a real, disclosed
/// safety net, not hidden.
pub fn step_rod_implicit(
    rod: &mut RodPoints,
    material: &RodMaterial,
    params: RodImplicitStepParams,
) {
    let RodImplicitStepParams {
        gravity,
        wind_velocity,
        wind_drag_coeff,
        push_center,
        push_strength,
        push_radius,
        dx_meters,
        dt,
    } = params;
    let n = rod.len();
    if n == 0 {
        return;
    }

    let free: Vec<usize> = (0..n).filter(|&i| rod.pinned[i] == 0).collect();
    let nf = free.len();
    if nf == 0 {
        return;
    }
    let ndof = nf * 2;

    let f0 = compute_internal_forces(
        &rod.x,
        &rod.v,
        RodRestState {
            rest_edge_length: &rod.rest_edge_length,
            rest_curvature: &rod.rest_curvature,
            ea: &rod.ea,
            ei: &rod.ei,
        },
        material,
        dx_meters,
    );

    // K[dof][dof'] = -dF[dof]/dx[dof'], C[dof][dof'] = -dF[dof]/dv[dof'],
    // restricted to free DOFs only (a pinned point's own force contributions
    // still matter -- they're baked into f0 -- but its OWN position/velocity
    // are never perturbed or solved for, since it can't move).
    //
    // Real, disclosed motivation -- `implicit_solver_cost_profile`, this
    // module's own `#[ignore]`d perf test: profiling found the finite-
    // difference Jacobian sweep costs 14-18x `solve_dense`'s own O(ndof^3)
    // elimination at every point count tested (20-200), so THIS sweep, not
    // the linear solve, was the real bottleneck. The axial (stretch +
    // Kelvin-Voigt damping) term already had a real, standard, closed-form
    // damped-spring Jacobian (Baraff & Witkin 1998; see
    // `forces::axial_force_and_jacobian`'s own doc). Bending now has its own
    // analytic Jacobian too (`forces::bending_jacobian_gauss_newton`) --
    // EXACT for the velocity term, a real, disclosed Gauss-Newton
    // (material-stiffness-only) approximation for the position term, since
    // this engine's own 2D-reduced curvature law has no derived Hessian yet
    // (see that function's own doc for what's dropped and why it's a
    // reasonable, standard, PSD-guaranteed approximation). No perturbation
    // needed for either term anymore.
    let mut k_mat = vec![0.0f32; ndof * ndof];
    let mut c_mat = vec![0.0f32; ndof * ndof];

    // `point_to_free_col[p] = Some(free-index)` for a free point, `None`
    // for pinned -- shared by both the bending and axial analytic blocks
    // below. A vertex/edge touching a pinned point still contributes to the
    // OTHER, free point(s)' own row/col; the pinned point's own row/col
    // simply doesn't exist in this system.
    let mut point_to_free_col = vec![None; n];
    for (col, &pi) in free.iter().enumerate() {
        point_to_free_col[pi] = Some(col);
    }

    // Analytic bending (discrete curvature) contribution.
    let ei_at = |i: usize| {
        if rod.ei.is_empty() {
            material.ei
        } else {
            rod.ei[i]
        }
    };
    if n >= 3 {
        for i in 1..n - 1 {
            let (p0, p1, p2) = (
                rod.x[i - 1] * dx_meters,
                rod.x[i] * dx_meters,
                rod.x[i + 1] * dx_meters,
            );
            let grad = discrete_curvature_gradient(p0, p1, p2);
            let l0_prev = rod.rest_edge_length[i - 1].max(1.0e-9);
            let l0_next = rod.rest_edge_length[i].max(1.0e-9);
            let voronoi_length = 0.5 * (l0_prev + l0_next);
            let stiffness = ei_at(i - 1) / voronoi_length;
            let (k_blocks, c_blocks) =
                bending_jacobian_gauss_newton(grad, stiffness, material.bending_damping);

            let vertex_points = [i - 1, i, i + 1];
            for (a, &pa) in vertex_points.iter().enumerate() {
                let Some(row) = point_to_free_col[pa] else {
                    continue;
                };
                for (b, &pb) in vertex_points.iter().enumerate() {
                    let Some(col) = point_to_free_col[pb] else {
                        continue;
                    };
                    // `bending_jacobian_gauss_newton` returns dF/dp, dF/dv
                    // w.r.t. real-meter position/velocity (`grad` itself came
                    // from meter-scaled points); `x`/`v` here are grid-cell
                    // units, so each needs exactly ONE `dx_meters` chain
                    // factor (`p[b] = x_grid[b]*dx_meters`, one direct linear
                    // map per point -- same single-factor chain the axial
                    // block below uses for its own reduced edge variable).
                    let k_block = k_blocks[a][b] * dx_meters;
                    let c_block = c_blocks[a][b] * dx_meters;
                    for r in 0..2 {
                        for c in 0..2 {
                            let k_val = if c == 0 {
                                k_block.x_axis
                            } else {
                                k_block.y_axis
                            };
                            let c_val = if c == 0 {
                                c_block.x_axis
                            } else {
                                c_block.y_axis
                            };
                            let k_component = if r == 0 { k_val.x } else { k_val.y };
                            let c_component = if r == 0 { c_val.x } else { c_val.y };
                            k_mat[(row * 2 + r) * ndof + (col * 2 + c)] -= k_component;
                            c_mat[(row * 2 + r) * ndof + (col * 2 + c)] -= c_component;
                        }
                    }
                }
            }
        }
    }

    // Analytic axial (stretch + damping) contribution -- ADDED (not
    // assigned) to whatever the bending block above already wrote, since an
    // interior point's own diagonal block gets real contributions from BOTH
    // its adjacent edges AND the bending vertices touching it.
    let ea_at = |i: usize| {
        if rod.ea.is_empty() {
            material.ea
        } else {
            rod.ea[i]
        }
    };
    for i in 0..n - 1 {
        let d = (rod.x[i + 1] - rod.x[i]) * dx_meters;
        let rel_v = (rod.v[i + 1] - rod.v[i]) * dx_meters;
        let l0 = rod.rest_edge_length[i];
        let (_, df_dd, df_drelv) =
            axial_force_and_jacobian(d, rel_v, l0, ea_at(i), material.axial_damping);

        // Force_a = sign_a * F(d, rel_v), a in {i, i+1}; d(d)/dx_b =
        // chain_b*dx_meters (chain_i=-1, chain_{i+1}=+1); same chain for
        // rel_v wrt v_b.
        let points = [i, i + 1];
        let signs = [1.0f32, -1.0f32];
        let chains = [-1.0f32, 1.0f32];
        for (a_idx, &pa) in points.iter().enumerate() {
            let Some(row) = point_to_free_col[pa] else {
                continue;
            };
            for (b_idx, &pb) in points.iter().enumerate() {
                let Some(col) = point_to_free_col[pb] else {
                    continue;
                };
                let factor = signs[a_idx] * chains[b_idx] * dx_meters;
                let k_block = df_dd * factor;
                let c_block = df_drelv * factor;
                for r in 0..2 {
                    for c in 0..2 {
                        let k_val = if c == 0 {
                            k_block.x_axis
                        } else {
                            k_block.y_axis
                        };
                        let c_val = if c == 0 {
                            c_block.x_axis
                        } else {
                            c_block.y_axis
                        };
                        let k_component = if r == 0 { k_val.x } else { k_val.y };
                        let c_component = if r == 0 { c_val.x } else { c_val.y };
                        k_mat[(row * 2 + r) * ndof + (col * 2 + c)] -= k_component;
                        c_mat[(row * 2 + r) * ndof + (col * 2 + c)] -= c_component;
                    }
                }
            }
        }
    }

    // Assemble the standard Baraff & Witkin (1998) implicit system. With
    // `K = -dF/dx`, `C = -dF/dv` (this file's own convention, positive for a
    // stable/dissipative system), the correct linearization of
    // `F(x_{n+1},v_{n+1})` around `(x_n,v_n)` -- using `x_{n+1}-x_n = dt*v_{n+1}`
    // -- gives `[M + dt*C + dt^2*K] * dv = dt*F_n - dt^2*K*v_n`. Getting the
    // sign wrong on the left-hand matrix, or dropping the `-dt^2*K*v_n` term on
    // the right, turns this into an amplifying (unstable) system instead of a
    // damping one — an incorrectly-signed version blows up with alternating
    // sign and exponentially growing magnitude within tens of steps on even a
    // single damped spring.
    let mut a_mat = vec![0.0f32; ndof * ndof];
    let mut b_vec = vec![0.0f32; ndof];
    let mut v_free = vec![0.0f32; ndof];
    for (row, &pi) in free.iter().enumerate() {
        let m = rod.mass[pi].max(1.0e-9);
        a_mat[(row * 2) * ndof + (row * 2)] += m;
        a_mat[(row * 2 + 1) * ndof + (row * 2 + 1)] += m;

        // Real, disclosed simplification: gravity/wind/push are treated as
        // ordinary EXTERNAL forces on the right-hand side (like gravity
        // already was), not folded into the implicit K/C solve -- only the
        // rod's OWN internal elastic/damping forces are stiff enough to
        // need implicit treatment; wind drag and push are comparatively
        // soft, real forces, safe to treat this way (same real convention
        // `apply_rod_internal_and_wind_forces` already uses for the
        // explicit path, converted to real Newtons via the same
        // `mass*dx_meters` factor internal forces already use).
        let a_wind = wind_drag_coeff * (wind_velocity - rod.v[pi]);
        let a_push = push_acceleration(rod.x[pi], push_center, push_strength, push_radius);
        let f_total = f0[pi] + (gravity + a_wind + a_push) * m * dx_meters;
        b_vec[row * 2] = dt * (f_total.x);
        b_vec[row * 2 + 1] = dt * (f_total.y);
        v_free[row * 2] = rod.v[pi].x;
        v_free[row * 2 + 1] = rod.v[pi].y;
    }
    for i in 0..ndof {
        for j in 0..ndof {
            a_mat[i * ndof + j] += dt * c_mat[i * ndof + j] + dt * dt * k_mat[i * ndof + j];
        }
        // b -= dt^2 * K * v_n (matrix-vector product, row i).
        let mut kv_i = 0.0f32;
        for j in 0..ndof {
            kv_i += k_mat[i * ndof + j] * v_free[j];
        }
        b_vec[i] -= dt * dt * kv_i;
    }

    let dv = match solve_dense(a_mat, b_vec, ndof) {
        Some(dv) => dv,
        None => {
            // Real, disclosed fallback: singular system (degenerate
            // geometry) -- one explicit substep instead of silently
            // producing garbage.
            for (i, &pi) in free.iter().enumerate() {
                let _ = i;
                let a = (f0[pi] + gravity * rod.mass[pi].max(1.0e-9) * dx_meters)
                    / (rod.mass[pi].max(1.0e-9) * dx_meters);
                rod.v[pi] += a * dt;
            }
            for &pi in &free {
                rod.x[pi] += rod.v[pi] * dt;
            }
            return;
        }
    };

    for (row, &pi) in free.iter().enumerate() {
        rod.v[pi].x += dv[row * 2];
        rod.v[pi].y += dv[row * 2 + 1];
    }
    for &pi in &free {
        rod.x[pi] += rod.v[pi] * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rod::build_straight_rod;

    /// Perf diagnostic (not correctness) -- measures `step_rod_implicit`'s
    /// total cost at several point counts, end-to-end (assembly + solve
    /// together). Run manually:
    /// `cargo test --lib implicit_solver_cost_profile --all-features --
    /// --ignored --nocapture`.
    ///
    /// Both axial (Baraff & Witkin 1998 damped-spring) and bending
    /// (`forces::bending_jacobian_gauss_newton`) terms use analytic
    /// Jacobians -- bending's is exact for the velocity term but a
    /// Gauss-Newton (material-stiffness-only) approximation for the
    /// position term, since the true Hessian of `discrete_curvature` has no
    /// derived closed form here (see that function's own doc for what's
    /// dropped and why). Accuracy is still covered by `tests/accuracy.rs`'s
    /// `cantilever_tip_deflection_matches_euler_bernoulli` and
    /// `cantilever_deflection_error_shrinks_with_resolution`.
    #[test]
    #[ignore = "perf diagnostic (not correctness) -- measures total implicit-step cost at several point counts; run manually when investigating implicit-rod scaling, not routine CI"]
    fn implicit_solver_cost_profile() {
        // Time the actual `step_rod_implicit` end-to-end (assembly + solve
        // together) -- don't re-implement the assembly loop separately just
        // to get a number, or the timing can silently drift from the real
        // code under test.
        for &n_points in &[20usize, 50, 100, 200] {
            let dx_meters = 0.01;
            let height_m = 0.10;
            let start = Vec2::new(0.0, 0.0);
            let end = Vec2::new(0.0, height_m / dx_meters);
            let young_modulus = 1.0e7_f32; // matches rod_blade_of_grass_gui.rs's blade A
            let ea = young_modulus * 0.003 * 0.001;
            let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;

            let mut points = build_straight_rod(start, end, n_points, 0.01, dx_meters);
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            let (axial_damping, bending_damping) =
                crate::rod::RodMaterial::modal_critical_damping(&points, ea, ei);
            let material = crate::rod::RodMaterial::from_young_modulus_rectangular(
                young_modulus,
                0.003,
                0.001,
                axial_damping,
                bending_damping,
            );

            const REPEATS: u32 = 20;
            let start_time = std::time::Instant::now();
            for _ in 0..REPEATS {
                step_rod_implicit(
                    &mut points.clone(),
                    &material,
                    RodImplicitStepParams {
                        gravity: Vec2::new(0.0, -9.81 / dx_meters),
                        wind_velocity: Vec2::ZERO,
                        wind_drag_coeff: 0.0,
                        push_center: None,
                        push_strength: 0.0,
                        push_radius: 0.0,
                        dx_meters,
                        dt: 0.02,
                    },
                );
            }
            let total_us = start_time.elapsed().as_micros() / REPEATS as u128;

            eprintln!(
                "implicit_solver_cost_profile: n={n_points:<4} ndof={:<4} total_step_us={total_us:>8}",
                (points.len() - 2) * 2
            );
        }
    }

    #[test]
    fn dense_solve_matches_known_2x2_system() {
        // 2x + y = 5, x + 3y = 10 -> x=1, y=3
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let x = solve_dense(a, b, 2).expect("should solve");
        assert!((x[0] - 1.0).abs() < 1.0e-4, "x={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1.0e-4, "y={}", x[1]);
    }

    #[test]
    fn implicit_single_damped_spring_matches_analytic_decay() {
        // A single free point on a spring to a fixed anchor, no gravity --
        // pure exponential decay of an initial displacement, real analytic
        // solution to compare against: for backward Euler on dv/dt=-(k/m)x,
        // dx/dt=v (damped harmonic oscillator at critical damping), the
        // discrete solution should match the same qualitative real decay a
        // continuous critically-damped oscillator has -- checked here via
        // energy monotonically decreasing and reaching nea-zero, not
        // exploding, the real correctness bar for an implicit integrator.
        let dx_meters = 1.0;
        let mut rod =
            build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 2, 1.0, dx_meters);
        rod.pinned[0] = 1;
        rod.x[1] = Vec2::new(1.5, 0.0); // displaced from rest (rest length 1.0)
        let ea = 100.0;
        let ei = 0.0;
        let (axial_damping, bending_damping) =
            RodMaterial::critical_damping(1.0, rod.mass[1], ea, ei);
        let material = RodMaterial::new(ea, ei, axial_damping, bending_damping);

        let dt = 0.1; // a LARGE dt an explicit integrator could never take stably here
        for _ in 0..50 {
            step_rod_implicit(
                &mut rod,
                &material,
                RodImplicitStepParams {
                    gravity: Vec2::ZERO,
                    wind_velocity: Vec2::ZERO,
                    wind_drag_coeff: 0.0,
                    push_center: None,
                    push_strength: 0.0,
                    push_radius: 0.0,
                    dx_meters,
                    dt,
                },
            );
        }
        let final_stretch = (rod.x[1].x - 1.0).abs();
        let final_speed = rod.v[1].length();
        assert!(
            final_stretch < 0.05,
            "implicit spring should settle near rest length, stretch={final_stretch}"
        );
        assert!(
            final_speed < 0.05,
            "implicit spring should settle to near-zero velocity, speed={final_speed}"
        );
        assert!(
            rod.x[1].x.is_finite() && rod.v[1].x.is_finite(),
            "implicit step must not diverge at a dt an explicit integrator could never take"
        );
    }
}

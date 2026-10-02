//! Implicit (backward Euler) integration of grain-grain normal contact
//! forces. Contact stiffness makes `critical_timestep` force ~3500 substeps
//! per frame whatever the grain count; a stiff mass-spring-damper system is
//! what `spacetime::rod::implicit` already integrates (Baraff & Witkin 1998,
//! "Large Steps in Cloth Simulation"), and this module reuses its assembly
//! pattern and its `solve_dense` (`pub(crate)`).
//!
//! # Method
//! `(M + dt*C + dt^2*K) * dv = dt*F_n - dt^2*K*v_n`, `K = -dF/dx`,
//! `C = -dF/dv`: the system of `rod::implicit::step_rod_implicit` (see there
//! for the derivation), one dense solve per step over every free grain's
//! (x, y) velocity.
//!
//! # Scope: normal contact force only
//! `F_n = kn*overlap - c_n*v_n` (Cundall & Strack 1979, the first term of
//! `grain_contact_law::resolve_contact_core_linear`). The Coulomb-capped
//! tangential and rolling springs are not included: once a capped spring
//! saturates (sliding) its force no longer depends on the unknown velocity,
//! and backward Euler gains nothing there. A complete integrator needs a
//! hybrid that falls back to explicit bursts during sliding.
//!
//! # Scope: no wall contact yet
//! Only grain-grain pairs; a floor or wall needs its own single-body
//! Jacobian (one fixed point instead of two free ones).

use glam::Vec2;

use crate::matter::particle::Grain;
use crate::rod::implicit::solve_dense;

/// Analytic Jacobian of the grain-grain normal contact force, built the way
/// `rod::forces::axial_force_and_jacobian` (finite-difference checked) builds
/// its own, not separately checked here.
///
/// `d = x_j - x_i`, `rel_v = v_j - v_i`, `r_sum = r_i + r_j`. Returns
/// `(force_on_j, df_dd, df_drelv)` -- force on `i` is the exact negation
/// (Newton's third law), matching `grain_contact_law::resolve_contact_pair`'s
/// own convention.
///
/// Let `l = |d|`, `dir = d/l`, `overlap = r_sum - l`, `v_n = rel_v . dir`,
/// `s = F_n = kn*overlap - c_n*v_n` (the force law of
/// `grain_contact_law::resolve_contact_core_linear` before its `.max(0.0)`,
/// linearized in the active, overlapping regime as `rod::implicit` does).
/// Force on `j` (repulsive, along `+dir`): `force = dir * s`.
///
/// `ds/dd = -kn*dir - c_n*(P*rel_v)/l` (`P = I - outer(dir,dir)`, since
/// `d(dir)/dd = P/l`). `df_dd = outer(dir, ds/dd) + s*(P/l)`, the product
/// rule on `dir*s`, the shape of `axial_force_and_jacobian`'s `df_dd`.
/// `ds/d(rel_v) = -c_n*dir`, so `df_drelv = -c_n*outer(dir,dir)`.
pub(crate) fn normal_contact_force_and_jacobian(
    d: Vec2,
    rel_v: Vec2,
    r_sum: f32,
    normal_stiffness: f32,
    normal_damping: f32,
) -> (Vec2, glam::Mat2, glam::Mat2) {
    let l = d.length().max(1.0e-9);
    let dir = d / l;
    let overlap = r_sum - l;
    let v_n = rel_v.dot(dir);
    let s = normal_stiffness * overlap - normal_damping * v_n;
    let force_on_j = dir * s;

    let outer_dir = glam::Mat2::from_cols(dir * dir.x, dir * dir.y);
    let p = glam::Mat2::IDENTITY - outer_dir;

    let ds_dd = dir * (-normal_stiffness) - (p * rel_v) * (normal_damping / l);
    let df_dd = glam::Mat2::from_cols(dir * ds_dd.x, dir * ds_dd.y) + p * (s / l);
    let df_drelv = outer_dir * (-normal_damping);

    (force_on_j, df_dd, df_drelv)
}

/// One implicit (backward Euler) step for grain-grain normal contact forces
/// across a population, structured like `rod::implicit::step_rod_implicit`:
/// assembles `K`/`C` from every active contact's analytic Jacobian (over
/// non-pinned grains), solves one dense system for `dv`, applies it.
///
/// `pinned[i] = true` removes grain `i` from the solved set (its velocity is
/// fixed this step), `rod::implicit`'s Dirichlet anchor, for a fixed floor
/// or wall grain (`grains_repose_angle.rs`'s large pinned floor grain).
///
/// `gravity` is an ordinary external force on the right-hand side, as in
/// `rod::implicit`: only the contact spring is stiff enough to need implicit
/// treatment.
///
/// A singular system (degenerate geometry) falls back to one explicit
/// substep for the free grains, as `rod::implicit` does.
pub fn step_grains_implicit_normal_only(
    grains: &mut [Grain],
    pinned: &[bool],
    normal_stiffness: f32,
    normal_damping: f32,
    gravity: Vec2,
    dt: f32,
) {
    let n = grains.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(pinned.len(), n);

    let free: Vec<usize> = (0..n).filter(|&i| !pinned[i]).collect();
    let nf = free.len();
    if nf == 0 {
        return;
    }
    let ndof = nf * 2;

    struct Pair {
        i: usize,
        j: usize,
    }
    let mut pairs: Vec<Pair> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let max_dist = grains[i].radius + grains[j].radius;
            if (grains[j].x - grains[i].x).length_squared() <= max_dist * max_dist {
                pairs.push(Pair { i, j });
            }
        }
    }

    let mut point_to_free_col = vec![None; n];
    for (col, &gi) in free.iter().enumerate() {
        point_to_free_col[gi] = Some(col);
    }

    let mut k_mat = vec![0.0f32; ndof * ndof];
    let mut c_mat = vec![0.0f32; ndof * ndof];
    let mut f0 = vec![Vec2::ZERO; n];

    for pair in &pairs {
        let (gi, gj) = (&grains[pair.i], &grains[pair.j]);
        let d = gj.x - gi.x;
        let rel_v = gj.v - gi.v;
        let r_sum = gi.radius + gj.radius;
        let (force_on_j, df_dd, df_drelv) =
            normal_contact_force_and_jacobian(d, rel_v, r_sum, normal_stiffness, normal_damping);

        f0[pair.j] += force_on_j;
        f0[pair.i] -= force_on_j;

        // Same real block-distribution pattern as `rod::implicit`'s own
        // axial assembly loop: two points {i, j}, force on j is `+`, on i is
        // `-`, and `d`/`rel_v` each change by `+1`/`-1` per point's own DOF
        // (`d = x_j - x_i`).
        let points = [pair.i, pair.j];
        let signs = [-1.0f32, 1.0f32];
        let chains = [-1.0f32, 1.0f32];
        for (a_idx, &pa) in points.iter().enumerate() {
            let Some(row) = point_to_free_col[pa] else {
                continue;
            };
            for (b_idx, &pb) in points.iter().enumerate() {
                let Some(col) = point_to_free_col[pb] else {
                    continue;
                };
                let factor = signs[a_idx] * chains[b_idx];
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

    let mut a_mat = vec![0.0f32; ndof * ndof];
    let mut b_vec = vec![0.0f32; ndof];
    let mut v_free = vec![0.0f32; ndof];
    for (row, &gi) in free.iter().enumerate() {
        let m = grains[gi].mass.max(1.0e-9);
        a_mat[(row * 2) * ndof + (row * 2)] += m;
        a_mat[(row * 2 + 1) * ndof + (row * 2 + 1)] += m;

        let f_total = f0[gi] + gravity * m;
        b_vec[row * 2] = dt * f_total.x;
        b_vec[row * 2 + 1] = dt * f_total.y;
        v_free[row * 2] = grains[gi].v.x;
        v_free[row * 2 + 1] = grains[gi].v.y;
    }
    for i in 0..ndof {
        for j in 0..ndof {
            a_mat[i * ndof + j] += dt * c_mat[i * ndof + j] + dt * dt * k_mat[i * ndof + j];
        }
        let mut kv_i = 0.0f32;
        for j in 0..ndof {
            kv_i += k_mat[i * ndof + j] * v_free[j];
        }
        b_vec[i] -= dt * dt * kv_i;
    }

    match solve_dense(a_mat, b_vec, ndof) {
        Some(dv) => {
            for (row, &gi) in free.iter().enumerate() {
                grains[gi].v.x += dv[row * 2];
                grains[gi].v.y += dv[row * 2 + 1];
            }
        }
        None => {
            // Singular system: one explicit substep for the free grains
            // instead of garbage, as `rod::implicit` does.
            for &gi in &free {
                let a =
                    (f0[gi] + gravity * grains[gi].mass.max(1.0e-9)) / grains[gi].mass.max(1.0e-9);
                grains[gi].v += a * dt;
            }
        }
    }
    for &gi in &free {
        let v = grains[gi].v;
        grains[gi].x += v * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An N-grain vertical stack on a fixed floor reaches the closed-form
    /// equilibrium overlap at every contact (`overlap_i =
    /// weight_above_contact_i / kn`) at dt = 1/60 s, far above this stiffness's
    /// `critical_timestep`.
    #[test]
    fn n_grain_stack_converges_to_closed_form_equilibrium_overlap() {
        let radius = 0.1_f32;
        let mass = 1.0_f32;
        let gravity = Vec2::new(0.0, -9.81);
        let normal_stiffness = 1.0e5_f32;
        let normal_damping = 20.0_f32; // real, moderate dashpot -- helps reach equilibrium, not required for correctness
        let n_free = 5usize; // grains 1..=5 free, grain 0 is the fixed floor

        // Floor (pinned) at y=0; free grains stacked directly touching
        // (zero initial overlap) at y = radius, 3*radius, 5*radius, ...
        let mut grains: Vec<Grain> = Vec::with_capacity(n_free + 1);
        grains.push(Grain::new(Vec2::new(0.0, -radius), radius, mass)); // floor, huge effective presence via pinning, real radius kept physical
        for k in 0..n_free {
            let y = radius + (2.0 * radius) * k as f32;
            grains.push(Grain::new(Vec2::new(0.0, y), radius, mass));
        }
        let mut pinned = vec![false; n_free + 1];
        pinned[0] = true;

        let dt = 1.0 / 60.0;
        let steps = 3000;
        for _ in 0..steps {
            step_grains_implicit_normal_only(
                &mut grains,
                &pinned,
                normal_stiffness,
                normal_damping,
                gravity,
                dt,
            );
        }

        // Contact i (grain i to grain i+1, grain 0 the floor) carries the
        // weight of the (n_free - i) free grains above it.
        for i in 0..n_free {
            let overlap = (grains[i].radius + grains[i + 1].radius)
                - (grains[i + 1].x - grains[i].x).length();
            let weight_above = (n_free - i) as f32 * mass * gravity.y.abs();
            let predicted_overlap = weight_above / normal_stiffness;
            let rel_err = (overlap - predicted_overlap).abs() / predicted_overlap.max(1.0e-9);
            assert!(
                rel_err < 0.02,
                "contact {i}: overlap={overlap}, predicted={predicted_overlap}, rel_err={rel_err}"
            );
        }
        // Every free grain has settled (a static scene), not merely passing
        // through a zero crossing of an oscillation.
        for &gi in &[1, n_free] {
            assert!(
                grains[gi].v.length() < 1.0e-2,
                "grain {gi} should have settled to near-zero velocity, v={:?}",
                grains[gi].v
            );
        }
    }
}

/// Diagnostic: wall-clock cost of advancing 1/60 s with one implicit step
/// against the explicit substeps `grain_contact_law::critical_timestep`
/// forces at the same stiffness. Both resolve the same normal-only physics
/// (same `kn`/`cn`, same pair list), so the comparison isolates the
/// integrator. Run manually:
/// `cargo test --release --lib grain_implicit_realtime_speedup -- --ignored --nocapture`.
#[cfg(test)]
mod perf_diagnostic {
    use super::*;
    use crate::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
    use std::time::Instant;

    fn explicit_normal_only_step(
        grains: &mut [Grain],
        pinned: &[bool],
        kn: f32,
        cn: f32,
        gravity: Vec2,
        dt: f32,
    ) {
        let n = grains.len();
        let mut force = vec![Vec2::ZERO; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = grains[j].x - grains[i].x;
                let l = d.length().max(1.0e-9);
                let r_sum = grains[i].radius + grains[j].radius;
                if l >= r_sum {
                    continue;
                }
                let dir = d / l;
                let overlap = r_sum - l;
                let rel_v = grains[j].v - grains[i].v;
                let v_n = rel_v.dot(dir);
                let f_n = (kn * overlap - cn * v_n).max(0.0);
                force[j] += dir * f_n;
                force[i] -= dir * f_n;
            }
        }
        for i in 0..n {
            if pinned[i] {
                continue;
            }
            let a = force[i] / grains[i].mass + gravity;
            grains[i].v += a * dt;
            grains[i].x += grains[i].v * dt;
        }
    }

    fn build_stack(n_free: usize, radius: f32, mass: f32) -> (Vec<Grain>, Vec<bool>) {
        let mut grains = Vec::with_capacity(n_free + 1);
        grains.push(Grain::new(Vec2::new(0.0, -radius), radius, mass));
        for k in 0..n_free {
            let y = radius + (2.0 * radius) * k as f32;
            grains.push(Grain::new(Vec2::new(0.0, y), radius, mass));
        }
        let mut pinned = vec![false; n_free + 1];
        pinned[0] = true;
        (grains, pinned)
    }

    #[test]
    #[ignore = "perf diagnostic, run explicitly with --release --ignored --nocapture"]
    fn grain_implicit_realtime_speedup() {
        let radius = 0.01_f32;
        let e_pa = 1.0e7_f32;
        let density = 1600.0_f32;
        let mass = density * std::f32::consts::PI * radius * radius;
        let gravity = Vec2::new(0.0, -9.81 / radius); // grid-unit gravity, same convention grains_repose_angle.rs uses
        let cfg = ContactLawConfig::dry_sand(e_pa, radius, mass * 0.5, 35.0, 0.20);
        let dt_explicit = critical_timestep(mass * 0.5, &cfg);
        let frame_dt = 1.0 / 60.0;
        let substeps_per_frame = (frame_dt / dt_explicit).ceil() as usize;

        for &n_free in &[10usize, 50, 200] {
            let (mut grains_explicit, pinned) = build_stack(n_free, radius, mass);
            let start = Instant::now();
            for _ in 0..substeps_per_frame {
                explicit_normal_only_step(
                    &mut grains_explicit,
                    &pinned,
                    cfg.normal_stiffness,
                    cfg.normal_damping,
                    gravity,
                    dt_explicit,
                );
            }
            let explicit_us = start.elapsed().as_micros();

            let (mut grains_implicit, _) = build_stack(n_free, radius, mass);
            let start = Instant::now();
            step_grains_implicit_normal_only(
                &mut grains_implicit,
                &pinned,
                cfg.normal_stiffness,
                cfg.normal_damping,
                gravity,
                frame_dt,
            );
            let implicit_us = start.elapsed().as_micros();

            println!(
                "n_free={n_free:<4} substeps_per_frame={substeps_per_frame:<8} explicit_us={explicit_us:<10} implicit_us={implicit_us:<10} speedup={:.1}x",
                explicit_us as f64 / implicit_us.max(1) as f64
            );
        }
    }
}

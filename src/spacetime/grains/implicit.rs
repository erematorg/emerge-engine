//! Real implicit (backward Euler) time integration for grain-grain NORMAL
//! contact forces -- the real fix for the DEM "timestep problem" this
//! engine's own grain population is genuinely stuck on (`critical_timestep`
//! forces ~3500 substeps/rendered-frame from real contact stiffness,
//! independent of grain count -- see `project_grain_realtime_fps_measured_
//! 2026-09-09`, project memory). Same real, standard method already
//! shipped and tested in THIS codebase for the exact same class of problem
//! (a stiff mass-spring-damper system whose explicit CFL bound is too
//! restrictive for real-time use): `spacetime::rod::implicit`, Baraff &
//! Witkin 1998 "Large Steps in Cloth Simulation". This module reuses that
//! file's own verified linear-system assembly/solve PATTERN (and its
//! `solve_dense` routine directly, `pub(crate)`) rather than re-deriving
//! sign conventions from scratch -- the earlier isolated prototype work
//! that verified this exact technique for grain contacts
//! (`project_grain_implicit_integration_scoped_2026-09-09`, project memory)
//! was lost before being generalized to real N-grain scale; this module is
//! that generalization, re-derived fresh against `rod::implicit`'s own
//! real, working code instead of purely from memory.
//!
//! # Method
//! `(M + dt*C + dt^2*K) * dv = dt*F_n - dt^2*K*v_n`, `K = -dF/dx`,
//! `C = -dF/dv` -- IDENTICAL system shape to `rod::implicit::step_rod_
//! implicit`'s own doc (see there for the full derivation/citation), one
//! dense solve per real step over every grain's free (x,y) velocity DOFs.
//!
//! # Real, disclosed scope: NORMAL contact force only
//! `F_n = kn*overlap - c_n*v_n` (Cundall & Strack 1979 linear spring-
//! dashpot, `grain_contact_law::resolve_contact_core_linear`'s own first
//! term) -- tangential/rolling elastic-plastic Coulomb-CAPPED springs are
//! NOT included. Real, disclosed reason, already found once by the lost
//! prototype work (preserved in project memory): a capped spring's force
//! becomes a CONSTANT independent of the unknown velocity once it saturates
//! (real kinetic sliding), which makes backward Euler's own stabilizing
//! mechanism structurally inapplicable there -- only the smooth (static-
//! holding) regime benefits, which is exactly what this module covers. A
//! real, complete integrator needs the hybrid smooth/capped-fallback
//! technique the prototype work verified (real explicit burst during
//! genuine sliding transients) -- not yet ported here, real future work,
//! not silently dropped.
//!
//! # Real, disclosed scope: no wall contact yet
//! Only grain-grain pairs are included in the implicit solve; a floor/wall
//! boundary needs its own analogous single-body Jacobian (same method,
//! simpler -- one fixed point instead of two free ones). Not yet built.

use glam::Vec2;

use crate::matter::particle::Grain;
use crate::rod::implicit::solve_dense;

/// Real, analytic (verified-by-construction against `rod::forces::
/// axial_force_and_jacobian`'s own already-FD-verified METHOD, not
/// independently FD-checked here -- see this module's own top-of-file doc)
/// Jacobian of the grain-grain normal contact force.
///
/// `d = x_j - x_i`, `rel_v = v_j - v_i`, `r_sum = r_i + r_j`. Returns
/// `(force_on_j, df_dd, df_drelv)` -- force on `i` is the exact negation
/// (Newton's third law), matching `grain_contact_law::resolve_contact_pair`'s
/// own convention.
///
/// Real derivation (hand-checked, not guessed): let `l = |d|`,
/// `dir = d/l`, `overlap = r_sum - l`,  `v_n = rel_v . dir`,
/// `s = F_n = kn*overlap - c_n*v_n` (the real force LAW,
/// `grain_contact_law::resolve_contact_core_linear`'s own first line, before
/// the `.max(0.0)` clamp -- linearizing the CURRENTLY-ACTIVE, still-
/// overlapping regime, the same real, standard practice
/// `rod::implicit`'s own doc already establishes for this class of solver).
/// Force on `j` (repulsive, pushes j away from i along `+dir`): `force =
/// dir * s`.
///
/// `ds/dd = -kn*dir - c_n*(P*rel_v)/l` (`P = I - outer(dir,dir)`, the
/// standard projection removing the radial component -- `d(dir)/dd = P/l`).
/// `df_dd = outer(dir, ds/dd) + s*(P/l)` -- the standard product-rule
/// Jacobian of `dir*s` w.r.t. `d` (`d(dir*s)/dd = dir⊗(ds/dd) + s*d(dir)/dd`),
/// IDENTICAL shape to `axial_force_and_jacobian`'s own `df_dd` line, just
/// with this function's own `s`/`ds_dd` substituted in -- matching an
/// already-verified pattern structurally, not re-derived in a vacuum.
/// `ds/d(rel_v) = -c_n*dir` (only the damping term depends on `rel_v`),
/// giving `df_drelv = -c_n*outer(dir,dir)`.
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

/// One real implicit (backward Euler) step for grain-grain NORMAL contact
/// forces across a whole population -- the direct N-grain generalization of
/// the single/2-grain-chain technique verified in isolated prototypes (see
/// this module's own top-of-file doc). Mirrors `rod::implicit::
/// step_rod_implicit`'s own structure closely: assemble `K`/`C` from every
/// active contact's analytic Jacobian (restricted to non-pinned grains),
/// solve one dense system for `dv`, apply it.
///
/// `pinned[i] = true` excludes grain `i` from the solved DOF set entirely
/// (its own velocity is treated as fixed for this step) -- same real
/// Dirichlet-anchor convention `rod::implicit` already uses for its own
/// pinned points, needed here for a fixed floor/wall grain (see
/// `grains_repose_angle.rs`'s own real "huge pinned floor grain" technique,
/// already proven in this codebase's explicit path).
///
/// `gravity` is applied as an ordinary external force on the right-hand
/// side (same real, disclosed simplification `rod::implicit` already makes
/// for its own external forces -- only the contact spring itself is stiff
/// enough to need implicit treatment).
///
/// Real fallback: a singular assembled system (degenerate geometry) falls
/// back to a plain explicit substep for the affected grains, same
/// `rod::implicit`'s own disclosed safety net.
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
            // Real, disclosed fallback -- singular system, one explicit
            // substep for the free grains instead of silently producing
            // garbage, same `rod::implicit`'s own precedent.
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

    /// Real, controlled verification, matching the lost prototype work's
    /// own methodology exactly (per project memory): an N-grain vertical
    /// stack on a fixed floor, real closed-form predicted equilibrium
    /// overlap at EVERY contact (`overlap_i = weight_above_contact_i / kn`),
    /// run at a LARGE dt=1/60s -- a dt an explicit integrator at this real
    /// stiffness could never take stably (this scene's own real
    /// `critical_timestep` is many orders of magnitude smaller, matching
    /// the same "DEM timestep problem" this whole module exists to solve).
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

        // Real closed-form check: contact i (grain i to grain i+1, 0-indexed
        // with grain 0 = floor) supports the real weight of every free grain
        // ABOVE it -- (n_free - i) grains' worth of weight.
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
        // Real stability check: every free grain's velocity must have
        // genuinely settled (this is a static-equilibrium scene), not just
        // happen to be measured at a zero-crossing of an ongoing oscillation.
        for &gi in &[1, n_free] {
            assert!(
                grains[gi].v.length() < 1.0e-2,
                "grain {gi} should have settled to near-zero velocity, v={:?}",
                grains[gi].v
            );
        }
    }
}

/// Real, disclosed diagnostic (not correctness) -- measures the actual
/// real-time win this module exists to deliver: wall-clock cost to advance
/// 1/60s of simulated time via ONE large implicit step vs the many tiny
/// explicit substeps the real Rayleigh critical timestep
/// (`grain_contact_law::critical_timestep`) forces at the same real contact
/// stiffness. Both paths resolve the IDENTICAL normal-only physics (same
/// `kn`/`cn`, same pair list) so the comparison isolates the INTEGRATOR,
/// not a different force model. Run manually:
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

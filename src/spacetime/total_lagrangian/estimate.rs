//! A one-solve local error estimate of a converged step, MEASURED beside the
//! step controller's doubling and never used by it (the user's decision: the
//! estimate's effectivity is shown per scene before any step is chosen by it).
//!
//! Source: Lacoma & Romero, Rev. Int. Mét. Num. Cálc. Dis. Ing. 22(1) 45-62
//! (2006), "estimador tipo 2", their Table II, eqs. 29-32 (read in full): the
//! improved solution `phi* = phi_n + Q`, `Q` the 3-point Gauss-Lobatto
//! quadrature of `phi' = (V, A)` over the step, with `U*_{1/2} = U_n + dt/2
//! V_n + dt^2/8 A_n`, `V*_{1/2} = V_n + 3 dt/8 A_n + dt/8 A_{n+1}` and
//! `A*_{1/2} = M^-1 F(U*, V*)` (29); `Q = dt/6 phi'_n + 4 dt/6 phi'_* + dt/6
//! phi'_{n+1}` (30); the estimate `e = phi_n + Q - phi_{n+1}` (31).
//!
//! Adaptations, declared (the source is linear `M U'' + C U' + K U = F`):
//! 1. `F(U, V) = f_ext + f_contact - f_int(U, V)`, `f_int` the internal
//!    force of the POINT stress at `U` (`Elastic::point_stress`), not the
//!    step's algorithmic one, with the dashpot's stress at the rate of `V`.
//! 2. The end accelerations are not formed by the step: each is one force
//!    and one consistent-mass solve (`M_eps`, Love and Sulsky's eq. 26).
//! 3. The contact force is the converged step's push, held over the step
//!    (exact for an enduring contact whose reaction is constant over it).
//! 4. A plastic or relaxing solid's stress at a time inside the step uses
//!    its state advanced to that time by its own update
//!    (`Elastic::committed` over `[t_n, t]`): with the start state only, a
//!    flowing particle's stress is off by `O(dt)` and the estimate loses
//!    its order (derived: its plastic increment over half a step).
//!
//! Removal criterion for adaptations 1-4: the effectivity measured on the
//! gates' scenes (`ESTIMATOR_DESIGN.md` sec. 5), or a source for E6 on
//! nonlinear solids with contact.

use super::*;

/// The estimate as the particles see it: their position, deformation
/// gradient and velocity errors (`e_d` interpolated, its gradient, `e_v`
/// interpolated), the same quantities the controller's norm reads; and,
/// to count the contact points' velocity alternation the quadrature of
/// `V` cannot follow, the largest particle speeds of the step's start
/// velocity, its mean `u / dt` and its end velocity.
pub(crate) type ParticleError = (Vec<DVec2>, Vec<DMat2>, Vec<DVec2>, [f64; 3]);

/// What the estimate reads from a converged step.
pub(crate) struct Converged<'a> {
    pub(crate) grid: &'a Grid,
    pub(crate) mass: &'a [f64],
    pub(crate) v0: &'a [f64],
    pub(crate) solids: &'a [Elastic],
    pub(crate) f_start: &'a [DMat2],
    pub(crate) lagrangian: bool,
    pub(crate) dt: f64,
    pub(crate) eps: f64,
    pub(crate) mass_factor: &'a Band,
    /// The weight, and the contact push at the solution (adaptation 3).
    pub(crate) applied: Vec<f64>,
    /// The step's grid displacement and start velocity.
    pub(crate) u: &'a [f64],
    pub(crate) v_n: &'a [f64],
}

/// Lacoma and Romero's estimate (module doc) of the converged step.
pub(crate) fn lobatto_error(s: &Converged<'_>) -> ParticleError {
    let (grid, dt) = (s.grid, s.dt);
    let n = s.f_start.len();
    let particles = s.v0.len().min(n);
    // A grid field's material gradient at a particle: on the reference
    // basis (`lagrangian`) the grid's own, else the step start's spatial
    // gradient carried back by `F_n` (as `solve`'s `deformation_at`).
    let gradient_of = |y: &[f64], p: usize| -> DMat2 {
        let (_, gu) = grid.read(y, p, 0);
        let (_, gv) = grid.read(y, p, 1);
        let grad = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
        if s.lagrangian {
            grad
        } else {
            grad * s.f_start[p]
        }
    };
    // `M_eps^-1 F(U, V)` with each solid's state advanced over `[t_n, t_n +
    // elapsed]` (adaptation 4).
    let massive = grid.massive();
    let acceleration = |w: &[f64], velocity: &[f64], elapsed: f64| -> Vec<f64> {
        let internal = parallel_scatter(grid.len, particles, |p, acc| {
            if s.v0[p] <= 0.0 {
                return;
            }
            let (f0, f) = (s.f_start[p], s.f_start[p] + gradient_of(w, p));
            let (c0, c) = (f0.transpose() * f0, f.transpose() * f);
            let solid = if elapsed > 0.0 && s.solids[p].carries_state() {
                s.solids[p].committed(c0, c, elapsed).solid
            } else {
                s.solids[p]
            };
            let f_dot = gradient_of(velocity, p);
            let e_dot = 0.5 * (f_dot.transpose() * f + f.transpose() * f_dot);
            let stress = solid.point_stress(c, e_dot);
            let first = if s.lagrangian {
                s.v0[p] * f * stress
            } else {
                s.v0[p] * f * stress * f0.transpose()
            };
            for a in 0..2 {
                let row = first.row(a);
                grid.visit(p, a, |k, _, g| acc[k] += row.dot(g));
            }
        });
        let rhs: Vec<f64> = (0..grid.len)
            .map(|k| {
                if massive[k] && grid.active[k] {
                    s.applied[k] - internal[k]
                } else {
                    0.0
                }
            })
            .collect();
        preconditioned_cg(
            &|y: &[f64]| grid.mass_times(y, s.mass, s.eps),
            &rhs,
            &|r: &[f64]| s.mass_factor.solve(r),
            &massive,
        )
    };
    let zero = vec![0.0; grid.len];
    let v_end: Vec<f64> = (0..grid.len)
        .map(|k| 2.0 * s.u[k] / dt - s.v_n[k])
        .collect();
    let a_n = acceleration(&zero, s.v_n, 0.0);
    let a_end = acceleration(s.u, &v_end, dt);
    // Eq. 29.
    let u_mid: Vec<f64> = (0..grid.len)
        .map(|k| 0.5 * dt * s.v_n[k] + dt * dt / 8.0 * a_n[k])
        .collect();
    let v_mid: Vec<f64> = (0..grid.len)
        .map(|k| s.v_n[k] + 3.0 * dt / 8.0 * a_n[k] + dt / 8.0 * a_end[k])
        .collect();
    let a_mid = acceleration(&u_mid, &v_mid, 0.5 * dt);
    // Eqs. 30-31: `e_d = Q_U - U_{n+1}` (`U_n = 0`), `e_v = V_n + Q_V -
    // V_{n+1}`.
    let e_d: Vec<f64> = (0..grid.len)
        .map(|k| dt / 6.0 * (s.v_n[k] + 4.0 * v_mid[k] + v_end[k]) - s.u[k])
        .collect();
    let e_v: Vec<f64> = (0..grid.len)
        .map(|k| s.v_n[k] + dt / 6.0 * (a_n[k] + 4.0 * a_mid[k] + a_end[k]) - v_end[k])
        .collect();
    let at = |y: &[f64], p: usize| DVec2::new(grid.read(y, p, 0).0, grid.read(y, p, 1).0);
    let mean: Vec<f64> = s.u.iter().map(|e| e / dt).collect();
    let fastest = |y: &[f64]| {
        (0..particles)
            .map(|p| at(y, p).length())
            .fold(0.0, f64::max)
    };
    (
        (0..n).map(|p| at(&e_d, p)).collect(),
        (0..n)
            .map(|p| {
                if p < particles {
                    gradient_of(&e_d, p)
                } else {
                    DMat2::ZERO
                }
            })
            .collect(),
        (0..n).map(|p| at(&e_v, p)).collect(),
        [fastest(s.v_n), fastest(&mean), fastest(&v_end)],
    )
}

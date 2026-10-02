//! Discrete elastic rod internal forces -- stretch (axial spring) + bending
//! (discrete curvature) + damping, specialized to 2D.
//!
//! Bergou, Wardetzky, Robinson, Audoly & Grinspun 2008, SIGGRAPH, "Discrete
//! Elastic Rods", eq. 1 (curvature binormal) and eq. 4-5 (bending energy).
//! See `mod.rs` for why the 3D binormal is a signed scalar in 2D.

use glam::{Mat2, Vec2};

use super::RodMaterial;

/// Discrete curvature at an interior vertex (Bergou et al. 2008, eq. 1, in
/// 2D). In 3D `kb` is a vector along the out-of-plane binormal with magnitude
/// `2*tan(turning_angle/2)`; in 2D the binormal is the plane's normal, so it
/// is this signed scalar. Dimensionless (about the turning angle for small
/// bends): the per-length normalization is in the bending force (division by
/// the rest Voronoi length).
///
/// Limitation of this closed form (Bergou 2008 §4.2): the denominator vanishes
/// as `e0`/`e1` become antiparallel (a very sharp bend). Fine for moderate
/// bending (a grass blade), not for arbitrary large deformation.
pub fn discrete_curvature(p0: Vec2, p1: Vec2, p2: Vec2) -> f32 {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let cross = e0.x * e1.y - e0.y * e1.x;
    let dot = e0.dot(e1);
    let chi = (e0.length() * e1.length() + dot).max(1.0e-9);
    2.0 * cross / chi
}

/// Analytic gradient of `discrete_curvature` w.r.t. its 3 input points,
/// hand-derived via the chain rule on that function's own closed form.
/// Verified against central differences in this module's own tests -- same
/// house discipline as `grid::kernel::axis_weights_derivative` and every
/// `*_vjp` function in `spacetime::transfer`: derive by hand, ship a
/// finite-difference check in the same file.
pub fn discrete_curvature_gradient(p0: Vec2, p1: Vec2, p2: Vec2) -> [Vec2; 3] {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let l0 = e0.length().max(1.0e-9);
    let l1 = e1.length().max(1.0e-9);
    let cross = e0.x * e1.y - e0.y * e1.x;
    let dot = e0.dot(e1);
    let chi = (l0 * l1 + dot).max(1.0e-9);
    let kappa = 2.0 * cross / chi;

    // d(cross)/d(p_k), where cross = e0.x*e1.y - e0.y*e1.x, e0 = p1-p0, e1 = p2-p1.
    let d_cross_p0 = Vec2::new(-e1.y, e1.x);
    let d_cross_p1 = Vec2::new(e1.y + e0.y, -e1.x - e0.x);
    let d_cross_p2 = Vec2::new(-e0.y, e0.x);

    // d(dot)/d(p_k), where dot = e0.e1.
    let d_dot_p0 = -e1;
    let d_dot_p1 = e1 - e0;
    let d_dot_p2 = e0;

    // d(chi)/d(p_k) = d(l0*l1)/d(p_k) + d(dot)/d(p_k), and d(l0)/d(p0) = -e0/l0, etc.
    let d_chi_p0 = (l1 / l0) * (-e0) + d_dot_p0;
    let d_chi_p1 = (l1 / l0) * e0 + (l0 / l1) * (-e1) + d_dot_p1;
    let d_chi_p2 = (l0 / l1) * e1 + d_dot_p2;

    let grad = |d_cross: Vec2, d_chi: Vec2| (2.0 / chi) * d_cross - (kappa / chi) * d_chi;
    [
        grad(d_cross_p0, d_chi_p0),
        grad(d_cross_p1, d_chi_p1),
        grad(d_cross_p2, d_chi_p2),
    ]
}

/// Per-point internal force (Newtons, real SI) from axial stretch, bending,
/// and damping. `x`/`v` are grid-cell units; `dx_meters` converts to/from
/// real meters for the stiffness terms, then back to a grid acceleration --
/// mirrors `gravity_to_grid`'s own `g_grid = g_SI / dx_meters` pattern (mass
/// handled explicitly here since force, unlike gravity, is not already
/// per-unit-mass).
///
/// `ea`/`ei` are PER-ELEMENT (length N-1/N-2, same shape as
/// `rest_edge_length`/`rest_curvature`) rather than the single scalar
/// `RodMaterial::ea`/`ei` -- real prior art `network::NetworkEdge::ea`/
/// `NetworkBendingVertex::ei` already does this for a branching
/// `RodNetwork`; this is the same non-uniform-stiffness capability for a
/// plain chain (a stem stiffer at its base than its growing tip). An EMPTY
/// slice falls back to `material.ea`/`material.ei` uniformly (the prior
/// single-scalar behavior, bit-for-bit) -- `Rod::new` normally fills these
/// to full length, but this fallback also covers any `RodPoints` built
/// directly (bypassing `Rod::new`, e.g. some existing tests) without
/// panicking or requiring every such call site to remember to pre-fill.
/// Damping stays scalar (`material.axial_damping`/`bending_damping`) -- out
/// of this phase's scope, not yet made per-element.
/// Bundles a rod's per-element rest/stiffness state -- the 4 parallel
/// arrays (same length convention as `rest_edge_length`, i.e. N-1/N-2 of
/// the point count) that always travel together, one per `RodPoints`. Real
/// fix for clippy::too_many_arguments on `compute_internal_forces` (was 4
/// loose slice params) rather than suppressing the lint.
pub struct RodRestState<'a> {
    pub rest_edge_length: &'a [f32],
    pub rest_curvature: &'a [f32],
    pub ea: &'a [f32],
    pub ei: &'a [f32],
}

pub fn compute_internal_forces(
    x: &[Vec2],
    v: &[Vec2],
    rest: RodRestState,
    material: &RodMaterial,
    dx_meters: f32,
) -> Vec<Vec2> {
    let RodRestState {
        rest_edge_length,
        rest_curvature,
        ea,
        ei,
    } = rest;
    let n = x.len();
    let mut force = vec![Vec2::ZERO; n];
    if n < 2 {
        return force;
    }
    let ea_at = |i: usize| if ea.is_empty() { material.ea } else { ea[i] };
    let ei_at = |i: usize| if ei.is_empty() { material.ei } else { ei[i] };

    // ── Axial stretch (Hookean spring along each edge) + Kelvin-Voigt axial damping ──
    for i in 0..n - 1 {
        let edge = (x[i + 1] - x[i]) * dx_meters;
        let l = edge.length().max(1.0e-9);
        let l0 = rest_edge_length[i].max(1.0e-9);
        let dir = edge / l;

        let f_stretch = ea_at(i) * (l - l0) / l0;

        // Strain-rate damping: relative velocity projected onto the edge direction.
        let rel_v = (v[i + 1] - v[i]) * dx_meters;
        let strain_rate = rel_v.dot(dir);
        let f_damp = material.axial_damping * strain_rate;

        let f = (f_stretch + f_damp) * dir;
        force[i] += f;
        force[i + 1] -= f;
    }

    // ── Bending (discrete curvature) + Rayleigh bending damping ──
    if n >= 3 {
        for i in 1..n - 1 {
            let (p0, p1, p2) = (x[i - 1] * dx_meters, x[i] * dx_meters, x[i + 1] * dx_meters);
            let kappa = discrete_curvature(p0, p1, p2);
            let grad = discrete_curvature_gradient(p0, p1, p2);

            let l0_prev = rest_edge_length[i - 1].max(1.0e-9);
            let l0_next = rest_edge_length[i].max(1.0e-9);
            let voronoi_length = 0.5 * (l0_prev + l0_next);

            let kappa_rest = rest_curvature[i - 1];
            let coeff = (ei_at(i - 1) / voronoi_length) * (kappa - kappa_rest);

            // kappa_dot = sum_k grad_k . v_k -- generalized-force construction
            // (Rayleigh 1873). `bending_damping` is declared N*m*s so that
            // `bending_damping * kappa_dot[1/s]` already has units N*m,
            // matching `coeff` [EI/l, N*m] directly -- do not divide by
            // voronoi_length again, that breaks dimensional consistency.
            let kappa_dot = grad[0].dot(v[i - 1] * dx_meters)
                + grad[1].dot(v[i] * dx_meters)
                + grad[2].dot(v[i + 1] * dx_meters);
            let damp_coeff = material.bending_damping * kappa_dot;

            let total_coeff = coeff + damp_coeff;
            force[i - 1] -= total_coeff * grad[0];
            force[i] -= total_coeff * grad[1];
            force[i + 1] -= total_coeff * grad[2];
        }
    }

    force
}

/// Per-point internal force from the BENDING term only (no axial stretch)
/// -- identical math to `compute_internal_forces`'s own "Bending" section,
/// intentionally duplicated rather than refactored out of that
/// already-tested, widely-used function. Real use: `step_rod_implicit`'s
/// finite-difference Jacobian sweep, once the axial term has its own
/// analytic Jacobian (`axial_force_and_jacobian` below) and no longer
/// needs perturbing -- calling this instead of the full function avoids
/// redoing axial work the FD sweep no longer needs (profiled real cost,
/// see `implicit_solver_cost_profile`).
pub fn compute_bending_forces_only(
    x: &[Vec2],
    v: &[Vec2],
    rest_edge_length: &[f32],
    rest_curvature: &[f32],
    ei: &[f32],
    material: &RodMaterial,
    dx_meters: f32,
) -> Vec<Vec2> {
    let n = x.len();
    let mut force = vec![Vec2::ZERO; n];
    if n < 3 {
        return force;
    }
    let ei_at = |i: usize| if ei.is_empty() { material.ei } else { ei[i] };

    for i in 1..n - 1 {
        let (p0, p1, p2) = (x[i - 1] * dx_meters, x[i] * dx_meters, x[i + 1] * dx_meters);
        let kappa = discrete_curvature(p0, p1, p2);
        let grad = discrete_curvature_gradient(p0, p1, p2);

        let l0_prev = rest_edge_length[i - 1].max(1.0e-9);
        let l0_next = rest_edge_length[i].max(1.0e-9);
        let voronoi_length = 0.5 * (l0_prev + l0_next);

        let kappa_rest = rest_curvature[i - 1];
        let coeff = (ei_at(i - 1) / voronoi_length) * (kappa - kappa_rest);

        let kappa_dot = grad[0].dot(v[i - 1] * dx_meters)
            + grad[1].dot(v[i] * dx_meters)
            + grad[2].dot(v[i + 1] * dx_meters);
        let damp_coeff = material.bending_damping * kappa_dot;

        let total_coeff = coeff + damp_coeff;
        force[i - 1] -= total_coeff * grad[0];
        force[i] -= total_coeff * grad[1];
        force[i + 1] -= total_coeff * grad[2];
    }

    force
}

/// Analytic axial force and its Jacobians, the standard damped-spring
/// Jacobian of cloth and mass-spring simulation (Provot 1995; Baraff & Witkin
/// 1998, this implicit scheme's basis). Returns `(force, dF/dd, dF/d(rel_v))`
/// for the edge vector in meters `d = (x[i+1]-x[i])*dx_meters` and `rel_v =
/// (v[i+1]-v[i])*dx_meters`, not yet chained to the four endpoint variables
/// (the caller does that; sign and `dx_meters` differ per endpoint).
///
/// Derivation: let `l=|d|`, `dir=d/l`, `s = f_stretch + f_damp` (the
/// scalar force magnitude along `dir`), `F = s*dir`. Using the standard
/// unit-vector-gradient identity `d(dir)/dd = (I - dir⊗dir)/l = P/l`:
/// ```text
/// ds/dd      = (ea/l0)*dir + (axial_damping/l)*(P*rel_v)
/// dF/dd      = dir⊗(ds/dd) + (s/l)*P
/// dF/d(rel_v) = axial_damping*(dir⊗dir)
/// ```
/// Checked against central differences of the force law
/// `compute_internal_forces` uses, in the tests below.
pub fn axial_force_and_jacobian(
    d: Vec2,
    rel_v: Vec2,
    l0: f32,
    ea: f32,
    axial_damping: f32,
) -> (Vec2, Mat2, Mat2) {
    let l = d.length().max(1.0e-9);
    let l0 = l0.max(1.0e-9);
    let dir = d / l;

    let f_stretch = ea * (l - l0) / l0;
    let strain_rate = rel_v.dot(dir);
    let f_damp = axial_damping * strain_rate;
    let s = f_stretch + f_damp;
    let force = dir * s;

    let outer_dir = Mat2::from_cols(dir * dir.x, dir * dir.y);
    let p = Mat2::IDENTITY - outer_dir;

    let ds_dd = dir * (ea / l0) + (p * rel_v) * (axial_damping / l);
    let df_dd = Mat2::from_cols(dir * ds_dd.x, dir * ds_dd.y) + p * (s / l);
    let df_drelv = outer_dir * axial_damping;

    (force, df_dd, df_drelv)
}

/// Analytic Jacobian of the bending force at one interior vertex, with
/// respect to the 3 points `discrete_curvature_gradient` takes (`grad` is its
/// output). Returns `(dF/dx, dF/dv)`, each a 3x3 grid of 2x2 blocks (`[a][b]`
/// = force at point `a` against point `b`).
///
/// `dF/dv` is exact: `F[a] = -total_coeff*grad[a]`, and the damping rate
/// `kappa_dot = sum_m grad[m].v[m]` is linear in `v` with `grad` (independent
/// of `v`) as coefficient, so `d(kappa_dot)/dv[b] = grad[b]`.
///
/// `dF/dx` is the Gauss-Newton approximation: the full derivative needs
/// `d(grad[a])/dx[b]`, the Hessian of `discrete_curvature`, not derived for
/// this 2D reduction. The kept term, `-(EI/l_v)*outer(grad[a],grad[b])`, is
/// the exact derivative of the part linear in `kappa` (the elastic
/// restoring term); the geometric-stiffness term and the damping cross-term
/// are dropped, a named omission (standard for energy-based force Jacobians,
/// e.g. Projective Dynamics) that grows with bending away from rest. Each
/// block is `coeff * outer(g,g)`, positive semi-definite, which helps the
/// implicit solve.
///
/// The tests check both against central differences: `dF/dv` tightly,
/// `dF/dx` near rest, with the gap growing for large bending, as expected.
pub fn bending_jacobian_gauss_newton(
    grad: [Vec2; 3],
    stiffness: f32,
    damping: f32,
) -> ([[Mat2; 3]; 3], [[Mat2; 3]; 3]) {
    let outer = |a: Vec2, b: Vec2| Mat2::from_cols(a * b.x, a * b.y);
    let mut k = [[Mat2::ZERO; 3]; 3];
    let mut c = [[Mat2::ZERO; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let o = outer(grad[a], grad[b]);
            k[a][b] = -stiffness * o;
            c[a][b] = -damping * o;
        }
    }
    (k, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_curvature_gradient_matches_finite_difference() {
        let h = 1.0e-4_f32;
        let cases = [
            (
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(2.0, 0.3),
            ),
            (
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.2),
                Vec2::new(1.8, 0.9),
            ),
            (
                Vec2::new(-0.5, 0.1),
                Vec2::new(0.6, -0.2),
                Vec2::new(1.7, 0.4),
            ),
        ];
        for (p0, p1, p2) in cases {
            let analytic = discrete_curvature_gradient(p0, p1, p2);
            let points = [p0, p1, p2];
            for k in 0..3 {
                for axis in 0..2 {
                    let mut plus = points;
                    let mut minus = points;
                    if axis == 0 {
                        plus[k].x += h;
                        minus[k].x -= h;
                    } else {
                        plus[k].y += h;
                        minus[k].y -= h;
                    }
                    let f_plus = discrete_curvature(plus[0], plus[1], plus[2]);
                    let f_minus = discrete_curvature(minus[0], minus[1], minus[2]);
                    let numeric = (f_plus - f_minus) / (2.0 * h);
                    let component = if axis == 0 {
                        analytic[k].x
                    } else {
                        analytic[k].y
                    };
                    let diff = (numeric - component).abs();
                    assert!(
                        diff < 1.0e-2,
                        "curvature gradient mismatch at points={points:?}, point {k}, axis {axis}: \
                         analytic={component:.6} numeric={numeric:.6} diff={diff:.2e}"
                    );
                }
            }
        }
    }

    #[test]
    fn straight_rod_has_zero_curvature() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(1.0, 0.0);
        let p2 = Vec2::new(2.0, 0.0);
        let kappa = discrete_curvature(p0, p1, p2);
        assert!(
            kappa.abs() < 1.0e-6,
            "straight rod should have zero curvature, got {kappa}"
        );
    }

    #[test]
    fn bent_rod_has_nonzero_curvature() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(1.0, 0.0);
        let p2 = Vec2::new(2.0, 0.5);
        let kappa = discrete_curvature(p0, p1, p2);
        assert!(
            kappa.abs() > 1.0e-3,
            "bent rod should have nonzero curvature, got {kappa}"
        );
    }

    #[test]
    fn zero_force_on_straight_rod_at_rest_length() {
        let material = RodMaterial::new(1000.0, 10.0, 0.0, 0.0);
        let x = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
        ];
        let v = vec![Vec2::ZERO; 3];
        let rest_edge_length = vec![1.0, 1.0];
        let rest_curvature = vec![0.0];
        let ea = vec![material.ea; 2];
        let ei = vec![material.ei; 1];
        let force = compute_internal_forces(
            &x,
            &v,
            RodRestState {
                rest_edge_length: &rest_edge_length,
                rest_curvature: &rest_curvature,
                ea: &ea,
                ei: &ei,
            },
            &material,
            1.0,
        );
        for (i, f) in force.iter().enumerate() {
            assert!(
                f.length() < 1.0e-4,
                "straight rod at rest length should have zero internal force, point {i}: {f:?}"
            );
        }
    }

    /// The force law under test, standalone (as `compute_internal_forces`'s
    /// axial section), so the central-difference check does not share code
    /// with what it checks.
    fn axial_force_reference(d: Vec2, rel_v: Vec2, l0: f32, ea: f32, axial_damping: f32) -> Vec2 {
        let l = d.length().max(1.0e-9);
        let dir = d / l;
        let f_stretch = ea * (l - l0) / l0;
        let strain_rate = rel_v.dot(dir);
        let f_damp = axial_damping * strain_rate;
        (f_stretch + f_damp) * dir
    }

    #[test]
    fn axial_force_and_jacobian_matches_finite_difference() {
        // Central-difference step for f32 h ~ cbrt(EPSILON), balancing
        // truncation (O(h^2)) against cancellation (O(EPSILON/h)).
        // `discrete_curvature_gradient`'s 1e-4 does not fit here: with `ea` of
        // 1000-2000 the cancellation error divided by 2h came out ~17%.
        let h = f32::EPSILON.cbrt();
        let cases = [
            (Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0), 1.0, 1000.0, 5.0),
            (Vec2::new(1.2, 0.3), Vec2::new(0.1, -0.05), 1.0, 500.0, 2.0),
            (
                Vec2::new(0.8, -0.2),
                Vec2::new(-0.2, 0.15),
                1.0,
                2000.0,
                10.0,
            ),
        ];
        // Relative tolerance (with a small absolute floor for near-zero
        // components) -- appropriate given `ea` spans 500-2000 across
        // cases, so a single fixed absolute tolerance would either be too
        // loose for the small case or too tight for the large one. Floor
        // of 3.0 (not 1.0) is itself a measured choice: perturbing
        // PERPENDICULAR to `dir` makes `l` an even function of `h` (length
        // barely changes to first order), so `f_stretch` there is O(h^2)
        // and the true analytic derivative is exactly 0 -- central
        // difference of that odd-in-h, cubic-leading-term
        // component has O(ea*h^2) truncation error (confirmed: measured
        // 0.012 at ea=1000, matching `(ea/2)*h^2` by hand), not a formula
        // bug. Still 2+ orders of magnitude tighter than the real
        // force/Jacobian magnitudes here (100s-1000s).
        let close_enough = |numeric: Vec2, analytic: Vec2| -> bool {
            let diff = (numeric - analytic).length();
            let scale = analytic.length().max(1.0);
            diff < 3.0e-2 * scale
        };
        for (d, rel_v, l0, ea, axial_damping) in cases {
            let (_, df_dd, df_drelv) = axial_force_and_jacobian(d, rel_v, l0, ea, axial_damping);

            for axis in 0..2 {
                let mut d_plus = d;
                let mut d_minus = d;
                if axis == 0 {
                    d_plus.x += h;
                    d_minus.x -= h;
                } else {
                    d_plus.y += h;
                    d_minus.y -= h;
                }
                let f_plus = axial_force_reference(d_plus, rel_v, l0, ea, axial_damping);
                let f_minus = axial_force_reference(d_minus, rel_v, l0, ea, axial_damping);
                let numeric = (f_plus - f_minus) / (2.0 * h);
                let analytic = if axis == 0 {
                    df_dd.x_axis
                } else {
                    df_dd.y_axis
                };
                assert!(
                    close_enough(numeric, analytic),
                    "dF/dd mismatch at d={d:?} axis={axis}: analytic={analytic:?} numeric={numeric:?}"
                );
            }

            for axis in 0..2 {
                let mut v_plus = rel_v;
                let mut v_minus = rel_v;
                if axis == 0 {
                    v_plus.x += h;
                    v_minus.x -= h;
                } else {
                    v_plus.y += h;
                    v_minus.y -= h;
                }
                let f_plus = axial_force_reference(d, v_plus, l0, ea, axial_damping);
                let f_minus = axial_force_reference(d, v_minus, l0, ea, axial_damping);
                let numeric = (f_plus - f_minus) / (2.0 * h);
                let analytic = if axis == 0 {
                    df_drelv.x_axis
                } else {
                    df_drelv.y_axis
                };
                assert!(
                    close_enough(numeric, analytic),
                    "dF/d(rel_v) mismatch at d={d:?} axis={axis}: analytic={analytic:?} numeric={numeric:?}"
                );
            }
        }
    }

    /// The force law under test, standalone (as `compute_bending_forces_only`
    /// per vertex, in whatever units `p`/`v` come in; no `dx_meters`, since
    /// `bending_jacobian_gauss_newton` is unit-agnostic), so the checks do not
    /// share code with what they check.
    fn bending_force_reference(
        p: [Vec2; 3],
        v: [Vec2; 3],
        stiffness: f32,
        kappa_rest: f32,
        damping: f32,
    ) -> [Vec2; 3] {
        let kappa = discrete_curvature(p[0], p[1], p[2]);
        let grad = discrete_curvature_gradient(p[0], p[1], p[2]);
        let kappa_dot = grad[0].dot(v[0]) + grad[1].dot(v[1]) + grad[2].dot(v[2]);
        let total_coeff = stiffness * (kappa - kappa_rest) + damping * kappa_dot;
        [
            -total_coeff * grad[0],
            -total_coeff * grad[1],
            -total_coeff * grad[2],
        ]
    }

    #[test]
    fn bending_jacobian_velocity_term_is_exact() {
        // dF/dv has NO dropped term (see the function's doc) -- this
        // should match finite differences as tightly as the axial Jacobian
        // does, not just "close enough for an approximation".
        let h = f32::EPSILON.cbrt();
        let p = [
            Vec2::new(-0.5, 0.1),
            Vec2::new(0.6, -0.2),
            Vec2::new(1.7, 0.4),
        ];
        let v = [
            Vec2::new(0.2, -0.1),
            Vec2::new(-0.1, 0.3),
            Vec2::new(0.05, 0.2),
        ];
        let stiffness = 800.0;
        let damping = 3.0;
        let kappa_rest = 0.05;
        let grad = discrete_curvature_gradient(p[0], p[1], p[2]);
        let (_, c) = bending_jacobian_gauss_newton(grad, stiffness, damping);

        for b in 0..3 {
            for axis in 0..2 {
                let mut v_plus = v;
                let mut v_minus = v;
                if axis == 0 {
                    v_plus[b].x += h;
                    v_minus[b].x -= h;
                } else {
                    v_plus[b].y += h;
                    v_minus[b].y -= h;
                }
                let f_plus = bending_force_reference(p, v_plus, stiffness, kappa_rest, damping);
                let f_minus = bending_force_reference(p, v_minus, stiffness, kappa_rest, damping);
                for (a, c_row) in c.iter().enumerate() {
                    let numeric = (f_plus[a] - f_minus[a]) / (2.0 * h);
                    let analytic = if axis == 0 {
                        c_row[b].x_axis
                    } else {
                        c_row[b].y_axis
                    };
                    let diff = (numeric - analytic).length();
                    let scale = analytic.length().max(1.0);
                    assert!(
                        diff < 1.0e-2 * scale,
                        "dF/dv mismatch a={a} b={b} axis={axis}: \
                         analytic={analytic:?} numeric={numeric:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn bending_jacobian_position_term_matches_near_rest_and_diverges_far_from_it() {
        // dF/dx drops the true Hessian's geometric-stiffness term (see the
        // function's doc) -- this is the honest test of that
        // disclosed approximation: close near rest (small `total_coeff`,
        // where the dropped term is small), and a real, EXPECTED
        // widening gap far from rest, not silently passed with a loose
        // tolerance that could just as easily be hiding a wrong formula.
        let stiffness = 800.0;
        let damping = 0.0; // isolate the position term from the (also-dropped) damping cross-term
        let h = f32::EPSILON.cbrt();
        let p = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.05), // near-straight: small real bend
        ];
        let v = [Vec2::ZERO; 3];
        let kappa = discrete_curvature(p[0], p[1], p[2]);
        let grad = discrete_curvature_gradient(p[0], p[1], p[2]);
        let (k, _) = bending_jacobian_gauss_newton(grad, stiffness, damping);

        let max_diff = |kappa_rest: f32| -> f32 {
            let mut max_diff = 0.0f32;
            for b in 0..3 {
                for axis in 0..2 {
                    let mut p_plus = p;
                    let mut p_minus = p;
                    if axis == 0 {
                        p_plus[b].x += h;
                        p_minus[b].x -= h;
                    } else {
                        p_plus[b].y += h;
                        p_minus[b].y -= h;
                    }
                    let f_plus = bending_force_reference(p_plus, v, stiffness, kappa_rest, damping);
                    let f_minus =
                        bending_force_reference(p_minus, v, stiffness, kappa_rest, damping);
                    for (a, k_row) in k.iter().enumerate() {
                        let numeric = (f_plus[a] - f_minus[a]) / (2.0 * h);
                        let analytic = if axis == 0 {
                            k_row[b].x_axis
                        } else {
                            k_row[b].y_axis
                        };
                        max_diff = max_diff.max((numeric - analytic).length());
                    }
                }
            }
            max_diff
        };

        // Near rest: kappa_rest almost equal to the kappa -> small total_coeff.
        let diff_near = max_diff(kappa - 1.0e-3);
        assert!(
            diff_near < 1.0,
            "near-rest Gauss-Newton dF/dx should closely match FD (dropped term genuinely \
             small there), got max diff {diff_near}"
        );

        // Far from rest: large synthetic deviation -> large total_coeff -> the dropped
        // geometric-stiffness term should now matter, a expected gap, not a bug.
        let diff_far = max_diff(kappa - 5.0);
        assert!(
            diff_far > diff_near,
            "far-from-rest gap ({diff_far}) should exceed the near-rest gap ({diff_near}) -- \
             the real, expected signature of the dropped Hessian term growing with the bending \
             moment, not noise"
        );
    }

    #[test]
    fn compute_bending_forces_only_matches_full_function_minus_axial() {
        // With EA = 0 the axial term is zero, so the bending-only force equals
        // the full function's output: the extracted function is the same
        // formula.
        let x = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.3),
            Vec2::new(3.0, 0.8),
        ];
        let v = vec![Vec2::ZERO; 4];
        let rest_edge_length = vec![1.0, 1.0, 1.0];
        let rest_curvature = vec![0.0, 0.0];
        let material = RodMaterial::new(0.0, 10.0, 0.0, 0.0);
        let ea = vec![0.0; 3];
        let ei = vec![10.0; 2];

        let full = compute_internal_forces(
            &x,
            &v,
            RodRestState {
                rest_edge_length: &rest_edge_length,
                rest_curvature: &rest_curvature,
                ea: &ea,
                ei: &ei,
            },
            &material,
            1.0,
        );
        let bending_only = compute_bending_forces_only(
            &x,
            &v,
            &rest_edge_length,
            &rest_curvature,
            &ei,
            &material,
            1.0,
        );
        for (i, (a, b)) in full.iter().zip(bending_only.iter()).enumerate() {
            assert!(
                (*a - *b).length() < 1.0e-6,
                "point {i}: full={a:?} bending_only={b:?}"
            );
        }
    }
}

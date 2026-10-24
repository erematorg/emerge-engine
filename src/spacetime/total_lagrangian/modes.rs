//! A body's linear modes on the step's own grid, for its flight between
//! contacts: the exact solution of the linearised elastic problem,
//! `M q'' + K q = 0`, mode by mode.
//!
//! The modes are those of the grid the step keeps on the reference
//! configuration (`contact_grid`), its consistent mass and the corotated
//! law's tangent at rest: `K phi = omega^2 M phi`, `phi^T M phi = 1`, the
//! mass-orthonormal basis `u = U q`, `U^T M U = I`, of Barbic's thesis
//! (CMU-CS-07-145, 2007, sec. 2.6). The grid stays on the reference, so a
//! displacement or velocity the step makes lies in the basis and its modal
//! coordinates are exact projections.
//!
//! Solved by two-sided Jacobi (Demmel and Veselic, LAPACK Working Note 15,
//! Algorithm 3.1) on `M`, then on `M^-1/2 (K + sigma M) M^-1/2`: the shift
//! `sigma` makes the second matrix definite, the case the algorithm
//! states, and is removed from the eigenvalues after. Dense: `n^3` per
//! sweep, 0.1 s for a 108-unknown block, measured; a large body needs a
//! sparse method (not built).
//!
//! A body whose particles all carry one proportional relaxation spectrum
//! (`Relaxation::proportional`) keeps these modes: in the linear limit
//! mechanism `l`'s force obeys `f_l' + theta_l f_l = kappa_l K u'`, through
//! the body's own stiffness, so the modes diagonalise it and each mode is
//! the generalised Zener oscillator `q'' = -omega^2 (q + sum_l s_l)`, `s_l'
//! + theta_l s_l = kappa_l q'`, with `s_il = phi_i^T f_l / omega_i^2`
//! (projected on the mass-orthonormal modes, `phi_i^T K u = omega_i^2 q_i`;
//! the mode's exact solution is `tests/common/zener.rs`'s).

use super::*;
use crate::matter::materials::solid::corotated::stress::corotated_second_pk_tangent;
pub(crate) use crate::tensor_algebra::dense::product;
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;

/// A free body's rigid modes in the plane: two translations and the
/// linearised rotation.
pub(crate) const RIGID_MODES: usize = 3;

/// A body's modes, ascending, the first `RIGID_MODES` its rigid motions.
#[derive(Clone, Debug)]
pub(crate) struct Modes {
    /// Angular frequencies, rad/s (the rigid ones set to zero).
    pub(crate) omega: Vec<f64>,
    /// Per mode, its value at every point of the step's state (particles,
    /// then contact points).
    pub(crate) at: Vec<Vec<DVec2>>,
    /// Per mode, its material gradient at every particle.
    pub(crate) gradients: Vec<Vec<DMat2>>,
    /// The body's relaxation mechanisms as `(theta, kappa)`, empty when it
    /// has none, and its record's strain limit (infinite without one).
    pub(crate) memory: Vec<(f64, f64)>,
    pub(crate) strain_limit: f64,
}

/// The material gradient `grad_X d` of the grid field `d` at point `p`, as
/// the step forms `F`'s increment.
fn gradient(grid: &Grid, d: &[f64], p: usize) -> DMat2 {
    let (_, gu) = grid.read(d, p, 0);
    let (_, gv) = grid.read(d, p, 1);
    DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
}

/// `K d`: the internal force's derivative at rest, `sum_p V0_p dP : grad
/// N`, with `dP = dS` there (`S = 0` at `F = I`) and `dS` the corotated
/// law's own tangent along `dC = dF + dF^T`.
fn stiffness_times(grid: &Grid, d: &[f64], (v0, solids): (&[f64], &[Elastic])) -> Vec<f64> {
    let mut out = vec![0.0; grid.len];
    for (p, (&volume, solid)) in v0.iter().zip(solids).enumerate() {
        if volume <= 0.0 {
            continue;
        }
        let g = gradient(grid, d, p);
        let ds = corotated_second_pk_tangent(
            DMat2::IDENTITY,
            g + g.transpose(),
            solid.bulk - solid.mu,
            solid.mu,
        );
        for a in 0..2 {
            let row = DVec2::new(ds.x_axis[a], ds.y_axis[a]);
            grid.visit(p, a, |k, _, grad| out[k] += volume * row.dot(grad));
        }
    }
    out
}

/// Mirrors the upper triangle's mean into both halves (round-off only).
pub(crate) fn symmetrise(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (a[i * n + j] + a[j * n + i]);
            (a[i * n + j], a[j * n + i]) = (mean, mean);
        }
    }
}

/// Demmel and Veselic's `tol`, "usually machine precision"; 64 eps here,
/// declared (the modes' orthonormality held to 5e-14 with 45 eps, measured).
pub(crate) const JACOBI_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// A body's dense mass and stiffness on the grid's active unknowns.
pub(crate) struct System {
    pub(crate) unknowns: Vec<usize>,
    pub(crate) n: usize,
    pub(crate) m: Vec<f64>,
    pub(crate) k: Vec<f64>,
}

/// The dense consistent mass and stiffness at rest of the body whose
/// `particles` points carry mass on `grid`, with the relaxed moduli of a
/// body that carries relaxation branches (its memory is then the
/// caller's, `damped_system`); `None` when a law is not the corotated
/// solid's or carries a dashpot. With `below_yield`, also the J2 law on
/// Hencky without Norton's flow and without branches: below its yield it
/// is Hencky's elasticity, whose tangent at rest is the corotated one with
/// the same `mu` and `lambda = bulk - mu` (plane strain), so the modes are
/// the same; the caller checks that the motion stays below yield
/// (`HenckyJ2::elastic_within`).
fn relaxed_system(
    grid: &Grid,
    (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
    (particles, below_yield): (usize, bool),
) -> Option<System> {
    let carried = |s: &Elastic| match s.law {
        ElasticLaw::Corotated => true,
        ElasticLaw::HenckyJ2(law, _) => below_yield && law.norton.is_none() && !s.is_relaxing(),
        _ => false,
    };
    if solids[..particles]
        .iter()
        .any(|s| !carried(s) || s.is_viscous())
    {
        return None;
    }
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let (mut m, mut k) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &column) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[column] = 1.0;
        let mass_column = grid.mass_times(&unit, mass, 0.0);
        let stiffness_column = stiffness_times(grid, &unit, (&v0[..particles], solids));
        for (i, &row) in unknowns.iter().enumerate() {
            m[i * n + j] = mass_column[row];
            k[i * n + j] = stiffness_column[row];
        }
    }
    symmetrise(&mut m, n);
    symmetrise(&mut k, n);
    Some(System { unknowns, n, m, k })
}

/// The generalised eigenpairs `k phi = omega^2 m phi`, `phi^T m phi = 1`,
/// of an `n` by `n` pair (`m` definite): `omega^2` ascending, the shift
/// removed, and the modes as the columns of a row-major `n` by `n` matrix;
/// also the shift, the scale of the eigenvalues' round-off. `None` when `m`
/// is not definite.
pub(crate) fn eigen_pairs(m: Vec<f64>, k: &[f64], n: usize) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let (d, q) = jacobi_eigen_symmetric_definite(m, n, JACOBI_TOLERANCE);
    if d.iter().any(|&e| e <= 0.0) {
        return None;
    }
    // `M^-1/2 = Q D^-1/2 Q^T`.
    let root: Vec<f64> = (0..n * n)
        .map(|ij| {
            (0..n)
                .map(|e| q[ij / n * n + e] * q[ij % n * n + e] / d[e].sqrt())
                .sum()
        })
        .collect();
    let mut reduced = product(&product(&root, k, n), &root, n);
    symmetrise(&mut reduced, n);
    let shift = (0..n).map(|i| reduced[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        reduced[i * n + i] += shift;
    }
    let (lambda, y) = jacobi_eigen_symmetric_definite(reduced, n, JACOBI_TOLERANCE);
    let phi = product(&root, &y, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| lambda[a].total_cmp(&lambda[b]));
    let squared = order.iter().map(|&e| lambda[e] - shift).collect();
    let sorted = (0..n * n)
        .map(|ij| phi[ij / n * n + order[ij % n]])
        .collect();
    Some((squared, sorted, shift))
}

/// A grid field's values at every point and its material gradients at the
/// particles, from its entries on `unknowns`.
pub(crate) fn field_at(
    grid: &Grid,
    (unknowns, values): (&[usize], &[f64]),
    (particles, points): (usize, usize),
) -> (Vec<DVec2>, Vec<DMat2>) {
    let mut field = vec![0.0; grid.len];
    for (&unknown, &value) in unknowns.iter().zip(values) {
        field[unknown] = value;
    }
    let at = (0..points)
        .map(|p| DVec2::new(grid.read(&field, p, 0).0, grid.read(&field, p, 1).0))
        .collect();
    let gradients = (0..particles).map(|p| gradient(grid, &field, p)).collect();
    (at, gradients)
}

/// A body's dense system with its relaxation (`damped_system`): the
/// system, the mechanisms as `(theta, kappa)` and the strain limit.
pub(crate) type DampedSystem = (System, Vec<(f64, f64)>, f64);

/// The dense system of a body with the relaxation its modes carry
/// (`common_memory`): the system with the relaxed moduli, the mechanisms as
/// `(theta, kappa)` (empty without relaxation) and the record's strain
/// limit; `None` when the modes do not carry the body's law.
pub(crate) fn damped_system(
    grid: &Grid,
    (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
    particles: usize,
) -> Option<DampedSystem> {
    let (memory, limit) = common_memory(&solids[..particles])?;
    Some((
        relaxed_system(grid, (mass, v0, solids), (particles, false))?,
        memory,
        limit,
    ))
}

/// The relaxation the modes carry for these particles: none (`Some` of an
/// empty list) when no particle relaxes; the mechanisms as `(theta, kappa)`
/// and the strain limit when every particle carries the same proportional
/// spectrum; `None` otherwise (the modes do not decouple).
fn common_memory(solids: &[Elastic]) -> Option<(Vec<(f64, f64)>, f64)> {
    if solids.iter().all(|s| !s.is_relaxing()) {
        return Some((Vec::new(), f64::INFINITY));
    }
    let (first, limit) = solids.first()?.relaxation.proportional()?;
    for s in solids {
        let (mechanisms, l) = s.relaxation.proportional()?;
        if mechanisms != first || l != limit {
            return None;
        }
    }
    Some((
        first.iter().map(|m| (m.theta, m.kappa_shear)).collect(),
        limit,
    ))
}

/// The modes of the body whose `particles` points carry mass on `grid`
/// (its contact points after them carry none). `None` when a law is not
/// the corotated solid's or J2's below yield (`relaxed_system`) or carries
/// a dashpot (another law's modes are not built), when its relaxation does
/// not decouple on them (`common_memory`), or when the mass is not
/// definite.
pub(crate) fn body_modes(
    grid: &Grid,
    (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
    (particles, points): (usize, usize),
) -> Option<Modes> {
    let (memory, strain_limit) = common_memory(&solids[..particles])?;
    let System { unknowns, n, m, k } = relaxed_system(grid, (mass, v0, solids), (particles, true))?;
    if n <= RIGID_MODES {
        return None;
    }
    let (squared, phi, _) = eigen_pairs(m, &k, n)?;
    let omega = squared
        .iter()
        .enumerate()
        .map(|(rank, &w2)| {
            if rank < RIGID_MODES {
                0.0
            } else {
                w2.max(0.0).sqrt()
            }
        })
        .collect();
    let (at, gradients) = (0..n)
        .map(|e| {
            let column: Vec<f64> = (0..n).map(|i| phi[i * n + e]).collect();
            field_at(grid, (&unknowns, &column), (particles, points))
        })
        .unzip();
    Some(Modes {
        omega,
        at,
        gradients,
        memory,
        strain_limit,
    })
}

/// A damped mode's propagator over `length`, row-major `(2 + N)` square:
/// `y(t + length) = P y(t)` for its state in energy variables, `y = (omega
/// q, q', omega s_l / sqrt(kappa_l))`, whose half square norm is the mode's
/// energy `1/2 q'^2 + 1/2 omega^2 q^2 + sum 1/2 (omega^2 / kappa_l) s_l^2`
/// (Blanc et al. 2016, Property 1, as in `tests/common/zener.rs`).
///
/// In these variables the mode's equations are `y' = A y` with `A = omega
/// W - D`: `W` skew-symmetric (`W_01 = 1`, `W_1(2+l) = -sqrt(kappa_l)`, and
/// their mirrors negated), `D = diag(0, 0, theta_l)` (derived on the page:
/// `(omega q)' = omega q'`, `q'' = -omega (omega q) - sum omega sqrt(kappa_l)
/// y_l`, `y_l' = omega sqrt(kappa_l) q' - theta_l y_l`). So `d|y|^2/dt =
/// -2 sum theta_l y_l^2 <= 0`: the exact `P` never lengthens `y`. `P` is
/// `e^(A length)` by `tensor_algebra::matrix_exponential::expm` (Higham
/// 2005, Algorithm 2.3).
pub(crate) fn damped_propagator(omega: f64, memory: &[(f64, f64)], length: f64) -> Vec<f64> {
    let n = 2 + memory.len();
    let mut a = vec![0.0; n * n];
    a[1] = omega * length;
    a[n] = -omega * length;
    for (l, &(theta, kappa)) in memory.iter().enumerate() {
        let coupling = omega * kappa.sqrt() * length;
        a[n + 2 + l] = -coupling;
        a[(2 + l) * n + 1] = coupling;
        a[(2 + l) * n + 2 + l] = -theta * length;
    }
    crate::tensor_algebra::matrix_exponential::expm(&a, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plasticine's mechanisms as `(theta, kappa)`.
    const PLASTICINE: [(f64, f64); 3] = [
        (654.876_730_621_607_9, 0.380_744_010_597_488_7),
        (3_865.458_262_006_148, 0.361_264_618_173_421_7),
        (25_438.050_835_431_713, 0.753_998_274_291_466_7),
    ];

    /// `y' = A y` integrated by the classical fourth-order Runge-Kutta
    /// method in `steps` steps: an independent check of the exponential.
    fn integrated(
        omega: f64,
        memory: &[(f64, f64)],
        length: f64,
        y: &[f64],
        steps: usize,
    ) -> Vec<f64> {
        let n = y.len();
        let derivative = |y: &[f64]| -> Vec<f64> {
            let mut d = vec![0.0; n];
            d[0] = omega * y[1];
            d[1] = -omega * y[0];
            for (l, &(theta, kappa)) in memory.iter().enumerate() {
                d[1] -= omega * kappa.sqrt() * y[2 + l];
                d[2 + l] = omega * kappa.sqrt() * y[1] - theta * y[2 + l];
            }
            d
        };
        let h = length / steps as f64;
        let mut y = y.to_vec();
        for _ in 0..steps {
            let at = |y: &[f64], d: &[f64], s: f64| -> Vec<f64> {
                y.iter().zip(d).map(|(a, b)| a + s * b).collect()
            };
            let k1 = derivative(&y);
            let k2 = derivative(&at(&y, &k1, 0.5 * h));
            let k3 = derivative(&at(&y, &k2, 0.5 * h));
            let k4 = derivative(&at(&y, &k3, h));
            for e in 0..n {
                y[e] += h / 6.0 * (k1[e] + 2.0 * k2[e] + 2.0 * k3[e] + k4[e]);
            }
        }
        y
    }

    /// The damped propagator against the integrated equations, at a low
    /// mode over a short frame and a high one over a real-time frame (where
    /// the squaring runs), and the undamped limit against the rotation.
    /// Criteria fixed before measuring: within 1e-10 of the start's size,
    /// and the state never lengthened.
    #[test]
    fn the_damped_propagator_is_the_modes_exponential() {
        for (omega, length, steps) in [(300.0, 2e-3, 20_000), (2.0e4, 1.0 / 60.0, 400_000)] {
            let y0 = [1.0, 0.3, 0.2, -0.1, 0.05];
            let p = damped_propagator(omega, &PLASTICINE, length);
            let y1: Vec<f64> = (0..5)
                .map(|r| (0..5).map(|c| p[r * 5 + c] * y0[c]).sum())
                .collect();
            let reference = integrated(omega, &PLASTICINE, length, &y0, steps);
            let size = y0.iter().map(|e| e * e).sum::<f64>().sqrt();
            let off = y1
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(off <= 1e-10 * size, "omega {omega}: off by {off:e}");
            assert!(y1.iter().map(|e| e * e).sum::<f64>() < size * size);
        }
        let (omega, length) = (300.0, 2e-3);
        let p = damped_propagator(omega, &[], length);
        let (c, s) = ((omega * length).cos(), (omega * length).sin());
        for (got, exact) in p.iter().zip([c, s, -s, c]) {
            assert!((got - exact).abs() < 1e-14, "{got} against {exact}");
        }
    }
}

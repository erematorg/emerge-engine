//! A semismooth Newton step on the contacts' multipliers, the displacement
//! eliminated through the step's own solve.
//!
//! Source: Hueber, Stadler and Wohlmuth, *A primal-dual active set algorithm
//! for three-dimensional contact problems with Coulomb friction*, SIAM J.
//! Sci. Comput. 30(2) (2008): the complementarity functions (4.1) and (7.1),
//! their semismooth Newton step with the sets frozen at the iterate (3.4),
//! (7.2)-(7.3): on a contact pressed, the gap closed; sticking, the slip
//! closed; slipping, the tangential multiplier `mu` times the normal one;
//! released, both zero. Theorem 4.1 (after Hintermueller, Ito and Kunisch,
//! SIAM J. Optim. 13 (2003)): locally superlinear.
//! Adaptation, this engine's: they solve for the displacement and the
//! multipliers together; here the step's nonlinear solve gives the
//! displacement at the current multipliers, its Jacobian the displacement's
//! answer to each active multiplier (`Response`), and the Newton step is
//! taken on the multipliers alone, `S delta = -g` with `S` the end gaps'
//! derivative in them (the Schur complement of the frozen system). The
//! uzawa update `lambda <- max(0, lambda - k g)` is the same iteration with
//! `S` replaced by `1 / k`: it converges linearly, slowly when `S` has small
//! eigenvalues (measured: a stiff block's 7 contacts after it stopped, the
//! deepest gap 2e5 to 1.6e4 times the tolerance over 350 passes).

use super::*;

/// One Newton step on the active multipliers from those the solve used
/// (`before`, per contact: normal, tangential), written into `contacts`.
/// Returns false, leaving `contacts` as they are, when the gaps'
/// derivative is singular.
pub(crate) fn newton_on_multipliers(
    grid: &Grid,
    (x, planes, friction): (&[DVec2], &[Plane], f64),
    solution: &Solution,
    contacts: &mut [Contact],
    before: &[(f64, f64)],
) -> bool {
    let responses = &solution.responses;
    let count = responses.len();
    if count == 0 {
        return false;
    }
    // The gap a response's row closes: the normal one, or sticking, the
    // slip.
    let gap = |u: &[f64], r: &Response| -> f64 {
        let c = &contacts[r.contact];
        let (g, shares) = contact_geometry(grid, x, planes, u, c);
        if r.tangential {
            if friction > 0.0 {
                contact_tangential_gap(grid, u, &shares)
            } else {
                0.0
            }
        } else {
            g
        }
    };
    let residual: Vec<f64> = responses.iter().map(|r| gap(&solution.u, r)).collect();
    let base = solution.u.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    // Column `j`: the gaps' derivative along the displacement's answer to
    // multiplier `j`, by central differences (exact against a plane, the
    // gap being linear in `u` there).
    let mut matrix = vec![vec![0.0; count]; count];
    for (j, column) in responses.iter().enumerate() {
        let size = column.du.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        if size == 0.0 {
            return false;
        }
        let h = 1e-6 * (grid.cell + base) / size;
        let plus: Vec<f64> = solution
            .u
            .iter()
            .zip(&column.du)
            .map(|(a, b)| a + h * b)
            .collect();
        let minus: Vec<f64> = solution
            .u
            .iter()
            .zip(&column.du)
            .map(|(a, b)| a - h * b)
            .collect();
        for (i, row) in responses.iter().enumerate() {
            matrix[i][j] = (gap(&plus, row) - gap(&minus, row)) / (2.0 * h);
        }
    }
    // The weighted gaps (`weighted_gap`) keep the active rows independent,
    // so `S` is regular: the diagonal shift that stood in for the
    // collocation points' redundancy (unsourced) is gone with it.
    let rhs: Vec<f64> = residual.iter().map(|g| -g).collect();
    let Some(delta) = dense_solve(matrix, rhs) else {
        return false;
    };
    for (r, d) in responses.iter().zip(delta) {
        let (normal, tangential) = before[r.contact];
        let c = &mut contacts[r.contact];
        if r.tangential {
            c.tangential = tangential + d;
        } else {
            c.multiplier = normal + d;
        }
    }
    true
}

/// `a y = b` by Gaussian elimination with partial pivoting; `None` when a
/// pivot vanishes against the matrix's size (singular to round-off).
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let size = a
        .iter()
        .flatten()
        .fold(0.0f64, |m, e| m.max(e.abs()))
        .max(f64::MIN_POSITIVE);
    for i in 0..n {
        let pivot = (i..n).max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs()))?;
        if a[pivot][i].abs() <= f64::EPSILON * size * n as f64 {
            return None;
        }
        a.swap(i, pivot);
        b.swap(i, pivot);
        let pivot_row = a[i].clone();
        for r in i + 1..n {
            let factor = a[r][i] / pivot_row[i];
            for (e, p) in a[r][i..].iter_mut().zip(&pivot_row[i..]) {
                *e -= factor * p;
            }
            b[r] -= factor * b[i];
        }
    }
    for i in (0..n).rev() {
        let tail: f64 = (i + 1..n).map(|c| a[i][c] * b[c]).sum();
        b[i] = (b[i] - tail) / a[i][i];
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_solve_matches_a_known_system() {
        let a = vec![
            vec![0.0, 2.0, 1.0],
            vec![1.0, -1.0, 0.0],
            vec![3.0, 0.0, 4.0],
        ];
        let y = [1.5, -2.0, 0.25];
        let b: Vec<f64> = a
            .iter()
            .map(|row| row.iter().zip(&y).map(|(p, q)| p * q).sum())
            .collect();
        let solved = dense_solve(a, b).expect("regular");
        for (s, e) in solved.iter().zip(y) {
            assert!((s - e).abs() < 1e-14, "{s} against {e}");
        }
    }

    #[test]
    fn dense_solve_refuses_a_singular_system() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(dense_solve(a, vec![1.0, 2.0]).is_none());
    }
}

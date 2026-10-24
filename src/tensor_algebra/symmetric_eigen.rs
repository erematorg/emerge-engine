//! Eigenvalues of a real symmetric matrix, and the nearest symmetric
//! positive semidefinite matrix. The 4x4 forms are test-only, the oracles
//! of `spacetime::solver::implicit_corotated`'s dense direct solve; the
//! definite `n` by `n` form gives a body's modes (`total_lagrangian::
//! modes`).

/// Nearest symmetric positive semidefinite matrix in the Frobenius norm:
/// eigendecompose, clamp every negative eigenvalue to zero, rebuild.
/// Unlike a diagonal shift, it never changes an eigenvalue that was already
/// non-negative.
///
/// Source: Higham 1988, the nearest symmetric positive semidefinite matrix;
/// the same clamping principle as Teran, Sifakis, Irving & Fedkiw 2005.
#[cfg(test)]
pub(crate) fn spd_project_symmetric_4x4(sym: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let (eigenvalues, v) = jacobi_eigen_symmetric_4x4(sym);
    let clamped = eigenvalues.map(|lambda| lambda.max(0.0));
    let mut out = [[0.0f32; 4]; 4];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, out_ij) in out_row.iter_mut().enumerate() {
            *out_ij = (0..4).map(|k| v[i][k] * clamped[k] * v[j][k]).sum();
        }
    }
    out
}

/// Cyclic Jacobi eigenvalue algorithm for a real symmetric 4×4 matrix.
/// Returns the eigenvalues and the eigenvector matrix (columns), so that
/// `sym == v * diag(eigenvalues) * v^T`.
///
/// Source: Golub & Van Loan, *Matrix Computations*, 4th ed., section 8.4.3.
#[cfg(test)]
pub(crate) fn jacobi_eigen_symmetric_4x4(mut a: [[f32; 4]; 4]) -> ([f32; 4], [[f32; 4]; 4]) {
    let mut v = [[0.0f32; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..50 {
        let off: f32 = (0..4)
            .flat_map(|p| ((p + 1)..4).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        if off < 1.0e-20 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let a_pq = a[p][q];
                if a_pq.abs() < 1.0e-12 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a_pq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                let a_pp = a[p][p];
                let a_qq = a[q][q];
                a[p][p] = a_pp - t * a_pq;
                a[q][q] = a_qq + t * a_pq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                let mut updated_p = [0.0f32; 4];
                let mut updated_q = [0.0f32; 4];
                for (i, row) in a.iter().enumerate() {
                    if i != p && i != q {
                        let a_ip = row[p];
                        let a_iq = row[q];
                        updated_p[i] = a_ip - s * (a_iq + tau * a_ip);
                        updated_q[i] = a_iq + s * (a_ip - tau * a_iq);
                    }
                }
                // `a` stays symmetric: each updated off-diagonal entry
                // writes into both its row and its mirrored column.
                for (i, (&up, &uq)) in updated_p.iter().zip(&updated_q).enumerate() {
                    if i != p && i != q {
                        a[i][p] = up;
                        a[p][i] = up;
                        a[i][q] = uq;
                        a[q][i] = uq;
                    }
                }
                for row in v.iter_mut() {
                    let v_ip = row[p];
                    let v_iq = row[q];
                    row[p] = v_ip - s * (v_iq + tau * v_ip);
                    row[q] = v_iq + s * (v_ip - tau * v_iq);
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// Two-sided Jacobi for a real symmetric positive definite `n` by `n`
/// matrix `h` (row-major). Returns the eigenvalues and the eigenvector
/// matrix `v` (row-major, its columns the eigenvectors), `h = v
/// diag(lambda) v^T`.
///
/// Source: Demmel and Veselic, *Jacobi's method is more accurate than QR*,
/// LAPACK Working Note 15 (SIAM J. Matrix Anal. Appl. 13, 1992), Algorithm
/// 3.1: its rotation and its stopping criterion, every pair `|h_ij| /
/// (h_ii h_jj)^(1/2) <= tol`. Read: sections 1 and 3's algorithm; its error
/// analyses are not used here. Rutishauser's rotation, the one of
/// `jacobi_eigen_symmetric_4x4`, in `f64` for any size.
pub(crate) fn jacobi_eigen_symmetric_definite(
    mut h: Vec<f64>,
    n: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut rotated = false;
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b, c) = (h[i * n + i], h[j * n + j], h[i * n + j]);
                if c.abs() <= tol * (a * b).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (b - a) / (2.0 * c);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = cs * t;
                h[i * n + i] = a - c * t;
                h[j * n + j] = b + c * t;
                h[i * n + j] = 0.0;
                h[j * n + i] = 0.0;
                for k in (0..n).filter(|&k| k != i && k != j) {
                    let tmp = h[i * n + k];
                    h[i * n + k] = cs * tmp - sn * h[j * n + k];
                    h[j * n + k] = sn * tmp + cs * h[j * n + k];
                    h[k * n + i] = h[i * n + k];
                    h[k * n + j] = h[j * n + k];
                }
                for k in 0..n {
                    let tmp = v[k * n + i];
                    v[k * n + i] = cs * tmp - sn * v[k * n + j];
                    v[k * n + j] = sn * tmp + cs * v[k * n + j];
                }
            }
        }
        if !rotated {
            break;
        }
    }
    ((0..n).map(|i| h[i * n + i]).collect(), v)
}

#[cfg(all(test, feature = "experimental"))]
mod definite_tests {
    use super::*;

    /// A 3 x 3 definite matrix's eigenpairs: `h v = lambda v`, `v`
    /// orthonormal.
    #[test]
    fn the_jacobi_eigenpairs_satisfy_the_definition() {
        let n = 3;
        let h = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
        let (lambda, v) = jacobi_eigen_symmetric_definite(h.clone(), n, 1e-15);
        for e in 0..n {
            for i in 0..n {
                let hv: f64 = (0..n).map(|k| h[i * n + k] * v[k * n + e]).sum();
                assert!((hv - lambda[e] * v[i * n + e]).abs() < 1e-13, "pair {e}");
            }
            for f in 0..n {
                let dot: f64 = (0..n).map(|k| v[k * n + e] * v[k * n + f]).sum();
                let expected = if e == f { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-13, "columns {e}, {f}");
            }
        }
    }
}

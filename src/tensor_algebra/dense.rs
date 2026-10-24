//! Small dense real matrices, row-major: what the modal solves and the
//! matrix exponential share.

/// `a b` for row-major `n` by `n` matrices.
pub(crate) fn product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|ij| (0..n).map(|e| a[ij / n * n + e] * b[e * n + ij % n]).sum())
        .collect()
}

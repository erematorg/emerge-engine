//! The exponential of a small dense real matrix, by scaling and squaring
//! with the [13/13] Pade approximant.
//!
//! Source: Higham, *The scaling and squaring method for the matrix
//! exponential revisited*, SIAM J. Matrix Anal. Appl. 26(4) (2005)
//! 1179-1193 (MIMS EPrint 2006.394), Algorithm 2.3 (read). Scale `A` by
//! `2^-s` until `||2^-s A||_1 <= theta_13 = 5.371920351148152` (Table 2.3),
//! evaluate `r_13 = (-U + V)^-1 (U + V)` with `U`, `V` from `A^2`, `A^4`,
//! `A^6` (lines 17-20, the coefficients `b_0..b_13` of p. 1185), and square
//! `s` times. Theorem 2.1 with Table 2.1: the truncation's backward error is
//! below the unit roundoff, and `||q_13(A)^-1||` at most 17, so the solve is
//! well conditioned.
//!
//! Adaptations, declared:
//! - Lines 9-14 (lower degrees when `||A||_1` is small) are left out: they
//!   save multiplications only. The degree-13 approximant is used for every
//!   norm, which Theorem 2.1 covers for `||A||_1 <= theta_13`. Removal
//!   criterion: the lower degrees are built if a profile shows this
//!   routine's cost matters to a frame.
//! - The preprocessing of lines 7-8 (a trace shift and balancing) is left
//!   out. Higham found it makes "essentially no difference" in his tests
//!   (p. 1188).
//! - The squaring phase's rounding is the open question the paper leaves
//!   (sec. 5); callers check what they need of the result.

use super::dense::product;

/// Higham's `b(0:13)`, the [13/13] Pade coefficients (p. 1185).
const B: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

/// Higham's `theta_13` (Table 2.3).
const THETA_13: f64 = 5.371_920_351_148_152;

/// `e^A` of the `n` by `n` matrix `a`, row-major.
pub(crate) fn expm(a: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n, "an n by n matrix");
    let norm = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let s = if norm > THETA_13 {
        (norm / THETA_13).log2().ceil() as i32
    } else {
        0
    };
    let scale = 2f64.powi(-s);
    let a: Vec<f64> = a.iter().map(|e| e * scale).collect();
    let identity: Vec<f64> = (0..n * n)
        .map(|k| if k / n == k % n { 1.0 } else { 0.0 })
        .collect();
    let a2 = product(&a, &a, n);
    let a4 = product(&a2, &a2, n);
    let a6 = product(&a2, &a4, n);
    let sum = |terms: &[(f64, &[f64])]| -> Vec<f64> {
        (0..n * n)
            .map(|k| terms.iter().map(|(c, m)| c * m[k]).sum())
            .collect()
    };
    // Line 18: U = A [A6 (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I].
    let inner_u = sum(&[(B[13], &a6), (B[11], &a4), (B[9], &a2)]);
    let u_bracket = {
        let high = product(&a6, &inner_u, n);
        let low = sum(&[(B[7], &a6), (B[5], &a4), (B[3], &a2), (B[1], &identity)]);
        sum(&[(1.0, &high), (1.0, &low)])
    };
    let u = product(&a, &u_bracket, n);
    // Line 19: V = A6 (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I.
    let inner_v = sum(&[(B[12], &a6), (B[10], &a4), (B[8], &a2)]);
    let v = {
        let high = product(&a6, &inner_v, n);
        let low = sum(&[(B[6], &a6), (B[4], &a4), (B[2], &a2), (B[0], &identity)]);
        sum(&[(1.0, &high), (1.0, &low)])
    };
    // Line 20: solve (-U + V) r = U + V.
    let left = sum(&[(-1.0, &u), (1.0, &v)]);
    let right = sum(&[(1.0, &u), (1.0, &v)]);
    let mut x = solve(left, right, n);
    // Line 21: square s times.
    for _ in 0..s {
        x = product(&x, &x, n);
    }
    x
}

/// `a^-1 b` for `n` by `n` row-major matrices, by Gaussian elimination with
/// partial pivoting.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .expect("a row");
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                b.swap(col * n + k, pivot * n + k);
            }
        }
        let d = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / d;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            for k in 0..n {
                b[row * n + k] -= factor * b[col * n + k];
            }
        }
    }
    for row in (0..n).rev() {
        let d = a[row * n + row];
        for k in 0..n {
            let mut value = b[row * n + k];
            for j in row + 1..n {
                value -= a[row * n + j] * b[j * n + k];
            }
            b[row * n + k] = value / d;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f64], b: &[f64], tolerance: f64) -> bool {
        let size = b.iter().fold(0.0f64, |m, e| m.max(e.abs())).max(1.0);
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tolerance * size)
    }

    /// The rotation generator `[[0, w], [-w, 0]]`: its exponential is the
    /// rotation by `w`, below and above `theta_13` (the squaring then runs).
    #[test]
    fn the_exponential_of_a_rotation_generator_is_the_rotation() {
        for w in [0.1, 3.0, 30.0, 400.0] {
            let e = expm(&[0.0, w, -w, 0.0], 2);
            let exact = [w.cos(), w.sin(), -w.sin(), w.cos()];
            assert!(
                close(&e, &exact, 1e-12 * w.max(1.0)),
                "w {w}: {e:?} against {exact:?}"
            );
        }
    }

    /// `e^A e^-A = I` for a non-normal 5 by 5 matrix of norm above
    /// `theta_13`, and `e^D` exact for a diagonal one.
    #[test]
    fn the_exponential_inverts_and_diagonalises() {
        let a: Vec<f64> = (0..25)
            .map(|k| ((k * 7 + 3) % 11) as f64 / 3.0 - 1.5)
            .collect();
        let minus: Vec<f64> = a.iter().map(|e| -e).collect();
        let identity: Vec<f64> = (0..25)
            .map(|k| if k / 5 == k % 5 { 1.0 } else { 0.0 })
            .collect();
        let product_ = product(&expm(&a, 5), &expm(&minus, 5), 5);
        assert!(close(&product_, &identity, 1e-12), "{product_:?}");
        let d = [-3.0, 0.5, 2.0];
        let mut diagonal = vec![0.0; 9];
        for (i, &v) in d.iter().enumerate() {
            diagonal[i * 3 + i] = v;
        }
        let e = expm(&diagonal, 3);
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (e[i * 3 + i] / v.exp() - 1.0).abs() < 1e-14,
                "{}",
                e[i * 3 + i]
            );
        }
    }
}

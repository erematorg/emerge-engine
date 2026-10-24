use super::matrix2::Matrix2;

/// Frobenius inner product of two 2×2 matrices, `A : B = sum_ij a_ij b_ij`:
/// the matrix version of the dot product.
#[inline]
pub(crate) fn frob<M: Matrix2>(a: M, b: M) -> M::Scalar {
    let ([a00, a01], [a10, a11]) = (a.columns()[0], a.columns()[1]);
    let ([b00, b01], [b10, b11]) = (b.columns()[0], b.columns()[1]);
    a00 * b00 + a01 * b01 + a10 * b10 + a11 * b11
}

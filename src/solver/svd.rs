use glam::{Mat2, Vec2};
use nalgebra as na;

/// 2×2 SVD via nalgebra. Returns (U, sigma, Vt) with non-negative singular values.
/// Reconstructs as: F = U * Mat2::from_diagonal(sigma) * Vt
pub(crate) fn svd2(f: Mat2) -> (Mat2, Vec2, Mat2) {
    // glam Mat2 is column-major: x_axis = col0, y_axis = col1
    // nalgebra Matrix2::new(a,b,c,d) fills row-by-row: [[a,b],[c,d]]
    let m = na::Matrix2::new(
        f.x_axis.x, f.y_axis.x,
        f.x_axis.y, f.y_axis.y,
    );
    let svd = m.svd(true, true);
    let u_na = svd.u.unwrap();
    let vt_na = svd.v_t.unwrap();
    let s = svd.singular_values;

    // Convert back: col0 = first column = rows 0 and 1 of nalgebra col 0
    let u = Mat2::from_cols(
        Vec2::new(u_na[(0, 0)], u_na[(1, 0)]),
        Vec2::new(u_na[(0, 1)], u_na[(1, 1)]),
    );
    let vt = Mat2::from_cols(
        Vec2::new(vt_na[(0, 0)], vt_na[(1, 0)]),
        Vec2::new(vt_na[(0, 1)], vt_na[(1, 1)]),
    );

    (u, Vec2::new(s[0], s[1]), vt)
}

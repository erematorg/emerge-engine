use glam::{Mat2, Vec2};

/// Analytical 2×2 SVD. Returns (U, sigma, Vt) with σ₁ ≥ |σ₂| and det(U)=det(V)=+1 such that
/// F = U · diag(sigma) · Vt.
///
/// Algorithm: direct analytical eigendecomposition of C = FᵀF (symmetric 2×2).
/// 1. Eigenvalues of C via the closed-form trace/discriminant formula
/// 2. Eigenvectors analytically from (C − λI)v = 0
/// 3. σᵢ = √λᵢ; U columns = F·vᵢ / σᵢ (fallback to complement for σᵢ ≈ 0)
/// 4. If F is inverted, encode the reflection in signed σ₂ so U and V stay proper rotations
///
/// No external library or Jacobi iteration needed — fully closed-form.
/// Reference: standard 2×2 symmetric eigendecomposition; McAdams et al. 2011 §2.
pub(crate) fn svd2(f: Mat2) -> (Mat2, Vec2, Mat2) {
    // F column-major: x_axis = (f00, f10), y_axis = (f01, f11)
    let f00 = f.x_axis.x;
    let f10 = f.x_axis.y;
    let f01 = f.y_axis.x;
    let f11 = f.y_axis.y;

    // C = FᵀF — symmetric 2×2: [[c00, c01],[c01, c11]]
    let c00 = f00 * f00 + f10 * f10;
    let c01 = f00 * f01 + f10 * f11;
    let c11 = f01 * f01 + f11 * f11;

    // Eigenvalues of symmetric [[a, b],[b, c]]:
    //   λ₁,₂ = (a+c)/2 ± √(((a−c)/2)² + b²)
    let mean = (c00 + c11) * 0.5;
    let half_diff = (c00 - c11) * 0.5;
    let disc = (half_diff * half_diff + c01 * c01).sqrt();

    let lambda1 = mean + disc; // larger eigenvalue
    let lambda2 = mean - disc; // smaller eigenvalue (≥ 0 for PSD C)

    // Eigenvectors via (C − λI)v = 0.
    // For λ₁: first row gives  (c00 − λ₁)·x + c01·y = 0  →  v₁ ∝ (c01, λ₁ − c00)
    //   = (c01, disc − half_diff).  If c01 ≈ 0: axis-aligned fallback.
    // v₂ = perpendicular to v₁ (exact in 2D, no second solve needed).
    let v1 = if c01.abs() > 1e-10 * (c00 + c11).max(1e-30) {
        let raw = Vec2::new(c01, lambda1 - c00); // (b, disc − half_diff)
        raw / raw.length()
    } else {
        // Already diagonal: larger eigenvalue determines axis
        if c00 >= c11 { Vec2::X } else { Vec2::Y }
    };
    let v2 = Vec2::new(-v1.y, v1.x); // orthogonal complement (exact)

    // V columns in descending singular-value order: [v1 | v2]
    let v_sorted = Mat2::from_cols(v1, v2);
    let vt = v_sorted.transpose();

    // Singular values σᵢ = √λᵢ (clamp negatives from floating-point noise)
    let mut sigma = Vec2::new(lambda1.max(0.0).sqrt(), lambda2.max(0.0).sqrt());

    // U columns: uᵢ = F·vᵢ / σᵢ.  For σᵢ ≈ 0 (rank-deficient F): orthogonal fallback.
    let fv1 = f * v1;
    let fv2 = f * v2;
    let u1 = if sigma.x > 1e-10 {
        fv1 / sigma.x
    } else {
        orthogonal_complement(fv2)
    };
    let u2 = if sigma.y > 1e-10 {
        fv2 / sigma.y
    } else {
        orthogonal_complement(u1)
    };

    let mut u = Mat2::from_cols(u1, u2);
    if u.determinant() < 0.0 {
        u.y_axis = -u.y_axis;
        sigma.y = -sigma.y;
    }

    (u, sigma, vt)
}

/// Returns a unit vector orthogonal to `v` in 2D.
/// Fallback when a singular value is zero (rank-deficient F).
#[inline]
fn orthogonal_complement(v: Vec2) -> Vec2 {
    let perp = Vec2::new(-v.y, v.x);
    let len = perp.length();
    if len > 1e-10 { perp / len } else { Vec2::X }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_svd(f: Mat2) {
        let (u, sigma, vt) = svd2(f);

        // Reconstruction: U · diag(σ) · Vᵀ ≈ F
        let sigma_mat = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        let r = u * sigma_mat * vt;
        let ef = [f.x_axis.x, f.x_axis.y, f.y_axis.x, f.y_axis.y];
        let er = [r.x_axis.x, r.x_axis.y, r.y_axis.x, r.y_axis.y];
        for (a, b) in ef.iter().zip(er.iter()) {
            assert!((a - b).abs() < 1e-5, "reconstruction {a} vs {b}\nF={f:?}");
        }

        // Proper rotations with the reflection, if any, carried by σ₂.
        assert!(
            (u.determinant() - 1.0).abs() < 1e-5,
            "det(U)={}",
            u.determinant()
        );
        assert!(
            (vt.determinant() - 1.0).abs() < 1e-5,
            "det(V)={}",
            vt.determinant()
        );
        assert!(sigma.x >= 0.0, "negative σ₁: {sigma:?}");
        assert!(
            sigma.x >= sigma.y.abs() - 1e-6,
            "σ not sorted by magnitude: {sigma:?}"
        );

        // U orthogonal: UᵀU ≈ I
        let utu = u.transpose() * u;
        assert!((utu.x_axis.x - 1.0).abs() < 1e-5, "U[0,0]={}", utu.x_axis.x);
        assert!((utu.y_axis.y - 1.0).abs() < 1e-5, "U[1,1]={}", utu.y_axis.y);
        assert!(utu.x_axis.y.abs() < 1e-5, "U off-diag={}", utu.x_axis.y);
    }

    #[test]
    fn identity() {
        check_svd(Mat2::IDENTITY);
    }
    #[test]
    fn diagonal() {
        check_svd(Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 1.5)));
    }
    #[test]
    fn shear() {
        check_svd(Mat2::from_cols(Vec2::new(1.0, 0.5), Vec2::new(0.5, 1.0)));
    }
    #[test]
    fn generic_deformation() {
        check_svd(Mat2::from_cols(Vec2::new(1.2, 0.3), Vec2::new(-0.1, 0.9)));
    }
    #[test]
    fn rank_deficient() {
        check_svd(Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0)));
    }
    #[test]
    fn negative_det() {
        check_svd(Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, -1.0)));
    }
    #[test]
    fn near_identity() {
        check_svd(Mat2::from_cols(
            Vec2::new(0.99, 0.01),
            Vec2::new(-0.01, 1.01),
        ));
    }
    #[test]
    fn negative_det_uses_signed_sigma() {
        let (u, sigma, vt) = svd2(Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, -1.0)));
        assert!(
            (u.determinant() - 1.0).abs() < 1e-5,
            "det(U)={}",
            u.determinant()
        );
        assert!(
            (vt.determinant() - 1.0).abs() < 1e-5,
            "det(V)={}",
            vt.determinant()
        );
        assert!(
            sigma.y <= 0.0,
            "expected signed σ₂ for inverted F, got {sigma:?}"
        );
    }
    #[test]
    fn pure_rotation() {
        let a: f32 = 0.7;
        check_svd(Mat2::from_cols(
            Vec2::new(a.cos(), a.sin()),
            Vec2::new(-a.sin(), a.cos()),
        ));
    }
}

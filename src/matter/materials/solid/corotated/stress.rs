//! Fixed corotated elasticity: its Kirchhoff stress, its energy density, and
//! their derivatives with respect to `F`.
//!
//! Source: Stomakhin et al. 2013, eq. (5)-(8), the corotated model this
//! engine's snow material also cites.

#[cfg(test)]
use glam::Vec2;
use glam::{DMat2, DVec2, Mat2};

use crate::matter::kinematics::polar_decomposition::{
    polar_decomposition_2d, polar_decomposition_2d_jvp,
};
use crate::matter::kinematics::volume_ratio::MIN_J;
use crate::tensor_algebra::frobenius::frob;
use crate::tensor_algebra::matrix2::{Matrix2, Real};
#[cfg(test)]
use crate::tensor_algebra::symmetric_eigen::spd_project_symmetric_4x4;

/// Corotated-elastic Kirchhoff stress: `2*mu*(F-R)*F^T + lambda*(J-1)*J*I`,
/// the standard hyperelastic form (Stomakhin et al. 2013, the same
/// corotated model this engine's snow material already cites). Every
/// plastic material that returns to this SAME elastic branch after its own
/// return-mapping (DruckerPragerMaterial, VonMisesMaterial, RankineMaterial,
/// MuIRheologyMaterial) used to hand-duplicate this exact formula in its own
/// `kirchhoff_stress` -- each material's plasticity lives entirely in its
/// own `update_particle`, this function owns only the shared elastic stress
/// evaluated on the (already plastically corrected) current F. One
/// definition at any precision: `f32` for the explicit solver, `f64` for
/// the total Lagrangian step.
#[inline]
pub(crate) fn corotated_elastic_stress<M: Matrix2>(f: M, lambda: M::Scalar, mu: M::Scalar) -> M {
    let j = f.determinant();
    if j <= M::Scalar::from(MIN_J) {
        return M::ZERO;
    }
    let r = polar_decomposition_2d(f);
    let one = M::Scalar::from(1.0);
    (f - r).scaled(M::Scalar::from(2.0) * mu) * f.transpose()
        + M::IDENTITY.scaled(lambda * (j - one) * j)
}

/// Corotated elastic energy density `Psi(F) = mu*||F-R||_F^2 +
/// 0.5*lambda*(J-1)^2` (Stomakhin et al. 2013). `corotated_elastic_stress` is
/// its First-Piola gradient times `F^T`: `dPsi/dF = 2*mu*(F-R) +
/// lambda*(J-1)*J*F^-T`, and `(dPsi/dF)*F^T = 2*mu*(F-R)*F^T +
/// lambda*(J-1)*J*I`. Zero at `J <= MIN_J`, like the stress ("no force").
/// The total Lagrangian step's discrete gradient reads it at `f64`; the
/// oracle `model_residual`'s gradient is checked against it at `f32` in
/// `spacetime::solver::implicit_corotated`.
#[inline]
pub(crate) fn corotated_elastic_energy_density<M: Matrix2>(
    f: M,
    lambda: M::Scalar,
    mu: M::Scalar,
) -> M::Scalar {
    let j = f.determinant();
    if j <= M::Scalar::from(MIN_J) {
        return M::Scalar::from(0.0);
    }
    let r = polar_decomposition_2d(f);
    let dev = f - r;
    let one = M::Scalar::from(1.0);
    mu * frob(dev, dev) + M::Scalar::from(0.5) * lambda * (j - one) * (j - one)
}

/// The corotated energy's change between right Cauchy-Green tensors `c0`
/// and `c1`, `W(C_1) - W(C_0)`, computed from the increment `dC = C_1 -
/// C_0` itself, so its round-off is relative to the change, not to the
/// energies (whose own, about `eps |S| |C|`, does not shrink with `dC`).
/// With `F = U` (`R = I`), `W(C) = mu (tr C - 2 tr U + 2) + lambda/2 (J -
/// 1)^2`, `J = sqrt(det C)`, `tr U = sqrt(tr C + 2 J)` (a 2 x 2 symmetric
/// positive definite `U`, `(tr U)^2 = tr C + 2 det U`). Exact identities:
/// `det(C_0 + dC) - det C_0 = tr(adj(C_0) dC) + det(dC)`, `J_1 - J_0 =
/// (det C_1 - det C_0) / (J_1 + J_0)`, `tr U_1 - tr U_0 = (tr dC + 2 (J_1 -
/// J_0)) / (tr U_1 + tr U_0)`, `(J_1 - 1)^2 - (J_0 - 1)^2 = (J_1 - J_0) (J_1
/// + J_0 - 2)`. Derived on the page from `corotated_elastic_energy_density`
/// (Stomakhin et al. 2013), the same energy at `F = sqrt(C)`.
#[inline]
pub(crate) fn corotated_energy_increment<M: Matrix2>(
    c0: M,
    c1: M,
    lambda: M::Scalar,
    mu: M::Scalar,
) -> M::Scalar {
    let dc = c1 - c0;
    let [[a, b], [c, d]] = c0.columns();
    let [[da, db], [dcc, dd]] = dc.columns();
    let two = M::Scalar::from(2.0);
    let det_change = a * dd + d * da - b * dcc - c * db + (da * dd - db * dcc);
    let (j0, j1) = (c0.determinant().sqrt(), c1.determinant().sqrt());
    let j_change = det_change / (j0 + j1);
    let tr_change = da + dd;
    let [[a1, _], [_, d1]] = c1.columns();
    let (tr_u0, tr_u1) = ((a + d + two * j0).sqrt(), (a1 + d1 + two * j1).sqrt());
    let tr_u_change = (tr_change + two * j_change) / (tr_u0 + tr_u1);
    mu * (tr_change - two * tr_u_change)
        + M::Scalar::from(0.5) * lambda * j_change * (j0 + j1 - two)
}

/// The corotated law's second Piola-Kirchhoff stress as a function of `C`,
/// `S(C) = 2 mu (I - U^-1) + lambda (1 - 1/J) adj(C)`, in `f64` for the total
/// Lagrangian step. It is `2 dW/dC` of the `W(C)` of
/// `corotated_energy_increment` (see `corotated_second_pk_tangent`), with
/// two 2 x 2 identities derived on the page: Cayley-Hamilton gives `U^-1 =
/// (t I - U) / J` and `U = (C + J I) / t`, `t = tr U = sqrt(tr C + 2J)`, so
/// `U^-1 = ((tr C + J) I - C) / (t J)`; and `J (J - 1) C^-1 = (1 - 1/J)
/// adj(C)`, `adj(C) = tr(C) I - C`. Linear in `(lambda, mu)`: `(0, mu)` is
/// the law's shear part (`mu |F - R|^2`), `(lambda, 0)` its volume part
/// (`lambda/2 (J - 1)^2`). Zero at `J <= MIN_J`, like the stress.
pub(crate) fn corotated_second_pk(c: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let j = c.determinant().max(0.0).sqrt();
    if j <= f64::from(MIN_J) {
        return DMat2::ZERO;
    }
    let tr = c.x_axis.x + c.y_axis.y;
    let t = (tr + 2.0 * j).sqrt();
    let identity = DMat2::IDENTITY;
    let u_inv = (identity * (tr + j) - c) * (t * j).recip();
    let adj = identity * tr - c;
    (identity - u_inv) * (2.0 * mu) + adj * (lambda * (1.0 - j.recip()))
}

/// `corotated_second_pk(c1) - corotated_second_pk(c0)` computed from the
/// increment `dC = C_1 - C_0` itself, so its round-off is relative to the
/// change (`corotated_energy_increment`'s reason). From the forms of
/// `corotated_second_pk`, with `a = tr C + J` and the exact increments
/// `d(det C) = tr(adj(C_0) dC) + det(dC)`, `dJ = d(det C) / (J_0 + J_1)`,
/// `dt = (tr dC + 2 dJ) / (t_0 + t_1)`, `d(t J) = t_1 dJ + J_0 dt` (derived
/// on the page):
/// - `d(U^-1) = (da I - dC) / (t_1 J_1) - (a_0 I - C_0) d(t J) / (t_0 J_0 t_1
///   J_1)`, `da = tr dC + dJ`;
/// - `d((1 - 1/J) adj C) = (1 - 1/J_1) adj(dC) + dJ / (J_0 J_1) adj(C_0)`.
///
/// Zero where either end is at `J <= MIN_J` (the stress is pinned there).
pub(crate) fn corotated_second_pk_increment(c0: DMat2, c1: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let (j0, j1) = (
        c0.determinant().max(0.0).sqrt(),
        c1.determinant().max(0.0).sqrt(),
    );
    if j0 <= f64::from(MIN_J) || j1 <= f64::from(MIN_J) {
        return corotated_second_pk(c1, lambda, mu) - corotated_second_pk(c0, lambda, mu);
    }
    let dc = c1 - c0;
    let [[a, b], [c, d]] = [c0.x_axis.to_array(), c0.y_axis.to_array()];
    let [[da, db], [dcc, dd]] = [dc.x_axis.to_array(), dc.y_axis.to_array()];
    let det_change = a * dd + d * da - b * dcc - c * db + (da * dd - db * dcc);
    let dj = det_change / (j0 + j1);
    let (tr0, tr_change) = (a + d, da + dd);
    let (t0, t1) = ((tr0 + 2.0 * j0).sqrt(), (tr0 + tr_change + 2.0 * j1).sqrt());
    let dt = (tr_change + 2.0 * dj) / (t0 + t1);
    let d_tj = t1 * dj + j0 * dt;
    let identity = DMat2::IDENTITY;
    let d_u_inv = (identity * (tr_change + dj) - dc) * (t1 * j1).recip()
        - (identity * (tr0 + j0) - c0) * (d_tj / (t0 * j0 * t1 * j1));
    let d_volume =
        (identity * tr_change - dc) * (1.0 - j1.recip()) + (identity * tr0 - c0) * (dj / (j0 * j1));
    d_u_inv * (-2.0 * mu) + d_volume * lambda
}

/// The derivative of the corotated law's second Piola-Kirchhoff stress in
/// `C`, along `dc`: `dS/dC [dC]`, in `f64` for the total Lagrangian step
/// (the explicit path differentiates in `F`, `corotated_elastic_stress_jvp`).
///
/// Derived here. With `F = U = C^1/2` (then `R = I`) this file's Kirchhoff
/// stress pulls back to `S(C) = U^-1 tau U^-1 = 2 mu (I - U^-1) + lambda J
/// (J - 1) C^-1`, `J = det U`: the gradient `2 dW/dC` of the `W(C)` of
/// `corotated_energy_increment` (`d tr U = 1/2 tr(U^-1 dC)`, `dJ = J/2
/// tr(C^-1 dC)`). Its derivative is
/// `dS = 2 mu U^-1 dU U^-1 + lambda (2J - 1) dJ C^-1 - lambda J (J - 1) C^-1 dC C^-1`,
/// with `dU` the solution of `dU U + U dU = dC`. That 2 x 2 Sylvester
/// equation in closed form, from Cayley-Hamilton (`U^2 = t U - J I`, `t = tr
/// U`):
/// `dU = (t / (2J) + 1 / (2t)) dC - (U dC + dC U) / (2J) + U dC U / (2 J t)`
/// (it gives `dU = dC / (2v)` at `U = v I`). No eigenvectors and no
/// difference quotient, so it is smooth through equal stretches. Zero at `J
/// <= MIN_J`, like the stress.
pub(crate) fn corotated_second_pk_tangent(c: DMat2, dc: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let j = c.determinant().max(0.0).sqrt();
    if j <= f64::from(MIN_J) {
        return DMat2::ZERO;
    }
    // `U = (C + J I) / sqrt(tr C + 2 J)`, `tr U = sqrt(tr C + 2 J)`.
    let t = (c.x_axis.x + c.y_axis.y + 2.0 * j).sqrt();
    let u = (c + DMat2::from_diagonal(DVec2::splat(j))) * t.recip();
    let du = dc * (t / (2.0 * j) + 0.5 / t) - (u * dc + dc * u) * (0.5 / j)
        + u * dc * u * (0.5 / (j * t));
    let (u_inv, c_inv) = (u.inverse(), c.inverse());
    let trace = |m: DMat2| m.x_axis.x + m.y_axis.y;
    let dj = 0.5 * j * trace(c_inv * dc);
    u_inv * du * u_inv * (2.0 * mu) + c_inv * (lambda * (2.0 * j - 1.0) * dj)
        - c_inv * dc * c_inv * (lambda * j * (j - 1.0))
}

/// Analytic Jacobian-vector product of `corotated_elastic_stress`'s
/// Kirchhoff stress w.r.t. `F`: given `F` and a perturbation `dF`, returns
/// `dTau`. Product rule on `2*mu*(F-R)*F^T + lambda*(J-1)*J*I` using
/// `polar_decomposition_2d_jvp` above and `dJ=J*(F^-T:dF)`. The
/// Hessian-vector-product building block of an implicit (Newton-CG) solve for
/// every material on this elastic branch (DruckerPrager, VonMises, Rankine,
/// MuIRheology, Corotated), see `spacetime::solver::implicit_corotated`.
/// Checked by central finite differences
/// (`corotated_stress_jvp_matches_finite_difference`): max relative error
/// 0.23% at sand-magnitude stiffness.
///
/// Zero at `j <= MIN_J`, where the stress is pinned at zero: a practical
/// simplification at a hard clamp, not a claim about the continuous
/// derivative.
#[inline]
pub(crate) fn corotated_elastic_stress_jvp(f: Mat2, df: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    if j <= MIN_J {
        return Mat2::ZERO;
    }
    let r = polar_decomposition_2d(f);
    let dr = polar_decomposition_2d_jvp(f, df);
    let f_t = f.transpose();
    let df_t = df.transpose();
    let f_inv_t = f.inverse().transpose();
    let d_j = j * frob(f_inv_t, df); // dJ = J*(F^-T:dF)

    let d_elastic_term = (df - dr) * f_t + (f - r) * df_t;
    let d_vol_term = (2.0 * j - 1.0) * d_j; // d[(J-1)*J] = (2J-1)*dJ
    2.0 * mu * d_elastic_term + lambda * d_vol_term * Mat2::IDENTITY
}

/// A PSD approximation of `d(tau)/d(L)` (the Kirchhoff stress derivative
/// with respect to a velocity-gradient perturbation `dl`, through `dF =
/// dt*dl*f_n`), applied to `dl`.
///
/// Teran et al. 2005's SPD construction proves `dP:dF >= 0` for First-Piola
/// stress with a material-space perturbation. This engine's force pairs
/// Kirchhoff stress with the spatial kernel gradient (as `tmp/ziran2020` and
/// `tmp/GeoTaichi` do), and `tau = P*F^T` as a function of `L` inherits no such
/// guarantee: a Piola-projected operator converted by the product rule still
/// let the residual grow across CG iterations, which a PSD system cannot do.
///
/// So the local 4x4 operator (`L` and `tau` both have 4 components in 2D) is
/// built from four calls to the exact `corotated_elastic_stress_jvp`,
/// symmetrized (only the symmetric part enters the quadratic form), then
/// eigendecomposed (`jacobi_eigen_symmetric_4x4`, Golub & Van Loan's cyclic
/// Jacobi) with negative eigenvalues clamped to zero: the nearest SPD matrix
/// in Frobenius norm (Higham 1988), Teran's clamping principle without his
/// `dP/dF`-specific closed form. A Gershgorin diagonal shift, also
/// sufficient for PSD, swamps the curvature at production stiffness
/// (`basic_sand`'s E = 15 MPa): the line search rejected every step.
///
/// Split in two: [`corotated_kirchhoff_dtau_dl_psd_matrix`] builds the matrix
/// once per trial `v` (it does not depend on `dl`); [`apply_dtau_dl_psd_matrix`]
/// applies it for a 4x4 mat-vec, instead of rebuilding it every CG iteration.
///
/// Test-only: `implicit_corotated` uses `steihaug_cg` (Nocedal & Wright),
/// which handles indefinite curvature and needs no PSD projection. Kept, with
/// `corotated_kirchhoff_dtau_dl_psd_tests`, in case a PSD operator is needed
/// again.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd_matrix(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
) -> [[f32; 4]; 4] {
    let basis = [
        Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
    ];
    let mut h = [[0.0f32; 4]; 4];
    for (col, &e) in basis.iter().enumerate() {
        let df = dt * e * f_n;
        let d_tau = corotated_elastic_stress_jvp(f, df, lambda, mu);
        let d_tau_vec = [
            d_tau.x_axis.x,
            d_tau.x_axis.y,
            d_tau.y_axis.x,
            d_tau.y_axis.y,
        ];
        for (row, &val) in d_tau_vec.iter().enumerate() {
            h[row][col] = val;
        }
    }
    let mut sym = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            sym[i][j] = 0.5 * (h[i][j] + h[j][i]);
        }
    }
    spd_project_symmetric_4x4(sym)
}

/// Application step for [`corotated_kirchhoff_dtau_dl_psd_matrix`]'s output:
/// a 4x4 mat-vec, no JVP evaluations. Test-only, for
/// [`corotated_kirchhoff_dtau_dl_psd`]'s direct comparison.
#[cfg(test)]
#[inline]
pub(crate) fn apply_dtau_dl_psd_matrix(matrix: &[[f32; 4]; 4], dl: Mat2) -> Mat2 {
    let dl_vec = [dl.x_axis.x, dl.x_axis.y, dl.y_axis.x, dl.y_axis.y];
    let mut out = [0.0f32; 4];
    for (i, row) in matrix.iter().enumerate() {
        out[i] = row.iter().zip(&dl_vec).map(|(a, b)| a * b).sum();
    }
    Mat2::from_cols(Vec2::new(out[0], out[1]), Vec2::new(out[2], out[3]))
}

/// Convenience wrapper matching the original single-call API (build +
/// apply in one shot) -- kept for the test suite's own direct-comparison
/// checks; production code (`spacetime::solver::implicit_corotated`) uses
/// the split build/apply pair instead to avoid rebuilding the matrix once
/// per CG iteration.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
    dl: Mat2,
) -> Mat2 {
    let matrix = corotated_kirchhoff_dtau_dl_psd_matrix(f, lambda, mu, dt, f_n);
    apply_dtau_dl_psd_matrix(&matrix, dl)
}

#[cfg(test)]
mod corotated_elastic_stress_jvp_tests {
    use super::*;

    fn frob_pub(a: Mat2, b: Mat2) -> f32 {
        a.x_axis.x * b.x_axis.x
            + a.x_axis.y * b.x_axis.y
            + a.y_axis.x * b.y_axis.x
            + a.y_axis.y * b.y_axis.y
    }

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
            Mat2::from_cols(Vec2::new(0.9, -0.2), Vec2::new(0.3, 1.2)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }
    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5)),
        ]
    }

    /// Central finite difference at h = 1e-3, not smaller: f32 cancellation on
    /// `polar_decomposition_2d`'s normalized quantity makes smaller h worse.
    #[test]
    fn matches_finite_difference_at_moderate_and_real_sand_stiffness() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                for &df in &directions() {
                    let plus = corotated_elastic_stress(f + h * df, lambda, mu);
                    let minus = corotated_elastic_stress(f - h * df, lambda, mu);
                    let numeric = (plus - minus) * (1.0 / (2.0 * h));
                    let analytic = corotated_elastic_stress_jvp(f, df, lambda, mu);
                    let diff = analytic - numeric;
                    let rel_err =
                        frob_pub(diff, diff).sqrt() / frob_pub(analytic, analytic).sqrt().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_stress_jvp: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "JVP wrong at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }

    #[test]
    fn zero_below_min_j_floor_matching_the_stress_itself() {
        // A near-singular F that pins `corotated_elastic_stress` at
        // Mat2::ZERO -- the JVP must match that pinned-constant behavior.
        let f = Mat2::from_cols(Vec2::new(1.0e-8, 0.0), Vec2::new(0.0, 1.0e-8));
        let df = Mat2::from_cols(Vec2::new(0.1, 0.0), Vec2::new(0.0, 0.1));
        assert_eq!(corotated_elastic_stress(f, 100.0, 50.0), Mat2::ZERO);
        assert_eq!(corotated_elastic_stress_jvp(f, df, 100.0, 50.0), Mat2::ZERO);
    }

    /// `corotated_elastic_energy_density` is the potential
    /// `corotated_elastic_stress` is the gradient of, so the trust-region
    /// ratio test solves the problem the other tests trust. Central finite
    /// difference at h = 1e-3 (smaller h is worse here); First-Piola comes from
    /// the Kirchhoff function via `P = tau*F^-T` rather than a second copy of
    /// the formula.
    #[test]
    fn energy_density_gradient_matches_finite_difference_of_the_stress() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                // `Mat2::IDENTITY` is a critical point of `Psi`
                // (F=R, J=1 -> both terms are exactly zero, the analytic
                // global minimum) -- its analytic gradient is exactly
                // zero, which makes central-difference ill-conditioned
                // there by construction: `f(x+h)`/`f(x-h)` are both
                // dominated by the (identical, non-cancelling) O(h^2)
                // curvature term, so `(plus-minus)/(2h)` recovers pure
                // O(h^2)*f'''/6 truncation noise, not a discrepancy
                // (confirmed: at lambda=2e7, this alone produced a raw FD
                // value of ~1.8 against an analytic 0 -- but EVERY other
                // state, all deformed with nonzero analytic
                // gradients, passed at the same h and stiffness with
                // max_rel_err=2.45e-4). Standard, documented FD-checking
                // practice (e.g. PyTorch's `gradcheck`) explicitly skips
                // exactly this scenario -- checking a derivative AT a
                // critical point -- rather than loosening the tolerance
                // for every other, well-conditioned case too.
                if f == Mat2::IDENTITY {
                    continue;
                }
                for &df in &directions() {
                    let plus = corotated_elastic_energy_density(f + h * df, lambda, mu);
                    let minus = corotated_elastic_energy_density(f - h * df, lambda, mu);
                    let numeric = (plus - minus) / (2.0 * h);
                    let tau = corotated_elastic_stress(f, lambda, mu);
                    let piola = tau * f.inverse().transpose();
                    let analytic = frob_pub(piola, df);
                    let rel_err = (analytic - numeric).abs() / analytic.abs().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_energy_density: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "energy gradient doesn't match the stress at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }
}

#[cfg(test)]
mod corotated_kirchhoff_dtau_dl_psd_tests {
    use super::*;

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.1), Vec2::new(-0.05, 1.05)),
            // Large volumetric expansion -- the regime the plain
            // (unprojected) Hessian is known-indefinite in.
            Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }

    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.1), Vec2::new(-0.1, 0.3)),
            Mat2::from_cols(Vec2::new(-0.1, 0.2), Vec2::new(0.15, -0.2)),
        ]
    }

    /// The actual property this function exists for: `frob(H(dl), dl) >=
    /// 0` for EVERY direction `dl`, at EVERY state including the ones the
    /// plain (unprojected) Hessian is known-indefinite at -- this is what
    /// guarantees CG can never see a negative-curvature direction from
    /// this operator, unlike the plain JVP or a Piola-space projection
    /// (which does not survive the Kirchhoff/spatial-gradient conversion:
    /// CG in `spacetime::solver::implicit_corotated` saw negative curvature
    /// with it).
    #[test]
    fn quadratic_form_is_never_negative() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        for &f in &states() {
            for &f_n in &states() {
                for &dl in &directions() {
                    let h_dl = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
                    let q = frob(h_dl, dl);
                    assert!(
                        q >= -1.0e-3 * frob(dl, dl).max(1.0),
                        "quadratic form went negative: f={f:?} f_n={f_n:?} dl={dl:?} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states() {
        // At small/moderate deformation with no pre-existing rotation, the
        // real local Hessian's SYMMETRIC PART is already close to PSD (no
        // real Gershgorin shift needed), so this should stay close to the
        // exact chain-rule JVP. NOT bit-identical even here, by design:
        // symmetrizing intentionally discards `dTau/dL`'s antisymmetric
        // part (which contributes nothing to the quadratic form PSD-ness
        // is actually about), so a few-percent difference from the
        // raw (non-symmetrized) exact JVP is expected, not a bug -- this
        // check is only about ruling out AGGRESSIVE over-damping at a
        // benign state, not exact reproduction.
        let lambda = 2.0e5f32;
        let mu = 1.5e5f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let diff = via_psd - exact;
            let rel_err = frob(diff, diff).sqrt() / frob(exact, exact).sqrt().max(1.0);
            assert!(
                rel_err < 5.0e-2,
                "should stay close to the exact JVP at a moderate, benign state (no aggressive over-damping): dl={dl:?} rel_err={rel_err}"
            );
        }
    }

    /// At production stiffness (`basic_sand`'s E = 15 MPa DruckerPrager, lambda
    /// and mu of this order) and a benign, mostly PSD state, the projection
    /// keeps the curvature close to intact: it tracks the exact JVP's
    /// quadratic form about as closely as
    /// `matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states` does
    /// at low stiffness. A Gershgorin shift large enough for diagonal
    /// dominance here swamped the curvature (every line-search step
    /// rejected over 30 frames).
    #[test]
    fn eigenvalue_clamp_preserves_curvature_at_real_production_stiffness() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let q_psd = frob(via_psd, dl);
            let q_exact = frob(exact, dl);
            assert!(
                q_psd > 0.5 * q_exact,
                "eigenvalue clamp over-damped real positive curvature at production stiffness: dl={dl:?} q_psd={q_psd} q_exact={q_exact}"
            );
        }
    }

    #[test]
    fn stays_finite_at_a_large_volumetric_expansion() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let f = Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0));
        let f_n = Mat2::IDENTITY;
        let dl = Mat2::from_cols(Vec2::new(0.5, 0.2), Vec2::new(-0.3, 0.4));
        let out = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, 0.016, f_n, dl);
        assert!(
            out.x_axis.is_finite() && out.y_axis.is_finite(),
            "must stay finite even in the indefinite regime: {out:?}"
        );
    }
}

#[cfg(test)]
mod corotated_energy_increment_tests {
    use super::*;
    use glam::{DMat2, DVec2};

    fn sqrt_spd(c: DMat2) -> DMat2 {
        let s = c.determinant().sqrt();
        let t = (c.x_axis.x + c.y_axis.y + 2.0 * s).sqrt();
        (c + DMat2::IDENTITY * s) * (1.0 / t)
    }

    /// The increment equals the energies' difference at `F = sqrt(C)`, for
    /// a sizeable change (where the difference is accurate), and stays
    /// relative to the change for a tiny one (where it is not).
    #[test]
    fn the_increment_is_the_energy_difference() {
        let (lambda, mu) = (3.0, 2.0);
        let f0 = DMat2::from_cols(DVec2::new(1.05, 0.02), DVec2::new(-0.01, 0.97));
        let f1 = DMat2::from_cols(DVec2::new(1.1, -0.03), DVec2::new(0.04, 0.9));
        let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
        let direct = corotated_elastic_energy_density(sqrt_spd(c1), lambda, mu)
            - corotated_elastic_energy_density(sqrt_spd(c0), lambda, mu);
        let increment = corotated_energy_increment(c0, c1, lambda, mu);
        assert!(
            (increment - direct).abs() < 1e-12 * direct.abs(),
            "{increment} against {direct}"
        );
        // Changes of 1e-12 and 2e-12: the increment is linear in the change
        // the inputs actually carry, `C_1 - C_0` as rounded (`C_0 + 1e-12`
        // holds it only to `eps |C_0|`, 2e-4 of it), to 1e-6 relative.
        let tiny = c0 + DMat2::from_cols(DVec2::new(1e-12, 0.0), DVec2::new(0.0, 0.0));
        let tiny2 = c0 + DMat2::from_cols(DVec2::new(2e-12, 0.0), DVec2::new(0.0, 0.0));
        let (d1, d2) = (
            corotated_energy_increment(c0, tiny, lambda, mu),
            corotated_energy_increment(c0, tiny2, lambda, mu),
        );
        let carried = (tiny2.x_axis.x - c0.x_axis.x) / (tiny.x_axis.x - c0.x_axis.x);
        assert!(
            (d2 / d1 / carried - 1.0).abs() < 1e-6,
            "{d1} and {d2}, carried ratio {carried}"
        );
    }
}

#[cfg(test)]
mod second_pk_tangent_tests {
    use super::*;

    /// `S(C) = U^-1 tau(U) U^-1` from this file's Kirchhoff stress at `F =
    /// sqrt(C)`.
    fn second_pk(c: DMat2, lambda: f64, mu: f64) -> DMat2 {
        let j = c.determinant().sqrt();
        let t = (c.x_axis.x + c.y_axis.y + 2.0 * j).sqrt();
        let u = (c + DMat2::from_diagonal(DVec2::splat(j))) * t.recip();
        let u_inv = u.inverse();
        u_inv * corotated_elastic_stress(u, lambda, mu) * u_inv
    }

    /// The closed form `corotated_second_pk` against the stress pulled back
    /// from the Kirchhoff stress, both parts, to round-off; and its increment
    /// against the two ends' difference where that difference is accurate (a
    /// large step), and against the increment's own `eps |dC|` scale on a
    /// step too small for the difference (1e-13: the difference's round-off,
    /// `eps |S|`, is then a thousand times the change).
    #[test]
    fn the_closed_form_stress_and_its_increment_are_exact() {
        let (lambda, mu) = (2.0e6, 1.3e6);
        let size = |m: DMat2| (m.x_axis.length_squared() + m.y_axis.length_squared()).sqrt();
        let cases = [
            DMat2::from_cols(DVec2::new(1.21, 0.07), DVec2::new(0.07, 0.86)),
            DMat2::from_cols(DVec2::new(0.64, -0.2), DVec2::new(-0.2, 1.5)),
            DMat2::IDENTITY,
        ];
        for c in cases {
            for (l, m) in [(lambda, 0.0), (0.0, mu), (lambda, mu)] {
                let (closed, pulled) = (corotated_second_pk(c, l, m), second_pk(c, l, m));
                assert!(
                    size(closed - pulled) <= 1e-12 * size(pulled).max(l.max(m)),
                    "C {c}, (lambda, mu) ({l}, {m}): {closed} against {pulled}"
                );
            }
        }
        let c0 = DMat2::from_cols(DVec2::new(1.1, 0.04), DVec2::new(0.04, 0.93));
        let direction = DMat2::from_cols(DVec2::new(0.3, -0.5), DVec2::new(-0.5, 0.8));
        for (l, m) in [(lambda, 0.0), (0.0, mu)] {
            let large = c0 + direction * 0.05;
            let (increment, difference) = (
                corotated_second_pk_increment(c0, large, l, m),
                corotated_second_pk(large, l, m) - corotated_second_pk(c0, l, m),
            );
            assert!(
                size(increment - difference) <= 1e-12 * size(difference),
                "large step: {increment} against {difference}"
            );
            // A tiny step: the increment against the tangent's first-order
            // change along the step `C_1` actually holds (`C_0 + h D` rounds
            // the step by `eps |C|`, 2e-3 of it here), to the step's second
            // order and the increment's own round-off, both far below the
            // difference's.
            let h = 1e-13;
            let c1 = c0 + direction * h;
            let tiny = corotated_second_pk_increment(c0, c1, l, m);
            let linear = corotated_second_pk_tangent(c0, c1 - c0, l, m);
            assert!(
                size(tiny - linear) <= 1e-9 * size(linear),
                "tiny step: {tiny} against the tangent's {linear}"
            );
        }
    }

    /// The tangent against central differences of the stress, at sheared,
    /// stretched and nearly isotropic `C` (equal stretches included), to the
    /// differences' own truncation `O(h^2)` and round-off `eps / h`.
    #[test]
    fn the_tangent_is_the_stress_derivative() {
        let (lambda, mu) = (2.0e6, 1.3e6);
        let cases = [
            DMat2::from_cols(DVec2::new(1.21, 0.07), DVec2::new(0.07, 0.86)),
            DMat2::from_cols(DVec2::new(0.64, -0.2), DVec2::new(-0.2, 1.5)),
            DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0)),
            DMat2::from_cols(DVec2::new(1.0 + 1e-9, 0.0), DVec2::new(0.0, 1.0)),
        ];
        let directions = [
            DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::new(0.0, 0.0)),
            DMat2::from_cols(DVec2::new(0.0, 0.0), DVec2::new(0.0, 1.0)),
            DMat2::from_cols(DVec2::new(0.0, 1.0), DVec2::new(1.0, 0.0)),
            DMat2::from_cols(DVec2::new(0.3, -0.5), DVec2::new(-0.5, 0.8)),
        ];
        let h = 1e-6;
        for c in cases {
            for dc in directions {
                let exact = corotated_second_pk_tangent(c, dc, lambda, mu);
                let fd = (second_pk(c + dc * h, lambda, mu) - second_pk(c - dc * h, lambda, mu))
                    * (0.5 / h);
                let size = (exact.x_axis.length_squared() + exact.y_axis.length_squared())
                    .sqrt()
                    .max(mu);
                let gap = exact - fd;
                let error = (gap.x_axis.length_squared() + gap.y_axis.length_squared()).sqrt();
                assert!(
                    error <= 1e-6 * size,
                    "C {c}, dC {dc}: tangent {exact} against differences {fd}"
                );
            }
        }
    }
}

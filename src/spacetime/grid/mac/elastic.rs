//! The shear part of an elastic solid on the staggered grid (step A2).
//!
//! Stomakhin, Schroeder, Jiang, Chai, Teran and Selle 2014 split the fixed
//! corotated energy into a shear part taken on the isochoric deformation
//! and a volume part (their eqs. 7 and 8): `psi = mu |F^ - R^|^2 + lambda
//! / 2 (J - 1)^2` with `F^ = J^(-1/d) F` and `R^` its rotation. The volume
//! part is the compressible projection's (`pressure::add_compressibility`,
//! `c^2 = lambda / rho`); this module gives the shear part's Kirchhoff
//! stress, which the faces receive as forces
//! (`transfer::stress_to_faces`).
//!
//! The stress of an energy of the isochoric deformation alone is the
//! deviatoric part of `dpsi/dF^ F^T` (the chain rule through `F^ = J^(-1/d)
//! F`), with `dpsi/dF^ = 2 mu (F^ - R^)` for the corotated energy.

use glam::{Mat2, Vec2};

/// Rotation of the polar decomposition of a 2D matrix with positive
/// determinant: the angle that maximises `tr(R^T F)`.
pub fn rotation(f: Mat2) -> Mat2 {
    let angle = (f.x_axis.y - f.y_axis.x).atan2(f.x_axis.x + f.y_axis.y);
    let (s, c) = angle.sin_cos();
    Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
}

/// `F^ = J^(-1/2) F`, the isochoric part in 2D.
fn isochoric(f: Mat2) -> Mat2 {
    f * f.determinant().max(f32::MIN_POSITIVE).sqrt().recip()
}

/// Deviatoric Kirchhoff stress of the shear energy, `dev(2 mu (F^ - R^)
/// F^^T)`, in the units of `mu` (per unit density on the gate scenes).
pub fn shear_kirchhoff(f: Mat2, mu: f32) -> Mat2 {
    let fh = isochoric(f);
    let tau = 2.0 * mu * (fh - rotation(fh)) * fh.transpose();
    let mean = 0.5 * (tau.x_axis.x + tau.y_axis.y);
    tau - Mat2::from_diagonal(Vec2::splat(mean))
}

/// Shear energy density `mu |F^ - R^|^2`.
pub fn shear_energy(f: Mat2, mu: f32) -> f32 {
    let fh = isochoric(f);
    let d = fh - rotation(fh);
    mu * (d.x_axis.length_squared() + d.y_axis.length_squared())
}

/// The deformation gradient's step over a substep with velocity gradient
/// `C`: `F <- R(dt C) F`, Stomakhin et al. 2014 section 5.9. The exact step
/// for a gradient held over the substep is `exp(dt C)`, whose determinant
/// `exp(dt tr C)` is always positive; `I + dt C`, its first-order
/// truncation, inverts `F` once `det(I + dt C) <= 0`. Measured in step A2:
/// at a block's impact on a wall, `|C|` reached 932 1/s at `dt` 1 ms, `det
/// F` went to -0.2 and then NaN. Their compromise, `R(M) = I + M` where
/// `det(I + M) > 0` and `R(M) = R(M / 2)^2` otherwise, is a truncated
/// exponential that keeps the determinant positive and costs nothing more
/// where the plain step is safe. Coarse where it acts: where the plain
/// step's determinant is negative it can be several times off the
/// exponential's (unit test); it guarantees the sign, not the accuracy.
pub fn deformation_step(m: Mat2) -> Mat2 {
    let plain = Mat2::IDENTITY + m;
    // A non-finite `M` never halves into a positive determinant: return
    // it as it is, so the NaN shows where it arose instead of recursing.
    if plain.determinant() > 0.0 || !m.is_finite() {
        plain
    } else {
        let half = deformation_step(0.5 * m);
        half * half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rotation stores no shear energy and no stress; nor does a pure
    /// volume change, which belongs to the projection.
    #[test]
    fn rotations_and_volume_changes_are_stress_free() {
        let r = rotation(Mat2::from_angle(0.7));
        let scaled = 1.3 * Mat2::from_angle(-0.4);
        for f in [r, scaled, Mat2::IDENTITY] {
            assert!(shear_energy(f, 5.0) < 1e-10, "{f:?}");
            let tau = shear_kirchhoff(f, 5.0);
            assert!(tau.x_axis.length() + tau.y_axis.length() < 1e-5, "{tau:?}");
        }
    }

    /// The step is `I + M` wherever that keeps the determinant positive,
    /// and positive everywhere else, close to the exponential.
    #[test]
    fn the_deformation_step_never_inverts() {
        let safe = Mat2::from_cols(Vec2::new(0.1, 0.2), Vec2::new(-0.3, 0.05));
        assert_eq!(deformation_step(safe), Mat2::IDENTITY + safe);
        // Strong compression along y with shear: I + M has determinant
        // 1 * (1 - 1.5) - 0.4 * 0.0 < 0.
        let violent = Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.4, -1.5));
        assert!((Mat2::IDENTITY + violent).determinant() < 0.0);
        let step = deformation_step(violent);
        assert!(step.determinant() > 0.0);
        // The compromise is coarse: here det 1/16 where the exponential
        // gives exp(tr M) = exp(-1.5) = 0.22. It guarantees the sign only.
        assert!((step.determinant() - 0.0625).abs() < 1e-6);
    }

    /// Small simple shear `F = I + g e_x e_y^T`: Kirchhoff shear stress
    /// `mu g` off the diagonal, as linear elasticity gives (shear modulus
    /// `mu`).
    #[test]
    fn small_shear_gives_mu_times_the_shear() {
        let g = 1.0e-3;
        let f = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(g, 1.0));
        let tau = shear_kirchhoff(f, 4.0);
        assert!((tau.y_axis.x - 4.0 * g).abs() < 1e-5, "{tau:?}");
        assert!((tau.x_axis.y - 4.0 * g).abs() < 1e-5, "{tau:?}");
    }

    /// Small uniaxial strain `F = diag(1, 1 + e)`: the shear part carries
    /// `mu e` of the axial stress (`(lambda + mu) e` with the projection's
    /// `lambda e`), the 2D uniaxial stiffness the scene 8 criterion uses.
    #[test]
    fn uniaxial_strain_carries_mu_e_of_the_axial_stress() {
        let e = 1.0e-3;
        let f = Mat2::from_diagonal(Vec2::new(1.0, 1.0 + e));
        let tau = shear_kirchhoff(f, 3.0);
        assert!((tau.y_axis.y - 3.0 * e).abs() < 1e-5, "{tau:?}");
    }
}

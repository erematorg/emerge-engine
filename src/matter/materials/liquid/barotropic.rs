//! A barotropic liquid in plane strain: its stored energy depends on its
//! volume alone, `W = U(J)`, with no shear stiffness; what the precise
//! path's water is at small amplitude and without viscosity.
//!
//! Sources: the stresses of a volume-only energy are Connolly, Mackenzie and
//! Gorash, Comput. Mech. 64:1273-1288 (2019) eqs. 13, 15 and 27 (read in
//! full): `S = J U'(J) C^-1`, `tau = J U'(J) I`, `C_vol = J (U' + U'' J)
//! C^-1 (x) C^-1 - 2 U' J C^-1 (.) C^-1`. The energy is quadratic, `U = K/2
//! (J - 1)^2`, `K` the liquid's measured modulus, so `J U' = K J (J - 1)` and
//! `J (U' + U'' J) = K J (2 J - 1)`. Its validity for water is measured in
//! `liquid_water` (`QUADRATIC_VALID_ABOVE_P0_PA`: within 1 % up to 9.6 MPa
//! above 0.1 MPa).
//!
//! Declared: the pressure is the gauge pressure about the state the modulus
//! was measured at (0.1 MPa for `LiquidWater`): the ambient pressure, uniform,
//! is not carried (no atmosphere in the engine). Plane strain, `F_33 = 1`:
//! `J = sqrt(det C)` of the in-plane block.

use glam::{DMat2, DVec2};

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

/// The stored energy per reference volume, `K/2 (J - 1)^2`, from `C`;
/// infinite where `det C <= 0`. The tests' reference: the path takes it from
/// `F` (`barotropic_energy_of_f`), without the cancellation of `sqrt(det C)
/// - 1`.
#[cfg(test)]
pub(crate) fn barotropic_energy(c: DMat2, bulk: f64) -> f64 {
    let det = c.determinant();
    if det <= 0.0 {
        return f64::INFINITY;
    }
    0.5 * bulk * (det.sqrt() - 1.0).powi(2)
}

/// The same energy from `F`, its round-off relative to the energy: `J -
/// 1 = (F_11 - 1) + (F_22 - 1) + (F_11 - 1)(F_22 - 1) - F_12 F_21`, each
/// `F_ii - 1` exact near 1 (Sterbenz), where `sqrt(det C) - 1` loses
/// `eps / |J - 1|` of it to cancellation (measured: 3e-11 of a ringing
/// block's energy at strains of 1e-5).
pub(crate) fn barotropic_energy_of_f(f: DMat2, bulk: f64) -> f64 {
    if f.determinant() <= 0.0 {
        return f64::INFINITY;
    }
    let (a, d) = (f.x_axis.x - 1.0, f.y_axis.y - 1.0);
    let j_minus_one = a + d + a * d - f.y_axis.x * f.x_axis.y;
    0.5 * bulk * j_minus_one * j_minus_one
}

/// The second Piola-Kirchhoff stress `S = K J (J - 1) C^-1`.
pub(crate) fn barotropic_second_pk(c: DMat2, bulk: f64) -> DMat2 {
    let j = c.determinant().sqrt();
    c.inverse() * (bulk * j * (j - 1.0))
}

/// The Kirchhoff stress `tau = K J (J - 1) I`, `J = det F`.
pub(crate) fn barotropic_kirchhoff(f: DMat2, bulk: f64) -> DMat2 {
    let j = f.determinant();
    DMat2::from_diagonal(DVec2::splat(bulk * j * (j - 1.0)))
}

/// `dS` along a symmetric `dC`, half the tangent of eq. 27 applied to it:
/// `K J (2 J - 1) / 2 (C^-1 : dC) C^-1 - K J (J - 1) C^-1 dC C^-1` (derived
/// on the page from `S`: `dJ = J/2 C^-1 : dC`, `dC^-1 = -C^-1 dC C^-1`).
pub(crate) fn barotropic_second_pk_tangent(c: DMat2, dc: DMat2, bulk: f64) -> DMat2 {
    let inverse = c.inverse();
    let j = c.determinant().sqrt();
    inverse * (0.5 * bulk * j * (2.0 * j - 1.0) * colon(inverse, dc))
        - (inverse * dc * inverse) * (bulk * j * (j - 1.0))
}

/// `U(J_1) - U(J_0) = K/2 (J_1 - J_0)(J_1 + J_0 - 2)` from the increment, so
/// its round-off is relative to the change: `J_1 - J_0 = (det_1 - det_0) /
/// (J_1 + J_0)`, the determinant's increment exact for 2 x 2 matrices, `d det
/// = tr(adj(C_0) dC) + det dC` (derived on the page).
pub(crate) fn barotropic_energy_increment(c0: DMat2, c1: DMat2, bulk: f64) -> f64 {
    let dc = c1 - c0;
    let adjugate_dot =
        c0.y_axis.y * dc.x_axis.x - c0.y_axis.x * dc.x_axis.y - c0.x_axis.y * dc.y_axis.x
            + c0.x_axis.x * dc.y_axis.y;
    let d_det = adjugate_dot + dc.determinant();
    let (j0, j1) = (c0.determinant().sqrt(), c1.determinant().sqrt());
    let dj = d_det / (j0 + j1);
    0.5 * bulk * dj * (j0 + j1 - 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BULK: f64 = 2.2e9;

    fn states() -> [DMat2; 3] {
        let f = |a: f64, b: f64, c: f64, d: f64| {
            let f = DMat2::from_cols(DVec2::new(a, b), DVec2::new(c, d));
            f.transpose() * f
        };
        [
            f(1.002, 0.001, -0.0005, 0.997),
            f(0.995, 0.02, 0.01, 1.001),
            DMat2::IDENTITY,
        ]
    }

    fn directions() -> [DMat2; 3] {
        [
            DMat2::from_cols(DVec2::X, DVec2::ZERO),
            DMat2::from_cols(DVec2::ZERO, DVec2::Y),
            DMat2::from_cols(DVec2::Y, DVec2::X),
        ]
    }

    fn size(a: DMat2) -> f64 {
        colon(a, a).sqrt()
    }

    /// `S = 2 dW/dC` and the tangent against central differences of `S`,
    /// to 1e-7 of their size plus the differences' floor (criteria fixed
    /// before running).
    #[test]
    fn the_stress_and_tangent_are_the_energy_derivatives() {
        for c in states() {
            let s = barotropic_second_pk(c, BULK);
            let h = 1e-7;
            for e in directions() {
                let d = (barotropic_energy(c + e * h, BULK) - barotropic_energy(c - e * h, BULK))
                    / (2.0 * h);
                assert!(
                    (d - 0.5 * colon(s, e)).abs() <= 1e-7 * size(s).max(1e-3 * BULK),
                    "{d} against {}",
                    0.5 * colon(s, e)
                );
                let fd = (barotropic_second_pk(c + e * h, BULK)
                    - barotropic_second_pk(c - e * h, BULK))
                    * (0.5 / h);
                let exact = barotropic_second_pk_tangent(c, e, BULK);
                assert!(size(fd - exact) <= 1e-6 * size(exact).max(BULK));
            }
        }
    }

    /// The exact increment against the energies' difference at a large step,
    /// and against the midpoint work at tiny ones (to `O(|dC|^3)` and the
    /// midpoint stress's round-off floor near `J = 1`, `16 eps K |dC|`).
    #[test]
    fn the_increment_is_the_energy_change_at_any_size() {
        let dir = DMat2::from_cols(DVec2::new(0.3, -0.2), DVec2::new(-0.2, 0.5));
        for c0 in states() {
            let large = c0 + dir * 0.01;
            let inc = barotropic_energy_increment(c0, large, BULK);
            let diff = barotropic_energy(large, BULK) - barotropic_energy(c0, BULK);
            assert!(
                (inc - diff).abs() <= 1e-12 * diff.abs().max(1.0),
                "{inc} vs {diff}"
            );
            for step in [1e-6, 1e-9, 1e-12] {
                let c1 = c0 + dir * step;
                let inc = barotropic_energy_increment(c0, c1, BULK);
                let mid = barotropic_second_pk(0.5 * (c0 + c1), BULK);
                let work = 0.5 * colon(mid, c1 - c0);
                let floor = 16.0 * f64::EPSILON * BULK * size(c1 - c0);
                assert!(
                    (inc - work).abs() <= 1e-10 * work.abs() + BULK * step.powi(3) + floor,
                    "step {step}: {inc:e} against {work:e}"
                );
            }
        }
    }

    /// The energy from `F` is the energy from `C`, and near `J = 1` it
    /// keeps its relative accuracy: at `F = diag(1 + d, 1)`, `K/2 d^2` to
    /// 4 eps where the `C` form is off by about `eps / d` (criterion fixed
    /// before running).
    #[test]
    fn the_energy_from_f_keeps_its_accuracy_near_the_rest_state() {
        let f = DMat2::from_cols(DVec2::new(1.002, 0.001), DVec2::new(-0.0005, 0.997));
        let (from_f, from_c) = (
            barotropic_energy_of_f(f, BULK),
            barotropic_energy(f.transpose() * f, BULK),
        );
        assert!(
            (from_f - from_c).abs() <= 1e-10 * from_c,
            "{from_f} vs {from_c}"
        );
        for d in [1e-5, 1e-7, 1e-9] {
            let f = DMat2::from_diagonal(DVec2::new(1.0 + d, 1.0));
            let exact = 0.5 * BULK * (f.x_axis.x - 1.0).powi(2);
            let got = barotropic_energy_of_f(f, BULK);
            assert!(
                (got - exact).abs() <= 4.0 * f64::EPSILON * exact,
                "d {d}: {got:e} against {exact:e}"
            );
        }
    }

    /// A uniaxial plane-strain stretch `F = diag(a, 1)`: `J = a`, the nominal
    /// stress `P_11 = K (a - 1)` and `P_22 = K a (a - 1)`: a pressure only, its
    /// small-amplitude modulus `K` (the P-wave modulus of a medium with no
    /// shear stiffness).
    #[test]
    fn a_uniaxial_stretch_carries_the_pressure_alone() {
        for a in [0.99, 1.0, 1.01] {
            let f = DMat2::from_diagonal(DVec2::new(a, 1.0));
            let p = f * barotropic_second_pk(f.transpose() * f, BULK);
            assert!((p.x_axis.x - BULK * (a - 1.0)).abs() <= 1e-15 * BULK);
            assert!((p.y_axis.y - BULK * a * (a - 1.0)).abs() <= 1e-15 * BULK);
            assert_eq!(p.x_axis.y, 0.0);
        }
    }
}

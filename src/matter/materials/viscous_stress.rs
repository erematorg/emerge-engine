//! The Newtonian viscous stress: how a material resists its own rate of
//! deformation, the dashpot of a Kelvin-Voigt solid and the viscous part of
//! a fluid's stress.
//!
//! Source: the linear isotropic rate law `sigma_v = 2 eta dev(D) + zeta
//! tr(D) I` (3D, `dev(D) = D - tr(D)/3 I`), shear viscosity `eta` and bulk
//! viscosity `zeta` in Pa s; in its Lagrangian form on the Green strain rate,
//! `S_D = phi tr(E_dot) I + 2 psi E_dot` (Barbic, CMU-CS-07-145, eq. 2.34),
//! the same law with `psi = eta`, `phi = zeta - 2 eta / 3`.
//! Adaptation, declared: plane strain, the engine's 2D. With `D_zz = 0`
//! the in-plane stress is `2 eta (D - tr(D)/3 I) + zeta tr(D) I` (the 3D
//! `tr/3`, not a 2D `tr/2`: `sigma_zz` exists and is not needed here), and
//! the dissipation rate `sigma_v : D = 2 eta (|D|^2 - tr(D)^2 / 3) + zeta
//! tr(D)^2 >= 0` for `eta, zeta >= 0` (`|D|^2 >= tr(D)^2 / 2` in 2D).

use crate::tensor_algebra::matrix2::Matrix2;

/// The trace of `m`.
fn trace<M: Matrix2>(m: M) -> M::Scalar {
    let [[a, _], [_, d]] = m.columns();
    a + d
}

/// The viscous stress of the symmetric rate `rate` (`D`, or `E_dot` in the
/// Lagrangian form), shear viscosity `eta`, bulk viscosity `zeta`, in the
/// plane-strain form of the module doc. One definition at any precision:
/// `f32` for the explicit solver, `f64` for the total Lagrangian step.
#[inline]
pub(crate) fn newtonian_viscous_stress<M: Matrix2>(rate: M, eta: M::Scalar, zeta: M::Scalar) -> M {
    let third = M::Scalar::from(1.0) / M::Scalar::from(3.0);
    let tr = trace(rate);
    (rate - M::IDENTITY.scaled(tr * third)).scaled(M::Scalar::from(2.0) * eta)
        + M::IDENTITY.scaled(zeta * tr)
}

/// The dissipation rate `sigma_v : rate` of `newtonian_viscous_stress`,
/// `2 eta (|rate|^2 - tr(rate)^2 / 3) + zeta tr(rate)^2`, never negative.
#[inline]
pub(crate) fn newtonian_dissipation<M: Matrix2>(
    rate: M,
    eta: M::Scalar,
    zeta: M::Scalar,
) -> M::Scalar {
    let [[a, b], [c, d]] = rate.columns();
    let squared = a * a + b * b + c * c + d * d;
    let tr = a + d;
    let third = M::Scalar::from(1.0) / M::Scalar::from(3.0);
    M::Scalar::from(2.0) * eta * (squared - tr * tr * third) + zeta * tr * tr
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat2, DVec2};

    fn contract(a: DMat2, b: DMat2) -> f64 {
        a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
    }

    /// The dissipation is the stress's work on its own rate, and never
    /// negative, for a shear, a compression and a mixed rate.
    #[test]
    fn dissipation_is_the_stress_work_and_never_negative() {
        let (eta, zeta) = (2.0, 0.7);
        for rate in [
            DMat2::from_cols(DVec2::new(0.0, 1.5), DVec2::new(1.5, 0.0)),
            DMat2::from_cols(DVec2::new(-2.0, 0.0), DVec2::new(0.0, -2.0)),
            DMat2::from_cols(DVec2::new(0.3, -0.4), DVec2::new(-0.4, 1.1)),
        ] {
            let work = contract(newtonian_viscous_stress(rate, eta, zeta), rate);
            let d = newtonian_dissipation(rate, eta, zeta);
            assert!((work - d).abs() < 1e-12 * d.abs().max(1.0), "{work} vs {d}");
            assert!(d >= 0.0);
        }
    }

    /// Uniaxial strain rate with `zeta = 2 eta / 3`: no lateral stress, an
    /// axial one of `2 eta` times the rate (the plane-strain bar the total
    /// Lagrangian path's damping test is built on).
    #[test]
    fn uniaxial_rate_with_zeta_two_thirds_eta_has_no_lateral_stress() {
        let eta = 1.3;
        let rate = DMat2::from_cols(DVec2::new(0.5, 0.0), DVec2::new(0.0, 0.0));
        let s = newtonian_viscous_stress(rate, eta, 2.0 * eta / 3.0);
        assert!(s.y_axis.y.abs() < 1e-15);
        assert!((s.x_axis.x - 2.0 * eta * 0.5).abs() < 1e-15);
    }
}

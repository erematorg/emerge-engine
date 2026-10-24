//! The deformation gradient `F` and how it follows the motion.
//!
//! Continuum kinematics: `dF/dt = L F`, with `L` the velocity gradient. For `L`
//! held constant over a step the exact solution is `F(t+dt) = exp(dt L) F(t)`.
//!
//! Source: Gurtin, Fried & Anand, *The Mechanics and Thermodynamics of
//! Continua* (2010), ch. 6 (the deformation gradient) and ch. 14 (motions with
//! constant velocity gradient).

use glam::{Mat2, Vec2};

/// Below this `|delta^2|` (a stretch or rotation under one radian per
/// substep, every substep the CFL bounds allow) the even and odd factors of
/// the exponential come from their Taylor series to `x^5`. The first
/// omitted terms, `x^6 / 12!` and `x^6 / 13!`, are then under 5e-9 of the
/// leading ones, below f32 resolution.
///
/// The series is used this far, rather than only near zero, because the
/// GPU copy cannot rely on its built-ins for small arguments: WGSL (W3C,
/// section 15.7.4 Floating Point Accuracy) gives `sinh` only the accuracy
/// of `(exp(x) - exp(-x)) * 0.5` and allows `sin` an absolute error of
/// 2^-11 on [-pi, pi], either of which can be the whole of a substep's
/// value. Both solvers use the same series so they agree.
const INCREMENT_SERIES_LIMIT: f32 = 1.0;

/// `exp(A) - I` for a 2x2 `A`, never formed as one plus a small number:
/// the exact deformation-gradient increment for a velocity gradient held
/// constant over one substep, minus the identity.
///
/// Continuum kinematics gives `dF/dt = L F`.  The exact update for constant
/// `L` is therefore `F(t+dt) = exp(dt L) F(t)`.  The commonly used forward-
/// Euler approximation `(I + dt L) F` has a systematic ratchet: two equal
/// and opposite diagonal rates multiply to `(1+a)(1-a)=1-a^2`, so a
/// perfectly reversible oscillation loses volume every cycle.  That error is
/// especially visible in a tension-only material because its slack direction
/// has no constitutive restoring force.
///
/// This uses the closed-form exponential of a real 2x2 matrix obtained from
/// Cayley-Hamilton.  The trigonometric branch covers complex-conjugate
/// eigenvalues (including rigid rotation); the hyperbolic branch covers real
/// eigenvalues.  Series expansions remove the removable singularity at zero.
///
/// A substep's `exp(dt L)` is within a few parts in 1e4 of the identity.
/// An f32 near 1 is spaced 1.19e-7 above it and 5.96e-8 below it, so
/// writing the increment as `1 + small` rounds the small part onto steps of
/// two different sizes, and a factor that should be just above or just
/// below 1 is quantized differently in each direction. Multiplied into an
/// `F` whose entries straddle 1 (a loaded body: 1.0027 across, 0.9899 along
/// the load), the rounding stops averaging out and pushes one way, every
/// substep. Measured on a resting self-weight column (E = 1e5 Pa, earth
/// gravity, 240 particles), with each particle's `F` re-integrated in f64
/// from the solver's own velocity gradients: the former update (this
/// increment formed as `1 + small`, then rescaled onto a carried volume)
/// drifted the mean `F_xx` by -3.6e-4 and `F_yy` by +3.0e-4 in 10 s, so `F`
/// read the body as less deformed than it was, while this form, applied as
/// `F + (exp(dt L) - I) F`, stayed within 5e-7 of the f64 integral.
/// Equilibrium holds `F` at the load, so in the solver that drift showed up
/// as the body's shape creeping instead (shorter and wider, without end, at
/// a rate proportional to the number of substeps).
///
/// Every term here stays small: `exp_m1` for the trace part and the
/// half-angle form `cosh x - 1 = 2 sinh^2(x / 2)` for the even part.
#[inline]
fn deformation_increment_exp_minus_identity(dt_velocity_gradient: Mat2) -> Mat2 {
    // glam is column-major: [[a,b],[c,d]] is stored as columns (a,c),(b,d).
    let a = dt_velocity_gradient.x_axis.x;
    let b = dt_velocity_gradient.y_axis.x;
    let c = dt_velocity_gradient.x_axis.y;
    let d = dt_velocity_gradient.y_axis.y;
    let half_trace = 0.5 * (a + d);
    let half_difference = 0.5 * (a - d);
    let delta_sq = half_difference * half_difference + b * c;

    // cosh(sqrt(x)) - 1 and sinh(sqrt(x))/sqrt(x), continued analytically
    // through x = 0 (cos and sin of sqrt(-x) for x < 0).
    let (even_minus_one, odd) = if delta_sq.abs() < INCREMENT_SERIES_LIMIT {
        let x = delta_sq;
        (
            x * (1.0 / 2.0
                + x * (1.0 / 24.0 + x * (1.0 / 720.0 + x * (1.0 / 40320.0 + x / 3628800.0)))),
            1.0 + x
                * (1.0 / 6.0
                    + x * (1.0 / 120.0
                        + x * (1.0 / 5040.0 + x * (1.0 / 362880.0 + x / 39916800.0)))),
        )
    } else if delta_sq > 0.0 {
        let delta = delta_sq.sqrt();
        let half = (0.5 * delta).sinh();
        (2.0 * half * half, delta.sinh() / delta)
    } else {
        let omega = (-delta_sq).sqrt();
        let half = (0.5 * omega).sin();
        (-2.0 * half * half, omega.sin() / omega)
    };

    // exp(A) = e^h (cosh I + odd (A - h I)) with h = tr(A) / 2; writing
    // s = e^h - 1 and e = cosh - 1 gives
    // exp(A) - I = (e + s (1 + e)) I + (1 + s) odd (A - h I).
    let scale_minus_one = half_trace.exp_m1();
    let traceless = dt_velocity_gradient - Mat2::from_diagonal(Vec2::splat(half_trace));
    Mat2::from_diagonal(Vec2::splat(
        even_minus_one + scale_minus_one * (1.0 + even_minus_one),
    )) + traceless * ((1.0 + scale_minus_one) * odd)
}

/// One substep of `dF/dt = L F`, as `F + (exp(dt L) - I) F`.
///
/// Applying the increment as a correction to `F`, rather than multiplying
/// by a matrix that sits on the f32 grid around 1, keeps each rounding
/// relative to the small part (see `deformation_increment_exp_minus_identity`),
/// so what rounding is left averages out instead of pushing one way.
///
/// The form this replaces multiplied by `exp(dt L)` and then rescaled the
/// product every substep onto a volume ratio carried beside `F`, to undo
/// the steady loss of determinant that product had. The rescale factor
/// was itself within a few ULP of 1, so it fed the same one-way rounding
/// back into the shape, and once the carried ratio and `det(F)` drifted
/// more than 1e-3 apart it stopped rescaling for good. Measured on one
/// particle driven by a prescribed oscillation, no solver
/// (`tests/probes/f_rounding_horizon.rs`, ROUND_FXX=1.0027
/// ROUND_FYY=0.9899, ROUND_AMP=0.4, 900 000 steps of 2.94e-4 s), error
/// against the f64 integral: the rescaled form held `ln det F` to -1.2e-7
/// but put -5.6e-3 into `ln(F_xx / F_yy)`; this form leaves +7.9e-5 and
/// +6.3e-5, about what an unbiased walk of f32 roundings reaches in that
/// many steps. On the anchored tension-only body of
/// `tests/probes/no_compression_drift_horizon.rs` (one 4.37 ms substep per
/// step), where shape error is what turned into volume, `max |J - 1|` after
/// 900 000 substeps goes from 0.131 to 0.0011.
///
/// The volume ratio is `det(F)`, as in any MPM: with no bias left to
/// cancel, nothing needs a second copy kept in step with it.
#[inline]
pub(crate) fn advance_deformation_gradient(f_old: Mat2, dt_velocity_gradient: Mat2) -> Mat2 {
    f_old + deformation_increment_exp_minus_identity(dt_velocity_gradient) * f_old
}

#[cfg(test)]
mod deformation_increment_tests {
    use super::*;

    fn deformation_increment_exp(a: Mat2) -> Mat2 {
        Mat2::IDENTITY + deformation_increment_exp_minus_identity(a)
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

    #[test]
    fn opposite_rates_are_exactly_reversible_without_volume_ratchet() {
        let a = Mat2::from_diagonal(Vec2::new(0.02, -0.01));
        let forward = deformation_increment_exp(a);
        let backward = deformation_increment_exp(-a);
        let round_trip = backward * forward;
        assert!(
            matrix_error(round_trip, Mat2::IDENTITY) < 1.0e-6,
            "exp(-A)exp(A) must be identity: {round_trip:?}"
        );
    }

    #[test]
    fn rigid_rotation_preserves_volume_and_matches_angle() {
        let angle = 0.37;
        let spin = Mat2::from_cols(Vec2::new(0.0, angle), Vec2::new(-angle, 0.0));
        let increment = deformation_increment_exp(spin);
        let expected = Mat2::from_angle(angle);
        assert!(
            matrix_error(increment, expected) < 1.0e-6,
            "matrix exponential of planar spin must be the matching rotation: \
             expected={expected:?} got={increment:?}"
        );
        assert!((increment.determinant() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn determinant_obeys_exact_continuity_identity() {
        let a = Mat2::from_cols(Vec2::new(0.11, -0.07), Vec2::new(0.13, -0.03));
        let increment = deformation_increment_exp(a);
        let expected_det = (a.x_axis.x + a.y_axis.y).exp();
        assert!(
            (increment.determinant() - expected_det).abs() < 2.0e-6,
            "det(exp(A)) must equal exp(trace(A)): expected={expected_det} got={}",
            increment.determinant()
        );
    }

    /// The same closed form in f64, row-major `[m00, m01, m10, m11]`.
    fn exp_f64(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
        let half_trace = 0.5 * (a + d);
        let half_difference = 0.5 * (a - d);
        let delta_sq = half_difference * half_difference + b * c;
        let (even, odd) = if delta_sq == 0.0 {
            (1.0, 1.0)
        } else if delta_sq > 0.0 {
            let delta = delta_sq.sqrt();
            (delta.cosh(), delta.sinh() / delta)
        } else {
            let omega = (-delta_sq).sqrt();
            (omega.cos(), omega.sin() / omega)
        };
        let scale = half_trace.exp();
        [
            scale * (even + (a - half_trace) * odd),
            scale * b * odd,
            scale * c * odd,
            scale * (even + (d - half_trace) * odd),
        ]
    }

    /// 200 000 substeps of an oscillating velocity gradient applied to a
    /// loaded `F` (1.0027 across, 0.9899 along the load, entries on both
    /// sides of 1), against the same exponential integrated in f64. The
    /// former update (the increment formed as `1 + small`, then rescaled
    /// onto a carried volume) ended this at -4.7e-3 (0.04 / s), -1.4e-3
    /// (0.4 / s) and -4.6e-4 (4 / s) in `ln(F_xx / F_yy)`, and -2.5e-3 in
    /// `ln det F` at 4 / s: a one-way drift. What this form leaves is the
    /// size of an unbiased walk of f32 roundings over that many steps.
    #[test]
    fn repeated_substeps_do_not_bias_a_loaded_shape_or_volume() {
        let dt = 2.94e-4_f32;
        let omega = std::f32::consts::TAU / 200.0;
        for amplitude in [0.04_f32, 0.4, 4.0] {
            let mut f = Mat2::from_diagonal(Vec2::new(1.0027, 0.9899));
            let mut exact = [f64::from(f.x_axis.x), 0.0, 0.0, f64::from(f.y_axis.y)];
            for step in 1..=200_000 {
                let phase = omega * step as f32;
                let c = Mat2::from_cols_array(&[
                    amplitude * phase.cos(),
                    amplitude * (phase * 0.37).sin(),
                    amplitude * (phase * 0.61).cos(),
                    -amplitude * phase.cos() + 0.3 * amplitude * (phase * 1.3).sin(),
                ]);
                let dc = dt * c;
                let m = exp_f64(
                    f64::from(dc.x_axis.x),
                    f64::from(dc.y_axis.x),
                    f64::from(dc.x_axis.y),
                    f64::from(dc.y_axis.y),
                );
                exact = [
                    m[0] * exact[0] + m[1] * exact[2],
                    m[0] * exact[1] + m[1] * exact[3],
                    m[2] * exact[0] + m[3] * exact[2],
                    m[2] * exact[1] + m[3] * exact[3],
                ];
                f = advance_deformation_gradient(f, dc);
            }
            let det_error = (f64::from(f.x_axis.x) * f64::from(f.y_axis.y)
                - f64::from(f.y_axis.x) * f64::from(f.x_axis.y))
            .ln()
                - (exact[0] * exact[3] - exact[1] * exact[2]).ln();
            let shape_error =
                (f64::from(f.x_axis.x) / f64::from(f.y_axis.y)).ln() - (exact[0] / exact[3]).ln();
            println!(
                "amplitude {amplitude}/s: ln det error {det_error:+.2e}, ln(F_xx/F_yy) error {shape_error:+.2e}"
            );
            assert!(
                det_error.abs() < 3.0e-4 && shape_error.abs() < 3.0e-4,
                "200 000 substeps at {amplitude}/s must not drift F one way: \
                 ln det error {det_error:+.2e}, ln(F_xx/F_yy) error {shape_error:+.2e}"
            );
        }
    }
}

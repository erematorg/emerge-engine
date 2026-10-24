use glam::{IVec2, Vec2};

#[derive(Clone, Copy, Debug)]
pub struct QuadraticWeights {
    pub base_cell: IVec2,
    pub wx: [f32; 3],
    pub wy: [f32; 3],
}

pub fn quadratic_weights(position: Vec2) -> QuadraticWeights {
    let base_cell = position.floor().as_ivec2();
    let diff = position - base_cell.as_vec2() - Vec2::splat(0.5);
    QuadraticWeights {
        base_cell,
        wx: axis_weights(diff.x),
        wy: axis_weights(diff.y),
    }
}

pub fn axis_weights(d: f32) -> [f32; 3] {
    let w0 = 0.5 * (0.5 - d).powi(2);
    let w1 = 0.75 - d.powi(2);
    let w2 = 0.5 * (0.5 + d).powi(2);
    [w0, w1, w2]
}

/// Analytic derivative `[dw0/dd, dw1/dd, dw2/dd]` of `axis_weights` w.r.t.
/// the fractional offset `d` -- the base building block for differentiating
/// through the kernel's own dependence on particle position, the last
/// confirmed-real gap in a fully general P2G/G2P adjoint (see
/// `p2g_stress_vjp`'s and `g2p_velocity_vjp`'s docs in `spacetime::transfer`,
/// both scoped to hold position fixed for exactly this reason).
///
/// Plain calculus on `axis_weights`'s own three closed-form quadratics:
///   dw0/dd = d(0.5*(0.5-d)^2)/dd = -(0.5-d)  = d - 0.5
///   dw1/dd = d(0.75-d^2)/dd      = -2d
///   dw2/dd = d(0.5*(0.5+d)^2)/dd = (0.5+d)
///
/// Note: `d = position - floor(position) - 0.5` in `quadratic_weights`, and
/// `floor` has zero derivative almost everywhere (only undefined at integer
/// cell boundaries, a measure-zero set -- the standard treatment, same as
/// how ReLU's kink is handled in ML autodiff), so `dd/d(position) = 1` a.e.
/// -- these ARE the weights' derivative w.r.t. the particle's own position
/// along this axis, not just w.r.t. `d` in the abstract.
///
/// Verified against central-difference numerical gradients in this module's
/// own tests. Does NOT yet wire into a full P2G/G2P position adjoint --
/// combining this with `cell_dist`'s own `-1` derivative and the product
/// rule across every scattered term (mass, velocity, C, stress) is real,
/// separate, still-open work.
pub fn axis_weights_derivative(d: f32) -> [f32; 3] {
    [d - 0.5, -2.0 * d, 0.5 + d]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_weights_sum_to_one() {
        for sample in [0.0f32, 0.1, 0.35, -0.2] {
            let ws = axis_weights(sample);
            let sum = ws[0] + ws[1] + ws[2];
            assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        }
    }

    #[test]
    fn axis_weights_derivative_matches_finite_difference() {
        let h = 1.0e-3_f32;
        for d in [0.0f32, 0.1, 0.35, -0.2, 0.49, -0.49] {
            let analytic = axis_weights_derivative(d);
            let w_plus = axis_weights(d + h);
            let w_minus = axis_weights(d - h);
            for i in 0..3 {
                let numeric = (w_plus[i] - w_minus[i]) / (2.0 * h);
                let diff = (numeric - analytic[i]).abs();
                assert!(
                    diff < 1.0e-3,
                    "axis_weights_derivative mismatch at d={d}, component {i}: \
                     analytic={:.6} numeric={numeric:.6} diff={diff:.2e}",
                    analytic[i]
                );
            }
        }
    }

    /// Independent of the analytic formula: `axis_weights` always sums to 1
    /// (checked above), so its derivative must always sum to 0, an algebraic
    /// constraint rather than another finite-difference check.
    #[test]
    fn axis_weights_derivative_sums_to_zero() {
        for d in [0.0f32, 0.1, 0.35, -0.2, 0.49, -0.49] {
            let dw = axis_weights_derivative(d);
            let sum = dw[0] + dw[1] + dw[2];
            assert!(sum.abs() < 1e-5, "d={d}: derivative sum={sum}, expected 0");
        }
    }
}

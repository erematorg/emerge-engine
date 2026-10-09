//! Time-integration pieces shared by the sub-solvers that advance their own
//! points (rods, grains).

use glam::Vec2;

/// `x += step` with compensated (Kahan 1965) summation: `compensation`
/// carries the part of each step f32 rounded away and adds it back on the
/// next. A stiff sub-solver's stable step makes each increment small next to
/// `x`'s grid coordinate, and a plain `+=` drops any increment under half
/// the f32 spacing there. Measured twice: an 81-point rod cantilever at 1 cm
/// cells, advanced plainly, froze every point within 5 s while its first
/// mode still rang at 5 mm (`tests/subsystem_time_steps.rs`,
/// `probe_cantilever_reference_absorption`); and two colliding grains
/// stopped converging as their step shrank (`tests/grains_pi_collisions.rs`).
pub(crate) fn advance_position(x: &mut Vec2, compensation: &mut Vec2, step: Vec2) {
    let y = step - *compensation;
    let t = *x + y;
    *compensation = (t - *x) - y;
    *x = t;
}

#[cfg(test)]
mod tests {
    use super::advance_position;
    use glam::Vec2;

    /// Half the spacing of f32 values around 40, where this test sits.
    fn half_ulp_at_40() -> f32 {
        0.5 * (40.0f32.next_up() - 40.0)
    }

    #[test]
    fn increments_below_half_an_ulp_add_up_when_compensated() {
        let start = Vec2::splat(40.0);
        let step = Vec2::splat(0.2 * half_ulp_at_40());
        let (mut plain, mut compensated, mut residual) = (start, start, Vec2::ZERO);
        let count = 100_000;
        for _ in 0..count {
            plain += step;
            advance_position(&mut compensated, &mut residual, step);
        }
        // The premise: plain f32 addition never moves.
        assert_eq!(plain, start);
        let expected = count as f32 * step.x;
        let moved = compensated - start;
        assert!(
            (moved.x - expected).abs() < 1.0e-3 * expected
                && (moved.y - expected).abs() < 1.0e-3 * expected,
            "moved {moved:?}, expected {expected} on each axis"
        );
    }
}

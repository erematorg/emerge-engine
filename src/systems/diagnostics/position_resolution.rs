//! How much of a body's motion over one substep survives f32 rounding.
//!
//! Positions advance by `x += v dt` in f32, which rounds to the nearest
//! representable value. When an increment is below half the spacing of
//! f32 values around `x` (half an ulp), that axis does not move at all. On
//! a loaded rod scene (`tests/probes/rod_load_ab.rs`) particles moving at
//! 1e-3 cells/s under a 7.4e-5 s substep had increments 17 to 20 times below
//! half an ulp at y = 25 and never moved, while their velocity followed the
//! rod beneath them.

use glam::Vec2;

use super::plugin::DiagnosticsPlugin;
use super::snapshot::SimSnapshot;
use crate::particle::Particle;

/// What f32 rounding does to the moving bodies' increments this substep.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PositionResolution {
    /// Bodies with a nonzero velocity.
    pub moving: usize,
    /// Share of them that do not move at all: every axis they move along
    /// has an increment under half an ulp.
    pub frozen: f32,
    /// Share of the total displacement (summed over axes and bodies) that
    /// is rounded away whole, from the axes whose increment is under half
    /// an ulp.
    pub displacement_lost: f32,
    /// Share of them with at least one axis under half an ulp, however
    /// small that axis's part of the motion.
    pub lost: f32,
    /// Share of them with at least one axis under 20 half-ulps: rounding
    /// changes that axis's step by more than 5 percent.
    pub coarse: f32,
}

/// Spacing of f32 values around `x`.
fn ulp(x: f32) -> f32 {
    let a = x.abs();
    a.next_up() - a
}

/// Measures `positions` advanced by `velocities` over `dt`.
pub fn position_resolution(
    positions: impl IntoIterator<Item = Vec2>,
    velocities: impl IntoIterator<Item = Vec2>,
    dt: f32,
) -> PositionResolution {
    // In half-ulps: see `PositionResolution::coarse`.
    const COARSE: f32 = 20.0;
    let (mut moving, mut frozen, mut lost, mut coarse) = (0usize, 0usize, 0usize, 0usize);
    let (mut total, mut erased) = (0.0f64, 0.0f64);
    for (x, v) in positions.into_iter().zip(velocities) {
        let d = v * dt;
        if d == Vec2::ZERO {
            continue;
        }
        moving += 1;
        let (mut smallest, mut largest) = (f32::INFINITY, 0.0f32);
        for (step, at) in [(d.x, x.x), (d.y, x.y)] {
            if step == 0.0 {
                continue;
            }
            let ratio = step.abs() / (0.5 * ulp(at));
            smallest = smallest.min(ratio);
            largest = largest.max(ratio);
            total += step.abs() as f64;
            if ratio < 1.0 {
                erased += step.abs() as f64;
            }
        }
        frozen += usize::from(largest < 1.0);
        lost += usize::from(smallest < 1.0);
        coarse += usize::from(smallest < COARSE);
    }
    let share = |n: usize| {
        if moving == 0 {
            0.0
        } else {
            n as f32 / moving as f32
        }
    };
    PositionResolution {
        moving,
        frozen: share(frozen),
        displacement_lost: if total > 0.0 {
            (erased / total) as f32
        } else {
            0.0
        },
        lost: share(lost),
        coarse: share(coarse),
    }
}

/// Logs `position_frozen`, `position_displacement_lost`,
/// `position_increment_lost` and `position_increment_coarse` for the
/// particles over the frame's mean substep (`configured_dt` over
/// `substeps_last_step`), not the last one, which is often the short
/// remainder that ends the frame.
pub struct PositionResolutionPlugin;

impl DiagnosticsPlugin for PositionResolutionPlugin {
    fn name(&self) -> &'static str {
        "position_resolution"
    }

    fn collect(&mut self, particles: &[Particle], snapshot: &SimSnapshot) -> Vec<(String, f32)> {
        let r = position_resolution(
            particles.iter().map(|p| p.x),
            particles.iter().map(|p| p.v),
            snapshot.configured_dt / snapshot.substeps_last_step.max(1) as f32,
        );
        vec![
            ("position_frozen".into(), r.frozen),
            ("position_displacement_lost".into(), r.displacement_lost),
            ("position_increment_lost".into(), r.lost),
            ("position_increment_coarse".into(), r.coarse),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_increment_under_half_an_ulp_does_not_move_the_position() {
        let x = Vec2::new(25.0, 25.0);
        let half = 0.5 * ulp(25.0);
        // Confirms the premise on f32 itself.
        assert_eq!(x.y + 0.5 * half, x.y);
        assert_ne!(x.y + 2.0 * half, x.y);
        let slow = position_resolution([x], [Vec2::new(0.0, -0.5 * half)], 1.0);
        let fast = position_resolution([x], [Vec2::new(0.0, -100.0 * half)], 1.0);
        let still = position_resolution([x], [Vec2::ZERO], 1.0);
        assert_eq!((slow.moving, slow.frozen, slow.lost), (1, 1.0, 1.0));
        assert_eq!(slow.displacement_lost, 1.0);
        assert_eq!(
            (fast.moving, fast.frozen, fast.lost, fast.coarse),
            (1, 0.0, 0.0, 0.0)
        );
        assert_eq!(fast.displacement_lost, 0.0);
        // Moving fast along y, barely along x: not frozen, x lost, and only
        // x's tiny part of the displacement rounded away.
        let mixed = position_resolution([x], [Vec2::new(0.5 * half, -100.0 * half)], 1.0);
        assert_eq!((mixed.frozen, mixed.lost), (0.0, 1.0));
        assert!(mixed.displacement_lost < 0.01);
        assert_eq!(still.moving, 0);
    }
}

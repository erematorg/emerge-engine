//! Real, shared cursor-driven radial force -- extracted for the exact
//! reason `gui_common::mod`'s own doc cites for `Gfx`/`cursor_to_grid`: a
//! bug fixed in one hand-rolled copy of this plumbing does not reach the
//! other copies. Confirmed live, 2026-08-26: `sand_water_saturation.rs`
//! shared ONE force value between LMB push and RMB pull/lift -- fine for
//! push, but far too weak for pull to ever separate a chunk from a packed
//! pile's own confinement (measured: 0.097 cells of real lift at the
//! shared value that gave push a clean, retained_fraction=1.000 feel).
//! 22 examples independently duplicate this exact push/pull cursor-force
//! block (`grep -rl push_weights examples/`); every one of them is a
//! candidate for the SAME shared-value mistake, not just this one scene.
//!
//! This crosses `gui_common`'s own stated boundary ("deliberately NOT in
//! scope: panel content... differs enough per example") on purpose: this
//! block is NOT panel content, it is the exact kind of "genuinely
//! identical, not just similar" mechanics that module's own doc says
//! justifies extending its scope.
//!
//! Deliberately keeps push and pull as SEPARATE strengths in the type
//! itself, not a single shared field with a sign flip -- that shared-field
//! shape is the root cause of the bug this module exists to stop from
//! recurring elsewhere.

use crate::emerge::particle::Particles;
use glam::Vec2;

/// Real `F = m*a` radial cursor force, in units of each particle's OWN
/// WEIGHT (`push_strength`/`pull_strength * m * g`) -- 1.0 exactly
/// cancels gravity, 2.0 nets 1g. Physically meaningful and scale-free:
/// stays correct at any gravity, cell size, or particle mass, unlike a
/// raw velocity poke (`Simulation::apply_radial_impulse`, which ignores
/// mass entirely and is why cursor interaction can feel arbitrary against
/// real gravity).
pub struct CursorForce {
    pub radius: f32,
    /// Strength for `apply(.., pulling: false)` -- disturbing/shoving a
    /// settled pile. Real, measured default that works well: 3.0.
    pub push_strength: f32,
    /// Strength for `apply(.., pulling: true)` -- lifting a chunk AGAINST
    /// its own weight and the surrounding material's confinement, a
    /// physically harder task than push. Needs meaningfully more force to
    /// achieve real separation -- see this module's own doc for the
    /// measured 0.097-cells-of-nothing failure at push-strength values.
    /// Real, measured default that works well: 7.0.
    pub pull_strength: f32,
}

impl CursorForce {
    pub const fn new(radius: f32, push_strength: f32, pull_strength: f32) -> Self {
        Self {
            radius,
            push_strength,
            pull_strength,
        }
    }

    /// Applies one substep's worth of the force to every particle within
    /// `radius` of `cursor`, `dv = (F/m) * dt` -- mass genuinely resists
    /// acceleration, matching how gravity itself is applied.
    pub fn apply(&self, particles: &mut Particles, cursor: Vec2, g: f32, dt: f32, pulling: bool) {
        let (weight, sign) = if pulling {
            (self.pull_strength, -1.0)
        } else {
            (self.push_strength, 1.0)
        };
        for i in 0..particles.len() {
            let d = particles.x[i] - cursor;
            let dist = d.length();
            if dist > 1.0e-4 && dist < self.radius {
                // Linear falloff, same profile the built-in impulse uses.
                let falloff = 1.0 - dist / self.radius;
                let force = (d / dist) * (weight * particles.mass[i] * g * falloff);
                particles.v[i] += (force / particles.mass[i]) * dt * sign;
            }
        }
    }
}

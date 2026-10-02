//! Shared cursor-driven radial force, extracted for the reason `gui_common::mod`
//! gives for `Gfx`/`cursor_to_grid`: a fix in one hand-rolled copy of this plumbing
//! does not reach the others. One force value shared between LMB push and RMB pull is
//! fine for push but far too weak for pull to separate a chunk from a packed pile's
//! confinement (0.097 cells of lift at the value that gives push a clean
//! retained_fraction=1.000). 22 examples duplicate this push/pull block
//! (`grep -rl push_weights examples/`), each a candidate for the same mistake.
//!
//! This crosses `gui_common`'s stated boundary ("deliberately NOT in scope: panel
//! content... differs enough per example") on purpose: this block is not panel
//! content but identical mechanics, which that module's doc says justifies extending
//! its scope.
//!
//! Push and pull are separate strengths in the type itself, not one shared field with
//! a sign flip, the shape behind that mistake.

use crate::emerge::particle::Particles;
use glam::Vec2;

/// Radial cursor force (F = m*a), in units of each particle's own weight
/// (`push_strength`/`pull_strength * m * g`): 1.0 exactly cancels gravity, 2.0 nets 1g.
/// Scale-free: correct at any gravity, cell size or particle mass, unlike a velocity
/// poke (`Simulation::apply_radial_impulse`, which ignores mass and makes cursor
/// interaction feel arbitrary against gravity).
pub struct CursorForce {
    pub radius: f32,
    /// Strength for `apply(.., pulling: false)` -- disturbing/shoving a
    /// settled pile. Measured default that works well: 3.0.
    pub push_strength: f32,
    /// Strength for `apply(.., pulling: true)` -- lifting a chunk AGAINST
    /// its own weight and the surrounding material's confinement, a
    /// physically harder task than push. Needs meaningfully more force to
    /// achieve real separation -- see this module's doc for the
    /// measured 0.097-cells-of-nothing failure at push-strength values.
    /// Measured default that works well: 7.0.
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
    /// `radius` of `cursor`, `dv = (F/m) * dt` -- mass resists
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

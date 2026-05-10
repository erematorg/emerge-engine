//! Gravitational force fields for MPM particles.
//!
//! `GravityWellField` models discrete macro-scale mass sources (stars, planets, large boulders)
//! that exert gravitational pull on every MPM particle.
//!
//! All positions are in **grid coordinates** (same units as `Particle::x`).
//!
//! # Physics
//! Plummer-softened gravity: **a = G·M·r̂ / (|r|² + ε²)^(3/2)**
//!
//! This is the proper gradient of the Plummer potential Φ = −G·M / √(r² + ε²).
//! Avoids the 1/r² singularity at r → 0. Reduces to Newtonian gravity for r >> ε.

use glam::Vec2;

use crate::solver::cutoff::smooth_cutoff;
use crate::fields::{ForceField, FADE_ONSET_RATIO};
use crate::particle::Particle;

/// Gravitational acceleration from one or more point-mass sources.
///
/// Intended for macro-scale bodies (celestial bodies, large terrain features) that
/// exert gravity on MPM continuum matter. Does **not** model particle-particle N-body —
/// for that, use `NBodyGravityField`.
pub struct GravityWellField {
    /// Point mass sources: `(position in grid coords, mass in simulation units)`.
    pub sources: Vec<(Vec2, f32)>,

    /// Gravitational constant G in simulation units.
    ///
    /// There is no universal default — tune to your scale.
    /// Small grid-scale scenes: ~0.1 simulation units.
    /// For SI: G = 6.674×10⁻¹¹ N·m²/kg² (only meaningful if grid_cell_size is set to SI).
    pub gravitational_constant: f32,

    /// Plummer softening length ε in grid coordinates.
    ///
    /// Prevents force divergence as r → 0.
    /// Rule of thumb: ε ≈ mean inter-particle spacing (typically 0.5–2.0 grid cells).
    pub softening: f32,

    /// Cutoff radius in grid coordinates. Gravity is zero beyond this distance.
    ///
    /// Default: `f32::INFINITY` (IRL — no cutoff).
    /// Set to a finite value as a performance approximation for large particle counts.
    pub cutoff: f32,

    /// Force-switch onset — smooth fade starts at this radius (must be ≤ cutoff).
    /// Only meaningful when cutoff is finite.
    pub switch_on: f32,
}

impl GravityWellField {
    /// Full-range gravity — IRL default, no distance cutoff.
    pub fn new(sources: Vec<(Vec2, f32)>, gravitational_constant: f32, softening: f32) -> Self {
        Self {
            sources,
            gravitational_constant,
            softening,
            cutoff: f32::INFINITY,
            switch_on: f32::INFINITY,
        }
    }

    /// Single stationary point mass — IRL default, no cutoff.
    pub fn point(position: Vec2, mass: f32, gravitational_constant: f32, softening: f32) -> Self {
        Self::new(vec![(position, mass)], gravitational_constant, softening)
    }

    /// Performance approximation: clamp gravity to zero beyond `cutoff` grid-cells.
    /// Smooth fade begins at 0.85 × cutoff.
    pub fn with_cutoff(mut self, cutoff: f32) -> Self {
        self.cutoff = cutoff;
        self.switch_on = FADE_ONSET_RATIO * cutoff;
        self
    }
}

impl ForceField for GravityWellField {
    fn acceleration(&self, particle: &Particle) -> Vec2 {
        let mut acc = Vec2::ZERO;
        let eps2 = self.softening * self.softening;

        for &(src_pos, src_mass) in &self.sources {
            let r_vec = src_pos - particle.x; // vector toward source
            let r2 = r_vec.length_squared();
            let r = r2.sqrt();

            if r >= self.cutoff || r < f32::EPSILON {
                continue;
            }

            // Plummer: a = G·M·r̂ / (r² + ε²)^(3/2)
            // Equivalent (avoids separate normalize): a = G·M·r_vec / (r² + ε²)^(3/2)
            let norm_s = r2 + eps2;
            let force_scale = self.gravitational_constant * src_mass / (norm_s * norm_s.sqrt());

            if !force_scale.is_finite() {
                continue;
            }

            let switch = smooth_cutoff(r, self.switch_on, self.cutoff);
            acc += r_vec * force_scale * switch;
        }

        acc
    }
}

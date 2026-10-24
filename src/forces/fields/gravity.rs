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

use crate::fields::Field;
use crate::particle::Particles;

/// Earth's surface gravitational acceleration in m/s², the value
/// `SimConfig::earth` converts to grid units: standard gravity `g_n`,
/// 9.80665 m/s² exactly (NIST CODATA, physics.nist.gov/cgi-bin/cuu/Value?gn),
/// rounded to three significant figures, 0.035% high. Kept rounded because
/// tests compare `earth()` scenes against analytic formulas written with
/// 9.81; moving to the exact value means updating those together.
pub const EARTH_GRAVITY_M_S2: f32 = 9.81;

/// Gravitational acceleration from one or more point-mass sources.
///
/// Intended for macro-scale bodies (celestial bodies, large terrain features) that
/// exert gravity on MPM continuum matter. Does **not** model particle-particle N-body --
/// for that, use `NBodyGravityField`.
#[derive(Debug)]
pub struct GravityWellField {
    /// Point mass sources: `(position in grid coords, mass in simulation units)`.
    pub sources: Vec<(Vec2, f32)>,

    /// Gravitational constant G in grid units: positions in cells, masses
    /// in grid mass (`(rho / reference_density_kg_m3) * cells^2`, as
    /// `SpawnRegion` spawns them), accelerations in cells/s².
    ///
    /// The real `G_SI` (6.674×10⁻¹¹ N·m²/kg²) converts through
    /// `SimConfig::gravitational_constant_from_si`, which needs the scene's
    /// `SimConfig::slice_thickness_m`.
    pub gravitational_constant: f32,

    /// Plummer softening length ε in grid coordinates.
    ///
    /// A regularization of the point-mass singularity at r → 0, not a property
    /// of the source: its value is the caller's choice and is declared as
    /// such. A real source has a physical size; replacing ε by it is open.
    pub softening: f32,
}

impl GravityWellField {
    /// Full-range gravity: Newtonian gravity has no range limit.
    pub const fn new(
        sources: Vec<(Vec2, f32)>,
        gravitational_constant: f32,
        softening: f32,
    ) -> Self {
        Self {
            sources,
            gravitational_constant,
            softening,
        }
    }

    /// Single stationary point mass.
    pub fn point(position: Vec2, mass: f32, gravitational_constant: f32, softening: f32) -> Self {
        Self::new(vec![(position, mass)], gravitational_constant, softening)
    }
}

impl Field for GravityWellField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let mut acc = Vec2::ZERO;
        let eps2 = self.softening * self.softening;
        let x = particles.x[i];

        for &(src_pos, src_mass) in &self.sources {
            let r_vec = src_pos - x; // vector toward source
            let r2 = r_vec.length_squared();
            let r = r2.sqrt();

            if r < f32::EPSILON {
                continue;
            }

            // Plummer: a = G·M·r̂ / (r² + ε²)^(3/2)
            // Equivalent (avoids separate normalize): a = G·M·r_vec / (r² + ε²)^(3/2)
            let norm_s = r2 + eps2;
            let force_scale = self.gravitational_constant * src_mass / (norm_s * norm_s.sqrt());

            if !force_scale.is_finite() {
                continue;
            }

            acc += r_vec * force_scale;
        }

        acc
    }
}

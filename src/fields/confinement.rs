//! Spatial confinement force fields.
//!
//! Useful for simulation domains (circular arena, plasma confinement, magnetic bottle analogues),
//! and spatial constraints (body bounds, material regions).

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;

/// Circular confinement: harmonic restoring force for particles outside radius.
///
/// Models a soft container — particles inside feel nothing, particles outside
/// are pulled back toward the surface with force proportional to overshoot.
///
/// ```text
///  r < radius:  no force
///  r > radius:  a = -stiffness * (r - radius) * r̂
/// ```
pub struct RadialConfinementField {
    /// Center of the confinement region in grid coordinates.
    pub center: Vec2,

    /// Confinement radius in grid coordinates.
    pub radius: f32,

    /// Restoring acceleration per unit overshoot (grid-units/s² per grid-unit).
    ///
    /// Higher values = harder wall. Start around 100–1000 and tune.
    /// Too high → particles oscillate at the boundary instead of settling.
    pub stiffness: f32,
}

impl RadialConfinementField {
    pub fn new(center: Vec2, radius: f32, stiffness: f32) -> Self {
        Self {
            center,
            radius,
            stiffness,
        }
    }
}

impl Field for RadialConfinementField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let r_vec = particles.x[i] - self.center;
        let dist = r_vec.length();
        let overshoot = dist - self.radius;

        if overshoot <= 0.0 || dist < f32::EPSILON {
            return Vec2::ZERO;
        }

        -r_vec * (self.stiffness * overshoot / dist)
    }
}

/// Rectangular (AABB) confinement: harmonic restoring force outside an axis-aligned box.
///
/// Useful for sandboxes or any rectangular simulation domain.
pub struct AabbConfinementField {
    /// Minimum corner of the box in grid coordinates.
    pub min: Vec2,
    /// Maximum corner of the box in grid coordinates.
    pub max: Vec2,

    /// Restoring acceleration per unit overshoot (grid-units/s² per grid-unit).
    pub stiffness: f32,
}

impl AabbConfinementField {
    pub fn new(min: Vec2, max: Vec2, stiffness: f32) -> Self {
        Self {
            min,
            max,
            stiffness,
        }
    }
}

impl Field for AabbConfinementField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let p = particles.x[i];
        let mut acc = Vec2::ZERO;

        // X axis
        if p.x < self.min.x {
            acc.x += self.stiffness * (self.min.x - p.x);
        } else if p.x > self.max.x {
            acc.x -= self.stiffness * (p.x - self.max.x);
        }

        // Y axis
        if p.y < self.min.y {
            acc.y += self.stiffness * (self.min.y - p.y);
        } else if p.y > self.max.y {
            acc.y -= self.stiffness * (p.y - self.max.y);
        }

        acc
    }
}

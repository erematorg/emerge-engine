//! Coulomb electrostatic force field for MPM particles.
//!
//! `CoulombField` models discrete external charged sources (electrodes, ion emitters,
//! confinement coils) that exert electrostatic forces on charged MPM particles.
//!
//! All positions are in **grid coordinates** (same units as `Particle::x`).
//!
//! # Physics
//! Softened Coulomb: **F = k·q_src·q_p·r̂ / (|r|² + ε²)^(3/2)**
//!
//! The softening ε prevents the 1/r² singularity as two charges overlap.
//! Sign convention: same-sign → repulsive, opposite-sign → attractive.
//!
//! # Particle charge encoding
//! Particles carry no `charge` field in the base struct.
//! Instead, `CoulombField` maps `material_id → charge_value`:
//! assign charge to a material, and every particle of that material feels the field.
//! This lets you make "ionised fluid" or "charged terrain" trivially.

use std::collections::HashMap;

use glam::Vec2;

use crate::fields::{FADE_ONSET_RATIO, Field};
use crate::particle::Particles;
use crate::solver::cutoff::smooth_cutoff;

/// Electrostatic acceleration from external point charges, affecting particles by material.
pub struct CoulombField {
    /// External charged point sources: `(position in grid coords, charge in sim units)`.
    ///
    /// Positive charge is proton-like. Negative is electron-like.
    /// These are macro-scale emitters/electrodes — not individual MPM particles.
    pub sources: Vec<(Vec2, f32)>,

    /// Per-material charge value. `material_id → charge`.
    ///
    /// Particles whose `material_id` is in this map are treated as charged.
    /// Unregistered material_ids are neutral (charge = 0.0, not affected).
    pub material_charges: HashMap<u32, f32>,

    /// Coulomb constant k in simulation units.
    ///
    /// IRL: k = 8.987×10⁹ N·m²/C² in vacuum.
    /// For grid-scale simulations: tune to balance EM forces relative to gravity (e.g. 1.0–100.0).
    pub coulomb_constant: f32,

    /// Plummer softening length ε in grid coordinates.
    ///
    /// Prevents singularity as charged particle approaches source.
    /// Rule of thumb: 0.5–1.0 grid cells.
    pub softening: f32,

    /// Cutoff radius in grid coordinates.
    ///
    /// Default: `f32::INFINITY` (IRL — Coulomb force has infinite range).
    /// Set to a finite value as a performance approximation for large particle counts.
    pub cutoff: f32,

    /// Force-switch onset for smooth fade (must be ≤ cutoff).
    /// Only meaningful when cutoff is finite.
    pub switch_on: f32,
}

impl CoulombField {
    /// Full-range Coulomb field — IRL default, no distance cutoff.
    pub fn new(
        sources: Vec<(Vec2, f32)>,
        material_charges: HashMap<u32, f32>,
        coulomb_constant: f32,
        softening: f32,
    ) -> Self {
        Self {
            sources,
            material_charges,
            coulomb_constant,
            softening,
            cutoff: f32::INFINITY,
            switch_on: f32::INFINITY,
        }
    }

    /// Performance approximation: clamp Coulomb force to zero beyond `cutoff` grid-cells.
    /// Smooth fade begins at 0.85 × cutoff.
    pub fn with_cutoff(mut self, cutoff: f32) -> Self {
        self.cutoff = cutoff;
        self.switch_on = FADE_ONSET_RATIO * cutoff;
        self
    }
}

impl Field for CoulombField {
    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let q_particle = match self.material_charges.get(&particles.material_id[i]) {
            Some(&q) if q.abs() > f32::EPSILON => q,
            _ => return Vec2::ZERO,
        };

        let mut acc = Vec2::ZERO;
        let eps2 = self.softening * self.softening;
        let inv_mass = if particles.mass[i] > f32::EPSILON {
            1.0 / particles.mass[i]
        } else {
            0.0
        };
        let x = particles.x[i];

        for &(src_pos, q_src) in &self.sources {
            let r_vec = x - src_pos; // vector from source to particle
            let r2 = r_vec.length_squared();
            let r = r2.sqrt();

            if r >= self.cutoff || r < f32::EPSILON {
                continue;
            }

            // Softened Coulomb: F = k·q_src·q_p·r_vec / (r² + ε²)^(3/2)
            // Same sign → positive force_scale → repulsive (r_vec points away from source).
            // Opposite sign → negative → attractive.
            let norm_s = r2 + eps2;
            let force_scale = self.coulomb_constant * q_src * q_particle / (norm_s * norm_s.sqrt());

            if !force_scale.is_finite() {
                continue;
            }

            let switch = smooth_cutoff(r, self.switch_on, self.cutoff);
            // Force → acceleration: divide by particle mass
            acc += r_vec * (force_scale * switch * inv_mass);
        }

        acc
    }
}

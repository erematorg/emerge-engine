//! Keller-Segel chemotaxis force field.
//!
//! Applies a per-particle acceleration proportional to the local gradient of a
//! scalar concentration field φ (pheromone, nutrient, morphogen, etc.):
//!
//!   a = χ · ∇φ(x_particle)
//!
//! where χ is the chemotactic sensitivity (positive = attraction, negative = repulsion).
//!
//! # Algorithm
//! Gradient is estimated by central finite differences on the stored grid snapshot:
//!   ∂φ/∂x ≈ (φ[ix+1,iy] − φ[ix−1,iy]) / 2
//!   ∂φ/∂y ≈ (φ[ix,iy+1] − φ[ix,iy−1]) / 2
//! where (ix,iy) = floor(particle position). Grid layout: φ[x*res+y].
//!
//! # Usage
//! ```rust,no_run
//! # extern crate emerge_engine as emerge;
//! # use emerge::prelude::*;
//! # use emerge::fields::ChemotaxisField;
//! # use emerge::{ScalarDiffusionField, ScalarDiffusionConfig};
//! // 1. Maintain a ScalarDiffusionField for pheromone.
//! let mut pheromone = ScalarDiffusionField::new(
//!     ScalarDiffusionConfig { diffusivity: 0.5, decay_rate: 0.1, ambient: 0.0 },
//!     |p| p.temperature,
//!     |p, d| p.temperature += d,
//!     64,
//! );
//!
//! // 2. Create a chemotaxis field that reads the pheromone gradient.
//! let mut chemo = ChemotaxisField::new(64, 1.0);  // grid_res=64, sensitivity=1.0
//!
//! // Per substep (after pheromone.apply):
//! chemo.sync_from(&pheromone);
//! // ... pass chemo as a Field to the solver
//! ```
//!
//! # Reference
//! Keller & Segel 1970, "Initiation of slime mold aggregation viewed as an instability".
//! PDE: ∂ρ/∂t = ∇·(D∇ρ − χ·ρ·∇φ)  →  particle force: a = χ·∇φ.

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;
use crate::thermodynamics::ScalarDiffusionField;

/// Gradient-following force derived from a scalar concentration field (Keller-Segel).
///
/// Call `sync_from(&scalar_field)` once per substep (after `scalar_field.apply()`)
/// to update the internal snapshot, then register with the solver as a `Field`.
pub struct ChemotaxisField {
    /// Grid resolution — must match the ScalarDiffusionField and MPM solver.
    grid_res: usize,
    /// Snapshot of φ on the grid. Layout: phi[x*grid_res+y].
    phi: Vec<f32>,
    /// Chemotactic sensitivity χ (grid-units/s² per φ-unit).
    /// Positive = move up gradient (attraction). Negative = move away (repulsion).
    pub sensitivity: f32,
    /// If set, only particles with this material_id receive the force.
    /// None = all particles (default).
    pub material_filter: Option<u32>,
}

impl ChemotaxisField {
    /// Create a new chemotaxis field.
    ///
    /// - `grid_res`: must match `ScalarDiffusionField` and `Simulation` grid resolution.
    /// - `sensitivity`: χ — positive for attraction, negative for repulsion.
    pub fn new(grid_res: usize, sensitivity: f32) -> Self {
        Self {
            grid_res,
            phi: vec![0.0; grid_res * grid_res],
            sensitivity,
            material_filter: None,
        }
    }

    /// Restrict force to a single material.  `None` (default) = all particles.
    pub fn with_material_filter(mut self, id: u32) -> Self {
        self.material_filter = Some(id);
        self
    }

    /// Copy the current scalar field snapshot from `source` into this field's internal grid.
    ///
    /// Call once per substep, **after** `source.apply()`, so the gradient reflects
    /// the just-computed diffusion step.
    pub fn sync_from(&mut self, source: &ScalarDiffusionField) {
        let src = source.current_phi();
        let n = self.grid_res * self.grid_res;
        debug_assert_eq!(src.len(), n, "ChemotaxisField grid_res mismatch");
        self.phi[..n].copy_from_slice(&src[..n]);
    }

    /// Estimate ∇φ at grid coordinate (x, y) using central differences.
    fn gradient_at(&self, x: i32, y: i32) -> Vec2 {
        let res = self.grid_res as i32;

        let idx = |xi: i32, yi: i32| -> f32 {
            if xi < 0 || yi < 0 || xi >= res || yi >= res {
                return 0.0;
            }
            self.phi[(xi * res + yi) as usize]
        };

        let dphidx = (idx(x + 1, y) - idx(x - 1, y)) * 0.5;
        let dphidy = (idx(x, y + 1) - idx(x, y - 1)) * 0.5;
        Vec2::new(dphidx, dphidy)
    }
}

impl Field for ChemotaxisField {
    fn prepare(&mut self, _particles: &Particles) {
        // Gradient is computed on-demand from the snapshot; no pre-computation needed.
    }

    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        if let Some(id) = self.material_filter
            && particles.material_id[i] != id
        {
            return Vec2::ZERO;
        }
        let ix = particles.x[i].x.floor() as i32;
        let iy = particles.x[i].y.floor() as i32;
        self.sensitivity * self.gradient_at(ix, iy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    const RES: usize = 8;

    /// Linear ramp along x: phi[x*RES+y] = x. Gradient should point purely in +x, zero in y.
    fn ramp_field() -> ChemotaxisField {
        let mut field = ChemotaxisField::new(RES, 1.0);
        for x in 0..RES {
            for y in 0..RES {
                field.phi[x * RES + y] = x as f32;
            }
        }
        field
    }

    fn particle_at(x: f32, y: f32, material_id: u32) -> Particles {
        let mut p = Particle::zeroed();
        p.x = Vec2::new(x, y);
        p.material_id = material_id;
        Particles::from(vec![p])
    }

    #[test]
    fn positive_sensitivity_attracts_up_the_gradient() {
        let field = ramp_field();
        let soa = particle_at(3.0, 4.0, 0);
        let a = field.acceleration(&soa, 0);
        assert!(
            a.x > 0.0,
            "should accelerate toward increasing phi (+x): {a:?}"
        );
        assert!(a.y.abs() < 1e-5, "ramp has no y-gradient: {a:?}");
    }

    #[test]
    fn negative_sensitivity_repels_down_the_gradient() {
        let mut field = ramp_field();
        field.sensitivity = -1.0;
        let soa = particle_at(3.0, 4.0, 0);
        let a = field.acceleration(&soa, 0);
        assert!(a.x < 0.0, "negative sensitivity must flip direction: {a:?}");
    }

    #[test]
    fn material_filter_zeroes_non_matching_particles() {
        let field = ramp_field().with_material_filter(7);
        let matching = particle_at(3.0, 4.0, 7);
        let other = particle_at(3.0, 4.0, 0);
        assert!(
            field.acceleration(&matching, 0).x > 0.0,
            "filtered material should still feel the force"
        );
        assert_eq!(
            field.acceleration(&other, 0),
            Vec2::ZERO,
            "non-matching material must get zero force"
        );
    }

    #[test]
    fn sync_from_matches_scalar_diffusion_field_layout() {
        use crate::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
        let scalar = ScalarDiffusionField::new(
            ScalarDiffusionConfig::default(),
            |p: &Particle| p.temperature,
            |p: &mut Particle, d: f32| p.temperature += d,
            RES,
        );
        let mut field = ChemotaxisField::new(RES, 1.0);
        // Must not panic on a real ScalarDiffusionField of matching grid_res (debug_assert
        // in sync_from would catch a layout/size mismatch).
        field.sync_from(&scalar);
    }
}

use glam::Vec2;

use crate::particle::{Particle, Particles};

/// Aggregate state for a set of particles — returned by spatial and material queries.
///
/// Use this to drive phase transitions, rendering effects, or any consumer system
/// that needs to read simulation state without iterating particles directly.
///
/// ```ignore
/// let s = solver.material_state(SNOW_ID);
/// if s.max_volume_ratio > MELT_THRESHOLD {
///     solver.phase_transition(|p| p.material_id == SNOW_ID && p.plastic_volume_ratio > MELT_THRESHOLD, WATER_ID);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MaterialState {
    /// Number of particles in the queried set.
    pub count: usize,
    /// Mean plastic Jacobian Jp. 1.0 = undeformed; < 1.0 = compressed; > 1.0 = expanded.
    pub avg_volume_ratio: f32,
    /// Peak Jp across the queried set. Useful for threshold triggers.
    pub max_volume_ratio: f32,
    /// Mean speed |v| in grid-cell units/s.
    pub avg_speed: f32,
    /// Mean det(F). < 1 = volumetrically compressed, > 1 = expanded.
    pub avg_det_f: f32,
    /// Mean particle density (mass / volume).
    pub avg_density: f32,
    /// Center of mass in grid coordinates.
    pub centroid: Vec2,
}

/// Aggregate state for all particles of the given material.
pub fn material_state_of(particles: &Particles, material_id: u32) -> MaterialState {
    let mut s = MaterialState::default();
    for i in particles.indices() {
        if particles.material_id[i] == material_id {
            s.accumulate(
                particles.x[i],
                particles.v[i].length(),
                particles.plastic_volume_ratio[i],
                particles.deformation_gradient[i].determinant(),
                particles.density[i],
            );
        }
    }
    s.finalize();
    s
}

/// Aggregate state for all particles of the given material from a CPU mirror slice.
/// Used by GpuSolver which maintains a `Vec<Particle>` mirror.
pub fn material_state_of_slice(particles: &[Particle], material_id: u32) -> MaterialState {
    let mut s = MaterialState::default();
    for p in particles {
        if p.material_id == material_id {
            s.accumulate(
                p.x,
                p.v.length(),
                p.plastic_volume_ratio,
                p.deformation_gradient.determinant(),
                p.density,
            );
        }
    }
    s.finalize();
    s
}

/// Aggregate state for all particles within `radius` grid-cells of `center`.
pub fn region_state_of(particles: &Particles, center: Vec2, radius: f32) -> MaterialState {
    let r2 = radius * radius;
    let mut s = MaterialState::default();
    for i in particles.indices() {
        if (particles.x[i] - center).length_squared() <= r2 {
            s.accumulate(
                particles.x[i],
                particles.v[i].length(),
                particles.plastic_volume_ratio[i],
                particles.deformation_gradient[i].determinant(),
                particles.density[i],
            );
        }
    }
    s.finalize();
    s
}

/// Aggregate state within radius from a CPU mirror slice (GpuSolver).
pub fn region_state_of_slice(particles: &[Particle], center: Vec2, radius: f32) -> MaterialState {
    let r2 = radius * radius;
    let mut s = MaterialState::default();
    for p in particles {
        if (p.x - center).length_squared() <= r2 {
            s.accumulate(
                p.x,
                p.v.length(),
                p.plastic_volume_ratio,
                p.deformation_gradient.determinant(),
                p.density,
            );
        }
    }
    s.finalize();
    s
}

impl MaterialState {
    /// Empty state — returned when no particles match the query.
    pub fn empty() -> Self {
        Self {
            count: 0,
            avg_volume_ratio: 0.0,
            max_volume_ratio: 0.0,
            avg_speed: 0.0,
            avg_det_f: 0.0,
            avg_density: 0.0,
            centroid: Vec2::ZERO,
        }
    }

    pub(crate) fn accumulate(&mut self, x: Vec2, speed: f32, jp: f32, det_f: f32, density: f32) {
        self.count += 1;
        self.centroid += x;
        self.avg_speed += speed;
        self.avg_volume_ratio += jp;
        self.max_volume_ratio = self.max_volume_ratio.max(jp);
        self.avg_det_f += det_f;
        self.avg_density += density;
    }

    pub(crate) fn finalize(&mut self) {
        if self.count > 0 {
            let n = self.count as f32;
            self.centroid /= n;
            self.avg_speed /= n;
            self.avg_volume_ratio /= n;
            self.avg_det_f /= n;
            self.avg_density /= n;
        }
    }
}

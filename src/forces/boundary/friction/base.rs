use glam::Vec2;

use crate::forces::boundary::{BoundaryCondition, apply_coulomb_wall, clamp_position_inside_grid};

/// Grid-level Coulomb wall boundary.
///
/// No-penetration (normal zeroed) + Coulomb friction on tangential component,
/// applied to grid cell velocities during grid update. Matches the Lagrangian
/// particle experience to first order — this is the standard MPM friction model.
///
/// `friction_coefficient = 0.0` → pure slip (same as SlipBoundary).
/// `friction_coefficient = 1.0` → strong friction.
/// IRL µ values: rock-on-rock ≈ 0.6, wet clay ≈ 0.2, ice ≈ 0.05.
///
/// # Note
/// This is grid-level friction (applied to grid cell velocities during grid update),
/// which is the standard MPM friction model. It matches the Lagrangian particle
/// experience to first order but is not per-surface-element friction.
#[derive(Debug, Clone, Copy)]
pub struct FrictionBoundary {
    pub thickness: usize,
    /// Coulomb friction coefficient µ ∈ [0, 1].
    /// 0 = slip (no friction), 1 = strong friction (full tangential damping at normal speed).
    pub friction_coefficient: f32,
}

impl FrictionBoundary {
    pub fn new(thickness: usize, friction_coefficient: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&friction_coefficient),
            "friction_coefficient must be in [0.0, 1.0], got {friction_coefficient}"
        );
        Self {
            thickness,
            friction_coefficient,
        }
    }
}

impl BoundaryCondition for FrictionBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        let t = self.thickness;
        let hi = grid_res.saturating_sub(t + 1);
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let mu = self.friction_coefficient;

        if x < t {
            apply_coulomb_wall(velocity, Vec2::X, mu);
        }
        if x > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_X, mu);
        }
        if y < t {
            apply_coulomb_wall(velocity, Vec2::Y, mu);
        }
        if y > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_Y, mu);
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

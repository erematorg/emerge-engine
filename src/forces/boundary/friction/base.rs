use glam::Vec2;

use crate::forces::boundary::{BoundaryCondition, apply_coulomb_wall, clamp_position_inside_grid};

/// Grid-level Coulomb wall boundary.
///
/// No-penetration (normal zeroed) + Coulomb friction on tangential component,
/// applied to grid cell velocities during grid update. Matches the Lagrangian
/// particle experience to first order -- this is the standard MPM friction model.
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
    /// Real, opt-in Material-Induced Boundary Friction (MIBF, Blatny & Gaume
    /// 2025) -- when `true`, a colliding node's real friction comes from
    /// `Grid::node_friction_at_index` (the mass-weighted average of nearby
    /// particles' own currently-computed internal friction, see
    /// `MaterialModel::current_friction_coefficient`'s own doc) whenever a
    /// friction-reporting particle actually touched that node this
    /// substep, falling back to `friction_coefficient` otherwise (a scene
    /// with no granular material, or a node no sand particle reached).
    /// `false` (default in `new()`) = every existing scene byte-identical.
    pub use_material_friction: bool,
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
            use_material_friction: false,
        }
    }
}

impl BoundaryCondition for FrictionBoundary {
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        self.apply_with_mu(cell_index, grid_res, velocity, self.friction_coefficient)
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }

    fn apply_to_grid_velocity_with_node_friction(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
        node_friction: Option<f32>,
    ) -> f32 {
        let mu = if self.use_material_friction {
            node_friction.unwrap_or(self.friction_coefficient)
        } else {
            self.friction_coefficient
        };
        self.apply_with_mu(cell_index, grid_res, velocity, mu)
    }
}

impl FrictionBoundary {
    /// Returns the total specific energy dissipated across every wall face
    /// this node touches -- a corner node genuinely rubs on two walls, and
    /// both do work.
    fn apply_with_mu(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
        mu: f32,
    ) -> f32 {
        let t = self.thickness;
        let hi = grid_res.saturating_sub(t + 1);
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let mut dissipated = 0.0;

        if x < t {
            dissipated += apply_coulomb_wall(velocity, Vec2::X, mu);
        }
        if x > hi {
            dissipated += apply_coulomb_wall(velocity, Vec2::NEG_X, mu);
        }
        if y < t {
            dissipated += apply_coulomb_wall(velocity, Vec2::Y, mu);
        }
        if y > hi {
            dissipated += apply_coulomb_wall(velocity, Vec2::NEG_Y, mu);
        }
        dissipated
    }
}

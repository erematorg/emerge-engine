use glam::Vec2;

use crate::forces::boundary::{
    BoundaryCondition, apply_coulomb_wall, assert_friction_coefficient, clamp_position_inside_grid,
};

/// Grid-level Coulomb wall boundary.
///
/// No-penetration (normal zeroed) + Coulomb friction on the tangential
/// component, applied to grid node velocities during the grid update: the
/// tangential impulse is at most `mu` times the normal one, the rigid-wall case
/// of the multi-material contact correction (Nairn, Hammerquist & Smith 2020,
/// eqs. 8-10, with the wall's mass infinite). It is grid-level friction, not
/// per-surface-element friction.
///
/// `friction_coefficient = 0.0` → pure slip (same as `SlipBoundary`).
/// The coefficient is a measured property of the pair of surfaces in contact.
#[derive(Debug, Clone, Copy)]
pub struct FrictionBoundary {
    pub thickness: usize,
    /// Coulomb friction coefficient µ >= 0, the ratio of tangential to normal
    /// impulse at sliding. No upper bound: rubber on dry rough surfaces exceeds
    /// 1 (Persson, Tartaglino, Albohr & Tosatti 2005, Phys. Rev. B 71, 035428).
    pub friction_coefficient: f32,
    /// Opt-in Material-Induced Boundary Friction (MIBF, Blatny & Gaume 2025).
    /// When `true`, a colliding node's friction comes from
    /// `Grid::node_friction_at_index` (the mass-weighted average of nearby
    /// particles' current internal friction, see
    /// `MaterialModel::current_friction_coefficient`) whenever a
    /// friction-reporting particle touched that node this substep, and from
    /// `friction_coefficient` otherwise (no granular material in the scene, or
    /// a node no sand particle reached). `false` (default in `new()`) keeps
    /// the fixed coefficient.
    pub use_material_friction: bool,
}

impl FrictionBoundary {
    pub fn new(thickness: usize, friction_coefficient: f32) -> Self {
        assert_friction_coefficient(friction_coefficient, "friction_coefficient");
        Self {
            thickness,
            friction_coefficient,
            use_material_friction: false,
        }
    }
}

impl BoundaryCondition for FrictionBoundary {
    /// Coulomb walls with this coefficient; the material-induced friction
    /// (one coefficient per node) has no such form yet.
    fn domain_walls(&self) -> Option<crate::boundary::DomainWalls> {
        (!self.use_material_friction).then_some(crate::boundary::DomainWalls {
            thickness: self.thickness,
            friction: self.friction_coefficient,
        })
    }

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
    /// this node touches -- a corner node rubs on two walls, and
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

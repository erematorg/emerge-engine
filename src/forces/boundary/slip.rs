use glam::Vec2;

use super::{
    BoundaryCondition, apply_sealed_wall_velocity, apply_slip_wall_velocity,
    clamp_position_inside_grid,
};

/// Frictionless walls on the four sides of the grid.
#[derive(Debug, Clone, Copy)]
pub struct SlipBoundary {
    pub thickness: usize,
    /// Whether matter at a wall node is held against the wall
    /// (`apply_sealed_wall_velocity`) instead of being free to leave it
    /// (`apply_slip_wall_velocity`, the default).
    pub sealed: bool,
}

impl SlipBoundary {
    /// Walls that matter can leave: the open-container case, and the right
    /// wall for solids and grains, which separate from a wall they are not
    /// pressed against.
    pub const fn new(thickness: usize) -> Self {
        Self {
            thickness,
            sealed: false,
        }
    }

    /// Walls of a sealed container, which a fluid filling it cannot leave
    /// (see `apply_sealed_wall_velocity`). Leave it to scenes whose walls
    /// really have no ambient air behind them: a liquid with a free surface
    /// would be held against an overhanging wall it should fall from.
    pub const fn sealed(thickness: usize) -> Self {
        Self {
            thickness,
            sealed: true,
        }
    }
}

impl BoundaryCondition for SlipBoundary {
    /// Frictionless by construction, so it dissipates nothing: a slip wall
    /// only removes the normal component, which is a no-penetration
    /// constraint and not frictional work. See `apply_coulomb_wall`'s doc.
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        if self.sealed {
            apply_sealed_wall_velocity(self.thickness, cell_index, grid_res, velocity);
        } else {
            apply_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
        }
        0.0
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }

    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        true
    }
}

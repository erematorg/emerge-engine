use glam::Vec2;

use crate::particle::Particles;

pub trait BoundaryCondition: Send + Sync + core::fmt::Debug {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2);
    /// Clamp particle position to the valid domain after G2P.
    /// Not a physical force — last-resort domain enforcement so particles never escape the grid.
    /// Proper no-penetration physics lives in `apply_to_grid_velocity`.
    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2;
    fn post_g2p_particle(&self, _particles: &mut Particles, _i: usize, _grid_res: usize, _dt: f32) {
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlipBoundary {
    pub thickness: usize,
}

impl SlipBoundary {
    pub fn new(thickness: usize) -> Self {
        Self { thickness }
    }
}

impl BoundaryCondition for SlipBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        apply_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

/// Grid-level slip boundary with a tighter inner keep-out zone.
///
/// Identical physics to `SlipBoundary` — no-penetration enforced on grid velocities.
/// `predictive_wall_min` shrinks the safe zone so fast particles hitting the boundary
/// layer are caught by `clamp_particle_position` before they can escape. The actual
/// wall physics is still the grid-level normal-zeroing, not a particle-level correction.
#[derive(Debug, Clone, Copy)]
pub struct PredictiveBoundary {
    pub thickness: usize,
    pub predictive_wall_min: f32,
}

impl PredictiveBoundary {
    pub fn new(thickness: usize, predictive_wall_min: f32) -> Self {
        Self {
            thickness,
            predictive_wall_min,
        }
    }
}

impl BoundaryCondition for PredictiveBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        apply_slip_wall_velocity(self.thickness, cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        let min = self.predictive_wall_min;
        let max = (grid_res as f32 - 1.0) - self.predictive_wall_min;
        position.clamp(Vec2::splat(min), Vec2::splat(max))
    }
}

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

/// Apply Coulomb wall friction along one wall face.
///
/// `outward_normal`: unit vector pointing away from the wall into the domain.
/// When the velocity has a component moving INTO the wall (v · outward_normal < 0),
/// zero the normal component and damp the tangential component by µ × |v_normal|.
fn apply_coulomb_wall(velocity: &mut Vec2, outward_normal: Vec2, mu: f32) {
    let v_n_scalar = velocity.dot(outward_normal);
    // Only act when moving into the wall.
    if v_n_scalar >= 0.0 {
        return;
    }
    let normal_speed = v_n_scalar.abs();
    let v_t = *velocity - v_n_scalar * outward_normal;
    let v_t_len = v_t.length();
    let friction_impulse = mu * normal_speed;
    // Tangential speed after friction: max(|v_t| − µ|v_n|, 0), direction preserved.
    *velocity = if v_t_len > friction_impulse {
        v_t * ((v_t_len - friction_impulse) / v_t_len)
    } else {
        Vec2::ZERO
    };
}

fn apply_slip_wall_velocity(
    thickness: usize,
    cell_index: usize,
    grid_res: usize,
    velocity: &mut Vec2,
) {
    let hi = grid_res - (thickness + 1);
    let x = cell_index / grid_res;
    let y = cell_index % grid_res;
    // Only block the inward component — let outward (escape) velocity pass through.
    // Standard MPM slip: no-penetration, free tangential slip.
    if x < thickness {
        velocity.x = velocity.x.max(0.0);
    }
    if x > hi {
        velocity.x = velocity.x.min(0.0);
    }
    if y < thickness {
        velocity.y = velocity.y.max(0.0);
    }
    if y > hi {
        velocity.y = velocity.y.min(0.0);
    }
}

fn clamp_position_inside_grid(thickness: usize, position: Vec2, grid_res: usize) -> Vec2 {
    let min = thickness.saturating_sub(1) as f32;
    let max = grid_res.saturating_sub(thickness) as f32;
    position.clamp(Vec2::splat(min), Vec2::splat(max))
}

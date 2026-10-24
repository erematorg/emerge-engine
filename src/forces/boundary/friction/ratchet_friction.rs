use glam::Vec2;

use crate::forces::boundary::{
    BoundaryCondition, apply_anisotropic_coulomb_wall, apply_coulomb_wall,
    assert_friction_coefficient, clamp_position_inside_grid,
};

/// Directional (anisotropic) Coulomb friction on the floor: sliding along
/// `easy_direction` meets `mu_easy`, sliding against it meets `mu_resist`.
///
/// The law is the forward/backward part of the anisotropic Coulomb friction
/// measured on snake skin by Hu, Nirody, Scott & Shelley 2009 ("The mechanics
/// of slithering locomotion", PNAS 106:10081, eq. 2): the friction force
/// opposes the sliding direction with a coefficient that depends on whether
/// the surface slides forward (`mu_f`, 0.11 on cloth) or backward (`mu_b`,
/// 0.14), set by scales that snag on asperities; on a smooth substrate the
/// coefficients become nearly equal. Their third, transverse coefficient has
/// no counterpart in a two-dimensional vertical slice, where the floor tangent
/// is one-dimensional. The same paper describes earthworms crawling by this
/// kind of ratcheting, engaging their hairs in the ground.
///
/// Declared simplification: the anisotropy belongs to the body's surface,
/// oriented by the body itself (the snake's tangent `s` in Hu et al.); here it
/// is attached to the floor and oriented by `easy_direction`, which the caller
/// sets. It applies to whatever slides on the floor.
///
/// SoftZoo, the published MPM soft-robot locomotion benchmark (`tmp/softzoo`),
/// uses only constant, symmetric Coulomb ground contact
/// (`engine/static/flat_surface.py`).
///
/// `easy_direction` and both coefficients are live (`set_easy_direction`,
/// `set_friction`), stored as `AtomicU32` (bit-cast f32) so the type stays
/// `Sync`, as `BoundaryCondition: Send + Sync` requires.
#[derive(Debug)]
pub struct RatchetFrictionBoundary {
    pub thickness: usize,
    /// Isotropic Coulomb coefficient of the side walls and ceiling, which have
    /// no ratchet surface.
    pub wall_friction: f32,
    easy_dir_x_bits: std::sync::atomic::AtomicU32,
    easy_dir_y_bits: std::sync::atomic::AtomicU32,
    mu_easy_bits: std::sync::atomic::AtomicU32,
    mu_resist_bits: std::sync::atomic::AtomicU32,
}

impl RatchetFrictionBoundary {
    /// `mu_easy`/`mu_resist`: the floor's coefficients for sliding along and
    /// against `easy_direction`. `wall_friction`: the side walls' and
    /// ceiling's own coefficient. All are measured surface properties, `>= 0`.
    pub fn new(
        thickness: usize,
        mu_easy: f32,
        mu_resist: f32,
        easy_direction: Vec2,
        wall_friction: f32,
    ) -> Self {
        assert_friction_coefficient(mu_easy, "mu_easy");
        assert_friction_coefficient(mu_resist, "mu_resist");
        assert_friction_coefficient(wall_friction, "wall_friction");
        let d = easy_direction.normalize_or_zero();
        Self {
            thickness,
            wall_friction,
            easy_dir_x_bits: std::sync::atomic::AtomicU32::new(d.x.to_bits()),
            easy_dir_y_bits: std::sync::atomic::AtomicU32::new(d.y.to_bits()),
            mu_easy_bits: std::sync::atomic::AtomicU32::new(mu_easy.to_bits()),
            mu_resist_bits: std::sync::atomic::AtomicU32::new(mu_resist.to_bits()),
        }
    }

    /// Update the easy direction live. Takes effect on the next substep.
    pub fn set_easy_direction(&self, direction: Vec2) {
        let d = direction.normalize_or_zero();
        self.easy_dir_x_bits
            .store(d.x.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.easy_dir_y_bits
            .store(d.y.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Update the floor's two coefficients live. Takes effect on the next substep.
    pub fn set_friction(&self, mu_easy: f32, mu_resist: f32) {
        assert_friction_coefficient(mu_easy, "mu_easy");
        assert_friction_coefficient(mu_resist, "mu_resist");
        self.mu_easy_bits
            .store(mu_easy.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.mu_resist_bits
            .store(mu_resist.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    fn mu_easy(&self) -> f32 {
        f32::from_bits(self.mu_easy_bits.load(std::sync::atomic::Ordering::Relaxed))
    }

    fn mu_resist(&self) -> f32 {
        f32::from_bits(
            self.mu_resist_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    fn easy_direction(&self) -> Vec2 {
        Vec2::new(
            f32::from_bits(
                self.easy_dir_x_bits
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            f32::from_bits(
                self.easy_dir_y_bits
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        )
    }
}

impl BoundaryCondition for RatchetFrictionBoundary {
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        let mut dissipated = 0.0;
        let t = self.thickness;
        let hi = grid_res.saturating_sub(t + 1);
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;

        // Side walls and ceiling: isotropic Coulomb friction with their own
        // coefficient.
        if x < t {
            dissipated += apply_coulomb_wall(velocity, Vec2::X, self.wall_friction);
        }
        if x > hi {
            dissipated += apply_coulomb_wall(velocity, Vec2::NEG_X, self.wall_friction);
        }
        if y > hi {
            dissipated += apply_coulomb_wall(velocity, Vec2::NEG_Y, self.wall_friction);
        }

        // Floor: sliding along `easy_direction` meets `mu_easy`, against it
        // `mu_resist`.
        if y < t {
            dissipated += apply_anisotropic_coulomb_wall(
                velocity,
                Vec2::Y,
                self.easy_direction(),
                self.mu_easy(),
                self.mu_resist(),
            );
        }
        dissipated
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

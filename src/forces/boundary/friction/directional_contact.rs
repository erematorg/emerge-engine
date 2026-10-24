use glam::Vec2;

use crate::forces::boundary::{apply_anisotropic_coulomb_wall, assert_friction_coefficient};

/// Directional (anisotropic) Coulomb friction for body-to-body multi-field
/// contact: the same law as `RatchetFrictionBoundary`'s floor (Hu, Nirody,
/// Scott & Shelley 2009, PNAS 106:10081, eq. 2, forward/backward part; see
/// `apply_anisotropic_coulomb_wall`), applied against an arbitrary contact
/// normal instead of the flat world floor, since a contact interface between
/// a body and real terrain particles can be sloped or uneven.
/// `easy_direction` is projected onto the local tangent, so sliding "along"
/// it keeps its meaning on a slope.
///
/// The law lives here; the multi-field contact resolution that applies it
/// (`spacetime::grid::contact`, Bardenhagen et al. 2001) is the
/// discretization. It acts on any body that opts particles into
/// `Particle::contact_group != 0`, living or not.
///
/// Stored as `AtomicU32` (bit-cast f32), like `RatchetFrictionBoundary`, so a
/// shared `Arc` lets a caller change the direction or the coefficients live.
#[derive(Debug)]
pub struct DirectionalContactGrip {
    mu_easy_bits: std::sync::atomic::AtomicU32,
    mu_resist_bits: std::sync::atomic::AtomicU32,
    easy_dir_x_bits: std::sync::atomic::AtomicU32,
    easy_dir_y_bits: std::sync::atomic::AtomicU32,
}

impl DirectionalContactGrip {
    pub fn new(mu_easy: f32, mu_resist: f32, easy_direction: Vec2) -> Self {
        assert_friction_coefficient(mu_easy, "mu_easy");
        assert_friction_coefficient(mu_resist, "mu_resist");
        let d = easy_direction.normalize_or_zero();
        Self {
            mu_easy_bits: std::sync::atomic::AtomicU32::new(mu_easy.to_bits()),
            mu_resist_bits: std::sync::atomic::AtomicU32::new(mu_resist.to_bits()),
            easy_dir_x_bits: std::sync::atomic::AtomicU32::new(d.x.to_bits()),
            easy_dir_y_bits: std::sync::atomic::AtomicU32::new(d.y.to_bits()),
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

    /// Update the friction coefficients live. Takes effect on the next substep.
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

    /// Apply the directional friction to the relative velocity `v_rel` against
    /// the contact's outward normal `n` (any unit vector). Returns the
    /// specific kinetic energy the friction dissipated, as
    /// `apply_coulomb_wall` does, so the contact can turn it into heat.
    pub(crate) fn resolve(&self, v_rel: &mut Vec2, n: Vec2) -> f32 {
        apply_anisotropic_coulomb_wall(
            v_rel,
            n,
            self.easy_direction(),
            self.mu_easy(),
            self.mu_resist(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a slope the easy direction is read along the local tangent: sliding
    /// down a 45-degree surface whose tangent points down-right, with the easy
    /// direction +x, meets `mu_easy`; the opposite slide meets `mu_resist`.
    #[test]
    fn easy_direction_is_projected_onto_the_contact_tangent() {
        let grip = DirectionalContactGrip::new(0.1, 0.9, Vec2::X);
        let n = Vec2::new(1.0, 1.0).normalize(); // outward normal of a slope
        let into = -0.5 * n;
        let down_right = Vec2::new(1.0, -1.0).normalize();

        let mut easy = into + 2.0 * down_right;
        let heat_easy = grip.resolve(&mut easy, n);
        let mut hard = into - 2.0 * down_right;
        let heat_hard = grip.resolve(&mut hard, n);

        // Tangential speed 2, normal speed 0.5: mu * 0.5 removed.
        assert!(
            (easy.length() - (2.0 - 0.1 * 0.5)).abs() < 1e-5,
            "got {easy}"
        );
        assert!(
            (hard.length() - (2.0 - 0.9 * 0.5)).abs() < 1e-5,
            "got {hard}"
        );
        assert!(heat_hard > heat_easy && heat_easy > 0.0);
    }
}

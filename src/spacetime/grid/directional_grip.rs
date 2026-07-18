//! `DirectionalContactGrip` -- split out of `mod.rs` (was ~110 of its ~1320
//! lines). Fully self-contained: only depends on `crate::boundary::
//! apply_coulomb_wall`, not on any `Grid`/`Cell` internals.

use glam::Vec2;

/// Directional (setae-style) Coulomb friction for the multi-field contact "grip"
/// field -- the real-terrain generalization of `RatchetFrictionBoundary`'s same
/// asymmetric-friction mechanism (see that type's doc for the full biomechanical
/// grounding: real crawlers break fore/aft slip symmetry structurally, not by
/// timing friction to muscle phase). `RatchetFrictionBoundary` only ever resolves
/// against a fixed, flat world floor (`normal = Vec2::Y` always); this generalizes
/// the same idea to an ARBITRARY contact normal, since a real multi-field contact
/// interface (a creature gripping actual terrain particles) can be sloped or
/// uneven, not just a flat boundary. `easy_direction` is projected onto the local
/// tangent plane (perpendicular to whatever normal the contact resolver fit that
/// substep) so "resist sliding this way less" still means the right thing on a
/// slope, not just on flat ground.
///
/// Deliberately generic, not tied to any one creature or body: this attaches to
/// the grid's contact resolution as a whole (the existing binary grip/rest
/// split, see `ContactCell` doc), so ANY body that opts particles into
/// `Particle::contact_group != 0` gets the same real, scalable mechanism for
/// free -- living or non-living, any body plan, matching every other primitive
/// in this engine (materials, force fields, boundaries) being creature-agnostic.
///
/// Stored as `AtomicU32` (bit-cast f32), same live-adjustable pattern as
/// `RatchetFrictionBoundary`, so a shared `Arc` reference lets a player/AI change
/// crawl direction or friction values from outside with no reconstruction.
#[derive(Debug)]
pub struct DirectionalContactGrip {
    mu_easy_bits: std::sync::atomic::AtomicU32,
    mu_resist_bits: std::sync::atomic::AtomicU32,
    easy_dir_x_bits: std::sync::atomic::AtomicU32,
    easy_dir_y_bits: std::sync::atomic::AtomicU32,
}

impl DirectionalContactGrip {
    pub fn new(mu_easy: f32, mu_resist: f32, easy_direction: Vec2) -> Self {
        assert!(
            (0.0..=1.0).contains(&mu_easy) && (0.0..=1.0).contains(&mu_resist),
            "mu_easy and mu_resist must be in [0.0, 1.0]"
        );
        let d = easy_direction.normalize_or_zero();
        Self {
            mu_easy_bits: std::sync::atomic::AtomicU32::new(mu_easy.to_bits()),
            mu_resist_bits: std::sync::atomic::AtomicU32::new(mu_resist.to_bits()),
            easy_dir_x_bits: std::sync::atomic::AtomicU32::new(d.x.to_bits()),
            easy_dir_y_bits: std::sync::atomic::AtomicU32::new(d.y.to_bits()),
        }
    }

    /// Update the preferred grip direction live (e.g. player/AI steering input).
    /// Takes effect on the very next substep; no reconstruction needed.
    pub fn set_easy_direction(&self, direction: Vec2) {
        let d = direction.normalize_or_zero();
        self.easy_dir_x_bits
            .store(d.x.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.easy_dir_y_bits
            .store(d.y.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Update the friction coefficients live. Set `mu_easy == mu_resist` to
    /// disable the directional asymmetry entirely (ordinary symmetric contact
    /// friction) whenever there's no real steering intent.
    pub fn set_friction(&self, mu_easy: f32, mu_resist: f32) {
        assert!(
            (0.0..=1.0).contains(&mu_easy) && (0.0..=1.0).contains(&mu_resist),
            "mu_easy and mu_resist must be in [0.0, 1.0]"
        );
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

    /// Apply directional Coulomb friction to `v_rel` against contact normal `n`
    /// (any unit vector, not just a flat floor's `Vec2::Y`). Projects
    /// `easy_direction` onto the tangent (`n` rotated 90°) to decide alignment,
    /// then defers to the same proven `apply_coulomb_wall` used everywhere else
    /// in this engine for the actual normal-clamp + tangential-reduction math.
    pub(super) fn resolve(&self, v_rel: &mut Vec2, n: Vec2) {
        let tangent = Vec2::new(-n.y, n.x);
        let v_t = v_rel.dot(tangent);
        let easy_t = self.easy_direction().dot(tangent);
        let aligned = v_t * easy_t >= 0.0;
        let mu = if aligned {
            self.mu_easy()
        } else {
            self.mu_resist()
        };
        crate::boundary::apply_coulomb_wall(v_rel, n, mu);
    }
}

use glam::Vec2;

use super::{BoundaryCondition, apply_coulomb_wall, clamp_position_inside_grid};

/// Directional (anisotropic) Coulomb floor friction — a real "ratchet" mechanism,
/// not a phase-gated one.
///
/// Research finding (checked against SoftZoo, the published MPM soft-robot
/// locomotion benchmark, `tmp/softzoo`): its ground contact uses only constant,
/// SYMMETRIC Coulomb friction (`Sticky`/`Slip`/`Separate` in
/// `engine/static/flat_surface.py`) — no activation-gated or phase-coupled
/// friction anywhere. Real crawlers (earthworms) don't gate friction on muscle
/// state either — they use a structurally asymmetric surface: setae (directional
/// bristles) that resist sliding one way and permit it the other, a real
/// mechanical ratchet independent of muscle timing. This is that mechanism,
/// applied to the floor wall's tangential (horizontal) friction: sliding in
/// `easy_direction` sees `mu_easy`, sliding against it sees `mu_resist`. Combined
/// with any traveling-wave contraction (no special phase-coordination required —
/// this is the point: the asymmetry lives in the boundary, not in choreographing
/// the gait against it), net drift accumulates in `easy_direction` because
/// backward slip is preferentially resisted.
/// `easy_direction` is LIVE, not baked in at construction — real animals decide
/// which way to anchor moment to moment (a real neural/behavioral choice, not a
/// fixed body plan), so this is `set_easy_direction`-updatable from outside
/// (e.g. every frame, from player/AI steering input) with no reconstruction and
/// no boundary-swap. Stored as two `AtomicU32` (bit-cast f32) rather than a plain
/// `Vec2` field so the type stays `Sync` or the `BoundaryCondition: Send + Sync`
/// bound is impossible — a `Cell` would be `Send` but not `Sync`.
#[derive(Debug)]
pub struct RatchetFrictionBoundary {
    pub thickness: usize,
    easy_dir_x_bits: std::sync::atomic::AtomicU32,
    easy_dir_y_bits: std::sync::atomic::AtomicU32,
    // Real bug found live, 2026-07-13: `mu_easy`/`mu_resist` used to be plain,
    // construction-only f32 fields -- meaning the ratchet's directional
    // asymmetry was ALWAYS active, even while a player provided zero steering
    // input. Combined with ordinary passive settling jitter (a body dropped
    // under gravity always wobbles a little while it settles), the ratchet
    // converted that jitter into a real, substantial (~18-unit) crawl BEFORE
    // any muscle activation ever ran -- confirmed via a real headless log
    // showing `act mean=0.00 max=0.00` (activation genuinely zero, an earlier
    // fix already gated it correctly) while drift still reached +18 units.
    // Real fix: make friction live-adjustable via atomics, same pattern as
    // `easy_direction` below, so the caller can set mu_easy==mu_resist
    // (symmetric, no ratchet effect at all) whenever there's no real steering
    // intent, and restore the real asymmetric values only while actively
    // steered -- matching the same "no bias without input" principle
    // `set_easy_direction`'s own doc already establishes for direction.
    mu_easy_bits: std::sync::atomic::AtomicU32,
    mu_resist_bits: std::sync::atomic::AtomicU32,
}

impl RatchetFrictionBoundary {
    pub fn new(thickness: usize, mu_easy: f32, mu_resist: f32, easy_direction: Vec2) -> Self {
        assert!(
            (0.0..=1.0).contains(&mu_easy) && (0.0..=1.0).contains(&mu_resist),
            "mu_easy and mu_resist must be in [0.0, 1.0]"
        );
        let d = easy_direction.normalize_or_zero();
        Self {
            thickness,
            easy_dir_x_bits: std::sync::atomic::AtomicU32::new(d.x.to_bits()),
            easy_dir_y_bits: std::sync::atomic::AtomicU32::new(d.y.to_bits()),
            mu_easy_bits: std::sync::atomic::AtomicU32::new(mu_easy.to_bits()),
            mu_resist_bits: std::sync::atomic::AtomicU32::new(mu_resist.to_bits()),
        }
    }

    /// Update the ratchet's preferred crawl direction live — e.g. driven by
    /// real-time player or AI steering input. Takes effect on the very next
    /// substep; no reconstruction, no boundary replacement.
    pub fn set_easy_direction(&self, direction: Vec2) {
        let d = direction.normalize_or_zero();
        self.easy_dir_x_bits
            .store(d.x.to_bits(), std::sync::atomic::Ordering::Relaxed);
        self.easy_dir_y_bits
            .store(d.y.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Update the ratchet's own friction coefficients live. Set `mu_easy ==
    /// mu_resist` to disable the directional asymmetry entirely (ordinary
    /// symmetric Coulomb friction, no ratchet effect) when there's no real
    /// steering intent, so passive settling jitter can't be converted into an
    /// unsolicited directional crawl. Takes effect on the very next substep.
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
}

impl BoundaryCondition for RatchetFrictionBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        let t = self.thickness;
        let hi = grid_res.saturating_sub(t + 1);
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;

        // Side and ceiling walls: plain symmetric slip+friction, same as
        // FrictionBoundary — the ratchet only applies to the floor, where a
        // resting/crawling body actually spends its contact time.
        let mu_side = 0.5 * (self.mu_easy() + self.mu_resist());
        if x < t {
            apply_coulomb_wall(velocity, Vec2::X, mu_side);
        }
        if x > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_X, mu_side);
        }
        if y > hi {
            apply_coulomb_wall(velocity, Vec2::NEG_Y, mu_side);
        }

        // Floor: directional friction. Tangential (horizontal) motion aligned
        // with the LIVE easy_direction gets mu_easy; opposing motion gets mu_resist.
        if y < t {
            let v_n_scalar = velocity.dot(Vec2::Y);
            if v_n_scalar < 0.0 {
                let easy_direction = self.easy_direction();
                let tangential = velocity.x;
                let aligned = tangential * easy_direction.x >= 0.0;
                let mu = if aligned {
                    self.mu_easy()
                } else {
                    self.mu_resist()
                };
                apply_coulomb_wall(velocity, Vec2::Y, mu);
            }
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

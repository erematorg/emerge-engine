use glam::Vec2;

use super::base::FrictionBoundary;
use crate::forces::boundary::BoundaryCondition;
use crate::particle::ParticleUpdateCtx;

/// Coulomb wall friction whose EFFECTIVE grip is modulated by each particle's own
/// muscle contraction PHASE — real anchoring behavior in soft-bodied peristaltic
/// crawling.
///
/// A symmetric contract/release cycle against CONSTANT friction produces near-zero
/// net drift: pushing forward during contraction is resisted exactly as much as the
/// recovery slide, so the two phases cancel (the same reason you can't swim forward
/// clapping your hands symmetrically underwater — verified directly in emerge's own
/// `basic_creature` demo diagnostics, which measured near-zero net locomotion under
/// plain `FrictionBoundary` regardless of muscle fiber direction). Real crawlers
/// (earthworms, inchworms) break this symmetry with setae/anchoring structures that
/// engage the substrate specifically during a segment's SHORTENING phase and
/// disengage during LENGTHENING (Trueman 1975, "The Locomotion of Soft-Bodied
/// Animals") — grip is gated on phase, not on activation magnitude alone. An
/// earlier magnitude-only design (grip ∝ activation, independent of whether the
/// segment was shortening or lengthening) measured WORSE than no grip at all
/// (near-total lockup, since a resting body sits continuously in the grip zone
/// regardless of phase) — this version fixes that by keying grip to the real
/// mechanical signal: the fiber-aligned strain RATE (derived from the already-
/// tracked `velocity_gradient` and `activation_dir`, no new stored state), which
/// is negative while a fiber is shortening (contracting → anchor) and positive
/// while lengthening (extending → release).
///
/// Base wall physics (no-penetration + Coulomb tangential damping) is identical to
/// `FrictionBoundary` — delegated to an inner instance. This only ADDS an extra,
/// phase-gated velocity damping to particles inside the boundary layer, via
/// `post_g2p_particle` — a per-particle hook every other boundary here leaves as a
/// no-op, so this composes with any material/creature that sets `activation` and
/// `activation_dir` without any grid- or GPU-side changes.
#[derive(Debug, Clone, Copy)]
pub struct GripFrictionBoundary {
    inner: FrictionBoundary,
    /// Extra grip strength per unit (activation × contraction rate), in [0, 1].
    /// 0.0 = identical to plain `FrictionBoundary` (no grip coupling, the
    /// symmetric-cycle case above). At `grip_gain = 1.0`, a particle that is both
    /// active AND actively shortening its fiber is fully anchored (horizontal
    /// velocity zeroed) — the real "power stroke" anchor; a particle that is
    /// relaxed, or actively lengthening (the recovery slide), is unaffected beyond
    /// the base Coulomb friction.
    pub grip_gain: f32,
}

impl GripFrictionBoundary {
    pub fn new(thickness: usize, friction_coefficient: f32, grip_gain: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&grip_gain),
            "grip_gain must be in [0.0, 1.0], got {grip_gain}"
        );
        Self {
            inner: FrictionBoundary::new(thickness, friction_coefficient),
            grip_gain,
        }
    }
}

impl BoundaryCondition for GripFrictionBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        self.inner
            .apply_to_grid_velocity(cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        self.inner.clamp_particle_position(position, grid_res)
    }

    fn post_g2p_particle(&self, ctx: &mut ParticleUpdateCtx, _grid_res: usize, _dt: f32) {
        if self.grip_gain <= 0.0 {
            return;
        }
        // Floor only -- this models real ground-contact crawling grip (earthworm/
        // inchworm setae engaging the SUBSTRATE), same floor-only scoping
        // `RatchetFrictionBoundary` uses for its directional friction ("the ratchet
        // only applies to the floor, where a resting/crawling body actually spends
        // its contact time"). Side/ceiling walls aren't a substrate a crawling body
        // grips, and the horizontal/vertical decomposition below only means
        // tangential/normal for the FLOOR specifically (compare
        // `FrictionBoundary`'s own `apply_coulomb_wall(velocity, Vec2::Y, mu)` call
        // for y<t, where Y is normal and X is tangential -- that convention doesn't
        // hold at the side walls).
        //
        // Boundary layer margin: wider than the wall's own solid region, since a
        // resting body settles with its lowest particles just ABOVE the wall
        // threshold (clamp_particle_position keeps them out of the solid cells),
        // not literally inside it. A contracting segment needs to reach the ground
        // to anchor, so the grip zone must cover that resting layer.
        let t = self.inner.thickness as f32 + 2.0;
        let near_floor = ctx.x.y < t;
        if !near_floor || ctx.activation <= 0.0 {
            return;
        }
        // Fiber-aligned strain rate: n0 . (velocity_gradient . n0) is the rate of
        // change of length per unit length along the fiber direction. Negative =
        // shortening (contracting → anchor phase); positive = lengthening
        // (extending → release phase, must NOT grip or the segment can never slide).
        let n = ctx.activation_dir;
        let len_sq = n.dot(n);
        if len_sq <= f32::EPSILON {
            return;
        }
        let n0 = n / len_sq.sqrt();
        let strain_rate = n0.dot(*ctx.velocity_gradient * n0);
        let contracting = (-strain_rate).clamp(0.0, 1.0);
        let grip = (self.grip_gain * ctx.activation * contracting).clamp(0.0, 1.0);
        // Only the horizontal (tangential-to-floor) component is damped -- the
        // vertical (normal) component is the wall's own no-penetration physics
        // (inner FrictionBoundary), not this mechanism's job. Matches the doc's own
        // claim above: "horizontal velocity zeroed", not the whole vector.
        ctx.v.x *= 1.0 - grip;
    }
}

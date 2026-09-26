//! Boundary conditions: the `BoundaryCondition` trait plus 5 real models.
//! The 3 Coulomb-friction variants (plain/grip/ratchet) are a real, tightly
//! related family -- grouped under `friction/` (see that module's own doc);
//! `heightmap`/`slip` are each standalone, one file apiece.
//!
//! Shared helpers (`apply_coulomb_wall`, `apply_slip_wall_velocity`,
//! `clamp_position_inside_grid`) and their direct unit tests live here,
//! since they're genuinely shared math, not any one model's own logic.

use glam::Vec2;

use crate::particle::ParticleUpdateCtx;

mod friction;
mod heightmap;
mod kinematic_obstacle;
mod slip;

pub use friction::{FrictionBoundary, GripFrictionBoundary, RatchetFrictionBoundary};
pub use heightmap::HeightmapBoundary;
pub use kinematic_obstacle::KinematicCircleBoundary;
pub use slip::SlipBoundary;

pub trait BoundaryCondition: Send + Sync + core::fmt::Debug {
    /// Correct one grid node's velocity, returning the specific kinetic
    /// energy this correction DISSIPATED as friction (grid velocity-squared
    /// units; multiply by `dx_meters^2` for J/kg).
    ///
    /// Returning it rather than discarding it is what lets the solver
    /// account for the energy instead of destroying it -- a frictional
    /// contact that silently removes kinetic energy violates the first law.
    /// A frictionless boundary returns 0, which is the honest answer and
    /// also the cheapest.
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32;
    /// Clamp particle position to the valid domain after G2P.
    /// Not a physical force -- last-resort domain enforcement so particles never escape the grid.
    /// Proper no-penetration physics lives in `apply_to_grid_velocity`.
    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2;
    /// Optional post-G2P per-particle hook (e.g. `GripFrictionBoundary`'s muscle
    /// grip). Takes a `ParticleUpdateCtx`, not `&mut Particles, i` -- same reason
    /// as `MaterialModel::update_particle`: only ever touches its own particle's
    /// fields, so G2P can run every particle's boundary hook in parallel too.
    fn post_g2p_particle(&self, _ctx: &mut ParticleUpdateCtx, _grid_res: usize, _dt: f32) {}

    /// Whether this boundary has a declared compatible wall discretisation
    /// for strict WC-MPM liquids. The conservative default is false: a
    /// post-G2P particle mutation is not automatically a fluid traction or
    /// no-penetration condition. Implementors must opt in explicitly.
    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        false
    }

    /// Real local contact normal + penetration overlap for a grain of the
    /// given `radius` at `position`, if it's touching this boundary's
    /// surface -- lets grain-vs-terrain contact produce a real rolling
    /// torque via `grain_contact_law::resolve_wall_contact`, the same real
    /// physics grain-vs-grain contact already has. Found missing live
    /// 2026-08-21 (see `heightmap.rs`'s own doc for the full story): without
    /// this, a grain resting on ANY boundary has literally no mechanism to
    /// ever start rolling from rest.
    ///
    /// `normal` must point AWAY from the surface (toward free space, where
    /// the grain is); `overlap = radius - distance_to_surface`. Default:
    /// `None` (no rolling torque from this boundary) -- safe and backward
    /// compatible; a boundary that's mostly outer box walls a grain rarely
    /// embeds into the way it rests on terrain doesn't need to opt in.
    fn grain_contact(
        &self,
        _position: Vec2,
        _radius: f32,
        _grid_res: usize,
    ) -> Option<(Vec2, f32)> {
        None
    }

    /// Real position backstop for a GRAIN specifically (radius-aware),
    /// after its own velocity integration -- see `clamp_particle_position`'s
    /// own doc for why a backstop exists at all (last-resort domain
    /// enforcement, not physics). Default just delegates to `clamp_particle_
    /// position` -- correct for any boundary with no grain-specific contact
    /// logic (e.g. `FrictionBoundary`, which only ever had the generic,
    /// point-particle clamp to begin with).
    ///
    /// `HeightmapBoundary` overrides this (see that file's own doc): its
    /// generic `clamp_particle_position` hardcodes a "+1" vertical
    /// clearance and checks straight-down `y`, ignoring both the grain's
    /// real radius and a sloped surface's own tilt. An earlier attempt
    /// (2026-08-21) tried layering a `grain_contact`-based correction on
    /// TOP of `clamp_particle_position`'s own terrain clamp, gated by a
    /// `grain_contact().is_some()` check -- but `grain_contact`'s `None` is
    /// ambiguous (it means both "no grain logic at all" and "genuinely not
    /// touching," and the generic clamp's own OVER-correction made the
    /// second case indistinguishable from the first, an endless fight
    /// between two disagreeing notions of "on the surface" that silently
    /// regressed outer-wall containment too when worked around with a bare
    /// boolean flag). This single method is the real fix: each boundary
    /// fully owns its own grain backstop end to end, no ambiguity to
    /// resolve at the call site.
    fn clamp_grain_position(&self, position: Vec2, radius: f32, grid_res: usize) -> Vec2 {
        let _ = radius;
        self.clamp_particle_position(position, grid_res)
    }

    /// Same as `apply_to_grid_velocity`, with an optional real, per-node
    /// Material-Induced Boundary Friction coefficient (`Grid::
    /// node_friction_at_index`, `None` when no friction-reporting particle
    /// touched this node) available for boundaries that want it -- see
    /// `MaterialModel::current_friction_coefficient`'s own doc for the real
    /// citation (Blatny & Gaume 2025). Default: ignore `node_friction`
    /// entirely and delegate to the ordinary method -- safe and backward
    /// compatible, the same "default no-op" shape `grain_contact`/
    /// `clamp_grain_position` already use above. Only `FrictionBoundary`
    /// overrides this; every other boundary keeps this default untouched.
    fn apply_to_grid_velocity_with_node_friction(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
        _node_friction: Option<f32>,
    ) -> f32 {
        self.apply_to_grid_velocity(cell_index, grid_res, velocity)
    }

    /// Real Newton's-third-law reaction hook: called once per grid cell
    /// this boundary's own `apply_to_grid_velocity[_with_node_friction]`
    /// actually corrected, with the real mass-weighted impulse the GRID
    /// lost at that cell (`-mass * (v_after - v_before)`) -- what the grid
    /// lost, a kinematically-driven obstacle boundary gains, letting it
    /// genuinely feel the fluid/solid it's pushing instead of having
    /// effectively infinite mass. Default: no-op (zero cost for every
    /// existing boundary -- `SlipBoundary`/`FrictionBoundary`/etc. are
    /// static geometry with nothing to accumulate into). Only a boundary
    /// that represents a real, externally-driven moving body (e.g. a
    /// kinematic obstacle) overrides this.
    fn on_grid_correction(&self, _cell_pos: Vec2, _reaction_impulse: Vec2) {}
}

/// Delegating impl so an `Arc<T>` can be boxed as a `BoundaryCondition` directly --
/// lets a caller keep its OWN clone of the `Arc` (e.g. to call
/// `RatchetFrictionBoundary::set_easy_direction` from a game loop) while the same
/// underlying instance is also installed on the solver, sharing state instead of
/// copying it.
impl<T: BoundaryCondition + ?Sized> BoundaryCondition for std::sync::Arc<T> {
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        (**self).apply_to_grid_velocity(cell_index, grid_res, velocity)
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        (**self).clamp_particle_position(position, grid_res)
    }

    fn post_g2p_particle(&self, ctx: &mut ParticleUpdateCtx, grid_res: usize, dt: f32) {
        (**self).post_g2p_particle(ctx, grid_res, dt);
    }

    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        (**self).is_strict_wc_mpm_fluid_compatible()
    }

    fn grain_contact(&self, position: Vec2, radius: f32, grid_res: usize) -> Option<(Vec2, f32)> {
        (**self).grain_contact(position, radius, grid_res)
    }

    fn clamp_grain_position(&self, position: Vec2, radius: f32, grid_res: usize) -> Vec2 {
        (**self).clamp_grain_position(position, radius, grid_res)
    }

    fn apply_to_grid_velocity_with_node_friction(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
        node_friction: Option<f32>,
    ) -> f32 {
        (**self).apply_to_grid_velocity_with_node_friction(
            cell_index,
            grid_res,
            velocity,
            node_friction,
        )
    }

    fn on_grid_correction(&self, cell_pos: Vec2, reaction_impulse: Vec2) {
        (**self).on_grid_correction(cell_pos, reaction_impulse);
    }
}

/// Apply Coulomb wall friction along one wall face.
///
/// `outward_normal`: unit vector pointing away from the wall into the domain.
/// When the velocity has a component moving INTO the wall (v · outward_normal < 0),
/// zero the normal component and damp the tangential component by µ × |v_normal|.
/// Returns the specific kinetic energy this Coulomb contact DISSIPATED,
/// `0.5 * (|v_t|^2 - |v_t_after|^2)`, in the grid's own velocity-squared
/// units (multiply by `dx_meters^2` for J/kg).
///
/// Only the TANGENTIAL part is reported, and that is deliberate. Sliding
/// friction doing work against a surface is unambiguously dissipation and
/// becomes heat, and Coulomb's own law says how much. The normal component
/// this function also removes is a no-penetration constraint against an
/// idealised rigid wall; where that energy goes in reality (wall
/// deformation, sound, heat on both sides) is a modelling choice this
/// engine does not make, so it is not reported as frictional heat.
///
/// Callers that do not care may ignore the value. Recording it is what
/// lets `energy::thermodynamics::frictional_heating` close the first law
/// instead of letting the energy vanish.
pub(crate) fn apply_coulomb_wall(velocity: &mut Vec2, outward_normal: Vec2, mu: f32) -> f32 {
    let v_n_scalar = velocity.dot(outward_normal);
    // Only act when moving into the wall.
    if v_n_scalar >= 0.0 {
        return 0.0;
    }
    let normal_speed = v_n_scalar.abs();
    let v_t = *velocity - v_n_scalar * outward_normal;
    let v_t_len = v_t.length();
    let friction_impulse = mu * normal_speed;
    // Tangential speed after friction: max(|v_t| - mu|v_n|, 0), direction kept.
    let v_t_after = (v_t_len - friction_impulse).max(0.0);
    *velocity = if v_t_len > friction_impulse {
        v_t * (v_t_after / v_t_len)
    } else {
        Vec2::ZERO
    };
    0.5 * (v_t_len * v_t_len - v_t_after * v_t_after)
}

pub(crate) const fn apply_slip_wall_velocity(
    thickness: usize,
    cell_index: usize,
    grid_res: usize,
    velocity: &mut Vec2,
) {
    let hi = grid_res - (thickness + 1);
    let x = cell_index / grid_res;
    let y = cell_index % grid_res;
    // Only block the inward component -- let outward (escape) velocity pass through.
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

pub(crate) fn clamp_position_inside_grid(
    thickness: usize,
    position: Vec2,
    grid_res: usize,
) -> Vec2 {
    let min = thickness.saturating_sub(1) as f32;
    let max = grid_res.saturating_sub(thickness) as f32;
    position.clamp(Vec2::splat(min), Vec2::splat(max))
}

#[cfg(test)]
mod boundary_physics_tests {
    use super::*;

    /// The defining property of a frictionless slip wall: tangential velocity
    /// passes through completely UNCHANGED (only the inward normal component
    /// is blocked). No existing test checked this precisely -- only whole-
    /// simulation "particles stay inside the domain" tests exist, which don't
    /// isolate this specific claim.
    #[test]
    fn slip_wall_preserves_tangential_velocity_exactly() {
        // x=0 (left wall zone), y=32 (mid-grid, clear of every other wall) --
        // isolates the left wall's check alone, avoids corner-cell double-hits.
        let mut v = Vec2::new(-3.0, 7.5); // moving into the wall, tangential=7.5
        apply_slip_wall_velocity(2, /* cell_index for x=0,y=32 */ 32, 64, &mut v);
        assert_eq!(
            v.y, 7.5,
            "tangential (Y) component must pass through exactly unchanged"
        );
        assert!(
            v.x >= 0.0,
            "inward (X) component must be blocked (>= 0, not still negative)"
        );
    }

    /// Outward-moving velocity (already leaving the wall) must be completely
    /// untouched by a slip wall -- "no-penetration" only blocks entry, it must
    /// never resist or clamp an escaping particle's velocity.
    #[test]
    fn slip_wall_does_not_touch_outward_velocity() {
        // x=0 (left wall zone), y=32 (mid-grid, clear of every other wall) --
        // isolates the left wall's check alone, avoids corner-cell double-hits.
        let mut v = Vec2::new(4.0, -2.0); // moving AWAY from the left wall
        apply_slip_wall_velocity(2, /* cell_index for x=0,y=32 */ 32, 64, &mut v);
        assert_eq!(
            v,
            Vec2::new(4.0, -2.0),
            "outward velocity must be completely untouched"
        );
    }

    /// mu=0 must behave IDENTICALLY to a pure slip wall -- FrictionBoundary's own
    /// doc comment claims this ("friction_coefficient = 0.0 -> pure slip (same as
    /// SlipBoundary)") but nothing verified it precisely until now.
    #[test]
    fn coulomb_wall_at_zero_friction_matches_pure_slip() {
        let mut v_friction = Vec2::new(-3.0, 7.5);
        apply_coulomb_wall(&mut v_friction, Vec2::X, 0.0);

        let mut v_slip = Vec2::new(-3.0, 7.5);
        apply_slip_wall_velocity(2, 0, 64, &mut v_slip);

        assert_eq!(
            v_friction.y, v_slip.y,
            "mu=0 tangential result must match pure slip exactly"
        );
        assert_eq!(
            v_friction.x, 0.0,
            "normal component always fully zeroed on impact"
        );
    }

    /// Real Coulomb friction law: tangential speed is reduced by EXACTLY
    /// mu * |v_normal| (not more, not less), direction preserved -- this is the
    /// actual physical law (friction force proportional to normal force), not
    /// just "friction slows things down somewhat."
    #[test]
    fn coulomb_wall_reduces_tangential_speed_by_exactly_mu_times_normal_speed() {
        let v_n = 4.0_f32; // normal speed into the wall
        let v_t = 10.0_f32; // tangential speed
        let mu = 0.3_f32;
        let mut v = Vec2::new(-v_n, v_t);
        apply_coulomb_wall(&mut v, Vec2::X, mu);

        let expected_v_t = v_t - mu * v_n; // = 10.0 - 1.2 = 8.8
        assert!(
            (v.y - expected_v_t).abs() < 1.0e-5,
            "tangential speed after friction should be exactly v_t - mu*v_n = {expected_v_t}, got {}",
            v.y
        );
        assert_eq!(v.x, 0.0, "normal component always fully zeroed on impact");
    }

    /// Real Coulomb friction can only decelerate, never reverse direction --
    /// once tangential speed would go negative, it clamps to exactly zero
    /// (a particle can't be pushed backward by its own friction).
    #[test]
    fn coulomb_wall_never_reverses_tangential_direction() {
        let mut v = Vec2::new(-10.0, 2.0); // huge normal speed, tiny tangential
        apply_coulomb_wall(&mut v, Vec2::X, 0.9); // friction_impulse = 9.0 > v_t=2.0
        assert_eq!(
            v,
            Vec2::ZERO,
            "when friction impulse exceeds tangential speed, result must be exactly zero, \
             never a reversed/negative tangential velocity"
        );
    }

    /// Outward-moving velocity must be completely untouched by Coulomb friction
    /// too, same as the slip wall -- friction only applies to genuine impacts.
    #[test]
    fn coulomb_wall_does_not_touch_outward_velocity() {
        let mut v = Vec2::new(4.0, -2.0); // moving away from the wall
        apply_coulomb_wall(&mut v, Vec2::X, 0.9);
        assert_eq!(
            v,
            Vec2::new(4.0, -2.0),
            "outward velocity must be completely untouched"
        );
    }
}

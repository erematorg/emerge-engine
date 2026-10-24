//! Boundary conditions: the `BoundaryCondition` trait plus 6 models.
//! The 2 Coulomb-friction walls (isotropic and directional) are grouped
//! under `friction/` with the directional law for body-to-body contact
//! (`DirectionalContactGrip`), see that module's doc;
//! `heightmap`/`kinematic_obstacle`/`slip`/`static_box` are each
//! standalone, one file apiece.
//! `signorini_coulomb` states the contact law itself (push but never
//! pull, Coulomb friction) for the solvers that enforce it on their own
//! unknowns.
//!
//! Every wall acts at the grid nodes it contains; grid node `i` sits at
//! `i + 0.5` (`grid::kernel::quadratic_weights`).
//!
//! Shared helpers (`apply_coulomb_wall`, `apply_slip_wall_velocity`,
//! `apply_sealed_wall_velocity`, `clamp_position_inside_grid`,
//! `assert_friction_coefficient`) and their direct unit tests live here,
//! since they're shared math, not any one model's own logic.

use glam::Vec2;

/// Walls on the domain's four sides, as the contact law sees them: planes
/// `thickness` cells in from each edge (where every outer wall's plane
/// sits, `position_clamp_bounds`), matter pushed but never pulled, and
/// Coulomb's coefficient along them (`signorini_coulomb`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainWalls {
    pub thickness: usize,
    pub friction: f32,
}

mod friction;
mod heightmap;
mod kinematic_obstacle;
pub mod signorini_coulomb;
mod slip;
mod static_box;

pub use friction::{DirectionalContactGrip, FrictionBoundary, RatchetFrictionBoundary};
pub use heightmap::HeightmapBoundary;
pub use kinematic_obstacle::KinematicCircleBoundary;
pub use slip::SlipBoundary;
pub use static_box::StaticBoxBoundary;

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

    /// Whether this boundary has a declared compatible wall discretisation
    /// for strict WC-MPM liquids, verified by a strict-fluid scene run
    /// against it. The conservative default is false; implementors opt in
    /// explicitly.
    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        false
    }

    /// The domain's walls as the contact law sees them (`DomainWalls`),
    /// for a solver that holds them by contact on its own unknowns (the
    /// total Lagrangian solid step). `None`: this boundary has no such
    /// form yet, and that solver refuses it.
    fn domain_walls(&self) -> Option<DomainWalls> {
        None
    }

    /// Local contact normal and penetration overlap for a grain of `radius` at
    /// `position`, if it touches this boundary's surface. Lets grain-vs-terrain
    /// contact produce a rolling torque via
    /// `grain_contact_law::resolve_wall_contact`, as grain-vs-grain contact
    /// does; without it a grain resting on a boundary cannot start rolling
    /// from rest (see `heightmap.rs`).
    ///
    /// `normal` must point AWAY from the surface (toward free space, where
    /// the grain is); `overlap = radius - distance_to_surface`. Default:
    /// `None` (no rolling torque from this boundary), for boundaries such as
    /// outer box walls that a grain rarely rests on.
    fn grain_contact(
        &self,
        _position: Vec2,
        _radius: f32,
        _grid_res: usize,
    ) -> Option<(Vec2, f32)> {
        None
    }

    /// Radius-aware position backstop for a grain, after its velocity
    /// integration (see `clamp_particle_position` for why a backstop exists:
    /// last-resort domain enforcement, not physics). The default delegates to
    /// `clamp_particle_position`, right for a boundary with no grain-specific
    /// contact logic (e.g. `FrictionBoundary`).
    ///
    /// `HeightmapBoundary` overrides it: its generic `clamp_particle_position`
    /// uses a fixed "+1" vertical clearance and checks straight-down `y`,
    /// ignoring the grain's radius and the slope. Each boundary owns its
    /// grain backstop end to end here, rather than the caller layering a
    /// `grain_contact`-based correction over the generic clamp: `None` from
    /// `grain_contact` means both "no grain logic" and "not touching", so the
    /// call site cannot tell the two apart.
    fn clamp_grain_position(&self, position: Vec2, radius: f32, grid_res: usize) -> Vec2 {
        let _ = radius;
        self.clamp_particle_position(position, grid_res)
    }

    /// Same as `apply_to_grid_velocity`, with an optional per-node
    /// Material-Induced Boundary Friction coefficient (`Grid::
    /// node_friction_at_index`, `None` when no friction-reporting particle
    /// touched this node) available for boundaries that want it -- see
    /// `MaterialModel::current_friction_coefficient`'s doc for the real
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

    /// Equal-and-opposite reaction hook: called once per grid cell this
    /// boundary's `apply_to_grid_velocity[_with_node_friction]` corrected,
    /// with the mass-weighted impulse the grid lost at that cell (`-mass *
    /// (v_after - v_before)`). A kinematically driven obstacle gains what the
    /// grid lost, so it feels the material it pushes. Default: no-op (static
    /// boundaries such as `SlipBoundary`/`FrictionBoundary` have nothing to
    /// accumulate into).
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
/// Only the TANGENTIAL part is reported. Sliding friction doing work against
/// a surface is dissipation and becomes heat, and Coulomb's own law says how
/// much. Removing the normal component makes the node's impact on the wall
/// perfectly inelastic: the kinetic energy `0.5 * v_n^2` per unit mass it
/// removes is a loss of the velocity-level discretization, not a modelled
/// physical process, and it is not reported anywhere yet (see
/// `forces/README.md`, Scope & Limits).
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

/// Anisotropic Coulomb friction along one wall face: the coefficient is
/// `mu_easy` when the tangential velocity points the same way as
/// `easy_direction`'s tangential component, `mu_resist` when it points against
/// it. This is the forward/backward part of the anisotropic friction law
/// measured on snake skin by Hu, Nirody, Scott & Shelley 2009 (PNAS
/// 106:10081, eq. 2); the transverse coefficient has no counterpart along a
/// one-dimensional wall tangent. No-penetration and the Coulomb reduction are
/// `apply_coulomb_wall`'s, and so is the returned dissipated energy.
pub(crate) fn apply_anisotropic_coulomb_wall(
    velocity: &mut Vec2,
    outward_normal: Vec2,
    easy_direction: Vec2,
    mu_easy: f32,
    mu_resist: f32,
) -> f32 {
    let tangent = Vec2::new(-outward_normal.y, outward_normal.x);
    let aligned = velocity.dot(tangent) * easy_direction.dot(tangent) >= 0.0;
    let mu = if aligned { mu_easy } else { mu_resist };
    apply_coulomb_wall(velocity, outward_normal, mu)
}

/// Panics unless `mu` is a valid Coulomb friction coefficient: finite and
/// `>= 0`. There is no upper bound: rubber on dry rough surfaces exceeds 1
/// (Persson, Tartaglino, Albohr & Tosatti 2005, Phys. Rev. B 71, 035428).
pub(crate) fn assert_friction_coefficient(mu: f32, name: &str) {
    assert!(
        mu.is_finite() && mu >= 0.0,
        "{name} must be a finite Coulomb friction coefficient >= 0, got {mu}"
    );
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

/// The wall nodes of `apply_slip_wall_velocity`, with the normal velocity
/// set to zero in BOTH directions: matter at a wall node can neither enter
/// the wall nor leave it. Tangential velocity passes through unchanged.
///
/// The one-sided rule above lets matter leave whenever its velocity points
/// away. The engine's equations of state give gauge pressure (zero at the
/// ambient state), so under that rule a material below ambient pressure
/// pulls itself off the wall as if ambient air stood behind it. Behind a
/// sealed wall there is no such air: a gas's absolute pressure stays
/// positive and it never leaves the wall, and a rarefaction reflects off
/// the wall as a rarefaction. Measured in
/// `tests/probes/gas_sound_speed.rs`: reflection coefficient +0.93 off this
/// wall, -0.84 off the one-sided one, where a rigid wall gives +1 and a
/// pressure-release surface -1.
pub(crate) const fn apply_sealed_wall_velocity(
    thickness: usize,
    cell_index: usize,
    grid_res: usize,
    velocity: &mut Vec2,
) {
    let hi = grid_res - (thickness + 1);
    let x = cell_index / grid_res;
    let y = cell_index % grid_res;
    if x < thickness || x > hi {
        velocity.x = 0.0;
    }
    if y < thickness || y > hi {
        velocity.y = 0.0;
    }
}

pub(crate) fn clamp_position_inside_grid(
    thickness: usize,
    position: Vec2,
    grid_res: usize,
) -> Vec2 {
    let (min, max) = position_clamp_bounds(thickness, grid_res);
    position.clamp(Vec2::splat(min), Vec2::splat(max))
}

/// The last-resort position bounds for outer walls `thickness` cells thick:
/// one cell past each wall plane, on every side alike.
///
/// The wall planes sit at `thickness` and `grid_res - thickness`
/// (`apply_slip_wall_velocity`'s wall nodes end half a cell before them).
/// The bounds used to be `thickness - 1` and `grid_res - thickness`: one
/// cell of room past the left wall and the floor, none past the right wall
/// and the ceiling, so a liquid at rest against the right wall sat on the
/// clamp (14 particles of a 3360-particle pool after 5 s) while it never
/// reached the left one (0.60 cells past its plane at most).
///
/// Both bounds also keep the quadratic stencil, `floor(x) - 1 ..= floor(x)
/// + 1`, inside the grid: `x >= 1` and `x < grid_res - 1`.
pub(crate) fn position_clamp_bounds(thickness: usize, grid_res: usize) -> (f32, f32) {
    let room = thickness.saturating_sub(1) as f32;
    let min = room.max(1.0);
    let max = (grid_res as f32 - room).min((grid_res as f32 - 1.0).next_down());
    (min, max)
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

    /// A sealed wall holds matter against it: the normal velocity is zeroed
    /// whether it points into the wall or away from it, and the tangential
    /// velocity is untouched, on every side of the grid.
    #[test]
    fn sealed_wall_zeroes_normal_velocity_in_both_directions() {
        let (thickness, res) = (2usize, 64usize);
        let index = |x: usize, y: usize| x * res + y;
        // (node, velocity in, expected out): left, right, bottom and top walls,
        // each away from the corners, once moving into the wall and once away.
        let cases = [
            (index(0, 32), Vec2::new(-3.0, 7.5), Vec2::new(0.0, 7.5)),
            (index(0, 32), Vec2::new(4.0, -2.0), Vec2::new(0.0, -2.0)),
            (index(63, 32), Vec2::new(3.0, 7.5), Vec2::new(0.0, 7.5)),
            (index(63, 32), Vec2::new(-4.0, -2.0), Vec2::new(0.0, -2.0)),
            (index(32, 0), Vec2::new(7.5, -3.0), Vec2::new(7.5, 0.0)),
            (index(32, 0), Vec2::new(-2.0, 4.0), Vec2::new(-2.0, 0.0)),
            (index(32, 63), Vec2::new(7.5, 3.0), Vec2::new(7.5, 0.0)),
            (index(32, 63), Vec2::new(-2.0, -4.0), Vec2::new(-2.0, 0.0)),
        ];
        for (cell, v_in, expected) in cases {
            let mut v = v_in;
            apply_sealed_wall_velocity(thickness, cell, res, &mut v);
            assert_eq!(v, expected, "cell {cell}, velocity in {v_in}");
        }
    }

    /// Away from the walls a sealed boundary does nothing.
    #[test]
    fn sealed_wall_leaves_interior_nodes_untouched() {
        let mut v = Vec2::new(-3.0, 7.5);
        apply_sealed_wall_velocity(2, 32 * 64 + 32, 64, &mut v);
        assert_eq!(v, Vec2::new(-3.0, 7.5));
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

    /// Coulomb friction: tangential speed drops by exactly mu * |v_normal|,
    /// direction preserved (friction force proportional to normal force).
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

    /// Coulomb friction only decelerates: once tangential speed would go
    /// negative it clamps to exactly zero, never reversing direction.
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
    /// too, same as the slip wall -- friction only applies to impacts.
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

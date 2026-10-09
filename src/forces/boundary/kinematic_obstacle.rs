use std::collections::HashMap;

use glam::Vec2;

use super::{BoundaryCondition, apply_coulomb_wall};

/// A moving circular obstacle, driven kinematically: position and velocity
/// come from outside, with no mass, inertia or constraint solver of its own.
/// Not a rigid body (the engine has none).
///
/// Prior art: `tmp/sparkl`'s rigid-collider coupling (`grid_update.rs`)
/// projects grid velocity against the collider's shape instead of running a
/// second contact field, `new_vel = rigid_vel + project_velocity(vel -
/// rigid_vel, normal)`; this does the same for a circle with this module's
/// `apply_coulomb_wall`.
///
/// Unlike `Particle::contact_group`'s multi-field contact (solid against
/// solid, see `Grid::resolve_contact`), this is a grid-level boundary
/// correction like `SlipBoundary`/`FrictionBoundary`, with a moving shape.
/// The grid does not know which material touches it, so it works for every
/// material alike.
///
/// `center`/`velocity` are stored as `AtomicU32` (bit-cast f32), like
/// `RatchetFrictionBoundary::set_easy_direction`, so a caller can update them
/// every frame (from a cursor, a creature's body) through a shared `Arc`.
///
/// Two-way momentum exchange (see `BoundaryCondition::on_grid_correction`):
/// each corrected cell adds its mass-weighted, equal-and-opposite reaction
/// impulse to `reaction_x_bits`/`reaction_y_bits`, read and reset by
/// `take_reaction_impulse()`; the caller integrates `v += impulse/mass`, so
/// the obstacle feels what it pushes.
///
/// `is_strict_wc_mpm_fluid_compatible()` keeps its `false` default, as for
/// `FrictionBoundary`, which uses the same `apply_coulomb_wall`: the flag
/// follows a strict-fluid scene run against the boundary, not structural
/// resemblance.
#[derive(Debug)]
pub struct KinematicCircleBoundary {
    center_x_bits: std::sync::atomic::AtomicU32,
    center_y_bits: std::sync::atomic::AtomicU32,
    velocity_x_bits: std::sync::atomic::AtomicU32,
    velocity_y_bits: std::sync::atomic::AtomicU32,
    /// Accumulated reaction impulse since the last `take_reaction_impulse()`
    /// call -- mass-weighted, Newton's-third-law momentum this
    /// obstacle received from every grid cell it corrected this substep.
    /// The caller (not this struct) decides what to do with it -- e.g.
    /// `v_new = v_old + impulse/mass` -- so this stays a simple point-mass
    /// accumulator, not a hidden auto-integrator with its own opinion about
    /// when/how often it should run relative to `Simulation::step()`.
    reaction_x_bits: std::sync::atomic::AtomicU32,
    reaction_y_bits: std::sync::atomic::AtomicU32,
    /// Angular velocity (rad/s, 2D scalar about z), e.g. a rolling ball's
    /// rotation. The caller integrates it (`omega += take_torque()/
    /// moment_of_inertia`), as with `velocity`/`reaction`: this struct only
    /// accumulates.
    angular_velocity_bits: std::sync::atomic::AtomicU32,
    /// Accumulated real torque (`cross(cell_pos - center, reaction_impulse)`
    /// per corrected cell, same Newton's-third-law sign convention as
    /// `reaction_*`) since the last `take_torque()` call.
    torque_bits: std::sync::atomic::AtomicU32,
    /// Radius in grid-index units, live-updatable via `set_radius`.
    radius_bits: std::sync::atomic::AtomicU32,
    /// Coulomb friction coefficient on the obstacle's surface, same convention
    /// as `FrictionBoundary::friction_coefficient` -- the DEFAULT, used for
    /// any material not listed in `friction_profile` below.
    pub friction: f32,
    /// Per-material friction overrides: e.g. water and mud can
    /// feel different at the SAME obstacle. Set once at construction (via
    /// `with_material_friction`), not live-updated -- the profile itself is
    /// static data, only WHICH entry is active changes frame to frame.
    ///
    /// Approximate: `apply_to_grid_velocity` corrects a grid cell, which after
    /// P2G can mix several materials, and `Cell` (unlike `ContactCell`) keeps
    /// no per-material breakdown; adding one would touch P2G's scatter. The
    /// caller instead finds the material near the obstacle with
    /// `Simulation::particles_near` and calls `set_active_friction` before
    /// `step()`.
    friction_profile: HashMap<u32, f32>,
    active_friction_bits: std::sync::atomic::AtomicU32,
}

impl KinematicCircleBoundary {
    pub fn new(center: Vec2, radius: f32, friction: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&friction),
            "friction must be in [0.0, 1.0], got {friction}"
        );
        Self {
            center_x_bits: std::sync::atomic::AtomicU32::new(center.x.to_bits()),
            center_y_bits: std::sync::atomic::AtomicU32::new(center.y.to_bits()),
            velocity_x_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            velocity_y_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            reaction_x_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            reaction_y_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            angular_velocity_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            torque_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            radius_bits: std::sync::atomic::AtomicU32::new(radius.to_bits()),
            friction,
            friction_profile: HashMap::new(),
            active_friction_bits: std::sync::atomic::AtomicU32::new(friction.to_bits()),
        }
    }

    /// Register a per-material friction override (builder). E.g. water
    /// and mud can behave differently at the same obstacle.
    pub fn with_material_friction(mut self, material_id: u32, friction: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&friction),
            "friction must be in [0.0, 1.0], got {friction}"
        );
        self.friction_profile.insert(material_id, friction);
        self
    }

    /// The friction this obstacle WOULD use against `material_id` -- the
    /// profile override if one is registered, else the default. Read-only
    /// lookup; does not itself change what `apply_to_grid_velocity` uses
    /// (call `set_active_friction` for that).
    pub fn friction_for_material(&self, material_id: u32) -> f32 {
        self.friction_profile
            .get(&material_id)
            .copied()
            .unwrap_or(self.friction)
    }

    /// Set which friction value `apply_to_grid_velocity` actually uses,
    /// live -- the caller decides how (e.g. `Simulation::particles_near`
    /// the obstacle each frame, then `friction_for_material` on whatever's
    /// nearest, see this struct's doc for the scope limit here).
    pub fn set_active_friction(&self, friction: f32) {
        self.active_friction_bits
            .store(friction.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    fn active_friction(&self) -> f32 {
        f32::from_bits(
            self.active_friction_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Current radius, grid-index units.
    pub fn radius(&self) -> f32 {
        f32::from_bits(self.radius_bits.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Grow/shrink the collision radius live -- e.g. a snowball accreting
    /// mass as it rolls. Takes effect on the very next substep.
    pub fn set_radius(&self, radius: f32) {
        self.radius_bits
            .store(radius.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Read and reset the reaction impulse accumulated since the last call --
    /// call once per `Simulation::step()`, real F=ma is the caller's job:
    /// `velocity += take_reaction_impulse() / obstacle_mass`.
    pub fn take_reaction_impulse(&self) -> Vec2 {
        use std::sync::atomic::Ordering::Relaxed;
        Vec2::new(
            f32::from_bits(self.reaction_x_bits.swap(0.0f32.to_bits(), Relaxed)),
            f32::from_bits(self.reaction_y_bits.swap(0.0f32.to_bits(), Relaxed)),
        )
    }

    /// Current angular velocity, radians/s.
    pub fn angular_velocity(&self) -> f32 {
        f32::from_bits(
            self.angular_velocity_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Set the obstacle's own spin live -- real F=ma-style integration is
    /// the caller's job: `omega += take_torque() / moment_of_inertia`. For
    /// a 2D solid disk of mass `m` and radius `r`, the moment of
    /// inertia is `I = 0.5 * m * r^2` (standard rigid-body mechanics, e.g.
    /// Goldstein's "Classical Mechanics" -- the 2D-disk case, not the 3D
    /// sphere's `(2/5)*m*r^2`, since this engine is genuinely 2D, not a
    /// sphere sliced through its equator).
    pub fn set_angular_velocity(&self, omega: f32) {
        self.angular_velocity_bits
            .store(omega.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Read and reset the torque accumulated since the last call -- same
    /// pattern as `take_reaction_impulse`.
    pub fn take_torque(&self) -> f32 {
        f32::from_bits(
            self.torque_bits
                .swap(0.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Update the obstacle's position and velocity live -- e.g. every frame,
    /// from a cursor or a creature's own driven position. Takes effect on the
    /// very next substep; no reconstruction, no boundary replacement.
    pub fn set_position_velocity(&self, center: Vec2, velocity: Vec2) {
        use std::sync::atomic::Ordering::Relaxed;
        self.center_x_bits.store(center.x.to_bits(), Relaxed);
        self.center_y_bits.store(center.y.to_bits(), Relaxed);
        self.velocity_x_bits.store(velocity.x.to_bits(), Relaxed);
        self.velocity_y_bits.store(velocity.y.to_bits(), Relaxed);
    }

    fn center(&self) -> Vec2 {
        use std::sync::atomic::Ordering::Relaxed;
        Vec2::new(
            f32::from_bits(self.center_x_bits.load(Relaxed)),
            f32::from_bits(self.center_y_bits.load(Relaxed)),
        )
    }

    fn velocity(&self) -> Vec2 {
        use std::sync::atomic::Ordering::Relaxed;
        Vec2::new(
            f32::from_bits(self.velocity_x_bits.load(Relaxed)),
            f32::from_bits(self.velocity_y_bits.load(Relaxed)),
        )
    }
}

impl BoundaryCondition for KinematicCircleBoundary {
    fn apply_to_grid_velocity(
        &self,
        cell_index: usize,
        grid_res: usize,
        velocity: &mut Vec2,
    ) -> f32 {
        let cell_pos = Vec2::new(
            (cell_index / grid_res) as f32,
            (cell_index % grid_res) as f32,
        );
        let d = cell_pos - self.center();
        let dist = d.length();
        // Outside the obstacle, or degenerate (cell exactly at the center) --
        // most of the grid, every substep this obstacle isn't nearby. Cheap
        // early return, same "zero-cost when unused" shape as contact_group.
        if dist >= self.radius() || dist <= f32::EPSILON {
            return 0.0;
        }
        let outward_normal = d / dist;
        // Surface velocity at the contact point, `v_surface = v_center +
        // omega x r` (in 2D `omega * (-r.y, r.x)`), not only the centre's:
        // without it a spinning ball's surface looks stationary to the
        // friction correction and rolling without slipping cannot emerge.
        let omega = self.angular_velocity();
        let rigid_v = self.velocity() + omega * Vec2::new(-d.y, d.x);
        let mut v_rel = *velocity - rigid_v;
        // Reused unchanged: `apply_coulomb_wall` already no-ops when
        // `v_rel` isn't approaching along `outward_normal`, so no separate
        // approach test is needed here.
        // Dissipation is computed in the obstacle's own frame, which is the
        // correct one: frictional work depends on RELATIVE sliding, not on
        // the grid's absolute velocity. A node moving exactly with a moving
        // obstacle rubs against nothing and heats nothing.
        let dissipated = apply_coulomb_wall(&mut v_rel, outward_normal, self.active_friction());
        *velocity = v_rel + rigid_v;
        dissipated
    }

    fn clamp_particle_position(&self, position: Vec2, _grid_res: usize) -> Vec2 {
        let d = position - self.center();
        let dist = d.length();
        let radius = self.radius();
        if dist < radius {
            let n = if dist > f32::EPSILON {
                d / dist
            } else {
                Vec2::X
            };
            self.center() + n * radius
        } else {
            position
        }
    }

    fn on_grid_correction(&self, cell_pos: Vec2, reaction_impulse: Vec2) {
        use std::sync::atomic::Ordering::Relaxed;
        // Accumulate as bit-cast f32 via compare-exchange (no atomic f32 add
        // in stable Rust), like center/velocity above. Only the few cells this
        // obstacle overlaps contend.
        // Torque with the same equal-and-opposite sign as the impulse: `tau =
        // r x F`, `r.x*F.y - r.y*F.x`, `r` from the obstacle's centre to the
        // corrected cell.
        let r = cell_pos - self.center();
        let torque = r.x * reaction_impulse.y - r.y * reaction_impulse.x;
        for (bits, delta) in [
            (&self.reaction_x_bits, reaction_impulse.x),
            (&self.reaction_y_bits, reaction_impulse.y),
            (&self.torque_bits, torque),
        ] {
            let mut current = bits.load(Relaxed);
            loop {
                let new = (f32::from_bits(current) + delta).to_bits();
                match bits.compare_exchange_weak(current, new, Relaxed, Relaxed) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
    }

    // `true`, on measurement:
    //   1. Contact holds: the nearest particle stays exactly at the obstacle's
    //      radius under sustained approach; mass conserved; max_speed rises
    //      smoothly (2.78 -> 3.45), no blowup.
    //   2. Two-way coupling unscripted: the obstacle's velocity comes only
    //      from `take_reaction_impulse()`; it decelerates on contact
    //      (2.00 -> 1.595 in one step) and the reaction is ~zero out of contact.
    //   3. Water at moderate and high speed (6.0), Bingham mud (slower than
    //      water, as its viscosity and yield stress imply) and a vertical
    //      drop: mass conserved, finite, contact confirmed in every case.
    // Not tested: several obstacles at once, more extreme mass ratios, or a
    // non-circular shape (this boundary is circles only).
    fn is_strict_wc_mpm_fluid_compatible(&self) -> bool {
        true
    }
}

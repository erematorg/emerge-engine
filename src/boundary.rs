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
    /// Coulomb coefficient when floor-tangential velocity aligns with `easy_direction`.
    pub mu_easy: f32,
    /// Coulomb coefficient when it opposes `easy_direction` — the "anchor" side.
    pub mu_resist: f32,
    easy_dir_x_bits: std::sync::atomic::AtomicU32,
    easy_dir_y_bits: std::sync::atomic::AtomicU32,
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
            mu_easy,
            mu_resist,
            easy_dir_x_bits: std::sync::atomic::AtomicU32::new(d.x.to_bits()),
            easy_dir_y_bits: std::sync::atomic::AtomicU32::new(d.y.to_bits()),
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
        let mu_side = 0.5 * (self.mu_easy + self.mu_resist);
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
                    self.mu_easy
                } else {
                    self.mu_resist
                };
                apply_coulomb_wall(velocity, Vec2::Y, mu);
            }
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        clamp_position_inside_grid(self.thickness, position, grid_res)
    }
}

/// Delegating impl so an `Arc<T>` can be boxed as a `BoundaryCondition` directly —
/// lets a caller keep its OWN clone of the `Arc` (e.g. to call
/// `RatchetFrictionBoundary::set_easy_direction` from a game loop) while the same
/// underlying instance is also installed on the solver, sharing state instead of
/// copying it.
impl<T: BoundaryCondition + ?Sized> BoundaryCondition for std::sync::Arc<T> {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        (**self).apply_to_grid_velocity(cell_index, grid_res, velocity);
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        (**self).clamp_particle_position(position, grid_res)
    }

    fn post_g2p_particle(&self, particles: &mut Particles, i: usize, grid_res: usize, dt: f32) {
        (**self).post_g2p_particle(particles, i, grid_res, dt);
    }
}

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

    fn post_g2p_particle(&self, particles: &mut Particles, i: usize, _grid_res: usize, _dt: f32) {
        if self.grip_gain <= 0.0 {
            return;
        }
        // Boundary layer margin: wider than the wall's own solid region, since a
        // resting body settles with its lowest particles just ABOVE the wall
        // threshold (clamp_particle_position keeps them out of the solid cells),
        // not literally inside it. A contracting segment needs to reach the ground
        // to anchor, so the grip zone must cover that resting layer.
        let t = self.inner.thickness as f32 + 2.0;
        let x = particles.x[i];
        let hi = _grid_res as f32 - t;
        let near_wall = x.x < t || x.x > hi || x.y < t || x.y > hi;
        if !near_wall || particles.activation[i] <= 0.0 {
            return;
        }
        // Fiber-aligned strain rate: n0 . (velocity_gradient . n0) is the rate of
        // change of length per unit length along the fiber direction. Negative =
        // shortening (contracting → anchor phase); positive = lengthening
        // (extending → release phase, must NOT grip or the segment can never slide).
        let n = particles.activation_dir[i];
        let len_sq = n.dot(n);
        if len_sq <= f32::EPSILON {
            return;
        }
        let n0 = n / len_sq.sqrt();
        let strain_rate = n0.dot(particles.velocity_gradient[i] * n0);
        let contracting = (-strain_rate).clamp(0.0, 1.0);
        let grip = (self.grip_gain * particles.activation[i] * contracting).clamp(0.0, 1.0);
        particles.v[i] *= 1.0 - grip;
    }
}

/// Heightmap terrain boundary — arbitrary ground profile + outer box walls.
///
/// The terrain is described by `heights[x]` in grid units for each x-column.
/// All grid cells at (x, y) with `y ≤ heights[x]` are treated as solid terrain.
/// The terrain surface normal is +Y (pointing up). Coulomb friction is applied on the
/// tangential (horizontal) velocity component at the surface.
///
/// Outer axis-aligned walls are always enforced (same as `SlipBoundary`), so the
/// heightmap sits inside the standard simulation domain.
///
/// # Coordinate convention
/// Y increases upward. `heights[0]` is the left column, `heights[grid_res-1]` is the right.
/// Heights beyond the array length clamp to the last value.
///
/// # Usage
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::HeightmapBoundary;
/// // Flat floor at y=3, with a hill at column 20–40 rising to y=10
/// let mut heights = vec![3.0f32; 64];
/// for x in 20..40 { heights[x] = 3.0 + (10.0 - 3.0) * (1.0 - ((x as f32 - 30.0) / 10.0).abs()); }
/// let boundary = HeightmapBoundary::new(heights, 0.4, 2);
/// ```
#[derive(Debug, Clone)]
pub struct HeightmapBoundary {
    /// Terrain surface height in grid cells for each x-column. Fractional values are supported.
    pub heights: Vec<f32>,
    /// Coulomb friction coefficient on the terrain surface. 0.0 = slip, 1.0 = full friction.
    pub friction: f32,
    /// Thickness of outer box walls (standard MPM boundary padding).
    pub wall_thickness: usize,
}

impl HeightmapBoundary {
    pub fn new(heights: Vec<f32>, friction: f32, wall_thickness: usize) -> Self {
        Self {
            heights,
            friction,
            wall_thickness,
        }
    }

    /// Flat floor at a constant height — equivalent to a floor-only boundary.
    pub fn flat_floor(grid_res: usize, floor_height: f32, friction: f32) -> Self {
        Self::new(vec![floor_height; grid_res], friction, 2)
    }

    /// Sample the terrain height at grid column x. Clamps to array bounds.
    #[inline]
    fn height_at(&self, x: usize) -> f32 {
        if self.heights.is_empty() {
            return 0.0;
        }
        self.heights[x.min(self.heights.len() - 1)]
    }
}

impl BoundaryCondition for HeightmapBoundary {
    fn apply_to_grid_velocity(&self, cell_index: usize, grid_res: usize, velocity: &mut Vec2) {
        let x = cell_index / grid_res;
        let y = cell_index % grid_res;
        let t = self.wall_thickness;
        let hi = grid_res.saturating_sub(t + 1);

        // Outer box walls — standard slip (no-penetration, free tangential).
        if x < t {
            velocity.x = velocity.x.max(0.0);
        }
        if x > hi {
            velocity.x = velocity.x.min(0.0);
        }
        if y > hi {
            velocity.y = velocity.y.min(0.0);
        }

        // Heightmap terrain: cells at or below terrain surface.
        let terrain_h = self.height_at(x);
        if (y as f32) <= terrain_h {
            // Block downward (into terrain) velocity component.
            if velocity.y < 0.0 {
                let v_n = velocity.y.abs();
                velocity.y = 0.0;
                // Coulomb friction on tangential (horizontal) component.
                if self.friction > 0.0 {
                    let friction_impulse = self.friction * v_n;
                    let v_t = velocity.x.abs();
                    velocity.x = if v_t > friction_impulse {
                        velocity.x * (1.0 - friction_impulse / v_t)
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    fn clamp_particle_position(&self, position: Vec2, grid_res: usize) -> Vec2 {
        // Outer walls.
        let wall_min = self.wall_thickness.saturating_sub(1) as f32;
        let wall_max = grid_res.saturating_sub(self.wall_thickness) as f32;
        let mut pos = position.clamp(Vec2::splat(wall_min), Vec2::splat(wall_max));

        // Terrain: push particles above the surface.
        let x_col = (pos.x as usize).min(grid_res.saturating_sub(1));
        let terrain_h = self.height_at(x_col);
        if pos.y < terrain_h + 1.0 {
            pos.y = terrain_h + 1.0;
        }

        pos
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

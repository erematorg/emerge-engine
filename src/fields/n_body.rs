//! N-body gravity via Barnes-Hut quadtree.
//!
//! For discrete macro-scale sources (stars, planets) use `GravityWellField` — it's O(sources)
//! per particle and simpler. Use `NBodyGravityField` when MPM particles themselves gravitate
//! each other (planetary terrain, accretion disks) — Barnes-Hut reduces this from O(N²) to
//! O(N log N) per substep at the cost of one tree rebuild.
//!
//! # Physics
//! Plummer-softened gravity: **a = G·M·r̂ / (|r|² + ε²)^(3/2)**
//!
//! Barnes-Hut approximation: treat a cluster of bodies as a single body at their
//! center of mass when the cluster is "far enough" away — width/distance < θ (theta).
//! θ = 0.5 is a common Barnes-Hut trade-off value in practice (e.g. GADGET-2's
//! default region uses values in this neighborhood, Springel 2005, MNRAS
//! 364:1105 — not independently confirmed as a specific universal default
//! from that paper itself); lower is more accurate, higher is faster.
//!
//! # Reference
//! Barnes & Hut 1986 (Nature). Ported from `crates/forces/src/core/gravity_math.rs`,
//! adapted to 2D (Vec2) and emerge's coordinate system (grid units).

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;

// ── Quadtree defaults ───────────────────────────────────────────────────────
// These control tree quality vs. build cost. Both are configurable via the
// builder methods on NBodyGravityField.

/// Default maximum tree depth. 8 levels → 4^8 = 65536 leaf nodes — fine for up to ~100k bodies.
const DEFAULT_MAX_DEPTH: usize = 8;

/// Default maximum bodies per leaf node before the node is subdivided.
const DEFAULT_MAX_BODIES_PER_NODE: usize = 4;

/// Fallback AABB half-size used when no bodies are present (empty tree sentinel).
/// Arbitrary but non-zero so tree traversal never divides by zero.
const EMPTY_TREE_HALF_SIZE: f32 = 1000.0;

/// Fractional padding added around the tightest-fitting AABB before building the tree.
/// Ensures bodies near the boundary are never just outside the root node.
const AABB_PADDING_RATIO: f32 = 0.1;

/// Absolute minimum padding (grid cells). Prevents degenerate zero-size AABB
/// when all bodies are at the same position.
const AABB_PADDING_MIN: f32 = 1.0;

/// Bundled gravity parameters threaded through the quadtree traversal.
/// Groups theta, eps2, softening, and G to stay under the argument-count limit.
#[derive(Clone, Copy)]
struct GravParams {
    theta: f32,
    /// Plummer softening squared: ε² = softening².
    eps2: f32,
    softening: f32,
    g: f32,
}

// ---------------------------------------------------------------------------
// Quadtree data structures
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Aabb2 {
    center: Vec2,
    half_size: Vec2,
}

impl Aabb2 {
    fn new(center: Vec2, half_size: Vec2) -> Self {
        Self { center, half_size }
    }

    /// Returns quadrant index 0-3 for a point.
    /// Bit 0: right(1)/left(0). Bit 1: bottom(1)/top(0).
    fn quadrant_of(&self, point: Vec2) -> usize {
        ((point.x >= self.center.x) as usize) | (((point.y < self.center.y) as usize) << 1)
    }

    fn quadrant_aabb(&self, quadrant: usize) -> Aabb2 {
        let quarter = self.half_size * 0.5;
        let x_sign = if (quadrant & 1) == 0 { -1.0 } else { 1.0 };
        let y_sign = if (quadrant & 2) == 0 { 1.0 } else { -1.0 };
        Aabb2::new(
            self.center + Vec2::new(x_sign * quarter.x, y_sign * quarter.y),
            quarter,
        )
    }
}

#[derive(Clone, Debug, Default)]
struct MassProps {
    total_mass: f32,
    center_of_mass: Vec2,
}

impl MassProps {
    fn add(&mut self, position: Vec2, mass: f32) {
        let new_total = self.total_mass + mass;
        if new_total > 0.0 {
            self.center_of_mass =
                (self.center_of_mass * self.total_mass + position * mass) / new_total;
            self.total_mass = new_total;
        }
    }
}

struct Node {
    aabb: Aabb2,
    depth: usize,
    mass: MassProps,
    /// Bodies: (particle_index, position, mass).
    bodies: Vec<(usize, Vec2, f32)>,
    children: [Option<Box<Node>>; 4],
    max_depth: usize,
    max_bodies_per_node: usize,
}

impl Node {
    fn new(aabb: Aabb2, depth: usize, max_depth: usize, max_bodies_per_node: usize) -> Self {
        Self {
            aabb,
            depth,
            mass: MassProps::default(),
            bodies: Vec::new(),
            children: [None, None, None, None],
            max_depth,
            max_bodies_per_node,
        }
    }

    fn is_far_enough(&self, pos: Vec2, theta: f32, softening: f32) -> bool {
        let dist = (pos - self.aabb.center).length();
        // Skip Barnes-Hut approximation if we're inside or touching the softening radius —
        // at these distances the multipole expansion is inaccurate.
        if dist < softening || self.mass.total_mass <= 0.0 {
            return false;
        }
        let width = self.aabb.half_size.x * 2.0;
        width / dist < theta
    }

    fn insert(&mut self, idx: usize, pos: Vec2, mass: f32) {
        self.mass.add(pos, mass);

        if self.depth >= self.max_depth
            || (self.bodies.len() < self.max_bodies_per_node && self.children[0].is_none())
        {
            self.bodies.push((idx, pos, mass));
            return;
        }

        if self.children[0].is_none() {
            for i in 0..4 {
                self.children[i] = Some(Box::new(Node::new(
                    self.aabb.quadrant_aabb(i),
                    self.depth + 1,
                    self.max_depth,
                    self.max_bodies_per_node,
                )));
            }
            let existing = std::mem::take(&mut self.bodies);
            for (e, p, m) in existing {
                let q = self.aabb.quadrant_of(p);
                if let Some(child) = &mut self.children[q] {
                    child.insert(e, p, m);
                }
            }
        }

        let q = self.aabb.quadrant_of(pos);
        if let Some(child) = &mut self.children[q] {
            child.insert(idx, pos, mass);
        }
    }

    fn acceleration_on(&self, idx: usize, pos: Vec2, gp: GravParams) -> Vec2 {
        if self.is_far_enough(pos, gp.theta, gp.softening) {
            let r_vec = self.mass.center_of_mass - pos;
            let r2 = r_vec.length_squared();
            let norm_s = r2 + gp.eps2;
            let scale = gp.g * self.mass.total_mass / (norm_s * norm_s.sqrt());
            return if scale.is_finite() {
                r_vec * scale
            } else {
                Vec2::ZERO
            };
        }

        if self.children.iter().all(|c| c.is_none()) {
            let mut total = Vec2::ZERO;
            for &(bidx, bpos, bmass) in &self.bodies {
                if bidx == idx {
                    continue;
                }
                let r_vec = bpos - pos;
                let r2 = r_vec.length_squared();
                // Skip self-interaction or particles within the softening radius.
                if r2 < gp.eps2 {
                    continue;
                }
                let norm_s = r2 + gp.eps2;
                let scale = gp.g * bmass / (norm_s * norm_s.sqrt());
                if scale.is_finite() {
                    total += r_vec * scale;
                }
            }
            return total;
        }

        let mut total = Vec2::ZERO;
        for child in self.children.iter().flatten() {
            total += child.acceleration_on(idx, pos, gp);
        }
        total
    }
}

struct Quadtree {
    root: Node,
}

impl Quadtree {
    fn build(bodies: &[(usize, Vec2, f32)], max_depth: usize, max_bodies_per_node: usize) -> Self {
        if bodies.is_empty() {
            return Self {
                root: Node::new(
                    Aabb2::new(Vec2::ZERO, Vec2::splat(EMPTY_TREE_HALF_SIZE)),
                    0,
                    max_depth,
                    max_bodies_per_node,
                ),
            };
        }

        let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
        let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
        for &(_, pos, _) in bodies {
            min_x = min_x.min(pos.x);
            min_y = min_y.min(pos.y);
            max_x = max_x.max(pos.x);
            max_y = max_y.max(pos.y);
        }
        let pad = ((max_x - min_x) + (max_y - min_y)) * AABB_PADDING_RATIO + AABB_PADDING_MIN;
        min_x -= pad;
        min_y -= pad;
        max_x += pad;
        max_y += pad;

        let center = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
        let half_size = Vec2::splat(((max_x - min_x).max(max_y - min_y)) * 0.5);

        let mut tree = Self {
            root: Node::new(
                Aabb2::new(center, half_size),
                0,
                max_depth,
                max_bodies_per_node,
            ),
        };
        for &(idx, pos, mass) in bodies {
            tree.root.insert(idx, pos, mass);
        }
        tree
    }

    fn acceleration_on(&self, idx: usize, pos: Vec2, gp: GravParams) -> Vec2 {
        self.root.acceleration_on(idx, pos, gp)
    }
}

// ---------------------------------------------------------------------------
// Field implementation
// ---------------------------------------------------------------------------

/// N-body gravitational acceleration via Barnes-Hut quadtree (O(N log N)).
///
/// On each substep `prepare()` builds a quadtree from the current particle snapshot,
/// then `acceleration()` queries it per particle.
///
/// # When to use
/// - Particles gravitate each other (accretion, planetary terrain at LP planetary scale).
/// - Use `GravityWellField` instead for fixed point-mass sources — much cheaper.
///
/// # Parameters
/// - `gravitational_constant`: G in simulation units. Tune to your scale.
/// - `softening`: Plummer ε (grid cells). Prevents divergence as r→0. Typ. 0.5–2.0.
/// - `theta`: Barnes-Hut opening angle. Must be in (0, 1].
///   0.5 = a common trade-off value in practice; lower = more accurate; higher = faster.
pub struct NBodyGravityField {
    pub gravitational_constant: f32,
    pub softening: f32,
    /// Barnes-Hut opening angle θ ∈ (0, 1]. Controls accuracy vs. speed trade-off.
    pub theta: f32,

    /// Maximum quadtree depth. Deeper trees handle dense clusters better at higher build cost.
    /// Default: `DEFAULT_MAX_DEPTH` (8).
    pub max_depth: usize,

    /// Maximum bodies stored in a leaf node before it is subdivided.
    /// Smaller = finer tree (more accuracy, slower build). Default: `DEFAULT_MAX_BODIES_PER_NODE` (4).
    pub max_bodies_per_node: usize,

    // Internal — rebuilt each substep by prepare().
    tree: Option<Quadtree>,
    /// Snapshot of (particle_index, position, mass) used to build the tree.
    /// Filtered to particles with positive mass only.
    snapshot: Vec<(usize, Vec2, f32)>,
}

impl NBodyGravityField {
    pub fn new(gravitational_constant: f32, softening: f32, theta: f32) -> Self {
        assert!(
            theta > 0.0 && theta <= 1.0,
            "theta must be in (0, 1]; got {theta}. \
             0.5 is a common Barnes-Hut trade-off value in practice."
        );
        Self {
            gravitational_constant,
            softening,
            theta,
            max_depth: DEFAULT_MAX_DEPTH,
            max_bodies_per_node: DEFAULT_MAX_BODIES_PER_NODE,
            tree: None,
            snapshot: Vec::new(),
        }
    }
}

impl Field for NBodyGravityField {
    fn prepare(&mut self, particles: &crate::particle::Particles) {
        self.snapshot.clear();
        self.snapshot.extend(
            particles
                .indices()
                .filter(|&i| particles.mass[i] > 0.0)
                .map(|i| (i, particles.x[i], particles.mass[i])),
        );
        self.tree = Some(Quadtree::build(
            &self.snapshot,
            self.max_depth,
            self.max_bodies_per_node,
        ));
    }

    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        let Some(tree) = &self.tree else {
            return Vec2::ZERO;
        };
        let gp = GravParams {
            theta: self.theta,
            eps2: self.softening * self.softening,
            softening: self.softening,
            g: self.gravitational_constant,
        };
        tree.acceleration_on(i, particles.x[i], gp)
    }
}

/// Orbital-mechanics helpers for placing bodies in stable orbits under
/// [`NBodyGravityField`]. All take `g` explicitly — pass the same `gravitational_constant`
/// you gave the field. Units are simulation units (grid cells, cells/s).
///
/// Use these to seed initial velocities so spawned bodies orbit rather than fall in.
pub mod orbit {
    /// Circular orbital speed for pure Newtonian gravity: `v = √(G·M / r)`.
    ///
    /// Valid when softening ε ≪ r. For softened gravity at small r use
    /// [`circular_velocity_softened`].
    pub fn circular_velocity(g: f32, central_mass: f32, orbit_radius: f32) -> f32 {
        if orbit_radius <= f32::EPSILON {
            return 0.0;
        }
        (g * central_mass / orbit_radius).sqrt()
    }

    /// Circular orbital speed for Plummer-softened gravity.
    ///
    /// Matches the force law used by [`super::NBodyGravityField`]:
    /// `m·v²/r = G·M·m·r / (r²+ε²)^{3/2}` → `v = r·√(G·M / (r²+ε²)^{3/2})`.
    pub fn circular_velocity_softened(
        g: f32,
        central_mass: f32,
        orbit_radius: f32,
        softening: f32,
    ) -> f32 {
        let norm_s = orbit_radius * orbit_radius + softening * softening;
        if norm_s <= f32::EPSILON {
            return 0.0;
        }
        orbit_radius * (g * central_mass / (norm_s * norm_s.sqrt())).sqrt()
    }

    /// Escape speed: `v = √(2·G·M / r)`. Below this a body stays bound.
    pub fn escape_velocity(g: f32, central_mass: f32, distance: f32) -> f32 {
        if distance <= f32::EPSILON {
            return 0.0;
        }
        (2.0 * g * central_mass / distance).sqrt()
    }

    /// Vis-viva speed at `distance` on an elliptical orbit of given `eccentricity`.
    ///
    /// `at_periapsis = true` evaluates at closest approach, `false` at apoapsis.
    /// `v = √(μ·(2/r − 1/a))` with `μ = G·M` and `a` the semi-major axis.
    pub fn elliptical_velocity(
        g: f32,
        central_mass: f32,
        distance: f32,
        eccentricity: f32,
        at_periapsis: bool,
    ) -> f32 {
        if distance <= f32::EPSILON {
            return 0.0;
        }
        let mu = g * central_mass;
        let sign = if at_periapsis { 1.0 } else { -1.0 };
        let semimajor = distance / (1.0 - eccentricity * sign);
        (mu * (2.0 / distance - 1.0 / semimajor)).max(0.0).sqrt()
    }
}

#[cfg(test)]
mod orbit_tests {
    use super::orbit::*;

    #[test]
    fn escape_is_sqrt2_times_circular() {
        // v_escape = √2 · v_circular for the same G, M, r (unsoftened).
        let (g, m, r) = (1.0, 100.0, 10.0);
        let vc = circular_velocity(g, m, r);
        let ve = escape_velocity(g, m, r);
        assert!(
            (ve / vc - 2.0_f32.sqrt()).abs() < 1e-5,
            "ve/vc = {}",
            ve / vc
        );
    }

    #[test]
    fn circular_orbit_is_zero_eccentricity_ellipse() {
        // A circular orbit is an ellipse with e=0: vis-viva must match circular_velocity.
        let (g, m, r) = (0.5, 200.0, 25.0);
        let vc = circular_velocity(g, m, r);
        let ve = elliptical_velocity(g, m, r, 0.0, true);
        assert!((vc - ve).abs() < 1e-4, "vc={vc} ve={ve}");
    }

    #[test]
    fn zero_radius_is_safe() {
        assert_eq!(circular_velocity(1.0, 1.0, 0.0), 0.0);
        assert_eq!(escape_velocity(1.0, 1.0, 0.0), 0.0);
    }
}

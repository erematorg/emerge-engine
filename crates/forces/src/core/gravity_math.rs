use self::spatial::QuadtreeNode;
/// Pure gravitational math — no Bevy ECS, no Entity references.
///
/// Target: migrate to emerge as the `ForceField` math kernel.
/// The ECS wrapper in `gravity.rs` stages entity data into the types here,
/// calls these functions, then writes results back into `AppliedForce`.
use bevy::math::Vec3;

pub const DEFAULT_GRAVITATIONAL_CONSTANT: f32 = 0.1;
/// Practical LP-0 guideline for exact mutual O(N²) gravity in realtime.
pub const MUTUAL_REALTIME_BODY_LIMIT: usize = 100;

/// Plummer-softened gravitational force: F = G·m₁·m₂·r / (r²+ε²)^(3/2).
///
/// Proper gradient of the Plummer potential Φ = -G·M/√(r²+ε²).
/// F→0 as r→0. Combined formula avoids a separate normalize call.
pub fn pair_force_vector(
    source_pos: Vec3,
    source_mass: f32,
    affected_pos: Vec3,
    affected_mass: f32,
    gravitational_constant: f32,
    softening_squared: f32,
) -> Option<Vec3> {
    let direction = source_pos - affected_pos;
    let distance_squared = direction.length_squared();
    if distance_squared <= f32::EPSILON {
        return None;
    }

    let norm_s = distance_squared + softening_squared;
    let force_scalar =
        gravitational_constant * source_mass * affected_mass / (norm_s * norm_s.sqrt());
    if !force_scalar.is_finite() {
        return None;
    }

    Some(direction * force_scalar)
}

/// Circular orbital velocity for pure Newtonian gravity (no softening).
/// v = sqrt(G·M / r)
pub fn orbital_velocity(central_mass: f32, orbit_radius: f32) -> f32 {
    (DEFAULT_GRAVITATIONAL_CONSTANT * central_mass / orbit_radius).sqrt()
}

/// Circular orbital velocity for Plummer-softened gravity.
/// Derived from F_centripetal = F_Plummer:
///   m·v²/r = G·M·m·r / (r²+ε²)^(3/2)
///   v = r · sqrt(G·M / (r²+ε²)^(3/2))
pub fn plummer_orbital_velocity(central_mass: f32, orbit_radius: f32, softening: f32) -> f32 {
    let norm_s = orbit_radius * orbit_radius + softening * softening;
    orbit_radius * (DEFAULT_GRAVITATIONAL_CONSTANT * central_mass / (norm_s * norm_s.sqrt())).sqrt()
}

/// Vis-viva velocity at a given distance for an elliptical orbit.
/// `is_periapsis = true` → point is closest approach; `false` → apoapsis.
pub fn elliptical_orbit_velocity(
    central_mass: f32,
    distance: f32,
    eccentricity: f32,
    is_periapsis: bool,
) -> f32 {
    let mu = DEFAULT_GRAVITATIONAL_CONSTANT * central_mass;
    let semimajor_axis = distance / (1.0 - eccentricity * if is_periapsis { 1.0 } else { -1.0 });
    (mu * (2.0 / distance - 1.0 / semimajor_axis)).sqrt()
}

/// Escape velocity: v = sqrt(2·G·M / r).
pub fn escape_velocity(central_mass: f32, distance: f32) -> f32 {
    (2.0 * DEFAULT_GRAVITATIONAL_CONSTANT * central_mass / distance).sqrt()
}

/// Barnes-Hut force on body `affected_idx` from the subtree rooted at `node`.
///
/// Bodies are keyed by index (not Entity) so this module is ECS-free.
/// The ECS layer builds the index↔Entity mapping before calling here.
pub fn barnes_hut_force(
    affected_idx: usize,
    affected_position: Vec3,
    affected_mass: f32,
    node: &QuadtreeNode,
    theta: f32,
    softening: f32,
    gravitational_constant: f32,
) -> Vec3 {
    let softening_squared = softening * softening;

    if node.is_far_enough(affected_position, theta) {
        let direction = node.mass_properties.center_of_mass - affected_position;
        let distance_squared = direction.length_squared();
        let norm_s = distance_squared + softening_squared;
        let force_scalar = gravitational_constant * affected_mass * node.mass_properties.total_mass
            / (norm_s * norm_s.sqrt());

        if !force_scalar.is_finite() {
            return Vec3::ZERO;
        }
        return direction * force_scalar;
    }

    if node.children.iter().all(|c| c.is_none()) {
        let mut total_force = Vec3::ZERO;
        for &(idx, position, mass) in &node.bodies {
            if idx == affected_idx {
                continue;
            }
            let direction = position - affected_position;
            let distance_squared = direction.length_squared();
            if distance_squared < 0.001 {
                continue;
            }
            let norm_s = distance_squared + softening_squared;
            let force_scalar =
                gravitational_constant * affected_mass * mass / (norm_s * norm_s.sqrt());
            if !force_scalar.is_finite() {
                continue;
            }
            total_force += direction * force_scalar;
        }
        return total_force;
    }

    let mut total_force = Vec3::ZERO;
    for child_node in node.children.iter().flatten() {
        total_force += barnes_hut_force(
            affected_idx,
            affected_position,
            affected_mass,
            child_node,
            theta,
            softening,
            gravitational_constant,
        );
    }
    total_force
}

/// Barnes-Hut quadtree — ECS-free (bodies keyed by usize index).
pub mod spatial {
    use bevy::math::{Vec2, Vec3};

    #[derive(Clone, Debug)]
    pub struct Aabb {
        pub center: Vec2,
        pub half_size: Vec2,
    }

    impl Aabb {
        pub fn new(center: Vec2, half_size: Vec2) -> Self {
            Self { center, half_size }
        }

        pub fn contains(&self, point: Vec2) -> bool {
            let min = self.center - self.half_size;
            let max = self.center + self.half_size;
            point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
        }

        /// Returns quadrant index (0-3) for a point.
        /// Bit 0: right=1 / left=0. Bit 1: bottom=1 / top=0.
        pub fn quadrant_of(&self, point: Vec2) -> usize {
            ((point.x >= self.center.x) as usize) | (((point.y < self.center.y) as usize) << 1)
        }

        pub fn quadrant_aabb(&self, quadrant: usize) -> Aabb {
            let quarter_size = self.half_size * 0.5;
            let x_sign = if (quadrant & 1) == 0 { -1.0 } else { 1.0 };
            let y_sign = if (quadrant & 2) == 0 { 1.0 } else { -1.0 };
            Aabb::new(
                self.center + Vec2::new(x_sign * quarter_size.x, y_sign * quarter_size.y),
                quarter_size,
            )
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct MassProperties {
        pub total_mass: f32,
        pub center_of_mass: Vec3,
    }

    impl MassProperties {
        pub fn add_body(&mut self, position: Vec3, mass: f32) {
            let new_total = self.total_mass + mass;
            if new_total > 0.0 {
                self.center_of_mass =
                    (self.center_of_mass * self.total_mass + position * mass) / new_total;
                self.total_mass = new_total;
            }
        }
    }

    #[derive(Debug)]
    pub struct QuadtreeNode {
        pub aabb: Aabb,
        pub depth: usize,
        pub mass_properties: MassProperties,
        /// Bodies stored as (body_index, position, mass).
        /// `body_index` is the caller's index, not an ECS Entity.
        pub bodies: Vec<(usize, Vec3, f32)>,
        pub children: [Option<Box<QuadtreeNode>>; 4],
        pub max_depth: usize,
        pub max_bodies_per_node: usize,
    }

    impl QuadtreeNode {
        pub fn new(aabb: Aabb, depth: usize, max_depth: usize, max_bodies_per_node: usize) -> Self {
            Self {
                aabb,
                depth,
                mass_properties: MassProperties::default(),
                bodies: Vec::new(),
                children: [None, None, None, None],
                max_depth,
                max_bodies_per_node,
            }
        }

        pub fn is_far_enough(&self, position: Vec3, theta: f32) -> bool {
            let pos_2d = Vec2::new(position.x, position.y);
            let distance = (pos_2d - self.aabb.center).length();
            if distance < 0.001 || self.mass_properties.total_mass <= 0.0 {
                return false;
            }
            let width = self.aabb.half_size.x * 2.0;
            width / distance < theta
        }

        pub fn insert(&mut self, idx: usize, position: Vec3, mass: f32) {
            self.mass_properties.add_body(position, mass);

            if self.depth >= self.max_depth
                || (self.bodies.len() < self.max_bodies_per_node && self.children[0].is_none())
            {
                self.bodies.push((idx, position, mass));
                return;
            }

            if self.children[0].is_none() {
                for i in 0..4 {
                    self.children[i] = Some(Box::new(QuadtreeNode::new(
                        self.aabb.quadrant_aabb(i),
                        self.depth + 1,
                        self.max_depth,
                        self.max_bodies_per_node,
                    )));
                }
                let existing = std::mem::take(&mut self.bodies);
                for (e, p, m) in existing {
                    let q = self.aabb.quadrant_of(p.truncate());
                    if let Some(child) = &mut self.children[q] {
                        child.insert(e, p, m);
                    }
                }
            }

            let quadrant = self.aabb.quadrant_of(position.truncate());
            if let Some(child) = &mut self.children[quadrant] {
                child.insert(idx, position, mass);
            }
        }
    }

    #[derive(Debug)]
    pub struct Quadtree {
        pub root: QuadtreeNode,
    }

    impl Quadtree {
        pub fn new(bounds: Aabb, max_depth: usize, max_bodies_per_node: usize) -> Self {
            Self {
                root: QuadtreeNode::new(bounds, 0, max_depth, max_bodies_per_node),
            }
        }

        /// Build from `(body_index, position, mass)` tuples.
        pub fn from_indexed_bodies(
            bodies: &[(usize, Vec3, f32)],
            max_depth: usize,
            max_bodies_per_node: usize,
        ) -> Self {
            if bodies.is_empty() {
                return Self::new(
                    Aabb::new(Vec2::ZERO, Vec2::new(1000.0, 1000.0)),
                    max_depth,
                    max_bodies_per_node,
                );
            }

            let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
            let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
            for (_, pos, _) in bodies {
                min_x = min_x.min(pos.x);
                min_y = min_y.min(pos.y);
                max_x = max_x.max(pos.x);
                max_y = max_y.max(pos.y);
            }

            let padding = ((max_x - min_x) + (max_y - min_y)) * 0.1;
            min_x -= padding;
            min_y -= padding;
            max_x += padding;
            max_y += padding;

            let center = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
            let half_size = Vec2::new((max_x - min_x) * 0.5, (max_y - min_y) * 0.5);
            let max_half = half_size.x.max(half_size.y);

            let mut tree = Self::new(
                Aabb::new(center, Vec2::splat(max_half)),
                max_depth,
                max_bodies_per_node,
            );
            for &(idx, pos, mass) in bodies {
                tree.root.insert(idx, pos, mass);
            }
            tree
        }
    }
}

//! Typed handles for materials and particle groups.
//!
//! Replaces raw `u32` material IDs and `Range<usize>` particle ranges with
//! named types — prevents mixing up IDs, enables grouped operations on particle sets.

use std::ops::Range;

use glam::Vec2;

use crate::particle::{Particle, Particles};
use crate::solver::query::MaterialState;

/// Typed handle for a registered material.
///
/// Wraps a `u32` material ID. Use instead of raw integers to prevent
/// accidentally mixing material IDs with other u32 values.
///
/// # Example
/// ```rust,no_run
/// # use emerge::solver::MpmSolver;
/// # use emerge::{SolverConfig, SpawnConfig, NewtonianFluidMaterial};
/// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
/// # let spawn = SpawnConfig { ..Default::default() };
/// # let mut solver = MpmSolver::new(config, spawn);
/// let water = solver.register_material(Box::new(NewtonianFluidMaterial::water(1000.0, 1e4)));
/// // spawn.material_id = water.id();
/// // solver.phase_transition(|p| p.temperature > 373.0, steam.id());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);

impl MaterialHandle {
    /// Raw material ID — for use in `SpawnConfig.material_id` and particle comparisons.
    #[inline]
    pub fn id(self) -> u32 {
        self.0
    }
}

impl From<u32> for MaterialHandle {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<MaterialHandle> for u32 {
    fn from(h: MaterialHandle) -> u32 {
        h.0
    }
}

impl std::fmt::Display for MaterialHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mat#{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// A named, typed handle to a contiguous range of particles in the solver.
///
/// Returned by `solver.spawn_group()`. LP uses this to track ownership
/// of creature bodies, terrain regions, fluid blobs, etc. without managing
/// raw indices.
///
/// # Example — LP creature management
/// ```rust,no_run
/// # use emerge::solver::{MpmSolver, ParticleGroup};
/// # use emerge::{SolverConfig, SpawnConfig};
/// # use glam::Vec2;
/// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
/// # let spawn = SpawnConfig { ..Default::default() };
/// # let mut solver = MpmSolver::new(config, spawn);
/// let creature: ParticleGroup = solver.spawn_group(SpawnConfig { ..Default::default() });
/// creature.set_activation(&mut solver.particles_mut(), 1.0);
/// let state = creature.state(solver.particles());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParticleGroup {
    pub range: Range<usize>,
    /// Optional debug label — shown in diagnostics.
    pub label: Option<String>,
}

impl ParticleGroup {
    pub fn new(range: Range<usize>) -> Self {
        Self { range, label: None }
    }

    pub fn named(range: Range<usize>, label: impl Into<String>) -> Self {
        Self { range, label: Some(label.into()) }
    }

    /// Number of particles in this group.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Collect all particles in this group into a `Vec<Particle>`.
    pub fn get_all(&self, all: &Particles) -> Vec<Particle> {
        self.range.clone().map(|i| all.get(i)).collect()
    }

    /// Aggregate physics state (centroid, speed, J, etc.) for this group.
    pub fn state(&self, all: &Particles) -> MaterialState {
        material_state_of_range(all, self.range.clone())
    }

    /// Set `activation` on every particle in the group.
    /// Use for muscle contraction: 0.0 = relaxed, 1.0 = fully contracted.
    pub fn set_activation(&self, all: &mut Particles, value: f32) {
        for i in self.range.clone() {
            all.activation[i] = value;
        }
    }

    /// Apply a spatially-varying activation function per particle.
    /// `f(position) -> activation` — e.g. traveling wave for peristaltic locomotion.
    pub fn set_activation_fn(&self, all: &mut Particles, f: impl Fn(Vec2) -> f32) {
        for i in self.range.clone() {
            all.activation[i] = f(all.x[i]).clamp(0.0, 1.0);
        }
    }

    /// Set temperature uniformly across the group.
    pub fn set_temperature(&self, all: &mut Particles, temp: f32) {
        for i in self.range.clone() {
            all.temperature[i] = temp;
        }
    }

    /// Set user_tag on every particle — LP uses this for creature ownership.
    pub fn set_tag(&self, all: &mut Particles, tag: u32) {
        for i in self.range.clone() {
            all.user_tag[i] = tag;
        }
    }

    /// Apply a velocity impulse to every particle in the group (with optional falloff).
    /// `falloff = None` → uniform impulse. `falloff = Some(center)` → linear distance falloff.
    pub fn apply_impulse(&self, all: &mut Particles, impulse: Vec2, falloff: Option<Vec2>) {
        for i in self.range.clone() {
            let scale = match falloff {
                None => 1.0,
                Some(center) => {
                    let d = (all.x[i] - center).length();
                    (1.0 - d * 0.1).max(0.0)
                }
            };
            all.v[i] += impulse * scale;
        }
    }

    /// Center of mass of this group.
    pub fn centroid(&self, all: &Particles) -> Vec2 {
        let len = self.range.len();
        if len == 0 {
            return Vec2::ZERO;
        }
        let sum: Vec2 = self.range.clone().map(|i| all.x[i]).sum();
        sum / len as f32
    }

    /// Count particles matching a predicate (e.g. those that transitioned material).
    pub fn count_where(&self, all: &Particles, pred: impl Fn(&Particle) -> bool) -> usize {
        self.range.clone().filter(|&i| pred(&all.get(i))).count()
    }
}

impl From<Range<usize>> for ParticleGroup {
    fn from(range: Range<usize>) -> Self {
        Self::new(range)
    }
}

impl ParticleGroup {
    /// Rebuild a group from a `user_tag` scan after particle removal shifts indices.
    ///
    /// Returns `None` if no particles with that tag exist.
    /// The resulting range covers all *contiguous* particles with the tag — if removal
    /// scattered matching particles across the array, use `solver.particles_with_tag(tag)`
    /// to collect indices individually instead.
    pub fn from_tag(particles: &Particles, tag: u32) -> Option<Self> {
        let first = particles.indices().find(|&i| particles.user_tag[i] == tag)?;
        let last = particles.indices().rev().find(|&i| particles.user_tag[i] == tag)?;
        Some(Self::new(first..last + 1))
    }
}

/// Aggregate MaterialState for a range of particles in a SoA store.
fn material_state_of_range(all: &Particles, range: std::ops::Range<usize>) -> MaterialState {
    let mut s = MaterialState::default();
    for i in range {
        let p = all.get(i);
        s.accumulate(
            p.x,
            p.v.length(),
            p.plastic_volume_ratio,
            p.deformation_gradient.determinant(),
            p.density,
        );
    }
    s.finalize();
    s
}

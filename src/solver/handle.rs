//! Typed handles for materials and particle groups.

use crate::solver::query::BodyState;

/// Typed handle for a registered material.
///
/// Wraps a `u32` material ID. Use instead of raw integers to prevent
/// accidentally mixing material IDs with other u32 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);

impl MaterialHandle {
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

/// A stable handle to a group of particles, identified by `user_tag`.
///
/// Physical indices change whenever particles sleep or wake — the tag is the
/// only stable identity. All operations delegate to `Simulation`'s tag-based API,
/// which uses `tag_index` for O(group_size) access.
///
/// # Example — LP creature management
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::solver::Simulation;
/// # use emerge::{SimConfig, SpawnRegion};
/// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
/// # let mut solver = Simulation::empty(config);
/// let creature = solver.add_body(SpawnRegion::for_sim(&config));
/// solver.set_group_activation(creature, 1.0);
/// let centroid = solver.group_centroid(creature);
/// let state = solver.group_state(creature);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticleGroup {
    pub tag: u32,
    /// Optional debug label — shown in diagnostics.
    pub label: Option<&'static str>,
}

impl ParticleGroup {
    pub fn new(tag: u32) -> Self {
        Self { tag, label: None }
    }

    pub fn named(tag: u32, label: &'static str) -> Self {
        Self {
            tag,
            label: Some(label),
        }
    }

    pub fn tag(self) -> u32 {
        self.tag
    }
}

impl From<u32> for ParticleGroup {
    fn from(tag: u32) -> Self {
        Self::new(tag)
    }
}

impl std::fmt::Display for ParticleGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.label {
            Some(l) => write!(f, "group({l}, tag={})", self.tag),
            None => write!(f, "group(tag={})", self.tag),
        }
    }
}

/// Aggregate BodyState for a tag — delegates to Simulation::group_state.
/// Kept here for callers that hold a ParticleGroup and want a one-liner.
pub fn group_state_of(_group: ParticleGroup, state: BodyState) -> BodyState {
    state
}

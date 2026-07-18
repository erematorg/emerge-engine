pub mod config;
pub mod cutoff;
pub mod density;
pub mod handle;
mod lifecycle;
mod particles;
mod queries;
pub mod query;
pub mod spatial_hash;
mod step;

pub use config::{SimConfig, SpawnRegion};
pub use cutoff::smooth_cutoff;
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use query::{BodyState, body_state_of, region_body_state_of};
// Only consumed by systems::gpu's own CFL scan -- unused (and correctly
// warned about) in a build without that feature.
#[cfg(feature = "gpu")]
pub(crate) use step::{affine_cfl_speed_contribution, cfl_bound};

use std::collections::{HashMap, HashSet};

use spatial_hash::SpatialHash;

use glam::{Mat2, Vec2};

use crate::thermodynamics::{ScalarDiffusionField, ThermalDiffusion};
use crate::{boundary::BoundaryCondition, fields::Field, materials::registry::MaterialRegistry};
use crate::{
    grid::Grid,
    particle::{Particle, Particles},
};

type PhaseRule = Box<dyn Fn(&Particle) -> Option<u32> + Send + Sync>;

pub struct Simulation {
    config: SimConfig,
    particles: Particles,
    /// Partition boundary: particles[0..active_count] are active, [active_count..N] sleeping.
    /// P2G / G2P only visit [0..active_count]. Maintained by sleep_particle / wake_particle.
    active_count: usize,
    /// Maps user_tag → physical indices of all particles with that tag.
    /// HashSet gives O(1) insert/remove on every sleep/wake swap.
    tag_index: HashMap<u32, HashSet<usize>>,
    /// Monotonically increasing counter — next tag issued by add_body.
    next_tag: u32,
    grid: Grid,
    materials: MaterialRegistry,
    boundaries: Vec<Box<dyn BoundaryCondition>>,
    /// Optional directional (setae-style) friction for the multi-field contact
    /// "grip" field — see `DirectionalContactGrip`'s doc. `None` (default) keeps
    /// the existing plain symmetric `contact_friction` behavior; every scene that
    /// never opts in is completely unaffected.
    contact_grip: Option<std::sync::Arc<crate::grid::DirectionalContactGrip>>,
    force_fields: Vec<(String, Box<dyn Field>)>,
    thermal: Option<ThermalDiffusion>,
    /// Scalar diffusion fields (pheromone, nutrients, morphogen) — run automatically each substep.
    scalar_fields: Vec<ScalarDiffusionField>,
    frame_index: u64,
    last_step_dt: f32,
    last_substeps: usize,
    last_vel_clamp_count: usize,
    last_j_projection_count: usize,
    last_sim_time_dropped: f32,
    last_timing: crate::diagnostics::StepTiming,
    /// Automatic phase transition rules, evaluated every substep.
    phase_rules: Vec<PhaseRule>,
    /// Spatial hash over active particles — rebuilt each substep after G2P.
    /// Turns O(N) radius queries into O(candidates_in_neighborhood).
    spatial_hash: SpatialHash,
    /// Scratch buffer for wake/sleep candidates — pre-allocated once, cleared per substep.
    /// Pattern from ziran2020 MpmSimulationBase: scratch_xp/scratch_vp member fields.
    scratch_indices: Vec<usize>,
}

impl std::fmt::Debug for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Simulation")
            .field("grid_res", &self.config.grid_res)
            .field("particles_total", &self.particles.len())
            .field("active", &self.active_count)
            .field("frame", &self.frame_index)
            .finish_non_exhaustive()
    }
}

pub(crate) fn initialize_particles(
    config: &SimConfig,
    spawn: SpawnRegion,
    rng: &mut LcgRng,
) -> Vec<Particle> {
    use crate::solver::config::SpawnShape;
    let mass = spawn.mass_override.unwrap_or(config.particle_mass);
    let mut particles = Vec::new();
    let half = spawn.box_size.as_vec2() * 0.5;
    let min = spawn.box_center - half;
    let max = spawn.box_center + half;

    let mut i = min.x;
    while i < max.x {
        let mut j = min.y;
        while j < max.y {
            let pos = Vec2::new(i, j);

            // Apply shape mask — skip particles outside the disk if disk shape is active.
            let inside = match spawn.shape {
                SpawnShape::Box => true,
                SpawnShape::Disk { radius } => (pos - spawn.box_center).length() <= radius,
            };

            if inside {
                let jitter_mag = spawn.position_jitter * spawn.spacing;
                let jx = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
                let jy = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
                let jittered_pos = pos + Vec2::new(jx, jy);
                let random = Vec2::new(rng.next_f32(), rng.next_f32());
                let velocity = (random - Vec2::splat(0.5)) * spawn.initial_velocity_scale;
                particles.push(Particle {
                    x: jittered_pos,
                    v: velocity,
                    velocity_gradient: Mat2::ZERO,
                    deformation_gradient: spawn.initial_deformation_gradient,
                    mass,
                    initial_volume: config.default_initial_volume,
                    volume: config.default_initial_volume,
                    density: mass / config.default_initial_volume,
                    material_id: spawn.material_id,
                    plastic_volume_ratio: 1.0,
                    hardening_scale: 1.0,
                    friction_hardening: 0.0,
                    log_volume_strain: 0.0,
                    temperature: 0.0,
                    user_tag: 0,
                    activation: 0.0,
                    activation_dir: Vec2::ZERO,
                    muscle_group_id: 0,
                    contact_group: 0,
                    sleeping: 0,
                    pinned: 0,
                    scalar_field: 0.0,
                    _pad: 0,
                });
            }

            j += spawn.spacing;
        }
        i += spawn.spacing;
    }

    particles
}

#[derive(Debug)]
pub(crate) struct LcgRng {
    state: u32,
}

impl LcgRng {
    pub(crate) fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
}

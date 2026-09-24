pub mod body_state;
mod cfl;
pub mod config;
pub mod density;
pub mod handle;
mod lifecycle;
pub mod operator;
mod particles;
mod projection;
mod queries;
pub mod spatial_hash;
mod step;

pub use body_state::{BodyState, body_state_of, region_body_state_of};
pub use config::{SimConfig, SpawnRegion};
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use operator::{CoupledBody, IntoSimEntry, OperatorCtx, Plugin, Stage, StageOp};
// Only consumed by systems::gpu's own CFL scan -- unused (and correctly
// warned about) in a build without that feature.
#[cfg(feature = "gpu")]
pub(crate) use cfl::{affine_cfl_speed_contribution, cfl_bound};

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use spatial_hash::SpatialHash;

use glam::{Mat2, Vec2};

use crate::rod::Rod;
use crate::thermodynamics::{
    CosseratField, GranularFluidityField, ScalarDiffusionField, ThermalDiffusion,
};
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
    /// Simulation time accumulated across this `step()`'s substeps, waiting to
    /// be handed to the diffusion operators in ONE application -- see
    /// `do_substep`'s own note for the stability derivation that makes this
    /// correct (and why per-substep sub-cycling was ~31,000x redundant).
    pending_diffusion_dt: f32,
    /// Index of the substep currently running within this `step()`. Lets
    /// per-FRAME work (validation scans, phase rules) run once instead of
    /// once per substep -- see `do_substep`'s own notes.
    substep_index_in_frame: u32,
    /// Nonlocal Granular Fluidity field (see `energy::thermodynamics::
    /// granular_fluidity` module doc) -- `None` (default) for every scene
    /// that doesn't opt in, same zero-cost-when-unused property `thermal`
    /// already has.
    granular_fluidity: Option<GranularFluidityField>,
    /// Persistent per-particle gathered `g`, indexed by particle -- read by
    /// G2P (`G2PParams::nonlocal_fluidity`), written by the granular-
    /// fluidity pass at the end of the SAME substep (one-substep lag, same
    /// convention thermal/scalar diffusion already use). Empty when
    /// `granular_fluidity` is `None`.
    granular_fluidity_g: Vec<f32>,
    /// Cosserat micro-rotation field (see `energy::thermodynamics::
    /// cosserat_field` module doc) -- `None` (default) for every scene that
    /// doesn't opt in, same zero-cost-when-unused property `granular_fluidity`
    /// already has. Real grid-level angular-momentum channel, added to close
    /// the loop `matter::materials::solid::granular::cosserat`'s kinematics module leaves open.
    cosserat: Option<CosseratField>,
    /// Persistent per-particle gathered micro-rotation, one-substep-lag
    /// convention matching `granular_fluidity_g` exactly.
    cosserat_omega: Vec<f32>,
    /// Persistent per-particle gathered micro-curvature (kappa = grad(omega_c)),
    /// same one-substep-lag convention, read by G2P (`G2PParams::
    /// cosserat_curvature`), written by the Cosserat pass at the end of the
    /// SAME substep.
    cosserat_curvature: Vec<glam::Vec2>,
    frame_index: u64,
    /// "Sticky" fine-substep hold for `SimConfig::fluid_step_retry_enabled`
    /// (2026-08-08): `(held_dt, substeps_remaining)`. A one-off retry alone
    /// (see `do_substep_with_retry`) doesn't work -- measured directly: the
    /// very next substep re-derives its size from the ordinary CFL scan,
    /// forgetting the rejection immediately, so a sustained near-wall
    /// compression event just repeats the same reject-shrink-forget cycle
    /// substep after substep instead of ever holding fine resolution long
    /// enough to actually resolve the event (confirmed: a plain per-substep
    /// retry, swept across several thresholds, never reproduced the clean
    /// convergence a globally-tightened CFL coefficient did). This field is
    /// the fix: once a retry fires, `choose_substep_dt`'s result is capped
    /// to `held_dt` for `substeps_remaining` further substeps (decremented
    /// each one), holding the fine resolution through the actual event
    /// instead of relaxing on the very next substep. `None` = no hold
    /// active (every scene that never enables the feature stays here
    /// permanently, zero cost).
    fluid_sticky_fine_dt: Option<(f32, u32)>,
    /// Real max particle speed measured by the PREVIOUS `choose_substep_dt`
    /// call -- one-substep-lagged, since a substep's own max speed isn't
    /// known until its CFL fold completes. Feeds the near-wall gate's
    /// Mach-relative compression threshold
    /// (`SimConfig::fluid_near_wall_compression_mach_margin`); `0.0` at
    /// construction, which makes the very first substep's near-wall gate
    /// maximally sensitive (threshold collapses to 0.0, matching the OLD
    /// always-reactive behavior for exactly one substep) -- a safe,
    /// conservative cold-start default, not a special case: it errs toward
    /// too-cautious for one substep rather than too-permissive, and
    /// self-corrects the moment the first real fold measures an actual
    /// speed.
    last_max_particle_speed: f32,
    last_step_dt: f32,
    last_substeps: usize,
    last_vel_clamp_count: usize,
    last_j_projection_count: usize,
    last_sim_time_dropped: f32,
    last_timing: crate::diagnostics::StepTiming,
    /// `SimConfig::spatial_sort_enabled` cache: computed ONCE per outer
    /// `step()` call (not per substep -- real, measured: recomputing this
    /// O(N log N) sort every substep cost MORE than the P2G cache-locality
    /// win it was meant to provide, see `spatial_sort_order`'s own doc)
    /// and reused across every substep within that call. Particle positions
    /// shift only slightly substep-to-substep, so a step-stale order still
    /// captures most of the real locality benefit. Empty when the feature
    /// is off -- zero allocation cost in the default case.
    cached_spatial_sort_order: Vec<usize>,
    /// Automatic phase transition rules, evaluated every substep.
    phase_rules: Vec<PhaseRule>,
    /// Spatial hash over active particles. Turns O(N) radius queries into
    /// O(candidates_in_neighborhood) for `particles_near`/`count_near`/
    /// `particles_knn`/`region_state`.
    ///
    /// Lazily rebuilt: `step()` only marks it dirty (`spatial_hash_dirty`),
    /// it does NOT rebuild eagerly every frame — real, measured cost found
    /// 2026-08-03 (see `perf_opportunities_survey` memory): rebuilding
    /// unconditionally every step cost 16.4% of a step's total time even in
    /// scenes that never call any of the four query methods above. Mirrors
    /// the identical fix already shipped on the GPU path (`GpuSimulation`'s
    /// own lazy spatial-hash rebuild, 2026-07-12, 25% win at 100k particles)
    /// — this ports the same real technique to CPU. `RefCell` because the
    /// four query methods take `&self` (a real, established public API
    /// contract LP depends on) but need to trigger a rebuild internally —
    /// the classic "conceptually read-only, lazily-computed cache" case
    /// interior mutability exists for. Structural mutations that change
    /// particle count/positions outside `step()` (`add_body`, `remove_where`,
    /// `split_particles`, construction) still rebuild EAGERLY right after
    /// mutating and clear the dirty flag, so a query issued between two
    /// `step()` calls (LP's actual usage pattern) always sees fresh data —
    /// only the once-per-frame "rebuild whether or not anyone will query it"
    /// cost is what became lazy.
    spatial_hash: RefCell<SpatialHash>,
    /// See `spatial_hash`'s own doc. `true` right after `step()` runs a
    /// substep loop (positions moved, hash is stale); cleared by
    /// `ensure_spatial_hash_fresh` the first time any query method is
    /// actually called. Never true after an eager rebuild (spawn/remove/
    /// split), since those clear it immediately after rebuilding.
    spatial_hash_dirty: Cell<bool>,
    /// Discrete elastic rods (Cosserat-rod family, `spacetime::rod`) sharing
    /// this simulation's own MPM grid — see `step.rs`'s `do_substep` for the
    /// real scatter/gather insertion points. Empty for every scene that
    /// never calls `add_rod`/`with_rod` (zero-cost: 0-iteration loops).
    rods: Vec<Rod>,
    /// Discrete-element grain populations (`spacetime::grains`) sharing this
    /// simulation's own MPM grid — real, cited elastic-plastic rolling
    /// resistance (Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011),
    /// see `grains::coupling` for the real scatter/gather insertion points
    /// (mirroring `rods` above exactly). Empty for every scene that never
    /// calls `add_grain_population`/`with_grain_population` (zero-cost:
    /// 0-iteration loops). Not yet gated by any automatic oracle deciding
    /// where grains are needed — that's a real, separate, not-yet-built
    /// piece (see `project_dem_rolling_resistance_scoped` memory); today a
    /// caller decides explicitly, same as `add_rod`.
    grain_populations: Vec<crate::grains::population::GrainPopulation>,
    /// Branching rod/root topologies (`spacetime::rod::network`) sharing this
    /// simulation's own MPM grid — real scatter/gather/internal-force
    /// `CoupledBody` impl, mirroring `rods` above exactly (generalized from
    /// linear i-1/i+1 adjacency to explicit edge/bending topology). Empty
    /// for every scene that never calls `add_rod_network`/`with_rod_network`
    /// (zero-cost: 0-iteration loops).
    rod_networks: Vec<crate::rod::RodNetwork>,
    /// Scratch buffer for wake/sleep candidates — pre-allocated once, cleared per substep.
    /// Pattern from ziran2020 MpmSimulationBase: scratch_xp/scratch_vp member fields.
    scratch_indices: Vec<usize>,
    /// Genuinely new physics with no existing mechanism to fit -- see
    /// `operator` module doc. Empty for every scene today (no implementor
    /// exists yet); dispatched at each real stage in `do_substep` so this
    /// is a live, zero-cost-when-empty extension point, not just a type.
    stage_ops: Vec<Box<dyn operator::StageOp>>,
    /// Coupled sub-solvers registered through the generic `add`/`with`
    /// path (see `operator` module doc). `rods`/`grain_populations` above
    /// stay their own concrete `Vec`s for now (real duplication between
    /// them is the actual migration target, not yet done) -- this is where
    /// a future `CoupledBody` implementor lands once it exists.
    coupled_bodies: Vec<Box<dyn operator::CoupledBody>>,
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
                    internal_pressure: 0.0,
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
    pub(crate) const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    const fn next_u32(&mut self) -> u32 {
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

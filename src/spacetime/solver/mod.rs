pub mod body_state;
#[cfg(any(test, feature = "research-diagnostics"))]
mod boundary_diagnostics;
mod cfl;
pub mod config;
pub mod density;
pub mod handle;
mod implicit_corotated;
mod lifecycle;
mod particles;
mod projection;
mod queries;
pub mod spatial_hash;
mod step;

pub use body_state::{BodyState, body_state_of, region_body_state_of};
#[cfg(any(test, feature = "research-diagnostics"))]
pub use boundary_diagnostics::{
    AcceptedBoundaryImpulseLedger, BoundaryImpulseExperiment, BoundaryImpulseReport,
    BoundaryNodeImpulseLedger,
};
pub use config::{SimConfig, SpawnRegion};
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use particles::{HydrostaticState, hydrostatic_state};
// Only consumed by systems::gpu's own CFL scan -- unused (and correctly
// warned about) in a build without that feature.
#[cfg(feature = "gpu")]
pub(crate) use cfl::{
    affine_cfl_speed_contribution, cfl_bound, deformation_gradient_ode_dt_bound, is_near_wall,
    shock_viscosity_dt_bound, single_particle_instability_dt_bound,
};

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

/// The CPU MLS-MPM solver -- the crate's central type. Owns every particle,
/// the shared background grid, the material registry, boundary conditions,
/// force fields, and optional rod/grain/thermal subsystems, and advances
/// all of them together one frame at a time.
///
/// Construct with [`Simulation::new`] (a [`SimConfig`] plus an initial
/// [`SpawnRegion`] of particles) or [`Simulation::empty`] (no initial
/// particles, add bodies afterward via [`Simulation::add_body`]). Advance
/// simulated time with [`Simulation::step`], which always advances exactly
/// `config.dt` regardless of how many adaptive CFL substeps that requires
/// internally. Read state back via [`Simulation::particles`]/
/// `particles_mut`, [`Simulation::material_state`]/`region_state`, or
/// [`Simulation::particles_near`].
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::{SimConfig, Simulation, SpawnRegion};
///
/// let config = SimConfig::standard(64, 0.05, glam::Vec2::new(0.0, -0.05));
/// let mut sim = Simulation::new(config, SpawnRegion::for_sim(&config));
/// sim.step();
/// ```
pub struct Simulation {
    config: SimConfig,
    particles: Particles,
    /// Partition boundary: particles[0..active_count] are active, [active_count..N] sleeping.
    /// P2G / G2P only visit [0..active_count]. Maintained by sleep_particle / wake_particle.
    active_count: usize,
    /// Maps user_tag → physical indices of all particles with that tag.
    /// HashSet gives O(1) insert/remove on every sleep/wake swap.
    tag_index: HashMap<u32, HashSet<usize>>,
    /// Monotonically increasing counter -- next tag issued by add_body.
    next_tag: u32,
    grid: Grid,
    materials: MaterialRegistry,
    /// Per-particle banked frictional heating, kelvin -- see
    /// `spacetime::transfer::friction_heat` for why a rise this small has to
    /// be accumulated before it can be added to an `f32` temperature.
    /// Empty and untouched in every scene where nothing rubs.
    friction_heat_debt: Vec<f32>,
    boundaries: Vec<Box<dyn BoundaryCondition>>,
    /// True while `boundaries` still holds only the default `SlipBoundary`
    /// inserted at construction (see `empty`/`new`). The first
    /// `add_boundary_condition` call clears it instead of stacking on top:
    /// the default's zero-friction no-penetration correction runs first every
    /// substep and zeroes the into-wall velocity before a user's
    /// `FrictionBoundary` sees it, which silently cancels that friction (a
    /// grain coasted at constant velocity for 500,000+ steps on a
    /// `FrictionBoundary(2, 0.7)` floor).
    boundaries_are_default: bool,
    /// Optional directional (setae-style) friction for the multi-field contact
    /// "grip" field -- see `DirectionalContactGrip`'s doc. `None` (default) keeps
    /// the existing plain symmetric `contact_friction` behavior; every scene that
    /// never opts in is completely unaffected.
    contact_grip: Option<std::sync::Arc<crate::grid::DirectionalContactGrip>>,
    force_fields: Vec<(String, Box<dyn Field>)>,
    thermal: Option<ThermalDiffusion>,
    /// Scalar diffusion fields (pheromone, nutrients, morphogen) -- run automatically each substep.
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
    /// the loop `matter::materials::granular::cosserat`'s kinematics module leaves open.
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
    /// Fine-substep hold for `SimConfig::fluid_step_retry_enabled`:
    /// `(held_dt, substeps_remaining)`. A one-off retry (see
    /// `do_substep_with_retry`) is not enough: the next substep re-derives its
    /// size from the CFL scan and forgets the rejection, so a sustained
    /// near-wall compression repeats a reject-shrink-forget cycle every
    /// substep (a per-substep retry swept across several thresholds never
    /// matched a globally tightened CFL coefficient). Once a retry fires,
    /// `choose_substep_dt` is capped to `held_dt` for `substeps_remaining`
    /// further substeps. `None` = no hold active (always, in scenes that
    /// never enable the feature).
    fluid_sticky_fine_dt: Option<(f32, u32)>,
    /// Research diagnostic, `EMERGE_TRACK_BOUNDARY_BIAS` (see
    /// `transfer::diagnose_particle_divergence_decomposition`). Set by
    /// `do_substep` right after P2G with the `dt` that scatter used, filled
    /// with the final `tr(C)` at the end of the same call, then printed by
    /// `do_substep_with_retry` after its retry loop exits, so it reflects the
    /// accepted attempt. `None` unless that switch is set in a
    /// `research-diagnostics` build (see `research_switch`). Fields:
    /// `(tracked_index, trace_translation, trace_affine, trace_stress,
    /// trace_final, dt_used)`.
    pending_divergence_diagnostic: Option<(usize, f32, f32, f32, f32, f32)>,
    /// Structural wall-bounce impulse ledger (`boundary_diagnostics`), set by
    /// `enable_boundary_impulse_diagnostic`; absent from a default build.
    #[cfg(any(test, feature = "research-diagnostics"))]
    boundary_impulse_diagnostic: Option<boundary_diagnostics::BoundaryImpulseDiagnostic>,
    /// Max particle speed measured by the previous `choose_substep_dt` call
    /// (lagged one substep: a substep's max speed is known only after its CFL
    /// fold). Feeds the near-wall gate's Mach-relative compression threshold
    /// (`SimConfig::fluid_near_wall_compression_mach_margin`). `0.0` at
    /// construction, so the first substep's gate is maximally sensitive
    /// (threshold 0.0): cautious for one substep, corrected as soon as the
    /// first fold measures a speed.
    last_max_particle_speed: f32,
    last_step_dt: f32,
    last_substeps: usize,
    last_j_projection_count: usize,
    last_sim_time_dropped: f32,
    last_timing: crate::diagnostics::StepTiming,
    /// `SimConfig::spatial_sort_enabled` cache: computed ONCE per outer
    /// `step()` call (not per substep -- measured: recomputing this
    /// O(N log N) sort every substep cost MORE than the P2G cache-locality
    /// win it was meant to provide, see `spatial_sort_order`'s doc)
    /// and reused across every substep within that call. Particle positions
    /// shift only slightly substep-to-substep, so a step-stale order still
    /// captures most of the locality benefit. Empty when the feature
    /// is off -- zero allocation cost in the default case.
    cached_spatial_sort_order: Vec<usize>,
    /// Automatic phase transition rules, evaluated every substep.
    phase_rules: Vec<PhaseRule>,
    /// Spatial hash over active particles. Turns O(N) radius queries into
    /// O(candidates_in_neighborhood) for `particles_near`/`count_near`/
    /// `particles_knn`/`region_state`.
    ///
    /// Lazily rebuilt: `step()` only marks it dirty (`spatial_hash_dirty`).
    /// Rebuilding every step measured at 16.4% of a step's time even in
    /// scenes that never query it; `GpuSimulation` rebuilds lazily the same
    /// way. `RefCell` because the four queries take `&self` (public API LP
    /// uses) but may trigger a rebuild: a read-only, lazily computed cache.
    /// Mutations outside `step()` that change particle count or positions
    /// (`add_body`, `remove_where`, `split_particles`, construction) rebuild
    /// eagerly and clear the flag, so a query between two `step()` calls
    /// always sees fresh data.
    spatial_hash: RefCell<SpatialHash>,
    /// See `spatial_hash`'s doc. `true` right after `step()` runs a
    /// substep loop (positions moved, hash is stale); cleared by
    /// `ensure_spatial_hash_fresh` the first time any query method is
    /// actually called. Never true after an eager rebuild (spawn/remove/
    /// split), since those clear it immediately after rebuilding.
    spatial_hash_dirty: Cell<bool>,
    /// Discrete elastic rods (Cosserat-rod family, `spacetime::rod`) sharing
    /// this simulation's own MPM grid -- see `step.rs`'s `do_substep` for the
    /// real scatter/gather insertion points. Empty for every scene that
    /// never calls `add_rod`/`with_rod` (zero-cost: 0-iteration loops).
    rods: Vec<Rod>,
    /// Discrete-element grain populations (`spacetime::grains`) sharing this
    /// simulation's own MPM grid -- cited elastic-plastic rolling
    /// resistance (Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011),
    /// see `grains::coupling` for the scatter/gather insertion points
    /// (mirroring `rods` above). Empty for every scene that never
    /// calls `add_grain_population`/`with_grain_population` (zero-cost:
    /// 0-iteration loops). Nothing adds grains automatically: a caller
    /// decides where, as with `add_rod`, optionally guided by the
    /// `grains::oracle` packing-fraction signal and
    /// `enrich_region_into_grain`.
    grain_populations: Vec<crate::grains::population::GrainPopulation>,
    /// Scratch buffer for wake/sleep candidates -- pre-allocated once, cleared per substep.
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
    // Mass per particle from mass per CELL AREA: a lattice at `spacing` cells
    // packs `1/spacing^2` particles into each cell, so the grid sees
    // `grid_density` regardless of how finely the region is discretized.
    // Without the `spacing^2`, refining a scene silently multiplied its grid
    // density -- and therefore gravity relative to stiffness -- by
    // `1/spacing^2`. See `SimConfig::grid_density`.
    let mass = spawn
        .mass_override
        .unwrap_or(config.grid_density * spawn.spacing * spawn.spacing);
    let mut particles = Vec::new();
    let half = spawn.box_size.as_vec2() * 0.5;
    let min = spawn.box_center - half;
    let max = spawn.box_center + half;

    let mut i = min.x;
    while i < max.x {
        let mut j = min.y;
        while j < max.y {
            let pos = Vec2::new(i, j);

            // Apply shape mask -- skip particles outside the disk if disk shape is active.
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

    pub(crate) const fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
}

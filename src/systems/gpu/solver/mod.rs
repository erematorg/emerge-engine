use std::sync::Arc;

use crate::materials::registry::MaterialRegistry;
use crate::particle::Particle;
use crate::solver::config::SimConfig;

mod accessors;
mod device_lost;
mod encode_substep;
mod live_params;
mod particles;
mod profiling;
mod queries;
mod readback;
mod spawn;
mod step;

use super::buffers::GpuBuffers;
use super::pipeline::SimPipelines;
use super::step_params::{
    GpuAsflipParams, GpuDirectionalGripParams, GpuFieldEntry, GpuImpulseEntry,
    GpuMaterialMassParams, GpuResourceParams, GpuThermalParams, MAX_MATERIALS,
};

/// Workgroup sizes -- must match `@workgroup_size(...)` in the WGSL shaders.
/// grid_clear and grid_update are dispatched by active-block slot (`2 * NUM_BLOCKS`
/// workgroups, see grid_clear.wgsl/grid_update.wgsl), not grid resolution -- no WG_GRID
/// constant needed for either any more.
const WG_PARTICLES: u32 = 64; // p2g and g2p: 64-wide 1D workgroups

/// Shared between the wgpu map_async callback (any thread) and step_frame's poll.
type ReadbackResult = std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// GPU-backed MLS-MPM solver.
///
/// Pass sequence (see `encode_substep.rs` for the authoritative dispatch list --
/// several passes below are conditional, e.g. contact/mixture/thermal/resource):
///   Once per frame: particle_sort_clear → count → scan → scatter
///   Per substep:    active_block_refresh → grid_clear → p2g → grid_update → g2p → particles_update
///
/// Particles live in VRAM between frames; the CPU only touches them at spawn and for
/// plasticity readback (currently: none -- all plasticity runs in particles_update.wgsl).
pub struct GpuSimulation {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: GpuBuffers,
    pipelines: SimPipelines,
    config: SimConfig,
    registry: MaterialRegistry,
    /// CPU-side particle mirror. One frame behind the GPU when readback is strided.
    /// Access via `particles()` / `particles_mut()`. Do not replace the Vec directly.
    particles: Vec<Particle>,
    particle_count: usize,
    /// How many particles the per-particle GPU buffers (`buffers.particles`,
    /// `sorted_particle_ids`, `readback_staging`) actually have room for --
    /// always `>= particle_count`. `spawn_region` (spawn.rs) grows this with
    /// Vec-style amortized doubling on the rare call that exceeds it, so the
    /// common case (repeated small spawns under existing headroom) is a
    /// sub-range `write_buffer` instead of a full realloc + reupload + bind
    /// group rebuild every time. `remove_particles` (spawn.rs) resets this
    /// back down to its own new exact-fit size when it reallocates smaller --
    /// must stay in sync with the real buffer size everywhere buffers are
    /// reallocated, or the fast path in `spawn_region` would write past the
    /// buffer's actual end.
    particle_capacity: usize,
    last_sub_dt: f32,
    last_substeps: usize,
    /// One-frame-lagged max particle speed -- mirrors CPU's own
    /// `Simulation::last_max_particle_speed` exactly (same convention, same
    /// consumer: `SimConfig::fluid_near_wall_compression_mach_margin`'s
    /// predictive near-wall CFL tightening, see `step_frame`'s own scan).
    last_max_particle_speed: f32,
    frame_index: u64,
    /// `frame_index` at the most recent `spawn_region` call (0 = only the initial
    /// construction batch exists). Tracked so `step_frame`'s sleep-warmup window
    /// (`SLEEP_WARMUP_FRAMES`) can re-arm on every spawn, not just once at
    /// construction -- otherwise a particle spawned live long after frame 10 gets
    /// `sleep_threshold` applied on its very first substep at v=0, freezing it
    /// asleep before gravity ever touches it.
    last_spawn_frame: u64,
    /// GPU force-field entries -- uploaded to the force_fields_params uniform each substep.
    force_field_entries: Vec<GpuFieldEntry>,
    /// Frame counter used to stride CPU readbacks when all materials are GPU-resident.
    readback_frame: usize,
    /// Download CPU particle state every N step_frame calls when no CPU plasticity is needed.
    /// 1 = every frame (default, always accurate). 2+ = skip frames, reducing GPU stall cost.
    /// One-frame lag on sprite positions is invisible at 60fps.
    pub readback_stride: usize,
    /// Particle positions/materials changed -- sort + upload required before next GPU pass.
    /// Set by spawn, phase_transition, mark_particles_dirty().
    layout_dirty: bool,
    /// Pending impulses to apply on GPU at the start of the next step_frame.
    /// Applied via a dedicated compute pass that reads LIVE GPU particle positions,
    /// avoiding the stale-CPU-mirror artifacts from the old upload approach.
    pending_impulses: Vec<GpuImpulseEntry>,
    /// Pending force-sleep/force-wake-by-tag for the next step_frame, applied once in
    /// force_fields.wgsl then cleared. Minimal hook for LP's future chunk system -- see
    /// `sleep_tag`/`wake_tag` doc comments and the `GpuSleepWakeParams` layout.
    pending_sleep_tags: Vec<u32>,
    pending_wake_tags: Vec<u32>,
    /// Pending async readback -- Some while GPU → staging copy + mapping is in flight.
    /// Checked each step_frame; on completion, CPU particles are updated without blocking.
    /// Arc<Mutex<...>> so the wgpu callback (any thread) can signal the main thread.
    pending_readback: Option<ReadbackResult>,
    /// Count of async readback failures (`map_async` completing with `Err`) ever
    /// recovered from -- should be 0 in ordinary operation; nonzero signals something
    /// is stressing the GPU backend (rare on fast hardware, more likely on
    /// slow/software backends). See `GpuBuffers::abandon_readback`'s doc.
    pub readback_error_count: u64,
    /// Set once, permanently, if this instance's device is ever lost (e.g. an Out of
    /// Memory device loss under sustained load on slow/software GPU backends). A lost
    /// device cannot be un-lost; every further GPU call on it would panic, so
    /// `step_frame`/the blocking sync methods check this and become safe no-ops
    /// once set, rather than crashing. Always populated for `new()` instances;
    /// `with_device()` instances need one call to `enable_device_lost_detection()`
    /// first (see that method's doc for why it isn't automatic there -- a wgpu
    /// device can only have one lost-callback, so auto-registering on a
    /// possibly-shared device risks silently overwriting a caller's own).
    /// Callers should poll `device_lost_reason()` if they care why the sim went
    /// quiet -- this is deliberately observable, not silently swallowed.
    device_lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Per-pass GPU timestamp profiling -- see `enable_profiling()`. None unless explicitly
    /// turned on; zero cost to every other code path when not in use.
    profiling: Option<GpuProfiling>,
    /// Whether `profile_stamp` records the substep currently being encoded -- only the
    /// frame's last one, so the per-stage timestamps are one coherent substep.
    profile_this_substep: std::cell::Cell<bool>,
    /// One bind group per `step_params_pool` slot, built once and reused by every
    /// `step_frame()` call instead of being recreated per-substep-per-frame. At high
    /// substep counts, recreating thousands of bind groups every frame exhausts the
    /// GPU's descriptor allocator. The buffers a bind group points at
    /// (`step_params_pool[i]`) never change identity after construction, only their
    /// contents (rewritten every frame via `upload_step_params_at`) -- so the bind group
    /// itself can be built once and only needs rebuilding when `spawn_region`
    /// reallocates `buffers.particles` (see `rebuild_bind_group_pool`).
    bind_group_pool: Vec<wgpu::BindGroup>,
    /// Group 1 (contact subsystem) bind group -- built exactly once, see
    /// `SimPipelines::make_contact_bind_group`'s doc for why it never needs rebuilding
    /// the way `bind_group_pool` does.
    contact_bind_group: wgpu::BindGroup,
    /// Group 2 (thermal subsystem) bind group -- same "built once" shape as
    /// `contact_bind_group`, see `SimPipelines::make_thermal_bind_group`'s doc.
    thermal_bind_group: wgpu::BindGroup,
    /// Live day-night/ambient thermal diffusion state -- `enabled: 0` (default) skips
    /// all 4 thermal passes entirely, every existing scene pays nothing. Set via
    /// `attach_thermal_gpu`/`set_thermal_ambient`.
    thermal_params: GpuThermalParams,
    /// `heat_capacity` as passed to `attach_thermal_gpu`, retained CPU-side only for
    /// `phase_transition`'s real latent-heat debit (`ΔT = latent_heat / heat_capacity`,
    /// mirrors CPU's `Simulation::phase_transition`) -- NOT baked into `thermal_params`,
    /// whose `alpha` field already folds it into the diffusion coefficient and can't be
    /// un-folded back out. `None` until `attach_thermal_gpu` is called, matching CPU's
    /// `self.thermal.is_none()` gate (no thermal model configured = no debit applied).
    thermal_heat_capacity: Option<f32>,
    /// Group 3 (resource regrowth subsystem) bind group -- built exactly once, see
    /// `SimPipelines::make_resource_bind_group`'s doc.
    resource_bind_group: wgpu::BindGroup,
    /// Live resource-regrowth state -- `enabled: 0` (default) skips all 4 resource
    /// passes entirely. Set via `attach_resource_field_gpu`.
    resource_params: GpuResourceParams,
    /// Live ASFLIP state -- `enabled: 0` (default) makes `step_frame` dispatch the
    /// ordinary split g2p/particles_update pair unchanged; `enabled: 1` dispatches
    /// `g2p_asflip_fused` instead (see `SubstepGates::asflip_active`, `encode_substep.rs`).
    /// Shares `resource_bind_group` (group 3) -- see `SimPipelines::new`'s module doc
    /// comment on why. Set via `attach_asflip_gpu`.
    asflip_params: GpuAsflipParams,
    /// Live `ColorMode::GridVolume` material-mass state -- `enabled: 0` (default)
    /// skips the extra P2G scatter and grid_clear zeroing entirely. Shares
    /// `contact_bind_group` (group 1) -- see `SimPipelines::make_contact_bind_group`'s
    /// doc for why. Set via `attach_grid_material_render_gpu`.
    material_mass_params: GpuMaterialMassParams,
    /// Spatial acceleration for `particles_near`/`count_near`/`group_centroid` --
    /// ported from `solver::Simulation`'s `SpatialHash`.
    ///
    /// Lazily rebuilt: `step_frame()`'s default `readback_stride=1` means a readback
    /// completes every frame regardless of whether any query runs that frame.
    /// `RefCell` + `spatial_hash_dirty` defer the actual rebuild to the first query
    /// call after new data lands, instead of paying it unconditionally on every
    /// readback -- see `ensure_spatial_hash_fresh` in `queries.rs`. Matches the
    /// discipline the CPU `Simulation` follows for the same queries: hash rebuilt
    /// once per external `step()`, since LP queries happen between frames, never
    /// mid-substep. Zero staleness change: a query after a dirty
    /// readback still sees the exact same freshly-landed positions, just computed on
    /// demand.
    spatial_hash: std::cell::RefCell<crate::solver::spatial_hash::SpatialHash>,
    /// Set whenever `self.particles` changes and the spatial hash hasn't been rebuilt
    /// to match yet. Cleared by `ensure_spatial_hash_fresh` (queries.rs) on the first
    /// query after that, or by `rebuild_spatial_hash` (spawn.rs) for callers that need
    /// it fresh immediately (e.g. right after `spawn_region` returns a usable range).
    spatial_hash_dirty: std::cell::Cell<bool>,
    /// CPU-side wall-clock breakdown of the last `step_frame()` call (cfl_scan_ns,
    /// encode_ns, submit_ns, readback_ns, total_ns) -- `Instant::now()` calls are
    /// themselves nanosecond-cost, so these are always recorded, not gated behind
    /// `enable_profiling()`. Read via `last_cpu_timings_ns()`. `total_ns` minus the sum of
    /// the other four reveals any unbracketed cost.
    last_cpu_timings: (f32, f32, f32, f32, f32),
    /// Live directional grip friction state -- GPU counterpart to
    /// `DirectionalContactGrip`. Uploaded fresh every `step_frame` (see `step.rs`), so
    /// unlike `contact_bind_group` there's no buffer to rebuild here, just a plain
    /// field updated via `set_grip_direction`/`set_grip_friction`. Starts symmetric
    /// (no directional bias) -- real Coulomb friction at `config.contact_friction`,
    /// identical to every scene before this field existed until a caller opts in.
    grip_params: GpuDirectionalGripParams,
}

/// One [begin, end] timestamp pair per labeled compute pass in `encode_substep`, written
/// every substep (later substeps overwrite earlier ones within the same `step_frame()`
/// call -- fine for finding the dominant cost, since substeps cost about the same each
/// time; not meant to capture per-substep variance).
const PROFILE_PASS_LABELS: &[&str] = &[
    "active_block_refresh (sort)",
    "grid_clear",
    "p2g",
    "gather_contact_points",
    "grid_update",
    "resolve_contact",
    "g2p_update (gather+update+forces) / g2p_asflip_fused",
    "force_fields (ASFLIP only) / cfl_commit",
];

struct GpuProfiling {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    timestamp_period_ns: f32,
}

/// One bind group per `step_params_pool` slot -- see `GpuSimulation::bind_group_pool`'s
/// doc comment for why this is built once and reused rather than recreated per substep.
fn build_bind_group_pool(
    device: &wgpu::Device,
    pipelines: &SimPipelines,
    buffers: &GpuBuffers,
) -> Vec<wgpu::BindGroup> {
    buffers
        .step_params_pool
        .iter()
        .map(|step_params| pipelines.make_bind_group(device, buffers, step_params))
        .collect()
}

impl GpuSimulation {
    /// Create a GpuSimulation, initialize wgpu, upload initial particle and material data.
    ///
    /// `async` because wgpu adapter/device requests are async.
    /// In examples, wrap with `pollster::block_on(GpuSimulation::new(...))`.
    pub async fn new(
        config: SimConfig,
        particles: Vec<Particle>,
        registry: MaterialRegistry,
    ) -> Self {
        let instance = super::create_wgpu_instance();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");

        // Request the adapter's actual limits, not wgpu's conservative defaults (128MiB
        // storage binding). Hardware commonly supports far more (e.g. 2047MiB on desktop
        // GPUs) -- capping at the default artificially shrinks the single-buffer particle/grid
        // ceiling well below what the device can actually do.
        //
        // TIMESTAMP_QUERY requested opportunistically (only if the adapter actually supports
        // it) so `enable_profiling()` can work later without requiring it everywhere --
        // hardware/backends that lack it fall back to empty, identical to before this line
        // existed.
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("emerge_gpu"),
                required_features: features,
                required_limits: adapter.limits(),
                ..Default::default() // experimental_features, trace, memory_hints
            })
            .await
            .expect("failed to create wgpu device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let sim = Self::with_device(device, queue, config, particles, registry);

        // This device is EXCLUSIVELY ours (just created above, no other caller could
        // have registered a competing handler on it yet), so it's always safe to
        // enable device-lost detection automatically here.
        sim.enable_device_lost_detection();
        sim
    }

    /// Build a `GpuSimulation` on an existing device/queue so its GPU buffers can be
    /// shared with a renderer or surface on the same device -- required for the
    /// zero-readback [`crate::render::Renderer::render_gpu`] path. `new()` creates its
    /// own headless device instead, which is correct for compute-only or CPU-readback
    /// workflows but cannot share GPU buffers with another device.
    pub fn with_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: SimConfig,
        particles: Vec<Particle>,
        registry: MaterialRegistry,
    ) -> Self {
        // Real regression guard (corrected -- a first version of this check
        // read `MaterialParams::model` from `all_params()` and looked for
        // `10`, which is UNREACHABLE: `NaccMaterial::params()` deliberately
        // emits `ConstitutiveModel::NeoHookean as u32` (2), not its own
        // `constitutive_model()` value (10) -- see that method's own
        // comment ("GPU uses NeoHookean stress... Plasticity runs CPU-only
        // via needs_cpu_update=true"). So a real NaccMaterial's GPU stress
        // is NOT zero -- it silently runs `case 2u`'s NeoHookean law
        // (`kappa*ln(J)`) instead of NACC's own real volumetric law
        // (`kappa/2*(J^2-1)`, see `nacc.rs::kirchhoff_stress`), which is
        // the actual, original finding here (external review). Its
        // `needs_cpu_update` fallback (issue #5) DOES correctly re-project
        // `deformation_gradient` onto the real Cam-Clay yield surface every
        // frame, but the STRESS feeding that substep's P2G grid transfer is
        // still NeoHookean's, not NACC's. Must check the real trait method
        // (`constitutive_model()`, via `constitutive_model_of`), not the
        // GPU-upload params -- those are exactly the two things this bug
        // conflates. See `ConstitutiveModel::Nacc`'s own doc for the full
        // finding. Fail loudly here instead of silently running the wrong
        // constitutive law -- use `GranularFluidMaterial` instead, already
        // fully GPU-native.
        for id in 0..registry.len() as u32 {
            if registry.constitutive_model_of(id) == crate::materials::ConstitutiveModel::Nacc {
                panic!(
                    "NaccMaterial (material_id {id}) has no real GPU stress path -- its \
                     params() deliberately uploads as NeoHookean (model 2), so p2g.wgsl \
                     silently runs NeoHookean's kappa*ln(J) volumetric law instead of \
                     NACC's own kappa/2*(J^2-1) (see ConstitutiveModel::Nacc's own doc). \
                     Its CPU plasticity fallback (issue #5) keeps F on the right yield \
                     surface, but the stress driving grid momentum transfer is still \
                     wrong. Use GranularFluidMaterial instead for a GPU-native \
                     granular-fluid scene."
                );
            }
        }

        // Real fix, issue #29: `NoCompressionMaterial` has no `p2g.wgsl`/
        // `particles_update.wgsl` case at all (unlike NACC above, no
        // compensating CPU fallback exists either) -- an unrecognised
        // `mat.model` falls through to `default: { return mat2x2<f32>(); }`,
        // exact zero stress every substep. A cable/membrane/tendon would
        // silently free-fall with no tension resistance, contradicting the
        // material's entire purpose. Fail loudly here instead, same pattern
        // as NACC's own guard -- no real GPU stress path exists yet for
        // this model, see issue #29 for the WGSL-port option, not pursued
        // here.
        for id in 0..registry.len() as u32 {
            if registry.constitutive_model_of(id)
                == crate::materials::ConstitutiveModel::NoCompression
            {
                panic!(
                    "NoCompressionMaterial (material_id {id}) has no GPU stress path at all -- \
                     p2g.wgsl/particles_update.wgsl have no case for this model and there is no \
                     CPU fallback, so it would silently run with exact zero stress (see \
                     ConstitutiveModel::NoCompression's own doc). Use this material on the CPU \
                     Simulation backend instead until issue #29's real WGSL port lands."
                );
            }
        }

        let material_params = registry.all_params();

        // Run init_particle before uploading. Mirrors Simulation::spawn_region().
        // Materials that seed plastic state (Snow: Jp=1, Sand: q=neutral) start wrong
        // without this.
        let mut initialized = particles;
        for p in &mut initialized {
            registry.get(p.material_id).init_particle(p);
        }
        let particle_count = initialized.len();

        let buffers = GpuBuffers::new(
            &device,
            particle_count,
            config.grid_res,
            MAX_MATERIALS,
            config.max_substeps_per_step,
        );

        buffers.upload_particles(&queue, &initialized);
        buffers.upload_materials(&queue, &material_params);

        let mut pipelines = SimPipelines::new(&device, config.grid_res);
        pipelines.specialize_g2p_update(&device, registry.model_mask());
        // A zero-sized particle buffer (no initial particles -- e.g. LP constructs
        // empty, then adds terrain/water/creature via spawn_region) fails bind group
        // creation outright ("binding size is zero"). spawn_region already rebuilds
        // this pool once real particles exist; skip the doomed eager build until then.
        let bind_group_pool = if particle_count > 0 {
            build_bind_group_pool(&device, &pipelines, &buffers)
        } else {
            Vec::new()
        };
        // Contact group's buffers are all fixed grid_res²-sized, never reallocated by
        // spawn_region -- safe to build unconditionally, unlike bind_group_pool above.
        let contact_bind_group = pipelines.make_contact_bind_group(&device, &buffers);
        // Thermal group's buffers are also all fixed grid_res²-sized -- same reasoning.
        let thermal_bind_group = pipelines.make_thermal_bind_group(&device, &buffers);
        let thermal_params = GpuThermalParams::disabled();
        let resource_bind_group = pipelines.make_resource_bind_group(&device, &buffers);
        let resource_params = GpuResourceParams::disabled();
        let asflip_params = GpuAsflipParams::disabled();
        let material_mass_params = GpuMaterialMassParams::disabled();

        let mut spatial_hash = crate::solver::spatial_hash::SpatialHash::new(config.grid_cell_size);
        spatial_hash.rebuild(
            &initialized.iter().map(|p| p.x).collect::<Vec<_>>(),
            initialized.len(),
        );
        let grip_params = GpuDirectionalGripParams::symmetric(config.contact_friction);

        Self {
            device,
            queue,
            buffers,
            pipelines,
            config,
            registry,
            particles: initialized,
            particle_count,
            particle_capacity: particle_count,
            last_sub_dt: config.dt,
            last_substeps: 0,
            last_max_particle_speed: 0.0,
            frame_index: 0,
            last_spawn_frame: 0,
            force_field_entries: Vec::new(),
            readback_frame: 0,
            readback_stride: 1,
            layout_dirty: true, // seed particle_sort on first step_frame
            pending_impulses: Vec::new(),
            pending_sleep_tags: Vec::new(),
            pending_wake_tags: Vec::new(),
            pending_readback: None,
            readback_error_count: 0,
            device_lost: std::sync::Arc::new(std::sync::Mutex::new(None)),
            profiling: None,
            profile_this_substep: std::cell::Cell::new(true),
            last_cpu_timings: (0.0, 0.0, 0.0, 0.0, 0.0),
            bind_group_pool,
            contact_bind_group,
            thermal_bind_group,
            thermal_params,
            thermal_heat_capacity: None,
            resource_bind_group,
            resource_params,
            asflip_params,
            material_mass_params,
            spatial_hash: std::cell::RefCell::new(spatial_hash),
            spatial_hash_dirty: std::cell::Cell::new(false),
            grip_params,
        }
    }
}

// Trivial public accessors/setters (GPU handle sharing, CPU-mirror access, material
// registry management, frame/timing state) -- split into their own file (`mod
// accessors` declared up top alongside the other 9 submodules), see accessors.rs's
// own doc comment.

// White-box device-lost tests -- split into their own file (was ~240 lines inline
// here), see device_lost_tests.rs's own doc comment for why it must stay a
// submodule (super::* private-field access) rather than a standalone integration test.
#[cfg(test)]
mod device_lost_tests;

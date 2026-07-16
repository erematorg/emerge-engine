use std::sync::Arc;

use crate::materials::registry::MaterialRegistry;
use crate::particle::Particle;
use crate::solver::config::SimConfig;

mod device_lost;
mod particles;
mod profiling;
mod queries;
mod readback;
mod spawn;
mod step;

use super::buffers::GpuBuffers;
use super::pipeline::SimPipelines;
use super::step_params::{
    GpuDirectionalGripParams, GpuFieldEntry, GpuImpulseEntry, GpuThermalParams, MAX_MATERIALS,
    MAX_SLEEP_WAKE_TAGS,
};

/// Workgroup sizes — must match `@workgroup_size(...)` in the WGSL shaders.
/// grid_clear and grid_update are dispatched by active-block slot (`2 * NUM_BLOCKS`
/// workgroups, see grid_clear.wgsl/grid_update.wgsl), not grid resolution — no WG_GRID
/// constant needed for either any more.
const WG_PARTICLES: u32 = 64; // p2g and g2p: 64-wide 1D workgroups

/// Shared between the wgpu map_async callback (any thread) and step_frame's poll.
type ReadbackResult = std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// GPU-backed MLS-MPM solver.
///
/// Pass sequence:
///   Once per frame: particle_sort (identity permutation → sorted_particle_ids)
///   Per substep:    grid_clear → p2g → grid_update → g2p → particles_update → force_fields
///
/// Particles live in VRAM between frames; the CPU only touches them at spawn and for
/// plasticity readback (currently: none — all plasticity runs in particles_update.wgsl).
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
    last_sub_dt: f32,
    last_substeps: usize,
    frame_index: u64,
    /// GPU force-field entries — uploaded to the force_fields_params uniform each substep.
    force_field_entries: Vec<GpuFieldEntry>,
    /// Frame counter used to stride CPU readbacks when all materials are GPU-resident.
    readback_frame: usize,
    /// Download CPU particle state every N step_frame calls when no CPU plasticity is needed.
    /// 1 = every frame (default, always accurate). 2+ = skip frames, reducing GPU stall cost.
    /// One-frame lag on sprite positions is invisible at 60fps.
    pub readback_stride: usize,
    /// Particle positions/materials changed — sort + upload required before next GPU pass.
    /// Set by spawn, phase_transition, mark_particles_dirty().
    layout_dirty: bool,
    /// Pending impulses to apply on GPU at the start of the next step_frame.
    /// Applied via a dedicated compute pass that reads LIVE GPU particle positions,
    /// avoiding the stale-CPU-mirror artifacts from the old upload approach.
    pending_impulses: Vec<GpuImpulseEntry>,
    /// Pending force-sleep/force-wake-by-tag for the next step_frame, applied once in
    /// force_fields.wgsl then cleared. Minimal hook for LP's future chunk system — see
    /// `sleep_tag`/`wake_tag` doc comments and the `GpuSleepWakeParams` layout.
    pending_sleep_tags: Vec<u32>,
    pending_wake_tags: Vec<u32>,
    /// Pending async readback — Some while GPU → staging copy + mapping is in flight.
    /// Checked each step_frame; on completion, CPU particles are updated without blocking.
    /// Arc<Mutex<...>> so the wgpu callback (any thread) can signal the main thread.
    pending_readback: Option<ReadbackResult>,
    /// Real, honest count of async readback failures (`map_async` completing with
    /// `Err`) ever recovered from — should be 0 in ordinary operation on real
    /// hardware; nonzero is a real signal something is stressing the GPU backend
    /// (rare on fast hardware, more likely on slow/software backends). Added
    /// 2026-07-05 alongside the fix for the failure path leaking the staging
    /// buffer's mapped state — see `GpuBuffers::abandon_readback`'s doc.
    pub readback_error_count: u64,
    /// Set once, permanently, if this instance's device is ever lost (confirmed
    /// real cause of emerge issue #10 — see project memory
    /// `gpu_readback_error_path_bug_issue10`: a genuine `Out of Memory` device
    /// loss under sustained load on slow/software GPU backends). A lost device
    /// cannot be un-lost; every further GPU call on it would panic, so
    /// `step_frame`/the blocking sync methods check this and become safe no-ops
    /// once set, rather than crashing. Always populated for `new()` instances;
    /// `with_device()` instances need one call to `enable_device_lost_detection()`
    /// first (see that method's doc for why it isn't automatic there — a wgpu
    /// device can only have one lost-callback, so auto-registering on a
    /// possibly-shared device risks silently overwriting a caller's own).
    /// Callers should poll `device_lost_reason()` if they care why the sim went
    /// quiet — this is deliberately observable, not silently swallowed.
    device_lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Per-pass GPU timestamp profiling — see `enable_profiling()`. None unless explicitly
    /// turned on; zero cost to every other code path when not in use.
    profiling: Option<GpuProfiling>,
    /// One bind group per `step_params_pool` slot, built once and reused by every
    /// `step_frame()` call instead of being recreated per-substep-per-frame. At high
    /// substep counts (LP's stiff-terrain scenes routinely need ~5-6k substeps/frame)
    /// recreating thousands of bind groups every frame exhausted the GPU's descriptor
    /// allocator within seconds (`wgpu error: Out of Memory` from `queue.submit`,
    /// reported against LP's own scene 2026-07-01). The buffers a bind group points at
    /// (`step_params_pool[i]`) never change identity after construction, only their
    /// contents (rewritten every frame via `upload_step_params_at`) — so the bind group
    /// itself can be built once and only needs rebuilding when `spawn_region`
    /// reallocates `buffers.particles` (see `rebuild_bind_group_pool`).
    bind_group_pool: Vec<wgpu::BindGroup>,
    /// Group 1 (contact subsystem) bind group — built exactly once, see
    /// `SimPipelines::make_contact_bind_group`'s doc for why it never needs rebuilding
    /// the way `bind_group_pool` does.
    contact_bind_group: wgpu::BindGroup,
    /// Group 2 (thermal subsystem) bind group — same "built once" shape as
    /// `contact_bind_group`, see `SimPipelines::make_thermal_bind_group`'s doc.
    thermal_bind_group: wgpu::BindGroup,
    /// Live day-night/ambient thermal diffusion state — `enabled: 0` (default) skips
    /// all 4 thermal passes entirely, every existing scene pays nothing. Set via
    /// `attach_thermal_gpu`/`set_thermal_ambient`.
    thermal_params: GpuThermalParams,
    /// Real spatial acceleration for `particles_near`/`count_near`/`group_centroid` --
    /// ported from `solver::Simulation`'s already-proven `SpatialHash` (was previously
    /// wired into the CPU-only `Simulation` but not `GpuSimulation`, meaning every
    /// caller of these three query methods on the GPU path -- the one LP actually uses --
    /// paid a full O(N) linear scan per call regardless of how local the query was.
    ///
    /// Lazily rebuilt (2026-07-12): measured 6.76ms at 50k particles in isolation --
    /// real, not negligible, and `step_frame()`'s default `readback_stride=1` means a
    /// readback (and formerly, an eager rebuild) completes EVERY frame regardless of
    /// whether any query runs that frame. `RefCell` + `spatial_hash_dirty` defer the
    /// actual rebuild to the first query call after new data lands, instead of paying
    /// it unconditionally on every readback -- see `ensure_spatial_hash_fresh` in
    /// `queries.rs`. This brings the GPU path in line with a discipline the CPU
    /// `Simulation` already follows for the same queries (`ARCHITECTURE.md` §4: hash
    /// rebuilt once per external `step()`, "because LP queries happen between frames,
    /// never mid-substep") -- the GPU path was rebuilding MORE eagerly than that, for
    /// the same consumer. Same real-world precedent as Box2D's incremental broad-phase
    /// (only re-touch what actually needs it) and SPH's Verlet-list neighbor caching
    /// (retain a neighbor structure across steps rather than rebuilding every one) --
    /// not an invented shortcut. Zero staleness change: a query after a dirty readback
    /// still sees the exact same freshly-landed positions, just computed on demand.
    spatial_hash: std::cell::RefCell<crate::solver::spatial_hash::SpatialHash>,
    /// Set whenever `self.particles` changes and the spatial hash hasn't been rebuilt
    /// to match yet. Cleared by `ensure_spatial_hash_fresh` (queries.rs) on the first
    /// query after that, or by `rebuild_spatial_hash` (spawn.rs) for callers that need
    /// it fresh immediately (e.g. right after `spawn_region` returns a usable range).
    spatial_hash_dirty: std::cell::Cell<bool>,
    /// CPU-side wall-clock breakdown of the last `step_frame()` call (cfl_scan_ns,
    /// encode_ns, submit_ns, readback_ns, total_ns) — `Instant::now()` calls are
    /// themselves nanosecond-cost, so these are always recorded, not gated behind
    /// `enable_profiling()`. Read via `last_cpu_timings_ns()`. `total_ns` minus the sum of
    /// the other four reveals any unbracketed cost.
    last_cpu_timings: (f32, f32, f32, f32, f32),
    /// Live directional grip friction state — GPU counterpart to
    /// `DirectionalContactGrip`. Uploaded fresh every `step_frame` (see `step.rs`), so
    /// unlike `contact_bind_group` there's no buffer to rebuild here, just a plain
    /// field updated via `set_grip_direction`/`set_grip_friction`. Starts symmetric
    /// (no directional bias) — real Coulomb friction at `config.contact_friction`,
    /// identical to every scene before this field existed until a caller opts in.
    grip_params: GpuDirectionalGripParams,
}

/// One [begin, end] timestamp pair per labeled compute pass in `encode_substep`, written
/// every substep (later substeps overwrite earlier ones within the same `step_frame()`
/// call — fine for finding the dominant cost, since substeps cost about the same each
/// time; not meant to capture per-substep variance).
const PROFILE_PASS_LABELS: &[&str] = &[
    "active_block_refresh (sort)",
    "grid_clear",
    "p2g",
    "gather_contact_points",
    "grid_update",
    "resolve_contact",
    "g2p",
    "particles_update",
    "force_fields",
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

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
        // GPUs) — capping at the default artificially shrinks the single-buffer particle/grid
        // ceiling well below what the device can actually do.
        //
        // TIMESTAMP_QUERY requested opportunistically (only if the adapter actually supports
        // it) so `enable_profiling()` can work later without requiring it everywhere —
        // hardware/backends that lack it fall back to empty, identical to before this line
        // existed.
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
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

        // Real device-lost detection (confirmed cause of emerge issue #10, see
        // project memory) -- this device is EXCLUSIVELY ours (just created above,
        // no other caller could have registered a competing handler on it yet),
        // so it's always safe to enable it automatically here.
        sim.enable_device_lost_detection();
        sim
    }

    /// Build a `GpuSimulation` on an existing device/queue so its GPU buffers can be
    /// shared with a renderer or surface on the same device — required for the
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

        let pipelines = SimPipelines::new(&device);
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
            last_sub_dt: config.dt,
            last_substeps: 0,
            frame_index: 0,
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
            last_cpu_timings: (0.0, 0.0, 0.0, 0.0, 0.0),
            bind_group_pool,
            contact_bind_group,
            thermal_bind_group,
            thermal_params,
            spatial_hash: std::cell::RefCell::new(spatial_hash),
            spatial_hash_dirty: std::cell::Cell::new(false),
            grip_params,
        }
    }

    /// Returns (cfl_scan_ns, encode_ns, wait_ns, readback_ns, total_ns) from the last
    /// `step_frame()` call. `encode_ns` is pure CPU-side command-building time (bind
    /// group already cached, just recording dispatches); `wait_ns` (renamed from the
    /// old always-zero `submit_ns` -- multi-chunk frames now really do block between
    /// chunks) is time spent in `device.poll(wait_indefinitely())` between substep
    /// batches, i.e. real GPU execution time for scenes needing >64 substeps/frame.
    pub fn last_cpu_timings_ns(&self) -> (f32, f32, f32, f32, f32) {
        self.last_cpu_timings
    }

    /// Force every particle with `user_tag == tag` asleep, regardless of velocity,
    /// applied at the start of the next `step_frame()`. P2G still scatters for them
    /// (see `gpu_sleep_wake_phase1` memory note — sleeping particles must keep
    /// providing structural support); only their own gather/integration/force-field
    /// work is skipped.
    ///
    /// Minimal hook, not a chunk system: this just lets a caller (e.g. LP's future
    /// chunk loader, once it exists) force-sleep a tagged group by distance instead
    /// of waiting for velocity to drop. Mirrors the CPU `Simulation::sleep_tag` API.
    pub fn sleep_tag(&mut self, tag: u32) {
        if self.pending_sleep_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_sleep_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU sleep-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) — tag dropped"
            );
        }
    }

    /// Force every particle with `user_tag == tag` awake, regardless of grid activity.
    /// Mirrors the CPU `Simulation::wake_tag` API. See `sleep_tag` doc comment.
    pub fn wake_tag(&mut self, tag: u32) {
        if self.pending_wake_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_wake_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU wake-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) — tag dropped"
            );
        }
    }

    /// Mark CPU particles as layout-changed (positions/materials) — triggers sort + upload.
    pub fn mark_particles_dirty(&mut self) {
        self.layout_dirty = true;
    }

    /// Upload revised material params (e.g., if interactive sliders change them).
    pub fn upload_materials(&self) {
        self.buffers
            .upload_materials(&self.queue, &self.registry.all_params());
    }

    pub fn registry(&self) -> &MaterialRegistry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut MaterialRegistry {
        &mut self.registry
    }

    /// The wgpu Device — share with the LP render system to read the particle buffer directly.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// The wgpu Queue — share with the LP render system for command submission.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// The GPU particle storage buffer — bind this in LP's custom render shader.
    /// Layout: `array<Particle>`, each Particle is 112 bytes, repr(C).
    /// Stays in VRAM between frames; read-only from the render side.
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.particles
    }

    /// Read-only access to the CPU particle mirror (one frame behind GPU when strided).
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Mutable access to the CPU particle mirror.
    ///
    /// **CFL WARNING:** velocity changes bypass the solver's CFL clamp.
    /// For gameplay impulses use `apply_impulse` / `apply_radial_impulse` instead.
    /// After modifying, call `mark_particles_dirty()` so the GPU sees the changes.
    pub fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    /// Append a new particle region to the simulation.
    ///
    /// Generates particles CPU-side, appends to the internal mirror, recomputes
    /// initial volumes for all particles, then reallocates the GPU particle buffer
    /// to fit the new total and uploads all particles.
    ///
    /// Returns the index range the new particles occupy in the internal mirror.
    /// LP uses this as `creature_id → particle_range` for ownership tracking.
    ///
    /// Call before `step_frame` — mid-frame spawning is not supported.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn set_gravity(&mut self, gravity: glam::Vec2) {
        self.config.gravity = gravity;
    }

    /// Replace the default material and re-upload the materials buffer.
    pub fn set_default_material(&mut self, material: Box<dyn crate::materials::MaterialModel>) {
        self.registry.set_default(material);
        self.upload_materials();
    }

    pub fn gravity(&self) -> glam::Vec2 {
        self.config.gravity
    }

    /// The live GPU grid buffer (STORAGE | COPY_SRC).
    /// Layout: `array<Cell>` where Cell = { momentum: vec2, mass: f32, _pad: f32 } (16 bytes).
    /// Consumers (e.g. LP's metaball renderer) can bind this read-only in their own compute pass.
    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.grid
    }

    /// Register a material, auto-assigning the next available ID.
    ///
    /// Mirrors `Simulation::register_material` — use this instead of `set_material`
    /// when you don't want to track IDs manually. Returns a typed handle.
    ///
    /// LP pattern: call at world-init time to build a material palette, then
    /// use `handle.id()` in `SpawnRegion::for_sim(...).material(handle.id())`.
    pub fn register_material(
        &mut self,
        material: Box<dyn crate::materials::MaterialModel>,
    ) -> crate::solver::handle::MaterialHandle {
        let id = self.registry.next_id();
        self.registry.insert(id, material);
        self.upload_materials();
        crate::solver::handle::MaterialHandle(id)
    }

    /// Register or replace a material by explicit ID and re-upload the materials buffer.
    pub fn set_material(
        &mut self,
        material_id: u32,
        material: Box<dyn crate::materials::MaterialModel>,
    ) {
        self.registry.insert(material_id, material);
        self.upload_materials();
    }

    /// The sub-dt used in the last substep of the most recent `step_frame` call.
    pub fn effective_dt(&self) -> f32 {
        self.last_sub_dt
    }

    /// Number of substeps run during the most recent `step_frame` call.
    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    /// Total frames stepped since creation.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

// White-box device-lost tests -- split into their own file (was ~240 lines inline
// here), see device_lost_tests.rs's own doc comment for why it must stay a
// submodule (super::* private-field access) rather than a standalone integration test.
#[cfg(test)]
mod device_lost_tests;

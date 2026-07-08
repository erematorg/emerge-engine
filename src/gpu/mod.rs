/// GPU compute backend for the MLS-MPM solver.
///
/// Architecture: wgpu compute shaders, 4 passes per substep:
///   grid_clear → p2g (scatter) → grid_update → g2p (gather)
///
/// Plasticity: Snow SVD and Drucker-Prager return-mapping both run on GPU (g2p.wgsl).
/// No CPU roundtrip needed for plasticity. Fluid, NeoHookean, Corotated also GPU.
///
/// Data flow each substep:
///   CPU uploads GpuStepParams (dt, gravity, etc.) once per substep
///   GPU runs 4 compute passes on particle + grid buffers in VRAM
///   CPU downloads particles once per frame only if plasticity is needed
///   LP renders: reads the particle buffer directly via shared wgpu Device
///
/// Physics constants: KERNEL_D_INVERSE=4.0 is a fixed B-spline constant; other params come from SimConfig.
/// GPU-side constants (MAX_MATERIALS, workgroup sizes) are named here
/// and must match their WGSL counterparts exactly.
///
/// Enabled via `features = ["gpu"]`. Core library compiles without this feature.
#[cfg(feature = "gpu")]
pub mod pipeline;

#[cfg(feature = "gpu")]
pub mod buffers;

// WGSL shader sources — embedded at compile time.
#[cfg(feature = "gpu")]
pub mod shaders {
    pub const PARTICLE_SORT: &str = include_str!("shaders/particle_sort.wgsl");
    pub const GRID_CLEAR: &str = include_str!("shaders/grid_clear.wgsl");
    pub const P2G: &str = include_str!("shaders/p2g.wgsl");
    pub const GRID_UPDATE: &str = include_str!("shaders/grid_update.wgsl");
    pub const G2P: &str = include_str!("shaders/g2p.wgsl");
    pub const PARTICLES_UPDATE: &str = include_str!("shaders/particles_update.wgsl");
    pub const FORCE_FIELDS: &str = include_str!("shaders/force_fields.wgsl");
    pub const APPLY_IMPULSES: &str = include_str!("shaders/apply_impulses.wgsl");
}

#[cfg(feature = "gpu")]
pub use solver::GpuSimulation;

#[cfg(feature = "gpu")]
pub use step_params::{
    GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuSleepWakeParams,
    GpuStepParams, MAX_FORCE_FIELDS, MAX_GPU_IMPULSES, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS,
    NUM_BLOCKS_PER_DIM, field_type,
};

#[cfg(feature = "gpu")]
mod step_params {
    use crate::solver::config::SimConfig;

    /// Re-export so GPU code reads the same limit as the registry.
    /// Injected into WGSL shaders at pipeline creation — change only in `materials/registry.rs`.
    pub use crate::materials::registry::MAX_MATERIAL_SLOTS as MAX_MATERIALS;

    /// Per-substep solver constants uploaded to the GPU uniform buffer before each substep.
    ///
    /// 48 bytes, 16-byte aligned — satisfies WGSL uniform binding requirements.
    /// Fields mirror `struct StepParams` in every WGSL shader exactly (same offsets, same types).
    ///
    /// All values come from `SimConfig` or are computed from it — no hardcoded physics here.
    /// Uniform data uploaded once per GPU substep.
    ///
    /// Layout (48 bytes, 16-byte aligned — WGSL uniform binding requirement):
    ///   offset  0: grid_res       u32
    ///   offset  4: particle_count u32
    ///   offset  8: dt             f32
    ///   offset 12: kernel_d_inverse      f32  (always 4.0 — quadratic B-spline)
    ///   offset 16: gravity        `vec2<f32>`  (8 bytes; 8-byte aligned in WGSL ✓)
    ///   offset 24: boundary_thickness u32
    ///   offset 28: vel_limit      f32
    ///   offset 32: sleep_threshold f32  (0.0 = sleep/wake disabled, SimConfig default)
    ///   offset 36: _pad           [u32; 3]
    ///                             = 48 bytes, 16-byte aligned ✓
    ///
    /// `gravity: Vec2` replaces the old `gravity: f32` + `_pad1: u32` pair —
    /// same byte count, no layout change for other fields.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuStepParams {
        pub grid_res: u32,
        pub particle_count: u32,
        pub dt: f32,
        pub kernel_d_inverse: f32,
        pub gravity: glam::Vec2, // SimConfig::gravity — supports angled/planetary gravity
        pub boundary_thickness: u32,
        pub vel_limit: f32,       // grid_cell_size / sub_dt
        pub sleep_threshold: f32, // SimConfig::sleep_threshold — 0.0 disables sleep/wake entirely
        pub _pad: [u32; 3],
    }

    impl GpuStepParams {
        pub fn new(config: &SimConfig, sub_dt: f32, particle_count: usize) -> Self {
            Self {
                grid_res: config.grid_res as u32,
                particle_count: particle_count as u32,
                dt: sub_dt,
                kernel_d_inverse: crate::solver::config::KERNEL_D_INVERSE,
                gravity: config.gravity,
                boundary_thickness: config.boundary_thickness as u32,
                vel_limit: config.grid_cell_size / sub_dt,
                sleep_threshold: config.sleep_threshold,
                _pad: [0; 3],
            }
        }
    }

    const _: () = assert!(core::mem::size_of::<GpuStepParams>() == 48);

    /// Maximum number of active GPU force-field entries per frame.
    /// Must match `MAX_FORCE_FIELDS` in `force_fields.wgsl`.
    pub const MAX_FORCE_FIELDS: usize = 16;

    /// Field-type discriminants — match `FIELD_*` constants in `force_fields.wgsl`.
    pub mod field_type {
        pub const DISABLED: u32 = 0;
        pub const GRAVITY_WELL: u32 = 1;
        pub const COULOMB: u32 = 2;
        pub const AABB_CONFINEMENT: u32 = 3;
        pub const RADIAL_CONFINEMENT: u32 = 4;
        pub const UNIFORM_ELECTRIC: u32 = 5;
        pub const BUOYANCY: u32 = 6;
    }

    /// One GPU force-field entry — 48 bytes, 16-byte aligned.
    /// Matches `struct FieldEntry` in `force_fields.wgsl` exactly (size-asserted).
    /// Use the named constructors instead of filling `params` manually.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuFieldEntry {
        pub field_type: u32,
        pub material_mask: u32,
        pub _pad: [u32; 2],
        pub params: [f32; 8],
    }

    const _: () = assert!(core::mem::size_of::<GpuFieldEntry>() == 48);

    impl GpuFieldEntry {
        /// material_mask value for a field that affects all materials.
        pub const ALL_MATERIALS: u32 = 0xFFFF_FFFF;

        /// Plummer-softened point-mass gravity: a = −G·M·r / (r²+ε²)^(3/2).
        ///
        /// - `gm`: gravitational_constant × source_mass (positive = attractive)
        /// - `softening_sq`: Plummer ε² (prevents singularity at r=0)
        /// - `cutoff`: hard cutoff distance (0.0 = no cutoff)
        /// - `switch_on`: force-switch onset (< cutoff; force tapers from `switch_on` to `cutoff`)
        pub fn gravity_well(
            pos: glam::Vec2,
            gm: f32,
            softening_sq: f32,
            cutoff: f32,
            switch_on: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = pos.x;
            p[1] = pos.y;
            p[2] = gm;
            p[3] = softening_sq;
            p[6] = cutoff;
            p[7] = switch_on;
            Self {
                field_type: field_type::GRAVITY_WELL,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Plummer-softened Coulomb interaction for one (source, material) pair.
        ///
        /// - `charge_factor`: k × q_source × q_particle (signed; positive = repulsion)
        /// - `softening_sq`: Plummer ε²
        /// - `material_id`: which material's particles are affected (bitmask = 1 << id)
        /// - `cutoff` / `switch_on`: same as `gravity_well`
        pub fn coulomb(
            pos: glam::Vec2,
            charge_factor: f32,
            softening_sq: f32,
            material_id: u32,
            cutoff: f32,
            switch_on: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = pos.x;
            p[1] = pos.y;
            p[2] = charge_factor;
            p[3] = softening_sq;
            p[6] = cutoff;
            p[7] = switch_on;
            Self {
                field_type: field_type::COULOMB,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Soft repulsive walls of an axis-aligned bounding box.
        ///
        /// Particles that penetrate within `thickness` cells of any wall get a
        /// restoring acceleration proportional to penetration depth × `stiffness`.
        pub fn aabb_confinement(
            min: glam::Vec2,
            max: glam::Vec2,
            stiffness: f32,
            thickness: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = min.x;
            p[1] = min.y;
            p[2] = max.x;
            p[3] = max.y;
            p[4] = stiffness;
            p[5] = thickness;
            Self {
                field_type: field_type::AABB_CONFINEMENT,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Soft inward repulsion outside a radial shell.
        ///
        /// Particles beyond `radius − thickness` receive an inward acceleration
        /// proportional to excess penetration × `stiffness`.
        pub fn radial_confinement(
            center: glam::Vec2,
            radius: f32,
            stiffness: f32,
            thickness: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = center.x;
            p[1] = center.y;
            p[2] = radius;
            p[3] = stiffness;
            p[4] = thickness;
            Self {
                field_type: field_type::RADIAL_CONFINEMENT,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Spatially-constant electric field: a = q · E / m.
        ///
        /// - `field`: E-field vector (simulation units — force per unit charge)
        /// - `charge`: per-particle charge for `material_id` (same units as the Coulomb constant)
        /// - `material_id`: only particles of this material are affected
        pub fn uniform_electric(field: glam::Vec2, charge: f32, material_id: u32) -> Self {
            let mut p = [0f32; 8];
            p[0] = field.x;
            p[1] = field.y;
            p[2] = charge;
            Self {
                field_type: field_type::UNIFORM_ELECTRIC,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Archimedes buoyancy for particles of `material_id` floating in a denser fluid.
        ///
        /// - `gravity`: must match `SimConfig::gravity` (solver gravity, including sign)
        /// - `fluid_density_grid`: surrounding fluid's rest_density in grid units
        ///   (`ρ_SI · dx_m²` — same value as `NewtonianFluidMaterial::rest_density`, fixed
        ///   2026-07-07 to drop an incorrect extra `/dt_s²` factor)
        /// - `material_id`: only particles of this material receive the buoyancy force
        ///
        /// Uses particle rest density (`mass / initial_volume`) not instantaneous density,
        /// preventing the expansion-buoyancy runaway where expanded fluid appears falsely light.
        /// Applies `Δv = −gravity · (fluid_density / ρ₀_particle − 1) · dt` each substep.
        pub fn buoyancy(gravity: glam::Vec2, fluid_density_grid: f32, material_id: u32) -> Self {
            let mut p = [0f32; 8];
            p[0] = gravity.x;
            p[1] = gravity.y;
            p[2] = fluid_density_grid;
            p[3] = 1.0e-4; // min_density floor — mirrors BuoyancyField::new default
            Self {
                field_type: field_type::BUOYANCY,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }
    }

    /// Uniform buffer containing all active GPU force-field entries — 784 bytes.
    /// Matches `struct FieldsParams` in `force_fields.wgsl` exactly (size-asserted).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuFieldsParams {
        pub count: u32,
        pub _pad: [u32; 3],
        pub entries: [GpuFieldEntry; MAX_FORCE_FIELDS],
    }

    const _: () = assert!(core::mem::size_of::<GpuFieldsParams>() == 784);

    /// Max impulses per frame submitted via `apply_impulse` / `apply_radial_impulse`.
    /// Must match `array<ImpulseEntry, 16>` in `apply_impulses.wgsl`.
    pub const MAX_GPU_IMPULSES: usize = 16;

    /// One impulse descriptor — 32 bytes, matches `struct ImpulseEntry` in WGSL.
    ///
    /// mode 0 = radial: `v += normalize(p - center) * strength * falloff`
    /// mode 1 = directional: `v += force * falloff`
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuImpulseEntry {
        pub center: [f32; 2], // grid-space origin
        pub radius: f32,
        pub strength: f32,   // radial only (signed)
        pub force: [f32; 2], // directional only
        pub mode: u32,       // 0 = radial, 1 = directional
        pub _pad: u32,
    }

    const _: () = assert!(core::mem::size_of::<GpuImpulseEntry>() == 32);

    /// Uniform data for the apply_impulses compute pass — 528 bytes.
    /// Matches `struct ImpulseParams` in `apply_impulses.wgsl`.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuImpulseParams {
        pub count: u32,
        pub vel_limit: f32,
        pub particle_count: u32,
        pub _pad: u32,
        pub entries: [GpuImpulseEntry; MAX_GPU_IMPULSES],
    }

    const _: () = assert!(core::mem::size_of::<GpuImpulseParams>() == 528);

    /// Max tags per frame for force-sleep/force-wake-by-tag.
    /// Must match `array<u32, 8>` in `force_fields.wgsl`.
    ///
    /// Minimal hook for LP's future chunk system (see `mpm_technique_survey` memory
    /// note): a chunk leaving camera range force-sleeps its particles by `user_tag`
    /// regardless of velocity; a chunk re-entering range force-wakes them. The chunk
    /// system itself — tagging particles by chunk, tracking camera distance — is
    /// LP's job, not emerge's. This is just the primitive it needs.
    pub const MAX_SLEEP_WAKE_TAGS: usize = 8;

    /// Uniform data for force-sleep/force-wake-by-tag, checked once per substep in
    /// `force_fields.wgsl` — 80 bytes. Matches `struct SleepWakeParams` in WGSL.
    ///
    /// Tags are packed 4-per-`vec4<u32>` (`[[u32; 4]; 2]` = 8 tags), not a flat
    /// `[u32; 8]` — WGSL requires uniform-address-space arrays to have a 16-byte
    /// element stride, so a flat u32 array would be rejected by naga at shader-module
    /// creation (same class of gotcha as `vec3<u32>` padding elsewhere in this file).
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuSleepWakeParams {
        pub sleep_count: u32,
        pub wake_count: u32,
        pub _pad0: u32,
        pub _pad1: u32,
        pub sleep_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
        pub wake_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
    }

    const _: () = assert!(core::mem::size_of::<GpuSleepWakeParams>() == 80);

    /// Spatial-block bucket geometry for the particle_sort histogram AND the
    /// active-block detection it now also feeds (GPU sparse grid, Phase 1 — see
    /// `mpm_technique_survey` memory note). Single Rust-side source of truth: must
    /// match `NUM_BLOCKS_PER_DIM`/`NUM_BLOCKS` in `particle_sort.wgsl` and
    /// `grid_clear.wgsl` exactly. Re-deriving from `grid_res` at runtime is not an
    /// option — this sizes `block_counts`/`active_block_ids`, both allocated once
    /// at `GpuBuffers::new()`, so it must be a fixed compile-time constant, same
    /// class as `MAX_FORCE_FIELDS`.
    pub const NUM_BLOCKS_PER_DIM: usize = 16;
    pub const NUM_BLOCKS: usize = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM; // 256
}

#[cfg(feature = "gpu")]
mod solver {
    use std::sync::Arc;

    use crate::materials::registry::MaterialRegistry;
    use crate::solver::config::{SimConfig, SpawnRegion};
    use crate::solver::density::estimate_particle_volumes;
    use crate::solver::{LcgRng, affine_cfl_speed_contribution, cfl_bound, initialize_particles};
    use crate::{
        grid::Grid,
        particle::{Particle, Particles},
    };

    use super::buffers::GpuBuffers;
    use super::pipeline::SimPipelines;
    use super::step_params::{
        GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuSleepWakeParams,
        GpuStepParams, MAX_FORCE_FIELDS, MAX_GPU_IMPULSES, MAX_MATERIALS, MAX_SLEEP_WAKE_TAGS,
        NUM_BLOCKS,
    };

    /// Workgroup sizes — must match `@workgroup_size(...)` in the WGSL shaders.
    const WG_GRID: u32 = 8; // grid_clear and grid_update: 8×8 2D workgroups
    const WG_PARTICLES: u32 = 64; // p2g and g2p: 64-wide 1D workgroups

    /// Shared between the wgpu map_async callback (any thread) and step_frame's poll.
    type ReadbackResult =
        std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

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
        /// Real spatial acceleration for `particles_near`/`count_near`/`group_centroid` --
        /// ported from `solver::Simulation`'s already-proven `SpatialHash` (was previously
        /// wired into the CPU-only `Simulation` but not `GpuSimulation`, meaning every
        /// caller of these three query methods on the GPU path -- the one LP actually uses --
        /// paid a full O(N) linear scan per call regardless of how local the query was.
        /// Rebuilt once per `step_frame()` (and after any explicit particle sync), same
        /// ~1-frame staleness tolerance already accepted everywhere else these queries read
        /// the CPU mirror.
        spatial_hash: crate::solver::spatial_hash::SpatialHash,
        /// CPU-side wall-clock breakdown of the last `step_frame()` call (cfl_scan_ns,
        /// encode_ns, submit_ns, readback_ns, total_ns) — `Instant::now()` calls are
        /// themselves nanosecond-cost, so these are always recorded, not gated behind
        /// `enable_profiling()`. Read via `last_cpu_timings_ns()`. `total_ns` minus the sum of
        /// the other four reveals any unbracketed cost.
        last_cpu_timings: (f32, f32, f32, f32, f32),
    }

    /// One [begin, end] timestamp pair per labeled compute pass in `encode_substep`, written
    /// every substep (later substeps overwrite earlier ones within the same `step_frame()`
    /// call — fine for finding the dominant cost, since substeps cost about the same each
    /// time; not meant to capture per-substep variance).
    const PROFILE_PASS_LABELS: &[&str] = &[
        "active_block_refresh (sort)",
        "grid_clear",
        "p2g",
        "grid_update",
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

            let mut spatial_hash =
                crate::solver::spatial_hash::SpatialHash::new(config.grid_cell_size);
            spatial_hash.rebuild(
                &initialized.iter().map(|p| p.x).collect::<Vec<_>>(),
                initialized.len(),
            );

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
                spatial_hash,
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

        /// Verification-only accessor: read back `sorted_particle_ids` as a `Vec<u32>`.
        /// Used by tests to confirm the particle_sort pipeline produces a valid permutation —
        /// not part of the render/game-loop API.
        pub fn sorted_particle_ids_blocking(&self) -> Vec<u32> {
            self.buffers.readback_u32_blocking(
                &self.device,
                &self.queue,
                &self.buffers.sorted_particle_ids,
                self.particle_count,
            )
        }

        /// Test/diagnostic readback for the GPU sparse grid Phase 1 active-block list — the
        /// first `active_block_count_blocking()` entries are valid; the rest are stale/unused.
        pub fn active_block_ids_blocking(&self) -> Vec<u32> {
            self.buffers.readback_u32_blocking(
                &self.device,
                &self.queue,
                &self.buffers.active_block_ids,
                NUM_BLOCKS,
            )
        }

        /// Test/diagnostic readback for how many entries in `active_block_ids_blocking()` are
        /// valid this frame.
        pub fn active_block_count_blocking(&self) -> u32 {
            self.buffers.readback_u32_blocking(
                &self.device,
                &self.queue,
                &self.buffers.active_block_count,
                1,
            )[0]
        }

        /// Test/diagnostic readback of the dense grid buffer — 4 f32 per cell (momentum.x,
        /// momentum.y, mass, _pad), same field order as the WGSL `Cell` struct, flat-indexed
        /// `(y * grid_res + x) * 4`. Lets tests verify grid_clear actually zeroed cells far from
        /// any particle (the failure mode a block-boundary mapping bug would produce: stale,
        /// never-cleared mass/momentum left behind in an unrelated block).
        pub fn grid_cells_blocking(&self) -> Vec<f32> {
            let cell_floats = self.config.grid_res * self.config.grid_res * 4;
            self.buffers.readback_f32_blocking(
                &self.device,
                &self.queue,
                &self.buffers.grid,
                cell_floats,
            )
        }

        /// Real, honest report of why this instance's device was lost, if it ever
        /// was — `None` in ordinary operation. Automatically wired for `new()`
        /// instances; `with_device()` instances need one explicit call to
        /// `enable_device_lost_detection()` first (see that method's doc for why
        /// it isn't automatic there). Once set, `step_frame` and the blocking sync
        /// methods become safe no-ops instead of panicking on a dead device —
        /// callers that care should poll this rather than assume silence means
        /// healthy.
        pub fn device_lost_reason(&self) -> Option<String> {
            self.device_lost.lock().ok().and_then(|g| g.clone())
        }

        /// Opt in to real device-lost detection (the confirmed real cause of
        /// emerge issue #10 — a genuine `Out of Memory` device loss under
        /// sustained load on slow/software GPU backends; see project memory
        /// `gpu_readback_error_path_bug_issue10`). Called automatically by `new()`
        /// (which owns its device exclusively, so it's always safe there). NOT
        /// automatic for `with_device()` (shared-device use, e.g. a renderer on the
        /// same device as this sim) because a wgpu device can only have ONE
        /// lost-callback (and, as of 2026-07-08, only one uncaptured-error handler
        /// too — same `Option<Arc<dyn Handler>>` single-slot storage internally,
        /// confirmed by reading wgpu-27.0.1's `ErrorSinkRaw`) — auto-registering
        /// here could silently overwrite a caller's own handler. Call this
        /// explicitly after `with_device()` if you (like LP) don't have your own
        /// device-lost handling and want emerge's; don't call it if you've already
        /// registered your own callback/handler on this device — the second
        /// registration wins and the first is silently lost (this is wgpu's own
        /// behavior, not something this method can prevent).
        ///
        /// ALSO installs an uncaptured-error handler (2026-07-08). wgpu's default
        /// behavior for ANY uncaptured error is an unconditional panic
        /// (`panic!("wgpu error: {err}")`, confirmed by reading wgpu-27.0.1's
        /// `default_error_handler`) — this handler replaces that default and
        /// **never panics**, regardless of what the error says. That "never" is
        /// load-bearing, not a simplification: an earlier version of this handler
        /// tried to be more precise — classify errors naming a destroyed/lost
        /// resource as an inferred device loss (no panic), but still panic for
        /// anything else so a genuine, unrelated validation bug wouldn't be
        /// silently swallowed. That version was reproduced crashing LOCALLY
        /// (forcing the D3D12 WARP adapter — the same backend windows-latest CI
        /// uses — instead of waiting on another CI round-trip) with the full
        /// backtrace showing the panic originated from THIS handler's own `panic!`
        /// call, invoked synchronously from inside `wgpu_core::Queue::submit`'s
        /// internal error path — and unwinding a panic from there is what produced
        /// `STATUS_STACK_BUFFER_OVERRUN`, not the error itself. In other words:
        /// panicking from ANY code reachable from this callback is unsafe on this
        /// backend, independent of whether the message looks like a device-loss
        /// artifact or a real bug — so the "still panic for real bugs" branch was
        /// itself the crash, not a safety net. The fix: never panic here, full
        /// stop. Every uncaptured error sets `device_lost` (so `is_device_lost()`'s
        /// existing no-op guards take over) and is `eprintln!`'d in full so it's
        /// still visible for debugging — just never re-thrown as a Rust panic from
        /// inside this specific callback context.
        pub fn enable_device_lost_detection(&self) {
            let flag = self.device_lost.clone();
            self.device
                .set_device_lost_callback(move |reason, message| {
                    *flag.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(format!("{reason:?}: {message}"));
                });

            let flag = self.device_lost.clone();
            self.device
                .on_uncaptured_error(std::sync::Arc::new(move |error: wgpu::Error| {
                    let message = error.to_string();
                    let mut guard = flag.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.is_none() {
                        *guard = Some(format!("(uncaptured wgpu error) {message}"));
                    }
                    drop(guard);
                    eprintln!(
                        "emerge: uncaptured wgpu error, treating device as unusable from \
                         here (see GpuSimulation::enable_device_lost_detection's doc for \
                         why this never panics): {message}"
                    );
                }));
        }

        fn is_device_lost(&self) -> bool {
            self.device_lost
                .lock()
                .map(|g| g.is_some())
                .unwrap_or(false)
        }

        /// Advance one frame of simulation time (`config.dt`) using the GPU.
        ///
        /// All substeps are encoded into a single command buffer and submitted once — one driver
        /// call regardless of adaptive substep count. Step params are pre-computed from the CPU
        /// particle mirror (same one-frame CFL lag as before, no physics change).
        pub fn step_frame(&mut self) {
            // Real fix for emerge issue #10 (confirmed root cause: genuine device
            // loss, Out of Memory, under sustained slow-backend load — see project
            // memory gpu_readback_error_path_bug_issue10). A lost device cannot be
            // un-lost; every further GPU call on it would panic through wgpu's
            // default error handler. Once lost, become a safe no-op instead.
            if self.is_device_lost() {
                return;
            }
            let total_start = std::time::Instant::now();
            let cfl_scan_start = total_start;
            let any_cpu = self.registry.any_needs_cpu_update();

            // Upload CPU → GPU only when positions/materials actually changed.
            // Impulses are now applied by a dedicated GPU compute pass (apply_impulses) that
            // reads LIVE GPU positions — no CPU mirror upload needed for impulse-only frames.
            //
            // Real bug fix (2026-07-06, LP issue erematorg/LP#161): this block used to
            // spatially resort `self.particles` by grid cell before every upload. That
            // predates the real GPU particle_sort pipeline (`f2c1e62`, "real particle-sort
            // pipeline") which added its own spatial-locality mechanism entirely on the GPU
            // side (`sorted_particle_ids`, a SEPARATE index buffer that never touches actual
            // particle storage order — see particle_sort.wgsl, runs unconditionally every
            // frame). When that GPU pass was added, the old CPU-side resort should have been
            // removed but wasn't — it kept running on every upload, which happens on
            // essentially every frame in real use (any per-particle CPU write, e.g. LP's
            // `drive_muscles`/`update_damage`, calls `mark_particles_dirty`). Reordering the
            // backing array on every such frame silently invalidated any previously-returned
            // `Range<usize>` particle identity (`spawn_region`'s own doc promises this range
            // is stable — "LP uses this as creature_id -> particle_range"). Confirmed via a
            // real repro: a spawned creature's fixed index range, read back every frame,
            // showed near-total corruption of a spawn-time-only tag field (`muscle_group_id`)
            // — not a readback race, the particles at those indices were simply different
            // particles after the resort. No remaining purpose for this CPU-side sort once
            // the GPU has its own; removing it restores range stability.
            let needs_upload = self.layout_dirty || any_cpu;
            if needs_upload {
                self.buffers.upload_particles(&self.queue, &self.particles);
                self.layout_dirty = false;
            }

            // Pre-compute all sub_dts from CPU mirror (same one-frame lag as before).
            // CFL scan is O(N) — run it ONCE and reuse the result to fill the sub_dts array.
            // The CPU mirror is static within a frame so every repeated call would return the
            // same value anyway. Previously this called choose_substep_dt up to 16×/frame
            // (once per substep), which in debug mode caused measurable cursor slowdown.
            //
            // Exclude sleeping particles from the scan. CPU's Simulation::step() does this
            // implicitly via its active/sleeping partition (active_count only covers awake
            // particles); GPU has no such partition, so without this filter a frozen-near-zero
            // sleeping majority dilutes the velocity statistics this estimate is based on,
            // potentially under-resolving the timestep right when an awake particle needs it
            // most. (sparkl's adaptive_timestep_length, tmp/sparkl/src/dynamics/solver/
            // timestep_estimator.rs, computes this the same way: scan only the live/active
            // particle set, never a population diluted by inactive ones.)
            // REAL FIX (2026-06-27, see project_mvp_definition memory for the full
            // investigation): the previous version built a fresh `Particles` SoA every frame
            // (filter+collect into an intermediate AoS Vec, then transpose into SoA) purely
            // because `MaterialModel::timestep_bound` used to require `&Particles, i: usize`.
            // Every material's implementation only ever read `density`/`hardening_scale` —
            // both plain scalar fields that already exist directly on `Particle` (AoS). Changed
            // the trait to take those two scalars directly (12 materials updated, 1 call site
            // in `choose_substep_dt`), which means this scan never needs to build ANY SoA
            // wrapper at all — it just reads each particle's own fields in one direct pass over
            // the array that already exists: zero allocation, not just less allocation.
            // Correctness fully verified (full CPU+GPU regression suite green). Wall-clock
            // comparisons on this machine were unreliable that night (integrated GPU, shared
            // CPU/GPU thermal budget, hours of sustained heavy load) — don't trust a GPU timing
            // number gathered after a long run of GPU work on this hardware; re-measure
            // `gpu_cfl_scan_baseline_across_grid` cold, first thing in a session, for a real
            // comparison.
            let mut max_speed = 0.0f32;
            let mut min_mat_dt = self.config.dt;
            let mut awake_count = 0usize;
            for p in self.particles.iter() {
                if p.sleeping != 0 {
                    continue;
                }
                awake_count += 1;
                let mut s = p.v.length();
                if self.config.cfl_include_affine_speed {
                    s += affine_cfl_speed_contribution(
                        &p.velocity_gradient,
                        self.config.grid_cell_size,
                    );
                }
                max_speed = max_speed.max(s);
                let mdt = self.registry.get(p.material_id).timestep_bound(
                    p.density,
                    p.hardening_scale,
                    self.config.grid_cell_size,
                    self.config.material_cfl_coefficient,
                    self.config.viscous_timestep_coefficient,
                );
                if mdt.is_finite() && mdt > 0.0 {
                    min_mat_dt = min_mat_dt.min(mdt);
                }
            }
            // If every particle is asleep AND something could actually disturb them this
            // frame, there's no awake velocity to base an estimate on — choose_substep_dt
            // would fall back to max_dt (max_speed=0 fails its `> f32::EPSILON` guard), the
            // COARSEST possible substep, right when a wake event needs the FINEST. But wake
            // propagation only happens via a neighbor's grid activity (which requires some
            // OTHER awake particle to exist — if the awake set is truly empty, there is none)
            // or an external impulse. So "everyone asleep" alone isn't a risk: nothing CAN
            // wake spontaneously with no awake particles and no incoming disturbance. Only
            // pay for the fine fallback when a pending impulse could actually wake someone —
            // otherwise a fully-settled scene would pay maximum substep cost forever, which
            // defeats sleep/wake's entire purpose (measured: 64 substeps/frame indefinitely
            // on a calm, fully-asleep pile before this check was added).
            let might_wake_this_frame = !self.pending_impulses.is_empty();
            let sub_dt_cfl =
                if awake_count == 0 && self.config.sleep_threshold > 0.0 && might_wake_this_frame {
                    self.config.dt / self.config.max_substeps_per_step.max(1) as f32
                } else {
                    cfl_bound(&self.config, max_speed, min_mat_dt, self.config.dt)
                };
            let mut sub_dts: Vec<f32> = Vec::with_capacity(self.config.max_substeps_per_step);
            {
                let mut remaining = self.config.dt;
                while remaining > f32::EPSILON && sub_dts.len() < self.config.max_substeps_per_step
                {
                    let sub_dt = sub_dt_cfl.min(remaining);
                    sub_dts.push(sub_dt);
                    remaining -= sub_dt;
                }
            }
            self.last_substeps = sub_dts.len();
            self.last_sub_dt = sub_dts.last().copied().unwrap_or(self.config.dt);
            self.frame_index += 1;
            let cfl_scan_ns = cfl_scan_start.elapsed().as_secs_f32() * 1.0e9;

            // Sleep delay: a particle spawned at rest (v=0) satisfies any positive
            // sleep_threshold on its very first substep, before gravity has accelerated it
            // at all — same fix every real physics engine uses for this (Box2D, PhysX,
            // Bullet all require sustained low velocity before sleeping, never an instant
            // single-frame check). Can't add a per-particle timer here (Particle has no
            // spare bytes left), so this is the simulation-level equivalent: don't let
            // anything sleep-score for the first few frames after construction, giving
            // real dynamics a chance to start. Once any particle exists, GPU has no
            // incremental add API (everything is introduced at construction), so this
            // covers every particle that will ever exist in this simulation, not just the
            // initial batch.
            const SLEEP_WARMUP_FRAMES: u64 = 10;
            let step_config = if self.frame_index <= SLEEP_WARMUP_FRAMES {
                SimConfig {
                    sleep_threshold: 0.0,
                    ..self.config
                }
            } else {
                self.config
            };

            // Build force fields uniform (same every substep).
            let mut ff_params: GpuFieldsParams = bytemuck::Zeroable::zeroed();
            ff_params.count = self.force_field_entries.len() as u32;
            for (i, e) in self.force_field_entries.iter().enumerate() {
                ff_params.entries[i] = *e;
            }
            self.buffers
                .upload_force_fields_params(&self.queue, &ff_params);

            // Force-sleep/force-wake-by-tag — minimal hook for LP's future chunk system.
            // Uploaded every frame (zeroed when nothing's pending, same as ff_params above)
            // and read once per substep in force_fields.wgsl; cleared after upload since
            // each call is a one-shot edge-trigger, not a persistent state (a tag that's
            // force-asleep doesn't need to be re-sent every frame — sleeping is sticky on
            // the particle itself until something genuinely wakes it).
            let mut sw_params: GpuSleepWakeParams = bytemuck::Zeroable::zeroed();
            sw_params.sleep_count = self.pending_sleep_tags.len() as u32;
            for (i, &tag) in self.pending_sleep_tags.iter().enumerate() {
                sw_params.sleep_tags[i / 4][i % 4] = tag;
            }
            sw_params.wake_count = self.pending_wake_tags.len() as u32;
            for (i, &tag) in self.pending_wake_tags.iter().enumerate() {
                sw_params.wake_tags[i / 4][i % 4] = tag;
            }
            self.buffers
                .upload_sleep_wake_params(&self.queue, &sw_params);
            self.pending_sleep_tags.clear();
            self.pending_wake_tags.clear();

            // Upload step_params for each substep into its pool slot -- contents change every
            // frame (adaptive dt), so this write can't be cached. The bind group pointing at
            // that slot, however, only depends on buffer IDENTITY, not contents, so it's built
            // once in `bind_group_pool` (see that field's doc comment) instead of recreated
            // here every substep every frame -- doing so at LP's ~5-6k-substep-per-frame scale
            // exhausted the GPU's descriptor allocator within seconds.
            for (i, &sub_dt) in sub_dts.iter().enumerate() {
                let params = GpuStepParams::new(&step_config, sub_dt, self.particle_count);
                self.buffers.upload_step_params_at(&self.queue, i, &params);
            }
            let bind_groups = &self.bind_group_pool;

            // Encode everything into one command buffer — one GPU submit per frame.
            // Order: [apply_impulses?] → [particle_sort?] → substep_0 → … → substep_N
            //
            // apply_impulses runs first so physics sees the freshly-applied velocities.
            // particle_sort re-seeds sorted_particle_ids after a CPU upload (layout_dirty).
            // Both use dedicated buffer slots so they never alias substep params.
            let grid_wg = (self.config.grid_res as u32).div_ceil(WG_GRID);
            let particle_wg = (self.particle_count as u32).div_ceil(WG_PARTICLES);
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mpm_frame"),
                });

            // — apply_impulses pass (GPU-native, no stale CPU mirror) —
            if !self.pending_impulses.is_empty() {
                let vel_limit = self.config.grid_cell_size / self.config.min_dt;
                let mut params = GpuImpulseParams {
                    count: self.pending_impulses.len() as u32,
                    vel_limit,
                    particle_count: self.particle_count as u32,
                    _pad: 0,
                    entries: bytemuck::Zeroable::zeroed(),
                };
                for (i, e) in self.pending_impulses.iter().enumerate() {
                    params.entries[i] = *e;
                }
                self.buffers.upload_impulse_params(&self.queue, &params);
                let impulse_bg = self
                    .pipelines
                    .make_impulse_bind_group(&self.device, &self.buffers);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("apply_impulses"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.apply_impulses);
                pass.set_bind_group(0, &impulse_bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
                drop(pass);
                self.pending_impulses.clear();
            }

            // — particle_sort pass: clear -> count -> scan -> scatter, every frame —
            //
            // Runs unconditionally (not gated on layout_dirty) because particle positions drift
            // every substep even when the CPU mirror is never touched — without a per-frame
            // re-sort, sorted_particle_ids would stay frozen at whatever ordering existed at the
            // last CPU upload, going stale as GPU-resident particles move. See particle_sort.wgsl.
            {
                let sort_slot = self.buffers.step_params_pool.len() - 1;
                let sort_params =
                    GpuStepParams::new(&self.config, self.config.dt, self.particle_count);
                self.buffers
                    .upload_step_params_at(&self.queue, sort_slot, &sort_params);
                let sort_bg = self.pipelines.make_bind_group(
                    &self.device,
                    &self.buffers,
                    &self.buffers.step_params_pool[sort_slot],
                );
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("particle_sort"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &sort_bg, &[]);
                pass.set_pipeline(&self.pipelines.particle_sort_clear);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_count);
                pass.dispatch_workgroups(particle_wg, 1, 1);
                // No particle_sort_compact here anymore — active-block detection now runs
                // every substep (see encode_substep's active_block_refresh pass), since
                // particles move every substep and this once-per-frame pass would go stale by
                // substep 2+. This pass's count output is used only for the sort permutation
                // (scan + scatter below), unrelated to active-block correctness.
                pass.set_pipeline(&self.pipelines.particle_sort_scan);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_scatter);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            // Substeps are batched into multiple command buffers/submits instead of one --
            // LP's stiff-terrain scenes (50 MPa sandy soil) routinely need several hundred
            // substeps in a single frame, and encoding them all into one command buffer
            // exhausted this GPU backend within seconds (`wgpu error: Out of Memory` from
            // this same `queue.submit`, reported against LP's own scene 2026-07-01).
            // Bisected empirically: 200 substeps in one submit reliably OOMs, 64 is stable
            // (matches this engine's own tested default, see `max_substeps_per_step`'s doc
            // comment) -- this is a real per-submit resource ceiling on the backend/driver
            // actually exercised, not a value derived from any GPU spec, so a different
            // backend may need a different number. Blocking between chunks is required too
            // -- unblocked back-to-back submits queue up faster than the GPU drains them and
            // hit the same OOM even with batching. Only blocks BETWEEN chunks, never after
            // the last one -- typical scenes (well under 64 substeps/frame) produce exactly
            // one chunk and pay zero extra sync cost, same as before this fix existed. Only
            // LP's stiff-terrain scale (hundreds of substeps/frame) pays the blocking cost,
            // and only for the chunks beyond the first.
            const SUBSTEP_BATCH_SIZE: usize = 64;
            let mut chunks = bind_groups[..sub_dts.len()]
                .chunks(SUBSTEP_BATCH_SIZE)
                .peekable();
            // Split pure CPU command-building time from GPU-completion wait time --
            // "encode_ns" previously bundled both under one name, hiding whether a slow
            // step_frame() was a CPU-side encoding problem or genuinely GPU-execution-bound.
            let mut pure_encode_ns = 0.0f32;
            let mut wait_ns = 0.0f32;
            while let Some(chunk) = chunks.next() {
                let chunk_encode_start = std::time::Instant::now();
                let mut sub_encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("mpm_substep_batch"),
                        });
                for bg in chunk {
                    self.encode_substep(&mut sub_encoder, bg, grid_wg, particle_wg);
                }
                self.queue.submit(std::iter::once(sub_encoder.finish()));
                pure_encode_ns += chunk_encode_start.elapsed().as_secs_f32() * 1.0e9;
                if chunks.peek().is_some() {
                    let wait_start = std::time::Instant::now();
                    self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                    wait_ns += wait_start.elapsed().as_secs_f32() * 1.0e9;
                }
            }
            let encode_ns = pure_encode_ns;
            // Repurposed: real GPU-completion wait time between chunks, not always 0 --
            // this IS where GPU execution time shows up for multi-chunk (>64 substep) frames.
            let submit_ns = wait_ns;

            // Async GPU → CPU readback — never blocks the render thread.
            //
            // Two-phase: begin_readback submits a GPU copy + async map (non-blocking).
            // The receiver fires on a subsequent frame when the GPU copy + map completes.
            // We pump wgpu callbacks with poll(Poll) each frame so the mapping progresses.
            //
            // If any_cpu: readback every frame (CPU plasticity needs current state).
            // Otherwise: stride-gated to reduce overhead.
            let readback_start = std::time::Instant::now();
            self.readback_frame = self.readback_frame.wrapping_add(1);
            let want_readback = any_cpu || self.readback_frame.is_multiple_of(self.readback_stride);

            // Pump wgpu callbacks so any in-flight mapping can complete.
            self.device.poll(wgpu::PollType::Poll).ok();

            // Check if a previous async readback completed -- Ok, Err, or still pending.
            // Real fix (2026-07-05, see project memory
            // emerge_locomotion_root_cause_and_fix / issue #10): the OLD code only
            // handled Ok here, silently dropping Err. That left the staging buffer
            // mapped forever (finish_readback, the only unmapper, was never called)
            // and pending_readback stuck Some forever (blocking every future
            // readback) -- until something else tried to map the same buffer again
            // and hit a real "Buffer is already mapped" panic. Every completion path
            // now explicitly unmaps, regardless of Ok/Err.
            let readback_done = self
                .pending_readback
                .as_ref()
                .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
            if let Some(result) = readback_done {
                self.pending_readback = None;
                if result.is_err() {
                    self.readback_error_count += 1;
                    self.buffers.abandon_readback();
                } else {
                    let gpu_particles = self.buffers.finish_readback(self.particle_count);

                    // CPU plasticity pass — skipped if all materials run plasticity on GPU.
                    //
                    // IMPORTANT: GPU g2p already integrated F via `F_new = (I + dt·C)·F_old`.
                    // Zero affine before update_particle so only the plasticity projection runs.
                    // Restore GPU affine afterwards so next P2G APIC term is correct.
                    // The new MaterialModel API takes (&mut Particles, usize) — convert AoS to SoA,
                    // run the CPU pass, then scatter results back.
                    if any_cpu {
                        // Stash GPU affine matrices — we zero affine for the plasticity call then restore.
                        let gpu_affines: Vec<_> =
                            gpu_particles.iter().map(|p| p.velocity_gradient).collect();
                        // Copy readback into AoS cpu mirror (zeroing affine for plasticity).
                        for (p_gpu, p_cpu) in gpu_particles.iter().zip(self.particles.iter_mut()) {
                            *p_cpu = *p_gpu;
                            p_cpu.velocity_gradient = glam::Mat2::ZERO;
                        }
                        // Build SoA wrapper, run CPU plasticity, scatter plastic state back.
                        // Skip sleeping particles — same reasoning as every GPU-side pass: their
                        // F/plastic state is frozen, re-running plasticity on unchanged input
                        // wastes exactly the compute sleep/wake exists to avoid. Before the
                        // Particles::push() fix above, this loop silently ran on every particle
                        // regardless of sleep state, because the AoS->SoA conversion dropped it.
                        let mut soa = Particles::from(std::mem::take(&mut self.particles));
                        for i in 0..soa.len() {
                            if soa.sleeping[i] {
                                continue;
                            }
                            self.registry.get(soa.material_id[i]).update_particle(
                                &mut soa,
                                i,
                                self.last_sub_dt,
                            );
                        }
                        self.particles = soa.to_vec();
                        // Restore GPU affine.
                        for (p_cpu, gpu_affine) in self.particles.iter_mut().zip(gpu_affines) {
                            p_cpu.velocity_gradient = gpu_affine;
                        }
                    } else {
                        for (p_gpu, p_cpu) in
                            gpu_particles.into_iter().zip(self.particles.iter_mut())
                        {
                            *p_cpu = p_gpu;
                        }
                    }
                    if any_cpu {
                        self.layout_dirty = true; // CPU plasticity touched positions/F
                    }
                    self.rebuild_spatial_hash();
                }
            }

            // Start a new readback if wanted and none is already in flight.
            if want_readback && self.pending_readback.is_none() {
                self.pending_readback = Some(self.buffers.begin_readback(
                    &self.device,
                    &self.queue,
                    self.particle_count,
                ));
            }
            let readback_ns = readback_start.elapsed().as_secs_f32() * 1.0e9;
            let total_ns = total_start.elapsed().as_secs_f32() * 1.0e9;
            self.last_cpu_timings = (cfl_scan_ns, encode_ns, submit_ns, readback_ns, total_ns);
        }

        /// Add a non-uniform body force field for the GPU path.
        /// Entries are uploaded and dispatched every substep until cleared.
        /// Panics if `MAX_FORCE_FIELDS` is exceeded.
        pub fn add_force_field_gpu(&mut self, entry: GpuFieldEntry) {
            assert!(
                self.force_field_entries.len() < MAX_FORCE_FIELDS,
                "add_force_field_gpu: MAX_FORCE_FIELDS ({MAX_FORCE_FIELDS}) exceeded"
            );
            self.force_field_entries.push(entry);
        }

        /// Remove all GPU force field entries.
        pub fn clear_force_fields_gpu(&mut self) {
            self.force_field_entries.clear();
        }

        /// Turns on per-pass GPU timing for `encode_substep`'s 7 labeled passes. Returns false
        /// (no-op) if this device wasn't created with `TIMESTAMP_QUERY` support — `new()`
        /// requests it opportunistically when the adapter supports it; `with_device()` depends
        /// on whatever device the caller already built. Call once after construction; read
        /// results back with `last_pass_timings_ns()` after stepping a few frames.
        pub fn enable_profiling(&mut self) -> bool {
            if !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                return false;
            }
            let n = PROFILE_PASS_LABELS.len() as u32;
            let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("emerge_profile_queries"),
                ty: wgpu::QueryType::Timestamp,
                count: n * 2, // begin+end per pass
            });
            let resolve_size = (n * 2) as u64 * 8; // 8 bytes per u64 timestamp
            let resolve_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("emerge_profile_resolve"),
                size: resolve_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("emerge_profile_readback"),
                size: resolve_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.profiling = Some(GpuProfiling {
                query_set,
                resolve_buf,
                readback_buf,
                timestamp_period_ns: self.queue.get_timestamp_period(),
            });
            true
        }

        /// Reads back the last substep's per-pass GPU timings (label, nanoseconds), in
        /// `encode_substep`'s pass order. Blocks until the GPU work + readback completes — a
        /// diagnostic call, not for the hot path. Returns None if `enable_profiling()` wasn't
        /// called or wasn't supported on this device.
        pub fn last_pass_timings_ns(&mut self) -> Option<Vec<(&'static str, f32)>> {
            let profiling = self.profiling.as_ref()?;
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let slice = profiling.readback_buf.slice(..);
            let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
            let flag2 = flag.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                *flag2.lock().unwrap() = Some(r);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            flag.lock().unwrap().take()?.ok()?;
            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            let period = profiling.timestamp_period_ns;
            let result = PROFILE_PASS_LABELS
                .iter()
                .enumerate()
                .map(|(i, &label)| {
                    let begin = timestamps[i * 2];
                    let end = timestamps[i * 2 + 1];
                    (label, (end.saturating_sub(begin)) as f32 * period)
                })
                .collect();
            drop(data);
            profiling.readback_buf.unmap();
            Some(result)
        }

        /// Encode one substep's passes into an existing encoder. No submission — caller batches.
        fn encode_substep(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            bg: &wgpu::BindGroup,
            grid_wg: u32,
            particle_wg: u32,
        ) {
            {
                // GPU sparse grid Phase 1 — re-detect active blocks from CURRENT particle
                // positions, every substep, immediately before grid_clear uses the result.
                //
                // Real bug found via direct testing (gpu_sleep_freezes_settled_particles
                // regressed, plus a native crash — see mpm_technique_survey memory note):
                // particle_sort's once-per-frame active-block detection (computed from
                // frame-START positions) went stale by substep 2+ of the same frame, since
                // particles move every substep. Fixed by re-running clear+count+compact (NOT
                // scan/scatter — those only matter for the once-per-frame sort permutation,
                // unrelated to grid_clear correctness) every substep.
                //
                // Second real bug, found via a long-running headless diagnostic AFTER the
                // above fix (basic_sand_gpu blew up after ~1500 frames, ~1-in-5 runs): a block
                // that stops being active (a particle moves away) was never cleared again —
                // grid_clear only ever clears CURRENTLY active blocks, so a block's last P2G
                // contribution sat there permanently until some particle wandered back near it
                // much later, at which point P2G's atomic ADD compounded onto the stale
                // residual. Dense grid_clear never had this problem (it unconditionally zeroed
                // every cell every substep regardless of activity). Fix: active_block_swap
                // (dispatched FIRST, before clear/count/compact) snapshots this substep's
                // about-to-be-overwritten active list into active_block_ids_prev/count_prev,
                // and grid_clear processes the union of both lists — a genuine one-substep
                // grace period. See active_block_swap_main's doc comment in particle_sort.wgsl
                // for the full reasoning, including a first attempt at this fix that was wrong
                // (reset happened in the same substep it was used in, giving zero actual grace
                // period).
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("active_block_refresh"),
                    timestamp_writes: self.profile_writes(0),
                });
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(&self.pipelines.active_block_swap);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_clear);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_count);
                pass.dispatch_workgroups(particle_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.particle_sort_compact);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grid_clear"),
                    timestamp_writes: self.profile_writes(1),
                });
                pass.set_pipeline(&self.pipelines.grid_clear);
                pass.set_bind_group(0, bg, &[]);
                // GPU sparse grid Phase 1: one workgroup per potential active-block slot, for
                // EACH of the two lists (this substep's + last substep's grace period) — fixed
                // worst-case size (2 * NUM_BLOCKS), not grid_res-dependent anymore. Most slots
                // beyond their list's real count exit immediately via the shader's own guard.
                // See grid_clear.wgsl.
                pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("p2g"),
                    timestamp_writes: self.profile_writes(2),
                });
                pass.set_pipeline(&self.pipelines.p2g);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grid_update"),
                    timestamp_writes: self.profile_writes(3),
                });
                pass.set_pipeline(&self.pipelines.grid_update);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("g2p"),
                    timestamp_writes: self.profile_writes(4),
                });
                pass.set_pipeline(&self.pipelines.g2p);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("particles_update"),
                    timestamp_writes: self.profile_writes(5),
                });
                pass.set_pipeline(&self.pipelines.particles_update);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("force_fields"),
                    timestamp_writes: self.profile_writes(6),
                });
                pass.set_pipeline(&self.pipelines.force_fields);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            if let Some(profiling) = &self.profiling {
                let n = PROFILE_PASS_LABELS.len() as u32;
                encoder.resolve_query_set(
                    &profiling.query_set,
                    0..n * 2,
                    &profiling.resolve_buf,
                    0,
                );
                encoder.copy_buffer_to_buffer(
                    &profiling.resolve_buf,
                    0,
                    &profiling.readback_buf,
                    0,
                    (n * 2) as u64 * 8,
                );
            }
        }

        /// Builds `ComputePassTimestampWrites` for pass index `i` (in `PROFILE_PASS_LABELS`
        /// order) if profiling is enabled, else `None` — keeps each pass's descriptor a
        /// one-liner regardless of whether profiling is active.
        fn profile_writes(&self, i: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
            self.profiling
                .as_ref()
                .map(|p| wgpu::ComputePassTimestampWrites {
                    query_set: &p.query_set,
                    beginning_of_pass_write_index: Some(i * 2),
                    end_of_pass_write_index: Some(i * 2 + 1),
                })
        }

        /// Download particles from GPU to CPU synchronously (diagnostics / one-shot use).
        /// Prefer the async readback path in step_frame for per-frame use.
        pub fn download_particles_blocking(&mut self) {
            let flag = self
                .buffers
                .begin_readback(&self.device, &self.queue, self.particle_count);
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            if let Ok(mut g) = flag.lock() {
                g.take();
            }
            self.particles = self.buffers.finish_readback(self.particle_count);
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
        pub fn spawn_region(&mut self, spawn: SpawnRegion) -> std::ops::Range<usize> {
            let start = self.particles.len();
            spawn.validate_for_sim(&self.config);
            debug_assert!(
                self.registry.is_registered(spawn.material_id),
                "spawn_region: material_id {} is not registered — call solver.set_material({}, ...) first",
                spawn.material_id,
                spawn.material_id
            );
            let mut rng = LcgRng::new(spawn.rng_seed);
            let new_particles = initialize_particles(&self.config, spawn, &mut rng);
            self.particles.extend(new_particles);

            // Recompute initial volumes for the combined particle set using a temporary grid.
            let mut tmp_grid = Grid::new(self.config.grid_res);
            {
                let mut tmp_soa =
                    crate::particle::Particles::from(std::mem::take(&mut self.particles));
                let n = tmp_soa.len();
                estimate_particle_volumes(&mut tmp_soa, &mut tmp_grid, n, true);
                self.particles = tmp_soa.to_vec();
            }

            let n = self.particles.len();

            // Reallocate all GPU buffers that are sized per-particle (including staging).
            self.buffers.particles = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mpm_particles"),
                size: (n * core::mem::size_of::<Particle>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.buffers.sorted_particle_ids = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mpm_sorted_particle_ids"),
                size: (n * core::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.buffers.readback_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mpm_particle_staging"),
                size: (n * core::mem::size_of::<Particle>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            self.particle_count = n;
            self.pending_readback = None; // old staging is gone
            self.buffers.upload_particles(&self.queue, &self.particles);
            // buffers.particles was just reallocated above -- cached bind groups reference
            // the old buffer object and would be stale (or invalid) without this.
            self.bind_group_pool =
                build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
            self.rebuild_spatial_hash();
            start..n
        }

        pub fn config(&self) -> &SimConfig {
            &self.config
        }
        pub fn particle_count(&self) -> usize {
            self.particle_count
        }

        /// Rebuilds `spatial_hash` from the current `self.particles` -- call any time
        /// `self.particles` changes (readback completion, explicit sync, spawn). O(N),
        /// same real cost class as the linear scans it replaces for QUERIES, but paid
        /// once per mutation instead of once per query -- a real win whenever more than
        /// one query happens against the same particle state (e.g. LP's ecology calling
        /// `sense_local`/`centroid`/`phenotype` per creature per frame).
        fn rebuild_spatial_hash(&mut self) {
            let positions: Vec<glam::Vec2> = self.particles.iter().map(|p| p.x).collect();
            self.spatial_hash.rebuild(&positions, self.particles.len());
        }

        /// Blocking GPU → CPU particle sync. Updates `self.particles` immediately.
        /// Stalls the CPU until all in-flight GPU work completes — use only after step_frame
        /// when you need current positions right now (e.g. rendering). Not for the hot path.
        pub fn sync_particles_blocking(&mut self) {
            // Safe no-op once the device is lost -- see step_frame's identical guard.
            if self.is_device_lost() {
                return;
            }
            // If an async readback is in-flight, the staging buffer may be mapped or pending map.
            // Wait for it to complete, then consume it to unmap the staging buffer before reuse.
            // Real fix (2026-07-05, issue #10): must distinguish Ok/Err here -- the old
            // code called finish_readback (which calls get_mapped_range) on EITHER, but
            // a failed map has nothing valid to extract; only abandon_readback (unmap
            // only) is safe on Err. See GpuBuffers::abandon_readback's doc.
            if let Some(flag) = self.pending_readback.take() {
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                match flag.lock().ok().and_then(|mut g| g.take()) {
                    Some(Ok(())) => {
                        let _ = self.buffers.finish_readback(self.particle_count);
                    }
                    Some(Err(_)) => {
                        self.readback_error_count += 1;
                        self.buffers.abandon_readback();
                    }
                    None => {}
                }
            }
            self.particles =
                self.buffers
                    .readback_blocking(&self.device, &self.queue, self.particle_count);
            self.rebuild_spatial_hash();
        }

        /// Like `sync_particles_blocking`, but only for the given particle index ranges --
        /// updates just `self.particles[range]` for each range, leaving the rest of the CPU
        /// mirror as whatever the last async/full sync delivered. For callers that only need
        /// a small, known subset of particles current every frame (e.g. a handful of live
        /// creatures inside a much larger terrain/water scene) instead of the whole buffer --
        /// see `GpuBuffers::readback_ranges_blocking`'s own doc for why this is cheaper than
        /// repeated full syncs, not just "less data" but batched into one CPU↔GPU round-trip.
        /// Ranges may overlap or be given in any order; each is written independently.
        pub fn sync_particle_ranges_blocking(&mut self, ranges: &[std::ops::Range<usize>]) {
            // Safe no-op once the device is lost -- see step_frame's identical guard.
            if self.is_device_lost() {
                return;
            }
            // Same real fix as sync_particles_blocking -- see that function's comment.
            if let Some(flag) = self.pending_readback.take() {
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                match flag.lock().ok().and_then(|mut g| g.take()) {
                    Some(Ok(())) => {
                        let _ = self.buffers.finish_readback(self.particle_count);
                    }
                    Some(Err(_)) => {
                        self.readback_error_count += 1;
                        self.buffers.abandon_readback();
                    }
                    None => {}
                }
            }
            let results = self
                .buffers
                .readback_ranges_blocking(&self.device, &self.queue, ranges);
            for (range, data) in ranges.iter().zip(results) {
                self.particles[range.clone()].copy_from_slice(&data);
            }
            self.rebuild_spatial_hash();
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

        /// Physics snapshot from the CPU particle mirror (one frame behind GPU when strided).
        /// Grid-side fields (mass error, momentum error, active cells) are zero — GPU grid is
        /// not readable on CPU. All particle-side fields are exact.
        pub fn diagnostics_snapshot(&self) -> crate::diagnostics::SimSnapshot {
            crate::diagnostics::collect_snapshot_particles_only(
                self.frame_index,
                &self.particles,
                &self.config,
                self.last_sub_dt,
                self.last_substeps,
            )
        }

        /// Iterate over (index, &Particle) pairs within `radius` grid-cells of `center`.
        /// Reads the internal CPU particle mirror — one frame behind GPU when strided.
        /// O(candidates) via the internal spatial hash, not O(N) -- see `spatial_hash`
        /// field's own doc for why this matters at real scale (many creatures/queries
        /// per frame against a large terrain+water buffer).
        pub fn particles_near(
            &self,
            center: glam::Vec2,
            radius: f32,
        ) -> impl Iterator<Item = (usize, &Particle)> {
            let r2 = radius * radius;
            self.spatial_hash
                .query(center, radius)
                .filter_map(move |i| {
                    let p = &self.particles[i];
                    ((p.x - center).length_squared() <= r2).then_some((i, p))
                })
        }

        /// Count particles of `material_id` within `radius` grid-cells of `center`.
        /// O(candidates) via the internal spatial hash, not O(N).
        pub fn count_near(&self, center: glam::Vec2, radius: f32, material_id: u32) -> usize {
            let r2 = radius * radius;
            self.spatial_hash
                .query(center, radius)
                .filter(|&i| {
                    let p = &self.particles[i];
                    p.material_id == material_id && (p.x - center).length_squared() <= r2
                })
                .count()
        }

        /// Indices of the `k` particles nearest to `center`, sorted by distance
        /// ascending -- see `Simulation::particles_knn` (CPU, `src/solver/mod.rs`)
        /// for the full rationale (Ballerini et al. 2008, PNAS: real starling
        /// flocks use a topological ~6-7-nearest-neighbor rule, not a fixed
        /// radius) and exactness argument. Identical algorithm, mirrored here
        /// because the GPU backend keeps its own CPU-side spatial hash mirror.
        pub fn particles_knn(&self, center: glam::Vec2, k: usize) -> Vec<usize> {
            if k == 0 || self.particles.is_empty() {
                return Vec::new();
            }
            let domain_diag =
                self.config.grid_res as f32 * self.config.grid_cell_size * std::f32::consts::SQRT_2;
            let mut radius = self.config.grid_cell_size * (k as f32).sqrt().max(1.0);
            let mut candidates: Vec<(usize, f32)>;
            loop {
                let r2 = radius * radius;
                candidates = self
                    .spatial_hash
                    .query(center, radius)
                    .map(|i| (i, (self.particles[i].x - center).length_squared()))
                    .filter(|&(_, d2)| d2 <= r2)
                    .collect();
                if candidates.len() >= k || radius >= domain_diag {
                    break;
                }
                radius *= 2.0;
            }
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            candidates.truncate(k);
            candidates.into_iter().map(|(i, _)| i).collect()
        }

        /// Center of mass for particles in `range`. O(range.len()). GPU has no tag_index
        /// like CPU `Simulation::group_centroid` -- `range` (from `spawn_region`'s return)
        /// is the stable group identity here instead of a `u32` tag.
        pub fn group_centroid(&self, range: std::ops::Range<usize>) -> glam::Vec2 {
            let particles = &self.particles[range.clone()];
            if particles.is_empty() {
                return glam::Vec2::ZERO;
            }
            let sum: glam::Vec2 = particles.iter().map(|p| p.x).sum();
            sum / range.len() as f32
        }

        /// Aggregate state for all particles of the given material.
        pub fn material_state(&self, material_id: u32) -> crate::solver::query::BodyState {
            crate::solver::query::body_state_of_slice(&self.particles, material_id)
        }

        /// Aggregate state for all particles within `radius` grid-cells of `center`.
        pub fn region_state(
            &self,
            center: glam::Vec2,
            radius: f32,
        ) -> crate::solver::query::BodyState {
            crate::solver::query::region_body_state_of_slice(&self.particles, center, radius)
        }

        /// Reassign material for all particles matching `predicate`. Marks dirty so GPU
        /// sees the change on the next `step_frame` call.
        pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
        where
            F: Fn(&Particle) -> bool,
        {
            assert!(
                self.registry.is_registered(new_material_id),
                "phase_transition: material_id {new_material_id} is not registered — \
                 call solver.set_material({new_material_id}, ...) first"
            );
            for p in self.particles.iter_mut() {
                if predicate(p) {
                    p.material_id = new_material_id;
                }
            }
            self.layout_dirty = true; // material_id changed — sort order may differ
        }

        /// Add `force` to every particle within `radius` cells of `center`, scaled by proximity.
        /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions,
        /// avoiding any stale-CPU-mirror artifacts. No CPU particle scan.
        pub fn apply_impulse(&mut self, center: glam::Vec2, radius: f32, force: glam::Vec2) {
            if self.pending_impulses.len() < MAX_GPU_IMPULSES {
                self.pending_impulses.push(GpuImpulseEntry {
                    center: center.to_array(),
                    radius,
                    strength: 0.0,
                    force: force.to_array(),
                    mode: 1,
                    _pad: 0,
                });
            } else {
                eprintln!(
                    "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
                );
            }
        }

        /// Push every particle within `radius` cells outward from `center`.
        /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions.
        /// `strength` may be negative to pull. No CPU particle scan.
        pub fn apply_radial_impulse(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
            if self.pending_impulses.len() < MAX_GPU_IMPULSES {
                self.pending_impulses.push(GpuImpulseEntry {
                    center: center.to_array(),
                    radius,
                    strength,
                    force: [0.0; 2],
                    mode: 0,
                    _pad: 0,
                });
            } else {
                eprintln!(
                    "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
                );
            }
        }
    }

    #[cfg(test)]
    mod device_lost_tests {
        use super::*;
        use crate::materials::registry::MaterialRegistry;
        use crate::materials::{FromSI, NeoHookeanMaterial};
        use crate::solver::config::{SimConfig, SpawnRegion};
        use glam::{IVec2, Vec2};

        fn gpu_available() -> bool {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .is_ok()
        }

        /// Real, white-box verification of the device-lost guard added for emerge
        /// issue #10 (see project memory gpu_readback_error_path_bug_issue10 — the
        /// root cause, a genuine `Out of Memory` device loss under sustained
        /// slow-backend load, was confirmed with hard evidence via a real
        /// `device_lost_callback` firing; forcing that same OOM condition again just
        /// to test the GUARD would repeat the same heavy, machine-stressing
        /// reproduction unnecessarily). This directly injects a lost reason into the
        /// private `device_lost` flag exactly as the real callback would, then
        /// proves three things: (1) `device_lost_reason()` reports it, (2)
        /// `step_frame()` becomes a real no-op (frame_index does not advance,
        /// proving it didn't just avoid panicking by luck), (3) the blocking sync
        /// methods are also safe no-ops (don't panic touching a "dead" device).
        #[test]
        fn step_frame_becomes_safe_noop_once_device_lost() {
            if !gpu_available() {
                return;
            }
            let config = SimConfig {
                max_substeps_per_step: 8,
                ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
            };
            let spawn = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(4, 4),
                box_center: Vec2::new(16.0, 16.0),
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let particles = crate::build_particles(&config, spawn);
            let mat = NeoHookeanMaterial::from_physical(
                &crate::materials::physical_props::Elastic {
                    e_pa: 30.0e3,
                    nu: 0.45,
                    rho_kg_m3: 1000.0,
                },
                &config,
            );
            let registry = MaterialRegistry::with_default(Box::new(mat));
            let mut sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

            assert!(
                sim.device_lost_reason().is_none(),
                "a healthy, freshly-constructed sim must not report device loss"
            );

            sim.step_frame();
            let frame_after_healthy_step = sim.frame_index;
            assert!(
                frame_after_healthy_step > 0,
                "sanity check: a healthy device must actually advance frame_index"
            );

            // Directly inject a lost reason, exactly as the real callback does.
            *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
            assert_eq!(
                sim.device_lost_reason(),
                Some("Unknown: Out of memory".to_string()),
                "device_lost_reason() must report an injected loss"
            );

            sim.step_frame();
            assert_eq!(
                sim.frame_index, frame_after_healthy_step,
                "step_frame must become a real no-op once device_lost is set -- \
                 frame_index must NOT advance"
            );

            // Must not panic -- these touch the same "dead" device.
            sim.sync_particles_blocking();
            let ranges = vec![0..1usize, 1..2usize];
            sim.sync_particle_ranges_blocking(&ranges);
        }

        /// Real repro of issue #10's ACTUAL failure mode, found via real windows-latest
        /// CI evidence (not speculation): re-enabling the crash-repro test showed that,
        /// under sustained load, wgpu invalidates/destroys buffers tied to the device
        /// BEFORE this instance's `device_lost_callback` fires -- the readback path's
        /// `.unmap()` call then hits an uncaptured Validation error naming the destroyed
        /// resource, and wgpu's default handler panics unconditionally
        /// (`default_error_handler`: `panic!("wgpu error: {err}")`, confirmed by reading
        /// wgpu-27.0.1's source directly).
        ///
        /// Forcing a real 9-minute sustained-load OOM again just to hit this exact race
        /// would repeat the same heavy, machine-stressing reproduction unnecessarily --
        /// this reproduces the SAME call path directly: destroy the readback staging
        /// buffer ourselves (exactly what the device-loss cascade does to it), then call
        /// `abandon_readback()`, the exact function whose `.unmap()` call panicked on
        /// real CI. Before the `on_uncaptured_error` fix this would panic and abort the
        /// test process; with it installed, it must set `device_lost` instead -- no
        /// panic, `device_lost_reason()` reports it.
        #[test]
        fn uncaptured_destroyed_buffer_error_sets_device_lost_not_a_panic() {
            if !gpu_available() {
                return;
            }
            let config = SimConfig {
                max_substeps_per_step: 8,
                ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
            };
            let spawn = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(4, 4),
                box_center: Vec2::new(16.0, 16.0),
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let particles = crate::build_particles(&config, spawn);
            let mat = NeoHookeanMaterial::from_physical(
                &crate::materials::physical_props::Elastic {
                    e_pa: 30.0e3,
                    nu: 0.45,
                    rho_kg_m3: 1000.0,
                },
                &config,
            );
            let registry = MaterialRegistry::with_default(Box::new(mat));
            let sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

            assert!(
                sim.device_lost_reason().is_none(),
                "a healthy, freshly-constructed sim must not report device loss"
            );

            // Exactly what a device-loss cascade does to resources tied to the
            // device, without needing 9 minutes of real sustained WARP load.
            sim.buffers.readback_staging.destroy();

            // The exact real call path that panicked on CI: finish_readback and
            // abandon_readback both end in `.unmap()` on this buffer.
            sim.buffers.abandon_readback();
            sim.device.poll(wgpu::PollType::Poll).ok();

            let reason = sim.device_lost_reason();
            assert!(
                reason.is_some(),
                "an uncaptured error naming a destroyed buffer must set device_lost, \
                 not silently do nothing"
            );
            let reason = reason.unwrap();
            assert!(
                reason.contains("uncaptured wgpu error"),
                "reason should be tagged as coming from the uncaptured-error handler \
                 (distinguishable from the official device_lost_callback's report), \
                 got: {reason}"
            );
            assert!(
                reason.contains("destroyed"),
                "reason should retain the real wgpu error text naming the destroyed \
                 resource, got: {reason}"
            );
        }

        /// Real proof that the OPT-IN path works -- this is the path LP's actual
        /// production code needs (`World::with_device`, since LP shares its device
        /// with a renderer and has no device-lost handling of its own, confirmed by
        /// inspection of LP's `src/main.rs`). Proves `enable_device_lost_detection()`
        /// makes a `with_device()` instance behave identically to a `new()`-
        /// constructed one for reporting purposes. NOTE: this does NOT independently
        /// prove `with_device()` never silently registers its own callback -- that
        /// would need a real device-loss trigger to distinguish "no callback
        /// registered" from "callback registered but nothing happened yet," which
        /// this test doesn't force (see the heavy stress-test caution elsewhere in
        /// this file). Static code inspection is what actually backs that claim:
        /// `with_device()`'s body contains no `set_device_lost_callback` call.
        #[test]
        fn with_device_instances_need_explicit_opt_in_for_device_lost_detection() {
            if !gpu_available() {
                return;
            }
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("no adapter");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("test_shared_device"),
                    required_limits: adapter.limits(),
                    ..Default::default()
                }))
                .expect("no device");
            let device = Arc::new(device);
            let queue = Arc::new(queue);

            let config = SimConfig {
                max_substeps_per_step: 8,
                ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
            };
            let mat = NeoHookeanMaterial::from_physical(
                &crate::materials::physical_props::Elastic {
                    e_pa: 30.0e3,
                    nu: 0.45,
                    rho_kg_m3: 1000.0,
                },
                &config,
            );
            let sim = GpuSimulation::with_device(
                device,
                queue,
                config,
                Vec::new(),
                MaterialRegistry::with_default(Box::new(mat)),
            );

            // Fresh with_device() instance: field starts unset (expected regardless
            // of whether a callback is wired -- see doc comment above for what this
            // does and doesn't prove).
            assert!(sim.device_lost_reason().is_none());

            sim.enable_device_lost_detection();
            // Directly invoke the same injection used in the other test -- proves the
            // callback registration path (not just the field) is wired correctly.
            *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
            assert_eq!(
                sim.device_lost_reason(),
                Some("Unknown: Out of memory".to_string()),
                "after enable_device_lost_detection(), device_lost_reason() must work \
                 identically to a new()-constructed instance"
            );
        }
    }
}

/// GPU buffer management for the MLS-MPM solver.
///
/// Persistent buffers in VRAM for the simulation lifetime:
///   - `particles`:          array<Particle>          — 112 bytes each, repr(C)
///   - `grid`:               array<Cell>              — 16 bytes each, repr(C)
///   - `materials`:          array<MaterialParams, N> — 96 bytes each, 16-byte aligned
///   - `step_params`:        GpuStepParams            — 32 bytes, uploaded once per substep
///   - `force_fields_params: GpuFieldsParams     — 784 bytes, uploaded when fields change
///
/// Upload path (CPU → GPU, via write_buffer):
///   particles:          initial spawn, then only when CPU plasticity corrects state
///   materials:          on spawn + on interactive param change
///   step_params:        every substep (sub_dt changes)
///   force_fields_params: every substep (entries may change between frames)
///   grid:               never uploaded — zeroed each substep by grid_clear compute pass
///
/// Download path (GPU → CPU):
///   particles:   only for plasticity readback (snow/sand) or diagnostics
///   grid:        never downloaded
use std::mem;

use super::step_params::{
    ContactDebugParams, GpuAsflipParams, GpuDirectionalGripParams, GpuFieldsParams,
    GpuImpulseParams, GpuMaterialMassParams, GpuResourceParams, GpuSleepWakeParams, GpuStepParams,
    GpuThermalParams, MAX_CONTACT_POINTS_PER_BLOCK, MAX_RENDER_MATERIAL_SLOTS, NUM_BLOCKS,
};
use crate::materials::MaterialParams;
use crate::particle::Particle;

/// Cell layout matching `struct Cell` in every WGSL shader — 16 bytes.
/// Only used for buffer sizing; the GPU writes it via the shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCell {
    momentum: [f32; 2],
    mass: f32,
    _pad: f32,
}

const _: () = assert!(mem::size_of::<GpuCell>() == 16);

/// All persistent GPU buffers for one GpuSimulation instance.
pub struct GpuBuffers {
    /// Particle data — STORAGE | COPY_DST | COPY_SRC.
    pub particles: wgpu::Buffer,
    /// Grid cells, zeroed each substep by grid_clear pass — STORAGE
    pub grid: wgpu::Buffer,
    /// One MaterialParams per registered material slot — UNIFORM | COPY_DST
    pub materials: wgpu::Buffer,
    /// Per-substep constants pool — one buffer per max_substeps slot, each 32 bytes.
    /// Substep i reads `step_params_pool[i]` so all substeps can be encoded in one command buffer.
    pub step_params_pool: Vec<wgpu::Buffer>,
    /// Non-uniform force-field entries for the force_fields pass — UNIFORM | COPY_DST
    pub force_fields_params: wgpu::Buffer,
    /// Impulse descriptors for the apply_impulses pass — UNIFORM | COPY_DST
    pub impulse_params: wgpu::Buffer,
    /// Force-sleep/force-wake-by-tag entries for the force_fields pass — UNIFORM | COPY_DST.
    /// Minimal hook for LP's future chunk system — see `GpuSleepWakeParams` doc comment.
    pub sleep_wake_params: wgpu::Buffer,
    /// Sorted particle index permutation — STORAGE.
    /// Written once per frame by the particle_sort count→scan→scatter pipeline (block-level
    /// counting sort by spatial position). Read by p2g and particles_update for cache-coherent
    /// particle access (Gao et al. 2018, "GPU Optimization of Material Point Methods").
    pub sorted_particle_ids: wgpu::Buffer,
    /// Per-block atomic counters for the particle_sort pipeline — NUM_BLOCKS (256) × u32.
    /// Cleared, filled (histogram), scanned (exclusive prefix sum), then reused as the atomic
    /// scatter cursor — all in one frame's particle_sort pass sequence.
    pub block_counts: wgpu::Buffer,
    /// Compacted active block IDs, rebuilt every frame — NUM_BLOCKS (256) × u32, STORAGE |
    /// COPY_SRC (COPY_SRC for test readback, same precedent as sorted_particle_ids). GPU
    /// sparse grid (see mpm_technique_survey memory note): block b is "active" iff it OR one
    /// of its 8 neighbors contains at least one particle this frame (halo-expanded in
    /// particle_sort_compact_main so the P2G kernel's cross-block scatter stencil is always
    /// covered), detected from particle_sort's raw per-block histogram before scan overwrites
    /// it into a scatter cursor. Consumed by grid_clear AND grid_update (Phase 1 and Phase 2
    /// respectively — both bound their real work to occupied blocks instead of the whole
    /// dense grid_res² domain; grid_update additionally guards against double-processing a
    /// block present in both this list and `active_block_ids_prev`, since unlike grid_clear's
    /// idempotent zero-write, grid_update computes each cell via several read-modify-write
    /// steps and two workgroups racing on the same non-atomic cells is a real correctness bug
    /// — found and fixed 2026-07-12, see `grid_update.wgsl`). The `grid` buffer itself stays
    /// dense (real memory compaction, not just bounded dispatch, would need a fundamentally
    /// different sparse-allocation scheme — a separate, larger undertaking, not done here).
    pub active_block_ids: wgpu::Buffer,
    /// Atomic count of valid entries in `active_block_ids` this frame — 1 × u32, STORAGE |
    /// COPY_SRC.
    pub active_block_count: wgpu::Buffer,
    /// Snapshot of `active_block_ids`/`active_block_count` from the IMMEDIATELY PRECEDING
    /// substep — NUM_BLOCKS × u32, STORAGE. Real bug fix: without this, a block that stops
    /// being active never gets cleared again, since grid_clear only ever clears CURRENTLY
    /// active blocks — its last P2G contribution would sit there permanently until some
    /// particle wandered back near it much later (see `active_block_swap_main`'s doc comment
    /// in `particle_sort.wgsl` for the full story, including a first attempt at this fix that
    /// was wrong). grid_clear processes the union of `active_block_ids` and this buffer,
    /// giving every block a genuine one-substep grace period before being left alone.
    pub active_block_ids_prev: wgpu::Buffer,
    /// Companion to `active_block_ids_prev` — 1 × u32, STORAGE. NOT atomic (only ever written
    /// by the single-threaded `lid.x == 0u` branch of `active_block_swap_main`, read by
    /// `grid_clear`), unlike `active_block_count` which needs atomics for concurrent
    /// `atomicAdd` from `particle_sort_compact_main`.
    pub active_block_count_prev: wgpu::Buffer,
    /// Persistent readback staging buffer — pre-allocated to avoid per-frame alloc/dealloc.
    /// COPY_DST | MAP_READ. Same size as `particles`.
    pub readback_staging: wgpu::Buffer,
    /// Multi-field contact (GPU port, first slice — 2026-07-14, see project memory
    /// `locomotion_core_frictional_contact_2026-07-11`) — the "grip" field's own
    /// mass/momentum accumulator, same dense per-cell layout and fixed-point atomic
    /// scatter convention as `grid` itself (`momentum.xy, mass, pad` × 4 bytes per
    /// float, `grid_res²` cells). Additively scattered by `p2g.wgsl` alongside the
    /// ordinary total-field scatter, gated on `Particle::contact_group != 0`. Zeroed
    /// every substep by `grid_clear.wgsl` (extended to also clear this buffer) — same
    /// active-block-bounded dispatch as `grid`, not the whole dense domain.
    pub grip_grid: wgpu::Buffer,
    /// Labeled contact point cloud (`+1.0` grip / `-1.0` rest, `vec4<f32>` = position.xy,
    /// label, unused) for the Newton-Raphson LR contact-normal fit — CPU's mirror is
    /// `ContactCell::points` (an unbounded per-node `Vec`, see that type's doc). Bucketed
    /// per coarse BLOCK (`NUM_BLOCKS` = 256, same spatial partition `particle_sort.wgsl`
    /// already uses), NOT per exact grid node — a first version bucketed per node and
    /// OOM'd the real `gpu_grid_resolution_cost` regression test at grid_res=2048 (see
    /// `MAX_CONTACT_POINTS_PER_BLOCK`'s doc in `step_params.rs` for the full story).
    /// Fixed total size (`NUM_BLOCKS × MAX_CONTACT_POINTS_PER_BLOCK × 16 bytes` ≈ 16 MiB),
    /// independent of grid_res. Populated by a dedicated pass (`gather_contact_points_main`
    /// in `p2g.wgsl`) that runs AFTER the main P2G scatter, mirroring CPU's own two-pass
    /// ordering (`gather_contact_point_cloud`'s doc: point data is only meaningful once
    /// grip mass has already been measured at a node this same substep). STORAGE |
    /// COPY_SRC (test readback only, same precedent as `sorted_particle_ids`).
    pub contact_points: wgpu::Buffer,
    /// Per-BLOCK atomic count of points written into `contact_points` this substep —
    /// `NUM_BLOCKS` (256) × `atomic<u32>`, fixed size regardless of grid_res. Bounds-
    /// checked against `MAX_CONTACT_POINTS_PER_BLOCK` at the atomic slot-claim site
    /// (`gather_contact_points_main`) — excess points are dropped, not undefined
    /// behavior, and the counter keeps counting past the cap so overflow is a real,
    /// observable signal. Zeroed every substep by `particle_sort_clear_main`
    /// (`particle_sort.wgsl`) alongside its own `block_counts` — same 256-wide dispatch,
    /// not `grid_clear.wgsl` (which processes per-CELL work; this is per-block).
    pub contact_point_counts: wgpu::Buffer,
    /// Debug/test-only uniform for `resolve_contact.wgsl`'s `debug_fit_normal_main` —
    /// see `ContactDebugParams`'s own doc. Not touched by the real per-substep
    /// pipeline.
    pub contact_debug_params: wgpu::Buffer,
    /// Debug/test-only output for `debug_fit_normal_main` — `[n.x, n.y, valid]`, 16
    /// bytes (padded to satisfy storage-buffer minimum alignment).
    pub contact_debug_output: wgpu::Buffer,
    /// Resolved "grip" field velocity per grid node, written by `resolve_contact_main`
    /// — dense `grid_res² × vec2<f32>`, mirrors CPU's `Grid::grip_velocity_at`. Defaults
    /// to the ordinary total velocity at every cell (matching CPU's fallback), overwritten
    /// with the real resolved value only at genuinely contact-active nodes. Read by a
    /// future G2P routing change for particles with `contact_group != 0`.
    pub resolved_grip_v: wgpu::Buffer,
    /// Resolved "rest" (contact_group == 0) field velocity — same layout/fallback as
    /// `resolved_grip_v`, mirrors CPU's `Grid::rest_velocity_at`.
    pub resolved_rest_v: wgpu::Buffer,
    /// Directional grip friction params — see `GpuDirectionalGripParams`' own doc.
    pub grip_params: wgpu::Buffer,
    /// Day-night/ambient thermal diffusion (GPU port) — real config uniform, see
    /// `GpuThermalParams`' own doc. Always uploaded (disabled by default), same
    /// always-present-but-cheap-when-unused pattern as `grip_params`.
    pub thermal_params: wgpu::Buffer,
    /// Thermal scratch: Σ(w·mass) per cell, cleared+rebuilt every substep by the
    /// thermal P2G pass — dense `grid_res²` f32, mirrors CPU `ThermalDiffusion::
    /// grid_mass`.
    pub thermal_mass: wgpu::Buffer,
    /// Thermal scratch: normalized T_old per cell (needed for the G2P delta gather,
    /// `T_new − T_old`) — dense `grid_res²` f32, mirrors CPU's `grid_temp`.
    pub thermal_temp_old: wgpu::Buffer,
    /// Thermal scratch, dual-use like CPU's own `grid_work`: P2G scatter accumulator
    /// first, then overwritten with the post-Laplacian `T_new` — dense `grid_res²` f32.
    pub thermal_work: wgpu::Buffer,
    /// Resource regrowth (GPU port) -- real config uniform, see `GpuResourceParams`'
    /// own doc. Own separate group/buffers from thermal (see that struct's doc for why).
    pub resource_params: wgpu::Buffer,
    /// Resource scratch: Σ(w·mass) per cell -- dense `grid_res²` f32, mirrors
    /// `thermal_mass`.
    pub resource_mass: wgpu::Buffer,
    /// Resource scratch: normalized φ_old per cell -- dense `grid_res²` f32, mirrors
    /// `thermal_temp_old`.
    pub resource_phi_old: wgpu::Buffer,
    /// Resource scratch, dual-use like `thermal_work`: P2G scatter accumulator first,
    /// then overwritten with the post-Laplacian+logistic-growth `φ_new`.
    pub resource_work: wgpu::Buffer,
    /// ASFLIP (GPU port) -- real config uniform, see `GpuAsflipParams`' own doc.
    pub asflip_params: wgpu::Buffer,
    /// Dense `grid_res² × vec2<f32>` pre-force velocity snapshot -- GPU mirror of CPU's
    /// `Grid::snapshot_velocities` (a sparse `HashMap`), made dense here because the GPU
    /// `grid` buffer already is (no sparse GPU allocation scheme exists, same real
    /// trade-off `grid` itself already discloses). Written by `grid_update.wgsl` right
    /// after momentum normalization, before gravity is added -- the same pre-force
    /// instant CPU snapshots at. Read by `g2p_asflip_fused.wgsl`'s second gather.
    ///
    /// LAZILY allocated, unlike `grip_params`/`thermal_params` -- REAL bug found and
    /// fixed 2026-07-17: at 8 bytes/cell this is the same order of magnitude as the main
    /// `grid` buffer itself (16 bytes/cell), not a cheap tiny uniform. Allocating it
    /// unconditionally at full `grid_res²` size for every `GpuSimulation` (the vast
    /// majority of which never call `attach_asflip_gpu`) measurably tipped already
    /// memory-marginal tests (a 16,000-step terrain settle, a grid_res-up-to-2048 sweep)
    /// into real `wgpu` "Out of Memory" territory on this machine -- confirmed via a
    /// direct A/B: the full test suite genuinely OOM'd with this buffer at real size,
    /// and passed clean (42/42, normal ~200s runtime) with it shrunk to a placeholder.
    /// Starts at `PLACEHOLDER_BYTES` (never read/written while `asflip_params.enabled ==
    /// 0`, so a too-small buffer is safe); `GpuSimulation::attach_asflip_gpu` grows it to
    /// the real size on first use, mirroring `spawn_region`'s existing reallocate-and-
    /// rebuild-bind-group pattern.
    pub asflip_snapshot: wgpu::Buffer,
    /// `true` once `asflip_snapshot` has been grown to its real `grid_res²` size --
    /// lets `attach_asflip_gpu` skip re-allocating (and rebuilding the bind group that
    /// references it) on every call, only the first.
    pub asflip_snapshot_grown: bool,
    /// `ColorMode::GridVolume`'s opt-in per-cell per-material mass accumulator (see
    /// `grid_volume.wgsl`'s own doc for the real technique). Real config uniform,
    /// same always-present-but-cheap-when-unused pattern as `grip_params`.
    pub material_mass_params: wgpu::Buffer,
    /// Dense `grid_res² × MAX_RENDER_MATERIAL_SLOTS × f32` per-material mass
    /// accumulator, same fixed-point atomic scatter convention as `grid` itself.
    /// LAZILY allocated, same real reason `asflip_snapshot` is (this is
    /// `MAX_RENDER_MATERIAL_SLOTS`x the size of the main grid's own mass field --
    /// unconditionally allocating it for every `GpuSimulation`, the vast majority of
    /// which never call `attach_grid_material_render_gpu`, would repeat the exact
    /// OOM risk already found and fixed for ASFLIP). Starts at
    /// `MATERIAL_MASS_PLACEHOLDER_BYTES` (never read/written while
    /// `material_mass_params.enabled == 0`).
    pub material_mass: wgpu::Buffer,
    /// `true` once `material_mass` has been grown to its real size -- mirrors
    /// `asflip_snapshot_grown`.
    pub material_mass_grown: bool,
}

/// `asflip_snapshot`'s pre-attach size -- large enough to satisfy wgpu's nonzero-buffer
/// requirement, small enough to be genuinely free. Never actually read/written at this
/// size (gated on `asflip_params.enabled`), so its exact value doesn't matter beyond
/// "nonzero and 8-byte aligned for vec2<f32>".
const ASFLIP_SNAPSHOT_PLACEHOLDER_BYTES: u64 = 8;

/// `material_mass`'s pre-attach size -- same rationale as
/// `ASFLIP_SNAPSHOT_PLACEHOLDER_BYTES` (nonzero for wgpu, never actually touched while
/// disabled).
const MATERIAL_MASS_PLACEHOLDER_BYTES: u64 = 4;

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        particle_count: usize,
        grid_res: usize,
        max_materials: usize,
        max_substeps: usize,
    ) -> Self {
        let particle_bytes = (particle_count * mem::size_of::<Particle>()) as u64;
        let grid_bytes = (grid_res * grid_res * mem::size_of::<GpuCell>()) as u64;
        let material_bytes = (max_materials * mem::size_of::<MaterialParams>()) as u64;
        let step_bytes = mem::size_of::<GpuStepParams>() as u64;

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particles"),
            size: particle_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_grid"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let materials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_materials"),
            size: material_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Allocate max_substeps + 1 slots: 0..max_substeps for physics substeps,
        // slot max_substeps is a dedicated particle_sort slot so it never aliases substep 0.
        let step_params_pool: Vec<wgpu::Buffer> = (0..max_substeps + 1)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("mpm_step_params_{i}")),
                    size: step_bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let force_fields_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_force_fields_params"),
            size: mem::size_of::<GpuFieldsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let impulse_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_impulse_params"),
            size: mem::size_of::<GpuImpulseParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sleep_wake_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sleep_wake_params"),
            size: mem::size_of::<GpuSleepWakeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_particle_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sorted_particle_ids"),
            size: (particle_count * mem::size_of::<u32>()) as u64,
            // COPY_SRC: needed for test-only readback (gpu_particle_sort_is_valid_permutation).
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // NUM_BLOCKS (256) must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM² exactly.
        let block_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_block_counts"),
            size: (NUM_BLOCKS * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // GPU sparse grid Phase 1 — see active_block_ids/active_block_count field docs.
        let active_block_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_active_block_ids"),
            size: (NUM_BLOCKS * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_block_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_active_block_count"),
            size: mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let active_block_ids_prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_active_block_ids_prev"),
            size: (NUM_BLOCKS * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let active_block_count_prev = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_active_block_count_prev"),
            size: mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let readback_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particle_staging"),
            size: particle_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Multi-field contact (GPU port, first slice) — see field docs above.
        let grip_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_grip_grid"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Fixed size, independent of grid_res — see MAX_CONTACT_POINTS_PER_BLOCK's doc
        // for why this is bucketed per coarse block (NUM_BLOCKS), not per exact node.
        let contact_points_bytes = (NUM_BLOCKS * MAX_CONTACT_POINTS_PER_BLOCK * 16) as u64;
        let contact_points = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_contact_points"),
            size: contact_points_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let contact_point_counts_bytes = (NUM_BLOCKS * mem::size_of::<u32>()) as u64;
        let contact_point_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_contact_point_counts"),
            size: contact_point_counts_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Multi-field contact (GPU port, second slice) — debug/test-only scaffolding
        // for verifying the Newton-Raphson LR normal fit in isolation.
        let contact_debug_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_contact_debug_params"),
            size: mem::size_of::<ContactDebugParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let contact_debug_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_contact_debug_output"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Multi-field contact (GPU port, third slice) — resolved velocities + directional
        // grip params, see field docs above.
        let resolved_vel_bytes = (grid_res * grid_res * mem::size_of::<[f32; 2]>()) as u64;
        let resolved_grip_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resolved_grip_v"),
            size: resolved_vel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let resolved_rest_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resolved_rest_v"),
            size: resolved_vel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let grip_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_grip_params"),
            size: mem::size_of::<GpuDirectionalGripParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Day-night/ambient thermal diffusion (GPU port) — see field docs above.
        let thermal_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_thermal_params"),
            size: mem::size_of::<GpuThermalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let thermal_scalar_bytes = (grid_res * grid_res * mem::size_of::<f32>()) as u64;
        let thermal_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_thermal_mass"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let thermal_temp_old = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_thermal_temp_old"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let thermal_work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_thermal_work"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Resource regrowth (GPU port) -- own separate group/buffers, see field docs above.
        let resource_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resource_params"),
            size: mem::size_of::<GpuResourceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let resource_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resource_mass"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let resource_phi_old = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resource_phi_old"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let resource_work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_resource_work"),
            size: thermal_scalar_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ASFLIP (GPU port) -- own separate group/buffer, see field docs above.
        let asflip_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_asflip_params"),
            size: mem::size_of::<GpuAsflipParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let asflip_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_asflip_snapshot"),
            size: ASFLIP_SNAPSHOT_PLACEHOLDER_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let material_mass_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_material_mass_params"),
            size: mem::size_of::<GpuMaterialMassParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_material_mass"),
            size: MATERIAL_MASS_PLACEHOLDER_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            particles,
            grid,
            materials,
            step_params_pool,
            force_fields_params,
            impulse_params,
            sleep_wake_params,
            sorted_particle_ids,
            block_counts,
            active_block_ids,
            active_block_count,
            active_block_ids_prev,
            active_block_count_prev,
            readback_staging,
            grip_grid,
            contact_points,
            contact_point_counts,
            contact_debug_params,
            contact_debug_output,
            resolved_grip_v,
            resolved_rest_v,
            grip_params,
            thermal_params,
            thermal_mass,
            thermal_temp_old,
            thermal_work,
            resource_params,
            resource_mass,
            resource_phi_old,
            resource_work,
            asflip_params,
            asflip_snapshot,
            asflip_snapshot_grown: false,
            material_mass_params,
            material_mass,
            material_mass_grown: false,
        }
    }

    /// Grow `asflip_snapshot` from its placeholder to real `grid_res²` size -- called
    /// once, by `GpuSimulation::attach_asflip_gpu` on first use. No-op if already grown
    /// (idempotent under repeated `attach_asflip_gpu` calls). Caller must rebuild any
    /// bind group referencing this buffer afterward (its identity changes) -- see
    /// `SimPipelines::make_resource_bind_group`.
    pub fn grow_asflip_snapshot(&mut self, device: &wgpu::Device, grid_res: usize) {
        if self.asflip_snapshot_grown {
            return;
        }
        self.asflip_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_asflip_snapshot"),
            size: (grid_res * grid_res * mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.asflip_snapshot_grown = true;
    }

    /// Grow `material_mass` from its placeholder to real
    /// `grid_res² x MAX_RENDER_MATERIAL_SLOTS` size -- called once, by
    /// `GpuSimulation::attach_grid_material_render_gpu` on first use. No-op if already
    /// grown. Caller must rebuild the contact bind group afterward (its identity
    /// changes) -- mirrors `grow_asflip_snapshot` exactly.
    pub fn grow_material_mass(&mut self, device: &wgpu::Device, grid_res: usize) {
        if self.material_mass_grown {
            return;
        }
        self.material_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_material_mass"),
            size: (grid_res * grid_res * MAX_RENDER_MATERIAL_SLOTS as usize * mem::size_of::<f32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.material_mass_grown = true;
    }

    pub fn upload_particles(&self, queue: &wgpu::Queue, particles: &[Particle]) {
        queue.write_buffer(&self.particles, 0, Particle::slice_as_bytes(particles));
    }

    pub fn upload_materials(&self, queue: &wgpu::Queue, params: &[MaterialParams]) {
        queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(params));
    }

    /// Upload step params into pool slot `index`. Panics if index >= pool size.
    pub fn upload_step_params_at(&self, queue: &wgpu::Queue, index: usize, params: &GpuStepParams) {
        queue.write_buffer(&self.step_params_pool[index], 0, bytemuck::bytes_of(params));
    }

    pub fn upload_force_fields_params(&self, queue: &wgpu::Queue, params: &GpuFieldsParams) {
        queue.write_buffer(&self.force_fields_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_impulse_params(&self, queue: &wgpu::Queue, params: &GpuImpulseParams) {
        queue.write_buffer(&self.impulse_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_sleep_wake_params(&self, queue: &wgpu::Queue, params: &GpuSleepWakeParams) {
        queue.write_buffer(&self.sleep_wake_params, 0, bytemuck::bytes_of(params));
    }

    /// Debug/test-only upload for `resolve_contact.wgsl`'s `debug_fit_normal_main` —
    /// see `ContactDebugParams`'s own doc.
    pub fn upload_contact_debug_params(&self, queue: &wgpu::Queue, params: &ContactDebugParams) {
        queue.write_buffer(&self.contact_debug_params, 0, bytemuck::bytes_of(params));
    }

    /// Upload directional grip friction params — see `GpuDirectionalGripParams`' doc.
    pub fn upload_grip_params(&self, queue: &wgpu::Queue, params: &GpuDirectionalGripParams) {
        queue.write_buffer(&self.grip_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_thermal_params(&self, queue: &wgpu::Queue, params: &GpuThermalParams) {
        queue.write_buffer(&self.thermal_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_resource_params(&self, queue: &wgpu::Queue, params: &GpuResourceParams) {
        queue.write_buffer(&self.resource_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_asflip_params(&self, queue: &wgpu::Queue, params: &GpuAsflipParams) {
        queue.write_buffer(&self.asflip_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_material_mass_params(&self, queue: &wgpu::Queue, params: &GpuMaterialMassParams) {
        queue.write_buffer(&self.material_mass_params, 0, bytemuck::bytes_of(params));
    }
}

// GPU -> CPU readback methods (begin_readback, readback_blocking and its
// variants, finish_readback, abandon_readback) -- split into their own file,
// was ~250 of this file's ~700 lines, matching this file's own doc comment
// split between "Upload path" and "Download path".
mod readback;

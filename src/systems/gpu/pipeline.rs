/// Compute pipeline setup for MLS-MPM GPU passes.
///
/// Eleven passes per frame:
///   Once per frame, in order (block-level counting sort, see particle_sort.wgsl):
///     0a. particle_sort_clear    — zero the 256-entry block histogram + active_block_count
///     0b. particle_sort_count    — one thread per particle, build histogram
///     0c. particle_sort_compact  — GPU sparse grid Phase 1: record which blocks are occupied
///                                  (reads the RAW histogram, must run before scan overwrites it)
///     0d. particle_sort_scan     — one workgroup, exclusive prefix sum -> scatter cursor
///     0e. particle_sort_scatter  — one thread per particle, write sorted_particle_ids
///   Per substep:
///     1. grid_clear       — zero only cells in active blocks (see grid_clear.wgsl, Phase 1)
///     2. p2g              — scatter particles → grid (sorted access, 64-wide workgroups)
///     3. grid_update      — normalize momentum→velocity, apply gravity, enforce boundary —
///                            active-block dispatch too (see grid_update.wgsl, Phase 2)
///     4. g2p              — gather grid → particles, write v + velocity_gradient only
///     5. particles_update — F update, plasticity, volume/density, position, boundary (sorted)
///     6. force_fields     — apply non-uniform body forces after particles_update
///
/// TWO bind group layouts shared by all passes (split 2026-07-16 — a single 20-binding
/// layout hit a real, present limit: `create_bind_group_layout` failed on any adapter
/// exposing only the WebGPU-guaranteed baseline of 8 storage buffers per compute stage,
/// once contact's GPU port pushed the count to 14. `maxStorageBuffersPerShaderStage` is
/// validated per bind-group-layout, not aggregated across a pipeline's layouts, so
/// splitting genuinely fixes it rather than moving the count around. Every real pass sets
/// BOTH groups regardless of which bindings its own entry point references, same
/// philosophy as "passes that don't use a binding still share the same layout" below —
/// keeps `encode_substep`/`readback.rs` from needing per-shader reasoning about which
/// group is actually touched.
///
/// Group 0 — core MPM state, needed by nearly every pass (8 storage, at the baseline
/// limit with zero headroom; any future core addition needs its own new group, not a
/// squeeze into this one):
///   binding 0: particles            — storage read_write
///   binding 1: grid                 — storage read_write
///   binding 2: materials            — uniform (array<MaterialParams, MAX_MATERIALS>)
///   binding 3: step_params          — uniform (GpuStepParams, 32 bytes)
///   binding 4: force_fields_params  — uniform (GpuFieldsParams, 784 bytes)
///   binding 5: sorted_particle_ids  — storage read_write (u32 per particle)
///   binding 6: block_counts         — storage read_write (256 atomic<u32>, particle_sort only)
///   binding 7: sleep_wake_params    — uniform (GpuSleepWakeParams, 80 bytes)
///   binding 8: active_block_ids     — storage read_write (256 u32 — particle_sort writes,
///                                     grid_clear/grid_update read; GPU sparse grid)
///   binding 9: active_block_count   — storage read_write (1 atomic<u32> — same pair as above)
///   binding 10: active_block_ids_prev   — storage read_write (256 u32 — one-substep grace
///                                         period, same consumers as binding 8)
///   binding 11: active_block_count_prev — storage read_write (1 u32 — same pair as above)
///
/// Group 1 — multi-field contact subsystem, only touched by contact-related passes (6
/// storage, 2 headroom below the baseline limit). None of these buffers are
/// particle-count-scaled (all fixed grid_res²-sized), so unlike group 0's bind group
/// (rebuilt whenever `spawn_region` reallocates `buffers.particles`), this bind group is
/// built once at construction and never needs rebuilding:
///   binding 12: grip_grid               — storage read_write (multi-field contact "grip"
///                                         field mass/momentum, grid_res² cells — GPU port,
///                                         first slice, see buffers.rs doc)
///   binding 13: contact_points           — storage read_write (labeled contact point cloud,
///                                         grid_res² × MAX_CONTACT_POINTS_PER_NODE)
///   binding 14: contact_point_counts     — storage read_write (grid_res² atomic<u32>)
///   binding 15: contact_debug_params     — uniform (ContactDebugParams, 16 bytes,
///                                         debug/test-only, resolve_contact.wgsl)
///   binding 16: contact_debug_output     — storage read_write (debug/test-only)
///   binding 17: resolved_grip_v          — storage read_write (grid_res² vec2<f32>)
///   binding 18: resolved_rest_v          — storage read_write (grid_res² vec2<f32>)
///   binding 19: grip_params              — uniform (GpuDirectionalGripParams, 16 bytes)
///
/// Group 3 also carries ASFLIP's 2 bindings (28-29, GPU port) alongside resource
/// regrowth -- NOT because the two are related (they aren't), but because WebGPU's
/// baseline `max_bind_groups` is exactly 4 (confirmed against wgpu-types' own downlevel
/// defaults) and this pipeline already uses all 4 -- the same baseline-adapter safety
/// concern that forced the original group 0/1 split in the first place. A 5th group
/// would break on any adapter reporting only the guaranteed baseline. Group 3 has real
/// headroom (4 of 8 storage slots used), so ASFLIP's 2 bindings go there instead of a
/// new group:
///   binding 28: asflip_params  — uniform (GpuAsflipParams, 16 bytes)
///   binding 29: asflip_snapshot — storage read_write (grid_res² vec2<f32> pre-force
///                                velocity snapshot, see buffers.rs doc)
///
/// Passes that don't use a binding still share the same layout — avoids rebinding.
use super::buffers::GpuBuffers;
use super::shaders;
use super::step_params::{
    MAX_FORCE_FIELDS, MAX_MATERIALS, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS_PER_DIM,
    NUM_CONTACT_BLOCKS_PER_DIM,
};

/// All compiled compute pipelines for one GpuSimulation instance.
pub struct SimPipelines {
    /// Once per frame, in order: clear histogram -> count per-block -> compact (active-block
    /// list, GPU sparse grid Phase 1) -> scan (exclusive prefix sum) -> scatter into
    /// sorted_particle_ids. See particle_sort.wgsl for the algorithm.
    pub particle_sort_clear: wgpu::ComputePipeline,
    pub particle_sort_count: wgpu::ComputePipeline,
    pub particle_sort_compact: wgpu::ComputePipeline,
    pub particle_sort_scan: wgpu::ComputePipeline,
    pub particle_sort_scatter: wgpu::ComputePipeline,
    /// One-substep grace-period swap (snapshots active_block_ids/count into _prev, resets
    /// the current count to 0), dispatched FIRST each substep, before clear/count/compact —
    /// see active_block_swap_main's doc comment in particle_sort.wgsl for why.
    pub active_block_swap: wgpu::ComputePipeline,
    pub grid_clear: wgpu::ComputePipeline,
    pub p2g: wgpu::ComputePipeline,
    /// Multi-field contact (GPU port, first slice) — populates `contact_points` from
    /// each particle's 9-node stencil, gated on grip mass already being nonzero at that
    /// node (written by `p2g` immediately before this runs). See `p2g.wgsl`'s
    /// `gather_contact_points_main` doc for the full rationale.
    pub gather_contact_points: wgpu::ComputePipeline,
    pub grid_update: wgpu::ComputePipeline,
    /// Gather-only: writes v + velocity_gradient. No F update or plasticity.
    pub g2p: wgpu::ComputePipeline,
    /// F update + all plasticity + volume/density + position + boundary (sorted access).
    pub particles_update: wgpu::ComputePipeline,
    /// Post-particles_update: applies non-uniform body forces (gravity wells, Coulomb, etc.).
    pub force_fields: wgpu::ComputePipeline,
    /// Apply velocity impulses directly on GPU particle buffer — no CPU upload needed.
    pub apply_impulses: wgpu::ComputePipeline,
    /// Debug/test-only — runs the Newton-Raphson LR normal fit against one chosen
    /// block's point cloud in isolation. Not part of the real per-substep pipeline.
    /// See `resolve_contact.wgsl`'s `debug_fit_normal_main` doc.
    pub debug_fit_normal: wgpu::ComputePipeline,
    /// Multi-field contact resolution — the real per-substep pass (GPU port). Runs
    /// after grid_update, before g2p. See `resolve_contact.wgsl`'s `resolve_contact_main`
    /// doc.
    pub resolve_contact: wgpu::ComputePipeline,
    /// Day-night/ambient thermal diffusion (GPU port) — 4 passes mirroring CPU's own
    /// `ThermalDiffusion::apply` stages exactly: clear scratch, P2G scalar scatter,
    /// normalize+Laplacian+Newton-cooling, G2P delta-gather. Dispatched over the WHOLE
    /// dense grid every substep when enabled (no active-block optimization -- matches
    /// CPU's own unconditional-dense-grid behavior, real but bounded scope).
    pub thermal_clear: wgpu::ComputePipeline,
    pub thermal_p2g: wgpu::ComputePipeline,
    pub thermal_normalize_laplacian: wgpu::ComputePipeline,
    pub thermal_g2p: wgpu::ComputePipeline,
    /// Resource regrowth (GPU port) — same 4-pass shape as the thermal passes above,
    /// logistic growth as the reaction term instead of Newton cooling.
    pub resource_clear: wgpu::ComputePipeline,
    pub resource_p2g: wgpu::ComputePipeline,
    pub resource_normalize_laplacian: wgpu::ComputePipeline,
    pub resource_g2p: wgpu::ComputePipeline,
    /// ASFLIP (GPU port, Fei et al. 2021) — replaces `g2p` + `particles_update` for a
    /// substep, ONLY dispatched when `SimConfig::asflip_blend > 0.0` (see
    /// `SubstepGates::asflip_active`). Does both passes' jobs fused into one dispatch —
    /// see `g2p_asflip_fused.wgsl`'s own doc for why the fusion is structurally required
    /// (the adaptive position-correction gamma needs the pre-correction velocity to
    /// survive from the gather stage to the position-write stage, and `Particle` has no
    /// spare capacity for a second stored velocity).
    pub g2p_asflip_fused: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Group 1 — contact subsystem, see the module doc comment above for why this is a
    /// second layout rather than more entries in `bind_group_layout`.
    pub contact_bind_group_layout: wgpu::BindGroupLayout,
    /// Group 2 — thermal subsystem, see its own creation site doc for why this is a
    /// third layout.
    pub thermal_bind_group_layout: wgpu::BindGroupLayout,
    /// Group 3 — resource regrowth subsystem, also carries ASFLIP's 2 bindings (see the
    /// module doc comment's Group 3 entry for why they share a group).
    pub resource_bind_group_layout: wgpu::BindGroupLayout,
    /// Separate layout for apply_impulses — only needs particles + impulse_params.
    pub impulse_bind_group_layout: wgpu::BindGroupLayout,
}

/// A `read_write` storage-buffer binding, COMPUTE-visible — the shape shared by every
/// storage entry in the pipeline's bind group layout. Collapses what used to be a ~10-line
/// struct literal repeated 8 times into one call each, cutting real line count (not just
/// moving it) while every binding still gets its own doc comment at the call site.
const fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A `uniform` buffer binding, COMPUTE-visible — same rationale as `storage_entry`.
const fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SimPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mpm_bind_group_layout"),
            entries: &[
                storage_entry(0), // particles
                storage_entry(1), // grid
                uniform_entry(2), // materials (array<MaterialParams, MAX_MATERIALS>)
                uniform_entry(3), // step_params (GpuStepParams, 32 bytes)
                uniform_entry(4), // force_fields_params (GpuFieldsParams, 784 bytes)
                // 5: sorted_particle_ids — written by particle_sort; read by p2g and
                // particles_update for sorted access.
                storage_entry(5),
                // 6: block_counts — 256 atomic<u32>, particle_sort only.
                storage_entry(6),
                // 7: sleep_wake_params — GpuSleepWakeParams, 80 bytes. Only force_fields.wgsl
                // reads this; harmless for shaders that don't.
                uniform_entry(7),
                // 8: active_block_ids — 256 u32. GPU sparse grid: particle_sort writes,
                // grid_clear/grid_update read.
                storage_entry(8),
                // 9: active_block_count — 1 atomic<u32>. Same pair as binding 8.
                storage_entry(9),
                // 10: active_block_ids_prev — 256 u32. Snapshot of last substep's
                // active_block_ids — the one-substep grace period, see active_block_swap_main.
                storage_entry(10),
                // 11: active_block_count_prev — 1 plain u32, not atomic (only ever written by
                // active_block_swap_main's single lid.x==0u thread). Companion to binding 10.
                storage_entry(11),
            ],
        });

        // Group 1 — contact subsystem, split out 2026-07-16 (see module doc comment above)
        // to keep each layout within the WebGPU-guaranteed 8-storage-buffers-per-stage
        // baseline. Binding NUMBERS are kept exactly as they were under the old single
        // layout (12-19) — only which GROUP they belong to changed, so every WGSL shader
        // only needed its `@group(0)` -> `@group(1)` annotation updated on these specific
        // bindings, no renumbering.
        let contact_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mpm_contact_bind_group_layout"),
                entries: &[
                    // 12: grip_grid — multi-field contact "grip" field mass/momentum
                    // accumulator, same dense grid_res² layout and fixed-point atomic
                    // convention as `grid` (group 0 binding 1). GPU port first slice —
                    // see buffers.rs doc.
                    storage_entry(12),
                    // 13: contact_points — labeled contact point cloud (grid_res² ×
                    // MAX_CONTACT_POINTS_PER_NODE), read/written by gather_contact_points_main.
                    storage_entry(13),
                    // 14: contact_point_counts — grid_res² atomic<u32>, per-node point-cloud
                    // size.
                    storage_entry(14),
                    // 15: contact_debug_params — debug/test-only, resolve_contact.wgsl.
                    uniform_entry(15),
                    // 16: contact_debug_output — debug/test-only, resolve_contact.wgsl.
                    storage_entry(16),
                    // 17/18: resolved_grip_v / resolved_rest_v — resolve_contact_main writes,
                    // a future G2P routing change reads.
                    storage_entry(17),
                    storage_entry(18),
                    // 19: grip_params — directional grip friction, resolve_contact.wgsl.
                    uniform_entry(19),
                    // 30/31: material_mass / material_mass_params — `ColorMode::
                    // GridVolume`'s opt-in per-cell per-material mass accumulator
                    // (P2G writes it). Shares this group purely for bind-group-count
                    // economy (WebGPU's 4-group baseline is already fully used, same
                    // reason ASFLIP shares group 3 with resource regrowth) — nothing
                    // to do with contact thematically.
                    storage_entry(30),
                    uniform_entry(31),
                ],
            });

        // Group 2 — day-night/ambient thermal diffusion (GPU port, 2026-07-16). A real,
        // separate group rather than squeezing into group 0 (already at 8/8 storage,
        // zero headroom, per that group's own doc) or group 1 (wrong category — thermal
        // has nothing to do with contact). 3 storage + 1 uniform, well under the
        // baseline limit. Bindings 20-23, continuing the flat numbering the split
        // already established.
        let thermal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mpm_thermal_bind_group_layout"),
                entries: &[
                    // 20: thermal_params — GpuThermalParams (alpha, ambient, cooling_rate,
                    // enabled).
                    uniform_entry(20),
                    // 21: thermal_mass — Σ(w·mass) per cell, dense grid_res² f32.
                    storage_entry(21),
                    // 22: thermal_temp_old — normalized T_old per cell, needed for the G2P
                    // delta gather.
                    storage_entry(22),
                    // 23: thermal_work — dual-use: P2G scatter accumulator, then post-
                    // Laplacian T_new.
                    storage_entry(23),
                ],
            });

        // Group 3 — resource regrowth (GPU port, 2026-07-16). Own separate group from
        // thermal despite the near-identical shape (see `GpuResourceParams`' doc for
        // why: both would otherwise fight over the same particle.temperature carrier).
        // 4 storage + 2 uniform once ASFLIP's 2 bindings are added below, still well
        // under the baseline 8-storage-per-stage limit. Bindings 24-29.
        let resource_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mpm_resource_bind_group_layout"),
                entries: &[
                    // 24: resource_params — GpuResourceParams (diffusivity, ambient,
                    // resource_r, resource_k, enabled).
                    uniform_entry(24),
                    // 25: resource_mass — Σ(w·mass) per cell, dense grid_res² f32.
                    storage_entry(25),
                    // 26: resource_phi_old — normalized φ_old per cell.
                    storage_entry(26),
                    // 27: resource_work — dual-use: P2G scatter accumulator, then post-
                    // Laplacian+logistic-growth φ_new.
                    storage_entry(27),
                    // 28: asflip_params — GpuAsflipParams (blend, enabled). Shares this
                    // group with resource regrowth purely for bind-group-count economy
                    // (WebGPU's 4-group baseline is already fully used) — see the module
                    // doc comment's Group 3 entry.
                    uniform_entry(28),
                    // 29: asflip_snapshot — grid_res² vec2<f32> pre-force velocity
                    // snapshot, written by grid_update.wgsl, read by g2p_asflip_fused.wgsl.
                    storage_entry(29),
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mpm_pipeline_layout"),
            bind_group_layouts: &[
                &bind_group_layout,
                &contact_bind_group_layout,
                &thermal_bind_group_layout,
                &resource_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // MAX_MATERIALS: array-size constant — must be injected via string template
        // (naga requires CREATION_RESOLVED; WGSL `override` doesn't apply to array sizes).
        let p2g_src = patch_shader(shaders::P2G);
        let particles_update_src = patch_shader(shaders::PARTICLES_UPDATE);
        let g2p_asflip_fused_src = patch_shader(shaders::G2P_ASFLIP_FUSED);

        // MAX_FORCE_FIELDS / MAX_SLEEP_WAKE_TAGS: loop-bound constants — uses WGSL
        // `override` (proper pipeline specialization), not a hardcoded literal in the shader.
        let ff_consts: &[(&str, f64)] = &[
            ("MAX_FORCE_FIELDS", MAX_FORCE_FIELDS as f64),
            ("MAX_SLEEP_WAKE_TAGS", MAX_SLEEP_WAKE_TAGS as f64),
        ];

        // NUM_BLOCKS_PER_DIM: GPU sparse grid Phase 1/2 — single Rust-side source of truth,
        // shared by particle_sort's compaction pass and grid_clear/grid_update's block-guarded
        // dispatch.
        let block_consts: &[(&str, f64)] = &[("NUM_BLOCKS_PER_DIM", NUM_BLOCKS_PER_DIM as f64)];
        // NUM_CONTACT_BLOCKS_PER_DIM: dedicated finer contact-point partition (2026-07-18
        // re-partition, see MAX_CONTACT_POINTS_PER_BLOCK's doc in step_params.rs) --
        // separate override from NUM_BLOCKS_PER_DIM above, needed by p2g.wgsl's
        // gather_contact_points_main and resolve_contact.wgsl's gather_local_points/
        // debug_fit_normal_main.
        let contact_block_consts: &[(&str, f64)] = &[(
            "NUM_CONTACT_BLOCKS_PER_DIM",
            NUM_CONTACT_BLOCKS_PER_DIM as f64,
        )];
        // resolve_contact.wgsl declares BOTH overrides (its own NUM_BLOCKS_PER_DIM for
        // resolve_contact_main's active-block iteration, plus NUM_CONTACT_BLOCKS_PER_DIM
        // for gather_local_points' contact-block scan) -- every pipeline built from that
        // module needs both supplied.
        let resolve_contact_consts: &[(&str, f64)] = &[
            ("NUM_BLOCKS_PER_DIM", NUM_BLOCKS_PER_DIM as f64),
            (
                "NUM_CONTACT_BLOCKS_PER_DIM",
                NUM_CONTACT_BLOCKS_PER_DIM as f64,
            ),
        ];
        // grid_update needs BOTH the force-field loop bound AND the block-dispatch constant
        // (Phase 2 — see grid_update.wgsl doc comment).
        let grid_update_consts: &[(&str, f64)] = &[
            ("MAX_FORCE_FIELDS", MAX_FORCE_FIELDS as f64),
            ("MAX_SLEEP_WAKE_TAGS", MAX_SLEEP_WAKE_TAGS as f64),
            ("NUM_BLOCKS_PER_DIM", NUM_BLOCKS_PER_DIM as f64),
        ];

        // NUM_BLOCKS_PER_DIM is an `override` at the particle_sort.wgsl MODULE level (promoted
        // from a hardcoded const — see GPU sparse grid Phase 1), so every pipeline built from
        // this file needs it supplied at creation time, not just particle_sort_compact, which
        // is the only entry point that actually reads it.
        let particle_sort_clear = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_clear_main",
            "particle_sort_clear",
            block_consts,
            false,
        );
        let particle_sort_count = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_count_main",
            "particle_sort_count",
            block_consts,
            false,
        );
        // particle_sort_scan is the ONLY pipeline with var<workgroup> memory (scan_temp) —
        // see the skip_workgroup_zero_init doc on make_pipeline for the safety argument.
        // Every other pipeline keeps the WebGPU-mandated zero-init (false here = default ON).
        // GPU sparse grid Phase 1 — reads the raw histogram before scan overwrites it into a
        // scatter cursor, so must run between count and scan, never reordered.
        let particle_sort_compact = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_compact_main",
            "particle_sort_compact",
            block_consts,
            false,
        );
        // Dispatched FIRST each substep, before clear/count/compact in the per-substep
        // sequence (not the once-per-frame sort sequence) — see active_block_swap_main's doc
        // comment in particle_sort.wgsl for why.
        let active_block_swap = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "active_block_swap_main",
            "active_block_swap",
            block_consts,
            false,
        );
        let particle_sort_scan = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_scan_main",
            "particle_sort_scan",
            block_consts,
            true,
        );
        let particle_sort_scatter = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_scatter_main",
            "particle_sort_scatter",
            block_consts,
            false,
        );
        let grid_clear = make_pipeline(
            device,
            &pipeline_layout,
            shaders::GRID_CLEAR,
            "grid_clear_main",
            "grid_clear",
            block_consts,
            false,
        );
        // p2g.wgsl declares `override NUM_CONTACT_BLOCKS_PER_DIM` (needed by
        // gather_contact_points_main's contact_block_index call) -- both entry points
        // compiled from this same module need it supplied, even though p2g_main itself
        // doesn't reference it.
        let p2g = make_pipeline(
            device,
            &pipeline_layout,
            &p2g_src,
            "p2g_main",
            "p2g",
            contact_block_consts,
            false,
        );
        let gather_contact_points = make_pipeline(
            device,
            &pipeline_layout,
            &p2g_src,
            "gather_contact_points_main",
            "gather_contact_points",
            contact_block_consts,
            false,
        );
        let grid_update = make_pipeline(
            device,
            &pipeline_layout,
            shaders::GRID_UPDATE,
            "grid_update_main",
            "grid_update",
            grid_update_consts,
            false,
        );
        let g2p = make_pipeline(
            device,
            &pipeline_layout,
            shaders::G2P,
            "g2p_main",
            "g2p",
            &[],
            false,
        );
        let particles_update = make_pipeline(
            device,
            &pipeline_layout,
            &particles_update_src,
            "particles_update_main",
            "particles_update",
            &[],
            false,
        );
        let force_fields = make_pipeline(
            device,
            &pipeline_layout,
            shaders::FORCE_FIELDS,
            "force_fields_main",
            "force_fields",
            ff_consts,
            false,
        );

        // ASFLIP (GPU port) -- replaces g2p+particles_update for a substep, only when
        // SimConfig::asflip_blend > 0.0. See g2p_asflip_fused.wgsl's own doc for why this
        // is one fused kernel rather than two, and SimPipelines::g2p_asflip_fused's doc.
        let g2p_asflip_fused = make_pipeline(
            device,
            &pipeline_layout,
            &g2p_asflip_fused_src,
            "g2p_asflip_fused_main",
            "g2p_asflip_fused",
            &[],
            false,
        );

        // Impulse pass has a minimal 2-binding layout: particles + impulse_params.
        let impulse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mpm_impulse_bind_group_layout"),
                entries: &[
                    storage_entry(0), // particles
                    uniform_entry(1), // impulse_params
                ],
            });
        let impulse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mpm_impulse_pipeline_layout"),
                bind_group_layouts: &[&impulse_bind_group_layout],
                push_constant_ranges: &[],
            });
        let apply_impulses = make_pipeline(
            device,
            &impulse_pipeline_layout,
            shaders::APPLY_IMPULSES,
            "apply_impulses_main",
            "apply_impulses",
            &[],
            false,
        );

        // resolve_contact.wgsl declares BOTH `override NUM_BLOCKS_PER_DIM` (needed by
        // resolve_contact_main's active-block-neighbor gather) and
        // `override NUM_CONTACT_BLOCKS_PER_DIM` (needed by gather_local_points' contact-
        // block scan) -- every entry point compiled from this module needs both
        // supplied, even though debug_fit_normal_main itself doesn't reference either
        // (same requirement already established for p2g/gather_contact_points sharing
        // p2g.wgsl's own override).
        let debug_fit_normal = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOLVE_CONTACT,
            "debug_fit_normal_main",
            "debug_fit_normal",
            resolve_contact_consts,
            false,
        );
        let resolve_contact = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOLVE_CONTACT,
            "resolve_contact_main",
            "resolve_contact",
            resolve_contact_consts,
            false,
        );

        // Day-night/ambient thermal diffusion (GPU port) -- 4 passes, see field docs.
        let thermal_clear = make_pipeline(
            device,
            &pipeline_layout,
            shaders::THERMAL,
            "thermal_clear_main",
            "thermal_clear",
            &[],
            false,
        );
        let thermal_p2g = make_pipeline(
            device,
            &pipeline_layout,
            shaders::THERMAL,
            "thermal_p2g_main",
            "thermal_p2g",
            &[],
            false,
        );
        let thermal_normalize_laplacian = make_pipeline(
            device,
            &pipeline_layout,
            shaders::THERMAL,
            "thermal_normalize_laplacian_main",
            "thermal_normalize_laplacian",
            &[],
            false,
        );
        let thermal_g2p = make_pipeline(
            device,
            &pipeline_layout,
            shaders::THERMAL,
            "thermal_g2p_main",
            "thermal_g2p",
            &[],
            false,
        );

        // Resource regrowth (GPU port) -- same 4-pass shape, see field docs.
        let resource_clear = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOURCE_FIELD,
            "resource_clear_main",
            "resource_clear",
            &[],
            false,
        );
        let resource_p2g = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOURCE_FIELD,
            "resource_p2g_main",
            "resource_p2g",
            &[],
            false,
        );
        let resource_normalize_laplacian = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOURCE_FIELD,
            "resource_normalize_laplacian_main",
            "resource_normalize_laplacian",
            &[],
            false,
        );
        let resource_g2p = make_pipeline(
            device,
            &pipeline_layout,
            shaders::RESOURCE_FIELD,
            "resource_g2p_main",
            "resource_g2p",
            &[],
            false,
        );

        Self {
            particle_sort_clear,
            particle_sort_count,
            particle_sort_compact,
            particle_sort_scan,
            particle_sort_scatter,
            active_block_swap,
            grid_clear,
            p2g,
            gather_contact_points,
            grid_update,
            g2p,
            particles_update,
            force_fields,
            apply_impulses,
            debug_fit_normal,
            resolve_contact,
            thermal_clear,
            thermal_p2g,
            thermal_normalize_laplacian,
            thermal_g2p,
            resource_clear,
            resource_p2g,
            resource_normalize_laplacian,
            resource_g2p,
            g2p_asflip_fused,
            bind_group_layout,
            contact_bind_group_layout,
            thermal_bind_group_layout,
            resource_bind_group_layout,
            impulse_bind_group_layout,
        }
    }
}

// Bind-group construction (make_impulse_bind_group, make_bind_group,
// make_contact_bind_group, make_thermal_bind_group, make_resource_bind_group)
// -- split into their own file, was ~200 of this file's ~850 lines. Pipeline/
// layout CONSTRUCTION stays above; per-substep bind-group building lives there.
mod bind_groups;

/// Replaces `{{MAX_MATERIALS}}` with the Rust-side value.
/// Needed because naga requires array-size constants to be CREATION_RESOLVED (known at
/// shader-module creation time), so WGSL `override` constants cannot be used there.
/// MAX_FORCE_FIELDS is a loop bound only — it uses `override` and is handled via constants.
fn patch_shader(source: &str) -> String {
    source.replace("{{MAX_MATERIALS}}", &MAX_MATERIALS.to_string())
}

/// `skip_workgroup_zero_init`: opt-IN per pipeline, NOT a global default. WebGPU mandates
/// zeroing `var<workgroup>` memory before use, as a safety net against reading stale data from
/// a prior dispatch. Pass `true` ONLY if every `var<workgroup>` declared in this specific
/// shader is provably written by every thread before any read (barrier-guarded) — skipping the
/// zero-init then costs nothing in correctness and saves real time (measured: ~10-18% on the
/// one pipeline that currently qualifies, particle_sort_scan). This is NOT compiler-checked —
/// if a future edit to that shader (or a copy-pasted call site for a new shader) adds a
/// `var<workgroup>` without re-verifying the write-before-read invariant, this flag must be
/// re-audited or set back to `false`. Default to `false` for any new pipeline.
fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    entry_point: &str,
    label: &str,
    constants: &[(&str, f64)],
    skip_workgroup_zero_init: bool,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            zero_initialize_workgroup_memory: !skip_workgroup_zero_init,
        },
        cache: None,
    })
}

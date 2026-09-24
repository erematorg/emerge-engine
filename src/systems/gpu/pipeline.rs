/// Compute pipeline setup for MLS-MPM GPU passes.
///
/// See `solver/encode_substep.rs` for the authoritative per-substep dispatch
/// order (this module builds the pipeline objects; it doesn't own the order
/// they're dispatched in). Once per frame: a 4-pass block-level counting
/// sort (`particle_sort_clear → count → scan → scatter`, see
/// `particle_sort.wgsl`). Active-block detection (`particle_sort_compact`,
/// GPU sparse grid Phase 1) runs every substep, not once per frame --
/// particles move every substep, so a once-per-frame version goes stale by
/// substep 2 of a multi-substep step.
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
    MAX_FORCE_FIELDS, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS_PER_DIM, NUM_CONTACT_BLOCKS_PER_DIM,
};

// Bind-group-LAYOUT construction (the four `wgpu::BindGroupLayout`s shared by every
// pass, plus the impulse pass's own minimal layout) -- split into its own file, was
// ~160 of this file's ~730 lines. See layouts.rs's own doc.
mod layouts;
use layouts::{
    build_contact_bind_group_layout, build_core_bind_group_layout, build_impulse_bind_group_layout,
    build_resource_bind_group_layout, build_thermal_bind_group_layout,
};

// Compute-PIPELINE construction (the `wgpu::ComputePipeline`s built from those
// layouts, grouped the same way the module doc comment above already groups the
// eleven passes) -- split into its own file, was ~280 of this file's ~730 lines.
// See passes.rs's own doc.
mod passes;
use passes::{
    build_asflip_pipeline, build_contact_resolve_pipelines, build_g2p_and_update_pipelines,
    build_impulse_pipeline, build_p2g_and_grid_pipelines, build_resource_pipelines,
    build_sort_pipelines, build_thermal_pipelines,
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

impl SimPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = build_core_bind_group_layout(device);
        let contact_bind_group_layout = build_contact_bind_group_layout(device);
        let thermal_bind_group_layout = build_thermal_bind_group_layout(device);
        let resource_bind_group_layout = build_resource_bind_group_layout(device);

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

        let (
            particle_sort_clear,
            particle_sort_count,
            particle_sort_compact,
            active_block_swap,
            particle_sort_scan,
            particle_sort_scatter,
        ) = build_sort_pipelines(device, &pipeline_layout, block_consts);

        let (grid_clear, p2g, gather_contact_points, grid_update) = build_p2g_and_grid_pipelines(
            device,
            &pipeline_layout,
            block_consts,
            contact_block_consts,
            grid_update_consts,
        );

        let (g2p, particles_update, force_fields) =
            build_g2p_and_update_pipelines(device, &pipeline_layout, ff_consts);

        // ASFLIP (GPU port) -- replaces g2p+particles_update for a substep, only when
        // SimConfig::asflip_blend > 0.0. See g2p_asflip_fused.wgsl's own doc for why this
        // is one fused kernel rather than two, and SimPipelines::g2p_asflip_fused's doc.
        let g2p_asflip_fused = build_asflip_pipeline(device, &pipeline_layout);

        let impulse_bind_group_layout = build_impulse_bind_group_layout(device);
        let apply_impulses = build_impulse_pipeline(device, &impulse_bind_group_layout);

        let (debug_fit_normal, resolve_contact) =
            build_contact_resolve_pipelines(device, &pipeline_layout, resolve_contact_consts);

        // Day-night/ambient thermal diffusion (GPU port) -- 4 passes, see field docs.
        let (thermal_clear, thermal_p2g, thermal_normalize_laplacian, thermal_g2p) =
            build_thermal_pipelines(device, &pipeline_layout);

        // Resource regrowth (GPU port) -- same 4-pass shape, see field docs.
        let (resource_clear, resource_p2g, resource_normalize_laplacian, resource_g2p) =
            build_resource_pipelines(device, &pipeline_layout);

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

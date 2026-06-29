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
///     1. grid_clear       — zero only cells in active blocks (see grid_clear.wgsl)
///     2. p2g              — scatter particles → grid (sorted access, 64-wide workgroups)
///     3. grid_update      — normalize momentum→velocity, apply gravity, enforce boundary
///     4. g2p              — gather grid → particles, write v + velocity_gradient only
///     5. particles_update — F update, plasticity, volume/density, position, boundary (sorted)
///     6. force_fields     — apply non-uniform body forces after particles_update
///
/// Single bind group layout shared by all passes:
///   binding 0: particles            — storage read_write
///   binding 1: grid                 — storage read_write
///   binding 2: materials            — uniform (array<MaterialParams, MAX_MATERIALS>)
///   binding 3: step_params          — uniform (GpuStepParams, 32 bytes)
///   binding 4: force_fields_params  — uniform (GpuFieldsParams, 784 bytes)
///   binding 5: sorted_particle_ids  — storage read_write (u32 per particle)
///   binding 6: block_counts         — storage read_write (256 atomic<u32>, particle_sort only)
///   binding 7: sleep_wake_params    — uniform (GpuSleepWakeParams, 80 bytes)
///   binding 8: active_block_ids     — storage read_write (256 u32 — particle_sort writes,
///                                     grid_clear reads; GPU sparse grid Phase 1)
///   binding 9: active_block_count   — storage read_write (1 atomic<u32> — same pair as above)
///
/// Passes that don't use a binding still share the same layout — avoids rebinding.
use super::buffers::GpuBuffers;
use super::shaders;
use super::step_params::{
    MAX_FORCE_FIELDS, MAX_MATERIALS, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS_PER_DIM,
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
    pub grid_update: wgpu::ComputePipeline,
    /// Gather-only: writes v + velocity_gradient. No F update or plasticity.
    pub g2p: wgpu::ComputePipeline,
    /// F update + all plasticity + volume/density + position + boundary (sorted access).
    pub particles_update: wgpu::ComputePipeline,
    /// Post-particles_update: applies non-uniform body forces (gravity wells, Coulomb, etc.).
    pub force_fields: wgpu::ComputePipeline,
    /// Apply velocity impulses directly on GPU particle buffer — no CPU upload needed.
    pub apply_impulses: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Separate layout for apply_impulses — only needs particles + impulse_params.
    pub impulse_bind_group_layout: wgpu::BindGroupLayout,
}

impl SimPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mpm_bind_group_layout"),
            entries: &[
                // binding 0: particles — storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: grid — storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: materials — uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: step_params — uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: force_fields_params — uniform (GpuFieldsParams, 784 bytes)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: sorted_particle_ids — storage read_write (u32 per particle)
                // Written by particle_sort; read by p2g and particles_update for sorted access.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: block_counts — storage read_write (256 atomic<u32>, particle_sort only)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: sleep_wake_params — uniform (GpuSleepWakeParams, 80 bytes)
                // Only force_fields.wgsl reads this; harmless for shaders that don't.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 8: active_block_ids — storage read_write (256 u32). GPU sparse grid
                // Phase 1: particle_sort writes, grid_clear reads.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 9: active_block_count — storage read_write (1 atomic<u32>). Same
                // pair as binding 8.
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 10: active_block_ids_prev — storage read_write (256 u32). Snapshot
                // of last substep's active_block_ids — the one-substep grace period, see
                // active_block_swap_main.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 11: active_block_count_prev — storage read_write (1 plain u32, not
                // atomic — only ever written by active_block_swap_main's single lid.x==0u
                // thread). Companion to binding 10.
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mpm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // MAX_MATERIALS: array-size constant — must be injected via string template
        // (naga requires CREATION_RESOLVED; WGSL `override` doesn't apply to array sizes).
        let p2g_src = patch_shader(shaders::P2G);
        let particles_update_src = patch_shader(shaders::PARTICLES_UPDATE);

        // MAX_FORCE_FIELDS / MAX_SLEEP_WAKE_TAGS: loop-bound constants — uses WGSL
        // `override` (proper pipeline specialization), not a hardcoded literal in the shader.
        let ff_consts: &[(&str, f64)] = &[
            ("MAX_FORCE_FIELDS", MAX_FORCE_FIELDS as f64),
            ("MAX_SLEEP_WAKE_TAGS", MAX_SLEEP_WAKE_TAGS as f64),
        ];

        // NUM_BLOCKS_PER_DIM: GPU sparse grid Phase 1 — single Rust-side source of truth,
        // shared by particle_sort's compaction pass and grid_clear's block-guarded dispatch.
        let block_consts: &[(&str, f64)] = &[("NUM_BLOCKS_PER_DIM", NUM_BLOCKS_PER_DIM as f64)];

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
        let p2g = make_pipeline(
            device,
            &pipeline_layout,
            &p2g_src,
            "p2g_main",
            "p2g",
            &[],
            false,
        );
        let grid_update = make_pipeline(
            device,
            &pipeline_layout,
            shaders::GRID_UPDATE,
            "grid_update_main",
            "grid_update",
            ff_consts,
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

        // Impulse pass has a minimal 2-binding layout: particles + impulse_params.
        let impulse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mpm_impulse_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        Self {
            particle_sort_clear,
            particle_sort_count,
            particle_sort_compact,
            particle_sort_scan,
            particle_sort_scatter,
            active_block_swap,
            grid_clear,
            p2g,
            grid_update,
            g2p,
            particles_update,
            force_fields,
            apply_impulses,
            bind_group_layout,
            impulse_bind_group_layout,
        }
    }

    /// Build a bind group for the apply_impulses pass (particles + impulse_params).
    /// Created on-demand each cursor frame — cheap, no GPU work.
    pub fn make_impulse_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_impulse_bind_group"),
            layout: &self.impulse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.impulse_params.as_entire_binding(),
                },
            ],
        })
    }

    /// Build a bind group for one substep using the given step_params buffer slot.
    /// Cheap — wgpu bind groups are descriptor tables, not data copies.
    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
        step_params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: step_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.force_fields_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.sorted_particle_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.block_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.sleep_wake_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.active_block_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.active_block_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.active_block_ids_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.active_block_count_prev.as_entire_binding(),
                },
            ],
        })
    }
}

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

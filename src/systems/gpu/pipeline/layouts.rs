//! Bind-group-LAYOUT construction for `SimPipelines::new` -- split out of
//! `pipeline.rs` (was ~160 of its ~730 lines). Same reasoning as the
//! `bind_groups.rs` split: this file only builds the (device-independent-of-
//! particle-count) `wgpu::BindGroupLayout`s; the actual per-substep
//! `wgpu::BindGroup`s built from them live in `bind_groups.rs`, and the
//! compute-pipeline objects that share these layouts live in `passes.rs`.
//!
//! The four-group split itself (and why binding numbers land where they do)
//! is the module-level doc comment on `pipeline.rs` -- that's the load-bearing
//! story (WebGPU's baseline 8-storage-buffers-per-stage and 4-bind-groups
//! limits), kept there since it explains the whole file, not just this one.
//! The per-group comments below explain each individual binding.

/// A `read_write` storage-buffer binding, COMPUTE-visible -- the shape shared by every
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

/// A `uniform` buffer binding, COMPUTE-visible -- same rationale as `storage_entry`.
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

/// Group 0 -- core MPM state, needed by nearly every pass. See the module doc
/// comment on `pipeline.rs` for the full binding table and why this group is
/// at the baseline storage-buffer limit with zero headroom.
pub(super) fn build_core_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mpm_bind_group_layout"),
        entries: &[
            storage_entry(0), // particles
            storage_entry(1), // grid
            uniform_entry(2), // materials (array<MaterialParams, MAX_MATERIALS>)
            uniform_entry(3), // step_params (GpuStepParams, 32 bytes)
            uniform_entry(4), // force_fields_params (GpuFieldsParams, 784 bytes)
            // 5: sorted_particle_ids -- written by particle_sort; read by p2g and
            // particles_update for sorted access.
            storage_entry(5),
            // 6: block_counts -- 256 atomic<u32>, particle_sort only.
            storage_entry(6),
            // 7: sleep_wake_params -- GpuSleepWakeParams, 80 bytes. Only force_fields.wgsl
            // reads this; harmless for shaders that don't.
            uniform_entry(7),
            // 8: active_block_ids -- 256 u32. GPU sparse grid: particle_sort writes,
            // grid_clear/grid_update read.
            storage_entry(8),
            // 9: active_block_count -- 1 atomic<u32>. Same pair as binding 8.
            storage_entry(9),
            // 10: active_block_ids_prev -- 256 u32. Snapshot of last substep's
            // active_block_ids -- the one-substep grace period, see active_block_swap_and_clear_main.
            storage_entry(10),
            // 11: active_block_count_prev -- 1 plain u32, not atomic (only ever written by
            // active_block_swap_and_clear_main's single lid.x==0u thread). Companion to binding 10.
            storage_entry(11),
        ],
    })
}

/// Group 1 -- contact subsystem, split out 2026-07-16 (see the `pipeline.rs` module doc
/// comment) to keep each layout within the WebGPU-guaranteed 8-storage-buffers-per-stage
/// baseline. Binding NUMBERS are kept exactly as they were under the old single layout
/// (12-19) -- only which GROUP they belong to changed, so every WGSL shader only needed
/// its `@group(0)` -> `@group(1)` annotation updated on these specific bindings, no
/// renumbering.
pub(super) fn build_contact_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mpm_contact_bind_group_layout"),
        entries: &[
            // 12: grip_grid -- multi-field contact "grip" field mass/momentum
            // accumulator, same dense grid_res² layout and fixed-point atomic
            // convention as `grid` (group 0 binding 1). GPU port first slice --
            // see buffers.rs doc.
            storage_entry(12),
            // 13: contact_points -- labeled contact point cloud (grid_res² ×
            // MAX_CONTACT_POINTS_PER_NODE), read/written by gather_contact_points_main.
            storage_entry(13),
            // 14: contact_point_counts -- grid_res² atomic<u32>, per-node point-cloud
            // size.
            storage_entry(14),
            // 15: contact_debug_params -- debug/test-only, resolve_contact.wgsl.
            uniform_entry(15),
            // 16: contact_debug_output -- debug/test-only, resolve_contact.wgsl.
            storage_entry(16),
            // 17/18: resolved_grip_v / resolved_rest_v -- resolve_contact_main writes,
            // a future G2P routing change reads.
            storage_entry(17),
            storage_entry(18),
            // 19: grip_params -- directional grip friction, resolve_contact.wgsl.
            uniform_entry(19),
            // 30/31: material_mass / material_mass_params -- `ColorMode::
            // GridVolume`'s opt-in per-cell per-material mass accumulator
            // (P2G writes it). Shares this group purely for bind-group-count
            // economy (WebGPU's 4-group baseline is already fully used, same
            // reason ASFLIP shares group 3 with resource regrowth) -- nothing
            // to do with contact thematically.
            storage_entry(30),
            uniform_entry(31),
        ],
    })
}

/// Group 2 -- day-night/ambient thermal diffusion (GPU port, 2026-07-16). A real,
/// separate group rather than squeezing into group 0 (already at 8/8 storage,
/// zero headroom, per that group's own doc) or group 1 (wrong category -- thermal
/// has nothing to do with contact). 3 storage + 1 uniform, well under the
/// baseline limit. Bindings 20-23, continuing the flat numbering the split
/// already established.
pub(super) fn build_thermal_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mpm_thermal_bind_group_layout"),
        entries: &[
            // 20: thermal_params -- GpuThermalParams (alpha, ambient, cooling_rate,
            // enabled).
            uniform_entry(20),
            // 21: thermal_mass -- Σ(w·mass) per cell, dense grid_res² f32.
            storage_entry(21),
            // 22: thermal_temp_old -- normalized T_old per cell, needed for the G2P
            // delta gather.
            storage_entry(22),
            // 23: thermal_work -- dual-use: P2G scatter accumulator, then post-
            // Laplacian T_new.
            storage_entry(23),
            // adaptive_dt -- the GPU's own per-substep timestep state (see buffers.rs).
            // Lives in this group purely for bind-group economy: group 0's storage slots
            // are full and this one has room, same reason ASFLIP shares group 3.
            storage_entry(37),
        ],
    })
}

/// Group 3 -- resource regrowth (GPU port, 2026-07-16). Own separate group from
/// thermal despite the near-identical shape (see `GpuResourceParams`' doc for
/// why: both would otherwise fight over the same particle.temperature carrier).
/// 4 storage + 2 uniform once ASFLIP's 2 bindings are added below, still well
/// under the baseline 8-storage-per-stage limit. Bindings 24-29.
pub(super) fn build_resource_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mpm_resource_bind_group_layout"),
        entries: &[
            // 24: resource_params -- GpuResourceParams (diffusivity, ambient,
            // resource_r, resource_k, enabled).
            uniform_entry(24),
            // 25: resource_mass -- Σ(w·mass) per cell, dense grid_res² f32.
            storage_entry(25),
            // 26: resource_phi_old -- normalized φ_old per cell.
            storage_entry(26),
            // 27: resource_work -- dual-use: P2G scatter accumulator, then post-
            // Laplacian+logistic-growth φ_new.
            storage_entry(27),
            // 28: asflip_params -- GpuAsflipParams (blend, enabled). Shares this
            // group with resource regrowth purely for bind-group-count economy
            // (WebGPU's 4-group baseline is already fully used) -- see the module
            // doc comment's Group 3 entry.
            uniform_entry(28),
            // 29: asflip_snapshot -- grid_res² vec2<f32> pre-force velocity
            // snapshot, written by grid_update.wgsl, read by g2p_asflip_fused.wgsl.
            storage_entry(29),
            // 32-36: real GPU port of the CPU-proven Chorin-style fluid
            // incompressibility pressure projection (`fluid_pressure.wgsl`,
            // see its own module doc for the full real algorithm and
            // citations). Shares this group for the same bind-group-count
            // economy reason as ASFLIP/resource regrowth above -- nothing
            // thematically related, group 3 is simply the one with real
            // storage-slot headroom (4 free of 8 before this addition,
            // exactly used up by this feature). 31 is already taken by
            // group 1's `material_mass_params`, so this starts at 32.
            uniform_entry(32), // fluid_pressure_params
            storage_entry(33), // fp_divergence
            storage_entry(34), // fp_pressure_a
            storage_entry(35), // fp_pressure_b
            storage_entry(36), // fp_is_surface
        ],
    })
}

/// Impulse pass has a minimal 2-binding layout: particles + impulse_params.
/// A separate, dedicated `wgpu::PipelineLayout` (built alongside the pipeline
/// itself in `passes::build_impulse_pipeline`) rather than reusing the main
/// 4-group `pipeline_layout` -- apply_impulses only ever touches these two
/// buffers.
pub(super) fn build_impulse_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mpm_impulse_bind_group_layout"),
        entries: &[
            storage_entry(0), // particles
            uniform_entry(1), // impulse_params
        ],
    })
}

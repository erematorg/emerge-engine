//! Compute-pipeline construction for `SimPipelines::new` -- split out of
//! `pipeline.rs` (was ~280 of its ~730 lines). Bind-group-LAYOUT construction
//! lives in `layouts.rs`; this file only builds the `wgpu::ComputePipeline`s
//! that share those layouts, grouped the same way the `pipeline.rs` module
//! doc comment already describes the eleven passes: once-per-frame sort
//! passes, then the per-substep MPM passes, then each later-added subsystem
//! (contact resolution, thermal, resource regrowth) gets its own small group.
//!
//! Every grouped builder function here takes the shared `&wgpu::PipelineLayout`
//! (built in `SimPipelines::new` from the four bind-group layouts) plus
//! whichever WGSL `override` constants its shaders need, and returns a tuple
//! of the pipelines it built, in the same order `SimPipelines::new` destructures
//! them -- mirrors `systems::render::pipelines`' `(Pipeline, BindGroupLayout)`
//! tuple style, just wider tuples since one pipeline_layout is shared by many
//! more passes here.
use super::super::step_params::MAX_MATERIALS;
use super::shaders;

/// Replaces `{{MAX_MATERIALS}}` with the Rust-side value.
/// Needed because naga requires array-size constants to be CREATION_RESOLVED (known at
/// shader-module creation time), so WGSL `override` constants cannot be used there.
/// MAX_FORCE_FIELDS is a loop bound only -- it uses `override` and is handled via constants.
fn patch_shader(source: &str) -> String {
    source.replace("{{MAX_MATERIALS}}", &MAX_MATERIALS.to_string())
}

/// `skip_workgroup_zero_init`: opt-IN per pipeline, NOT a global default. WebGPU mandates
/// zeroing `var<workgroup>` memory before use, as a safety net against reading stale data from
/// a prior dispatch. Pass `true` ONLY if every `var<workgroup>` declared in this specific
/// shader is provably written by every thread before any read (barrier-guarded) -- skipping the
/// zero-init then costs nothing in correctness and saves real time (measured: ~10-18% on the
/// one pipeline that currently qualifies, particle_sort_scan). This is NOT compiler-checked --
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

/// Once per frame, in order (block-level counting sort, see particle_sort.wgsl): clear
/// histogram -> count per-block -> compact (active-block list, GPU sparse grid Phase 1)
/// -> scan (exclusive prefix sum) -> scatter into sorted_particle_ids.
/// `active_block_swap_and_clear` lives in the same shader module but is dispatched FIRST
/// each substep, before count/compact -- see the swap pass's doc in particle_sort.wgsl. All
/// six need `block_consts` (`NUM_BLOCKS_PER_DIM`) supplied, even entry points that don't read it
/// -- NUM_BLOCKS_PER_DIM is an `override` at the particle_sort.wgsl MODULE level.
pub(super) fn build_sort_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    block_consts: &[(&str, f64)],
) -> (
    wgpu::ComputePipeline, // particle_sort_clear
    wgpu::ComputePipeline, // particle_sort_count
    wgpu::ComputePipeline, // particle_sort_compact
    wgpu::ComputePipeline, // active_block_swap_and_clear
    wgpu::ComputePipeline, // contact_counts_clear
    wgpu::ComputePipeline, // particle_sort_scan
    wgpu::ComputePipeline, // particle_sort_scatter
) {
    let particle_sort_clear = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "particle_sort_clear_main",
        "particle_sort_clear",
        block_consts,
        false,
    );
    let particle_sort_count = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "particle_sort_count_main",
        "particle_sort_count",
        block_consts,
        false,
    );
    // GPU sparse grid Phase 1 -- reads the raw histogram before scan overwrites it into a
    // scatter cursor, so must run between count and scan, never reordered.
    let particle_sort_compact = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "particle_sort_compact_main",
        "particle_sort_compact",
        block_consts,
        false,
    );
    // Dispatched FIRST each substep, before clear/count/compact in the per-substep
    // sequence (not the once-per-frame sort sequence) -- see active_block_swap_and_clear_main's doc
    // comment in particle_sort.wgsl for why.
    let active_block_swap_and_clear = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "active_block_swap_and_clear_main",
        "active_block_swap_and_clear",
        block_consts,
        false,
    );
    // particle_sort_scan is the ONLY pipeline with var<workgroup> memory (scan_temp) --
    // see the skip_workgroup_zero_init doc on make_pipeline for the safety argument.
    // Every other pipeline keeps the WebGPU-mandated zero-init (false here = default ON).
    let contact_counts_clear = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "contact_counts_clear_main",
        "contact_counts_clear",
        block_consts,
        false,
    );
    let particle_sort_scan = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "particle_sort_scan_main",
        "particle_sort_scan",
        block_consts,
        true,
    );
    let particle_sort_scatter = make_pipeline(
        device,
        layout,
        shaders::PARTICLE_SORT,
        "particle_sort_scatter_main",
        "particle_sort_scatter",
        block_consts,
        false,
    );

    (
        particle_sort_clear,
        particle_sort_count,
        particle_sort_compact,
        active_block_swap_and_clear,
        contact_counts_clear,
        particle_sort_scan,
        particle_sort_scatter,
    )
}

/// grid_clear -> p2g (+ gather_contact_points) -> grid_update, the first half of the
/// per-substep MPM sequence. `p2g.wgsl` declares `override NUM_CONTACT_BLOCKS_PER_DIM`
/// (needed by `gather_contact_points_main`'s contact_block_index call) -- both entry
/// points compiled from that module need it supplied, even though p2g_main itself
/// doesn't reference it. grid_update needs BOTH the force-field loop bound AND the
/// block-dispatch constant (Phase 2 -- see grid_update.wgsl doc comment), passed in via
/// `grid_update_consts`.
pub(super) fn build_p2g_and_grid_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    grid_clear_consts: &[(&str, f64)],
    contact_block_consts: &[(&str, f64)],
    grid_update_consts: &[(&str, f64)],
) -> (
    wgpu::ComputePipeline, // grid_clear
    wgpu::ComputePipeline, // p2g
    wgpu::ComputePipeline, // gather_contact_points
    wgpu::ComputePipeline, // grid_update
) {
    // MAX_MATERIALS: array-size constant -- must be injected via string template
    // (naga requires CREATION_RESOLVED; WGSL `override` doesn't apply to array sizes).
    let p2g_src = patch_shader(shaders::P2G);

    let grid_clear = make_pipeline(
        device,
        layout,
        shaders::GRID_CLEAR,
        "grid_clear_main",
        "grid_clear",
        grid_clear_consts,
        false,
    );
    let p2g = make_pipeline(
        device,
        layout,
        &p2g_src,
        "p2g_main",
        "p2g",
        contact_block_consts,
        // p2g_main writes every word of its workgroup tile (and the tile origin) before a
        // barrier, ahead of any read -- see its own "no early return" setup block.
        true,
    );
    // Multi-field contact (GPU port, first slice) -- populates `contact_points` from
    // each particle's 9-node stencil, gated on grip mass already being nonzero at that
    // node (written by `p2g` immediately before this runs). See `p2g.wgsl`'s
    // `gather_contact_points_main` doc for the full rationale.
    let gather_contact_points = make_pipeline(
        device,
        layout,
        &p2g_src,
        "gather_contact_points_main",
        "gather_contact_points",
        contact_block_consts,
        false,
    );
    let grid_update = make_pipeline(
        device,
        layout,
        shaders::GRID_UPDATE,
        "grid_update_main",
        "grid_update",
        grid_update_consts,
        false,
    );

    (grid_clear, p2g, gather_contact_points, grid_update)
}

/// The fused per-particle substep tail `g2p_update` (G2P gather -> F update/plasticity/
/// position -> force fields, one dispatch), specialized to the material models in
/// `models_present` (`MaterialRegistry::model_mask`; `u32::MAX` = generic).
pub(super) fn build_g2p_update_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    ff_consts: &[(&str, f64)],
    models_present: u32,
) -> wgpu::ComputePipeline {
    // MAX_MATERIALS: array-size constant, same rationale as p2g_src above. The gather and
    // force-field functions are shared WGSL snippets appended to their host module (WGSL
    // has no includes; module-scope declarations are order-independent).
    let g2p_update_src = patch_shader(&format!(
        "{}
{}
{}",
        shaders::PARTICLES_UPDATE,
        shaders::G2P_GATHER_INC,
        shaders::FORCE_FIELDS_APPLY_INC
    ));
    let mut consts = ff_consts.to_vec();
    consts.push(("MODELS_PRESENT", f64::from(models_present)));
    make_pipeline(
        device,
        layout,
        &g2p_update_src,
        "g2p_update_main",
        "g2p_update",
        &consts,
        false,
    )
}

/// The GPU's own per-substep CFL: turns the minimum bound the particles folded into
/// `adaptive_dt[2]` this substep into the next substep's timestep. See adaptive_cfl.wgsl.
pub(super) fn build_cfl_commit_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    make_pipeline(
        device,
        layout,
        shaders::ADAPTIVE_CFL,
        "cfl_commit_main",
        "cfl_commit",
        &[],
        false,
    )
}

/// Standalone force fields + sleep/wake, used only after the ASFLIP fused G2P (which
/// replaces `g2p_update`'s gather+update but not its force-field stage).
pub(super) fn build_force_fields_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    ff_consts: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let force_fields_src = format!(
        "{}
{}",
        shaders::FORCE_FIELDS,
        shaders::FORCE_FIELDS_APPLY_INC
    );
    make_pipeline(
        device,
        layout,
        &force_fields_src,
        "force_fields_main",
        "force_fields",
        ff_consts,
        false,
    )
}

/// ASFLIP (GPU port, Fei et al. 2021) -- replaces g2p+particles_update for a substep,
/// only when `SimConfig::asflip_blend > 0.0`. See `g2p_asflip_fused.wgsl`'s own doc for
/// why this is one fused kernel rather than two, and `SimPipelines::g2p_asflip_fused`'s
/// own field doc.
pub(super) fn build_asflip_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let g2p_asflip_fused_src = patch_shader(shaders::G2P_ASFLIP_FUSED);
    make_pipeline(
        device,
        layout,
        &g2p_asflip_fused_src,
        "g2p_asflip_fused_main",
        "g2p_asflip_fused",
        &[],
        false,
    )
}

/// Impulse pass -- a dedicated single-pipeline layout (particles + impulse_params only,
/// built in `layouts::build_impulse_bind_group_layout`), separate from the main
/// 4-group `pipeline_layout` shared by every other pass in this file.
pub(super) fn build_impulse_pipeline(
    device: &wgpu::Device,
    impulse_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let impulse_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mpm_impulse_pipeline_layout"),
        bind_group_layouts: &[impulse_bind_group_layout],
        push_constant_ranges: &[],
    });
    make_pipeline(
        device,
        &impulse_pipeline_layout,
        shaders::APPLY_IMPULSES,
        "apply_impulses_main",
        "apply_impulses",
        &[],
        false,
    )
}

/// Multi-field contact resolution -- `debug_fit_normal` (debug/test-only, runs the
/// Newton-Raphson LR normal fit against one chosen block's point cloud in isolation)
/// and `resolve_contact` (the real per-substep pass, runs after grid_update, before
/// g2p). `resolve_contact.wgsl` declares BOTH `override NUM_BLOCKS_PER_DIM` (needed by
/// resolve_contact_main's active-block-neighbor gather) and `override
/// NUM_CONTACT_BLOCKS_PER_DIM` (needed by gather_local_points' contact-block scan) --
/// every entry point compiled from this module needs both supplied, even though
/// debug_fit_normal_main itself doesn't reference either (same requirement already
/// established for p2g/gather_contact_points sharing p2g.wgsl's own override).
pub(super) fn build_contact_resolve_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    resolve_contact_consts: &[(&str, f64)],
) -> (
    wgpu::ComputePipeline, // debug_fit_normal
    wgpu::ComputePipeline, // resolve_contact
) {
    let debug_fit_normal = make_pipeline(
        device,
        layout,
        shaders::RESOLVE_CONTACT,
        "debug_fit_normal_main",
        "debug_fit_normal",
        resolve_contact_consts,
        false,
    );
    let resolve_contact = make_pipeline(
        device,
        layout,
        shaders::RESOLVE_CONTACT,
        "resolve_contact_main",
        "resolve_contact",
        resolve_contact_consts,
        false,
    );

    (debug_fit_normal, resolve_contact)
}

/// Day-night/ambient thermal diffusion (GPU port) -- 4 passes mirroring CPU's own
/// `ThermalDiffusion::apply` stages exactly: clear scratch, P2G scalar scatter,
/// normalize+Laplacian+Newton-cooling, G2P delta-gather. None of these entry points
/// take WGSL `override` constants.
pub(super) fn build_thermal_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> (
    wgpu::ComputePipeline, // thermal_clear
    wgpu::ComputePipeline, // thermal_p2g
    wgpu::ComputePipeline, // thermal_normalize_laplacian
    wgpu::ComputePipeline, // thermal_g2p
) {
    let thermal_clear = make_pipeline(
        device,
        layout,
        shaders::THERMAL,
        "thermal_clear_main",
        "thermal_clear",
        &[],
        false,
    );
    let thermal_p2g = make_pipeline(
        device,
        layout,
        shaders::THERMAL,
        "thermal_p2g_main",
        "thermal_p2g",
        &[],
        false,
    );
    let thermal_normalize_laplacian = make_pipeline(
        device,
        layout,
        shaders::THERMAL,
        "thermal_normalize_laplacian_main",
        "thermal_normalize_laplacian",
        &[],
        false,
    );
    let thermal_g2p = make_pipeline(
        device,
        layout,
        shaders::THERMAL,
        "thermal_g2p_main",
        "thermal_g2p",
        &[],
        false,
    );

    (
        thermal_clear,
        thermal_p2g,
        thermal_normalize_laplacian,
        thermal_g2p,
    )
}

/// Resource regrowth (GPU port) -- same 4-pass shape as the thermal passes above,
/// logistic growth as the reaction term instead of Newton cooling.
pub(super) fn build_resource_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> (
    wgpu::ComputePipeline, // resource_clear
    wgpu::ComputePipeline, // resource_p2g
    wgpu::ComputePipeline, // resource_normalize_laplacian
    wgpu::ComputePipeline, // resource_g2p
) {
    let resource_clear = make_pipeline(
        device,
        layout,
        shaders::RESOURCE_FIELD,
        "resource_clear_main",
        "resource_clear",
        &[],
        false,
    );
    let resource_p2g = make_pipeline(
        device,
        layout,
        shaders::RESOURCE_FIELD,
        "resource_p2g_main",
        "resource_p2g",
        &[],
        false,
    );
    let resource_normalize_laplacian = make_pipeline(
        device,
        layout,
        shaders::RESOURCE_FIELD,
        "resource_normalize_laplacian_main",
        "resource_normalize_laplacian",
        &[],
        false,
    );
    let resource_g2p = make_pipeline(
        device,
        layout,
        shaders::RESOURCE_FIELD,
        "resource_g2p_main",
        "resource_g2p",
        &[],
        false,
    );

    (
        resource_clear,
        resource_p2g,
        resource_normalize_laplacian,
        resource_g2p,
    )
}

/// Real GPU port of the CPU-proven Chorin-style fluid incompressibility
/// pressure projection -- see `fluid_pressure.wgsl`'s own module doc for the
/// full real algorithm, citations, and why Jacobi (not CPU's exact DCT
/// solve) is the right GPU technique. 4 pipelines: divergence+classification
/// setup, the two alternating Jacobi sweep directions, and the final
/// per-cell-mass momentum correction.
pub(super) fn build_fluid_pressure_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> (
    wgpu::ComputePipeline, // fluid_pressure_setup
    wgpu::ComputePipeline, // fluid_pressure_jacobi_a_to_b
    wgpu::ComputePipeline, // fluid_pressure_jacobi_b_to_a
    wgpu::ComputePipeline, // fluid_pressure_correct
) {
    let fluid_pressure_setup = make_pipeline(
        device,
        layout,
        shaders::FLUID_PRESSURE,
        "fluid_pressure_setup_main",
        "fluid_pressure_setup",
        &[],
        false,
    );
    let fluid_pressure_jacobi_a_to_b = make_pipeline(
        device,
        layout,
        shaders::FLUID_PRESSURE,
        "fluid_pressure_jacobi_a_to_b_main",
        "fluid_pressure_jacobi_a_to_b",
        &[],
        false,
    );
    let fluid_pressure_jacobi_b_to_a = make_pipeline(
        device,
        layout,
        shaders::FLUID_PRESSURE,
        "fluid_pressure_jacobi_b_to_a_main",
        "fluid_pressure_jacobi_b_to_a",
        &[],
        false,
    );
    let fluid_pressure_correct = make_pipeline(
        device,
        layout,
        shaders::FLUID_PRESSURE,
        "fluid_pressure_correct_main",
        "fluid_pressure_correct",
        &[],
        false,
    );

    (
        fluid_pressure_setup,
        fluid_pressure_jacobi_a_to_b,
        fluid_pressure_jacobi_b_to_a,
        fluid_pressure_correct,
    )
}

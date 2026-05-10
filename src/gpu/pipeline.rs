/// Compute pipeline setup for MLS-MPM GPU passes.
///
/// Seven passes per frame/substep:
///   Once per frame:
///     0. particle_sort    — write identity permutation to sorted_particle_ids (placeholder GPU sort)
///   Per substep:
///     1. grid_clear       — zero all grid cells (one thread per cell, 8×8 workgroups)
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
///   binding 4: force_fields_params  — uniform (GpuForceFieldsParams, 784 bytes)
///   binding 5: sorted_particle_ids  — storage read_write (u32 per particle)
///
/// Passes that don't use a binding still share the same layout — avoids rebinding.
use super::buffers::GpuBuffers;
use super::shaders;
use super::step_params::{MAX_FORCE_FIELDS, MAX_MATERIALS};

/// All compiled compute pipelines for one GpuSolver instance.
pub struct MpmPipelines {
    /// Once per frame: initializes sorted_particle_ids to identity permutation.
    pub particle_sort: wgpu::ComputePipeline,
    pub grid_clear: wgpu::ComputePipeline,
    pub p2g: wgpu::ComputePipeline,
    pub grid_update: wgpu::ComputePipeline,
    /// Gather-only: writes v + velocity_gradient. No F update or plasticity.
    pub g2p: wgpu::ComputePipeline,
    /// F update + all plasticity + volume/density + position + boundary (sorted access).
    pub particles_update: wgpu::ComputePipeline,
    /// Post-particles_update: applies non-uniform body forces (gravity wells, Coulomb, etc.).
    pub force_fields: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl MpmPipelines {
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
                // binding 4: force_fields_params — uniform (GpuForceFieldsParams, 784 bytes)
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mpm_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Patch shader sources: inject Rust-side constants so WGSL never has its own copies.
        // Any change to MAX_MATERIAL_SLOTS or MAX_FORCE_FIELDS propagates here automatically.
        let p2g_src = patch_shader(shaders::P2G);
        let particles_update_src = patch_shader(shaders::PARTICLES_UPDATE);
        let force_fields_src = patch_shader(shaders::FORCE_FIELDS);
        let grid_update_src = patch_shader(shaders::GRID_UPDATE);

        let particle_sort = make_pipeline(
            device,
            &pipeline_layout,
            shaders::PARTICLE_SORT,
            "particle_sort_main",
            "particle_sort",
        );
        let grid_clear = make_pipeline(
            device,
            &pipeline_layout,
            shaders::GRID_CLEAR,
            "grid_clear_main",
            "grid_clear",
        );
        let p2g = make_pipeline(device, &pipeline_layout, &p2g_src, "p2g_main", "p2g");
        let grid_update = make_pipeline(
            device,
            &pipeline_layout,
            &grid_update_src,
            "grid_update_main",
            "grid_update",
        );
        let g2p = make_pipeline(device, &pipeline_layout, shaders::G2P, "g2p_main", "g2p");
        let particles_update = make_pipeline(
            device,
            &pipeline_layout,
            &particles_update_src,
            "particles_update_main",
            "particles_update",
        );
        let force_fields = make_pipeline(
            device,
            &pipeline_layout,
            &force_fields_src,
            "force_fields_main",
            "force_fields",
        );

        Self {
            particle_sort,
            grid_clear,
            p2g,
            grid_update,
            g2p,
            particles_update,
            force_fields,
            bind_group_layout,
        }
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
            ],
        })
    }
}

/// Inject Rust-side constants into a WGSL shader source.
/// Replaces `{{MAX_MATERIALS}}` and `{{MAX_FORCE_FIELDS}}` placeholders.
fn patch_shader(source: &str) -> String {
    source
        .replace("{{MAX_MATERIALS}}", &MAX_MATERIALS.to_string())
        .replace("{{MAX_FORCE_FIELDS}}", &MAX_FORCE_FIELDS.to_string())
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: &str,
    entry_point: &str,
    label: &str,
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
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

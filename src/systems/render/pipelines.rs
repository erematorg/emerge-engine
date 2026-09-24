//! wgpu pipeline construction for the renderer -- split out of `Renderer::new`
//! (was ~230 of `mod.rs`'s ~895 lines, three near-identical bind-group-layout /
//! shader-module / pipeline blocks inlined in one constructor). Each pipeline
//! is fully self-contained: no dependency on the others or on `Renderer`'s own
//! fields, so extracting them changes nothing about when/how they're built.

use std::mem;

use super::gpu_types::InstanceData;
use super::{CURVATURE_FLOW_SHADER, PREP_SHADER, RENDER_SHADER};

pub(super) fn bgl_storage_ro(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

pub(super) fn bgl_storage_rw(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

pub(super) fn bgl_uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        count: None,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    }
}

/// Instanced-quad particle draw pipeline (the CPU and GPU-compute paths both
/// feed the same vertex/instance buffers, just filled differently).
pub(super) fn build_particle_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render_particles"),
        source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_particles_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            }),
        ),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: mem::size_of::<InstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 24,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 4,
                        },
                    ],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (render_pipeline, render_bgl)
}

/// `prep_instances.wgsl` compute pipeline -- fills the instance buffer directly
/// from the particle storage buffer for the zero-readback GPU render path.
pub(super) fn build_prep_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("prep_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(3, wgpu::ShaderStages::COMPUTE),
        ],
    });

    let prep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("prep_instances"),
        source: wgpu::ShaderSource::Wgsl(PREP_SHADER.into()),
    });

    let prep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("prep_instances_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&prep_bgl],
                push_constant_ranges: &[],
            }),
        ),
        module: &prep_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (prep_pipeline, prep_bgl)
}

/// `grid_volume.wgsl` pipeline -- samples the solver's own P2G mass field
/// directly instead of per-particle splats (see that shader's own doc).
/// `grid_volume.wgsl`'s `grid_visibility_step_main` pipeline -- the SAME
/// real hysteresis technique `build_visibility_step_pipeline` already
/// ships for the curvature-flow surface, ported to this mode's own
/// `mass_floor` discard.
pub(super) fn build_grid_visibility_step_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid_visibility_step_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid_volume"),
        source: wgpu::ShaderSource::Wgsl(super::GRID_VOLUME_SHADER.into()),
    });
    let pipeline = build_compute_pipeline(
        device,
        "grid_visibility_step_pipeline",
        &bgl,
        &shader,
        "grid_visibility_step_main",
    );
    (pipeline, bgl)
}

pub(super) fn build_grid_volume_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let grid_volume_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid_volume_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(1, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(3, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(4, wgpu::ShaderStages::FRAGMENT),
        ],
    });

    let grid_volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grid_volume"),
        source: wgpu::ShaderSource::Wgsl(super::GRID_VOLUME_SHADER.into()),
    });

    let grid_volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("grid_volume_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&grid_volume_bgl],
                push_constant_ranges: &[],
            }),
        ),
        vertex: wgpu::VertexState {
            module: &grid_volume_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &grid_volume_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (grid_volume_pipeline, grid_volume_bgl)
}

fn build_curvature_flow_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("curvature_flow"),
        source: wgpu::ShaderSource::Wgsl(CURVATURE_FLOW_SHADER.into()),
    })
}

fn build_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            }),
        ),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// `curvature_flow.wgsl`'s `clear_surface_main` pipeline -- zeroes the
/// fixed-point atomic splat buffer before each frame's splat pass (same
/// real need `grid_clear.wgsl` already serves for the physics grid).
pub(super) fn build_surface_clear_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_clear_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE), // unused by this entry point, layout shared with splat's bg
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
            // Real mass-weighted temperature atomic, also cleared here --
            // see `surface_temp_atomic`'s own doc in the shader.
            bgl_storage_rw(3, wgpu::ShaderStages::COMPUTE),
            // Real volume-preserving-correction totals, also cleared here --
            // see `pre_total_atomic`'s own doc in the shader.
            bgl_storage_rw(4, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(5, wgpu::ShaderStages::COMPUTE),
            // N-material extension's per-cell mass array, also cleared here
            // (gated, real cost only when opted in) -- see
            // `surface_material_mass_atomic`'s own doc in the shader.
            bgl_storage_rw(6, wgpu::ShaderStages::COMPUTE),
            // Yu & Turk neighbourhood moments, also cleared here -- see
            // `surface_moments_atomic`'s own doc in the shader.
            bgl_storage_rw(7, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "surface_clear_pipeline",
        &bgl,
        &shader,
        "clear_surface_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `splat_density_main` pipeline -- scatters real
/// particle mass onto the finer auxiliary surface buffer via the same
/// quadratic B-spline kernel P2G uses for the physics grid (see that
/// shader's own top doc).
pub(super) fn build_surface_splat_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_splat_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
            // Real mass-weighted temperature atomic scatter -- see
            // `surface_temp_atomic`'s own doc in the shader.
            bgl_storage_rw(3, wgpu::ShaderStages::COMPUTE),
            // Real ground-truth total mass accumulation -- see
            // `pre_total_atomic`'s own doc in the shader.
            bgl_storage_rw(4, wgpu::ShaderStages::COMPUTE),
            // Unused by this entry point (only cleared, filled later by
            // `post_total_reduce_main`) but the layout is shared with clear.
            bgl_storage_rw(5, wgpu::ShaderStages::COMPUTE),
            // N-material extension's per-cell mass array -- see
            // `surface_material_mass_atomic`'s own doc in the shader.
            bgl_storage_rw(6, wgpu::ShaderStages::COMPUTE),
            // Yu & Turk neighbourhood moments -- written by
            // `splat_moments_main`, read back by `splat_density_main` to fit
            // each particle's kernel shape. See that buffer's own shader doc.
            bgl_storage_rw(7, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "surface_splat_pipeline",
        &bgl,
        &shader,
        "splat_density_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `splat_moments_main` pipeline -- scatters the
/// weighted moments of the particle distribution onto the physics grid so the
/// splat pass can fit each particle a neighbourhood-derived kernel shape
/// (Yu & Turk 2013). Shares the splat pass's own bind group layout: it reads
/// the same particles and params and writes the same moments buffer, so no
/// separate layout or bind group is needed.
pub(super) fn build_surface_moments_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = build_curvature_flow_shader(device);
    build_compute_pipeline(
        device,
        "surface_moments_pipeline",
        bgl,
        &shader,
        "splat_moments_main",
    )
}

/// `curvature_flow.wgsl`'s `convert_atomic_to_float_main` pipeline --
/// converts the settled fixed-point splat buffer into the first plain-f32
/// ping-pong buffer (see that entry point's own doc for why this can't
/// fold into the splat pass itself).
pub(super) fn build_surface_convert_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_convert_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
            // Real, persistent raw-splat history for neighborhood-clamped
            // temporal smoothing -- see that entry point's own doc.
            bgl_storage_rw(3, wgpu::ShaderStages::COMPUTE),
            // Real mass-weighted temperature: settled atomic in, plain f32
            // out -- see `surface_temp_atomic_ro`'s own doc in the shader.
            bgl_storage_ro(4, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(5, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "surface_convert_pipeline",
        &bgl,
        &shader,
        "convert_atomic_to_float_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `curvature_iterate_main` pipeline -- one real
/// mean-curvature smoothing step, ping-ponged between two plain float
/// buffers across several dispatches (see that entry point's own doc for
/// the real cited equation).
pub(super) fn build_surface_iterate_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_iterate_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "surface_iterate_pipeline",
        &bgl,
        &shader,
        "curvature_iterate_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `post_total_reduce_main` pipeline -- sums the
/// settled (post-curvature-flow) density into a real total, half of the
/// volume-preserving correction, see that entry point's own "Pass 1d" doc.
pub(super) fn build_post_total_reduce_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_total_reduce_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "post_total_reduce_pipeline",
        &bgl,
        &shader,
        "post_total_reduce_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `volume_correct_main` pipeline -- the other half
/// of the real volume-preserving correction: rescales the settled density
/// by `pre_total/post_total`, see that entry point's own "Pass 1d" doc.
pub(super) fn build_volume_correct_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("volume_correct_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_ro(1, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(2, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(3, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "volume_correct_pipeline",
        &bgl,
        &shader,
        "volume_correct_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `temp_avg_main` pipeline -- one-shot real
/// temperature recovery (mass-weighted sum / final settled density), see
/// that entry point's own "Pass 1c" doc for why this is split from the
/// diffusion step below.
pub(super) fn build_temp_avg_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("temp_avg_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_ro(1, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(2, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(3, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline =
        build_compute_pipeline(device, "temp_avg_pipeline", &bgl, &shader, "temp_avg_main");
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `temp_diffuse_main` pipeline -- the real 2D heat
/// equation (Fourier's law) applied to the recovered temperature field, see
/// that entry point's own "Pass 1c" doc for the real cited stability bound.
pub(super) fn build_temp_diffuse_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("temp_diffuse_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "temp_diffuse_pipeline",
        &bgl,
        &shader,
        "temp_diffuse_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `light_diffuse_main` pipeline -- the real
/// diffusion approximation to light transport (see that entry point's own
/// "Pass 1e" doc for the full real derivation, including the cited von
/// Neumann stability bound). Reads the current fluence + the diffused
/// temperature field (source) + the real per-material `OpticalTable`
/// (sigma_a/sigma_s), writes the next fluence.
pub(super) fn build_light_diffuse_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("light_diffuse_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_storage_ro(2, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(3, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(4, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "light_diffuse_pipeline",
        &bgl,
        &shader,
        "light_diffuse_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `wave_step_main` pipeline -- the real, persistent
/// 2D wave-equation step (see that entry point's own "Pass 2b" doc for the
/// full real-technique citation). 4 storage buffers (density read, wave
/// current read, wave previous read, wave next write) + 1 uniform.
pub(super) fn build_wave_step_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("wave_step_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_ro(1, wgpu::ShaderStages::COMPUTE),
            bgl_storage_ro(2, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(3, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(4, wgpu::ShaderStages::COMPUTE),
            // Prev-frame settled density, for the temporal (not spatial)
            // disturbance forcing term.
            bgl_storage_ro(5, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "wave_step_pipeline",
        &bgl,
        &shader,
        "wave_step_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `visibility_step_main` pipeline -- the real
/// hysteresis (Schmitt-trigger) visible/invisible state step (see that
/// entry point's own "Pass 2c" doc). 1 storage read (density) + 1 storage
/// read_write (persistent visibility state) + 1 uniform.
pub(super) fn build_visibility_step_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("visibility_step_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "visibility_step_pipeline",
        &bgl,
        &shader,
        "visibility_step_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `band_hysteresis_step_main` pipeline -- the real
/// hysteresis color-band state step (see that entry point's own "Pass 2d"
/// doc). Same 3-binding shape as `build_visibility_step_pipeline`
/// (density read + persistent state read_write + uniform), a separate
/// pipeline since it's a distinct entry point/persistent buffer.
pub(super) fn build_band_hysteresis_step_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("band_hysteresis_step_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::COMPUTE),
            bgl_storage_rw(1, wgpu::ShaderStages::COMPUTE),
            bgl_uniform(2, wgpu::ShaderStages::COMPUTE),
        ],
    });
    let shader = build_curvature_flow_shader(device);
    let pipeline = build_compute_pipeline(
        device,
        "band_hysteresis_step_pipeline",
        &bgl,
        &shader,
        "band_hysteresis_step_main",
    );
    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `vs_main`/`fs_main` render pipeline -- final
/// extraction + Beer-Lambert/gradient-shading composite, reading the
/// settled, smoothed surface buffer (see that shader's own doc).
pub(super) fn build_surface_render_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_render_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(1, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(3, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(4, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(5, wgpu::ShaderStages::FRAGMENT),
            // Real mass-weighted temperature, single-phase only -- see
            // `surface_temp_final`'s own doc in the shader for why
            // `fs_main_dual_phase` doesn't get this (already at the real
            // 8-storage-buffer WebGPU-guaranteed minimum).
            bgl_storage_ro(6, wgpu::ShaderStages::FRAGMENT),
            // N-material extension, single-phase only -- see
            // `surface_material_mass`'s own doc in the shader.
            bgl_storage_ro(7, wgpu::ShaderStages::FRAGMENT),
            // Real diffused light fluence, single-phase only -- see
            // `surface_light_phi`'s own doc in the shader.
            bgl_storage_ro(8, wgpu::ShaderStages::FRAGMENT),
        ],
    });
    let shader = build_curvature_flow_shader(device);

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("surface_render_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            }),
        ),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bgl)
}

/// `curvature_flow.wgsl`'s `vs_main`/`fs_main_dual_phase` render pipeline --
/// the two-phase extension's own final composite, reading TWO
/// independently-smoothed surface buffers and picking the real, locally
/// denser phase per pixel (see that shader's own doc).
pub(super) fn build_surface_dual_render_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("surface_dual_render_bgl"),
        entries: &[
            bgl_storage_ro(0, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(1, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(3, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(4, wgpu::ShaderStages::FRAGMENT),
            // Real per-phase wave/hysteresis state -- see
            // `curvature_flow.wgsl`'s own Pass 3b doc for the exact
            // storage-buffer-count reasoning (8 total, at the WebGPU
            // guaranteed minimum).
            bgl_storage_ro(5, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(6, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(7, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(8, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(9, wgpu::ShaderStages::FRAGMENT),
            bgl_storage_ro(10, wgpu::ShaderStages::FRAGMENT),
        ],
    });
    let shader = build_curvature_flow_shader(device);

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("surface_dual_render_pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            }),
        ),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main_dual_phase"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: output_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bgl)
}

/// emerge particle renderer — physics-driven, no assets, no Bevy.
///
/// # Architecture
///
/// Two-pass GPU pipeline (zero CPU↔GPU roundtrip):
///   1. **Compute** — `prep_instances.wgsl`: reads `Particle[]` from VRAM, writes
///      `InstanceData[]` (F-deformed quad + color). One thread per particle.
///   2. **Render** — `render_particles.wgsl`: instanced draw, optional disc clip.
///
/// LP owns the wgpu surface and calls `render_raw` each frame with emerge's
/// `particle_buffer()`. The renderer composites onto whatever LP has already drawn.
///
/// # Extending
///
/// - `set_color_mode(ColorMode::ByPhysics)` — Beer-Lambert + thermal (wire σ_a via `set_optical_params`)
/// - `set_optical_params(slot, sigma_a)` — LP registers per-material absorption (placeholder, no-op until ByPhysics lands)
/// - Future passes (curvature flow, SS-SSS) added as separate structs, same `render_raw` entry point
///
/// # Usage
///
/// ```no_run
/// # #[cfg(feature = "render")]
/// # {
/// use emerge::render::{ColorMode, MpmRenderer};
/// # let (device, queue, solver, surface_format, output_view) = unimplemented!();
/// let mut renderer = MpmRenderer::new(&device, &solver, surface_format);
/// renderer.set_camera(&queue, solver.config().grid_res as u32, 1.0, true);
/// renderer.set_color_mode(ColorMode::ByMaterial);
/// renderer.render_raw(&device, &queue, &solver.particle_buffer(), solver.particle_count(), &output_view, false);
/// # }
/// ```
use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu::GpuSolver;

const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_WG: u32 = 64;

/// Particle color visualization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// Color by material_id — 16-slot fixed palette. Default.
    #[default]
    ByMaterial = 0,
    /// Color by particle speed — blue (slow) → red (fast).
    ByVelocity = 1,
    /// Color by det(F) — blue (compressed) / white (rest) / red (expanded).
    ByVolume = 2,
    /// Physics-derived: Beer-Lambert absorption + blackbody thermal glow.
    /// Requires `set_optical_params` per material slot. LP's production mode.
    ByPhysics = 3,
    /// Blackbody thermal emission only — diagnostic for temperature field.
    ByThermal = 4,
    /// Muscle activation [0,1] → cool→warm gradient. Creature diagnostic.
    ByActivation = 5,
}

// ── GPU-side structs (repr(C), bytemuck) ─────────────────────────────────────

/// RenderConfig uploaded to the compute shader uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderConfig {
    mode: u32,
    particle_count: u32,
    /// Maps |v| → [0, 1] for velocity heat map. Typically 1 / (grid_cell_size / sub_dt).
    vel_scale: f32,
    _pad: u32,
}

/// Per-instance data written by the compute pass, read as vertex attributes.
/// Layout (48 bytes) must match `struct InstanceData` in prep_instances.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceData {
    deform_col0: [f32; 2], // offset  0
    deform_col1: [f32; 2], // offset  8
    position: [f32; 2],    // offset 16
    _pad: [f32; 2],        // offset 24
    color: [f32; 4],       // offset 32
}
const _: () = assert!(mem::size_of::<InstanceData>() == 48);

/// Camera uniform uploaded before each render call.
/// Layout (80 bytes) must match `struct Camera` in render_particles.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraParams {
    view_proj: [f32; 16],
    particle_scale: f32,
    round_particles: u32,
    _pad: [f32; 2],
}
const _: () = assert!(mem::size_of::<CameraParams>() == 80);

// ── MpmRenderer ────────────────────────────────────────────────────────────

/// Standalone wgpu debug renderer — draws MPM particles as deformed ellipses.
///
/// Particles are rendered using the particle's deformation gradient F, so
/// compression, shear, and volume change are visually apparent.
pub struct MpmRenderer {
    prep_pipeline: wgpu::ComputePipeline,
    prep_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    config_buffer: wgpu::Buffer,
    max_particles: usize,
    color_mode: ColorMode,
    vel_scale: f32,
}

impl MpmRenderer {
    /// Create a renderer compatible with `output_format` (match your swapchain surface format).
    pub fn new(
        device: &wgpu::Device,
        solver: &GpuSolver,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_raw(device, solver.particle_count().max(1), output_format)
    }

    /// Create a renderer without a GpuSolver reference.
    /// Use when embedding in a Bevy render node (particle count may be updated lazily).
    pub fn new_raw(
        device: &wgpu::Device,
        max_particles: usize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let max_particles = max_particles.max(1);

        // ── Persistent buffers ────────────────────────────────────────────
        let instance_buffer = make_instance_buffer(device, max_particles);

        // Unit quad: 4 corners in [-0.5, 0.5]² (vec2, 8 bytes each). Immutable after creation.
        let quad_verts: &[u8] = bytemuck::cast_slice::<[f32; 2], u8>(&[
            [-0.5f32, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_verts"),
            contents: quad_verts,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let quad_idx: &[u8] = bytemuck::cast_slice::<u16, u8>(&[0u16, 1, 2, 0, 2, 3]);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_idx"),
            contents: quad_idx,
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_camera"),
            size: mem::size_of::<CameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_config"),
            size: mem::size_of::<RenderConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Compute pipeline (prep_instances) ────────────────────────────
        let prep_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_prep_bgl"),
            entries: &[
                // binding 0: particles (read-only storage)
                bgl_entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // binding 1: instances (read-write storage — written by this pass)
                bgl_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                // binding 2: render config uniform
                bgl_entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });

        let prep_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_prep_pl"),
            bind_group_layouts: &[&prep_layout],
            push_constant_ranges: &[],
        });
        let prep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prep_instances"),
            source: wgpu::ShaderSource::Wgsl(PREP_SHADER.into()),
        });
        let prep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("render_prep_pipeline"),
            layout: Some(&prep_pl_layout),
            module: &prep_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Render pipeline (render_particles) ───────────────────────────
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_bgl"),
            entries: &[bgl_entry(
                0,
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            )],
        });

        let render_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl_layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_particles"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_particles_pipeline"),
            layout: Some(&render_pl_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    // Slot 0: per-vertex quad positions (vec2, 8 bytes).
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        }],
                    },
                    // Slot 1: per-instance InstanceData (48 bytes).
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
                            // offset 24: pad — not bound to any shader location
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
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Stable render bind group — camera_buffer identity never changes.
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        Self {
            prep_pipeline,
            prep_layout,
            render_pipeline,
            render_bind_group,
            instance_buffer,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            config_buffer,
            max_particles,
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Set orthographic camera: maps grid space [0, `grid_res`] → NDC [-1, 1].
    /// Y is flipped so that grid Y=0 appears at the top of the screen.
    ///
    /// `particle_scale` — uniform scale for the deformed quad in grid cells.
    /// 1.0 fills one grid cell; use ~0.5–0.8 for ppc≥2 to avoid overlap.
    ///
    /// `round_particles` — clip to a smooth disc (true) or draw full deformed quad (false).
    pub fn set_camera(
        &self,
        queue: &wgpu::Queue,
        grid_res: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let gr = grid_res as f32;
        // Column-major orthographic matrix: grid [0,gr]² → NDC [-1,1]² with Y flipped.
        // col 0: (2/gr, 0, 0, 0), col 1: (0, -2/gr, 0, 0), col 2: (0,0,1,0), col 3: (-1,1,0,1)
        let view_proj: [f32; 16] = [
            2.0 / gr,
            0.0,
            0.0,
            0.0,
            0.0,
            -2.0 / gr,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
        ];
        let params = CameraParams {
            view_proj,
            particle_scale,
            round_particles: round_particles as u32,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Set the color visualization mode.
    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }

    /// Set the velocity scale for `ByVelocity` mode.
    /// A particle moving at `1.0 / vel_scale` grid-cells/s will appear fully red.
    /// Default: 0.05 (saturates at 20 cells/s, typical for slow sims).
    pub fn set_vel_scale(&mut self, vel_scale: f32) {
        self.vel_scale = vel_scale;
    }

    /// Register per-material optical absorption coefficients for `ByPhysics` mode.
    ///
    /// `slot`: material_id % 16. `sigma_a`: [r, g, b] absorption per grid-cell.
    /// Real optics values (LP year-1): water=[0.06,0.014,0.007], sand=[1.2,0.96,0.80].
    ///
    /// No-op until `ByPhysics` shader support lands. Call now to future-proof LP startup.
    pub fn set_optical_params(&mut self, _slot: usize, _sigma_a: [f32; 3]) {}

    // ── Rendering ─────────────────────────────────────────────────────────────

    /// Render all particles from `solver` into `output_view`.
    ///
    /// Clears `output_view` to a dark background first. For compositing into an
    /// existing scene (e.g. a Bevy render node), use `render_raw` with `clear=false`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        solver: &GpuSolver,
        output_view: &wgpu::TextureView,
    ) {
        self.render_raw(
            device,
            queue,
            &solver.particle_buffer(),
            solver.particle_count(),
            output_view,
            true,
        );
    }

    /// Render particles directly from a GPU buffer into `output_view`.
    ///
    /// `clear`: if true, clears the view to the background color first.
    ///          if false, composites particles on top of existing content (LoadOp::Load).
    /// Use `clear=false` when embedding in a Bevy render node.
    pub fn render_raw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buffer: &wgpu::Buffer,
        count: usize,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        if count == 0 {
            return;
        }

        // Grow instance buffer if spawn_region added more particles.
        if count > self.max_particles {
            self.instance_buffer = make_instance_buffer(device, count);
            self.max_particles = count;
        }

        // Upload render config.
        let config = RenderConfig {
            mode: self.color_mode as u32,
            particle_count: count as u32,
            vel_scale: self.vel_scale,
            _pad: 0,
        };
        queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(&config));

        // Rebuild prep bind group each frame — particle_buffer may have been reallocated.
        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_prep_bg"),
            layout: &self.prep_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.config_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles"),
        });

        // Pass 1: compute — fill instance_buffer from particle data.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prep_instances"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.prep_pipeline);
            cpass.set_bind_group(0, &prep_bg, &[]);
            cpass.dispatch_workgroups((count as u32 + PREP_WG - 1) / PREP_WG, 1, 1);
        }

        // Pass 2: render — instanced draw of deformed quads.
        {
            let load_op = if clear {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.05,
                    g: 0.05,
                    b: 0.08,
                    a: 1.0,
                })
            } else {
                wgpu::LoadOp::Load
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_particles"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..6, 0, 0..count as u32);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_instance_buffer(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render_instances"),
        size: (count * mem::size_of::<InstanceData>()) as u64,
        // STORAGE: written by compute pass. VERTEX: read by render pass.
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

fn bgl_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

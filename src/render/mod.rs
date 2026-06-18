/// emerge particle renderer -- physics-driven, no assets.
///
/// # Two rendering paths
///
/// **CPU path** (`render_slice`):
///   Builds `InstanceData` per particle on CPU, uploads via `write_buffer`.
///
/// **GPU path** (`render_gpu`):
///   Runs `prep_instances.wgsl` compute to fill instance buffer directly from
///   the particle storage buffer -- zero CPU readback, zero stall.
///   Pass `sim.particle_buffer()` + `sim.particle_count()`. No `sync_particles_blocking()`.
use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::Mat2;
use wgpu::util::DeviceExt;

use crate::particle::{Particle, Particles};

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const PREP_WG: u32 = 64;

// ── Color mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    ByMaterial = 0,
    ByVelocity = 1,
    ByVolume = 2,
    ByPhysics = 3,
    ByThermal = 4,
    ByActivation = 5,
}

// ── GPU-side structs (must match WGSL) ────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceData {
    deform_col0: [f32; 2],
    deform_col1: [f32; 2],
    position: [f32; 2],
    _pad: [f32; 2],
    color: [f32; 4],
}
const _: () = assert!(mem::size_of::<InstanceData>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraParams {
    view_proj: [f32; 16],
    particle_scale: f32,
    round_particles: u32,
    _pad: [f32; 2],
}
const _: () = assert!(mem::size_of::<CameraParams>() == 80);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderConfig {
    mode: u32,
    particle_count: u32,
    vel_scale: f32,
    _pad: u32,
}
const _: () = assert!(mem::size_of::<RenderConfig>() == 16);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OpticalTable {
    slots: [[f32; 4]; 16],
}
const _: () = assert!(mem::size_of::<OpticalTable>() == 256);

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer, // VERTEX | COPY_DST — drawn as per-instance attributes
    storage_instances: wgpu::Buffer, // STORAGE | COPY_SRC — compute write target (GPU path)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    max_particles: usize,

    prep_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    render_config_buf: wgpu::Buffer,
    optical_table_buf: wgpu::Buffer,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    sigma_a: [[f32; 3]; 16],
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        max_particles: usize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let cap = max_particles.max(1);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            // VERTEX for draw; COPY_DST for both the CPU fill path and the GPU compute copy.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // GPU compute write target. Kept distinct from the vertex buffer: wgpu treats a
        // read_write storage buffer as an exclusive usage, so sharing one buffer for both
        // compute-write and vertex-read trips its usage tracker. Copied into instance_buffer.
        let storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances_storage"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_verts"),
            contents: bytemuck::cast_slice::<[f32; 2], u8>(&[
                [-0.5f32, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_idx"),
            contents: bytemuck::cast_slice::<u16, u8>(&[0u16, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_camera"),
            size: mem::size_of::<CameraParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_config_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_config"),
            size: mem::size_of::<RenderConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let optical_table_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_optics"),
            size: mem::size_of::<OpticalTable>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Render pipeline ---------------------------------------------------------
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

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Compute pipeline (prep_instances) ---------------------------------------
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

        Self {
            render_pipeline,
            render_bind_group,
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            max_particles: cap,
            prep_pipeline,
            prep_bgl,
            render_config_buf,
            optical_table_buf,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            sigma_a: [[0.3f32; 3]; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Call at init and on every resize.
    pub fn set_camera(
        &self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let gr = grid_res as f32;
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                _pad: [0.0; 2],
            }),
        );
    }

    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }
    pub fn set_vel_scale(&mut self, s: f32) {
        self.vel_scale = s;
    }

    pub fn set_optical_params(&mut self, slot: usize, sigma_a: [f32; 3]) {
        self.sigma_a[slot % 16] = sigma_a;
    }

    pub fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(queue, &self.optical_table_buf, &self.sigma_a);
    }

    // ── GPU compute render path ────────────────────────────────────────────────

    /// Zero-readback GPU render. No `sync_particles_blocking()` needed.
    pub fn render_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        if particle_count == 0 {
            return;
        }
        self.ensure_capacity(device, particle_count);

        queue.write_buffer(
            &self.render_config_buf,
            0,
            bytemuck::bytes_of(&RenderConfig {
                mode: self.color_mode as u32,
                particle_count: particle_count as u32,
                vel_scale: self.vel_scale,
                _pad: 0,
            }),
        );
        write_optical_table(queue, &self.optical_table_buf, &self.sigma_a);

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prep_bg"),
            layout: &self.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.storage_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.render_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_gpu"),
        });
        // Compute: fill the storage instance buffer from the particle buffer.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prep_instances"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.prep_pipeline);
            cp.set_bind_group(0, &prep_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(PREP_WG), 1, 1);
        }
        // GPU->GPU copy into the vertex buffer (decouples storage and vertex roles).
        let bytes = (particle_count * mem::size_of::<InstanceData>()) as u64;
        enc.copy_buffer_to_buffer(&self.storage_instances, 0, &self.instance_buffer, 0, bytes);
        // Render: draw instanced quads from the vertex buffer.
        self.draw_pass(&mut enc, output_view, clear, particle_count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── CPU render path ────────────────────────────────────────────────────────

    /// CPU-fill render for the SoA `Particles` store (CPU `Simulation`).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &Particles,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles.iter() {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(&p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_soa"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    pub fn render_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[Particle],
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_cpu"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.max_particles {
            let size = (count * mem::size_of::<InstanceData>()) as u64;
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.max_particles = count;
        }
    }

    fn draw_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: bool,
        count: usize,
    ) {
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, &self.render_bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
        rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..count as u32);
    }

    fn particle_color(&self, p: &Particle) -> [f32; 4] {
        match self.color_mode {
            ColorMode::ByMaterial => material_palette(p.material_id),
            ColorMode::ByVelocity => heat(p.v.length() * self.vel_scale),
            ColorMode::ByVolume => heat(det2(p.deformation_gradient) * 0.5),
            ColorMode::ByPhysics => {
                let sigma = self.sigma_a[p.material_id as usize % 16];
                let j = det2(p.deformation_gradient).clamp(0.05, 4.0);
                let od = 1.0 / j;
                let r = (-sigma[0] * od).exp();
                let g = (-sigma[1] * od).exp();
                let b = (-sigma[2] * od).exp();
                let t = (p.temperature / 5000.0).clamp(0.0, 1.0);
                let glow = t * t * 2.0;
                let [er, eg, eb, _] = heat(0.5 + t * 0.5);
                [
                    (r + er * glow).min(1.0),
                    (g + eg * glow).min(1.0),
                    (b + eb * glow).min(1.0),
                    1.0,
                ]
            }
            ColorMode::ByThermal => {
                let t = (p.temperature / 1500.0).clamp(0.0, 1.0);
                let [r, g, b, _] = heat(t);
                [
                    r * (0.1 + t * 0.9),
                    g * (0.1 + t * 0.9),
                    b * (0.1 + t * 0.9),
                    1.0,
                ]
            }
            ColorMode::ByActivation => heat(p.activation.clamp(0.0, 1.0) * 0.8),
        }
    }
}

// ── BGL helpers ───────────────────────────────────────────────────────────────

fn bgl_storage_ro(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_storage_rw(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
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

// ── Color helpers (CPU path) ──────────────────────────────────────────────────

fn write_optical_table(queue: &wgpu::Queue, buf: &wgpu::Buffer, sigma_a: &[[f32; 3]; 16]) {
    let mut table = OpticalTable {
        slots: [[0.0; 4]; 16],
    };
    for (i, s) in sigma_a.iter().enumerate() {
        table.slots[i] = [s[0], s[1], s[2], 0.0];
    }
    queue.write_buffer(buf, 0, bytemuck::bytes_of(&table));
}

fn det2(f: Mat2) -> f32 {
    f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x
}

fn heat(t: f32) -> [f32; 4] {
    let c = t.clamp(0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - (c - 0.5).abs() * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    [r, g, b, 1.0]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn material_palette(id: u32) -> [f32; 4] {
    match id % 16 {
        0 => [0.35, 0.65, 1.00, 1.0],
        1 => [0.90, 0.80, 0.30, 1.0],
        2 => [0.80, 0.90, 1.00, 1.0],
        3 => [0.50, 0.85, 0.50, 1.0],
        4 => [1.00, 0.45, 0.20, 1.0],
        5 => [0.85, 0.35, 0.35, 1.0],
        6 => [0.65, 0.40, 0.85, 1.0],
        7 => [0.40, 0.85, 0.80, 1.0],
        8 => [0.90, 0.60, 0.40, 1.0],
        9 => [0.50, 0.50, 0.90, 1.0],
        10 => [0.70, 0.90, 0.40, 1.0],
        11 => [1.00, 0.80, 0.20, 1.0],
        12 => [0.85, 0.50, 0.75, 1.0],
        13 => [0.40, 0.70, 0.50, 1.0],
        14 => [0.60, 0.60, 0.60, 1.0],
        _ => [1.00, 1.00, 1.00, 1.0],
    }
}

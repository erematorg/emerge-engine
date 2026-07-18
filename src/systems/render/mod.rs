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
use wgpu::util::DeviceExt;

use crate::particle::{Particle, Particles};

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const GRID_VOLUME_SHADER: &str = include_str!("shaders/grid_volume.wgsl");
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
    /// Generic second scalar carrier (resource/grass level, pheromone, nutrients).
    /// See `Particle::scalar_field`'s own doc. Distinct wire value (6, not the next
    /// unused slot after ByActivation's implicit WGSL else-branch) so the GPU shader's
    /// existing fallback `else` can keep meaning ByActivation without renumbering it.
    ByScalarField = 6,
}

/// Mirrors `grid_volume.wgsl`'s `GridVolumeParams` -- see that shader's own doc for
/// the real technique (samples the solver's own P2G mass field directly instead of
/// per-particle splats).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridVolumeParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    grid_res: u32,
    mass_floor: f32,
    material_mass_enabled: u32,
    _pad1: f32,
}
const _: () = assert!(mem::size_of::<GridVolumeParams>() == 32);

/// Bundles `render_grid_volume`'s buffer args -- same real precedent as
/// `spacetime::transfer::P2GParticleState` (a struct instead of a suppressed
/// argument-count lint).
pub struct GridVolumeSource<'a> {
    /// `GpuSimulation::grid_buffer()`.
    pub grid: &'a wgpu::Buffer,
    /// `GpuSimulation::material_mass_buffer()` -- pass it regardless of whether
    /// `attach_grid_material_render_gpu` was called; `material_mass_enabled` gates
    /// whether the shader actually reads it.
    pub material_mass: &'a wgpu::Buffer,
    pub material_mass_enabled: bool,
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
    /// rgb = sigma_a, absorption coefficient (Beer-Lambert). .w = sigma_s, reduced
    /// scattering coefficient (single scalar, not per-channel -- real tissue
    /// scattering is much less wavelength-dependent than absorption in the visible
    /// range, Jacques 2013, a legitimate simplification for that reason).
    slots: [[f32; 4]; 16],
    /// .x = specular Fresnel base reflectance R0 (Schlick 1994 approximation),
    /// rest padding. Real, cited, but bounded: this renderer has no surface-normal
    /// estimation (it tints particle instances, doesn't raytrace a reconstructed
    /// surface), so this is a constant near-normal-incidence reflectance, NOT a
    /// full view-angle-dependent Fresnel term -- honestly a simplification, not a
    /// claim of full BRDF accuracy.
    specular: [[f32; 4]; 16],
}
const _: () = assert!(mem::size_of::<OpticalTable>() == 512);

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

    grid_volume_pipeline: wgpu::RenderPipeline,
    grid_volume_bgl: wgpu::BindGroupLayout,
    grid_volume_params_buf: wgpu::Buffer,
    /// Cached ortho projection + grid_res (set by `set_camera`) -- lets
    /// `render_grid_volume` take just (device, queue, grid_buf, material_mass_buf,
    /// view, clear) instead of repeating width/height/grid_res, keeping it under
    /// clippy's argument-count lint.
    cached_ortho: (f32, f32, f32, f32),
    cached_grid_res: u32,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    sigma_a: [[f32; 3]; 16],
    /// Reduced scattering coefficient per material slot (single scalar -- see
    /// `OpticalTable`'s own doc for why this isn't per-channel).
    sigma_s: [f32; 16],
    /// Specular Fresnel base reflectance R0 per material slot (see `OpticalTable`'s
    /// own doc for the real-but-bounded caveat).
    specular_r0: [f32; 16],
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

        // Grid-volume pipeline (samples the solver's own grid mass field) -----------
        let grid_volume_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_volume_params"),
            size: mem::size_of::<GridVolumeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_volume_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_volume_bgl"),
            entries: &[
                bgl_storage_ro(0, wgpu::ShaderStages::FRAGMENT),
                bgl_uniform(1, wgpu::ShaderStages::FRAGMENT),
                bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
                bgl_storage_ro(3, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let grid_volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid_volume"),
            source: wgpu::ShaderSource::Wgsl(GRID_VOLUME_SHADER.into()),
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
            grid_volume_pipeline,
            grid_volume_bgl,
            grid_volume_params_buf,
            cached_ortho: (1.0, 0.0, 1.0, 0.0),
            cached_grid_res: 1,
            optical_table_buf,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            sigma_a: [[0.3f32; 3]; 16],
            sigma_s: [0.0f32; 16],
            specular_r0: [0.0f32; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Call at init and on every resize.
    pub fn set_camera(
        &mut self,
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
        self.cached_ortho = (sx, tx, sy, ty);
        self.cached_grid_res = grid_res;
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

    /// Reduced scattering coefficient for `slot` -- see `OpticalTable`'s doc for
    /// what this represents physically (real subsurface scattering, single-
    /// scattering approximation) and its real citation (Jacques 2013).
    pub fn set_optical_scattering(&mut self, slot: usize, sigma_s: f32) {
        self.sigma_s[slot % 16] = sigma_s;
    }

    /// Specular Fresnel base reflectance R0 for `slot` -- see `OpticalTable`'s doc
    /// for the real-but-bounded caveat (constant near-normal reflectance, no
    /// surface-normal-dependent angle term).
    pub fn set_specular_r0(&mut self, slot: usize, r0: f32) {
        self.specular_r0[slot % 16] = r0;
    }

    pub fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );
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
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

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

    // ── Grid-volume render path ────────────────────────────────────────────────

    /// Renders the solver's own grid mass field directly (see `grid_volume.wgsl`'s
    /// own doc for the real technique). Requires `set_camera` to have been called
    /// first (same as `render_gpu` needs for its own bind group) -- reuses the
    /// identical cached orthographic projection/grid_res so both modes line up on
    /// screen without re-deriving them.
    pub fn render_grid_volume(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: GridVolumeSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        queue.write_buffer(
            &self.grid_volume_params_buf,
            0,
            bytemuck::bytes_of(&GridVolumeParams {
                sx,
                tx,
                sy,
                ty,
                grid_res: self.cached_grid_res,
                // REAL BUG FOUND AND FIXED 2026-07-18: 1e-4 is "any trace of mass at
                // all" -- combined with bilinear smoothing (which spreads a full
                // cell's worth of falloff outward from even ONE occupied neighbor),
                // this made the rendered shape visibly overshoot the real particle-
                // occupied extent (reported live: "overlaps," puffy edges bigger than
                // true sizing) and let single sparse/low-mass cells (e.g. a lone
                // stray water particle) render as isolated blocky rectangles. Real
                // per-particle cell-mass scale at this project's typical demo density
                // (~4.0) and spacing (~0.5) is order 0.5-4 per occupied cell (B-spline
                // center weight up to 0.75 * particle mass, several particles/cell in
                // steady state) -- 0.15 requires genuine, non-trivial local density
                // before showing anything, tightening the visible edge to real
                // occupied cells instead of any measurable trace.
                mass_floor: 0.15,
                material_mass_enabled: source.material_mass_enabled as u32,
                _pad1: 0.0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_volume_bg"),
            layout: &self.grid_volume_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_volume_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: source.material_mass.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_grid_volume"),
        });
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
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_grid_volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
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
            rp.set_pipeline(&self.grid_volume_pipeline);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
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
}

// particle_color (the CPU-path per-particle color computation) is split into
// color.rs alongside the rest of the "Color helpers" section below -- see
// that file's own doc comment.
mod color;
use color::write_optical_table;

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

// Test suite split into its own file -- was ~150 of this file's ~930 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`.
#[cfg(test)]
mod tests;

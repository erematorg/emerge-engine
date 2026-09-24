//! Orbit/motion trail renderer -- a fading polyline per particle, the same
//! real, standard N-body-visualization technique astronomy viz tools (e.g.
//! REBOUND's own OpenGL/js visualizers) use to show an orbit's shape over
//! time, not just an instantaneous dot.
//!
//! Fully self-contained: owns its own tiny camera uniform/bind group rather
//! than reaching into [`super::Renderer`]'s much larger internal state, and
//! its own encoder/submit cycle (mirrors [`super::Renderer::render`]'s
//! `clear: bool` pattern) -- so it composes with the main particle renderer
//! via ordinary `LoadOp::Load`, not shared `wgpu::RenderPass` plumbing. This
//! keeps it a general, opt-in capability any demo can add alongside
//! `Renderer`, not something welded into one example.
//!
//! wgpu core has no native line-width control, so trails draw as 1px
//! `LineStrip` segments -- a disclosed simplification; a thicker ribbon
//! would need a quad-per-segment approach instead, not built here.

use std::collections::VecDeque;

use glam::{Mat4, Vec2};

const TRAILS_SHADER: &str = include_str!("shaders/trails.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TrailCamera {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TrailVertex {
    position: [f32; 2],
    age: f32,
    color: [f32; 4],
}

pub struct TrailRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    histories: Vec<VecDeque<Vec2>>,
    colors: Vec<[f32; 4]>,
    max_len: usize,
    scratch: Vec<TrailVertex>,
    /// (start_vertex, vertex_count) per particle, recomputed by `upload` every render.
    ranges: Vec<(u32, u32)>,
}

impl TrailRenderer {
    /// `max_particles` bounds how many independent trails can be tracked;
    /// `max_len` is the per-trail sample cap (older samples drop off the
    /// tail once exceeded).
    pub fn new(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        max_particles: usize,
        max_len: usize,
    ) -> Self {
        let max_particles = max_particles.max(1);
        let max_len = max_len.max(2);

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail_camera"),
            size: std::mem::size_of::<TrailCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trail_camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trail_camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trails"),
            source: wgpu::ShaderSource::Wgsl(TRAILS_SHADER.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("trails_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&camera_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TrailVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 12,
                            shader_location: 2,
                        },
                    ],
                }],
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
                topology: wgpu::PrimitiveTopology::LineStrip,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_capacity = max_particles * max_len;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail_vertices"),
            size: (vertex_capacity * std::mem::size_of::<TrailVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_buf,
            camera_bg,
            vertex_buffer,
            vertex_capacity,
            histories: vec![VecDeque::with_capacity(max_len); max_particles],
            colors: vec![[1.0, 1.0, 1.0, 0.6]; max_particles],
            max_len,
            scratch: Vec::new(),
            ranges: Vec::new(),
        }
    }

    /// Sets a trail's own base color (alpha here is the trail's peak
    /// opacity at the newest point -- per-vertex age fade multiplies it
    /// down toward the tail). Defaults to translucent white.
    pub fn set_color(&mut self, index: usize, rgba: [f32; 4]) {
        if let Some(c) = self.colors.get_mut(index) {
            *c = rgba;
        }
    }

    /// Same view-projection convention as [`super::Renderer::set_camera_centered`]
    /// -- pass the identical matrix each frame to keep trails aligned with
    /// the main particle render.
    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        let data = TrailCamera {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&data));
    }

    /// Records one new sample per particle. Real, load-bearing calling
    /// convention: call this on a FIXED SIMULATED-TIME cadence (e.g. once
    /// per simulated day via an accumulator the caller owns), never once
    /// per rendered frame or once per `sim.step()` call directly -- either
    /// of those ties trail resolution to render rate / playback speed
    /// instead of orbital angular progress. A fast-orbiting body sampled
    /// too coarsely (e.g. once per frame at a high steps-per-frame
    /// multiplier) draws as a visibly faceted polygon instead of a smooth
    /// ellipse -- a real, reported regression the first version of this doc
    /// caused by recommending per-frame sampling.
    pub fn push(&mut self, positions: &[Vec2]) {
        for (i, &pos) in positions.iter().enumerate() {
            let Some(h) = self.histories.get_mut(i) else {
                break;
            };
            h.push_back(pos);
            if h.len() > self.max_len {
                h.pop_front();
            }
        }
    }

    /// Drops all recorded history (e.g. on a demo's own scene reset).
    pub fn clear(&mut self) {
        for h in &mut self.histories {
            h.clear();
        }
    }

    fn upload(&mut self, queue: &wgpu::Queue) {
        self.scratch.clear();
        self.ranges.clear();
        for (i, h) in self.histories.iter().enumerate() {
            let start = self.scratch.len();
            let n = h.len();
            if n >= 2 {
                let color = self.colors[i];
                // Newest sample first (age=0), oldest last (age=1) --
                // `VecDeque`'s back is the most recently pushed point.
                for (age_idx, &pos) in h.iter().rev().enumerate() {
                    let age = age_idx as f32 / (n - 1) as f32;
                    self.scratch.push(TrailVertex {
                        position: [pos.x, pos.y],
                        age,
                        color,
                    });
                }
            }
            let count = (self.scratch.len() - start) as u32;
            self.ranges.push((start as u32, count));
        }
        if self.scratch.len() > self.vertex_capacity {
            self.scratch.truncate(self.vertex_capacity);
        }
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    /// Owns its own encoder/submit cycle, same `clear` convention as
    /// [`super::Renderer::render`] -- pass `clear: false` to composite on
    /// top of an already-drawn frame (`LoadOp::Load`).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        self.upload(queue);
        // Always runs the pass (even with zero drawable trails yet, e.g. the
        // first few frames) so `clear: true` still actually clears --
        // skipping the whole pass here would silently drop that when this
        // is the frame's first draw call, leaving the swapchain texture's
        // previous (undefined) contents on screen.
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

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_trails"),
        });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_trails"),
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
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.camera_bg, &[]);
            for &(start, count) in &self.ranges {
                if count < 2 {
                    continue;
                }
                let stride = std::mem::size_of::<TrailVertex>() as u64;
                let byte_start = start as u64 * stride;
                let byte_end = (start + count) as u64 * stride;
                rp.set_vertex_buffer(0, self.vertex_buffer.slice(byte_start..byte_end));
                rp.draw(0..count, 0..1);
            }
        }
        queue.submit(std::iter::once(enc.finish()));
    }
}

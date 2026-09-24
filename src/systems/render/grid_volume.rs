//! Grid-native volumetric render path -- split out of `mod.rs`, same
//! "verbatim `impl Renderer` method, no behavior change" pattern already
//! used for `surface_reconstruction.rs`. Reads the shared MPM grid buffer
//! directly (per-cell mass/material), distinct from the particle-instance
//! paths (`render_gpu`/`render_slice`) and from curvature-flow surface
//! reconstruction.

use super::Renderer;
use super::color::write_optical_table;
use super::gpu_types::{GridVisibilityParams, GridVolumeParams, GridVolumeSource};

impl Renderer {
    /// Renders the solver's own grid mass field directly (see `grid_volume.wgsl`'s
    /// own doc for the real technique). Requires `set_camera` to have been called
    /// first (same as `render_gpu` needs for its own bind group) -- reuses the
    /// identical cached orthographic projection/grid_res so both modes line up on
    /// screen without re-deriving them.
    pub fn render_grid_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: GridVolumeSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        let grid_res = self.cached_grid_res;
        self.ensure_grid_visibility_capacity(device, grid_res);
        // Real per-particle cell-mass scale here is order 0.5-4 per occupied
        // cell; 0.15 requires non-trivial local density before showing anything,
        // instead of any measurable trace (which combined with bilinear smoothing
        // would overshoot true particle extent). SAME floor the visibility step
        // below gates on, so the hysteresis band and the raw discard agree.
        // Scaled by the caller's real full-cell mass (see
        // `Renderer::grid_reference_cell_mass`'s own doc). 0.15 is now a
        // FRACTION of a full cell -- "needs non-trivial local density before
        // showing anything" -- instead of an absolute number that silently
        // assumed a particular density calibration. Defaults to 1.0, so every
        // existing caller keeps the exact previous threshold.
        let mass_floor = 0.15 * self.grid_reference_cell_mass;
        queue.write_buffer(
            &self.grid_volume_params_buf,
            0,
            bytemuck::bytes_of(&GridVolumeParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                grid_res,
                mass_floor,
                material_mass_enabled: source.material_mass_enabled as u32,
                _pad1: 0.0,
                _pad2: [0.0, 0.0],
            }),
        );
        queue.write_buffer(
            &self.grid_visibility_params_buf,
            0,
            bytemuck::bytes_of(&GridVisibilityParams {
                grid_res,
                mass_floor,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        // Real hysteresis visibility step -- see `grid_volume.wgsl`'s own
        // `grid_visibility_step_main` doc. Reads the SAME raw grid buffer
        // the render pass below samples, must run before it in this
        // encoder so `fs_main`'s discard sees this frame's decision, not
        // last frame's.
        let grid_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_visibility_step_bg"),
            layout: &self.grid_visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.grid_visibility_params_buf.as_entire_binding(),
                },
            ],
        });

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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.grid_visibility_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_grid_volume"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_visibility_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.grid_visibility_step_pipeline);
            cp.set_bind_group(0, &grid_visibility_step_bg, &[]);
            cp.dispatch_workgroups(grid_res.div_ceil(8), grid_res.div_ceil(8), 1);
        }
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
}

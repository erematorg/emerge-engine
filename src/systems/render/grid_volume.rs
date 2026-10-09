//! Grid-native volumetric render path -- split out of `mod.rs`, same
//! "verbatim `impl Renderer` method, no behavior change" pattern already
//! used for `surface_reconstruction.rs`. Reads the shared MPM grid buffer
//! directly (per-cell mass/material), distinct from the particle-instance
//! paths (`render_gpu`/`render_slice`) and from curvature-flow surface
//! reconstruction.

use super::color::write_optical_table;
use super::gpu_types::{GridPeakParams, GridVolumeParams, GridVolumeSource};
use super::{LightPassSource, Renderer, free_surface_cell_step};

impl Renderer {
    /// Renders the solver's own grid mass field directly (see `grid_volume.wgsl`'s
    /// doc for the technique), through the camera `set_camera` or
    /// `set_camera_region` last set, so it lines up with the other modes. The
    /// grid's resolution comes from `source.grid_res`.
    pub fn render_grid_volume(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: GridVolumeSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        let grid_res = source.grid_res;
        self.ensure_grid_peak_capacity(device, grid_res);
        // Before the draw's bind group takes `light_transmittance_buf`.
        self.ensure_light_pass_capacity(device, grid_res);
        // Open air for the legacy column-depth term: a cell under 0.15 of a
        // full cell (see `Renderer::grid_reference_cell_mass`). The edge of
        // the matter no longer reads it (see `grid_peak_main`).
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
                reference_cell_mass: self.grid_reference_cell_mass,
                // The grid volume reads the physics grid itself: one cell
                // per physics cell.
                free_surface_step: self.grid_reference_cell_mass * free_surface_cell_step(1.0),
                _pad2: 0.0,
            }),
        );
        queue.write_buffer(
            &self.grid_peak_params_buf,
            0,
            bytemuck::bytes_of(&GridPeakParams {
                grid_res,
                _pad: [0; 3],
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
            &self.holds_shape,
        );

        // Local peak of the cell mass (`grid_volume.wgsl`'s `grid_peak_main`),
        // from the grid buffer the render pass below samples, encoded before
        // it so `fs_main` reads this frame's.
        let grid_peak_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_peak_bg"),
            layout: &self.grid_peak_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid_peak_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.grid_peak_params_buf.as_entire_binding(),
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
                    resource: self.grid_peak_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.physical_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.light_transmittance_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_grid_volume"),
        });
        self.encode_light_pass(
            device,
            queue,
            &mut enc,
            grid_res,
            LightPassSource::Grid {
                grid: source.grid,
                material_mass: source.material_mass,
                material_mass_enabled: source.material_mass_enabled,
            },
        );
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_peak"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.grid_peak_pipeline);
            cp.set_bind_group(0, &grid_peak_bg, &[]);
            cp.dispatch_workgroups(grid_res.div_ceil(8), grid_res.div_ceil(8), 1);
        }
        let load = if clear {
            wgpu::LoadOp::Clear(self.clear_color())
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

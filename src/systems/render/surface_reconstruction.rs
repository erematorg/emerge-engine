//! Curvature-flow surface reconstruction rendering -- split out of `mod.rs`
//! purely for LOC (this was ~1600 lines of a single 2945-line impl block,
//! matching this project's own established split-file-per-feature
//! convention, e.g. `spacetime::rod`'s `gravitropism.rs`/`forces.rs`).
//! No behavior change -- same `impl Renderer` methods, moved verbatim.
//!
//! See `curvature_flow.wgsl` for the technique (van der Laan et al.
//! 2009 mean curvature flow on a particle-splatted auxiliary buffer).

use super::*;

/// The ten GPU buffers one phase of the surface-reconstruction pipeline
/// binds. Bundled so `encode_phase_pipeline` needs no
/// `#[allow(clippy::too_many_arguments)]` -- fixing the cause (buffers that
/// always travel together as one phase's buffer set) rather than silencing
/// the lint. Pure regrouping: every buffer is passed through unchanged.
#[derive(Clone, Copy)]
struct PhasePipelineBuffers<'a> {
    params_buf: &'a wgpu::Buffer,
    atomic_buf: &'a wgpu::Buffer,
    a_buf: &'a wgpu::Buffer,
    b_buf: &'a wgpu::Buffer,
    raw_splat_history_buf: &'a wgpu::Buffer,
    temp_atomic_buf: &'a wgpu::Buffer,
    temp_float_buf: &'a wgpu::Buffer,
    pre_total_buf: &'a wgpu::Buffer,
    post_total_buf: &'a wgpu::Buffer,
    material_mass_buf: &'a wgpu::Buffer,
}

impl Renderer {
    // ── Curvature-flow surface reconstruction ──────────────────────────────────

    /// Surface reconstruction finer than the physics grid: van der Laan et
    /// al. 2009 mean curvature flow on a particle-splatted buffer whose
    /// resolution is independent of `grid_res` (see `curvature_flow.wgsl`).
    /// Needs `set_camera` first: its orthographic projection is reused,
    /// rescaled to `surface_res`. `material_slot` picks one `OpticalTable`
    /// slot for the whole surface (single dominant material, see the shader).
    pub fn render_surface_reconstruction(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: SurfaceReconstructionSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let SurfaceReconstructionSource {
            particle_buf,
            particle_count,
            grid_res,
            material_slot,
            material_mass_enabled,
            dt,
        } = source;
        if particle_count == 0 {
            return;
        }
        self.ensure_surface_capacity(device, grid_res);
        let surface_res = self.surface_res;
        if material_mass_enabled {
            self.ensure_surface_material_mass_capacity(device, surface_res);
        }
        // Before the draw's bind group takes `light_transmittance_buf`.
        self.ensure_light_pass_capacity(device, grid_res);
        self.ensure_light_surface_capacity(device, surface_res);

        queue.write_buffer(
            &self.surface_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: -1, // v1 behavior: every particle contributes
                material_mass_enabled: material_mass_enabled as u32,
                dt,
                splat_width_cells: self.splat_width_cells,
                anisotropy_strength: self.anisotropy_strength,
            }),
        );

        let (sx, tx, sy, ty) = self.surface_render_projection(grid_res, surface_res);
        let free_surface_step = self.grid_reference_cell_mass
            * free_surface_cell_step(
                surface_res as f32 / grid_res.max(1) as f32 * self.splat_width_cells,
            );

        queue.write_buffer(
            &self.surface_render_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                // See `SURFACE_MASS_FLOOR_FRACTION`'s doc: sized off what
                // ONE isolated particle's B-spline splat can concentrate into a
                // single surface cell, not copied from `render_grid_volume`'s
                // own (differently-diluted) units.
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                material_slot,
                material_mass_enabled: material_mass_enabled as u32,
                reference_cell_mass: self.grid_reference_cell_mass,
                edge_reference_depth: self.edge_reference_depth,
                free_surface_step,
                light_res: grid_res,
                _pad: [0; 2],
            }),
        );

        queue.write_buffer(
            &self.light_diffuse_params_buf,
            0,
            bytemuck::bytes_of(&LightDiffuseParams {
                surface_res,
                material_slot,
                emission_reference_k: self.emission_reference_k,
                display_white_mean: self.display_white_mean(),
                luminous_emission_w_m3: self.luminous_emission[material_slot as usize % 16],
                reference_cell_mass: self.grid_reference_cell_mass,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                wave_force_coeff: self.wave_force_coeff,
                _pad: [0; 2],
            }),
        );

        queue.write_buffer(
            &self.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        queue.write_buffer(
            &self.band_hysteresis_params_buf,
            0,
            bytemuck::bytes_of(&BandHysteresisParams {
                surface_res,
                reference_cell_mass: self.grid_reference_cell_mass,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_splat_bg"),
            layout: &self.surface_splat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pre_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.post_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_moments_buf.as_entire_binding(),
                },
            ],
        });
        // clear_surface_main only ever reads binding 1/2/3/4/5/6 (its own
        // atomic buffers + params); binding 0 is unused but the layout is
        // shared with the splat pass, so bind SOMETHING real there too.
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_clear_bg"),
            layout: &self.surface_clear_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pre_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.post_total_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_moments_buf.as_entire_binding(),
                },
            ],
        });
        let convert_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_convert_bg"),
            layout: &self.surface_convert_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.raw_splat_history_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.surface_temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_iterate_a_to_b"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_iterate_b_to_a"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        // Thermal-diffusion pair (`curvature_flow.wgsl`'s Pass 1c):
        // `temp_avg_bg` recovers temperature from the mass-weighted scatter
        // with this frame's settled density (`surface_a_buf`);
        // `temp_diffuse_bg` runs one heat-equation step from
        // `surface_temp_b_buf` back into `surface_temp_float_buf`, the buffer
        // `fs_main` binds.
        let temp_avg_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temp_avg_bg"),
            layout: &self.temp_avg_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_temp_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        let temp_diffuse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temp_diffuse_bg"),
            layout: &self.temp_diffuse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_temp_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_params_buf.as_entire_binding(),
                },
            ],
        });
        // Light-diffusion pair (`curvature_flow.wgsl`'s Pass 1e), after
        // temp_diffuse: its emission source is the diffused temperature.
        // `light_cur_idx`/`light_next_idx` alternate the 2 ping-pong buffers
        // (see `light_frame_index` for why 2 suffice where the wave field
        // needs 3).
        let light_cur_idx = (self.light_frame_index % 2) as usize;
        let light_next_idx = ((self.light_frame_index + 1) % 2) as usize;
        let light_diffuse_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_diffuse_bg"),
            layout: &self.light_diffuse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.light_phi_bufs[light_cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.light_phi_bufs[light_next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.light_diffuse_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
            ],
        });
        // Persistent wave field (`curvature_flow.wgsl`'s Pass 2b). Three
        // buffers rotate through "current" (read with neighbour offsets),
        // "previous" (own index, read) and "next" (own index, write): wgpu
        // rejects one buffer bound read-only and read_write in one dispatch,
        // even for index-disjoint access. After this dispatch the "next"
        // buffer holds the new state, which `render_bg` reads.
        let cur_idx = (self.wave_frame_index % 3) as usize;
        let prev_idx = ((self.wave_frame_index + 2) % 3) as usize;
        let next_idx = ((self.wave_frame_index + 1) % 3) as usize;
        let wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });

        // Hysteresis visibility step: one persistent buffer (own index only,
        // no stencil, so no usage-scope conflict), reading this frame's
        // settled density (`surface_a_buf`).
        let visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });

        // Hysteresis color-band step, the same single-buffer shape (Pass 2d).
        let band_hysteresis_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("band_hysteresis_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });

        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_render_bg"),
            layout: &self.surface_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.surface_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.surface_temp_float_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_material_mass_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.light_phi_bufs[light_next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.light_transmittance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.physical_render_params_buf.as_entire_binding(),
                },
            ],
        });

        let cell_count = surface_res * surface_res;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_surface_reconstruction"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_clear"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_clear_pipeline);
            cp.set_bind_group(0, &clear_bg, &[]);
            // Covers both the surface buffers (sized off `surface_res`) and
            // the moments buffer (sized off `grid_res`); neither is reliably
            // the larger at every `surface_res_multiplier`, and the shader
            // guards each store against its own range.
            let clear_threads = cell_count.max(grid_res * grid_res * MOMENTS_PER_CELL as u32);
            cp.dispatch_workgroups(clear_threads.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        if self.anisotropy_strength > 0.0 {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_moments"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_moments_pipeline);
            cp.set_bind_group(0, &splat_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(SURFACE_SPLAT_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_splat"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_splat_pipeline);
            cp.set_bind_group(0, &splat_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(SURFACE_SPLAT_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_convert"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_convert_pipeline);
            cp.set_bind_group(0, &convert_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        // `temp_avg_main` divides the mass-weighted temperature sum by mass,
        // which only means something against the mass the sum was splatted
        // with. Both the curvature iterate loop and the volume correction
        // change `surface_a_buf`, so this runs straight after convert, raw
        // over raw. (Run after the smoothing, a raw numerator over a smoothed
        // denominator blew up wherever mass moved out of a cell: rim cells of
        // 300 K water rendered saturated red.) `temp_diffuse` below still
        // smooths the result.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temp_avg"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.temp_avg_pipeline);
            cp.set_bind_group(0, &temp_avg_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        // `CURVATURE_ITERATIONS` real dispatches, ping-ponged -- kept EVEN so
        // the settled result always lands back in `surface_a` (see this
        // struct's own field doc, and the module-level const assertion
        // below), letting `render_bg` above bind `surface_a`
        // unconditionally rather than choosing at runtime.
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        for i in 0..self.curvature_iterations {
            let bg = if i % 2 == 0 {
                &iterate_a_to_b
            } else {
                &iterate_b_to_a
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("surface_iterate"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_iterate_pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Thermal-diffusion step (`curvature_flow.wgsl`'s Pass 1c), before
        // the render pass reads `surface_temp_float_buf` and before the
        // volume correction below: rescaling mass first shifted every
        // recovered temperature by the rescale factor (a test caught it). An
        // average temperature is intensive and needs no volume preservation.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temp_diffuse"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.temp_diffuse_pipeline);
            cp.set_bind_group(0, &temp_diffuse_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Light-diffusion step, after temp_diffuse (see `light_diffuse_bg`).
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("light_diffuse"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.light_diffuse_pipeline);
            cp.set_bind_group(0, &light_diffuse_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        self.light_frame_index = self.light_frame_index.wrapping_add(1);
        // Volume-preserving correction (`curvature_flow.wgsl`'s Pass 1d),
        // after temperature recovery and before the wave, visibility and band
        // steps and the render pass, which all use the corrected mass.
        {
            let post_total_reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post_total_reduce_bg"),
                layout: &self.post_total_reduce_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.surface_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.post_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.surface_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("post_total_reduce"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.post_total_reduce_pipeline);
            cp.set_bind_group(0, &post_total_reduce_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let volume_correct_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volume_correct_bg"),
                layout: &self.volume_correct_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.pre_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.post_total_atomic_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.surface_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.surface_params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("volume_correct"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.volume_correct_pipeline);
            cp.set_bind_group(0, &volume_correct_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        // First real frame since `wave_density_prev_buf` was last a zero
        // placeholder: seed it from this frame's own settled density so the
        // wave step below reads prev == now (force = 0) instead of a false
        // "density appeared from nothing" kick. See `wave_prev_seeded`'s own
        // doc.
        if !self.wave_prev_seeded {
            enc.copy_buffer_to_buffer(
                &self.surface_a_buf,
                0,
                &self.wave_density_prev_buf,
                0,
                (cell_count as u64) * mem::size_of::<f32>() as u64,
            );
            self.wave_prev_seeded = true;
        }
        // Wave-equation step, excited by this frame's settled density
        // (`surface_a_buf`): after the iterate loop, before the render pass
        // reads the wave field.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wave_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.wave_step_pipeline);
            cp.set_bind_group(0, &wave_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Snapshot this frame's settled density as "previous" for next
        // frame's temporal-disturbance wave excitation. Must happen AFTER
        // the wave step above (which needed this frame's density as "now").
        // Plain buffer copy, no shader needed.
        enc.copy_buffer_to_buffer(
            &self.surface_a_buf,
            0,
            &self.wave_density_prev_buf,
            0,
            (cell_count as u64) * mem::size_of::<f32>() as u64,
        );
        // Hysteresis visibility step, also from the settled density;
        // independent of the wave step (neither reads the other).
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.visibility_step_pipeline);
            cp.set_bind_group(0, &visibility_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Hysteresis color-band step, also from the settled density.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("band_hysteresis_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.band_hysteresis_step_pipeline);
            cp.set_bind_group(0, &band_hysteresis_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Light pass from the settled density, the same field the draw reads.
        self.encode_light_pass(
            device,
            queue,
            &mut enc,
            grid_res,
            LightPassSource::Surface {
                density: &self.surface_a_buf,
                material_mass: &self.surface_material_mass_buf,
                material_mass_enabled,
                surface_res,
                material_slot,
            },
        );

        let load = if clear {
            wgpu::LoadOp::Clear(self.clear_color())
        } else {
            wgpu::LoadOp::Load
        };
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_render"),
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
            rp.set_pipeline(&self.surface_render_pipeline);
            rp.set_bind_group(0, &render_bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        // Rotate which of the 3 wave buffers plays which role next call --
        // see the bind-group construction above's doc for the full
        // rotation reasoning.
        self.wave_frame_index = self.wave_frame_index.wrapping_add(1);
    }

    /// Encodes clear+splat+convert+`CURVATURE_ITERATIONS`-iterate for ONE
    /// phase into the shared encoder -- the common body both
    /// `render_surface_reconstruction` (inlined, unchanged, zero risk to
    /// already-shipped code) and `render_surface_reconstruction_dual_phase`
    /// (below, calls this twice) both need. `params_buf` must already carry
    /// the `phase_filter_material_id` for this specific phase.
    fn encode_phase_pipeline(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        surface_res: u32,
        bufs: PhasePipelineBuffers<'_>,
        // N-material extension's own buffer -- always bound (layout is
        // shared with the single-phase path), but a harmless dead-code
        // path here: both dual-phase calls set `material_mass_enabled: 0`
        // in `params_buf`, so the shader branch that reads this never
        // executes.
    ) {
        let PhasePipelineBuffers {
            params_buf,
            atomic_buf,
            a_buf,
            b_buf,
            raw_splat_history_buf,
            temp_atomic_buf,
            temp_float_buf,
            pre_total_buf,
            post_total_buf,
            material_mass_buf,
        } = bufs;
        let cell_count = surface_res * surface_res;
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_clear_bg"),
            layout: &self.surface_clear_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pre_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: post_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_mass_buf.as_entire_binding(),
                },
                // Shared across both phases, not per-phase: the fit is a
                // geometric property of where matter is, see the moments
                // buffer's own shader doc.
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_moments_buf.as_entire_binding(),
                },
            ],
        });
        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_splat_bg"),
            layout: &self.surface_splat_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pre_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: post_total_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_mass_buf.as_entire_binding(),
                },
                // Shared across both phases, not per-phase: the fit is a
                // geometric property of where matter is, see the moments
                // buffer's own shader doc.
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.surface_moments_buf.as_entire_binding(),
                },
            ],
        });
        let convert_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_convert_bg"),
            layout: &self.surface_convert_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_splat_history_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: temp_atomic_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: temp_float_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_iterate_a_to_b"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let iterate_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_iterate_b_to_a"),
            layout: &self.surface_iterate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_clear"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_clear_pipeline);
            cp.set_bind_group(0, &clear_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_splat"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_splat_pipeline);
            cp.set_bind_group(0, &splat_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(SURFACE_SPLAT_WG), 1, 1);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_convert"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_convert_pipeline);
            cp.set_bind_group(0, &convert_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        for i in 0..self.curvature_iterations {
            let bg = if i % 2 == 0 {
                &iterate_a_to_b
            } else {
                &iterate_b_to_a
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_iterate"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.surface_iterate_pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Volume-preserving correction for this phase, as in
        // `render_surface_reconstruction` (Pass 1d).
        {
            let post_total_reduce_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phase_post_total_reduce_bg"),
                layout: &self.post_total_reduce_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: post_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_post_total_reduce"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.post_total_reduce_pipeline);
            cp.set_bind_group(0, &post_total_reduce_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
        {
            let volume_correct_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("phase_volume_correct_bg"),
                layout: &self.volume_correct_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pre_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: post_total_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("phase_volume_correct"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.volume_correct_pipeline);
            cp.set_bind_group(0, &volume_correct_bg, &[]);
            cp.dispatch_workgroups(cell_count.div_ceil(SURFACE_CLEAR_WG), 1, 1);
        }
    }

    /// Two-phase extension of `render_surface_reconstruction` (see
    /// `curvature_flow.wgsl`'s own "two-phase extension" doc and
    /// `DualPhaseSurfaceSource`'s doc): runs the clear/splat/convert/
    /// iterate pipeline TWICE, once per material, into two fully
    /// independent buffer sets, so each phase gets its own real,
    /// independently-smoothed surface instead of merging at a shared
    /// interface into one blob. The final composite picks whichever
    /// phase has more real local density at each pixel (discrete
    /// winner-take-all, matching `grid_volume.wgsl`'s own "dominant
    /// material wins" convention).
    pub fn render_surface_reconstruction_dual_phase(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: DualPhaseSurfaceSource,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let DualPhaseSurfaceSource {
            particle_buf,
            particle_count,
            grid_res,
            material_id_a,
            material_id_b,
            dt,
        } = source;
        if particle_count == 0 {
            return;
        }
        self.ensure_surface_capacity(device, grid_res);
        let surface_res = self.surface_res;

        queue.write_buffer(
            &self.surface_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: material_id_a as i32,
                // N-material extension is a single-phase-only mechanism,
                // unrelated to this 2-phase filter -- always off here.
                material_mass_enabled: 0,
                dt,
                splat_width_cells: self.splat_width_cells,
                anisotropy_strength: self.anisotropy_strength,
            }),
        );
        queue.write_buffer(
            &self.phase_b_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceParams {
                grid_res,
                surface_res,
                particle_count: particle_count as u32,
                phase_filter_material_id: material_id_b as i32,
                material_mass_enabled: 0,
                dt,
                splat_width_cells: self.splat_width_cells,
                anisotropy_strength: self.anisotropy_strength,
            }),
        );

        // Identical for both phases -- they share one camera/surface_res.
        let (sx, tx, sy, ty) = self.surface_render_projection(grid_res, surface_res);
        let free_surface_step = self.grid_reference_cell_mass
            * free_surface_cell_step(
                surface_res as f32 / grid_res.max(1) as f32 * self.splat_width_cells,
            );

        queue.write_buffer(
            &self.surface_render_params_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                material_slot: material_id_a,
                material_mass_enabled: 0,
                reference_cell_mass: self.grid_reference_cell_mass,
                edge_reference_depth: self.edge_reference_depth,
                free_surface_step,
                light_res: grid_res,
                _pad: [0; 2],
            }),
        );
        queue.write_buffer(
            &self.render_params_b_buf,
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                sx,
                tx,
                sy,
                ty,
                light_dir: [self.light_dir.0, self.light_dir.1],
                surface_res,
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                material_slot: material_id_b,
                material_mass_enabled: 0,
                reference_cell_mass: self.grid_reference_cell_mass,
                edge_reference_depth: self.edge_reference_depth,
                free_surface_step,
                light_res: grid_res,
                _pad: [0; 2],
            }),
        );

        // Wave/visibility/band params shared by both phases (only
        // `surface_res`/`mass_floor` matter, identical for both): one write
        // covers both dispatches, as in the single-phase path.
        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                wave_force_coeff: self.wave_force_coeff,
                _pad: [0; 2],
            }),
        );
        queue.write_buffer(
            &self.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        queue.write_buffer(
            &self.band_hysteresis_params_buf,
            0,
            bytemuck::bytes_of(&BandHysteresisParams {
                surface_res,
                reference_cell_mass: self.grid_reference_cell_mass,
                _pad1: 0,
                _pad2: 0,
            }),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_surface_reconstruction_dual_phase"),
        });

        self.encode_phase_pipeline(
            device,
            &mut enc,
            particle_buf,
            particle_count,
            surface_res,
            PhasePipelineBuffers {
                params_buf: &self.surface_params_buf,
                atomic_buf: &self.surface_atomic_buf,
                a_buf: &self.surface_a_buf,
                b_buf: &self.surface_b_buf,
                raw_splat_history_buf: &self.raw_splat_history_buf,
                temp_atomic_buf: &self.surface_temp_atomic_buf,
                temp_float_buf: &self.surface_temp_float_buf,
                pre_total_buf: &self.pre_total_atomic_buf,
                post_total_buf: &self.post_total_atomic_buf,
                material_mass_buf: &self.surface_material_mass_buf,
            },
        );
        self.encode_phase_pipeline(
            device,
            &mut enc,
            particle_buf,
            particle_count,
            surface_res,
            PhasePipelineBuffers {
                params_buf: &self.phase_b_params_buf,
                atomic_buf: &self.phase_b_atomic_buf,
                a_buf: &self.phase_b_a_buf,
                b_buf: &self.phase_b_b_buf,
                raw_splat_history_buf: &self.phase_b_raw_splat_history_buf,
                temp_atomic_buf: &self.phase_b_temp_atomic_buf,
                temp_float_buf: &self.phase_b_temp_float_buf,
                pre_total_buf: &self.phase_b_pre_total_atomic_buf,
                post_total_buf: &self.phase_b_post_total_atomic_buf,
                material_mass_buf: &self.surface_material_mass_buf,
            },
        );

        // Per-phase wave/visibility/band steps with the same
        // `wave_frame_index` rotation (see `render_surface_reconstruction`),
        // on phase A's buffers and phase B's own set. After both
        // `encode_phase_pipeline` calls (they settle `surface_a_buf`/
        // `phase_b_a_buf`) and before the render pass.
        let iterate_wg_x = surface_res.div_ceil(8);
        let iterate_wg_y = surface_res.div_ceil(8);
        let cur_idx = (self.wave_frame_index % 3) as usize;
        let prev_idx = ((self.wave_frame_index + 2) % 3) as usize;
        let next_idx = ((self.wave_frame_index + 1) % 3) as usize;
        let phase_a_wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_wave_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_wave_step_bg"),
            layout: &self.wave_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_wave_bufs[cur_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.phase_b_wave_bufs[prev_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.phase_b_wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.wave_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.phase_b_wave_density_prev_buf.as_entire_binding(),
                },
            ],
        });
        let phase_a_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_visibility_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_visibility_step_bg"),
            layout: &self.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_a_band_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_a_band_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });
        let phase_b_band_step_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phase_b_band_step_bg"),
            layout: &self.band_hysteresis_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.band_hysteresis_params_buf.as_entire_binding(),
                },
            ],
        });
        // First real frame since either phase's `*_wave_density_prev_buf`
        // was last a zero placeholder: seed it from this frame's own
        // settled density before its wave step runs. See the single-phase
        // path's `wave_prev_seeded` check for the full doc.
        let wave_history_bytes = (surface_res * surface_res) as u64 * mem::size_of::<f32>() as u64;
        if !self.wave_prev_seeded {
            enc.copy_buffer_to_buffer(
                &self.surface_a_buf,
                0,
                &self.wave_density_prev_buf,
                0,
                wave_history_bytes,
            );
            self.wave_prev_seeded = true;
        }
        if !self.phase_b_wave_prev_seeded {
            enc.copy_buffer_to_buffer(
                &self.phase_b_a_buf,
                0,
                &self.phase_b_wave_density_prev_buf,
                0,
                wave_history_bytes,
            );
            self.phase_b_wave_prev_seeded = true;
        }
        for (label, bg) in [
            ("phase_a_wave_step", &phase_a_wave_step_bg),
            ("phase_b_wave_step", &phase_b_wave_step_bg),
            ("phase_a_visibility_step", &phase_a_visibility_step_bg),
            ("phase_b_visibility_step", &phase_b_visibility_step_bg),
            ("phase_a_band_step", &phase_a_band_step_bg),
            ("phase_b_band_step", &phase_b_band_step_bg),
        ] {
            let pipeline = if label.ends_with("wave_step") {
                &self.wave_step_pipeline
            } else if label.ends_with("visibility_step") {
                &self.visibility_step_pipeline
            } else {
                &self.band_hysteresis_step_pipeline
            };
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            cp.set_pipeline(pipeline);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Per-phase snapshot -- feeds the NEXT frame's wave step. Always
        // unconditional (unlike the seed above): every frame's own settled
        // density becomes next frame's "previous", not just the first.
        enc.copy_buffer_to_buffer(
            &self.surface_a_buf,
            0,
            &self.wave_density_prev_buf,
            0,
            wave_history_bytes,
        );
        enc.copy_buffer_to_buffer(
            &self.phase_b_a_buf,
            0,
            &self.phase_b_wave_density_prev_buf,
            0,
            wave_history_bytes,
        );

        let dual_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_dual_render_bg"),
            layout: &self.surface_dual_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.phase_b_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.surface_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.render_params_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.phase_b_wave_bufs[next_idx].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.phase_b_visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.phase_b_band_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.physical_render_params_buf.as_entire_binding(),
                },
            ],
        });

        let load = if clear {
            wgpu::LoadOp::Clear(self.clear_color())
        } else {
            wgpu::LoadOp::Load
        };
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_dual_render"),
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
            rp.set_pipeline(&self.surface_dual_render_pipeline);
            rp.set_bind_group(0, &dual_bg, &[]);
            rp.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(enc.finish()));
        // Rotate which of the 3 wave-buffer slots plays which role next call
        // -- same real rotation `render_surface_reconstruction` performs,
        // shared here since both phases index off the SAME frame counter
        // (see the wave-step bind-group construction above's doc).
        self.wave_frame_index = self.wave_frame_index.wrapping_add(1);
    }

    /// The orthographic projection for THIS pass's own, finer coordinate
    /// units -- shared by `render_surface_reconstruction` and its
    /// dual-phase sibling (identical for both, one camera/surface_res), so
    /// they can't independently drift like `cursor_grid`/`set_camera` once
    /// did. Not a fresh computation: `set_camera`'s `sx/sy` scale as
    /// `1/grid_res` and `tx/ty` are resolution-independent, so rescaling the
    /// cached transform by `grid_res/surface_res` on `sx/sy` alone gives the
    /// same projection in this pass's own units.
    fn surface_render_projection(&self, grid_res: u32, surface_res: u32) -> (f32, f32, f32, f32) {
        let (sx_grid, tx, sy_grid, ty) = self.cached_ortho;
        let res_ratio = grid_res as f32 / surface_res as f32;
        (sx_grid * res_ratio, tx, sy_grid * res_ratio, ty)
    }

    /// Grows the three curvature-flow surface buffers together when a
    /// caller's `grid_res * SURFACE_RES_MULTIPLIER` exceeds the currently
    /// allocated `surface_res` -- same lazy-growth pattern `ensure_capacity`
    /// already uses for the particle instance buffers.
    pub(super) fn ensure_surface_capacity(&mut self, device: &wgpu::Device, grid_res: u32) {
        // Grown BEFORE the surface early-return below: the moments buffer is
        // sized off `grid_res` alone, so it can need growing on a call where
        // the (multiplier-scaled) surface buffers do not.
        if grid_res > self.surface_moments_res {
            self.surface_moments_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surface_moments"),
                size: (grid_res as u64)
                    * (grid_res as u64)
                    * MOMENTS_PER_CELL
                    * mem::size_of::<i32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.surface_moments_res = grid_res;
        }
        let needed = grid_res * self.surface_res_multiplier;
        if needed <= self.surface_res {
            return;
        }
        let cell_count = (needed * needed) as u64;
        let float_size = cell_count * mem::size_of::<f32>() as u64;
        self.surface_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Mass-weighted temperature pair, grown together (see
        // `surface_temp_atomic_buf`).
        self.surface_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_float"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Ping-pong partner, grown together -- see `temp_avg_pipeline`'s own
        // doc.
        self.surface_temp_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_temp_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_a"),
            size: float_size,
            // COPY_SRC: see the constructor's own placeholder allocation doc.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Two-phase extension's own phase-B buffers, same sizes -- grown
        // together so `render_surface_reconstruction_dual_phase` never has
        // to special-case a mismatched capacity between the two phases.
        self.phase_b_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Phase B's own temperature pair, grown together -- see
        // `surface_temp_atomic_buf`'s doc.
        self.phase_b_temp_atomic_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_atomic"),
            size: cell_count * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_temp_float_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_temp_float"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_a"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_b"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.phase_b_raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_raw_splat_history"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Light-fluence field, grown together; a resize resets it to zero, as
        // the wave field below.
        self.light_phi_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "light_phi_0",
                    _ => "light_phi_1",
                }),
                size: float_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        // Wave field, grown together; a resize resets it to flat, as
        // `surface_a_buf`/`surface_b_buf`.
        self.wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "wave_0",
                    1 => "wave_1",
                    _ => "wave_2",
                }),
                size: float_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        // Temporal-disturbance history, grown together (see
        // `wave_density_prev_buf`).
        self.wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_density_prev"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Back to a zero placeholder -- needs reseeding before the next
        // `wave_step`, see `wave_prev_seeded`'s doc.
        self.wave_prev_seeded = false;
        // Hysteresis visibility state, grown together; a resize resets it to
        // "not visible", like the constructor's placeholder.
        self.visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Hysteresis color-band state, grown together; a resize resets it to
        // band 0, like the constructor's placeholder.
        self.band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("band_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.raw_splat_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_splat_history"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Phase B's wave/visibility/band state, grown together, with the same
        // resets as phase A's.
        self.phase_b_wave_bufs = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "phase_b_wave_0",
                    1 => "phase_b_wave_1",
                    _ => "phase_b_wave_2",
                }),
                size: float_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        self.phase_b_wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_wave_density_prev"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_wave_prev_seeded = false;
        self.phase_b_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_visibility_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.phase_b_band_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phase_b_band_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_res = needed;
    }

    /// N-material extension (see `surface_material_mass_buf`): grows it to
    /// `surface_res² × MAX_RENDER_MATERIAL_SLOTS × 4` bytes. Not called from
    /// `ensure_surface_capacity`: only when a caller opts in
    /// (`SurfaceReconstructionSource::material_mass_enabled`), so a
    /// `Renderer` that never does keeps the 4-byte placeholder, like
    /// `GpuBuffers::grow_material_mass` on the solver side.
    pub(super) fn ensure_surface_material_mass_capacity(
        &mut self,
        device: &wgpu::Device,
        surface_res: u32,
    ) {
        if self.surface_material_mass_res >= surface_res {
            return;
        }
        let cell_count = (surface_res * surface_res) as u64;
        self.surface_material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_material_mass"),
            size: cell_count * MAX_RENDER_MATERIAL_SLOTS as u64 * mem::size_of::<i32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.surface_material_mass_res = surface_res;
    }

    /// Grows `grid_peak_buf` when a caller's `grid_res` exceeds the
    /// currently allocated `grid_peak_res` -- same lazy-growth
    /// pattern `ensure_surface_capacity` uses above, just keyed on the
    /// solver's own `grid_res` instead of the finer `surface_res`.
    pub(super) fn ensure_grid_peak_capacity(&mut self, device: &wgpu::Device, grid_res: u32) {
        if grid_res <= self.grid_peak_res {
            return;
        }
        let cell_count = (grid_res * grid_res) as u64;
        self.grid_peak_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_peak"),
            size: cell_count * mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.grid_peak_res = grid_res;
    }
}

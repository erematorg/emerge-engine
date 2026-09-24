//! Curvature-flow surface reconstruction rendering -- split out of `mod.rs`
//! purely for LOC (this was ~1600 lines of a single 2945-line impl block,
//! matching this project's own established split-file-per-feature
//! convention, e.g. `spacetime::rod`'s `gravitropism.rs`/`forces.rs`).
//! No behavior change -- same `impl Renderer` methods, moved verbatim.
//!
//! See `curvature_flow.wgsl` for the real technique (van der Laan et al.
//! 2009 mean curvature flow on a particle-splatted auxiliary buffer).

use super::*;

impl Renderer {
    // ── Curvature-flow surface reconstruction ──────────────────────────────────

    /// Real, finer-than-physics-grid surface reconstruction (see
    /// `curvature_flow.wgsl`'s own top doc for the full real technique --
    /// van der Laan et al. 2009 mean curvature flow on a particle-splatted
    /// auxiliary buffer, resolution-independent from the solver's own
    /// `grid_res`). Requires `set_camera` to have been called first (reuses
    /// its cached orthographic projection, rescaled to this pass's own
    /// finer `surface_res` -- see the real derivation in this function's
    /// body). `material_slot` picks ONE `OpticalTable` slot for the whole
    /// surface (real v1 scope: single dominant material, not full per-
    /// material phase-fraction separation -- see the shader's own doc).
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
                // See `SURFACE_MASS_FLOOR_FRACTION`'s own doc: sized off what
                // ONE isolated particle's B-spline splat can concentrate into a
                // single surface cell, not copied from `render_grid_volume`'s
                // own (differently-diluted) units.
                mass_floor: SURFACE_MASS_FLOOR_FRACTION * self.grid_reference_cell_mass,
                material_slot,
                material_mass_enabled: material_mass_enabled as u32,
                reference_cell_mass: self.grid_reference_cell_mass,
                edge_reference_depth: self.edge_reference_depth,
            }),
        );

        queue.write_buffer(
            &self.light_diffuse_params_buf,
            0,
            bytemuck::bytes_of(&LightDiffuseParams {
                surface_res,
                material_slot,
                _pad0: 0,
                _pad1: 0,
            }),
        );

        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                _pad: [0; 3],
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
        // Real thermal-diffusion pair (see `curvature_flow.wgsl`'s own
        // "Pass 1c" doc): `temp_avg_bg` recovers real temperature from the
        // mass-weighted scatter using THIS frame's now-settled density
        // (`surface_a_buf`); `temp_diffuse_bg` then runs one real heat-
        // equation step, reading `surface_temp_b_buf` (avg's output) and
        // writing back into `surface_temp_float_buf` (the buffer `fs_main`
        // already binds -- no render-bind-group change needed).
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
        // Real light-diffusion pair (`curvature_flow.wgsl`'s "Pass 1e") --
        // must run AFTER temp_diffuse above (needs the FINAL, already-
        // diffused temperature as its own real emission source, same real
        // ordering reasoning `temp_diffuse_bg`'s own doc gives for running
        // after the density iterate loop). `light_cur_idx`/`light_next_idx`
        // alternate which of the 2 ping-pong buffers is read vs written
        // this frame -- simpler 2-way rotation than the wave field's own
        // 3-way (see `light_frame_index`'s own doc for why one fewer buffer
        // suffices here).
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
            ],
        });
        // Real, persistent (across frames) wave field -- see `curvature_
        // flow.wgsl`'s own "Pass 2b" doc. THREE distinct physical buffers
        // rotate through the "current" (read with neighbor offsets),
        // "previous" (self-index read only), and "next" (self-index write
        // only) roles each call -- wgpu's usage-scope validator rejects
        // binding the SAME buffer as both read-only and read_write within
        // one dispatch (confirmed via a real validation error when a
        // cheaper 2-buffer aliasing scheme was tried first), even though
        // that scheme's actual access pattern was index-disjoint and
        // logically hazard-free -- 3 buffers is the correct, always-valid
        // way to satisfy that rule for a leapfrog integrator. After this
        // dispatch, the buffer that served as "next" holds the freshly
        // computed state, which is what `render_bg` below must read.
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

        // Real hysteresis visibility step -- a SINGLE persistent buffer
        // (no rotation needed: self-index read+write only, no neighbor
        // stencil, so no wgpu usage-scope conflict). Reads this frame's
        // settled density (`surface_a_buf`), same as the wave step above.
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

        // Real hysteresis color-band step -- same single-buffer, one-way-
        // downstream shape as the visibility step above (see "Pass 2d"
        // doc in the shader).
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
        // ROOT FIX (2026-08-13): `temp_avg_main` recovers per-cell AVERAGE
        // temperature by dividing the raw mass-weighted temperature sum by
        // mass -- and, in this file's own already-stated words, "that ratio is
        // only meaningful against the mass value the weighted sum was
        // ORIGINALLY splatted against".
        //
        // That principle was previously enforced against ONE mutator of
        // `surface_a_buf` (the volume-preserving correction) by moving this
        // pass before it -- but the curvature iterate loop below mutates the
        // very same buffer, and used to run FIRST. So the division paired a
        // RAW numerator with a SMOOTHED denominator: wherever curvature flow
        // moves mass out of a cell the denominator shrinks while the numerator
        // does not, and the recovered "temperature" blows up far past any real
        // value.
        //
        // Visible symptom that traced back to here: `fs_main`'s blackbody term
        // is `heat(0.5 + t_norm*0.5) * t_norm^2 * 2` with
        // `t_norm = clamp(avg_temp/5000, 0, 1)`, and `heat(1.0)` is PURE RED.
        // So rim/thin cells of 300 K water rendered as saturated red-pink
        // added on top of the correct body colour -- measured as body
        // (0.078, 0.43, 0.59) vs artifact (1.0, 0.43, 0.59): identical green
        // and blue, red alone driven to 1.0, which is exactly an additive
        // heat(1.0).
        //
        // Running it HERE -- straight after convert, before any smoothing --
        // divides raw by raw, which is the pairing the algorithm actually
        // requires. `temp_diffuse` below still smooths the resulting
        // temperature field, so nothing is lost.
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
        // Real thermal-diffusion pair (see `curvature_flow.wgsl`'s own
        // "Pass 1c" doc) -- must run AFTER the density iterate loop above
        // (needs the final settled `surface_a_buf`) and BEFORE the render
        // pass below reads `surface_temp_float_buf` for blackbody emission.
        // Real ordering requirement, caught by a real test failure: this
        // MUST run BEFORE the volume-preserving correction below --
        // `temp_avg_main` divides `surface_temp_float_buf` (raw weighted
        // sum) by `surface_a_buf` (mass) to recover a real per-cell AVERAGE
        // temperature; that ratio is only meaningful against the mass value
        // the weighted sum was ORIGINALLY splatted against. Rescaling mass
        // first (for a completely different, unrelated reason -- volume
        // preservation) before this division silently corrupted the
        // recovered temperature by the same rescale factor (confirmed via
        // the real test: hot/cold both shifted by the identical ratio).
        // Average temperature is an INTENSIVE quantity -- it doesn't need
        // "volume preservation" at all, so it must be computed from the
        // real, as-settled mass, before that mass is corrected for anything
        // else.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temp_diffuse"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.temp_diffuse_pipeline);
            cp.set_bind_group(0, &temp_diffuse_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real light-diffusion step -- see `light_diffuse_bg`'s own doc for
        // why this must run after temp_diffuse above.
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
        // Real volume-preserving correction (see `curvature_flow.wgsl`'s own
        // "Pass 1d" doc) -- must run AFTER temperature recovery above (see
        // that step's own doc for why) and BEFORE the wave/visibility/band
        // steps and the render pass, all of which need the CORRECTED mass
        // for alpha/banding/excitation purposes.
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
        // Real wave-equation step, reading this frame's now-settled density
        // (`surface_a_buf`) as its excitation source -- must run AFTER the
        // iterate loop above finishes writing it, and BEFORE the render
        // pass below reads the wave field for shading.
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
        // Real hysteresis visibility step -- also reads this frame's now-
        // settled density, independent of the wave step above (order
        // between the two doesn't matter, neither reads the other's
        // output).
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.visibility_step_pipeline);
            cp.set_bind_group(0, &visibility_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
        }
        // Real hysteresis color-band step -- also reads this frame's now-
        // settled density, independent of the wave/visibility steps above.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("band_hysteresis_step"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.band_hysteresis_step_pipeline);
            cp.set_bind_group(0, &band_hysteresis_step_bg, &[]);
            cp.dispatch_workgroups(iterate_wg_x, iterate_wg_y, 1);
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
        // see the bind-group construction above's own doc for the full
        // rotation reasoning.
        self.wave_frame_index = self.wave_frame_index.wrapping_add(1);
    }

    /// Encodes clear+splat+convert+`CURVATURE_ITERATIONS`-iterate for ONE
    /// phase into the shared encoder -- the real, common body both
    /// `render_surface_reconstruction` (inlined, unchanged, zero risk to
    /// already-shipped code) and `render_surface_reconstruction_dual_phase`
    /// (below, calls this twice) both need. `params_buf` must already carry
    /// the real `phase_filter_material_id` for this specific phase.
    #[allow(clippy::too_many_arguments)]
    fn encode_phase_pipeline(
        &self,
        device: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        surface_res: u32,
        params_buf: &wgpu::Buffer,
        atomic_buf: &wgpu::Buffer,
        a_buf: &wgpu::Buffer,
        b_buf: &wgpu::Buffer,
        raw_splat_history_buf: &wgpu::Buffer,
        temp_atomic_buf: &wgpu::Buffer,
        temp_float_buf: &wgpu::Buffer,
        pre_total_buf: &wgpu::Buffer,
        post_total_buf: &wgpu::Buffer,
        // N-material extension's own buffer -- always bound (layout is
        // shared with the single-phase path), but a harmless dead-code
        // path here: both dual-phase calls set `material_mass_enabled: 0`
        // in `params_buf`, so the shader branch that reads this never
        // executes.
        material_mass_buf: &wgpu::Buffer,
    ) {
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
        // Real volume-preserving correction for THIS phase -- see
        // `render_surface_reconstruction`'s own identical step for the full
        // doc (`curvature_flow.wgsl`'s "Pass 1d").
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
    /// `DualPhaseSurfaceSource`'s doc): runs the real clear/splat/convert/
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
            }),
        );

        // Real per-phase wave/visibility/band params -- SHARED across both
        // phases (only `surface_res`/`mass_floor` matter here, identical for
        // both), same single-write-covers-both-dispatches convention the
        // single-phase path already established.
        queue.write_buffer(
            &self.wave_params_buf,
            0,
            bytemuck::bytes_of(&WaveStepParams {
                surface_res,
                _pad: [0; 3],
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
            &self.surface_params_buf,
            &self.surface_atomic_buf,
            &self.surface_a_buf,
            &self.surface_b_buf,
            &self.raw_splat_history_buf,
            &self.surface_temp_atomic_buf,
            &self.surface_temp_float_buf,
            &self.pre_total_atomic_buf,
            &self.post_total_atomic_buf,
            &self.surface_material_mass_buf,
        );
        self.encode_phase_pipeline(
            device,
            &mut enc,
            particle_buf,
            particle_count,
            surface_res,
            &self.phase_b_params_buf,
            &self.phase_b_atomic_buf,
            &self.phase_b_a_buf,
            &self.phase_b_b_buf,
            &self.phase_b_raw_splat_history_buf,
            &self.phase_b_temp_atomic_buf,
            &self.phase_b_temp_float_buf,
            &self.phase_b_pre_total_atomic_buf,
            &self.phase_b_post_total_atomic_buf,
            &self.surface_material_mass_buf,
        );

        // Real per-phase wave/visibility/band steps -- reuses the SAME
        // wave_frame_index rotation (see `render_surface_reconstruction`'s
        // own doc for the 3-buffer reasoning) applied to phase A's existing
        // buffers AND phase B's own separate set, so both phases get the
        // identical proven flicker fixes. Must run AFTER both
        // `encode_phase_pipeline` calls above (they need the now-settled
        // `surface_a_buf`/`phase_b_a_buf`) and BEFORE the render pass below
        // (which reads their output).
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
            ],
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
        // (see the wave-step bind-group construction above's own doc).
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
        // Real mass-weighted temperature pair, grown together -- see
        // `surface_temp_atomic_buf`'s own doc.
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
        // `surface_temp_atomic_buf`'s own doc.
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
        // Real, persistent light-fluence diffusion field, grown together --
        // same real, disclosed reset-to-zero behavior as the wave field
        // just below (no light has diffused yet right after a resize,
        // which is simply true, not a bug).
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
        // Real wave field, grown together -- a resize resets it to flat
        // (zero), the same real, disclosed behavior `surface_a_buf`/
        // `surface_b_buf` already have on resize (a rare, one-time event,
        // and "undisturbed" is a physically sensible reset state).
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
        // Real temporal-disturbance history, grown together -- see
        // `wave_density_prev_buf`'s own doc.
        self.wave_density_prev_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave_density_prev"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Back to a zero placeholder -- needs reseeding before the next
        // `wave_step`, see `wave_prev_seeded`'s own doc.
        self.wave_prev_seeded = false;
        // Real hysteresis visibility state, grown together -- a resize
        // resets it to all-"not visible" (the same real, disclosed,
        // harmless one-time bias the constructor's own placeholder
        // allocation already has).
        self.visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_state"),
            size: float_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Real hysteresis color-band state, grown together -- a resize
        // resets it to band 0, same real, disclosed harmless bias the
        // constructor's own placeholder already has.
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
        // Phase B's own wave/visibility/band state, grown together -- same
        // real, disclosed reset-to-flat/not-visible/band-0 bias as phase
        // A's own fields above.
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

    /// N-material extension (see `surface_material_mass_buf`'s own doc) --
    /// grows it to `surface_res² × MAX_RENDER_MATERIAL_SLOTS × 4` bytes.
    /// Deliberately NOT called from `ensure_surface_capacity` above: only
    /// invoked when a caller actually opts in
    /// (`SurfaceReconstructionSource::material_mass_enabled`), so a
    /// `Renderer` that never opts in keeps paying only the 4-byte
    /// placeholder -- same real, disclosed lazy-growth reasoning as
    /// `GpuBuffers::grow_material_mass` on the solver side (`buffers.rs`),
    /// not the always-grow-together convention every other surface buffer
    /// above uses.
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

    /// Grows `grid_visibility_buf` when a caller's `grid_res` exceeds the
    /// currently allocated `grid_visibility_res` -- same lazy-growth
    /// pattern `ensure_surface_capacity` uses above, just keyed on the
    /// solver's own `grid_res` instead of the finer `surface_res`.
    pub(super) fn ensure_grid_visibility_capacity(&mut self, device: &wgpu::Device, grid_res: u32) {
        if grid_res <= self.grid_visibility_res {
            return;
        }
        let cell_count = (grid_res * grid_res) as u64;
        self.grid_visibility_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_visibility_state"),
            size: cell_count * mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.grid_visibility_res = grid_res;
    }
}

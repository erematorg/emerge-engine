//! The actual per-frame GPU dispatch: `step_frame` (CFL scan, uploads, encode,
//! submit, async readback) and `encode_substep` (the 7 labeled compute passes).
//!
//! Split out of `gpu/solver/mod.rs` -- the highest-risk slice: everything here
//! touches live wgpu device/buffer state and timing-sensitive submit/poll
//! ordering (see the OOM/substep-batching and active-block-grace-period
//! comments below).

use super::super::step_params::{
    GpuFieldsParams, GpuImpulseParams, GpuSleepWakeParams, GpuStepParams, NUM_BLOCKS_PER_DIM,
};
use super::encode_substep::SubstepGates;
use super::{GpuSimulation, WG_PARTICLES};

use crate::particle::Particles;
use crate::solver::config::SimConfig;
use crate::solver::{
    affine_cfl_speed_contribution, cfl_bound, deformation_gradient_ode_dt_bound, is_near_wall,
    shock_viscosity_dt_bound, single_particle_instability_dt_bound,
};

impl GpuSimulation {
    /// Advance one frame of simulation time (`config.dt`) using the GPU.
    ///
    /// All substeps are encoded into a single command buffer and submitted once -- one driver
    /// call regardless of adaptive substep count. Step params are pre-computed from the CPU
    /// particle mirror (same one-frame CFL lag as before, no physics change).
    pub fn step_frame(&mut self) {
        // A lost device cannot be un-lost; every further GPU call on it would panic
        // through wgpu's default error handler. Once lost, become a safe no-op instead.
        if self.is_device_lost() {
            return;
        }
        let total_start = std::time::Instant::now();
        let cfl_scan_start = total_start;
        let any_cpu = self.registry.any_needs_cpu_update();
        // Materials can be registered/replaced after construction; keep the per-particle
        // update pipeline specialized to the models actually present (no-op unless the
        // set of models changed -- see `SimPipelines::specialize_g2p_update`).
        self.pipelines
            .specialize_g2p_update(&self.device, self.registry.model_mask());

        // Upload CPU → GPU only when positions/materials actually changed.
        // Impulses are now applied by a dedicated GPU compute pass (apply_impulses) that
        // reads LIVE GPU positions -- no CPU mirror upload needed for impulse-only frames.
        //
        // Do not resort `self.particles` by grid cell here -- GPU `particle_sort` already
        // provides spatial locality via a separate index buffer (`sorted_particle_ids`)
        // that never touches actual particle storage order. Resorting the backing array
        // would invalidate `spawn_region`'s promised stable `Range<usize>` particle
        // identity (LP uses this as creature_id -> particle_range).
        let needs_upload = self.layout_dirty || any_cpu;
        if needs_upload {
            self.buffers.upload_particles(&self.queue, &self.particles);
            self.layout_dirty = false;
        }

        // Pre-compute all sub_dts from CPU mirror (same one-frame lag as before).
        // CFL scan is O(N) -- run it ONCE and reuse the result to fill the sub_dts array.
        // The CPU mirror is static within a frame so every repeated call would return the
        // same value anyway.
        //
        // Exclude sleeping particles from the scan. CPU's Simulation::step() does this
        // implicitly via its active/sleeping partition (active_count only covers awake
        // particles); GPU has no such partition, so without this filter a frozen-near-zero
        // sleeping majority dilutes the velocity statistics this estimate is based on,
        // potentially under-resolving the timestep right when an awake particle needs it
        // most. (sparkl's adaptive_timestep_length computes this the same way: scan only
        // the live/active particle set, never a population diluted by inactive ones.)
        //
        // `MaterialModel::timestep_bound` takes `density`/`hardening_scale` directly
        // (plain scalar fields on `Particle`), so this scan reads the AoS array in one
        // direct pass without building any SoA wrapper.
        let mut max_speed = 0.0f32;
        let mut min_mat_dt = self.config.dt;
        let mut near_wall_gravity_scale = 1.0f32;
        let mut awake_count = 0usize;
        // Real CPU/GPU parity fix (2026-09-16): CPU's own `choose_substep_dt`
        // returns immediately when `adaptive_timestep` is off (exactly `config.dt`,
        // one substep, zero scan cost) -- this GPU scan never had that early
        // return at all, so `adaptive_timestep: false` was silently a no-op here.
        // Found via a real regression: `gpu_and_cpu_shock_viscosity_match_under_
        // forced_compression` (tests/gpu.rs) sets this flag specifically to force
        // one identical substep on both backends for a controlled comparison --
        // GPU was only ever "passing" by coincidentally landing on 1 substep from
        // its OLD, incomplete CFL scan; once the real terms below were added, GPU
        // correctly started recommending 2 substeps for that scene's forced
        // compression, breaking the coincidence. Gating the whole scan here
        // restores the real, intended parity: non-adaptive means exactly one
        // substep of `config.dt`, unconditionally, matching CPU exactly.
        if self.config.adaptive_timestep {
            for p in self.particles.iter() {
                if p.sleeping != 0 {
                    continue;
                }
                awake_count += 1;
                let grad_norm = (p.velocity_gradient.x_axis.length_squared()
                    + p.velocity_gradient.y_axis.length_squared())
                .sqrt();
                let mut s = p.v.length();
                if self.config.cfl_include_affine_speed {
                    s += affine_cfl_speed_contribution(
                        &p.velocity_gradient,
                        self.config.grid_cell_size,
                    );
                }
                max_speed = max_speed.max(s);

                // Real, engine-wide CFL parity fix (2026-09-16): CPU's own
                // `choose_substep_dt` (spacetime/solver/cfl.rs) has several real, cited
                // per-particle stability terms this GPU scan never computed -- it only
                // ever saw raw particle speed plus a material's REST-state acoustic
                // bound. Confirmed live: a GPU-only fluid scene (byte-identical scene
                // geometry/EOS to a CPU twin that stays stable) explodes on violent
                // impact while `sub`/`cfl` both looked nominal -- these missing terms
                // are exactly what CPU had that GPU didn't. Ported via the SAME shared
                // functions CPU now also calls (single source of truth, no duplicated
                // formula to drift) -- see each function's own doc for citations.
                let owns_state = self.registry.owns_deformation_volume_state(p.material_id);
                // Real, PREDICTIVE (not reactive) near-wall tightening for a strict
                // fluid (`SimConfig::fluid_near_wall_cfl_scale`) -- mirrors CPU's own
                // `is_near_wall` gate exactly, including its Mach-number-based
                // compression-margin check (`fluid_near_wall_compression_mach_margin`)
                // using this same one-frame-lagged `last_max_particle_speed` CPU uses.
                let near_wall = self.config.fluid_near_wall_cfl_scale != 1.0
                    && owns_state
                    && is_near_wall(p.x, self.config.grid_res, self.config.boundary_thickness);
                let material_cfl = if near_wall && {
                    let j = p.volume / p.initial_volume;
                    let threshold = match self.registry.rest_acoustic_c2(p.material_id) {
                        Some(c2_rest) if c2_rest > f32::EPSILON => {
                            let mach = self.last_max_particle_speed / c2_rest.sqrt();
                            (mach * mach) * self.config.fluid_near_wall_compression_mach_margin
                        }
                        _ => self.config.fluid_near_wall_compression_threshold,
                    };
                    (j - 1.0).abs() > threshold
                } {
                    self.config.material_cfl_coefficient / self.config.fluid_near_wall_cfl_scale
                } else {
                    self.config.material_cfl_coefficient
                };
                if near_wall {
                    near_wall_gravity_scale =
                        near_wall_gravity_scale.max(self.config.fluid_near_wall_cfl_scale);
                }

                let mdt = self.registry.get(p.material_id).timestep_bound(
                    p.density,
                    p.hardening_scale,
                    self.config.grid_cell_size,
                    material_cfl,
                    self.config.viscous_timestep_coefficient,
                );
                if mdt.is_finite() && mdt > 0.0 {
                    min_mat_dt = min_mat_dt.min(mdt);
                }

                // Deformation-gradient ODE stability -- unconditional on material type,
                // see the function's own doc.
                let deformation_dt =
                    deformation_gradient_ode_dt_bound(grad_norm, self.config.cfl_coefficient);
                if deformation_dt.is_finite() && deformation_dt > 0.0 {
                    min_mat_dt = min_mat_dt.min(deformation_dt);
                }

                // Von Neumann-Richtmyer shock-viscosity stability correction -- tightens
                // dt on live compression rate, not just accumulated J drift.
                if let Some(shock_dt) = shock_viscosity_dt_bound(
                    grad_norm,
                    owns_state,
                    self.registry.rest_acoustic_c2(p.material_id),
                    self.registry.get(p.material_id).params().eos_power,
                    self.config.grid_cell_size,
                    material_cfl,
                ) {
                    min_mat_dt = min_mat_dt.min(shock_dt);
                }

                // Sun, Shinar & Schroeder 2020 single-particle instability bound -- the
                // exact isolated-particle feedback mechanism this scene's own runaway
                // "hot potato" outliers (1 grid neighbor, |v| into the hundreds) match.
                if owns_state {
                    let rest_density = self.registry.get(p.material_id).params().rest_density;
                    let j = p.volume / p.initial_volume;
                    if let Some(single_particle_dt) = single_particle_instability_dt_bound(
                        true,
                        rest_density,
                        j,
                        self.registry.rest_acoustic_c2(p.material_id),
                        self.config.grid_cell_size,
                    ) {
                        min_mat_dt = min_mat_dt.min(single_particle_dt);
                    }
                }
            }
            self.last_max_particle_speed = max_speed;
            // Real, standard "additional stability condition" for explicit integration
            // under a body force (Bridson, "Fluid Simulation for Computer Graphics" ch.
            // 3; Foster & Fedkiw 2001) -- see CPU's own `choose_substep_dt` tail for the
            // full derivation. Ported here (was previously CPU-only): GPU fluids sit
            // under the exact same gravity and the same at-rest gap this term closes.
            let g = self.config.gravity.length();
            if g > f32::EPSILON {
                let gravity_dt = (self.config.cfl_coefficient * self.config.grid_cell_size
                    / (g * near_wall_gravity_scale))
                    .sqrt();
                if gravity_dt.is_finite() && gravity_dt > 0.0 {
                    min_mat_dt = min_mat_dt.min(gravity_dt);
                }
            }
        } else {
            self.last_max_particle_speed = 0.0;
        }
        // If every particle is asleep AND something could actually disturb them this
        // frame, there's no awake velocity to base an estimate on -- choose_substep_dt
        // would fall back to max_dt (max_speed=0 fails its `> f32::EPSILON` guard), the
        // COARSEST possible substep, right when a wake event needs the FINEST. But wake
        // propagation only happens via a neighbor's grid activity (which requires some
        // OTHER awake particle to exist -- if the awake set is truly empty, there is none)
        // or an external impulse. So "everyone asleep" alone isn't a risk: nothing CAN
        // wake spontaneously with no awake particles and no incoming disturbance. Only
        // pay for the fine fallback when a pending impulse could actually wake someone --
        // otherwise a fully-settled scene would pay maximum substep cost forever, which
        // defeats sleep/wake's entire purpose.
        let might_wake_this_frame = !self.pending_impulses.is_empty();
        let sub_dt_cfl = if !self.config.adaptive_timestep {
            self.config.dt
        } else if awake_count == 0 && self.config.sleep_threshold > 0.0 && might_wake_this_frame {
            self.config.dt / self.config.max_substeps_per_step.max(1) as f32
        } else {
            cfl_bound(&self.config, max_speed, min_mat_dt, self.config.dt)
        };
        let mut sub_dts: Vec<f32> = Vec::with_capacity(self.config.max_substeps_per_step);
        {
            let mut remaining = self.config.dt;
            while remaining > f32::EPSILON && sub_dts.len() < self.config.max_substeps_per_step {
                let sub_dt = sub_dt_cfl.min(remaining);
                sub_dts.push(sub_dt);
                remaining -= sub_dt;
            }
        }
        // The GPU re-picks each substep's dt from the post-update particle state and can
        // only go BELOW `sub_dt_cfl` (see `adaptive_cfl.wgsl`), so the frame may need
        // more substeps than this estimate. Encode a margin: substeps beyond the frame's
        // time return immediately, at the cost of their (tiny) dispatch. A frame violent
        // enough to exhaust even the margin advances less than `config.dt` -- the same
        // honest dropped time the CPU loop reports when it hits `max_substeps_per_step`.
        const ADAPTIVE_SUBSTEP_MARGIN: f32 = 1.15;
        let encoded_substeps = (((sub_dts.len() as f32) * ADAPTIVE_SUBSTEP_MARGIN).ceil() as usize)
            .clamp(sub_dts.len(), self.config.max_substeps_per_step)
            .max(1);
        self.buffers.upload_adaptive_dt(
            &self.queue,
            sub_dts[0].min(self.config.dt),
            self.config.dt,
        );
        self.last_sub_dt = sub_dts.last().copied().unwrap_or(self.config.dt);
        self.frame_index += 1;
        let cfl_scan_ns = cfl_scan_start.elapsed().as_secs_f32() * 1.0e9;

        // Sleep delay: a particle spawned at rest (v=0) satisfies any positive
        // sleep_threshold on its very first substep, before gravity has accelerated it
        // at all -- same fix every real physics engine uses for this (Box2D, PhysX,
        // Bullet all require sustained low velocity before sleeping, never an instant
        // single-frame check). Can't add a per-particle timer here (Particle has no
        // spare bytes left), so this is the simulation-level equivalent: don't let
        // anything sleep-score for the first few frames after the most recent spawn,
        // giving real dynamics a chance to start.
        //
        // Window re-arms from `last_spawn_frame` (updated by `spawn_region`), not just
        // frame 0 -- otherwise a particle spawned live mid-scene (e.g. a paint tool) would
        // get `sleep_threshold` applied at v=0 on its very first substep and freeze
        // asleep before gravity ever touched it.
        const SLEEP_WARMUP_FRAMES: u64 = 10;
        let step_config = if self.frame_index <= self.last_spawn_frame + SLEEP_WARMUP_FRAMES {
            SimConfig {
                sleep_threshold: 0.0,
                ..self.config
            }
        } else {
            self.config
        };

        // Build force fields uniform (same every substep).
        let mut ff_params: GpuFieldsParams = bytemuck::Zeroable::zeroed();
        ff_params.count = self.force_field_entries.len() as u32;
        for (i, e) in self.force_field_entries.iter().enumerate() {
            ff_params.entries[i] = *e;
        }
        self.buffers
            .upload_force_fields_params(&self.queue, &ff_params);

        // Multi-field contact (GPU port) -- directional grip friction, uploaded once per
        // frame like ff_params above. `self.grip_params` starts symmetric (no
        // directional bias, identical to every scene before this existed) and is only
        // live-adjustable via `set_grip_direction`/`set_grip_friction` -- a real
        // GPU-side `DirectionalContactGrip` equivalent, matching CPU's own
        // atomics-based live-adjustable pattern (plain field here since GpuSimulation
        // isn't Arc-shared across threads the way CPU's boundary conditions are).
        self.buffers
            .upload_grip_params(&self.queue, &self.grip_params);

        // Day-night/ambient thermal diffusion (GPU port) -- uploaded once per frame,
        // same pattern as grip_params above. `enabled == 0` (the default, every
        // existing scene) makes the 4 thermal passes below skip their dispatch
        // entirely, not just early-return per-thread -- real, not just disabled-in-name.
        self.buffers
            .upload_thermal_params(&self.queue, &self.thermal_params);
        let thermal_active = self.thermal_params.enabled != 0;

        // Resource regrowth (GPU port) -- same upload + real dispatch-skip pattern as
        // thermal above.
        self.buffers
            .upload_resource_params(&self.queue, &self.resource_params);
        let resource_active = self.resource_params.enabled != 0;

        // ASFLIP (GPU port) -- same upload + real dispatch-skip pattern as thermal/
        // resource above, but the "skip" here means the fused g2p_asflip_fused pass
        // REPLACES g2p+particles_update rather than an extra pass being skipped
        // entirely -- see SubstepGates::asflip_active's use in encode_substep.rs.
        self.buffers
            .upload_asflip_params(&self.queue, &self.asflip_params);
        let asflip_active = self.asflip_params.enabled != 0;

        // `ColorMode::GridVolume` material-mass tracking -- same upload pattern, real
        // per-substep cost (an extra P2G atomic scatter + grid_clear zeroing) only
        // when `attach_grid_material_render_gpu` has been called.
        self.buffers
            .upload_material_mass_params(&self.queue, &self.material_mass_params);

        // Force-sleep/force-wake-by-tag -- minimal hook for LP's future chunk system.
        // Uploaded every frame (zeroed when nothing's pending, same as ff_params above)
        // and read once per substep in force_fields.wgsl; cleared after upload since
        // each call is a one-shot edge-trigger, not a persistent state (a tag that's
        // force-asleep doesn't need to be re-sent every frame -- sleeping is sticky on
        // the particle itself until something genuinely wakes it).
        let mut sw_params: GpuSleepWakeParams = bytemuck::Zeroable::zeroed();
        sw_params.sleep_count = self.pending_sleep_tags.len() as u32;
        for (i, &tag) in self.pending_sleep_tags.iter().enumerate() {
            sw_params.sleep_tags[i / 4][i % 4] = tag;
        }
        sw_params.wake_count = self.pending_wake_tags.len() as u32;
        for (i, &tag) in self.pending_wake_tags.iter().enumerate() {
            sw_params.wake_tags[i / 4][i % 4] = tag;
        }
        self.buffers
            .upload_sleep_wake_params(&self.queue, &sw_params);
        self.pending_sleep_tags.clear();
        self.pending_wake_tags.clear();

        // force_fields_main is a provable no-op for every particle this frame when none
        // of these are true -- no fields configured, no tag-based sleep/wake pending,
        // and sleep-scoring disabled (the pass's only other job). Even with an empty loop
        // body it still reads+writes every particle's full 128-byte struct, so skipping
        // the whole dispatch (not just the loop) when unneeded avoids that memory traffic
        // -- same principle as the lazy spatial hash and sparse-grid active-block dispatch.
        let force_fields_needed = ff_params.count > 0
            || sw_params.sleep_count > 0
            || sw_params.wake_count > 0
            || step_config.sleep_threshold > 0.0;

        // Mirrors CPU's `Grid::has_contact_activity()` gate (`transfer.rs`):
        // `resolve_contact`/`gather_contact_points` are structurally required whenever ANY
        // particle uses multi-field contact (g2p.wgsl unconditionally reads their output),
        // but for the common case where NO particle ever sets `contact_group`, this is
        // provable dead work. A plain O(N) scan of the CPU particle mirror (same
        // "compute once per frame" pattern as `force_fields_needed` above) is far cheaper
        // than the GPU passes it gates.
        let contact_active = self.particles[..self.particle_count]
            .iter()
            .any(|p| p.contact_group != 0);

        // Upload step_params for each substep into its pool slot -- contents change every
        // frame (adaptive dt), so this write can't be cached. The bind group pointing at
        // that slot, however, only depends on buffer IDENTITY, not contents, so it's built
        // once in `bind_group_pool` (see that field's doc comment) instead of recreated
        // here every substep every frame -- doing so at LP's ~5-6k-substep-per-frame scale
        // exhausted the GPU's descriptor allocator within seconds.
        {
            // Every encoded substep gets the same params: `dt`/`vel_limit` are no longer
            // read by the shaders (they take those from `adaptive_dt`), but `dt_cap` is --
            // it is the CPU's frame-start CFL choice, the ceiling the GPU may not exceed.
            let params = GpuStepParams::new(
                &step_config,
                sub_dt_cfl,
                self.particle_count,
                contact_active,
            );
            for i in 0..encoded_substeps {
                self.buffers.upload_step_params_at(&self.queue, i, &params);
            }
        }
        let bind_groups = &self.bind_group_pool;

        // Encode everything into one command buffer -- one GPU submit per frame.
        // Order: [apply_impulses?] → [particle_sort?] → substep_0 → … → substep_N
        //
        // apply_impulses runs first so physics sees the freshly-applied velocities.
        // particle_sort re-seeds sorted_particle_ids after a CPU upload (layout_dirty).
        // Both use dedicated buffer slots so they never alias substep params.
        let particle_wg = (self.particle_count as u32).div_ceil(WG_PARTICLES);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm_frame"),
            });

        // -- apply_impulses pass (GPU-native, no stale CPU mirror) --
        if !self.pending_impulses.is_empty() {
            let vel_limit = self.config.grid_cell_size / self.config.min_dt;
            let mut params = GpuImpulseParams {
                count: self.pending_impulses.len() as u32,
                vel_limit,
                particle_count: self.particle_count as u32,
                _pad: 0,
                entries: bytemuck::Zeroable::zeroed(),
            };
            for (i, e) in self.pending_impulses.iter().enumerate() {
                params.entries[i] = *e;
            }
            self.buffers.upload_impulse_params(&self.queue, &params);
            let impulse_bg = self
                .pipelines
                .make_impulse_bind_group(&self.device, &self.buffers);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply_impulses"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.apply_impulses);
            pass.set_bind_group(0, &impulse_bg, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            drop(pass);
            self.pending_impulses.clear();
        }

        // -- particle_sort pass: clear -> count -> scan -> scatter, every frame --
        //
        // Runs unconditionally (not gated on layout_dirty) because particle positions drift
        // every substep even when the CPU mirror is never touched -- without a per-frame
        // re-sort, sorted_particle_ids would stay frozen at whatever ordering existed at the
        // last CPU upload, going stale as GPU-resident particles move. See particle_sort.wgsl.
        {
            let sort_slot = self.buffers.step_params_pool.len() - 1;
            let sort_params = GpuStepParams::new(
                &self.config,
                self.config.dt,
                self.particle_count,
                contact_active,
            );
            self.buffers
                .upload_step_params_at(&self.queue, sort_slot, &sort_params);
            // Reuse the cached bind group from `bind_group_pool` rather than calling
            // `pipelines.make_bind_group(...)` fresh here -- creating one every
            // `step_frame` call exhausts the GPU's descriptor allocator over a long run.
            // `bind_group_pool` already contains one bind group per `step_params_pool`
            // slot, including this sort slot (see `build_bind_group_pool`), rebuilt only
            // when `buffers` reallocates. The bind group only depends on buffer IDENTITY,
            // not contents (which `upload_step_params_at` above rewrites in place), so
            // reusing the cached entry is correct, not just faster.
            let sort_bg = &self.bind_group_pool[sort_slot];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_sort"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, sort_bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
            pass.set_pipeline(&self.pipelines.particle_sort_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_count);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            // No particle_sort_compact here anymore -- active-block detection now runs
            // every substep (see encode_substep's active_block_refresh pass), since
            // particles move every substep and this once-per-frame pass would go stale by
            // substep 2+. This pass's count output is used only for the sort permutation
            // (scan + scatter below), unrelated to active-block correctness.
            pass.set_pipeline(&self.pipelines.particle_sort_scan);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_scatter);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Substeps are submitted in small batches as soon as they are encoded, so the
        // GPU starts on the first ones while the CPU is still encoding the rest.
        // Measured (dam break, 61 substeps, one batch per frame): the GPU sat idle for
        // the whole ~23ms encode, and the first substeps after that idle gap ran far
        // slower than the last ones (last 1-4 substeps at their profiled ~0.7ms each,
        // the frame as a whole at ~1.2ms/substep) -- an integrated GPU drops its clock
        // while idle and ramps back up only under sustained load.
        const SUBSTEP_SUBMIT_BATCH: usize = 8;
        // Blocking every `SUBSTEP_BLOCK_EVERY` substeps is still required: encoding a
        // few hundred substeps without letting the GPU drain exhausts this backend's
        // descriptor allocator (200 in one submit reliably OOMs, 64 is stable -- a
        // per-backend/driver ceiling, not from any GPU spec). Typical scenes (under 64
        // substeps/frame) never block.
        const SUBSTEP_BLOCK_EVERY: usize = 64;
        let mut chunks = bind_groups[..encoded_substeps]
            .chunks(SUBSTEP_SUBMIT_BATCH)
            .peekable();
        // Split pure CPU command-building time from GPU-completion wait time --
        // "encode_ns" previously bundled both under one name, hiding whether a slow
        // step_frame() was a CPU-side encoding problem or genuinely GPU-execution-bound.
        // How often the per-substep active-block re-detection has to run. A block is
        // marked active when it OR ANY of its 8 neighbours holds particles
        // (`particle_sort_compact_main`), so a particle's 3x3 scatter stencil stays inside
        // the marked region until it has travelled about a block minus the stencil reach.
        // Per substep a particle moves at most `max_speed * sub_dt` (and never more than
        // one cell, since `g2p` clamps to `vel_limit = grid_cell_size / sub_dt`), so the
        // list can safely be reused for that many substeps. The acoustic CFL makes this a
        // large margin for a stiff fluid -- 0.03 cells per substep on the dam break, where
        // re-detecting every substep cost ~20us of a ~170us substep -- but a slow, coarse
        // scene gets 1 and behaves exactly as before.
        let block_size_cells = self.config.grid_res.div_ceil(NUM_BLOCKS_PER_DIM) as f32;
        let margin_cells = (block_size_cells - 1.5).max(0.5);
        // 4x headroom on the frame-start max speed for anything that accelerates mid-frame
        // (impacts); the per-substep displacement is capped at one cell regardless.
        const SPEED_HEADROOM: f32 = 4.0;
        let per_substep_travel = (max_speed * SPEED_HEADROOM * sub_dt_cfl)
            .clamp(f32::MIN_POSITIVE, self.config.grid_cell_size);
        let active_block_refresh_interval =
            ((margin_cells / per_substep_travel) as usize).clamp(1, SUBSTEP_SUBMIT_BATCH);

        let mut pure_encode_ns = 0.0f32;
        let mut wait_ns = 0.0f32;
        let mut first_chunk = true;
        let mut since_block = 0usize;
        let mut substep_counter = 0usize;
        while let Some(chunk) = chunks.next() {
            let chunk_encode_start = std::time::Instant::now();
            let mut sub_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("mpm_substep_batch"),
                    });
            if first_chunk {
                self.profile_frame_marker(&mut sub_encoder, false);
                first_chunk = false;
            }
            {
                let mut pass = sub_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mpm_substeps"),
                    timestamp_writes: None,
                });
                // Groups 1-3 never change between substeps; group 0 (the substep's
                // `StepParams` slot) is set by `encode_substep`.
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                for bg in chunk {
                    let refresh_active_blocks =
                        substep_counter.is_multiple_of(active_block_refresh_interval);
                    substep_counter += 1;
                    // Profile a substep from the middle of the frame: the last encoded
                    // ones are the adaptive-timestep margin and usually do nothing.
                    self.profile_this_substep
                        .set(substep_counter == encoded_substeps / 2);
                    self.encode_substep(
                        &mut pass,
                        bg,
                        particle_wg,
                        SubstepGates {
                            force_fields_needed,
                            contact_active,
                            thermal_active,
                            resource_active,
                            asflip_active,
                            // Reuses the SAME `SimConfig` field CPU's own
                            // `step.rs` already gates on -- real, automatic
                            // parity, not a separate GPU-only flag. Default 0
                            // (every existing scene, CPU or GPU) is a true no-op
                            // (see `fluid_pressure_iterations > 0` gate in
                            // `encode_substep.rs`).
                            fluid_pressure_iterations: self.config.fluid_pressure_iterations,
                            refresh_active_blocks,
                        },
                    );
                }
            }
            let last_chunk = chunks.peek().is_none();
            if last_chunk {
                self.profile_frame_marker(&mut sub_encoder, true);
            }
            self.queue.submit(std::iter::once(sub_encoder.finish()));
            pure_encode_ns += chunk_encode_start.elapsed().as_secs_f32() * 1.0e9;
            since_block += chunk.len();
            if !last_chunk && since_block >= SUBSTEP_BLOCK_EVERY {
                let wait_start = std::time::Instant::now();
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                wait_ns += wait_start.elapsed().as_secs_f32() * 1.0e9;
                since_block = 0;
            }
        }
        let encode_ns = pure_encode_ns;
        // Repurposed: real GPU-completion wait time between chunks, not always 0 --
        // this IS where GPU execution time shows up for multi-chunk (>64 substep) frames.
        let submit_ns = wait_ns;

        // Async GPU → CPU readback -- never blocks the render thread.
        //
        // Two-phase: begin_readback submits a GPU copy + async map (non-blocking).
        // The receiver fires on a subsequent frame when the GPU copy + map completes.
        // We pump wgpu callbacks with poll(Poll) each frame so the mapping progresses.
        //
        // If any_cpu: readback every frame (CPU plasticity needs current state).
        // Otherwise: stride-gated to reduce overhead.
        let readback_start = std::time::Instant::now();
        self.readback_frame = self.readback_frame.wrapping_add(1);
        let want_readback = any_cpu || self.readback_frame.is_multiple_of(self.readback_stride);

        // Pump wgpu callbacks so any in-flight mapping can complete.
        self.device.poll(wgpu::PollType::Poll).ok();

        // The real substep count and dropped time come from the GPU's own time
        // accounting, read back without blocking: one frame late unless the caller
        // waits with `sync_frame_stats`. If the previous readback is still in flight
        // the GPU is more than a frame behind and this frame's stats are skipped.
        self.collect_frame_stats();
        if self.pending_frame_stats.is_none() && !self.is_device_lost() {
            self.pending_frame_stats = Some(
                self.buffers
                    .begin_frame_stats_readback(&self.device, &self.queue),
            );
        }

        // Check if a previous async readback completed -- Ok, Err, or still pending.
        // Every completion path must explicitly unmap regardless of Ok/Err -- an
        // unhandled Err leaves the staging buffer mapped forever (finish_readback, the
        // only unmapper, never called) and pending_readback stuck Some forever, until
        // something else tries to map the same buffer and panics.
        let readback_done = self
            .pending_readback
            .as_ref()
            .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
        if let Some(result) = readback_done {
            self.pending_readback = None;
            // The device-lost check at the TOP of step_frame only guards against a
            // device that was ALREADY lost before this call started -- it says nothing
            // about a device that dies DURING this same call (e.g. an earlier
            // queue.submit() in this frame's chunked substep loop triggers an
            // uncaptured OOM). Re-check here: once lost, the staging buffer may already
            // be destroyed regardless of what the async result claims, so both the Ok
            // and Err branches are skipped, not just one.
            if self.is_device_lost() {
                // Do nothing -- neither finish_readback nor abandon_readback is
                // safe to call once the device is confirmed lost.
            } else if result.is_err() {
                self.readback_error_count += 1;
                self.buffers.abandon_readback();
            } else {
                let gpu_particles = self.buffers.finish_readback(self.particle_count);

                // CPU plasticity pass -- skipped if all materials run plasticity on GPU.
                //
                // IMPORTANT: GPU g2p already integrated F via `F_new = (I + dt·C)·F_old`.
                // Zero affine before update_particle so only the plasticity projection runs.
                // Restore GPU affine afterwards so next P2G APIC term is correct.
                // Convert AoS to SoA, run the CPU pass via a per-particle
                // `ParticleUpdateCtx`, then scatter results back.
                if any_cpu {
                    // Stash GPU affine matrices -- we zero affine for the plasticity call then restore.
                    let gpu_affines: Vec<_> =
                        gpu_particles.iter().map(|p| p.velocity_gradient).collect();
                    // Copy readback into AoS cpu mirror (zeroing affine for plasticity).
                    for (p_gpu, p_cpu) in gpu_particles.iter().zip(self.particles.iter_mut()) {
                        *p_cpu = *p_gpu;
                        p_cpu.velocity_gradient = glam::Mat2::ZERO;
                    }
                    // Build SoA wrapper, run CPU plasticity, scatter plastic state back.
                    // Skip sleeping particles -- same reasoning as every GPU-side pass: their
                    // F/plastic state is frozen, re-running plasticity on unchanged input
                    // wastes exactly the compute sleep/wake exists to avoid.
                    let mut soa = Particles::from(std::mem::take(&mut self.particles));
                    for i in 0..soa.len() {
                        if soa.sleeping[i] {
                            continue;
                        }
                        let material_id = soa.material_id[i];
                        self.registry
                            .get(material_id)
                            .update_particle(&mut soa.update_ctx(i), self.last_sub_dt);
                    }
                    self.particles = soa.to_vec();
                    // Restore GPU affine.
                    for (p_cpu, gpu_affine) in self.particles.iter_mut().zip(gpu_affines) {
                        p_cpu.velocity_gradient = gpu_affine;
                    }
                } else {
                    for (p_gpu, p_cpu) in gpu_particles.into_iter().zip(self.particles.iter_mut()) {
                        *p_cpu = p_gpu;
                    }
                }
                if any_cpu {
                    self.layout_dirty = true; // CPU plasticity touched positions/F
                }
                // Defer the actual O(N) rebuild to the first query that needs it
                // (ensure_spatial_hash_fresh, queries.rs) instead of paying it on every
                // readback completion regardless of whether a query runs this frame --
                // see spatial_hash's doc in mod.rs.
                self.spatial_hash_dirty.set(true);
            }
        }

        // Start a new readback if wanted and none is already in flight -- guarded by
        // is_device_lost() for the same reason as the completion-check block above:
        // a mid-call device loss shouldn't kick off a fresh async copy/map against a
        // buffer that may already be gone.
        if want_readback && self.pending_readback.is_none() && !self.is_device_lost() {
            self.pending_readback = Some(self.buffers.begin_readback(
                &self.device,
                &self.queue,
                self.particle_count,
            ));
        }
        let readback_ns = readback_start.elapsed().as_secs_f32() * 1.0e9;
        let total_ns = total_start.elapsed().as_secs_f32() * 1.0e9;
        self.last_cpu_timings = (cfl_scan_ns, encode_ns, submit_ns, readback_ns, total_ns);
    }

    /// Wait for the GPU to finish the last stepped frame and read its real substep
    /// count and dropped time into `last_substeps` / `last_sim_time_dropped`, which are
    /// otherwise one frame behind (`step_frame` never blocks on them).
    pub fn sync_frame_stats(&mut self) {
        if self.pending_frame_stats.is_some() && !self.is_device_lost() {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        self.collect_frame_stats();
    }

    /// Apply a completed frame stats readback, if there is one. Never blocks.
    fn collect_frame_stats(&mut self) {
        let done = self
            .pending_frame_stats
            .as_ref()
            .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
        let Some(result) = done else {
            return;
        };
        self.pending_frame_stats = None;
        if self.is_device_lost() {
            return;
        }
        if result.is_err() {
            self.readback_error_count += 1;
            self.buffers.abandon_frame_stats_readback();
            return;
        }
        let stats = self.buffers.finish_frame_stats_readback();
        self.last_substeps = stats.executed_substeps as usize;
        self.last_sim_time_dropped = stats.dropped_time;
    }
}

// Live-adjustable params (add/clear_force_field_gpu, set_grip_direction/
// friction, attach_thermal_gpu, set_thermal_ambient, attach_resource_field_gpu)
// split into sibling module live_params.rs (declared in solver/mod.rs);
// encode_substep (the 7-8 per-substep compute passes) split into sibling
// module encode_substep.rs -- was ~440 combined of this file's ~1000 lines.
// step_frame (above) is the one thing that stays here, per this file's own
// top-of-file doc comment on why it's the highest-risk slice.

//! The actual per-frame GPU dispatch: `step_frame` (CFL scan, uploads, encode,
//! submit, async readback) and `encode_substep` (the 7 labeled compute passes).
//!
//! Split out of `gpu/solver/mod.rs` -- the highest-risk slice, deliberately
//! done last and alone: everything here touches live wgpu device/buffer
//! state, timing-sensitive submit/poll ordering, and has real, previously-
//! debugged failure modes documented inline (see the OOM/substep-batching and
//! active-block-grace-period comments below). Pure mechanical move via exact
//! line-range extraction, not retyped, to eliminate transcription risk in
//! code this precise.

use super::super::step_params::{
    GpuFieldEntry, GpuFieldsParams, GpuImpulseParams, GpuSleepWakeParams, GpuStepParams,
    GpuThermalParams, MAX_FORCE_FIELDS, NUM_BLOCKS,
};
use super::{GpuSimulation, PROFILE_PASS_LABELS, WG_PARTICLES};
use crate::particle::Particles;
use crate::solver::config::SimConfig;
use crate::solver::{affine_cfl_speed_contribution, cfl_bound};

impl GpuSimulation {
    /// Advance one frame of simulation time (`config.dt`) using the GPU.
    ///
    /// All substeps are encoded into a single command buffer and submitted once — one driver
    /// call regardless of adaptive substep count. Step params are pre-computed from the CPU
    /// particle mirror (same one-frame CFL lag as before, no physics change).
    pub fn step_frame(&mut self) {
        // Real fix for emerge issue #10 (confirmed root cause: genuine device
        // loss, Out of Memory, under sustained slow-backend load — see project
        // memory gpu_readback_error_path_bug_issue10). A lost device cannot be
        // un-lost; every further GPU call on it would panic through wgpu's
        // default error handler. Once lost, become a safe no-op instead.
        if self.is_device_lost() {
            return;
        }
        let total_start = std::time::Instant::now();
        let cfl_scan_start = total_start;
        let any_cpu = self.registry.any_needs_cpu_update();

        // Upload CPU → GPU only when positions/materials actually changed.
        // Impulses are now applied by a dedicated GPU compute pass (apply_impulses) that
        // reads LIVE GPU positions — no CPU mirror upload needed for impulse-only frames.
        //
        // Real bug fix (2026-07-06, LP issue erematorg/LP#161): this block used to
        // spatially resort `self.particles` by grid cell before every upload. That
        // predates the real GPU particle_sort pipeline (`f2c1e62`, "real particle-sort
        // pipeline") which added its own spatial-locality mechanism entirely on the GPU
        // side (`sorted_particle_ids`, a SEPARATE index buffer that never touches actual
        // particle storage order — see particle_sort.wgsl, runs unconditionally every
        // frame). When that GPU pass was added, the old CPU-side resort should have been
        // removed but wasn't — it kept running on every upload, which happens on
        // essentially every frame in real use (any per-particle CPU write, e.g. LP's
        // `drive_muscles`/`update_damage`, calls `mark_particles_dirty`). Reordering the
        // backing array on every such frame silently invalidated any previously-returned
        // `Range<usize>` particle identity (`spawn_region`'s own doc promises this range
        // is stable — "LP uses this as creature_id -> particle_range"). Confirmed via a
        // real repro: a spawned creature's fixed index range, read back every frame,
        // showed near-total corruption of a spawn-time-only tag field (`muscle_group_id`)
        // — not a readback race, the particles at those indices were simply different
        // particles after the resort. No remaining purpose for this CPU-side sort once
        // the GPU has its own; removing it restores range stability.
        let needs_upload = self.layout_dirty || any_cpu;
        if needs_upload {
            self.buffers.upload_particles(&self.queue, &self.particles);
            self.layout_dirty = false;
        }

        // Pre-compute all sub_dts from CPU mirror (same one-frame lag as before).
        // CFL scan is O(N) — run it ONCE and reuse the result to fill the sub_dts array.
        // The CPU mirror is static within a frame so every repeated call would return the
        // same value anyway. Previously this called choose_substep_dt up to 16×/frame
        // (once per substep), which in debug mode caused measurable cursor slowdown.
        //
        // Exclude sleeping particles from the scan. CPU's Simulation::step() does this
        // implicitly via its active/sleeping partition (active_count only covers awake
        // particles); GPU has no such partition, so without this filter a frozen-near-zero
        // sleeping majority dilutes the velocity statistics this estimate is based on,
        // potentially under-resolving the timestep right when an awake particle needs it
        // most. (sparkl's adaptive_timestep_length, tmp/sparkl/src/dynamics/solver/
        // timestep_estimator.rs, computes this the same way: scan only the live/active
        // particle set, never a population diluted by inactive ones.)
        // REAL FIX (2026-06-27, see project_mvp_definition memory for the full
        // investigation): the previous version built a fresh `Particles` SoA every frame
        // (filter+collect into an intermediate AoS Vec, then transpose into SoA) purely
        // because `MaterialModel::timestep_bound` used to require `&Particles, i: usize`.
        // Every material's implementation only ever read `density`/`hardening_scale` —
        // both plain scalar fields that already exist directly on `Particle` (AoS). Changed
        // the trait to take those two scalars directly (12 materials updated, 1 call site
        // in `choose_substep_dt`), which means this scan never needs to build ANY SoA
        // wrapper at all — it just reads each particle's own fields in one direct pass over
        // the array that already exists: zero allocation, not just less allocation.
        // Correctness fully verified (full CPU+GPU regression suite green). Wall-clock
        // comparisons on this machine were unreliable that night (integrated GPU, shared
        // CPU/GPU thermal budget, hours of sustained heavy load) — don't trust a GPU timing
        // number gathered after a long run of GPU work on this hardware; re-measure
        // `gpu_cfl_scan_baseline_across_grid` cold, first thing in a session, for a real
        // comparison.
        let mut max_speed = 0.0f32;
        let mut min_mat_dt = self.config.dt;
        let mut awake_count = 0usize;
        for p in self.particles.iter() {
            if p.sleeping != 0 {
                continue;
            }
            awake_count += 1;
            let mut s = p.v.length();
            if self.config.cfl_include_affine_speed {
                s +=
                    affine_cfl_speed_contribution(&p.velocity_gradient, self.config.grid_cell_size);
            }
            max_speed = max_speed.max(s);
            let mdt = self.registry.get(p.material_id).timestep_bound(
                p.density,
                p.hardening_scale,
                self.config.grid_cell_size,
                self.config.material_cfl_coefficient,
                self.config.viscous_timestep_coefficient,
            );
            if mdt.is_finite() && mdt > 0.0 {
                min_mat_dt = min_mat_dt.min(mdt);
            }
        }
        // If every particle is asleep AND something could actually disturb them this
        // frame, there's no awake velocity to base an estimate on — choose_substep_dt
        // would fall back to max_dt (max_speed=0 fails its `> f32::EPSILON` guard), the
        // COARSEST possible substep, right when a wake event needs the FINEST. But wake
        // propagation only happens via a neighbor's grid activity (which requires some
        // OTHER awake particle to exist — if the awake set is truly empty, there is none)
        // or an external impulse. So "everyone asleep" alone isn't a risk: nothing CAN
        // wake spontaneously with no awake particles and no incoming disturbance. Only
        // pay for the fine fallback when a pending impulse could actually wake someone —
        // otherwise a fully-settled scene would pay maximum substep cost forever, which
        // defeats sleep/wake's entire purpose (measured: 64 substeps/frame indefinitely
        // on a calm, fully-asleep pile before this check was added).
        let might_wake_this_frame = !self.pending_impulses.is_empty();
        let sub_dt_cfl =
            if awake_count == 0 && self.config.sleep_threshold > 0.0 && might_wake_this_frame {
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
        self.last_substeps = sub_dts.len();
        self.last_sub_dt = sub_dts.last().copied().unwrap_or(self.config.dt);
        self.frame_index += 1;
        let cfl_scan_ns = cfl_scan_start.elapsed().as_secs_f32() * 1.0e9;

        // Sleep delay: a particle spawned at rest (v=0) satisfies any positive
        // sleep_threshold on its very first substep, before gravity has accelerated it
        // at all — same fix every real physics engine uses for this (Box2D, PhysX,
        // Bullet all require sustained low velocity before sleeping, never an instant
        // single-frame check). Can't add a per-particle timer here (Particle has no
        // spare bytes left), so this is the simulation-level equivalent: don't let
        // anything sleep-score for the first few frames after construction, giving
        // real dynamics a chance to start. Once any particle exists, GPU has no
        // incremental add API (everything is introduced at construction), so this
        // covers every particle that will ever exist in this simulation, not just the
        // initial batch.
        const SLEEP_WARMUP_FRAMES: u64 = 10;
        let step_config = if self.frame_index <= SLEEP_WARMUP_FRAMES {
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

        // Multi-field contact (GPU port) — directional grip friction, uploaded once per
        // frame like ff_params above. `self.grip_params` starts symmetric (no
        // directional bias, identical to every scene before this existed) and is only
        // live-adjustable via `set_grip_direction`/`set_grip_friction` — a real
        // GPU-side `DirectionalContactGrip` equivalent, matching CPU's own
        // atomics-based live-adjustable pattern (plain field here since GpuSimulation
        // isn't Arc-shared across threads the way CPU's boundary conditions are).
        self.buffers
            .upload_grip_params(&self.queue, &self.grip_params);

        // Day-night/ambient thermal diffusion (GPU port) — uploaded once per frame,
        // same pattern as grip_params above. `enabled == 0` (the default, every
        // existing scene) makes the 4 thermal passes below skip their dispatch
        // entirely, not just early-return per-thread — real, not just disabled-in-name.
        self.buffers
            .upload_thermal_params(&self.queue, &self.thermal_params);
        let thermal_active = self.thermal_params.enabled != 0;

        // Force-sleep/force-wake-by-tag — minimal hook for LP's future chunk system.
        // Uploaded every frame (zeroed when nothing's pending, same as ff_params above)
        // and read once per substep in force_fields.wgsl; cleared after upload since
        // each call is a one-shot edge-trigger, not a persistent state (a tag that's
        // force-asleep doesn't need to be re-sent every frame — sleeping is sticky on
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
        // and sleep-scoring disabled (the pass's only other job). Real cost found via
        // profiling (2026-07-12): the pass still reads+writes every particle's full
        // 128-byte struct even with an empty loop body, ~1ms/17.5% of a substep at 50k
        // particles for pure memory traffic with nothing to show for it. Skipping the
        // whole dispatch (not just the loop) when genuinely unneeded is the same
        // "don't pay for provably unnecessary work" principle already applied to the
        // lazy spatial hash and the sparse-grid active-block dispatch.
        let force_fields_needed = ff_params.count > 0
            || sw_params.sleep_count > 0
            || sw_params.wake_count > 0
            || step_config.sleep_threshold > 0.0;

        // Real GPU gap fixed 2026-07-15, mirroring CPU's `Grid::has_contact_activity()`
        // gate (`transfer.rs`): `resolve_contact`/`gather_contact_points` are structurally
        // required whenever ANY particle uses multi-field contact (g2p.wgsl unconditionally
        // reads their output for every scene), but for the common case where NO particle
        // ever sets `contact_group`, this is entirely provable dead work -- measured at
        // 37.5%/5.66ms of a substep on a pure fluid scene with zero contact particles,
        // paid by every existing example except the two snake_on_terrain ones. A plain O(N)
        // scan of the CPU particle mirror (same "compute once per frame" pattern as
        // `force_fields_needed` above) is far cheaper than the GPU passes it gates.
        let contact_active = self.particles[..self.particle_count]
            .iter()
            .any(|p| p.contact_group != 0);

        // Upload step_params for each substep into its pool slot -- contents change every
        // frame (adaptive dt), so this write can't be cached. The bind group pointing at
        // that slot, however, only depends on buffer IDENTITY, not contents, so it's built
        // once in `bind_group_pool` (see that field's doc comment) instead of recreated
        // here every substep every frame -- doing so at LP's ~5-6k-substep-per-frame scale
        // exhausted the GPU's descriptor allocator within seconds.
        for (i, &sub_dt) in sub_dts.iter().enumerate() {
            let params =
                GpuStepParams::new(&step_config, sub_dt, self.particle_count, contact_active);
            self.buffers.upload_step_params_at(&self.queue, i, &params);
        }
        let bind_groups = &self.bind_group_pool;

        // Encode everything into one command buffer — one GPU submit per frame.
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

        // — apply_impulses pass (GPU-native, no stale CPU mirror) —
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

        // — particle_sort pass: clear -> count -> scan -> scatter, every frame —
        //
        // Runs unconditionally (not gated on layout_dirty) because particle positions drift
        // every substep even when the CPU mirror is never touched — without a per-frame
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
            let sort_bg = self.pipelines.make_bind_group(
                &self.device,
                &self.buffers,
                &self.buffers.step_params_pool[sort_slot],
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_sort"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &sort_bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_pipeline(&self.pipelines.particle_sort_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_count);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            // No particle_sort_compact here anymore — active-block detection now runs
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

        // Substeps are batched into multiple command buffers/submits instead of one --
        // LP's stiff-terrain scenes (50 MPa sandy soil) routinely need several hundred
        // substeps in a single frame, and encoding them all into one command buffer
        // exhausted this GPU backend within seconds (`wgpu error: Out of Memory` from
        // this same `queue.submit`, reported against LP's own scene 2026-07-01).
        // Bisected empirically: 200 substeps in one submit reliably OOMs, 64 is stable
        // (matches this engine's own tested default, see `max_substeps_per_step`'s doc
        // comment) -- this is a real per-submit resource ceiling on the backend/driver
        // actually exercised, not a value derived from any GPU spec, so a different
        // backend may need a different number. Blocking between chunks is required too
        // -- unblocked back-to-back submits queue up faster than the GPU drains them and
        // hit the same OOM even with batching. Only blocks BETWEEN chunks, never after
        // the last one -- typical scenes (well under 64 substeps/frame) produce exactly
        // one chunk and pay zero extra sync cost, same as before this fix existed. Only
        // LP's stiff-terrain scale (hundreds of substeps/frame) pays the blocking cost,
        // and only for the chunks beyond the first.
        const SUBSTEP_BATCH_SIZE: usize = 64;
        let mut chunks = bind_groups[..sub_dts.len()]
            .chunks(SUBSTEP_BATCH_SIZE)
            .peekable();
        // Split pure CPU command-building time from GPU-completion wait time --
        // "encode_ns" previously bundled both under one name, hiding whether a slow
        // step_frame() was a CPU-side encoding problem or genuinely GPU-execution-bound.
        let mut pure_encode_ns = 0.0f32;
        let mut wait_ns = 0.0f32;
        while let Some(chunk) = chunks.next() {
            let chunk_encode_start = std::time::Instant::now();
            let mut sub_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("mpm_substep_batch"),
                    });
            for bg in chunk {
                self.encode_substep(
                    &mut sub_encoder,
                    bg,
                    particle_wg,
                    force_fields_needed,
                    contact_active,
                    thermal_active,
                );
            }
            self.queue.submit(std::iter::once(sub_encoder.finish()));
            pure_encode_ns += chunk_encode_start.elapsed().as_secs_f32() * 1.0e9;
            if chunks.peek().is_some() {
                let wait_start = std::time::Instant::now();
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                wait_ns += wait_start.elapsed().as_secs_f32() * 1.0e9;
            }
        }
        let encode_ns = pure_encode_ns;
        // Repurposed: real GPU-completion wait time between chunks, not always 0 --
        // this IS where GPU execution time shows up for multi-chunk (>64 substep) frames.
        let submit_ns = wait_ns;

        // Async GPU → CPU readback — never blocks the render thread.
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

        // Check if a previous async readback completed -- Ok, Err, or still pending.
        // Real fix (2026-07-05, see project memory
        // emerge_locomotion_root_cause_and_fix / issue #10): the OLD code only
        // handled Ok here, silently dropping Err. That left the staging buffer
        // mapped forever (finish_readback, the only unmapper, was never called)
        // and pending_readback stuck Some forever (blocking every future
        // readback) -- until something else tried to map the same buffer again
        // and hit a real "Buffer is already mapped" panic. Every completion path
        // now explicitly unmaps, regardless of Ok/Err.
        let readback_done = self
            .pending_readback
            .as_ref()
            .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
        if let Some(result) = readback_done {
            self.pending_readback = None;
            // REAL BUG FOUND AND FIXED 2026-07-15 (see project memory
            // gpu_readback_error_path_bug_issue10 for the original 2026-07-05
            // investigation this extends): the device-lost check at the TOP of
            // step_frame only guards against a device that was ALREADY lost before
            // this call started. It says nothing about a device that dies DURING
            // this same call -- e.g. an earlier queue.submit() in this frame's own
            // chunked substep loop above triggers an uncaptured OOM error, which
            // sets `device_lost` via the registered callback, and THEN this exact
            // block runs and blindly trusts whatever the async flag says (Ok or
            // Err) without re-checking. Confirmed via a real 16,000-step GPU
            // long-horizon contact test: crashed with "Buffer 'mpm_particle_staging'
            // has been destroyed" inside `finish_readback`'s `get_mapped_range` --
            // the exact same failure class the original issue #10 fix already
            // covers for `sync_particles_blocking`'s ENTRY guard, just reachable
            // here through a path that guard never touches (mid-call loss, not
            // pre-call loss). Once the device is lost, the staging buffer may
            // already be destroyed regardless of what the async result claims --
            // Ok can't be trusted post-loss either, so this skips BOTH the Ok and
            // Err branches (not just adding a check to one), avoiding the unmap()
            // panic risk in `abandon_readback` too.
            if self.is_device_lost() {
                // Do nothing -- neither finish_readback nor abandon_readback is
                // safe to call once the device is confirmed lost.
            } else if result.is_err() {
                self.readback_error_count += 1;
                self.buffers.abandon_readback();
            } else {
                let gpu_particles = self.buffers.finish_readback(self.particle_count);

                // CPU plasticity pass — skipped if all materials run plasticity on GPU.
                //
                // IMPORTANT: GPU g2p already integrated F via `F_new = (I + dt·C)·F_old`.
                // Zero affine before update_particle so only the plasticity projection runs.
                // Restore GPU affine afterwards so next P2G APIC term is correct.
                // The new MaterialModel API takes (&mut Particles, usize) — convert AoS to SoA,
                // run the CPU pass, then scatter results back.
                if any_cpu {
                    // Stash GPU affine matrices — we zero affine for the plasticity call then restore.
                    let gpu_affines: Vec<_> =
                        gpu_particles.iter().map(|p| p.velocity_gradient).collect();
                    // Copy readback into AoS cpu mirror (zeroing affine for plasticity).
                    for (p_gpu, p_cpu) in gpu_particles.iter().zip(self.particles.iter_mut()) {
                        *p_cpu = *p_gpu;
                        p_cpu.velocity_gradient = glam::Mat2::ZERO;
                    }
                    // Build SoA wrapper, run CPU plasticity, scatter plastic state back.
                    // Skip sleeping particles — same reasoning as every GPU-side pass: their
                    // F/plastic state is frozen, re-running plasticity on unchanged input
                    // wastes exactly the compute sleep/wake exists to avoid. Before the
                    // Particles::push() fix above, this loop silently ran on every particle
                    // regardless of sleep state, because the AoS->SoA conversion dropped it.
                    let mut soa = Particles::from(std::mem::take(&mut self.particles));
                    for i in 0..soa.len() {
                        if soa.sleeping[i] {
                            continue;
                        }
                        self.registry.get(soa.material_id[i]).update_particle(
                            &mut soa,
                            i,
                            self.last_sub_dt,
                        );
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

    /// Add a non-uniform body force field for the GPU path.
    /// Entries are uploaded and dispatched every substep until cleared.
    /// Panics if `MAX_FORCE_FIELDS` is exceeded.
    pub fn add_force_field_gpu(&mut self, entry: GpuFieldEntry) {
        assert!(
            self.force_field_entries.len() < MAX_FORCE_FIELDS,
            "add_force_field_gpu: MAX_FORCE_FIELDS ({MAX_FORCE_FIELDS}) exceeded"
        );
        self.force_field_entries.push(entry);
    }

    /// Remove all GPU force field entries.
    pub fn clear_force_fields_gpu(&mut self) {
        self.force_field_entries.clear();
    }

    /// Update the preferred grip direction live (e.g. player/AI steering input) --
    /// GPU counterpart to `DirectionalContactGrip::set_easy_direction`. Takes effect
    /// on the very next `step_frame` (re-uploaded fresh every frame, see that
    /// function's own `grip_params` upload). Only has a visible effect on particles
    /// with `contact_group != 0` and only once `set_grip_friction` has set
    /// `mu_easy != mu_resist` -- symmetric friction (the default) makes direction
    /// irrelevant, same "no bias without input" principle as `RatchetFrictionBoundary`.
    ///
    /// HONEST STATUS (2026-07-16): this API is real and correctly wired -- verified
    /// reaching `resolve_contact.wgsl`'s `grip_params` uniform. The RESULTING
    /// directional effect is correctly signed (the "easy" direction genuinely retains
    /// more speed) but its MAGNITUDE is measurably unstable run to run, unlike CPU's
    /// `DirectionalContactGrip` which is consistent -- see
    /// `gpu_directional_grip_is_direction_aware`'s `#[ignore]` reason in `tests/gpu.rs`
    /// for the real measured numbers and the likely (not confirmed) root cause. Usable
    /// for real, but don't present it as equivalent to CPU's steering yet.
    pub fn set_grip_direction(&mut self, direction: glam::Vec2) {
        self.grip_params.easy_direction = direction.normalize_or_zero();
    }

    /// Update the grip friction coefficients live -- GPU counterpart to
    /// `DirectionalContactGrip::set_friction`. Set `mu_easy == mu_resist` to disable
    /// directional asymmetry entirely (ordinary symmetric Coulomb friction) whenever
    /// there's no real steering intent.
    pub fn set_grip_friction(&mut self, mu_easy: f32, mu_resist: f32) {
        assert!(
            (0.0..=1.0).contains(&mu_easy) && (0.0..=1.0).contains(&mu_resist),
            "set_grip_friction: mu_easy and mu_resist must be in [0.0, 1.0]"
        );
        self.grip_params.mu_easy = mu_easy;
        self.grip_params.mu_resist = mu_resist;
    }

    /// Attach day-night/ambient thermal diffusion -- GPU counterpart to CPU's
    /// `Simulation::with_thermal`/`thermal_config_mut`. Real PDE (Fourier's law
    /// `∂T/∂t = α·∇²T` plus Newton cooling), see `GpuThermalParams`' own doc. Enables
    /// all 4 thermal passes starting next `step_frame`; call `set_thermal_ambient`
    /// afterward for live day-night oscillation.
    ///
    /// - `conductivity_w_m_k` / `heat_capacity_j_kg_k`: real SI material constants
    /// - `grid_cell_size_m`: physical cell size (pass `SimConfig::dx_meters`, NOT
    ///   `grid_cell_size` which is always 1.0 -- same trap `ThermalConfig`'s own doc
    ///   warns about)
    /// - `ambient`: background temperature; `cooling_rate`: Newton cooling k_c (1/s,
    ///   0.0 = none)
    pub fn attach_thermal_gpu(
        &mut self,
        conductivity_w_m_k: f32,
        heat_capacity_j_kg_k: f32,
        grid_cell_size_m: f32,
        ambient: f32,
        cooling_rate: f32,
    ) {
        let alpha =
            conductivity_w_m_k / (heat_capacity_j_kg_k * grid_cell_size_m * grid_cell_size_m);
        self.thermal_params = GpuThermalParams {
            alpha,
            ambient,
            cooling_rate,
            enabled: 1,
        };
    }

    /// Update the ambient/boundary temperature live (e.g. day-night oscillation) --
    /// GPU counterpart to CPU's `thermal_config_mut().ambient = ...`. No-op (with a
    /// debug assert) if thermal hasn't been attached yet.
    pub fn set_thermal_ambient(&mut self, ambient: f32) {
        debug_assert!(
            self.thermal_params.enabled != 0,
            "set_thermal_ambient: call attach_thermal_gpu first"
        );
        self.thermal_params.ambient = ambient;
    }

    /// Encode one substep's passes into an existing encoder. No submission — caller batches.
    fn encode_substep(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        particle_wg: u32,
        force_fields_needed: bool,
        contact_active: bool,
        thermal_active: bool,
    ) {
        {
            // GPU sparse grid Phase 1 — re-detect active blocks from CURRENT particle
            // positions, every substep, immediately before grid_clear uses the result.
            //
            // Real bug found via direct testing (gpu_sleep_freezes_settled_particles
            // regressed, plus a native crash — see mpm_technique_survey memory note):
            // particle_sort's once-per-frame active-block detection (computed from
            // frame-START positions) went stale by substep 2+ of the same frame, since
            // particles move every substep. Fixed by re-running clear+count+compact (NOT
            // scan/scatter — those only matter for the once-per-frame sort permutation,
            // unrelated to grid_clear correctness) every substep.
            //
            // Second real bug, found via a long-running headless diagnostic AFTER the
            // above fix (basic_sand_gpu blew up after ~1500 frames, ~1-in-5 runs): a block
            // that stops being active (a particle moves away) was never cleared again —
            // grid_clear only ever clears CURRENTLY active blocks, so a block's last P2G
            // contribution sat there permanently until some particle wandered back near it
            // much later, at which point P2G's atomic ADD compounded onto the stale
            // residual. Dense grid_clear never had this problem (it unconditionally zeroed
            // every cell every substep regardless of activity). Fix: active_block_swap
            // (dispatched FIRST, before clear/count/compact) snapshots this substep's
            // about-to-be-overwritten active list into active_block_ids_prev/count_prev,
            // and grid_clear processes the union of both lists — a genuine one-substep
            // grace period. See active_block_swap_main's doc comment in particle_sort.wgsl
            // for the full reasoning, including a first attempt at this fix that was wrong
            // (reset happened in the same substep it was used in, giving zero actual grace
            // period).
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("active_block_refresh"),
                timestamp_writes: self.profile_writes(0),
            });
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_pipeline(&self.pipelines.active_block_swap);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_count);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            pass.set_pipeline(&self.pipelines.particle_sort_compact);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_clear"),
                timestamp_writes: self.profile_writes(1),
            });
            pass.set_pipeline(&self.pipelines.grid_clear);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            // GPU sparse grid Phase 1: one workgroup per potential active-block slot, for
            // EACH of the two lists (this substep's + last substep's grace period) — fixed
            // worst-case size (2 * NUM_BLOCKS), not grid_res-dependent anymore. Most slots
            // beyond their list's real count exit immediately via the shader's own guard.
            // See grid_clear.wgsl.
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("p2g"),
                timestamp_writes: self.profile_writes(2),
            });
            pass.set_pipeline(&self.pipelines.p2g);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        // Skipped entirely (not just an empty loop body) when NO particle anywhere has
        // `contact_group != 0` this frame -- mirrors CPU's `Grid::has_contact_activity()`
        // gate exactly (`gather_contact_point_cloud` in `transfer.rs` is a documented no-op
        // in that case). See `contact_active`'s doc (computed in `step_frame`) for the real
        // measured cost this avoids (37.5%/5.66ms of a substep on a pure fluid scene).
        if contact_active {
            // Multi-field contact (GPU port, first slice) -- must run strictly after p2g
            // (reads grip mass p2g just scattered) and strictly before grid_update, same
            // ordering CPU's own step.rs enforces between scatter_particles_to_grid,
            // gather_contact_point_cloud, and update_velocities. A real, separate compute
            // pass (not folded into p2g_main itself) specifically so this barrier is
            // enforced -- see p2g.wgsl's gather_contact_points_main doc.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gather_contact_points"),
                timestamp_writes: self.profile_writes(3),
            });
            pass.set_pipeline(&self.pipelines.gather_contact_points);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_update"),
                timestamp_writes: self.profile_writes(4),
            });
            pass.set_pipeline(&self.pipelines.grid_update);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            // GPU sparse grid Phase 2: same active-block dispatch pattern as grid_clear (see
            // grid_update.wgsl's doc comment) -- was the last remaining O(grid_res²)-dispatch
            // pass; now bounded to occupied blocks (+ one substep's grace period) instead.
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
        }
        // Skipped entirely under the same `contact_active` gate as `gather_contact_points`
        // above -- safe ONLY because `g2p.wgsl` itself is gated on the identical flag (see
        // `contact_active`'s doc): when false, G2P reads the plain `grid` velocity directly
        // instead of `resolved_rest_v`/`resolved_grip_v`, so this pass never needing to have
        // populated them is correct, not just "probably fine" -- both gates were added
        // together, mirroring CPU's single `contact_active` check in `transfer.rs` exactly.
        if contact_active {
            // Multi-field contact (GPU port) -- must run after grid_update (needs the
            // DECODED, gravity-applied total velocity grid_update just produced) and
            // before g2p (which will read the resolved velocities this pass writes),
            // same ordering CPU's own step.rs enforces between update_velocities and
            // resolve_contact. See resolve_contact.wgsl's resolve_contact_main doc.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resolve_contact"),
                timestamp_writes: self.profile_writes(5),
            });
            pass.set_pipeline(&self.pipelines.resolve_contact);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g2p"),
                timestamp_writes: self.profile_writes(6),
            });
            pass.set_pipeline(&self.pipelines.g2p);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particles_update"),
                timestamp_writes: self.profile_writes(7),
            });
            pass.set_pipeline(&self.pipelines.particles_update);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        // Skipped entirely (not just an empty loop body) when force_fields_main is
        // provably a no-op for every particle this frame -- see force_fields_needed's
        // doc comment above (step_frame) for the full reasoning and the real measured
        // cost this avoids. When skipped, the velocity this pass would have re-clamped
        // is exactly what g2p already clamped to (particles_update's only effect on v
        // is multiplicative damping, never amplifying), so this is a correctness-
        // preserving skip, not an approximation.
        if force_fields_needed {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("force_fields"),
                timestamp_writes: self.profile_writes(8),
            });
            pass.set_pipeline(&self.pipelines.force_fields);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        }
        // Day-night/ambient thermal diffusion (GPU port) — skipped ENTIRELY (not just
        // early-returning per-thread) when no thermal system is attached, same
        // dispatch-skip discipline as contact_active/force_fields_needed above. Runs
        // after force_fields, matching CPU's own `ThermalDiffusion::apply` ordering
        // ("after force fields, before state projection") — fully decoupled from
        // mechanics (operates only on particle.temperature), so exact ordering
        // relative to force_fields doesn't affect correctness, just matches CPU's own
        // call site for consistency.
        if thermal_active {
            let grid_res = self.config.grid_res as u32;
            let cell_wg = (grid_res * grid_res).div_ceil(64);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("thermal_clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.thermal_clear);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("thermal_p2g"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.thermal_p2g);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("thermal_normalize_laplacian"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.thermal_normalize_laplacian);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("thermal_g2p"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.thermal_g2p);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        if let Some(profiling) = &self.profiling {
            let n = PROFILE_PASS_LABELS.len() as u32;
            encoder.resolve_query_set(&profiling.query_set, 0..n * 2, &profiling.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(
                &profiling.resolve_buf,
                0,
                &profiling.readback_buf,
                0,
                (n * 2) as u64 * 8,
            );
        }
    }
}

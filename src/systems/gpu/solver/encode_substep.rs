//! `encode_substep` -- the 7-8 labeled compute passes for one MLS-MPM substep,
//! split out of `step.rs` (was ~320 of its ~1000 lines). Self-contained: takes
//! its encoder/bind-group/gates as explicit params, captures no per-frame
//! timing-sensitive state (that stays in `step_frame`, `step.rs`, per that
//! file's own "highest-risk, done last and alone" doc comment).

use super::super::step_params::NUM_BLOCKS;
use super::GpuSimulation;

/// Per-substep dispatch-skip gates -- bundled instead of 4 separate bool args to
/// `encode_substep` (crossed the project's own no-`#[allow]` line for argument count,
/// same real precedent as `P2GParticleState` in `spacetime::transfer`: a struct, not a
/// suppressed lint). Each `true` means the corresponding real, measured optional pass
/// actually runs this substep; `false` means it's skipped entirely, not just a no-op.
#[derive(Clone, Copy)]
pub(super) struct SubstepGates {
    pub(super) force_fields_needed: bool,
    pub(super) contact_active: bool,
    pub(super) thermal_active: bool,
    pub(super) resource_active: bool,
    /// `true` means `g2p_asflip_fused` runs INSTEAD OF the ordinary `g2p` +
    /// `particles_update` pair for this substep -- not an extra optional pass being
    /// skipped, a REPLACEMENT of two passes with one. See `g2p_asflip_fused.wgsl`'s own
    /// doc for why fusion is structurally required.
    pub(super) asflip_active: bool,
}

impl GpuSimulation {
    /// Encode one substep's passes into an existing encoder. No submission — caller batches.
    pub(super) fn encode_substep(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        particle_wg: u32,
        gates: SubstepGates,
    ) {
        let SubstepGates {
            force_fields_needed,
            contact_active,
            thermal_active,
            resource_active,
            asflip_active,
        } = gates;
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
        }
        if asflip_active {
            // ASFLIP (GPU port) -- REPLACES the ordinary g2p + particles_update pair
            // below with one fused dispatch. See g2p_asflip_fused.wgsl's own doc for
            // why fusion is structurally required. Reuses the "g2p" profiling slot;
            // the "particles_update" slot is simply unused this substep (profiling is
            // diagnostic-only, not correctness-relevant).
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("g2p_asflip_fused"),
                timestamp_writes: self.profile_writes(6),
            });
            pass.set_pipeline(&self.pipelines.g2p_asflip_fused);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
            pass.dispatch_workgroups(particle_wg, 1, 1);
        } else {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("g2p"),
                    timestamp_writes: self.profile_writes(6),
                });
                pass.set_pipeline(&self.pipelines.g2p);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
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
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
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
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        // Resource regrowth (GPU port) — same real dispatch-skip discipline as thermal
        // above. Independent system (own buffers/group), can run alongside thermal in
        // the same frame (both gated separately) even though both currently carry
        // state in particle.temperature -- a real scene using both simultaneously
        // would need a genuine second carrier, same limitation the CPU precedent has.
        if resource_active {
            let grid_res = self.config.grid_res as u32;
            let cell_wg = (grid_res * grid_res).div_ceil(64);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resource_clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.resource_clear);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resource_p2g"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.resource_p2g);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resource_normalize_laplacian"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.resource_normalize_laplacian);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("resource_g2p"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.resource_g2p);
                pass.set_bind_group(0, bg, &[]);
                pass.set_bind_group(1, &self.contact_bind_group, &[]);
                pass.set_bind_group(2, &self.thermal_bind_group, &[]);
                pass.set_bind_group(3, &self.resource_bind_group, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        if let Some(profiling) = &self.profiling {
            let n = super::PROFILE_PASS_LABELS.len() as u32;
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

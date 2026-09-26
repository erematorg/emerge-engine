//! `encode_substep` -- the 7-8 labeled dispatch stages for one MLS-MPM substep,
//! split out of `step.rs` (was ~320 of its ~1000 lines). Self-contained: takes
//! its compute pass/bind-group/gates as explicit params, captures no per-frame
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
    /// Real GPU port of the CPU-proven fluid incompressibility pressure
    /// projection (`fluid_pressure.wgsl`) -- mirrors CPU's own
    /// `SimConfig::fluid_pressure_iterations` exactly (0 = off, the default,
    /// every existing scene unaffected; N = real outer-corrector-pass count,
    /// same PISO/SIMPLE-family repeated-correction technique CPU's own
    /// `step.rs` doc already explains). 0 for every scene as of this
    /// writing -- no caller sets it above 0 yet (see the real-time fluid
    /// pressure-projection plan for the real remaining contact-detection
    /// wiring that will).
    pub(super) fluid_pressure_iterations: u32,
    /// Re-detect which grid blocks are occupied this substep (swap+clear -> count ->
    /// compact). `false` reuses the previous substep's list, which stays correct while no
    /// particle can have moved far enough to scatter outside it -- see `step_frame`'s
    /// `active_block_refresh_interval` for the real displacement bound this is derived
    /// from. Measured at ~20us of a ~170us substep on the dam-break demo.
    pub(super) refresh_active_blocks: bool,
}

impl GpuSimulation {
    /// Encode one substep's dispatches into an already-open compute pass. The caller sets
    /// bind groups 1-3 once per pass (they never change between substeps); this sets group
    /// 0 (the per-substep `StepParams` slot). No submission -- caller batches.
    ///
    /// Every stage used to open its own compute pass and re-set all four bind groups:
    /// ~7 passes and ~28 `set_bind_group` calls per substep, measured at ~0.4-0.5ms of
    /// CPU encode per substep (27-33ms per 60-substep frame) -- more than the GPU work
    /// itself on the dam-break demo. wgpu already synchronizes successive dispatches
    /// within one pass (each dispatch is its own usage scope, barriers inserted between
    /// them), which `active_block_refresh`'s four dispatches always relied on.
    pub(super) fn encode_substep(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        bg: &wgpu::BindGroup,
        particle_wg: u32,
        gates: SubstepGates,
    ) {
        let SubstepGates {
            refresh_active_blocks,
            force_fields_needed,
            contact_active,
            thermal_active,
            resource_active,
            asflip_active,
            fluid_pressure_iterations,
        } = gates;
        pass.set_bind_group(0, bg, &[]);
        // Contact-point counters: refilled by `gather_contact_points` and read by
        // `resolve_contact` every substep, so they are zeroed every substep -- unlike the
        // active-block lists around them, which may be reused (see `refresh_active_blocks`).
        if contact_active {
            pass.set_pipeline(&self.pipelines.contact_counts_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256, grid-stride
        }
        if refresh_active_blocks {
            // GPU sparse grid Phase 1 -- re-detect active blocks from CURRENT particle
            // positions, every substep, immediately before grid_clear uses the result.
            // particle_sort's once-per-frame detection (computed from frame-START
            // positions) would go stale by substep 2+ since particles move every
            // substep, so clear+count+compact reruns every substep (NOT scan/scatter --
            // those only matter for the once-per-frame sort permutation).
            //
            // A block that stops being active (a particle moves away) must still get
            // cleared once more -- grid_clear only clears CURRENTLY active blocks, so
            // without this a block's last P2G contribution would sit there permanently
            // until a particle wandered back near it, and P2G's atomic ADD would
            // compound onto the stale residual. active_block_swap_and_clear (dispatched FIRST,
            // before clear/count/compact) snapshots this substep's about-to-be-
            // overwritten active list into active_block_ids_prev/count_prev, and
            // grid_clear processes the union of both lists -- a one-substep grace
            // period. See the swap pass's doc comment in particle_sort.wgsl.
            self.profile_stamp(pass, 0, false);
            pass.set_pipeline(&self.pipelines.active_block_swap_and_clear);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            pass.set_pipeline(&self.pipelines.particle_sort_count);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            pass.set_pipeline(&self.pipelines.particle_sort_compact);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
            self.profile_stamp(pass, 0, true);
        }
        {
            self.profile_stamp(pass, 1, false);
            pass.set_pipeline(&self.pipelines.grid_clear);
            // GPU sparse grid Phase 1: one workgroup per potential active-block slot, for
            // EACH of the two lists (this substep's + last substep's grace period) -- fixed
            // worst-case size (2 * NUM_BLOCKS), not grid_res-dependent anymore. Most slots
            // beyond their list's real count exit immediately via the shader's own guard.
            // See grid_clear.wgsl.
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
            self.profile_stamp(pass, 1, true);
        }
        {
            self.profile_stamp(pass, 2, false);
            pass.set_pipeline(&self.pipelines.p2g);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            self.profile_stamp(pass, 2, true);
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
            self.profile_stamp(pass, 3, false);
            pass.set_pipeline(&self.pipelines.gather_contact_points);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            self.profile_stamp(pass, 3, true);
        }
        {
            self.profile_stamp(pass, 4, false);
            pass.set_pipeline(&self.pipelines.grid_update);
            // GPU sparse grid Phase 2: same active-block dispatch pattern as grid_clear (see
            // grid_update.wgsl's doc comment) -- was the last remaining O(grid_res²)-dispatch
            // pass; now bounded to occupied blocks (+ one substep's grace period) instead.
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
            self.profile_stamp(pass, 4, true);
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
            self.profile_stamp(pass, 5, false);
            pass.set_pipeline(&self.pipelines.resolve_contact);
            pass.dispatch_workgroups(2 * NUM_BLOCKS as u32, 1, 1);
            self.profile_stamp(pass, 5, true);
        }
        // Real GPU port of the CPU-proven Chorin-style fluid incompressibility
        // pressure projection (`fluid_pressure.wgsl`) -- runs after grid_update
        // (needs the real, gravity/boundary-applied velocity field) and before
        // G2P (particles must gather the CORRECTED field), same real ordering
        // CPU's own `step.rs` doc already establishes for this exact mechanism.
        // `fluid_pressure_iterations == 0` (every scene as of this writing --
        // no caller sets it above 0 yet) skips all of this entirely, same
        // dispatch-skip discipline as `contact_active`/`force_fields_needed`
        // above -- zero cost, byte-identical behavior to before this feature
        // existed.
        if fluid_pressure_iterations > 0 {
            // Real per-CELL passes (grid_res x grid_res domain, workgroup_size
            // 16x16 in fluid_pressure.wgsl) -- NOT `particle_wg`, which is
            // sized for the particle count and would under- or over-dispatch
            // depending on how particle_count compares to grid_res². Ceiling
            // division so a grid_res not a multiple of 16 is still fully
            // covered (the shader's own bounds check discards the excess).
            let grid_wg = (self.config.grid_res as u32).div_ceil(16);
            {
                pass.set_pipeline(&self.pipelines.fluid_pressure_setup);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
            // Real, fixed, even sweep count per outer iteration -- matches
            // CPU's own `GS_CORRECTION_SWEEPS=10` (see `pressure.rs`'s own
            // doc for why 10, not fewer or more: a live per-sweep convergence
            // dump showed the max per-cell delta dropping ~100x by sweep 10,
            // an order of magnitude below the pressure field's own working
            // scale). EVEN so the final answer always lands back in
            // `fp_pressure_a`, which `fluid_pressure_correct` reads
            // unconditionally -- see fluid_pressure.wgsl's own module doc.
            const JACOBI_SWEEPS: u32 = 30;
            for sweep in 0..JACOBI_SWEEPS {
                let pipeline = if sweep % 2 == 0 {
                    &self.pipelines.fluid_pressure_jacobi_a_to_b
                } else {
                    &self.pipelines.fluid_pressure_jacobi_b_to_a
                };
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.fluid_pressure_correct);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
        }
        if asflip_active {
            // ASFLIP (GPU port) -- REPLACES the gather + update half of `g2p_update` with
            // one fused dispatch of its own. See g2p_asflip_fused.wgsl's own doc for why
            // fusion is structurally required.
            self.profile_stamp(pass, 6, false);
            pass.set_pipeline(&self.pipelines.g2p_asflip_fused);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            self.profile_stamp(pass, 6, true);
            // Its force-field stage stays a separate pass. Skipped entirely when
            // force_fields_main is provably a no-op for every particle this frame -- see
            // force_fields_needed's doc comment (step_frame).
            if force_fields_needed {
                self.profile_stamp(pass, 7, false);
                pass.set_pipeline(&self.pipelines.force_fields);
                pass.dispatch_workgroups(particle_wg, 1, 1);
                self.profile_stamp(pass, 7, true);
            }
        } else {
            // G2P gather -> F update/plasticity/position -> force fields + sleep/wake, one
            // dispatch. The force-field stage runs unconditionally here: when
            // `force_fields_needed` is false it is a no-op (no fields, no tags, sleep
            // threshold 0, and the velocity clamp is already satisfied by the gather's).
            self.profile_stamp(pass, 6, false);
            pass.set_pipeline(&self.pipelines.g2p_update);
            pass.dispatch_workgroups(particle_wg, 1, 1);
            self.profile_stamp(pass, 6, true);
        }
        // Day-night/ambient thermal diffusion (GPU port) -- skipped ENTIRELY (not just
        // early-returning per-thread) when no thermal system is attached, same
        // dispatch-skip discipline as contact_active/force_fields_needed above. Runs
        // after force_fields, matching CPU's own `ThermalDiffusion::apply` ordering
        // ("after force fields, before state projection") -- fully decoupled from
        // mechanics (operates only on particle.temperature), so exact ordering
        // relative to force_fields doesn't affect correctness, just matches CPU's own
        // call site for consistency.
        if thermal_active {
            let grid_res = self.config.grid_res as u32;
            let cell_wg = (grid_res * grid_res).div_ceil(64);
            {
                pass.set_pipeline(&self.pipelines.thermal_clear);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.thermal_p2g);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.thermal_normalize_laplacian);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.thermal_g2p);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        // Resource regrowth (GPU port) -- same real dispatch-skip discipline as thermal
        // above. Independent system (own buffers/group), can run alongside thermal in
        // the same frame (both gated separately) even though both currently carry
        // state in particle.temperature -- a real scene using both simultaneously
        // would need a genuine second carrier, same limitation the CPU precedent has.
        if resource_active {
            let grid_res = self.config.grid_res as u32;
            let cell_wg = (grid_res * grid_res).div_ceil(64);
            {
                pass.set_pipeline(&self.pipelines.resource_clear);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.resource_p2g);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.resource_normalize_laplacian);
                pass.dispatch_workgroups(cell_wg, 1, 1);
            }
            {
                pass.set_pipeline(&self.pipelines.resource_g2p);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        // The GPU's own CFL, LAST in the substep: it folds the particles' post-update
        // bounds (accumulated by `g2p_update`) into the NEXT substep's timestep, so every
        // pass that consumes this substep's dt -- thermal diffusion and resource regrowth
        // included -- must already have run. (Dispatching it earlier silently gave those
        // two the next substep's dt, and zero on the frame's last substep: the thermal
        // cooling test measured 95.5 where the analytical answer is 68.5.)
        {
            self.profile_stamp(pass, 8, false);
            pass.set_pipeline(&self.pipelines.cfl_commit);
            pass.dispatch_workgroups(1, 1, 1);
            self.profile_stamp(pass, 8, true);
        }
    }
}

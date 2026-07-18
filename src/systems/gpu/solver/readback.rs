use super::GpuSimulation;
use crate::systems::gpu::step_params::{
    ContactDebugParams, MAX_CONTACT_POINTS_PER_BLOCK, NUM_BLOCKS, NUM_CONTACT_BLOCKS,
};

impl GpuSimulation {
    /// Download particles from GPU to CPU synchronously (diagnostics / one-shot use).
    /// Prefer the async readback path in step_frame for per-frame use.
    pub fn download_particles_blocking(&mut self) {
        let flag = self
            .buffers
            .begin_readback(&self.device, &self.queue, self.particle_count);
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        if let Ok(mut g) = flag.lock() {
            g.take();
        }
        self.particles = self.buffers.finish_readback(self.particle_count);
    }

    /// Verification-only accessor: read back `sorted_particle_ids` as a `Vec<u32>`.
    /// Used by tests to confirm the particle_sort pipeline produces a valid permutation —
    /// not part of the render/game-loop API.
    pub fn sorted_particle_ids_blocking(&self) -> Vec<u32> {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.sorted_particle_ids,
            self.particle_count,
        )
    }

    /// Test/diagnostic readback for the GPU sparse grid Phase 1 active-block list — the
    /// first `active_block_count_blocking()` entries are valid; the rest are stale/unused.
    pub fn active_block_ids_blocking(&self) -> Vec<u32> {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.active_block_ids,
            NUM_BLOCKS,
        )
    }

    /// Test/diagnostic readback for how many entries in `active_block_ids_blocking()` are
    /// valid this frame.
    pub fn active_block_count_blocking(&self) -> u32 {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.active_block_count,
            1,
        )[0]
    }

    /// Test/diagnostic readback of the dense grid buffer — 4 f32 per cell (momentum.x,
    /// momentum.y, mass, _pad), same field order as the WGSL `Cell` struct, flat-indexed
    /// `(y * grid_res + x) * 4`. Lets tests verify grid_clear actually zeroed cells far from
    /// any particle (the failure mode a block-boundary mapping bug would produce: stale,
    /// never-cleared mass/momentum left behind in an unrelated block).
    pub fn grid_cells_blocking(&self) -> Vec<f32> {
        let cell_floats = self.config.grid_res * self.config.grid_res * 4;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.grid,
            cell_floats,
        )
    }

    /// Test/diagnostic readback of the multi-field contact "grip" accumulator (GPU port,
    /// first slice) — same 4-f32-per-cell layout as `grid_cells_blocking`. Lets tests
    /// verify grip mass/momentum scatter matches CPU's `Grid::add_grip_mass_momentum`.
    pub fn grip_grid_cells_blocking(&self) -> Vec<f32> {
        let cell_floats = self.config.grid_res * self.config.grid_res * 4;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.grip_grid,
            cell_floats,
        )
    }

    /// Test/diagnostic readback of the per-block contact point-cloud counts (GPU port) —
    /// `NUM_CONTACT_BLOCKS` (4096) `u32` entries, one per dedicated contact-point spatial
    /// block (see `MAX_CONTACT_POINTS_PER_BLOCK`'s doc in `step_params.rs`). A count can
    /// exceed `MAX_CONTACT_POINTS_PER_BLOCK` on overflow (a real, observable signal, not
    /// silently capped) — callers must clamp before indexing `contact_points_blocking`.
    pub fn contact_point_counts_blocking(&self) -> Vec<u32> {
        self.buffers.readback_u32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.contact_point_counts,
            NUM_CONTACT_BLOCKS,
        )
    }

    /// Test/diagnostic readback of the full contact point-cloud buffer (GPU port) —
    /// `NUM_CONTACT_BLOCKS * MAX_CONTACT_POINTS_PER_BLOCK` `vec4<f32>` entries
    /// (position.x, position.y, label, unused), flat-indexed
    /// `block * MAX_CONTACT_POINTS_PER_BLOCK + slot`. Only the first
    /// `min(count, MAX_CONTACT_POINTS_PER_BLOCK)` entries per block (per
    /// `contact_point_counts_blocking`) are meaningful; the rest are stale/unused.
    pub fn contact_points_blocking(&self) -> Vec<f32> {
        let floats = NUM_CONTACT_BLOCKS * MAX_CONTACT_POINTS_PER_BLOCK * 4;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.contact_points,
            floats,
        )
    }

    /// Debug/test-only: runs `resolve_contact.wgsl`'s `debug_fit_normal_main` — the SAME
    /// neighbor-expanded, distance-filtered `gather_local_points` the real per-substep
    /// `resolve_cell` uses — centered on `node_pos`. Returns `(normal, valid)` — `valid`
    /// is `false` if the fit found no confident answer (mirrors CPU's
    /// `fit_contact_normal_lr`'s `Option<Vec2>`). Verifies the Newton-Raphson LR fit's
    /// WGSL port against a known reference case, the same way CPU's own
    /// `fit_contact_normal_lr_tests` module unit-tests the fit separately from the full
    /// `resolve_contact` integration. Blocking — test/diagnostic use only.
    ///
    /// CHANGED 2026-07-18 (GPU sparse-contact perf pass): `target_block`/`point_count`
    /// are vestigial — the shader no longer reads one un-expanded block's raw points (an
    /// assumption that only held by coincidence at the old coarse partition's
    /// block_size=4, false in general and definitely false against the new, finer,
    /// dedicated contact partition). Kept as parameters only because removing them would
    /// also require reshaping `ContactDebugParams`'s uniform layout for no real benefit.
    pub fn debug_fit_contact_normal_blocking(&self, node_pos: glam::Vec2) -> (glam::Vec2, bool) {
        let params = ContactDebugParams {
            node_pos,
            target_block: 0,
            point_count: 0,
        };
        self.buffers
            .upload_contact_debug_params(&self.queue, &params);

        let bg = self.pipelines.make_bind_group(
            &self.device,
            &self.buffers,
            &self.buffers.step_params_pool[0],
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm_debug_fit_normal"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("debug_fit_normal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.debug_fit_normal);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &self.contact_bind_group, &[]);
            pass.set_bind_group(2, &self.thermal_bind_group, &[]);
            pass.set_bind_group(3, &self.resource_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let out = self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.contact_debug_output,
            4,
        );
        (glam::Vec2::new(out[0], out[1]), out[2] > 0.0)
    }

    /// Test/diagnostic readback of the resolved "grip" field velocity per node —
    /// `grid_res² × vec2<f32>`, written by `resolve_contact_main`. See
    /// `GpuBuffers::resolved_grip_v`'s own doc.
    pub fn resolved_grip_v_blocking(&self) -> Vec<f32> {
        let floats = self.config.grid_res * self.config.grid_res * 2;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.resolved_grip_v,
            floats,
        )
    }

    /// Test/diagnostic readback of the resolved "rest" field velocity per node — same
    /// layout as `resolved_grip_v_blocking`.
    pub fn resolved_rest_v_blocking(&self) -> Vec<f32> {
        let floats = self.config.grid_res * self.config.grid_res * 2;
        self.buffers.readback_f32_blocking(
            &self.device,
            &self.queue,
            &self.buffers.resolved_rest_v,
            floats,
        )
    }
}

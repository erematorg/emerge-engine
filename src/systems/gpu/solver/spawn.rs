//! Particle-population lifecycle on `GpuSimulation`: spawning (which
//! reallocates the per-particle GPU buffers) and CPU<->GPU resync.
//!
//! Split out of `gpu/solver/mod.rs` -- distinct from construction (mod.rs),
//! read-only queries (queries.rs), particle mutation (particles.rs), and the
//! actual dispatch loop (step.rs).

use crate::grid::Grid;
use crate::particle::Particle;
use crate::solver::LcgRng;
use crate::solver::density::estimate_particle_volumes;
use crate::solver::initialize_particles;

use super::GpuSimulation;
use super::build_bind_group_pool;

impl GpuSimulation {
    pub fn spawn_region(
        &mut self,
        spawn: crate::solver::config::SpawnRegion,
    ) -> std::ops::Range<usize> {
        let start = self.particles.len();
        spawn.validate_for_sim(&self.config);
        debug_assert!(
            self.registry.is_registered(spawn.material_id),
            "spawn_region: material_id {} is not registered — call solver.set_material({}, ...) first",
            spawn.material_id,
            spawn.material_id
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        self.particles.extend(new_particles);

        // Recompute initial volumes for the combined particle set using a temporary grid.
        let mut tmp_grid = Grid::new(self.config.grid_res);
        {
            let mut tmp_soa = crate::particle::Particles::from(std::mem::take(&mut self.particles));
            let n = tmp_soa.len();
            estimate_particle_volumes(&mut tmp_soa, &mut tmp_grid, n, true);
            self.particles = tmp_soa.to_vec();
        }

        let n = self.particles.len();

        // Reallocate all GPU buffers that are sized per-particle (including staging).
        self.buffers.particles = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particles"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffers.sorted_particle_ids = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sorted_particle_ids"),
            size: (n * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.buffers.readback_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particle_staging"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.particle_count = n;
        // Real bug fix (2026-07-18): re-arm the sleep-warmup window (see
        // `last_spawn_frame`'s own doc) so freshly-spawned particles get the same
        // "don't sleep-score yet" grace period the initial construction batch
        // already got, instead of getting the real sleep_threshold applied at
        // v=0 on their very first substep -- confirmed live as the actual cause
        // of `material_sandbox_gpu`'s painted particles never responding to
        // gravity (Force impulses still worked, since that's a separate wake path).
        self.last_spawn_frame = self.frame_index;
        self.pending_readback = None; // old staging is gone
        self.buffers.upload_particles(&self.queue, &self.particles);
        // buffers.particles was just reallocated above -- cached bind groups reference
        // the old buffer object and would be stale (or invalid) without this.
        self.bind_group_pool = build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
        self.rebuild_spatial_hash();
        start..n
    }

    /// Rebuilds `spatial_hash` from the current `self.particles` immediately -- for
    /// callers (spawn, explicit sync) that need a query to be safe to call right after
    /// this returns, without depending on the lazy dirty-flag path (`ensure_spatial_hash_fresh`
    /// in queries.rs) firing first. O(N), same real cost class as the linear scans it
    /// replaces for QUERIES, but paid once per mutation instead of once per query -- a
    /// real win whenever more than one query happens against the same particle state
    /// (e.g. LP's ecology calling `sense_local`/`centroid`/`phenotype` per creature per
    /// frame). Readback completion in `step.rs` does NOT call this directly anymore --
    /// it just marks `spatial_hash_dirty`, deferring the rebuild to the first query that
    /// actually needs it (see `spatial_hash`'s own doc in `mod.rs` for why).
    pub(super) fn rebuild_spatial_hash(&mut self) {
        let positions: Vec<glam::Vec2> = self.particles.iter().map(|p| p.x).collect();
        self.spatial_hash
            .borrow_mut()
            .rebuild(&positions, self.particles.len());
        self.spatial_hash_dirty.set(false);
    }

    /// Remove all particles where `predicate` returns true. Returns count removed.
    ///
    /// GPU counterpart to `Simulation::remove_particles` (CPU) -- same predicate
    /// API (LP pattern: tag with a sentinel, then `remove_particles(|p| p.user_tag
    /// == DEAD)`). Unlike CPU's in-place `Vec::retain`, this reallocates every
    /// per-particle GPU buffer to the smaller size and re-uploads -- the exact
    /// same reallocate-and-reupload pattern `spawn_region` already uses to grow,
    /// just shrinking instead. Real, same-cost-class operation, not free -- don't
    /// call every frame for large removals any more than you'd call `spawn_region`
    /// every frame (see that method's own doc).
    ///
    /// Calls `sync_particles_blocking` first: the predicate needs genuinely
    /// current particle state (e.g. current temperature for an evaporation rule),
    /// not whatever the CPU mirror happened to hold from the last readback.
    pub fn remove_particles<F: Fn(&Particle) -> bool>(&mut self, predicate: F) -> usize {
        self.sync_particles_blocking();
        let before = self.particles.len();
        self.particles.retain(|p| !predicate(p));
        let removed = before - self.particles.len();
        if removed == 0 {
            return 0;
        }

        let n = self.particles.len();
        self.buffers.particles = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particles"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.buffers.sorted_particle_ids = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sorted_particle_ids"),
            size: (n * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.buffers.readback_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particle_staging"),
            size: (n * core::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.particle_count = n;
        self.pending_readback = None; // old staging is gone
        self.buffers.upload_particles(&self.queue, &self.particles);
        // buffers.particles was just reallocated above -- cached bind groups reference
        // the old buffer object and would be stale (or invalid) without this.
        self.bind_group_pool = build_bind_group_pool(&self.device, &self.pipelines, &self.buffers);
        self.rebuild_spatial_hash();
        removed
    }

    /// Blocking GPU → CPU particle sync. Updates `self.particles` immediately.
    /// Stalls the CPU until all in-flight GPU work completes — use only after step_frame
    /// when you need current positions right now (e.g. rendering). Not for the hot path.
    pub fn sync_particles_blocking(&mut self) {
        // Safe no-op once the device is lost -- see step_frame's identical guard.
        if self.is_device_lost() {
            return;
        }
        // If an async readback is in-flight, the staging buffer may be mapped or pending map.
        // Wait for it to complete, then consume it to unmap the staging buffer before reuse.
        // Real fix (2026-07-05, issue #10): must distinguish Ok/Err here -- the old
        // code called finish_readback (which calls get_mapped_range) on EITHER, but
        // a failed map has nothing valid to extract; only abandon_readback (unmap
        // only) is safe on Err. See GpuBuffers::abandon_readback's doc.
        if let Some(flag) = self.pending_readback.take() {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            match flag.lock().ok().and_then(|mut g| g.take()) {
                Some(Ok(())) => {
                    let _ = self.buffers.finish_readback(self.particle_count);
                }
                Some(Err(_)) => {
                    self.readback_error_count += 1;
                    self.buffers.abandon_readback();
                }
                None => {}
            }
        }
        self.particles =
            self.buffers
                .readback_blocking(&self.device, &self.queue, self.particle_count);
        self.rebuild_spatial_hash();
    }

    /// Like `sync_particles_blocking`, but only for the given particle index ranges --
    /// updates just `self.particles[range]` for each range, leaving the rest of the CPU
    /// mirror as whatever the last async/full sync delivered. For callers that only need
    /// a small, known subset of particles current every frame (e.g. a handful of live
    /// creatures inside a much larger terrain/water scene) instead of the whole buffer --
    /// see `GpuBuffers::readback_ranges_blocking`'s own doc for why this is cheaper than
    /// repeated full syncs, not just "less data" but batched into one CPU↔GPU round-trip.
    /// Ranges may overlap or be given in any order; each is written independently.
    pub fn sync_particle_ranges_blocking(&mut self, ranges: &[std::ops::Range<usize>]) {
        // Safe no-op once the device is lost -- see step_frame's identical guard.
        if self.is_device_lost() {
            return;
        }
        // Same real fix as sync_particles_blocking -- see that function's comment.
        if let Some(flag) = self.pending_readback.take() {
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            match flag.lock().ok().and_then(|mut g| g.take()) {
                Some(Ok(())) => {
                    let _ = self.buffers.finish_readback(self.particle_count);
                }
                Some(Err(_)) => {
                    self.readback_error_count += 1;
                    self.buffers.abandon_readback();
                }
                None => {}
            }
        }
        let results = self
            .buffers
            .readback_ranges_blocking(&self.device, &self.queue, ranges);
        for (range, data) in ranges.iter().zip(results) {
            self.particles[range.clone()].copy_from_slice(&data);
        }
        self.rebuild_spatial_hash();
    }
}

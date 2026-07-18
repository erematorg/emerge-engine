//! GPU -> CPU readback methods for `GpuBuffers` -- split out of `buffers.rs`
//! (was ~250 of its ~700 lines), matching that file's own doc comment split
//! between "Upload path" and "Download path". Construction and uploads stay
//! in `buffers.rs`; every blocking/async readback variant lives here.

use std::mem;

use super::GpuBuffers;
use crate::particle::Particle;

impl GpuBuffers {
    /// Begin an async GPU → CPU readback. Non-blocking — returns a shared flag set when done.
    /// Caller polls the flag each frame via `try_lock` + `take`. Staging buffer must be idle.
    pub fn begin_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> {
        use std::sync::{Arc, Mutex};
        let byte_count = (particle_count * mem::size_of::<Particle>()) as u64;

        // Submit copy GPU → staging (non-blocking GPU command).
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm_readback_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.particles, 0, &self.readback_staging, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));

        // Kick off async mapping — callback fires when copy is complete and buffer is ready.
        let flag: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let flag_cb = flag.clone();
        self.readback_staging
            .slice(..byte_count)
            .map_async(wgpu::MapMode::Read, move |r| {
                *flag_cb.lock().expect("emerge: GPU readback flag poisoned") = Some(r);
            });
        flag
    }

    /// Blocking GPU → CPU readback of specific particle index ranges only — e.g. a
    /// handful of live creatures scattered through a much larger terrain/water
    /// buffer, where reading the WHOLE buffer every frame (via `readback_blocking`)
    /// would stall on copying/mapping particles the caller doesn't even need this
    /// frame. All ranges are copied within a SINGLE encoder/submit/poll (batched,
    /// not one blocking round-trip per range — the per-call CPU↔GPU synchronization
    /// overhead, not just data volume, is what makes repeated small blocking
    /// readbacks expensive in practice). Returns one `Vec<Particle>` per input
    /// range, same order, each sized `range.len()`.
    ///
    /// Panics if the combined byte size of all ranges exceeds the readback staging
    /// buffer's capacity (sized for the full particle count at construction, so any
    /// subset always fits).
    pub fn readback_ranges_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ranges: &[std::ops::Range<usize>],
    ) -> Vec<Vec<Particle>> {
        let particle_size = mem::size_of::<Particle>() as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm_readback_ranges_blocking"),
        });
        let mut staging_offset = 0u64;
        let mut spans = Vec::with_capacity(ranges.len());
        for range in ranges {
            let byte_count = (range.len() as u64) * particle_size;
            encoder.copy_buffer_to_buffer(
                &self.particles,
                (range.start as u64) * particle_size,
                &self.readback_staging,
                staging_offset,
                byte_count,
            );
            spans.push((staging_offset, byte_count));
            staging_offset += byte_count;
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let total_bytes = staging_offset;
        let slice = self.readback_staging.slice(..total_bytes.max(1));
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let mapped = slice.get_mapped_range();
        let results = spans
            .iter()
            .map(|&(offset, len)| {
                let start = offset as usize;
                let end = start + len as usize;
                bytemuck::cast_slice::<u8, Particle>(&mapped[start..end]).to_vec()
            })
            .collect();
        drop(mapped);
        self.readback_staging.unmap();
        results
    }

    /// Blocking GPU → CPU readback. Submits copy, waits for GPU idle, maps, returns particles.
    /// Stalls the CPU until the GPU completes all in-flight work — only call from tests or
    /// parity-mode helpers, never from the render/game loop.
    pub fn readback_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: usize,
    ) -> Vec<Particle> {
        let byte_count = (particle_count * mem::size_of::<Particle>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm_readback_blocking"),
        });
        encoder.copy_buffer_to_buffer(&self.particles, 0, &self.readback_staging, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = self.readback_staging.slice(..byte_count);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let mapped = slice.get_mapped_range();
        let particles = bytemuck::cast_slice::<u8, Particle>(&mapped).to_vec();
        drop(mapped);
        self.readback_staging.unmap();
        particles
    }

    /// Blocking GPU → CPU readback of an arbitrary u32 storage buffer (e.g.
    /// `sorted_particle_ids`). Test/verification use only — uses a throwaway staging buffer,
    /// never call from the render/game loop.
    pub fn readback_u32_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<u32> {
        let byte_count = (count * mem::size_of::<u32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_u32_readback_staging"),
            size: byte_count,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm_u32_readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = staging.slice(..byte_count);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        values
    }

    /// Test/diagnostic readback of `count` f32 values from any storage buffer with COPY_SRC.
    /// Used to inspect the dense `grid` buffer directly (4 f32 per `Cell`: momentum.x,
    /// momentum.y, mass, _pad — same field order as the WGSL `Cell` struct in every shader)
    /// without exposing the crate-private `GpuCell` type outside this module.
    pub fn readback_f32_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<f32> {
        let byte_count = (count * mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_f32_readback_staging"),
            size: byte_count,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm_f32_readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_count);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let slice = staging.slice(..byte_count);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        values
    }

    /// Read mapped staging data into a Vec and unmap. Call only after the readback receiver fires.
    pub fn finish_readback(&self, particle_count: usize) -> Vec<Particle> {
        let byte_count = (particle_count * mem::size_of::<Particle>()) as u64;
        let slice = self.readback_staging.slice(..byte_count);
        let mapped = slice.get_mapped_range();
        let particles: Vec<Particle> = bytemuck::cast_slice::<u8, Particle>(&mapped).to_vec();
        drop(mapped);
        self.readback_staging.unmap();
        particles
    }

    /// Release a FAILED readback's mapping without extracting data (there's nothing
    /// valid to read on `Err`). Real bug this closes (found 2026-07-05, see project
    /// memory): `map_async`'s callback firing at all -- Ok OR Err -- means wgpu
    /// considers the buffer mapped; only `finish_readback`/this function's call to
    /// `unmap()` releases that state. The old code only ever unmapped on the Ok path,
    /// so a single `Err` (rare on fast hardware, far more likely on a slow/software
    /// backend where async completion timing differs) left the staging buffer
    /// permanently mapped -- silently disabling every future readback for the rest of
    /// the run, then panicking ("Buffer is already mapped") the next time anything
    /// tried to map it again (reproduced locally forcing a software WARP-style
    /// adapter; plausibly the same root cause behind emerge issue #10's
    /// STATUS_STACK_BUFFER_OVERRUN on CI, manifesting differently per-driver).
    pub fn abandon_readback(&self) {
        self.readback_staging.unmap();
    }
}

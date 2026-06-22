/// GPU buffer management for the MLS-MPM solver.
///
/// Persistent buffers in VRAM for the simulation lifetime:
///   - `particles`:          array<Particle>          — 112 bytes each, repr(C)
///   - `grid`:               array<Cell>              — 16 bytes each, repr(C)
///   - `materials`:          array<MaterialParams, N> — 96 bytes each, 16-byte aligned
///   - `step_params`:        GpuStepParams            — 32 bytes, uploaded once per substep
///   - `force_fields_params: GpuFieldsParams     — 784 bytes, uploaded when fields change
///
/// Upload path (CPU → GPU, via write_buffer):
///   particles:          initial spawn, then only when CPU plasticity corrects state
///   materials:          on spawn + on interactive param change
///   step_params:        every substep (sub_dt changes)
///   force_fields_params: every substep (entries may change between frames)
///   grid:               never uploaded — zeroed each substep by grid_clear compute pass
///
/// Download path (GPU → CPU):
///   particles:   only for plasticity readback (snow/sand) or diagnostics
///   grid:        never downloaded
use std::mem;

use super::step_params::{GpuFieldsParams, GpuImpulseParams, GpuStepParams};
use crate::materials::MaterialParams;
use crate::particle::Particle;

/// Cell layout matching `struct Cell` in every WGSL shader — 16 bytes.
/// Only used for buffer sizing; the GPU writes it via the shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCell {
    momentum: [f32; 2],
    mass: f32,
    _pad: f32,
}

const _: () = assert!(mem::size_of::<GpuCell>() == 16);

/// All persistent GPU buffers for one GpuSimulation instance.
pub struct GpuBuffers {
    /// Particle data — STORAGE | COPY_DST | COPY_SRC.
    pub particles: wgpu::Buffer,
    /// Grid cells, zeroed each substep by grid_clear pass — STORAGE
    pub grid: wgpu::Buffer,
    /// One MaterialParams per registered material slot — UNIFORM | COPY_DST
    pub materials: wgpu::Buffer,
    /// Per-substep constants pool — one buffer per max_substeps slot, each 32 bytes.
    /// Substep i reads `step_params_pool[i]` so all substeps can be encoded in one command buffer.
    pub step_params_pool: Vec<wgpu::Buffer>,
    /// Non-uniform force-field entries for the force_fields pass — UNIFORM | COPY_DST
    pub force_fields_params: wgpu::Buffer,
    /// Impulse descriptors for the apply_impulses pass — UNIFORM | COPY_DST
    pub impulse_params: wgpu::Buffer,
    /// Sorted particle index permutation — STORAGE.
    /// Written once per frame by the particle_sort count→scan→scatter pipeline (block-level
    /// counting sort by spatial position). Read by p2g and particles_update for cache-coherent
    /// particle access (Gao et al. 2018, "GPU Optimization of Material Point Methods").
    pub sorted_particle_ids: wgpu::Buffer,
    /// Per-block atomic counters for the particle_sort pipeline — NUM_BLOCKS (256) × u32.
    /// Cleared, filled (histogram), scanned (exclusive prefix sum), then reused as the atomic
    /// scatter cursor — all in one frame's particle_sort pass sequence.
    pub block_counts: wgpu::Buffer,
    /// Persistent readback staging buffer — pre-allocated to avoid per-frame alloc/dealloc.
    /// COPY_DST | MAP_READ. Same size as `particles`.
    pub readback_staging: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        particle_count: usize,
        grid_res: usize,
        max_materials: usize,
        max_substeps: usize,
    ) -> Self {
        let particle_bytes = (particle_count * mem::size_of::<Particle>()) as u64;
        let grid_bytes = (grid_res * grid_res * mem::size_of::<GpuCell>()) as u64;
        let material_bytes = (max_materials * mem::size_of::<MaterialParams>()) as u64;
        let step_bytes = mem::size_of::<GpuStepParams>() as u64;

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particles"),
            size: particle_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_grid"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let materials = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_materials"),
            size: material_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Allocate max_substeps + 1 slots: 0..max_substeps for physics substeps,
        // slot max_substeps is a dedicated particle_sort slot so it never aliases substep 0.
        let step_params_pool: Vec<wgpu::Buffer> = (0..max_substeps + 1)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("mpm_step_params_{i}")),
                    size: step_bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let force_fields_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_force_fields_params"),
            size: mem::size_of::<GpuFieldsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let impulse_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_impulse_params"),
            size: mem::size_of::<GpuImpulseParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sorted_particle_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_sorted_particle_ids"),
            size: (particle_count * mem::size_of::<u32>()) as u64,
            // COPY_SRC: needed for test-only readback (gpu_particle_sort_is_valid_permutation).
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // NUM_BLOCKS (256) must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM² exactly.
        let block_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_block_counts"),
            size: (256 * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let readback_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm_particle_staging"),
            size: particle_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            particles,
            grid,
            materials,
            step_params_pool,
            force_fields_params,
            impulse_params,
            sorted_particle_ids,
            block_counts,
            readback_staging,
        }
    }

    pub fn upload_particles(&self, queue: &wgpu::Queue, particles: &[Particle]) {
        queue.write_buffer(&self.particles, 0, Particle::slice_as_bytes(particles));
    }

    pub fn upload_materials(&self, queue: &wgpu::Queue, params: &[MaterialParams]) {
        queue.write_buffer(&self.materials, 0, bytemuck::cast_slice(params));
    }

    /// Upload step params into pool slot `index`. Panics if index >= pool size.
    pub fn upload_step_params_at(&self, queue: &wgpu::Queue, index: usize, params: &GpuStepParams) {
        queue.write_buffer(&self.step_params_pool[index], 0, bytemuck::bytes_of(params));
    }

    pub fn upload_force_fields_params(&self, queue: &wgpu::Queue, params: &GpuFieldsParams) {
        queue.write_buffer(&self.force_fields_params, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_impulse_params(&self, queue: &wgpu::Queue, params: &GpuImpulseParams) {
        queue.write_buffer(&self.impulse_params, 0, bytemuck::bytes_of(params));
    }

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
}

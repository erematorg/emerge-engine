//! GPU buffers for rendering a CPU `Simulation` through the grid-volume and
//! surface-reconstruction paths.
//!
//! Those paths read GPU-resident buffers that `GpuSimulation` already keeps
//! (`grid_buffer`, `material_mass_buffer`, `particle_buffer`); the CPU solver
//! keeps its state in host memory. `CpuRenderBridge` owns the three buffers
//! and rebuilds them from the CPU state each frame a GPU-read mode is shown,
//! in the layouts those paths expect, so a CPU demo wires them with one field
//! instead of its own copy of the upload.

use glam::IVec2;

use super::gpu_types::{GridVolumeSource, SurfaceReconstructionSource};
use crate::grid::Grid;
use crate::particle::{Particle, Particles};
use crate::systems::gpu::MAX_RENDER_MATERIAL_SLOTS;

/// Floats per grid cell in the grid-volume layout: `[Σ m·T, 0, mass, 0]`,
/// the same as `GpuSimulation::grid_buffer()`.
const GRID_CHANNELS: usize = 4;
const SLOTS: usize = MAX_RENDER_MATERIAL_SLOTS as usize;

/// Owns and refreshes the buffers the GPU render paths read, for a CPU
/// `Simulation`.
///
/// Grid layout is row-major, `y * grid_res + x`, 4 floats per cell: channel
/// 2 is the grid's own P2G mass (`Grid::mass_at`), channel 0 the
/// mass-weighted temperature `Σ m·T` when `carries_temperature` is on (0
/// otherwise). Per-material mass and the temperature are scattered to each
/// particle's nearest node, not through P2G's quadratic kernel: enough to
/// pick and blend material colours and to place hot cells, not a physics
/// quantity.
pub struct CpuRenderBridge {
    grid_res: usize,
    carries_temperature: bool,
    grid_buf: wgpu::Buffer,
    material_mass_buf: wgpu::Buffer,
    particle_buf: wgpu::Buffer,
    particle_capacity: usize,
    particle_count: usize,
    // Reused every upload: a fresh allocation per frame was measurable at
    // demo particle counts.
    dense: Vec<f32>,
    material_mass: Vec<f32>,
    particles: Vec<Particle>,
}

impl CpuRenderBridge {
    /// Buffers for a `grid_res`-cell simulation. The particle buffer starts
    /// empty and grows on the first `upload_particles`.
    pub fn new(device: &wgpu::Device, grid_res: usize) -> Self {
        let cells = grid_res * grid_res;
        Self {
            grid_res,
            carries_temperature: false,
            grid_buf: storage_buffer(
                device,
                "cpu_render_bridge_grid",
                cells * GRID_CHANNELS * std::mem::size_of::<f32>(),
            ),
            material_mass_buf: storage_buffer(
                device,
                "cpu_render_bridge_material_mass",
                cells * SLOTS * std::mem::size_of::<f32>(),
            ),
            particle_buf: storage_buffer(device, "cpu_render_bridge_particles", 0),
            particle_capacity: 0,
            particle_count: 0,
            dense: Vec::new(),
            material_mass: Vec::new(),
            particles: Vec::new(),
        }
    }

    /// Also scatter the mass-weighted temperature into grid channel 0, which
    /// the grid-volume path reads for thermal emission. Off by default.
    pub fn with_temperature(mut self, carries_temperature: bool) -> Self {
        self.carries_temperature = carries_temperature;
        self
    }

    /// Rebuilds the grid and per-material mass buffers from `grid` and
    /// `particles` and uploads them, for `Renderer::render_grid_volume`.
    pub fn upload_grid(&mut self, queue: &wgpu::Queue, particles: &Particles, grid: &Grid) {
        fill_grid_channels(
            particles,
            grid,
            self.grid_res,
            self.carries_temperature,
            &mut self.dense,
        );
        queue.write_buffer(&self.grid_buf, 0, bytemuck::cast_slice(&self.dense));
        fill_material_mass(particles, self.grid_res, &mut self.material_mass);
        queue.write_buffer(
            &self.material_mass_buf,
            0,
            bytemuck::cast_slice(&self.material_mass),
        );
    }

    /// The source for `Renderer::render_grid_volume`, reading what the last
    /// `upload_grid` wrote, with per-material colouring on.
    pub fn grid_volume_source(&self) -> GridVolumeSource<'_> {
        GridVolumeSource {
            grid: &self.grid_buf,
            material_mass: &self.material_mass_buf,
            material_mass_enabled: true,
            grid_res: self.grid_res as u32,
        }
    }

    /// Uploads `particles` as the `Particle` array the surface path reads
    /// (`Particle` is `repr(C)`/`Pod`), growing the buffer when the particle
    /// count has grown.
    pub fn upload_particles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &Particles,
    ) {
        self.particles.clear();
        self.particles.extend(particles.iter());
        if self.particles.len() > self.particle_capacity {
            self.particle_capacity = self.particles.len();
            self.particle_buf = storage_buffer(
                device,
                "cpu_render_bridge_particles",
                self.particle_capacity * std::mem::size_of::<Particle>(),
            );
        }
        self.particle_count = self.particles.len();
        queue.write_buffer(&self.particle_buf, 0, bytemuck::cast_slice(&self.particles));
    }

    /// The source for `Renderer::render_surface_reconstruction`, reading
    /// what the last `upload_particles` wrote. `material_slot` colours the
    /// surface when `material_mass_enabled` is off; `dt` is the solver's
    /// `Simulation::mean_substep_dt` (see `SurfaceReconstructionSource::dt`).
    pub fn surface_source(
        &self,
        material_slot: u32,
        material_mass_enabled: bool,
        dt: f32,
    ) -> SurfaceReconstructionSource<'_> {
        SurfaceReconstructionSource {
            particle_buf: &self.particle_buf,
            particle_count: self.particle_count,
            grid_res: self.grid_res as u32,
            material_slot,
            material_mass_enabled,
            dt,
        }
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // A zero-sized storage binding fails validation; 4 bytes keeps an
        // empty particle buffer bindable.
        size: size.max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A particle's nearest grid node, clamped into the domain.
fn nearest_cell(x: glam::Vec2, grid_res: usize) -> usize {
    let max = grid_res as i32 - 1;
    let cx = (x.x.round() as i32).clamp(0, max) as usize;
    let cy = (x.y.round() as i32).clamp(0, max) as usize;
    cy * grid_res + cx
}

fn fill_grid_channels(
    particles: &Particles,
    grid: &Grid,
    grid_res: usize,
    carries_temperature: bool,
    dense: &mut Vec<f32>,
) {
    dense.clear();
    dense.resize(grid_res * grid_res * GRID_CHANNELS, 0.0);
    for y in 0..grid_res {
        for x in 0..grid_res {
            let idx = y * grid_res + x;
            dense[idx * GRID_CHANNELS + 2] = grid.mass_at(IVec2::new(x as i32, y as i32));
        }
    }
    if carries_temperature {
        for i in 0..particles.len() {
            let idx = nearest_cell(particles.x[i], grid_res);
            dense[idx * GRID_CHANNELS] += particles.mass[i] * particles.temperature[i];
        }
    }
}

fn fill_material_mass(particles: &Particles, grid_res: usize, material_mass: &mut Vec<f32>) {
    material_mass.clear();
    material_mass.resize(grid_res * grid_res * SLOTS, 0.0);
    for i in 0..particles.len() {
        let idx = nearest_cell(particles.x[i], grid_res);
        let slot = particles.material_id[i] as usize % SLOTS;
        material_mass[idx * SLOTS + slot] += particles.mass[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
    use glam::Vec2;

    fn stepped_scene() -> Simulation {
        let config = SimConfig::standard(24, 0.01, Vec2::new(0.0, -0.3));
        let mut sim = Simulation::new(
            config,
            SpawnRegion::for_sim(&config)
                .at(Vec2::splat(12.0))
                .disk(4.0)
                .spacing(0.5),
        )
        .with_default_material(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        for (i, t) in sim.particles_mut().temperature.iter_mut().enumerate() {
            *t = 280.0 + (i % 7) as f32;
        }
        sim.step();
        // Set after stepping: only one material is registered, and the bridge
        // reads ids without consulting the registry.
        sim.particles_mut().material_id[0] = 3;
        sim
    }

    /// Channel 2 is the grid's own mass, channel 0 carries exactly Σ m·T when
    /// asked and stays 0 otherwise, and the per-material mass accounts for
    /// every particle's mass in its own slot.
    #[test]
    fn grid_channels_and_material_mass_match_the_cpu_state() {
        let sim = stepped_scene();
        let res = sim.config().grid_res;
        let particles = sim.particles();

        let mut dense = Vec::new();
        fill_grid_channels(particles, sim.grid(), res, true, &mut dense);
        for y in 0..res {
            for x in 0..res {
                let idx = y * res + x;
                assert_eq!(
                    dense[idx * GRID_CHANNELS + 2],
                    sim.grid().mass_at(IVec2::new(x as i32, y as i32))
                );
            }
        }
        let heat: f64 = (0..particles.len())
            .map(|i| f64::from(particles.mass[i] * particles.temperature[i]))
            .sum();
        let scattered: f64 = dense.chunks(GRID_CHANNELS).map(|c| f64::from(c[0])).sum();
        assert!(
            (scattered - heat).abs() < 1e-3 * heat,
            "{scattered} vs {heat}"
        );

        fill_grid_channels(particles, sim.grid(), res, false, &mut dense);
        assert!(dense.chunks(GRID_CHANNELS).all(|c| c[0] == 0.0));

        let mut material_mass = Vec::new();
        fill_material_mass(particles, res, &mut material_mass);
        for slot in 0..SLOTS {
            let expected: f32 = (0..particles.len())
                .filter(|&i| particles.material_id[i] as usize % SLOTS == slot)
                .map(|i| particles.mass[i])
                .sum();
            let got: f32 = material_mass.chunks(SLOTS).map(|c| c[slot]).sum();
            assert!(
                (got - expected).abs() <= 1e-4 * expected.max(1.0),
                "slot {slot}"
            );
        }
    }
}

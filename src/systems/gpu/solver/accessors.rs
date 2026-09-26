//! Trivial public accessors/setters on `GpuSimulation`: GPU handle sharing
//! (device/queue/buffers, for LP's renderer), CPU-mirror access, material
//! registry management, and frame/timing state.
//!
//! Split out of `gpu/solver/mod.rs` -- distinct from construction (`new`/
//! `with_device`, which stay in mod.rs alongside the `GpuSimulation` struct
//! definition itself), spatial queries (`queries.rs`), particle mutation
//! (`particles.rs`), and the dispatch loop (`step.rs`). Everything here is a
//! plain field read/write or a small registry delegation -- no wgpu command
//! encoding.

use std::sync::Arc;

use crate::materials::registry::MaterialRegistry;
use crate::particle::Particle;
use crate::solver::config::SimConfig;

use super::super::step_params::MAX_SLEEP_WAKE_TAGS;
use super::GpuSimulation;

impl GpuSimulation {
    /// Returns (cfl_scan_ns, encode_ns, wait_ns, readback_ns, total_ns) from the last
    /// `step_frame()` call. `encode_ns` is pure CPU-side command-building time (bind
    /// group already cached, just recording dispatches); `wait_ns` (renamed from the
    /// old always-zero `submit_ns` -- multi-chunk frames now really do block between
    /// chunks) is time spent in `device.poll(wait_indefinitely())` between substep
    /// batches, i.e. real GPU execution time for scenes needing >64 substeps/frame.
    pub fn last_cpu_timings_ns(&self) -> (f32, f32, f32, f32, f32) {
        self.last_cpu_timings
    }

    /// Force every particle with `user_tag == tag` asleep, regardless of velocity,
    /// applied at the start of the next `step_frame()`. P2G still scatters for them
    /// (see `gpu_sleep_wake_phase1` memory note -- sleeping particles must keep
    /// providing structural support); only their own gather/integration/force-field
    /// work is skipped.
    ///
    /// Minimal hook, not a chunk system: this just lets a caller (e.g. LP's future
    /// chunk loader, once it exists) force-sleep a tagged group by distance instead
    /// of waiting for velocity to drop. Mirrors the CPU `Simulation::sleep_tag` API.
    pub fn sleep_tag(&mut self, tag: u32) {
        if self.pending_sleep_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_sleep_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU sleep-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) -- tag dropped"
            );
        }
    }

    /// Force every particle with `user_tag == tag` awake, regardless of grid activity.
    /// Mirrors the CPU `Simulation::wake_tag` API. See `sleep_tag` doc comment.
    pub fn wake_tag(&mut self, tag: u32) {
        if self.pending_wake_tags.len() < MAX_SLEEP_WAKE_TAGS {
            self.pending_wake_tags.push(tag);
        } else {
            eprintln!(
                "emerge: GPU wake-tag queue full ({MAX_SLEEP_WAKE_TAGS}/frame max) -- tag dropped"
            );
        }
    }

    /// Mark CPU particles as layout-changed (positions/materials) -- triggers sort + upload.
    pub fn mark_particles_dirty(&mut self) {
        self.layout_dirty = true;
    }

    /// Upload revised material params (e.g., if interactive sliders change them).
    pub fn upload_materials(&self) {
        self.buffers
            .upload_materials(&self.queue, &self.registry.all_params());
    }

    pub fn registry(&self) -> &MaterialRegistry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut MaterialRegistry {
        &mut self.registry
    }

    /// The wgpu Device -- share with the LP render system to read the particle buffer directly.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// The wgpu Queue -- share with the LP render system for command submission.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// The GPU particle storage buffer -- bind this in LP's custom render shader.
    /// Layout: `array<Particle>`, each Particle is 112 bytes, repr(C).
    /// Stays in VRAM between frames; read-only from the render side.
    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.particles
    }

    /// Read-only access to the CPU particle mirror (one frame behind GPU when strided).
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    /// Mutable access to the CPU particle mirror.
    ///
    /// **CFL WARNING:** velocity changes bypass the solver's CFL clamp.
    /// For gameplay impulses use `apply_impulse` / `apply_radial_impulse` instead.
    /// After modifying, call `mark_particles_dirty()` so the GPU sees the changes.
    /// Puts every particle into hydrostatic equilibrium under the current
    /// gravity, so a body spawned "at rest" genuinely starts at rest.
    ///
    /// The GPU mirror of `Simulation::settle_hydrostatic`; both call the
    /// same `hydrostatic_state`, so equilibrium means the same thing on
    /// either path. See that method for why a pool spawned at uniform
    /// density is not at rest.
    ///
    /// Call it after spawning and before the first step. Marks the particle
    /// buffer dirty so the corrected state reaches the GPU.
    pub fn settle_hydrostatic(&mut self) {
        let gravity_magnitude = self.config.gravity.length();
        if gravity_magnitude <= 0.0 {
            return;
        }
        let mut surface: std::collections::HashMap<(u32, i32), f32> =
            std::collections::HashMap::new();
        for p in &self.particles {
            let top = surface
                .entry((p.material_id, p.x.x.floor() as i32))
                .or_insert(f32::NEG_INFINITY);
            *top = top.max(p.x.y);
        }
        for i in 0..self.particles.len() {
            let p = self.particles[i];
            let Some(&top) = surface.get(&(p.material_id, p.x.x.floor() as i32)) else {
                continue;
            };
            let Some(state) = crate::spacetime::solver::hydrostatic_state(
                self.registry.get(p.material_id),
                gravity_magnitude,
                top - p.x.y,
                p.initial_volume,
                p.mass,
            ) else {
                continue;
            };
            let p = &mut self.particles[i];
            p.deformation_gradient = state.deformation_gradient;
            p.volume = state.volume;
            p.density = state.density;
        }
        self.mark_particles_dirty();
    }

    pub fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn set_gravity(&mut self, gravity: glam::Vec2) {
        self.config.gravity = gravity;
    }

    /// Replace the default material and re-upload the materials buffer.
    pub fn set_default_material(&mut self, material: Box<dyn crate::materials::MaterialModel>) {
        self.registry.set_default(material);
        self.upload_materials();
    }

    pub fn gravity(&self) -> glam::Vec2 {
        self.config.gravity
    }

    /// The live GPU grid buffer (STORAGE | COPY_SRC).
    /// Layout: `array<Cell>` where Cell = { momentum: vec2, mass: f32, _pad: f32 } (16 bytes).
    /// Consumers (e.g. LP's metaball renderer) can bind this read-only in their own compute pass.
    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.grid
    }

    /// The live GPU per-material mass buffer for `ColorMode::GridVolume`'s
    /// material-aware coloring. Layout: `array<f32>`, `grid_res² x
    /// MAX_RENDER_MATERIAL_SLOTS` entries, cell-major then slot-minor (`[cell_idx *
    /// MAX_RENDER_MATERIAL_SLOTS + material_id % MAX_RENDER_MATERIAL_SLOTS]`). Only
    /// meaningful after `attach_grid_material_render_gpu` -- before that, this reads
    /// the (tiny, placeholder-sized) never-written buffer.
    pub fn material_mass_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.material_mass
    }

    /// Register a material, auto-assigning the next available ID.
    ///
    /// Mirrors `Simulation::register_material` -- use this instead of `set_material`
    /// when you don't want to track IDs manually. Returns a typed handle.
    ///
    /// LP pattern: call at world-init time to build a material palette, then
    /// use `handle.id()` in `SpawnRegion::for_sim(...).material(handle.id())`.
    pub fn register_material(
        &mut self,
        material: Box<dyn crate::materials::MaterialModel>,
    ) -> crate::solver::handle::MaterialHandle {
        let id = self.registry.next_id();
        self.registry.insert(id, material);
        self.upload_materials();
        crate::solver::handle::MaterialHandle(id)
    }

    /// Register or replace a material by explicit ID and re-upload the materials buffer.
    pub fn set_material(
        &mut self,
        material_id: u32,
        material: Box<dyn crate::materials::MaterialModel>,
    ) {
        self.registry.insert(material_id, material);
        self.upload_materials();
    }

    /// The sub-dt used in the last substep of the most recent `step_frame` call.
    pub fn effective_dt(&self) -> f32 {
        self.last_sub_dt
    }

    /// Number of substeps run during the most recent `step_frame` call.
    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    /// Total frames stepped since creation.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }
}

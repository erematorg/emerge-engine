/// GPU compute backend for the MLS-MPM solver.
///
/// Architecture: wgpu compute shaders, 4 passes per substep:
///   grid_clear → p2g (scatter) → grid_update → g2p (gather)
///
/// Plasticity: Snow SVD and Drucker-Prager return-mapping both run on GPU (g2p.wgsl).
/// No CPU roundtrip needed for plasticity. Fluid, NeoHookean, Corotated also GPU.
///
/// Data flow each substep:
///   CPU uploads GpuStepParams (dt, gravity, etc.) once per substep
///   GPU runs 4 compute passes on particle + grid buffers in VRAM
///   CPU downloads particles once per frame only if plasticity is needed
///   LP renders: reads the particle buffer directly via shared wgpu Device
///
/// Physics constants: KERNEL_D_INVERSE=4.0 is a fixed B-spline constant; other params come from SimConfig.
/// GPU-side constants (MAX_MATERIALS, workgroup sizes) are named here
/// and must match their WGSL counterparts exactly.
///
/// Enabled via `features = ["gpu"]`. Core library compiles without this feature.
#[cfg(feature = "gpu")]
pub mod pipeline;

#[cfg(feature = "gpu")]
pub mod buffers;

// WGSL shader sources — embedded at compile time.
#[cfg(feature = "gpu")]
pub mod shaders {
    pub const PARTICLE_SORT: &str = include_str!("shaders/particle_sort.wgsl");
    pub const GRID_CLEAR: &str = include_str!("shaders/grid_clear.wgsl");
    pub const P2G: &str = include_str!("shaders/p2g.wgsl");
    pub const GRID_UPDATE: &str = include_str!("shaders/grid_update.wgsl");
    pub const G2P: &str = include_str!("shaders/g2p.wgsl");
    pub const PARTICLES_UPDATE: &str = include_str!("shaders/particles_update.wgsl");
    pub const FORCE_FIELDS: &str = include_str!("shaders/force_fields.wgsl");
    pub const APPLY_IMPULSES: &str = include_str!("shaders/apply_impulses.wgsl");
}

#[cfg(feature = "gpu")]
pub use solver::GpuSimulation;

#[cfg(feature = "gpu")]
pub use step_params::{
    GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuStepParams,
    MAX_FORCE_FIELDS, MAX_GPU_IMPULSES, field_type,
};

#[cfg(feature = "gpu")]
mod step_params {
    use crate::solver::config::SimConfig;

    /// Re-export so GPU code reads the same limit as the registry.
    /// Injected into WGSL shaders at pipeline creation — change only in `materials/registry.rs`.
    pub use crate::materials::registry::MAX_MATERIAL_SLOTS as MAX_MATERIALS;

    /// Per-substep solver constants uploaded to the GPU uniform buffer before each substep.
    ///
    /// 32 bytes, 16-byte aligned — satisfies WGSL uniform binding requirements.
    /// Fields mirror `struct StepParams` in every WGSL shader exactly (same offsets, same types).
    ///
    /// All values come from `SimConfig` or are computed from it — no hardcoded physics here.
    /// Uniform data uploaded once per GPU substep.
    ///
    /// Layout (32 bytes, 16-byte aligned — WGSL uniform binding requirement):
    ///   offset  0: grid_res       u32
    ///   offset  4: particle_count u32
    ///   offset  8: dt             f32
    ///   offset 12: kernel_d_inverse      f32  (always 4.0 — quadratic B-spline)
    ///   offset 16: gravity        vec2<f32>  (8 bytes; 8-byte aligned in WGSL ✓)
    ///   offset 24: boundary_thickness u32
    ///   offset 28: vel_limit      f32
    ///                             = 32 bytes, 16-byte aligned ✓
    ///
    /// `gravity: Vec2` replaces the old `gravity: f32` + `_pad1: u32` pair —
    /// same byte count, no layout change for other fields.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuStepParams {
        pub grid_res: u32,
        pub particle_count: u32,
        pub dt: f32,
        pub kernel_d_inverse: f32,
        pub gravity: glam::Vec2, // SimConfig::gravity — supports angled/planetary gravity
        pub boundary_thickness: u32,
        pub vel_limit: f32, // grid_cell_size / sub_dt
    }

    impl GpuStepParams {
        pub fn new(config: &SimConfig, sub_dt: f32, particle_count: usize) -> Self {
            Self {
                grid_res: config.grid_res as u32,
                particle_count: particle_count as u32,
                dt: sub_dt,
                kernel_d_inverse: crate::solver::config::KERNEL_D_INVERSE,
                gravity: config.gravity,
                boundary_thickness: config.boundary_thickness as u32,
                vel_limit: config.grid_cell_size / sub_dt,
            }
        }
    }

    const _: () = assert!(core::mem::size_of::<GpuStepParams>() == 32);

    /// Maximum number of active GPU force-field entries per frame.
    /// Must match `MAX_FORCE_FIELDS` in `force_fields.wgsl`.
    pub const MAX_FORCE_FIELDS: usize = 16;

    /// Field-type discriminants — match `FIELD_*` constants in `force_fields.wgsl`.
    pub mod field_type {
        pub const DISABLED: u32 = 0;
        pub const GRAVITY_WELL: u32 = 1;
        pub const COULOMB: u32 = 2;
        pub const AABB_CONFINEMENT: u32 = 3;
        pub const RADIAL_CONFINEMENT: u32 = 4;
        pub const UNIFORM_ELECTRIC: u32 = 5;
        pub const BUOYANCY: u32 = 6;
    }

    /// One GPU force-field entry — 48 bytes, 16-byte aligned.
    /// Matches `struct FieldEntry` in `force_fields.wgsl` exactly (size-asserted).
    /// Use the named constructors instead of filling `params` manually.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuFieldEntry {
        pub field_type: u32,
        pub material_mask: u32,
        pub _pad: [u32; 2],
        pub params: [f32; 8],
    }

    const _: () = assert!(core::mem::size_of::<GpuFieldEntry>() == 48);

    impl GpuFieldEntry {
        /// material_mask value for a field that affects all materials.
        pub const ALL_MATERIALS: u32 = 0xFFFF_FFFF;

        /// Plummer-softened point-mass gravity: a = −G·M·r / (r²+ε²)^(3/2).
        ///
        /// - `gm`: gravitational_constant × source_mass (positive = attractive)
        /// - `softening_sq`: Plummer ε² (prevents singularity at r=0)
        /// - `cutoff`: hard cutoff distance (0.0 = no cutoff)
        /// - `switch_on`: force-switch onset (< cutoff; force tapers from `switch_on` to `cutoff`)
        pub fn gravity_well(
            pos: glam::Vec2,
            gm: f32,
            softening_sq: f32,
            cutoff: f32,
            switch_on: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = pos.x;
            p[1] = pos.y;
            p[2] = gm;
            p[3] = softening_sq;
            p[6] = cutoff;
            p[7] = switch_on;
            Self {
                field_type: field_type::GRAVITY_WELL,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Plummer-softened Coulomb interaction for one (source, material) pair.
        ///
        /// - `charge_factor`: k × q_source × q_particle (signed; positive = repulsion)
        /// - `softening_sq`: Plummer ε²
        /// - `material_id`: which material's particles are affected (bitmask = 1 << id)
        /// - `cutoff` / `switch_on`: same as `gravity_well`
        pub fn coulomb(
            pos: glam::Vec2,
            charge_factor: f32,
            softening_sq: f32,
            material_id: u32,
            cutoff: f32,
            switch_on: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = pos.x;
            p[1] = pos.y;
            p[2] = charge_factor;
            p[3] = softening_sq;
            p[6] = cutoff;
            p[7] = switch_on;
            Self {
                field_type: field_type::COULOMB,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Soft repulsive walls of an axis-aligned bounding box.
        ///
        /// Particles that penetrate within `thickness` cells of any wall get a
        /// restoring acceleration proportional to penetration depth × `stiffness`.
        pub fn aabb_confinement(
            min: glam::Vec2,
            max: glam::Vec2,
            stiffness: f32,
            thickness: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = min.x;
            p[1] = min.y;
            p[2] = max.x;
            p[3] = max.y;
            p[4] = stiffness;
            p[5] = thickness;
            Self {
                field_type: field_type::AABB_CONFINEMENT,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Soft inward repulsion outside a radial shell.
        ///
        /// Particles beyond `radius − thickness` receive an inward acceleration
        /// proportional to excess penetration × `stiffness`.
        pub fn radial_confinement(
            center: glam::Vec2,
            radius: f32,
            stiffness: f32,
            thickness: f32,
        ) -> Self {
            let mut p = [0f32; 8];
            p[0] = center.x;
            p[1] = center.y;
            p[2] = radius;
            p[3] = stiffness;
            p[4] = thickness;
            Self {
                field_type: field_type::RADIAL_CONFINEMENT,
                material_mask: Self::ALL_MATERIALS,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Spatially-constant electric field: a = q · E / m.
        ///
        /// - `field`: E-field vector (simulation units — force per unit charge)
        /// - `charge`: per-particle charge for `material_id` (same units as the Coulomb constant)
        /// - `material_id`: only particles of this material are affected
        pub fn uniform_electric(field: glam::Vec2, charge: f32, material_id: u32) -> Self {
            let mut p = [0f32; 8];
            p[0] = field.x;
            p[1] = field.y;
            p[2] = charge;
            Self {
                field_type: field_type::UNIFORM_ELECTRIC,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }

        /// Archimedes buoyancy for particles of `material_id` floating in a denser fluid.
        ///
        /// - `gravity`: must match `SimConfig::gravity` (solver gravity, including sign)
        /// - `fluid_density_grid`: surrounding fluid's rest_density in grid units
        ///   (`ρ_SI · dx_m² / dt_s²` — same value as `NewtonianFluidMaterial::rest_density`)
        /// - `material_id`: only particles of this material receive the buoyancy force
        ///
        /// Uses particle rest density (`mass / initial_volume`) not instantaneous density,
        /// preventing the expansion-buoyancy runaway where expanded fluid appears falsely light.
        /// Applies `Δv = −gravity · (fluid_density / ρ₀_particle − 1) · dt` each substep.
        pub fn buoyancy(gravity: glam::Vec2, fluid_density_grid: f32, material_id: u32) -> Self {
            let mut p = [0f32; 8];
            p[0] = gravity.x;
            p[1] = gravity.y;
            p[2] = fluid_density_grid;
            p[3] = 1.0e-4; // min_density floor — mirrors BuoyancyField::new default
            Self {
                field_type: field_type::BUOYANCY,
                material_mask: 1 << material_id,
                _pad: [0; 2],
                params: p,
            }
        }
    }

    /// Uniform buffer containing all active GPU force-field entries — 784 bytes.
    /// Matches `struct FieldsParams` in `force_fields.wgsl` exactly (size-asserted).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuFieldsParams {
        pub count: u32,
        pub _pad: [u32; 3],
        pub entries: [GpuFieldEntry; MAX_FORCE_FIELDS],
    }

    const _: () = assert!(core::mem::size_of::<GpuFieldsParams>() == 784);

    /// Max impulses per frame submitted via `apply_impulse` / `apply_radial_impulse`.
    /// Must match `array<ImpulseEntry, 16>` in `apply_impulses.wgsl`.
    pub const MAX_GPU_IMPULSES: usize = 16;

    /// One impulse descriptor — 32 bytes, matches `struct ImpulseEntry` in WGSL.
    ///
    /// mode 0 = radial: `v += normalize(p - center) * strength * falloff`
    /// mode 1 = directional: `v += force * falloff`
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuImpulseEntry {
        pub center: [f32; 2], // grid-space origin
        pub radius: f32,
        pub strength: f32,   // radial only (signed)
        pub force: [f32; 2], // directional only
        pub mode: u32,       // 0 = radial, 1 = directional
        pub _pad: u32,
    }

    const _: () = assert!(core::mem::size_of::<GpuImpulseEntry>() == 32);

    /// Uniform data for the apply_impulses compute pass — 528 bytes.
    /// Matches `struct ImpulseParams` in `apply_impulses.wgsl`.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct GpuImpulseParams {
        pub count: u32,
        pub vel_limit: f32,
        pub particle_count: u32,
        pub _pad: u32,
        pub entries: [GpuImpulseEntry; MAX_GPU_IMPULSES],
    }

    const _: () = assert!(core::mem::size_of::<GpuImpulseParams>() == 528);
}

#[cfg(feature = "gpu")]
mod solver {
    use std::sync::Arc;

    use crate::materials::registry::MaterialRegistry;
    use crate::solver::config::{SimConfig, SpawnRegion};
    use crate::solver::density::estimate_particle_volumes;
    use crate::solver::{LcgRng, choose_substep_dt, initialize_particles};
    use crate::{
        grid::Grid,
        particle::{Particle, Particles},
    };

    use super::buffers::GpuBuffers;
    use super::pipeline::SimPipelines;
    use super::step_params::{
        GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuStepParams,
        MAX_FORCE_FIELDS, MAX_GPU_IMPULSES, MAX_MATERIALS,
    };

    /// Workgroup sizes — must match `@workgroup_size(...)` in the WGSL shaders.
    const WG_GRID: u32 = 8; // grid_clear and grid_update: 8×8 2D workgroups
    const WG_PARTICLES: u32 = 64; // p2g and g2p: 64-wide 1D workgroups

    /// Shared between the wgpu map_async callback (any thread) and step_frame's poll.
    type ReadbackResult =
        std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

    /// GPU-backed MLS-MPM solver.
    ///
    /// Pass sequence:
    ///   Once per frame: particle_sort (identity permutation → sorted_particle_ids)
    ///   Per substep:    grid_clear → p2g → grid_update → g2p → particles_update → force_fields
    ///
    /// Particles live in VRAM between frames; the CPU only touches them at spawn and for
    /// plasticity readback (currently: none — all plasticity runs in particles_update.wgsl).
    pub struct GpuSimulation {
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        buffers: GpuBuffers,
        pipelines: SimPipelines,
        config: SimConfig,
        registry: MaterialRegistry,
        /// CPU-side particle mirror. One frame behind the GPU when readback is strided.
        /// Access via `particles()` / `particles_mut()`. Do not replace the Vec directly.
        particles: Vec<Particle>,
        particle_count: usize,
        last_sub_dt: f32,
        last_substeps: usize,
        frame_index: u64,
        /// GPU force-field entries — uploaded to the force_fields_params uniform each substep.
        force_field_entries: Vec<GpuFieldEntry>,
        /// Frame counter used to stride CPU readbacks when all materials are GPU-resident.
        readback_frame: usize,
        /// Download CPU particle state every N step_frame calls when no CPU plasticity is needed.
        /// 1 = every frame (default, always accurate). 2+ = skip frames, reducing GPU stall cost.
        /// One-frame lag on sprite positions is invisible at 60fps.
        pub readback_stride: usize,
        /// Particle positions/materials changed — sort + upload required before next GPU pass.
        /// Set by spawn, phase_transition, mark_particles_dirty().
        layout_dirty: bool,
        /// Pending impulses to apply on GPU at the start of the next step_frame.
        /// Applied via a dedicated compute pass that reads LIVE GPU particle positions,
        /// avoiding the stale-CPU-mirror artifacts from the old upload approach.
        pending_impulses: Vec<GpuImpulseEntry>,
        /// Pending async readback — Some while GPU → staging copy + mapping is in flight.
        /// Checked each step_frame; on completion, CPU particles are updated without blocking.
        /// Arc<Mutex<...>> so the wgpu callback (any thread) can signal the main thread.
        pending_readback: Option<ReadbackResult>,
    }

    impl GpuSimulation {
        /// Create a GpuSimulation, initialize wgpu, upload initial particle and material data.
        ///
        /// `async` because wgpu adapter/device requests are async.
        /// In examples, wrap with `pollster::block_on(GpuSimulation::new(...))`.
        pub async fn new(
            config: SimConfig,
            particles: Vec<Particle>,
            registry: MaterialRegistry,
        ) -> Self {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("no suitable GPU adapter found");

            // Request the adapter's actual limits, not wgpu's conservative defaults (128MiB
            // storage binding). Hardware commonly supports far more (e.g. 2047MiB on desktop
            // GPUs) — capping at the default artificially shrinks the single-buffer particle/grid
            // ceiling well below what the device can actually do.
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("emerge_gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default() // experimental_features, trace, memory_hints
                })
                .await
                .expect("failed to create wgpu device");

            let device = Arc::new(device);
            let queue = Arc::new(queue);
            Self::with_device(device, queue, config, particles, registry)
        }

        /// Build a `GpuSimulation` on an existing device/queue so its GPU buffers can be
        /// shared with a renderer or surface on the same device — required for the
        /// zero-readback [`crate::render::Renderer::render_gpu`] path. `new()` creates its
        /// own headless device instead, which is correct for compute-only or CPU-readback
        /// workflows but cannot share GPU buffers with another device.
        pub fn with_device(
            device: Arc<wgpu::Device>,
            queue: Arc<wgpu::Queue>,
            config: SimConfig,
            particles: Vec<Particle>,
            registry: MaterialRegistry,
        ) -> Self {
            let material_params = registry.all_params();

            // Run init_particle before uploading. Mirrors Simulation::spawn_region().
            // Materials that seed plastic state (Snow: Jp=1, Sand: q=neutral) start wrong
            // without this.
            let mut initialized = particles;
            for p in &mut initialized {
                registry.get(p.material_id).init_particle(p);
            }
            let particle_count = initialized.len();

            let buffers = GpuBuffers::new(
                &device,
                particle_count,
                config.grid_res,
                MAX_MATERIALS,
                config.max_substeps_per_step,
            );

            buffers.upload_particles(&queue, &initialized);
            buffers.upload_materials(&queue, &material_params);

            let pipelines = SimPipelines::new(&device);

            Self {
                device,
                queue,
                buffers,
                pipelines,
                config,
                registry,
                particles: initialized,
                particle_count,
                last_sub_dt: config.dt,
                last_substeps: 0,
                frame_index: 0,
                force_field_entries: Vec::new(),
                readback_frame: 0,
                readback_stride: 1,
                layout_dirty: true, // seed particle_sort on first step_frame
                pending_impulses: Vec::new(),
                pending_readback: None,
            }
        }

        /// Mark CPU particles as layout-changed (positions/materials) — triggers sort + upload.
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

        /// The wgpu Device — share with the LP render system to read the particle buffer directly.
        pub fn device(&self) -> &Arc<wgpu::Device> {
            &self.device
        }

        /// The wgpu Queue — share with the LP render system for command submission.
        pub fn queue(&self) -> &Arc<wgpu::Queue> {
            &self.queue
        }

        /// The GPU particle storage buffer — bind this in LP's custom render shader.
        /// Layout: `array<Particle>`, each Particle is 112 bytes, repr(C).
        /// Stays in VRAM between frames; read-only from the render side.
        pub fn particle_buffer(&self) -> &wgpu::Buffer {
            &self.buffers.particles
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

        /// Advance one frame of simulation time (`config.dt`) using the GPU.
        ///
        /// All substeps are encoded into a single command buffer and submitted once — one driver
        /// call regardless of adaptive substep count. Step params are pre-computed from the CPU
        /// particle mirror (same one-frame CFL lag as before, no physics change).
        pub fn step_frame(&mut self) {
            let any_cpu = self.registry.any_needs_cpu_update();

            // Upload CPU → GPU only when positions/materials actually changed.
            // Impulses are now applied by a dedicated GPU compute pass (apply_impulses) that
            // reads LIVE GPU positions — no CPU mirror upload needed for impulse-only frames.
            let needs_upload = self.layout_dirty || any_cpu;
            if needs_upload {
                let res = self.config.grid_res as u32;
                self.particles.sort_unstable_by_key(|p| {
                    let cx = (p.x.x as u32).min(res.saturating_sub(1));
                    let cy = (p.x.y as u32).min(res.saturating_sub(1));
                    cy * res + cx
                });
                self.buffers.upload_particles(&self.queue, &self.particles);
                self.layout_dirty = false;
            }

            // Pre-compute all sub_dts from CPU mirror (same one-frame lag as before).
            // CFL scan is O(N) — run it ONCE and reuse the result to fill the sub_dts array.
            // The CPU mirror is static within a frame so every repeated call would return the
            // same value anyway. Previously this called choose_substep_dt up to 16×/frame
            // (once per substep), which in debug mode caused measurable cursor slowdown.
            let particles_soa = crate::particle::Particles::from(self.particles.clone());
            let sub_dt_cfl = choose_substep_dt(
                &self.config,
                &particles_soa,
                particles_soa.len(),
                &self.registry,
                self.config.dt,
            );
            let mut sub_dts: Vec<f32> = Vec::with_capacity(self.config.max_substeps_per_step);
            {
                let mut remaining = self.config.dt;
                while remaining > f32::EPSILON && sub_dts.len() < self.config.max_substeps_per_step
                {
                    let sub_dt = sub_dt_cfl.min(remaining);
                    sub_dts.push(sub_dt);
                    remaining -= sub_dt;
                }
            }
            self.last_substeps = sub_dts.len();
            self.last_sub_dt = sub_dts.last().copied().unwrap_or(self.config.dt);
            self.frame_index += 1;

            // Build force fields uniform (same every substep).
            let mut ff_params: GpuFieldsParams = bytemuck::Zeroable::zeroed();
            ff_params.count = self.force_field_entries.len() as u32;
            for (i, e) in self.force_field_entries.iter().enumerate() {
                ff_params.entries[i] = *e;
            }
            self.buffers
                .upload_force_fields_params(&self.queue, &ff_params);

            // Upload step_params for each substep into its pool slot and build a bind group.
            // All uploads happen before the command buffer executes — pool ensures each substep
            // reads its own dt from a distinct buffer.
            let bind_groups: Vec<wgpu::BindGroup> = sub_dts
                .iter()
                .enumerate()
                .map(|(i, &sub_dt)| {
                    let params = GpuStepParams::new(&self.config, sub_dt, self.particle_count);
                    self.buffers.upload_step_params_at(&self.queue, i, &params);
                    self.pipelines.make_bind_group(
                        &self.device,
                        &self.buffers,
                        &self.buffers.step_params_pool[i],
                    )
                })
                .collect();

            // Encode everything into one command buffer — one GPU submit per frame.
            // Order: [apply_impulses?] → [particle_sort?] → substep_0 → … → substep_N
            //
            // apply_impulses runs first so physics sees the freshly-applied velocities.
            // particle_sort re-seeds sorted_particle_ids after a CPU upload (layout_dirty).
            // Both use dedicated buffer slots so they never alias substep params.
            let grid_wg = (self.config.grid_res as u32).div_ceil(WG_GRID);
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
                let sort_params =
                    GpuStepParams::new(&self.config, self.config.dt, self.particle_count);
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
                pass.set_pipeline(&self.pipelines.particle_sort_clear);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_count);
                pass.dispatch_workgroups(particle_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.particle_sort_scan);
                pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 256 == NUM_BLOCKS
                pass.set_pipeline(&self.pipelines.particle_sort_scatter);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            for bg in &bind_groups {
                self.encode_substep(&mut encoder, bg, grid_wg, particle_wg);
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            // Async GPU → CPU readback — never blocks the render thread.
            //
            // Two-phase: begin_readback submits a GPU copy + async map (non-blocking).
            // The receiver fires on a subsequent frame when the GPU copy + map completes.
            // We pump wgpu callbacks with poll(Poll) each frame so the mapping progresses.
            //
            // If any_cpu: readback every frame (CPU plasticity needs current state).
            // Otherwise: stride-gated to reduce overhead.
            self.readback_frame = self.readback_frame.wrapping_add(1);
            let want_readback = any_cpu || self.readback_frame.is_multiple_of(self.readback_stride);

            // Pump wgpu callbacks so any in-flight mapping can complete.
            self.device.poll(wgpu::PollType::Poll).ok();

            // Check if a previous async readback completed.
            let readback_done = self
                .pending_readback
                .as_ref()
                .and_then(|flag| flag.lock().ok().and_then(|mut g| g.take()));
            if readback_done.map(|r| r.is_ok()).unwrap_or(false) {
                let gpu_particles = self.buffers.finish_readback(self.particle_count);
                self.pending_readback = None;

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
                    let mut soa = Particles::from(std::mem::take(&mut self.particles));
                    for i in 0..soa.len() {
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
            }

            // Start a new readback if wanted and none is already in flight.
            if want_readback && self.pending_readback.is_none() {
                self.pending_readback = Some(self.buffers.begin_readback(
                    &self.device,
                    &self.queue,
                    self.particle_count,
                ));
            }
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

        /// Encode one substep's passes into an existing encoder. No submission — caller batches.
        fn encode_substep(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            bg: &wgpu::BindGroup,
            grid_wg: u32,
            particle_wg: u32,
        ) {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grid_clear"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.grid_clear);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("p2g"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.p2g);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("grid_update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.grid_update);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(grid_wg, grid_wg, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("g2p"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.g2p);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("particles_update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.particles_update);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("force_fields"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.force_fields);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }

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

        /// Read-only access to the CPU particle mirror (one frame behind GPU when strided).
        pub fn particles(&self) -> &[Particle] {
            &self.particles
        }

        /// Mutable access to the CPU particle mirror.
        ///
        /// **CFL WARNING:** velocity changes bypass the solver's CFL clamp.
        /// For gameplay impulses use `apply_impulse` / `apply_radial_impulse` instead.
        /// After modifying, call `mark_particles_dirty()` so the GPU sees the changes.
        pub fn particles_mut(&mut self) -> &mut Vec<Particle> {
            &mut self.particles
        }

        /// Append a new particle region to the simulation.
        ///
        /// Generates particles CPU-side, appends to the internal mirror, recomputes
        /// initial volumes for all particles, then reallocates the GPU particle buffer
        /// to fit the new total and uploads all particles.
        ///
        /// Returns the index range the new particles occupy in the internal mirror.
        /// LP uses this as `creature_id → particle_range` for ownership tracking.
        ///
        /// Call before `step_frame` — mid-frame spawning is not supported.
        pub fn spawn_region(&mut self, spawn: SpawnRegion) -> std::ops::Range<usize> {
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
                let mut tmp_soa =
                    crate::particle::Particles::from(std::mem::take(&mut self.particles));
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
            self.pending_readback = None; // old staging is gone
            self.buffers.upload_particles(&self.queue, &self.particles);
            start..n
        }

        pub fn config(&self) -> &SimConfig {
            &self.config
        }
        pub fn particle_count(&self) -> usize {
            self.particle_count
        }

        /// Blocking GPU → CPU particle sync. Updates `self.particles` immediately.
        /// Stalls the CPU until all in-flight GPU work completes — use only after step_frame
        /// when you need current positions right now (e.g. rendering). Not for the hot path.
        pub fn sync_particles_blocking(&mut self) {
            // If an async readback is in-flight, the staging buffer may be mapped or pending map.
            // Wait for it to complete, then consume it to unmap the staging buffer before reuse.
            if let Some(flag) = self.pending_readback.take() {
                self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
                if flag.lock().ok().and_then(|mut g| g.take()).is_some() {
                    let _ = self.buffers.finish_readback(self.particle_count);
                }
            }
            self.particles =
                self.buffers
                    .readback_blocking(&self.device, &self.queue, self.particle_count);
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
        /// Layout: array<Cell> where Cell = { momentum: vec2, mass: f32, _pad: f32 } (16 bytes).
        /// Consumers (e.g. LP's metaball renderer) can bind this read-only in their own compute pass.
        pub fn grid_buffer(&self) -> &wgpu::Buffer {
            &self.buffers.grid
        }

        /// Register a material, auto-assigning the next available ID.
        ///
        /// Mirrors `Simulation::register_material` — use this instead of `set_material`
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

        /// Physics snapshot from the CPU particle mirror (one frame behind GPU when strided).
        /// Grid-side fields (mass error, momentum error, active cells) are zero — GPU grid is
        /// not readable on CPU. All particle-side fields are exact.
        pub fn diagnostics_snapshot(&self) -> crate::diagnostics::SimSnapshot {
            crate::diagnostics::collect_snapshot_particles_only(
                self.frame_index,
                &self.particles,
                &self.config,
                self.last_sub_dt,
                self.last_substeps,
            )
        }

        /// Iterate over (index, &Particle) pairs within `radius` grid-cells of `center`.
        /// Reads the internal CPU particle mirror — one frame behind GPU when strided.
        pub fn particles_near(
            &self,
            center: glam::Vec2,
            radius: f32,
        ) -> impl Iterator<Item = (usize, &Particle)> {
            let r2 = radius * radius;
            self.particles
                .iter()
                .enumerate()
                .filter(move |(_, p)| (p.x - center).length_squared() <= r2)
        }

        /// Count particles of `material_id` within `radius` grid-cells of `center`. O(N).
        pub fn count_near(&self, center: glam::Vec2, radius: f32, material_id: u32) -> usize {
            let r2 = radius * radius;
            self.particles
                .iter()
                .filter(|p| p.material_id == material_id && (p.x - center).length_squared() <= r2)
                .count()
        }

        /// Aggregate state for all particles of the given material.
        pub fn material_state(&self, material_id: u32) -> crate::solver::query::BodyState {
            crate::solver::query::body_state_of_slice(&self.particles, material_id)
        }

        /// Aggregate state for all particles within `radius` grid-cells of `center`.
        pub fn region_state(
            &self,
            center: glam::Vec2,
            radius: f32,
        ) -> crate::solver::query::BodyState {
            crate::solver::query::region_body_state_of_slice(&self.particles, center, radius)
        }

        /// Reassign material for all particles matching `predicate`. Marks dirty so GPU
        /// sees the change on the next `step_frame` call.
        pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
        where
            F: Fn(&Particle) -> bool,
        {
            assert!(
                self.registry.is_registered(new_material_id),
                "phase_transition: material_id {new_material_id} is not registered — \
                 call solver.set_material({new_material_id}, ...) first"
            );
            for p in self.particles.iter_mut() {
                if predicate(p) {
                    p.material_id = new_material_id;
                }
            }
            self.layout_dirty = true; // material_id changed — sort order may differ
        }

        /// Add `force` to every particle within `radius` cells of `center`, scaled by proximity.
        /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions,
        /// avoiding any stale-CPU-mirror artifacts. No CPU particle scan.
        pub fn apply_impulse(&mut self, center: glam::Vec2, radius: f32, force: glam::Vec2) {
            if self.pending_impulses.len() < MAX_GPU_IMPULSES {
                self.pending_impulses.push(GpuImpulseEntry {
                    center: center.to_array(),
                    radius,
                    strength: 0.0,
                    force: force.to_array(),
                    mode: 1,
                    _pad: 0,
                });
            } else {
                eprintln!(
                    "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
                );
            }
        }

        /// Push every particle within `radius` cells outward from `center`.
        /// Applied on the GPU at the start of the next step_frame — reads LIVE GPU positions.
        /// `strength` may be negative to pull. No CPU particle scan.
        pub fn apply_radial_impulse(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
            if self.pending_impulses.len() < MAX_GPU_IMPULSES {
                self.pending_impulses.push(GpuImpulseEntry {
                    center: center.to_array(),
                    radius,
                    strength,
                    force: [0.0; 2],
                    mode: 0,
                    _pad: 0,
                });
            } else {
                eprintln!(
                    "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) — impulse dropped"
                );
            }
        }
    }
}

//! GPU uniform-buffer parameter structs: per-substep step params, force-field
//! entries, impulse entries, sleep/wake params, and spatial-block constants.
//!
//! Split out of `gpu/mod.rs` -- pure `#[repr(C)]`/`bytemuck::Pod` data plus
//! constructors. No wgpu device/buffer handling lives here; that's
//! `gpu::solver`.

use crate::solver::config::SimConfig;

/// Re-export so GPU code reads the same limit as the registry.
/// Injected into WGSL shaders at pipeline creation — change only in `materials/registry.rs`.
pub use crate::materials::registry::MAX_MATERIAL_SLOTS as MAX_MATERIALS;

/// Per-substep solver constants uploaded to the GPU uniform buffer before each substep.
///
/// 48 bytes, 16-byte aligned — satisfies WGSL uniform binding requirements.
/// Fields mirror `struct StepParams` in every WGSL shader exactly (same offsets, same types).
///
/// All values come from `SimConfig` or are computed from it — no hardcoded physics here.
/// Uniform data uploaded once per GPU substep.
///
/// Layout (48 bytes, 16-byte aligned — WGSL uniform binding requirement):
///   offset  0: grid_res       u32
///   offset  4: particle_count u32
///   offset  8: dt             f32
///   offset 12: kernel_d_inverse      f32  (always 4.0 — quadratic B-spline)
///   offset 16: gravity        `vec2<f32>`  (8 bytes; 8-byte aligned in WGSL ✓)
///   offset 24: boundary_thickness u32
///   offset 28: vel_limit      f32
///   offset 32: sleep_threshold f32  (0.0 = sleep/wake disabled, SimConfig default)
///   offset 36: contact_friction f32 (SimConfig::contact_friction, GPU port — repurposes
///                             the first of 3 original pad slots, see field doc)
///   offset 40: grid_cell_size f32 (SimConfig::grid_cell_size, repurposes the second
///                             original pad slot — read by `resolve_contact.wgsl`'s
///                             normal fit + Baumgarte cap, previously hardcoded to 1.0
///                             there, a real latent bug for any config with a non-default
///                             grid_cell_size, fixed 2026-07-15)
///   offset 44: contact_active u32 (0/1 — repurposes the third pad slot. True iff any
///                             particle anywhere has `contact_group != 0` this frame.
///                             Mirrors CPU's `Grid::has_contact_activity()` gate in
///                             `transfer.rs` — lets `g2p.wgsl` skip straight to the plain
///                             grid velocity, and lets `resolve_contact`/`gather_contact_
///                             points` be skipped entirely, for every scene that never
///                             uses multi-field contact. Fixed 2026-07-15: this gate did
///                             not exist on GPU before, so EVERY scene paid full contact-
///                             resolution cost regardless of use (measured: resolve_contact
///                             alone was 37.5%/5.66ms of a substep on a pure fluid scene
///                             with zero contact particles).
///                             = 48 bytes, 16-byte aligned ✓
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
    pub vel_limit: f32,       // grid_cell_size / sub_dt
    pub sleep_threshold: f32, // SimConfig::sleep_threshold — 0.0 disables sleep/wake entirely
    /// Multi-field contact (GPU port) — `SimConfig::contact_friction`, read by
    /// `resolve_contact.wgsl`. Repurposes the first of the original 3 `_pad` u32
    /// slots — total struct size/offsets of every other field are UNCHANGED, so
    /// shaders that don't care about contact (their own `StepParams` copy still
    /// declares `_pad0: u32`) read harmless bits and never touch this value.
    pub contact_friction: f32,
    /// `SimConfig::grid_cell_size` — read by `resolve_contact.wgsl`'s normal fit
    /// (penalty scaling) and Baumgarte correction cap. Was hardcoded to 1.0 in that
    /// shader before this field existed; harmless while every real config used the
    /// default 1.0, but a real correctness gap the moment one didn't.
    pub grid_cell_size: f32,
    /// True (nonzero) iff any particle anywhere has `contact_group != 0` this frame —
    /// see this field's doc in the layout comment above.
    pub contact_active: u32,
}

impl GpuStepParams {
    pub fn new(
        config: &SimConfig,
        sub_dt: f32,
        particle_count: usize,
        contact_active: bool,
    ) -> Self {
        Self {
            grid_res: config.grid_res as u32,
            particle_count: particle_count as u32,
            dt: sub_dt,
            kernel_d_inverse: crate::solver::config::KERNEL_D_INVERSE,
            gravity: config.gravity,
            boundary_thickness: config.boundary_thickness as u32,
            vel_limit: config.grid_cell_size / sub_dt,
            sleep_threshold: config.sleep_threshold,
            contact_friction: config.contact_friction,
            grid_cell_size: config.grid_cell_size,
            contact_active: contact_active as u32,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuStepParams>() == 48);

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
    pub const LINEAR_DRAG: u32 = 7;
    pub const SPATIAL_DRAG_CYLINDER: u32 = 8;
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
    ///   (`ρ_SI · dx_m²` — same value as `NewtonianFluidMaterial::rest_density`, fixed
    ///   2026-07-07 to drop an incorrect extra `/dt_s²` factor)
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

    /// Linear drag toward a target/ambient flow velocity: a = k·(v_target − v_particle) —
    /// see `LinearDragField`'s (CPU) doc comment for the real physics (Stokes drag /
    /// Rayleigh friction) this mirrors exactly. River current, wind-blown sand, any scene
    /// needing sustained directional flow instead of gravity settling into a static
    /// pool/pile.
    ///
    /// - `target_velocity`: the ambient flow velocity particles relax toward
    /// - `drag_coefficient`: relaxation rate k (1/time); decay timescale is 1/k
    /// - `material_mask`: general bitmask (`1 << material_id`, OR together for several,
    ///   or `Self::ALL_MATERIALS`) — NOT a single `material_id` like most other
    ///   constructors here, matching `LinearDragField`'s own CPU-side parameter exactly
    ///   for real CPU/GPU parity.
    pub fn linear_drag(
        target_velocity: glam::Vec2,
        drag_coefficient: f32,
        material_mask: u32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = target_velocity.x;
        p[1] = target_velocity.y;
        p[2] = drag_coefficient;
        Self {
            field_type: field_type::LINEAR_DRAG,
            material_mask,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Spatially-varying wind/current drag: same `a = k·(target(x) − v)` mechanism as
    /// `linear_drag`, but `target` is sampled from the real, exact closed-form solution
    /// for 2D potential flow around a circular cylinder (uniform stream + doublet
    /// superposition — see CPU's `SpatialDragField`/its test module doc for the derivation
    /// and citations). WGSL has no function pointers, so unlike CPU's generic
    /// `target_velocity_fn: fn(Vec2) -> Vec2`, this GPU port bakes this ONE specific
    /// analytic formula into its own field-type case in `force_fields.wgsl` — the real,
    /// disclosed trade-off of porting a fn-pointer-based mechanism to a shader.
    ///
    /// - `cylinder_center`: the flow singularity's position, in grid coordinates
    /// - `free_stream_u`: undisturbed flow speed far from the cylinder (+X direction)
    /// - `radius`: cylinder radius `a` (the doublet strength is `U·a²`)
    /// - `drag_coefficient`: same relaxation rate `k` as `linear_drag`
    pub fn spatial_drag_potential_flow_cylinder(
        cylinder_center: glam::Vec2,
        free_stream_u: f32,
        radius: f32,
        drag_coefficient: f32,
        material_mask: u32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = cylinder_center.x;
        p[1] = cylinder_center.y;
        p[2] = free_stream_u;
        p[3] = radius;
        p[4] = drag_coefficient;
        Self {
            field_type: field_type::SPATIAL_DRAG_CYLINDER,
            material_mask,
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

/// Max tags per frame for force-sleep/force-wake-by-tag.
/// Must match `array<u32, 8>` in `force_fields.wgsl`.
///
/// Minimal hook for LP's future chunk system (see `mpm_technique_survey` memory
/// note): a chunk leaving camera range force-sleeps its particles by `user_tag`
/// regardless of velocity; a chunk re-entering range force-wakes them. The chunk
/// system itself — tagging particles by chunk, tracking camera distance — is
/// LP's job, not emerge's. This is just the primitive it needs.
pub const MAX_SLEEP_WAKE_TAGS: usize = 8;

/// Uniform data for force-sleep/force-wake-by-tag, checked once per substep in
/// `force_fields.wgsl` — 80 bytes. Matches `struct SleepWakeParams` in WGSL.
///
/// Tags are packed 4-per-`vec4<u32>` (`[[u32; 4]; 2]` = 8 tags), not a flat
/// `[u32; 8]` — WGSL requires uniform-address-space arrays to have a 16-byte
/// element stride, so a flat u32 array would be rejected by naga at shader-module
/// creation (same class of gotcha as `vec3<u32>` padding elsewhere in this file).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSleepWakeParams {
    pub sleep_count: u32,
    pub wake_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub sleep_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
    pub wake_tags: [[u32; 4]; MAX_SLEEP_WAKE_TAGS / 4],
}

const _: () = assert!(core::mem::size_of::<GpuSleepWakeParams>() == 80);

/// Spatial-block bucket geometry for the particle_sort histogram AND the
/// active-block detection it now also feeds (GPU sparse grid, Phase 1 — see
/// `mpm_technique_survey` memory note). Single Rust-side source of truth: must
/// match `NUM_BLOCKS_PER_DIM`/`NUM_BLOCKS` in `particle_sort.wgsl` and
/// `grid_clear.wgsl` exactly. Re-deriving from `grid_res` at runtime is not an
/// option — this sizes `block_counts`/`active_block_ids`, both allocated once
/// at `GpuBuffers::new()`, so it must be a fixed compile-time constant, same
/// class as `MAX_FORCE_FIELDS`.
pub const NUM_BLOCKS_PER_DIM: usize = 16;
pub const NUM_BLOCKS: usize = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM; // 256

/// Multi-field contact (GPU port, first slice — see project memory
/// `locomotion_core_frictional_contact_2026-07-11`'s 2026-07-14 GPU-port entry): fixed
/// capacity for the labeled contact point cloud (`+1.0` grip / `-1.0` rest) that the
/// Newton-Raphson LR normal fit will read.
///
/// REAL BUG FOUND AND FIXED while building this: the first version of this bucketed
/// points per exact GRID NODE (`grid_res² × capacity`) — this OOM'd the existing
/// `gpu_grid_resolution_cost` regression test at grid_res=2048 (a ~4 GiB allocation),
/// confirmed via the real failure, not predicted in advance. Fixed by bucketing per
/// coarse BLOCK instead (`NUM_BLOCKS` = 256, fixed regardless of grid_res — the same
/// spatial partition `particle_sort.wgsl` already uses), so total memory
/// (`NUM_BLOCKS × MAX_CONTACT_POINTS_PER_BLOCK × 16 bytes` ≈ 16 MiB) is now CONSTANT
/// at any grid resolution, matching how `active_block_ids`/`block_counts` are already
/// sized. Trade-off, disclosed: a future `resolve_contact` pass processing a specific
/// node must scan its own block's (and, for stencil spillover at a block boundary,
/// neighboring blocks') point bucket and filter by real distance, rather than reading
/// an exact per-node list directly — more per-node filtering work, but bounded,
/// constant memory instead of a real OOM.
///
/// 4096 per block is a real, disclosed cap (like the CPU's own bounded caps, e.g.
/// `fit_contact_normal_lr`'s 15-iteration Newton cap) — chosen generously since a
/// block covers many cells' worth of interface, not just one node; the atomic
/// slot-claim is bounds-checked (points beyond the cap are dropped, not undefined
/// behavior), and `contact_point_counts` keeps counting past the cap so overflow is a
/// real, observable signal, not silently absorbed.
pub const MAX_CONTACT_POINTS_PER_BLOCK: usize = 4096;

/// Debug/test-only uniform for `resolve_contact.wgsl`'s `debug_fit_normal_main` — picks
/// which block's point cloud to run the Newton-Raphson LR normal fit against and what
/// `node_pos` to center it on. Not part of the real per-substep pipeline; exists solely
/// to verify `fit_contact_normal_lr`'s WGSL port in isolation, the same way CPU's own
/// `fit_contact_normal_lr_tests` module unit-tests the fit separately from the full
/// `resolve_contact` integration.
/// Field order matters: `node_pos` (`vec2<f32>`) must start at an 8-byte-aligned
/// offset per WGSL uniform-address-space rules (same reasoning as `GpuStepParams`'s
/// own `gravity` field) — putting it FIRST (offset 0) satisfies that without needing
/// explicit padding, unlike the u32 fields which only need 4-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactDebugParams {
    pub node_pos: glam::Vec2,
    pub target_block: u32,
    pub point_count: u32,
}

const _: () = assert!(core::mem::size_of::<ContactDebugParams>() == 16);

/// Directional (setae-style) grip friction — GPU mirror of `DirectionalContactGrip`
/// (`src/spacetime/grid/mod.rs`). Always uploaded, every substep contact is active:
/// `mu_easy == mu_resist` (both set to `SimConfig::contact_friction` when no directional
/// bias is in play) makes `resolve_direction_aware` (`resolve_contact.wgsl`) reduce
/// exactly to plain symmetric Coulomb friction — see that function's own doc for why
/// this is ONE code path, not two. Field order: `easy_direction` first (8-byte
/// alignment), matching `ContactDebugParams`' own convention.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDirectionalGripParams {
    pub easy_direction: glam::Vec2,
    pub mu_easy: f32,
    pub mu_resist: f32,
}

impl GpuDirectionalGripParams {
    /// Plain symmetric Coulomb friction at `friction` — no directional bias.
    pub fn symmetric(friction: f32) -> Self {
        Self {
            easy_direction: glam::Vec2::X,
            mu_easy: friction,
            mu_resist: friction,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuDirectionalGripParams>() == 16);

/// Grid-based Fourier heat diffusion — GPU mirror of `ThermalDiffusion`/`ThermalConfig`
/// (`src/energy/thermodynamics/diffusion.rs`). Implements the same real PDE:
/// `∂T/∂t = α·∇²T` (Fourier's law) plus Newton cooling `dT/dt = −k_c·(T−ambient)`.
/// `dt` itself is NOT stored here — the thermal pass reads `step_params.dt` (group 0)
/// directly, since substep `dt` is already the single source of truth uploaded there
/// every substep; duplicating it here would risk the two going out of sync.
/// `enabled == 0` skips all 4 thermal passes entirely (see `contact_active`'s identical
/// gate-when-unused pattern) — every scene that never attaches thermal pays nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuThermalParams {
    /// Thermal diffusivity α = k / (c_p · dx²), grid-units²/s — see
    /// `ThermalConfig::alpha_grid`'s own doc for the real derivation/units.
    pub alpha: f32,
    /// Ambient/boundary temperature — empty cells and Newton cooling both relax toward this.
    pub ambient: f32,
    /// Newton cooling rate k_c, 1/s. 0.0 = no cooling (adiabatic walls).
    pub cooling_rate: f32,
    /// 0 = no thermal system attached (default, every existing scene) — skips all 4
    /// thermal passes. 1 = attached and active.
    pub enabled: u32,
}

impl GpuThermalParams {
    pub fn disabled() -> Self {
        Self {
            alpha: 0.0,
            ambient: 0.0,
            cooling_rate: 0.0,
            enabled: 0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuThermalParams>() == 16);

/// Generic reaction-diffusion resource field — GPU mirror of `ScalarDiffusionField`
/// (`src/energy/thermodynamics/scalar_field.rs`), specialized to the ONE real source
/// term its own CPU test module actually uses: logistic growth (Verhulst 1838,
/// `dφ/dt = r·φ·(1−φ/K)`). Same real PDE shape as `GpuThermalParams` (scatter ->
/// normalize -> Laplacian+reaction -> gather), its own separate group/buffers AND its
/// own separate carrier field (`particle.scalar_field`, not `particle.temperature`) --
/// composes freely with `GpuThermalParams` in the same scene (real fix, 2026-07-17;
/// they used to share `temperature` and could never both be attached at once).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuResourceParams {
    /// Diffusivity D, grid-units²/s -- spatial spread rate.
    pub diffusivity: f32,
    /// Value assigned to empty cells (no particle mass) and domain boundaries.
    pub ambient: f32,
    /// Logistic growth rate r, 1/s.
    pub resource_r: f32,
    /// Logistic carrying capacity K.
    pub resource_k: f32,
    /// 0 = no resource system attached (default) -- skips all 4 passes entirely.
    pub enabled: u32,
    pub _pad: [u32; 3],
}

impl GpuResourceParams {
    pub fn disabled() -> Self {
        Self {
            diffusivity: 0.0,
            ambient: 0.0,
            resource_r: 0.0,
            resource_k: 0.0,
            enabled: 0,
            _pad: [0; 3],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuResourceParams>() == 32);

/// ASFLIP (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the Material Point
/// Method: A Scheme for Easier Separation and Less Dissipation") -- GPU mirror of
/// `SimConfig::asflip_blend`. `enabled == 0` (the default, every existing scene) makes
/// `grid_update.wgsl` skip the pre-force velocity snapshot write entirely and the fused
/// `g2p_asflip_fused.wgsl` pass never gets dispatched (see `SubstepGates::asflip_active`)
/// -- zero cost, byte-identical behavior to before ASFLIP existed, matching every other
/// opt-in GPU subsystem's own gate.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAsflipParams {
    /// Blend factor [0, 1] -- see `SimConfig::asflip_blend`'s own doc for the real
    /// derivation and the ~0.97 reference value (`nepluno/pyasflip`).
    pub blend: f32,
    /// 0 = ASFLIP disabled (default) -- skips the snapshot write and the fused G2P+
    /// position pass, falling back to the ordinary split g2p/particles_update passes
    /// unchanged. 1 = attached and active.
    pub enabled: u32,
    pub _pad: [u32; 2],
}

impl GpuAsflipParams {
    pub fn disabled() -> Self {
        Self {
            blend: 0.0,
            enabled: 0,
            _pad: [0; 2],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuAsflipParams>() == 16);

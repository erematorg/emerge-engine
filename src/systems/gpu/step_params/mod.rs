//! GPU uniform-buffer parameter structs: per-substep step params, force-field
//! entries, impulse entries, sleep/wake params, spatial-block constants, and
//! opt-in per-substep subsystem params (thermal/resource/ASFLIP/material-mass).
//!
//! Split out of `gpu/mod.rs` -- pure `#[repr(C)]`/`bytemuck::Pod` data plus
//! constructors. No wgpu device/buffer handling lives here; that's
//! `gpu::solver`.
//!
//! Was a single ~650-line file (`step_params.rs`); its own header already named
//! five concerns, and a sixth ("opt-in subsystem params" -- thermal, resource,
//! ASFLIP, material-mass, each following the same `enabled`/`disabled()` gate
//! pattern) had grown alongside it since. Split one file per concern, all
//! re-exported flat here so every existing `step_params::X` path (both
//! crate-internal `use super::step_params::{...}` call sites and `gpu::mod`'s
//! own `pub use`) keeps working unchanged:
//!   - `substep.rs`        -- `GpuStepParams`, the core per-substep uniform
//!   - `force_fields.rs`   -- `GpuFieldEntry`/`GpuFieldsParams` + `field_type`
//!   - `impulses.rs`       -- `GpuImpulseEntry`/`GpuImpulseParams`
//!   - `sleep_wake.rs`     -- `GpuSleepWakeParams`
//!   - `spatial_blocks.rs` -- sort/contact block-partition constants +
//!     `ContactDebugParams`/`GpuDirectionalGripParams`
//!   - `subsystems.rs`     -- `GpuThermalParams`/`GpuResourceParams`/
//!     `GpuAsflipParams`/`GpuMaterialMassParams`

/// Re-export so GPU code reads the same limit as the registry.
/// Injected into WGSL shaders at pipeline creation -- change only in `materials/registry.rs`.
pub use crate::materials::registry::MAX_MATERIAL_SLOTS as MAX_MATERIALS;

mod force_fields;
mod impulses;
mod sleep_wake;
mod spatial_blocks;
mod substep;
mod subsystems;

pub use force_fields::{GpuFieldEntry, GpuFieldsParams, MAX_FORCE_FIELDS, field_type};
pub use impulses::{GpuImpulseEntry, GpuImpulseParams, MAX_GPU_IMPULSES};
pub use sleep_wake::{GpuSleepWakeParams, MAX_SLEEP_WAKE_TAGS};
pub use spatial_blocks::{
    ContactDebugParams, GpuDirectionalGripParams, MAX_CONTACT_POINTS_PER_BLOCK, NUM_BLOCKS,
    NUM_BLOCKS_PER_DIM, NUM_CONTACT_BLOCKS, NUM_CONTACT_BLOCKS_PER_DIM,
};
pub use substep::GpuStepParams;
pub use subsystems::{
    GpuAsflipParams, GpuFluidPressureParams, GpuMaterialMassParams, GpuResourceParams,
    GpuThermalParams, MAX_RENDER_MATERIAL_SLOTS,
};

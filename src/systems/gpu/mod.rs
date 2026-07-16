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
    pub const RESOLVE_CONTACT: &str = include_str!("shaders/resolve_contact.wgsl");
    pub const THERMAL: &str = include_str!("shaders/thermal.wgsl");
}

#[cfg(feature = "gpu")]
pub use solver::GpuSimulation;

#[cfg(feature = "gpu")]
pub use step_params::{
    GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuSleepWakeParams,
    GpuStepParams, MAX_CONTACT_POINTS_PER_BLOCK, MAX_FORCE_FIELDS, MAX_GPU_IMPULSES,
    MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS, NUM_BLOCKS_PER_DIM, field_type,
};

#[cfg(feature = "gpu")]
mod step_params;

#[cfg(feature = "gpu")]
mod solver;

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
    pub const RESOURCE_FIELD: &str = include_str!("shaders/resource_field.wgsl");
    pub const G2P_ASFLIP_FUSED: &str = include_str!("shaders/g2p_asflip_fused.wgsl");
}

#[cfg(feature = "gpu")]
pub use solver::GpuSimulation;

#[cfg(feature = "gpu")]
pub use step_params::{
    GpuFieldEntry, GpuFieldsParams, GpuImpulseEntry, GpuImpulseParams, GpuSleepWakeParams,
    GpuStepParams, MAX_CONTACT_POINTS_PER_BLOCK, MAX_FORCE_FIELDS, MAX_GPU_IMPULSES,
    MAX_RENDER_MATERIAL_SLOTS, MAX_SLEEP_WAKE_TAGS, NUM_BLOCKS, NUM_BLOCKS_PER_DIM,
    NUM_CONTACT_BLOCKS, NUM_CONTACT_BLOCKS_PER_DIM, field_type,
};

#[cfg(feature = "gpu")]
mod step_params;

#[cfg(feature = "gpu")]
mod solver;

/// Every real `wgpu::Instance` construction in this crate (production and
/// tests) must go through here — NOT `InstanceDescriptor::default()`
/// directly. Default selects the Fxc DX12 shader compiler, which fails to
/// compile `resolve_contact.wgsl` on the D3D12 WARP software adapter CI
/// runs on (`windows-latest` has no real GPU): FXC cannot unroll one of its
/// loops ("array reference cannot be used as an l-value... forcing loop to
/// unroll... unable to unroll... 6 iterations"), a known real FXC weakness
/// with dynamically-indexed local arrays that Dxc doesn't share. `StaticDxc`
/// (the `static-dxc` Cargo feature, which statically links Microsoft's own
/// DirectXShaderCompiler via `mach-dxcompiler-rs`) fixes it without needing
/// to ship a separate `dxcompiler.dll`. That dependency is Windows-only by
/// its own `Cargo.toml` target cfg, so this has zero effect on Linux/macOS
/// builds or the `ubuntu-latest` CI job.
#[cfg(feature = "gpu")]
pub(crate) fn create_wgpu_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

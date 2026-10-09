//! GPU impulse entries -- split out of `step_params.rs`, see that module's own
//! doc comment for the full file map.

/// Max impulses per frame submitted via `apply_impulse` / `apply_radial_impulse`.
/// Must match `array<ImpulseEntry, 16>` in `apply_impulses.wgsl`.
pub const MAX_GPU_IMPULSES: usize = 16;

/// One impulse descriptor -- 32 bytes, matches `struct ImpulseEntry` in WGSL.
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

/// Uniform data for the apply_impulses compute pass -- 528 bytes.
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

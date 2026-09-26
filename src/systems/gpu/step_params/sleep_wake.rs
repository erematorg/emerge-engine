//! GPU force-sleep/force-wake-by-tag params -- split out of `step_params.rs`,
//! see that module's own doc comment for the full file map.

/// Max tags per frame for force-sleep/force-wake-by-tag.
/// Must match `array<u32, 8>` in `force_fields.wgsl`.
///
/// Minimal hook for LP's future chunk system (see `mpm_technique_survey` memory
/// note): a chunk leaving camera range force-sleeps its particles by `user_tag`
/// regardless of velocity; a chunk re-entering range force-wakes them. The chunk
/// system itself -- tagging particles by chunk, tracking camera distance -- is
/// LP's job, not emerge's. This is just the primitive it needs.
pub const MAX_SLEEP_WAKE_TAGS: usize = 8;

/// Uniform data for force-sleep/force-wake-by-tag, checked once per substep in
/// `force_fields.wgsl` -- 80 bytes. Matches `struct SleepWakeParams` in WGSL.
///
/// Tags are packed 4-per-`vec4<u32>` (`[[u32; 4]; 2]` = 8 tags), not a flat
/// `[u32; 8]` -- WGSL requires uniform-address-space arrays to have a 16-byte
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

//! Spatial-block partition constants (P2G-sort/active-block detection, and the
//! separate finer contact-point partition) plus the contact-debug and
//! directional-grip uniforms that reference them -- split out of
//! `step_params.rs`, see that module's own doc comment for the full file map.

/// Spatial-block bucket geometry for the particle_sort histogram AND the
/// active-block detection it now also feeds (GPU sparse grid, Phase 1 -- see
/// `mpm_technique_survey` memory note). Single Rust-side source of truth: must
/// match `NUM_BLOCKS_PER_DIM`/`NUM_BLOCKS` in `particle_sort.wgsl` and
/// `grid_clear.wgsl` exactly. Re-deriving from `grid_res` at runtime is not an
/// option -- this sizes `block_counts`/`active_block_ids`, both allocated once
/// at `GpuBuffers::new()`, so it must be a fixed compile-time constant, same
/// class as `MAX_FORCE_FIELDS`.
pub const NUM_BLOCKS_PER_DIM: usize = 16;
pub const NUM_BLOCKS: usize = NUM_BLOCKS_PER_DIM * NUM_BLOCKS_PER_DIM; // 256

/// Multi-field contact (GPU port): fixed capacity for the labeled contact point cloud
/// (`+1.0` grip / `-1.0` rest) that the Newton-Raphson LR normal fit reads. Bucketed
/// per coarse BLOCK, not per grid node -- total memory is CONSTANT at any grid
/// resolution, matching how `active_block_ids`/`block_counts` are already sized
/// (bucketing per node would scale as `grid_res² × capacity`, OOMing at high res).
///
/// This partition is DEDICATED to contact points (`NUM_CONTACT_BLOCKS_PER_DIM`,
/// below), independent of `NUM_BLOCKS_PER_DIM` (the P2G-sort/active-block partition).
/// Sharing the coarser sort partition made `gather_local_points` scan far more
/// candidate points per node than the ~1.5-cell kernel reach actually needs -- the
/// canonical uniform-grid neighbor-search mismatch (Green, "Particle Simulation using
/// CUDA", GPU Gems 3; Ihmsen et al. 2011: bucket size should match interaction radius).
/// A finer, dedicated partition (64×64 blocks vs 16×16) closes most of that gap
/// without changing the fit's inputs/outputs (same points feed the same Newton-Raphson
/// fit, only how they're fetched changes).
///
/// 256 per block (vs 4096 for the coarser partition) keeps the same per-cell headroom
/// for the now much smaller block area, so total memory is unchanged. The atomic
/// slot-claim is bounds-checked (points beyond the cap are dropped, not UB), and
/// `contact_point_counts` keeps counting past the cap so overflow stays observable.
pub const MAX_CONTACT_POINTS_PER_BLOCK: usize = 256;

/// Dedicated spatial partition for contact-point bucketing (`gather_contact_points_main`
/// writes, `resolve_contact.wgsl`'s `gather_local_points`/`debug_fit_normal_main` read) --
/// deliberately SEPARATE from `NUM_BLOCKS_PER_DIM` (the P2G-sort/active-block-detection
/// partition), which serves an unrelated purpose and has no reason to share bucket
/// geometry with contact-point neighbor search. See `MAX_CONTACT_POINTS_PER_BLOCK`'s
/// doc for sizing rationale. Fixed regardless of `grid_res`, same reasoning as
/// `NUM_BLOCKS_PER_DIM` (sizes buffers allocated once at `GpuBuffers::new()`).
pub const NUM_CONTACT_BLOCKS_PER_DIM: usize = 64;
pub const NUM_CONTACT_BLOCKS: usize = NUM_CONTACT_BLOCKS_PER_DIM * NUM_CONTACT_BLOCKS_PER_DIM; // 4096

/// Debug/test-only uniform for `resolve_contact.wgsl`'s `debug_fit_normal_main` -- picks
/// which block's point cloud to run the Newton-Raphson LR normal fit against and what
/// `node_pos` to center it on. Not part of the real per-substep pipeline; exists solely
/// to verify `fit_contact_normal_lr`'s WGSL port in isolation, the same way CPU's own
/// `fit_contact_normal_lr_tests` module unit-tests the fit separately from the full
/// `resolve_contact` integration.
/// Field order matters: `node_pos` (`vec2<f32>`) must start at an 8-byte-aligned
/// offset per WGSL uniform-address-space rules (same reasoning as `GpuStepParams`'s
/// own `gravity` field) -- putting it FIRST (offset 0) satisfies that without needing
/// explicit padding, unlike the u32 fields which only need 4-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContactDebugParams {
    pub node_pos: glam::Vec2,
    pub target_block: u32,
    pub point_count: u32,
}

const _: () = assert!(core::mem::size_of::<ContactDebugParams>() == 16);

/// Directional (setae-style) grip friction -- GPU mirror of `DirectionalContactGrip`
/// (`src/spacetime/grid/mod.rs`). Always uploaded, every substep contact is active:
/// `mu_easy == mu_resist` (both set to `SimConfig::contact_friction` when no directional
/// bias is in play) makes `resolve_direction_aware` (`resolve_contact.wgsl`) reduce
/// exactly to plain symmetric Coulomb friction -- see that function's own doc for why
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
    /// Plain symmetric Coulomb friction at `friction` -- no directional bias.
    pub fn symmetric(friction: f32) -> Self {
        Self {
            easy_direction: glam::Vec2::X,
            mu_easy: friction,
            mu_resist: friction,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuDirectionalGripParams>() == 16);

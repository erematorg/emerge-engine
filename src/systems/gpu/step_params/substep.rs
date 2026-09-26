//! Core per-substep step params -- split out of `step_params.rs`, see that
//! module's own doc comment for the full file map.

use crate::solver::config::SimConfig;

/// Per-substep solver constants uploaded to the GPU uniform buffer before each substep.
///
/// 64 bytes, 16-byte aligned -- satisfies WGSL uniform binding requirements.
/// Fields mirror `struct StepParams` in every WGSL shader exactly (same offsets, same types).
///
/// All values come from `SimConfig` or are computed from it -- no hardcoded physics here.
/// Uniform data uploaded once per GPU substep.
///
/// Layout (64 bytes, 16-byte aligned -- WGSL uniform binding requirement):
///   offset  0: grid_res       u32
///   offset  4: particle_count u32
///   offset  8: dt             f32
///   offset 12: kernel_d_inverse      f32  (always 4.0 -- quadratic B-spline)
///   offset 16: gravity        `vec2<f32>`  (8 bytes; 8-byte aligned in WGSL ✓)
///   offset 24: boundary_thickness u32
///   offset 28: vel_limit      f32
///   offset 32: sleep_threshold f32  (0.0 = sleep/wake disabled, SimConfig default)
///   offset 36: contact_friction f32 (SimConfig::contact_friction, GPU port -- repurposes
///                             the first of 3 original pad slots, see field doc)
///   offset 40: grid_cell_size f32 (SimConfig::grid_cell_size, repurposes the second
///                             original pad slot -- read by `resolve_contact.wgsl`'s
///                             normal fit + Baumgarte cap; must not be left at a
///                             hardcoded 1.0, or non-default grid_cell_size configs
///                             desync from the CPU path)
///   offset 44: contact_active u32 (0/1 -- repurposes the third pad slot. True iff any
///                             particle anywhere has `contact_group != 0` this frame.
///                             Mirrors CPU's `Grid::has_contact_activity()` gate in
///                             `transfer.rs` -- lets `g2p.wgsl` skip straight to the plain
///                             grid velocity, and lets `resolve_contact`/`gather_contact_
///                             points` be skipped entirely, for every scene that never
///                             uses multi-field contact.
///                             + cfl_coefficient/material_cfl_coefficient/min_dt/dt_cap
///                             (the GPU's own per-substep CFL, see adaptive_cfl.wgsl)
///                             = 64 bytes, 16-byte aligned ✓
///
/// `gravity: Vec2` replaces the old `gravity: f32` + `_pad1: u32` pair --
/// same byte count, no layout change for other fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStepParams {
    pub grid_res: u32,
    pub particle_count: u32,
    pub dt: f32,
    pub kernel_d_inverse: f32,
    pub gravity: glam::Vec2, // SimConfig::gravity -- supports angled/planetary gravity
    pub boundary_thickness: u32,
    pub vel_limit: f32,       // grid_cell_size / sub_dt
    pub sleep_threshold: f32, // SimConfig::sleep_threshold -- 0.0 disables sleep/wake entirely
    /// Multi-field contact (GPU port) -- `SimConfig::contact_friction`, read by
    /// `resolve_contact.wgsl`. Repurposes the first of the original 3 `_pad` u32
    /// slots -- total struct size/offsets of every other field are UNCHANGED, so
    /// shaders that don't care about contact (their own `StepParams` copy still
    /// declares `_pad0: u32`) read harmless bits and never touch this value.
    pub contact_friction: f32,
    /// `SimConfig::grid_cell_size` -- read by `resolve_contact.wgsl`'s normal fit
    /// (penalty scaling) and Baumgarte correction cap. Must not be hardcoded to
    /// 1.0 there -- non-default grid_cell_size configs depend on this.
    pub grid_cell_size: f32,
    /// True (nonzero) iff any particle anywhere has `contact_group != 0` this frame --
    /// see this field's doc in the layout comment above.
    pub contact_active: u32,
    /// `SimConfig::cfl_coefficient` -- the GPU re-derives each substep's own CFL bound
    /// from the post-update particle state (see `adaptive_cfl.wgsl`), so it needs the
    /// same coefficients the CPU scan uses.
    pub cfl_coefficient: f32,
    /// `SimConfig::material_cfl_coefficient` -- see `cfl_coefficient`.
    pub material_cfl_coefficient: f32,
    /// `SimConfig::min_dt` -- floor on the adaptive substep.
    pub min_dt: f32,
    /// The CPU scan's own frame-start substep size: the adaptive substep never goes
    /// ABOVE it, only below, so the GPU can only tighten what the CPU already approved.
    pub dt_cap: f32,
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
            cfl_coefficient: config.cfl_coefficient,
            material_cfl_coefficient: config.material_cfl_coefficient,
            min_dt: config.min_dt,
            dt_cap: sub_dt,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuStepParams>() == 64);

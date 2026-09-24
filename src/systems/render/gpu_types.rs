//! GPU-side wire structs for the renderer -- split out of `mod.rs` (was its own
//! "GPU-side structs (must match WGSL)" section, ~90 of the file's ~895 lines).
//! Every `repr(C)` struct here must stay byte-identical to its WGSL counterpart;
//! the `size_of` asserts are the real guard against silent drift.

use std::mem;

use bytemuck::{Pod, Zeroable};

/// Mirrors `grid_volume.wgsl`'s `GridVolumeParams` -- see that shader's own doc for
/// the real technique (samples the solver's own P2G mass field directly instead of
/// per-particle splats).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridVolumeParams {
    pub(super) sx: f32,
    pub(super) tx: f32,
    pub(super) sy: f32,
    pub(super) ty: f32,
    /// Real light direction, sourced from `Renderer::set_light_dir` -- see
    /// that method's own doc for why this replaced a value hardcoded
    /// separately (and inconsistently) in each fragment shader.
    pub(super) light_dir: [f32; 2],
    pub(super) grid_res: u32,
    pub(super) mass_floor: f32,
    pub(super) material_mass_enabled: u32,
    pub(super) _pad1: f32,
    pub(super) _pad2: [f32; 2],
}
const _: () = assert!(mem::size_of::<GridVolumeParams>() == 48);

/// Bundles `render_grid_volume`'s buffer args -- same real precedent as
/// `spacetime::transfer::P2GParticleState` (a struct instead of a suppressed
/// argument-count lint).
pub struct GridVolumeSource<'a> {
    /// `GpuSimulation::grid_buffer()`.
    pub grid: &'a wgpu::Buffer,
    /// `GpuSimulation::material_mass_buffer()` -- pass it regardless of whether
    /// `attach_grid_material_render_gpu` was called; `material_mass_enabled` gates
    /// whether the shader actually reads it.
    pub material_mass: &'a wgpu::Buffer,
    pub material_mass_enabled: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceData {
    pub(super) deform_col0: [f32; 2],
    pub(super) deform_col1: [f32; 2],
    pub(super) position: [f32; 2],
    /// Real, per-particle blackbody-emission factor (`color::
    /// blackbody_glow_factor`, shared with `ByPhysics`'s own thermal-glow
    /// term) -- grounds `CameraParams::glow_strength`'s round-particle
    /// soft-glow in the particle's OWN physical state instead of applying
    /// the same brightness boost uniformly regardless of what a particle
    /// represents. 0.0 = no glow contribution (cold/untracked-temperature
    /// particles), byte-identical to the old always-on-uniform behavior
    /// only for particles that are genuinely hot.
    pub(super) emission: f32,
    pub(super) _pad: [f32; 1],
    pub(super) color: [f32; 4],
}
const _: () = assert!(mem::size_of::<InstanceData>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraParams {
    pub(super) view_proj: [f32; 16],
    pub(super) particle_scale: f32,
    pub(super) round_particles: u32,
    /// Real, general, opt-in soft-glow strength for round particles (0.0 =
    /// old hard-edged disc, byte-identical for every existing caller of
    /// `set_camera`/`set_camera_centered` that never passes a nonzero
    /// value). See `render_particles.wgsl`'s own fragment shader doc for
    /// the falloff formula.
    pub(super) glow_strength: f32,
    /// Real, general, opt-in point-light position (grid coords) for
    /// billboard-sphere Lambertian shading -- see `set_light_source`'s own
    /// doc and `render_particles.wgsl`'s `fs_main` for the real technique
    /// (reconstructs a hemisphere normal from each particle's own on-quad
    /// position, shades by its dot product with the direction to this
    /// light). Ignored unless `shading_strength > 0.0`.
    pub(super) light_pos: [f32; 2],
    /// 0.0 = no shading (old flat-disc look, byte-identical default).
    pub(super) shading_strength: f32,
    /// Real inverse-square-law reference distance (grid units): a particle
    /// exactly this far from `light_pos` gets intensity 1.0, closer is
    /// brighter, farther is dimmer, matching real physical falloff rather
    /// than the direction-only shading this replaces. `0.0` (never set)
    /// disables the falloff entirely -- see `set_light_source`'s own doc.
    pub(super) light_reference_distance: f32,
    pub(super) _pad: [f32; 1],
}
const _: () = assert!(mem::size_of::<CameraParams>() == 96);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RenderConfig {
    pub(super) mode: u32,
    pub(super) particle_count: u32,
    pub(super) vel_scale: f32,
    pub(super) _pad: u32,
}
const _: () = assert!(mem::size_of::<RenderConfig>() == 16);

/// Mirrors `curvature_flow.wgsl`'s `SurfaceParams` -- shared by the clear,
/// splat, convert, and iterate compute passes (all four only ever need
/// `grid_res`/`surface_res`/`particle_count`, one buffer covers all of them).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SurfaceParams {
    pub(super) grid_res: u32,
    pub(super) surface_res: u32,
    pub(super) particle_count: u32,
    /// -1 = no filter (every particle contributes, v1 behavior). >= 0 =
    /// only particles with this exact `material_id` are splatted -- the
    /// two-phase extension's own filter (see `curvature_flow.wgsl`'s doc).
    pub(super) phase_filter_material_id: i32,
    /// N-material extension (single-phase path only, see `curvature_
    /// flow.wgsl`'s doc): 0 = disabled, splat/clear skip the extra 16-slot
    /// per-cell atomic work entirely (zero cost, same convention as
    /// `p2g.wgsl`'s own `material_mass_params.enabled` gate). 1 = enabled.
    /// Always 0 on the dual-phase path -- its own 2-phase filter above is
    /// a different, unrelated mechanism.
    pub(super) material_mass_enabled: u32,
    /// Real simulation timestep (`SimConfig::dt`) -- see `curvature_flow.
    /// wgsl`'s own `SurfaceParams::dt` doc for why this is the real
    /// physical quantity `splat_density_main`'s velocity-stretch extension
    /// needs, not a render-frame time. Ignored by every other pass sharing
    /// this struct.
    pub(super) dt: f32,
    /// Splat kernel width in physics-grid cells -- see `curvature_flow.
    /// wgsl`'s own `SurfaceParams::splat_width_cells` doc. 1.0 = original
    /// full MPM B-spline support.
    pub(super) splat_width_cells: f32,
    /// Yu & Turk 2013 neighbourhood-fitted kernel anisotropy strength, used by
    /// `splat_density_main` only. 0.0 disables the fit entirely (the previous
    /// behaviour, bit for bit); 1.0 applies it at full strength. See
    /// `Renderer::set_anisotropy_strength` and the `render::anisotropy` module.
    pub(super) anisotropy_strength: f32,
}
const _: () = assert!(mem::size_of::<SurfaceParams>() == 32);

/// Mirrors `curvature_flow.wgsl`'s `LightDiffuseParams` -- the real,
/// persistent light-fluence diffusion pass's own uniform (see that shader's
/// own "Pass 1e" doc for the full real technique).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct LightDiffuseParams {
    pub(super) surface_res: u32,
    pub(super) material_slot: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}
const _: () = assert!(mem::size_of::<LightDiffuseParams>() == 16);

/// Mirrors `curvature_flow.wgsl`'s `WaveStepParams` -- the real, persistent
/// (across frames) 2D wave-equation pass's own uniform. See that shader's
/// own "Pass 2b" doc for the real technique (same cited numerical scheme as
/// `energy::acoustics::WaveEquation2D`, reimplemented for this GPU-resident
/// buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct WaveStepParams {
    pub(super) surface_res: u32,
    pub(super) _pad: [u32; 3],
}
const _: () = assert!(mem::size_of::<WaveStepParams>() == 16);

/// Mirrors `curvature_flow.wgsl`'s `VisibilityParams` -- the real
/// hysteresis (Schmitt-trigger) visible/invisible state pass's own
/// uniform. See that shader's own "Pass 2c" doc.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct VisibilityParams {
    pub(super) surface_res: u32,
    pub(super) mass_floor: f32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}
const _: () = assert!(mem::size_of::<VisibilityParams>() == 16);

/// Mirrors `grid_volume.wgsl`'s `GridVisibilityParams` -- the SAME real
/// hysteresis technique as `VisibilityParams` above, ported to the
/// grid-native render path's own `mass_floor` discard (a separate buffer
/// since this operates at `grid_res`, not `surface_res`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GridVisibilityParams {
    pub(super) grid_res: u32,
    pub(super) mass_floor: f32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}
const _: () = assert!(mem::size_of::<GridVisibilityParams>() == 16);

/// Mirrors `curvature_flow.wgsl`'s `BandHysteresisParams` -- the real
/// hysteresis color-band state pass's own uniform. See that shader's own
/// "Pass 2d" doc.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct BandHysteresisParams {
    pub(super) surface_res: u32,
    /// Mass of one fully-occupied cell -- see the shader's own doc. 1.0
    /// reproduces the previous absolute-threshold behaviour exactly.
    pub(super) reference_cell_mass: f32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}
const _: () = assert!(mem::size_of::<BandHysteresisParams>() == 16);

/// Mirrors `curvature_flow.wgsl`'s `SurfaceRenderParams` -- the final
/// extraction/composite fragment pass's own params. `sx/tx/sy/ty` are the
/// SAME orthographic-projection math `CameraParams` uses, but computed
/// against `surface_res`, not the physics `grid_res` -- see
/// `Renderer::render_surface_reconstruction`'s own doc.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SurfaceRenderParams {
    pub(super) sx: f32,
    pub(super) tx: f32,
    pub(super) sy: f32,
    pub(super) ty: f32,
    /// Same real light direction as `GridVolumeParams::light_dir` -- see
    /// `Renderer::set_light_dir`'s own doc.
    pub(super) light_dir: [f32; 2],
    pub(super) surface_res: u32,
    pub(super) mass_floor: f32,
    /// Fallback slot used when `material_mass_enabled` is 0 -- unchanged
    /// v1 behavior for every existing caller.
    pub(super) material_slot: u32,
    /// N-material extension, single-phase path only (see `SurfaceParams`'s
    /// own doc for the mechanism). Was `_pad1: f32`, an unused pad field --
    /// same offset, same size, `SurfaceRenderParams` stays 48 bytes.
    pub(super) material_mass_enabled: u32,
    /// Real per-cell mass scale this scene's densities are expressed in, same
    /// value `BandHysteresisParams` already carries -- see `Renderer::
    /// set_grid_reference_cell_mass`. `fs_main` divides by it before
    /// quantizing depth into bands, which is what makes the band range a
    /// dimensionless "how many reference cell-masses deep is this" rather than
    /// an absolute mass that only happens to be right for one material.
    /// Was one half of `_pad2` -- same offset, same size, struct stays 48 bytes.
    pub(super) reference_cell_mass: f32,
    /// Floor on optical depth for edge color, in the SAME dimensionless band
    /// units as the quantizer above (see `Renderer::set_edge_reference_depth`).
    /// Was the other half of `_pad2`.
    pub(super) edge_reference_depth: f32,
}
const _: () = assert!(mem::size_of::<SurfaceRenderParams>() == 48);

/// Bundles `render_surface_reconstruction`'s buffer/scene args -- same real
/// precedent as `GridVolumeSource` above (a struct instead of a suppressed
/// argument-count lint).
pub struct SurfaceReconstructionSource<'a> {
    pub particle_buf: &'a wgpu::Buffer,
    pub particle_count: usize,
    /// The solver's own physics grid resolution -- the real auxiliary
    /// surface buffer is allocated at `grid_res * SURFACE_RES_MULTIPLIER`,
    /// finer than this, not equal to it (see `curvature_flow.wgsl`'s own
    /// doc for why that's the whole point of this render path).
    pub grid_res: u32,
    /// Which `OpticalTable` slot colors the whole reconstructed surface
    /// when `material_mass_enabled` is false -- real v1 fallback, see this
    /// method's own doc.
    pub material_slot: u32,
    /// N-material extension: when true, `fs_main` ignores `material_slot`
    /// and colors each pixel from its cell's own majority-mass material
    /// instead (same real technique as `grid_volume.wgsl`'s own
    /// `dominant_material`, see `curvature_flow.wgsl`'s doc). Opt-in --
    /// false costs nothing beyond a 4-byte placeholder buffer.
    pub material_mass_enabled: bool,
    /// Real simulation timestep (`SimConfig::dt`) -- feeds `splat_density_
    /// main`'s real velocity-stretch extension (see `curvature_flow.wgsl`'s
    /// own `SurfaceParams::dt` doc for the full physical grounding). Pass
    /// the same `dt` the `Simulation` this scene came from was constructed
    /// with.
    pub dt: f32,
}

/// Bundles `render_surface_reconstruction_dual_phase`'s args (see that
/// method's own doc, and `curvature_flow.wgsl`'s "two-phase extension" doc
/// for the real technique) -- two independently-smoothed surfaces sharing
/// one particle buffer, filtered by `material_id`. `material_id` doubles as
/// the `OpticalTable` color slot for its own phase, same real convention
/// `SurfaceReconstructionSource::material_slot` already uses.
pub struct DualPhaseSurfaceSource<'a> {
    pub particle_buf: &'a wgpu::Buffer,
    pub particle_count: usize,
    pub grid_res: u32,
    pub material_id_a: u32,
    pub material_id_b: u32,
    /// Real simulation timestep (`SimConfig::dt`) -- same real velocity-
    /// stretch extension as `SurfaceReconstructionSource::dt` (see that
    /// field's own doc), since this path shares the exact same
    /// `splat_density_main` compute shader.
    pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct OpticalTable {
    /// rgb = sigma_a, absorption coefficient (Beer-Lambert). .w = sigma_s, reduced
    /// scattering coefficient (single scalar, not per-channel -- real tissue
    /// scattering is much less wavelength-dependent than absorption in the visible
    /// range, Jacques 2013, a legitimate simplification for that reason).
    pub(super) slots: [[f32; 4]; 16],
    /// .x = specular Fresnel base reflectance R0 (Schlick 1994 approximation),
    /// rest padding. Real, cited, but bounded: this renderer has no surface-normal
    /// estimation (it tints particle instances, doesn't raytrace a reconstructed
    /// surface), so this is a constant near-normal-incidence reflectance, NOT a
    /// full view-angle-dependent Fresnel term -- honestly a simplification, not a
    /// claim of full BRDF accuracy.
    pub(super) specular: [[f32; 4]; 16],
}
const _: () = assert!(mem::size_of::<OpticalTable>() == 512);

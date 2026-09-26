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
    /// Full-cell mass used to form the dimensionless density ratio rho/rho0.
    pub(super) reference_cell_mass: f32,
    pub(super) _pad2: [f32; 2],
}
const _: () = assert!(mem::size_of::<GridVolumeParams>() == 48);

/// Bundles `render_gpu`'s own args -- same real precedent as `GridVolumeSource`
/// below and `spacetime::transfer::P2GParticleState` (a struct instead of a
/// suppressed argument-count lint). `interp_alpha` was the 8th argument that
/// tripped `clippy::too_many_arguments`; grouping it with the other
/// already-together-traveling per-call params fixes the root cause instead of
/// `#[allow]`ing it.
pub struct GpuRenderParams<'a> {
    pub particle_buf: &'a wgpu::Buffer,
    pub particle_count: usize,
    pub output_view: &'a wgpu::TextureView,
    pub clear: bool,
    /// See `RenderConfig::interp_alpha`'s own doc.
    pub interp_alpha: f32,
}

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
    pub(super) _pad: [f32; 2],
    pub(super) color: [f32; 4],
}
const _: () = assert!(mem::size_of::<InstanceData>() == 48);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraParams {
    pub(super) view_proj: [f32; 16],
    pub(super) particle_scale: f32,
    pub(super) round_particles: u32,
    pub(super) _pad: [f32; 2],
}
const _: () = assert!(mem::size_of::<CameraParams>() == 80);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RenderConfig {
    pub(super) mode: u32,
    pub(super) particle_count: u32,
    pub(super) vel_scale: f32,
    /// Render-time position blend factor for the "Fix Your Timestep" (Gaffer
    /// 2004) GPU interpolation path -- `mix(prev_positions[id], p.x,
    /// interp_alpha)` in `prep_instances.wgsl`. `1.0` (the CPU-path
    /// `basic_fluids.rs`/`snake_on_terrain.rs` convention when no snapshot has
    /// been taken) means "use the current position unblended," byte-identical
    /// to this field's former role as unused padding. Real fix for the
    /// "sudden acceleration" symptom root-caused 2026-09-15: every GPU demo
    /// already runs `FixedStepController` real-time-decoupled stepping but
    /// none interpolated the leftover fractional step, so uneven real
    /// per-step cost showed up as uneven position jumps on screen.
    pub(super) interp_alpha: f32,
}
const _: () = assert!(mem::size_of::<RenderConfig>() == 16);

/// `snapshot_positions.wgsl`'s own uniform -- just the active particle count,
/// so the extraction pass can bounds-check the same way `prep_instances.wgsl`
/// already does via `RenderConfig::particle_count` (a separate small uniform
/// rather than reusing that buffer: the snapshot runs BEFORE `render_gpu`
/// writes `RenderConfig` for the frame, so sharing it would create a fragile
/// ordering dependency for no real benefit -- 16 bytes is the standard
/// minimum-uniform-size convention already used by every buffer in this file).
/// `_pad` MUST stay three plain `u32`s, not `[u32; 3]` mirrored as WGSL
/// `vec3<u32>` -- confirmed via a real runtime wgpu validation panic ("size 16
/// where the shader expects 32"): `vec3` has a 16-byte alignment in WGSL's
/// uniform address space, silently inflating the true GPU-side struct size
/// past this exactly-16-byte Rust layout. See `snapshot_positions.wgsl`'s own
/// matching comment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct SnapshotConfig {
    pub(super) particle_count: u32,
    pub(super) _pad: [u32; 3],
}
const _: () = assert!(mem::size_of::<SnapshotConfig>() == 16);

/// One physical-scale/radiance contract shared by all `ByPhysics` GPU paths.
///
/// Every vector is four-wide to make the Rust/WGSL uniform layout explicit.
/// `spatial.z` is 1 only after a caller supplies a validated
/// `PhysicalRenderContract`; zero means legacy, dimensionless rendering.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PhysicalRenderParams {
    /// x = dx [m], y = out-of-plane view thickness [m], z = enabled flag.
    pub(super) spatial: [f32; 4],
    /// Linear RGB spectral-band radiance [W / (m^2 sr)].
    pub(super) incident_radiance: [f32; 4],
    /// Linear RGB spectral-band radiance [W / (m^2 sr)].
    pub(super) background_radiance: [f32; 4],
    /// Radiance mapped to linear display value 1; strictly positive RGB.
    pub(super) display_white_radiance: [f32; 4],
    /// Normalized world-space direction; w is padding.
    pub(super) camera_direction: [f32; 4],
    /// Normalized world-space direction; w is padding.
    pub(super) light_direction: [f32; 4],
    /// x = thermal-emission exposure anchor [K]: the temperature that renders
    /// at full brightness when no radiance contract is in force (see
    /// `Renderer::set_emission_reference_temperature`). 0 means unset, and
    /// `blackbody.inc.wgsl` substitutes its own documented default -- the
    /// zeroed buffer this struct starts as is therefore valid, not a bug.
    /// y/z/w reserved.
    pub(super) emission: [f32; 4],
}
const _: () = assert!(mem::size_of::<PhysicalRenderParams>() == 112);

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
    /// Thermal-emission exposure anchor [K]. This pass's bind group does not
    /// carry the shared `PhysicalRenderParams`, so the two numbers
    /// `blackbody.inc.wgsl` needs travel here instead. Was `_pad0`.
    pub(super) emission_reference_k: f32,
    /// Mean display-white radiance [W/(m^2 sr)], or 0 when no
    /// `PhysicalRenderContract` is in force. Was `_pad1`.
    pub(super) display_white_mean: f32,
    /// Volumetric luminous source `S` for the rendered material, `W/m^3` --
    /// see `MaterialModel::luminous_emission_w_m3`. 0 for anything that does
    /// not glow on its own, which is almost everything.
    pub(super) luminous_emission_w_m3: f32,
    /// Mass of one fully-occupied cell, so the source can be weighted by how
    /// much emitting matter a cell actually holds instead of glowing in
    /// empty space.
    pub(super) reference_cell_mass: f32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}
const _: () = assert!(mem::size_of::<LightDiffuseParams>() == 32);

/// Mirrors `curvature_flow.wgsl`'s `WaveStepParams` -- the real, persistent
/// (across frames) 2D wave-equation pass's own uniform. See that shader's
/// own "Pass 2b" doc for the real technique (same cited numerical scheme as
/// `energy::acoustics::WaveEquation2D`, reimplemented for this GPU-resident
/// buffer).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct WaveStepParams {
    pub(super) surface_res: u32,
    /// See `curvature_flow.wgsl`'s own `WaveStepParams` doc -- real, generic,
    /// derived from `MaterialModel::owns_deformation_volume_state()` by the
    /// caller, not a per-material-ID special case. 0.0 (default) = inert.
    pub(super) wave_force_coeff: f32,
    pub(super) _pad: [u32; 2],
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

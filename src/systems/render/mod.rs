/// emerge particle renderer -- physics-driven, no assets.
///
/// # Two rendering paths
///
/// **CPU path** (`render_slice`):
///   Builds `InstanceData` per particle on CPU, uploads via `write_buffer`.
///
/// **GPU path** (`render_gpu`):
///   Runs `prep_instances.wgsl` compute to fill instance buffer directly from
///   the particle storage buffer -- zero CPU readback, zero stall.
///   Pass `sim.particle_buffer()` + `sim.particle_count()`. No `sync_particles_blocking()`.
use std::mem;

use glam::{Mat2, Vec2};

use crate::particle::{Particle, Particles};
use crate::systems::gpu::MAX_RENDER_MATERIAL_SLOTS;

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const PREP_SHADER: &str = include_str!("shaders/prep_instances.wgsl");
const GRID_VOLUME_SHADER: &str = include_str!("shaders/grid_volume.wgsl");
const CURVATURE_FLOW_SHADER: &str = include_str!("shaders/curvature_flow.wgsl");
const PREP_WG: u32 = 64;
const SURFACE_CLEAR_WG: u32 = 64;
const SURFACE_SPLAT_WG: u32 = 64;
/// Finer-than-physics-grid resolution multiplier (see `curvature_flow.wgsl`'s
/// own top doc). Cost scales with the SQUARE of this constant --
/// `(N*grid_res)^2` f32 cells through every pass (splat, convert,
/// curvature-iterate, thermal diffusion, wave, visibility, band-hysteresis).
const SURFACE_RES_MULTIPLIER: u32 = 6;
/// Real, fixed EVEN iteration count -- keeping this even means the settled
/// result always lands in the SAME buffer (`surface_a`) regardless of N,
/// avoiding a runtime-conditional final bind group (see
/// `render_surface_reconstruction`'s own doc). 12 is within van der Laan et
/// al. 2009's own "several iterations per frame" range.
const CURVATURE_ITERATIONS: u32 = 12;
/// Number of flat depth bands `curvature_flow.wgsl` quantizes optical depth
/// into -- must stay equal to that file's own (file-scope) `DEPTH_BANDS`,
/// which as of 2026-08-14 is itself the single source of truth for
/// `fs_main`/`shade_phase`/`band_hysteresis_step_main` all agreeing with
/// each other; this Rust copy and `grid_volume.wgsl`'s own separate copy
/// are the two that still have to be kept in sync by hand (no cross-
/// language/cross-shader-module const sharing exists to do it for us).
const DEPTH_BANDS: f32 = 4.0;
/// Scalars stored per physics-grid cell by the Yu & Turk moments pass
/// (`[m0, m1x, m1y, m2xx, m2xy, m2yy]`) -- must match `MOMENTS_PER_CELL` in
/// `curvature_flow.wgsl`.
const MOMENTS_PER_CELL: u64 = 6;

/// `∫W² dA` for the 2D tensor-product quadratic B-spline this renderer splats
/// with, at unit width.
///
/// The 1D quadratic B-spline is `3/4 − x²` for `|x| < 1/2` and
/// `½(3/2 − |x|)²` for `1/2 ≤ |x| < 3/2`. Integrating its square gives
/// `0.45 + 0.10 = 0.55`, and the 2D tensor product squares that: `0.55² =
/// 0.3025`. Used only to derive [`Renderer::set_particle_spacing_cells`]'s
/// splat width -- see there for what it is for.
const BSPLINE_2D_SELF_OVERLAP: f32 = 0.3025;

/// Effective neighbour count a density estimate needs before its sampling
/// noise stops being visible.
///
/// A particle density field `ρ(x) = Σ mⱼ W(x − xⱼ)` fluctuates with where the
/// particles happen to fall, and the size of that fluctuation is set by how
/// many particles actually contribute to each evaluation. This is the standard
/// 2D SPH figure; the 3D literature's ~50 does not port, since neighbour count
/// grows with the dimension of the support.
const TARGET_EFFECTIVE_NEIGHBORS: f32 = 20.0;

/// Correction for kernel truncation at a boundary.
///
/// [`TARGET_EFFECTIVE_NEIGHBORS`] describes a COMPLETE neighbourhood, which a
/// particle only has in the interior. At a free surface or against a wall the
/// support is cut roughly in half -- there are no neighbours past the boundary
/// -- so the same kernel width delivers about half the effective samples, and
/// the estimate is correspondingly noisier exactly where the eye is drawn.
/// This is the standard SPH boundary-deficiency problem.
///
/// Sizing the kernel for the interior therefore under-serves the only region
/// that is actually visible: a rendered fluid IS its boundary. Budgeting for
/// the truncated case would cost `sqrt(2)` in width.
///
/// Tried live 2026-08-13 at `2.0` (giving `splat_width_cells ≈ 1.74`): the
/// interior-only derivation (this factor at `1.0`, width ≈ 1.23) had already
/// been confirmed to smooth the bulk with no reported instability; ADDING
/// this factor is what triggered a live-reported regression -- a gap opening
/// at the free surface, worsening as `surface_res_multiplier` was raised
/// further, on top of an already-raised value. Reverted to `1.0` (a no-op)
/// pending a proper visual diagnosis of that interaction; the boundary-
/// deficiency reasoning above may still be correct, but is not re-applied
/// blind after one regression already came from stacking it on other raised
/// sliders untested. See `set_particle_spacing_cells`'s own doc.
const BOUNDARY_TRUNCATION_FACTOR: f32 = 1.0;
/// Surface-reconstruction visibility/edge-alpha threshold, as a fraction of
/// [`Renderer::set_grid_reference_cell_mass`]. The B-spline splat here can
/// never concentrate more than `BSPLINE_CENTER_COEFF²` (0.5625) of one
/// particle's own mass into a single surface cell -- at `basic_fluids_gui`'s
/// calibration that peak is `0.025 * 0.5625 / 0.1 ≈ 0.14` of a full cell, so
/// the old `0.15` (copied from `render_grid_volume`'s unrelated calibration,
/// never re-derived for this path's finer units) made an isolated particle
/// structurally unable to ever cross the visibility hysteresis' 1.3x
/// turn-on threshold -- not a flicker, a guaranteed miss. `0.07` leaves
/// headroom on both sides of the ~0.14 ceiling.
const SURFACE_MASS_FLOOR_FRACTION: f32 = 0.07;
/// Default [`Renderer::set_edge_reference_depth`].
///
/// The shader's own doc gives a LOWER bound of one interior depth-band step
/// (`1.0 / DEPTH_BANDS`), and that bound was tried live on 2026-08-13. It is
/// correct as a bound and wrong as a default: unpinning optical depth this far
/// makes the per-particle density ripple drive colour directly, and the fluid
/// surface renders as blue speckle over white rather than a solid body. The
/// ripple is pre-existing -- a high floor was flattening it out of sight, not
/// preventing it -- and the real fix for it is the neighbourhood-fitted splat
/// kernel in `render::anisotropy`, not this knob.
///
/// So this stays high enough to keep shading flat until that lands, at which
/// point lowering it becomes worth retrying. Adjustable via
/// [`Renderer::set_edge_reference_depth`].
const DEFAULT_EDGE_REFERENCE_DEPTH: f32 = 3.0;
const _: () = assert!(
    CURVATURE_ITERATIONS.is_multiple_of(2),
    "CURVATURE_ITERATIONS must stay even so the result always settles in surface_a"
);

// ── Color mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    ByMaterial = 0,
    ByVelocity = 1,
    ByVolume = 2,
    ByPhysics = 3,
    ByThermal = 4,
    ByActivation = 5,
    /// Generic second scalar carrier (resource/grass level, pheromone, nutrients).
    /// See `Particle::scalar_field`'s own doc. Distinct wire value (6, not the next
    /// unused slot after ByActivation's implicit WGSL else-branch) so the GPU shader's
    /// existing fallback `else` can keep meaning ByActivation without renumbering it.
    ByScalarField = 6,
}

// GPU-side wire structs (InstanceData/CameraParams/RenderConfig/OpticalTable,
// GridVolumeParams/GridVolumeSource) live in gpu_types.rs -- see that file's doc.
mod gpu_types;
use gpu_types::{CameraParams, InstanceData, OpticalTable, RenderConfig};
pub use gpu_types::{DualPhaseSurfaceSource, GridVolumeSource, SurfaceReconstructionSource};

// GPU buffer allocation (the RenderBuffers struct + its own constructor)
// lives in buffers.rs -- see that file's doc.
mod buffers;
use buffers::RenderBuffers;

// Grid-native volumetric render path (`render_grid_volume`) lives in
// grid_volume.rs -- see that file's doc.
mod grid_volume;

// wgpu pipeline construction (the three build_*_pipeline functions + their
// bind-group-layout helpers) lives in pipelines.rs -- see that file's doc.
mod pipelines;

// Yu & Turk 2013 anisotropic surface kernels -- fits each particle's splat
// shape to its own neighborhood instead of to its (isotropic, for a fluid)
// deformation gradient. See that file's doc for why the F-driven source
// degenerates for liquids.
pub mod anisotropy;

// Curvature-flow surface reconstruction (render_surface_reconstruction[_dual_phase]
// + its own capacity helpers) lives in surface_reconstruction.rs -- see that file's doc.
mod surface_reconstruction;

// Real, general, opt-in window+event-loop runner (`DemoApp`/`run_demo`) --
// extracted from the winit/wgpu boilerplate every example was hand-
// duplicating. See that file's own doc for the full rationale (real,
// published capability, not example-only tooling).
pub mod demo_harness;

// Real, general, opt-in fading-polyline orbit/motion trail renderer -- see
// that file's own doc. Self-contained (own pipeline/camera/vertex buffer),
// not wired into `Renderer` itself, so any demo can add it independently.
pub mod trails;
use gpu_types::{
    BandHysteresisParams, LightDiffuseParams, SurfaceParams, SurfaceRenderParams, VisibilityParams,
    WaveStepParams,
};
use pipelines::{
    build_band_hysteresis_step_pipeline, build_grid_visibility_step_pipeline,
    build_grid_volume_pipeline, build_light_diffuse_pipeline, build_particle_pipeline,
    build_post_total_reduce_pipeline, build_prep_pipeline, build_surface_clear_pipeline,
    build_surface_convert_pipeline, build_surface_dual_render_pipeline,
    build_surface_iterate_pipeline, build_surface_moments_pipeline, build_surface_render_pipeline,
    build_surface_splat_pipeline, build_temp_avg_pipeline, build_temp_diffuse_pipeline,
    build_visibility_step_pipeline, build_volume_correct_pipeline, build_wave_step_pipeline,
};
pub use trails::TrailRenderer;

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer, // VERTEX | COPY_DST — drawn as per-instance attributes
    storage_instances: wgpu::Buffer, // STORAGE | COPY_SRC — compute write target (GPU path)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    max_particles: usize,

    prep_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    render_config_buf: wgpu::Buffer,
    optical_table_buf: wgpu::Buffer,

    grid_volume_pipeline: wgpu::RenderPipeline,
    grid_volume_bgl: wgpu::BindGroupLayout,
    grid_volume_params_buf: wgpu::Buffer,
    /// Real, persistent (across frames) hysteresis visible/invisible state
    /// for the grid-native render path -- the SAME technique as
    /// `visibility_buf` above (see `curvature_flow.wgsl`'s "Pass 2c" doc),
    /// ported to `grid_volume.wgsl`'s own `mass_floor` discard. Separate
    /// buffer/resolution tracking since this operates at `grid_res`, not
    /// `surface_res`.
    grid_visibility_step_pipeline: wgpu::ComputePipeline,
    grid_visibility_step_bgl: wgpu::BindGroupLayout,
    grid_visibility_buf: wgpu::Buffer,
    grid_visibility_params_buf: wgpu::Buffer,
    /// Resolution `grid_visibility_buf` is currently allocated at --
    /// `ensure_grid_visibility_capacity` regrows it when a caller's
    /// `grid_res` exceeds this, same lazy-growth convention as `surface_res`.
    grid_visibility_res: u32,
    /// Cached ortho projection + grid_res (set by `set_camera`) -- lets
    /// `render_grid_volume` take just (device, queue, grid_buf, material_mass_buf,
    /// view, clear) instead of repeating width/height/grid_res, keeping it under
    /// clippy's argument-count lint.
    cached_ortho: (f32, f32, f32, f32),
    cached_grid_res: u32,
    /// Real, cached glow strength -- see `set_glow_strength`'s own doc.
    /// Needed here because `set_camera_centered` re-writes the WHOLE
    /// `CameraParams` buffer (including this field) on every resize/pan/
    /// zoom, so it must remember the last value `set_glow_strength` set,
    /// not silently reset it to 0.0 on the next camera update.
    cached_glow_strength: f32,
    /// Real, cached point-light position + shading strength for the
    /// round-particle billboard-sphere Lambertian shading -- see
    /// `set_light_source`'s own doc. Same "must survive `set_camera_
    /// centered`'s whole-buffer rewrite" reason as `cached_glow_strength`.
    cached_light_pos: Vec2,
    cached_shading_strength: f32,
    cached_light_reference_distance: f32,
    /// When true, `render`/`render_slice` draw every particle as an
    /// undeformed disc (identity in place of `p.deformation_gradient`)
    /// instead of the real per-particle quad deformation -- see
    /// `set_rigid_render`'s own doc for why. Default `false`: byte-
    /// identical to every existing caller.
    rigid_render: bool,
    /// Real light direction for `render_grid_volume`/`render_surface_
    /// reconstruction`(`_dual_phase`)'s Lambertian + specular shading -- set
    /// via `set_light_dir`, defaults to the same value those shaders used
    /// to hardcode directly (so behavior is unchanged until a caller
    /// explicitly wires in a real value, e.g. `SimConfig::light_dir`).
    light_dir: (f32, f32),

    /// Mass of ONE fully-occupied grid cell in the caller's own units, used to
    /// scale the grid-volume/surface density thresholds. Defaults to `1.0`,
    /// which reproduces the previous hardcoded absolute thresholds exactly.
    ///
    /// Real bug this exists to fix (2026-08-13): those thresholds (e.g.
    /// `grid_volume.rs`'s `mass_floor = 0.15`) were absolute constants written
    /// against a scene whose "per occupied cell mass is order 0.5-4" (that
    /// file's own comment). A physically-calibrated scene has no reason to land
    /// in that range -- `basic_fluids_gui.rs` uses the REAL water density
    /// `rho0 = 1000 kg/m^3 * dx^2 = 0.1` grid units, so a *completely full*
    /// cell weighs 0.1, i.e. LESS than the 0.15 floor. Every cell was therefore
    /// discarded and the fluid rendered as near-empty, while the raw-particle
    /// mode showed it perfectly -- the three render modes visibly disagreeing
    /// about where the fluid was.
    ///
    /// Callers should pass their fluid's `rest_density * cell_area` (in grid
    /// units, cell_area = 1). Thresholds then mean "this fraction of a full
    /// cell", which is scale-free and correct for any density calibration.
    grid_reference_cell_mass: f32,

    /// Curvature-flow smoothing passes per frame, defaulting to
    /// `CURVATURE_ITERATIONS`. Settable via `set_curvature_iterations` so a
    /// caller can trade surface smoothness against cost instead of being stuck
    /// with one compile-time value: van der Laan et al. 2009 only prescribes
    /// "several iterations per frame", and the right number genuinely depends
    /// on the scene -- a thin sheet of fast water wants more smoothing than a
    /// settled pool, and a perf-bound scene wants fewer.
    ///
    /// Always kept EVEN (see `CURVATURE_ITERATIONS`' own assert): the passes
    /// ping-pong between `surface_a`/`surface_b`, so an even count is what
    /// guarantees the settled result lands back in `surface_a` and the final
    /// bind group can stay compile-time constant.
    curvature_iterations: u32,

    /// Surface-grid fineness relative to the physics grid, defaulting to
    /// `SURFACE_RES_MULTIPLIER`. Settable via `set_surface_res_multiplier`.
    ///
    /// This is the single biggest quality/cost dial in the surface path and it
    /// was a compile-time constant: the reconstruction runs at
    /// `grid_res * multiplier` and cost scales with the SQUARE of it, through
    /// every pass (splat, convert, each curvature iteration, thermal, wave,
    /// visibility, band hysteresis). At the default 6 with a 64-cell physics
    /// grid that is a 384x384 field -- ~147k cells per pass. Halving it to 3
    /// quarters that; raising it sharpens the surface at 4x the cost per step
    /// up. The right value genuinely depends on the scene and the perf budget,
    /// which is exactly why it should not be baked in.
    surface_res_multiplier: u32,

    /// Width of ONE particle's surface splat, in physics-grid cells.
    /// Defaults to `1.0` = the full MPM quadratic B-spline support (1.5
    /// cells radius), which is exactly the previous behaviour.
    ///
    /// Deliberately material-AGNOSTIC: this is a property of how densely a
    /// scene samples its material with particles, not of what the material
    /// is. Sand, snow, a plant's tissue and water all splat the same way --
    /// any of them looks like overlapping metaballs if each particle is drawn
    /// much wider than its own spacing, and looks speckled/holey if drawn much
    /// narrower. The MPM B-spline width is the right kernel for scattering
    /// MASS TO THE GRID; it is not automatically the right width for DRAWING
    /// a particle, and the two only coincide at one particular spacing.
    ///
    /// A good starting value is the scene's particle spacing in cells (so each
    /// particle covers roughly the patch it actually represents). Cost falls
    /// as the SQUARE of it -- the splat loop is O(radius^2).
    splat_width_cells: f32,

    /// Floor on optical depth for edge color, in the DIMENSIONLESS band units
    /// `fs_main`/`shade_phase` quantize into (i.e. multiples of
    /// `grid_reference_cell_mass`, not an absolute mass).
    ///
    /// It exists so a thin edge still reads as solid material colour instead
    /// of a washed-out grey ring: `exp(-sigma_a * depth)` barely absorbs
    /// anything at near-zero depth for the small `sigma_a` most materials
    /// have. Set too HIGH it does the opposite damage -- it becomes the `max`
    /// for nearly every pixel, pinning optical depth to a constant so the
    /// whole surface shades flat with no depth variation at all.
    ///
    /// Default is the shader's own stated minimum, one interior depth-band
    /// step (`1.0 / DEPTH_BANDS`), rather than a value picked by eye.
    edge_reference_depth: f32,

    /// Strength of the Yu & Turk neighbourhood-fitted splat anisotropy.
    /// 0.0 skips the fit entirely (previous behaviour, bit for bit), 1.0 is
    /// the full fitted shape, in between blends toward isotropic. See the
    /// `render::anisotropy` module doc.
    anisotropy_strength: f32,

    // ── Curvature-flow surface reconstruction (see curvature_flow.wgsl) ────
    surface_clear_pipeline: wgpu::ComputePipeline,
    surface_clear_bgl: wgpu::BindGroupLayout,
    surface_splat_pipeline: wgpu::ComputePipeline,
    /// Yu & Turk moments scatter -- runs immediately before the splat pass and
    /// shares its bind group layout entirely.
    surface_moments_pipeline: wgpu::ComputePipeline,
    surface_splat_bgl: wgpu::BindGroupLayout,
    surface_convert_pipeline: wgpu::ComputePipeline,
    surface_convert_bgl: wgpu::BindGroupLayout,
    surface_iterate_pipeline: wgpu::ComputePipeline,
    surface_iterate_bgl: wgpu::BindGroupLayout,
    surface_render_pipeline: wgpu::RenderPipeline,
    surface_render_bgl: wgpu::BindGroupLayout,
    /// Fixed-point atomic splat target (see `curvature_flow.wgsl`'s own
    /// `clear_surface_main`/`splat_density_main` doc).
    surface_atomic_buf: wgpu::Buffer,
    /// Real mass-weighted temperature: fixed-point atomic scatter target +
    /// converted plain-f32 buffer `fs_main` samples for blackbody emission
    /// -- see `surface_temp_atomic`'s own doc in the shader. Single-phase
    /// only (grown/reused by phase A of the dual-phase path too, same as
    /// `surface_atomic_buf` itself -- see that field's own sharing note
    /// below); phase B gets its own separate pair.
    surface_temp_atomic_buf: wgpu::Buffer,
    surface_temp_float_buf: wgpu::Buffer,
    /// Real volume-preserving correction (`curvature_flow.wgsl`'s "Pass
    /// 1d"): `pre_total_atomic_buf` accumulates the TRUE ground-truth
    /// particle mass during splat; `post_total_atomic_buf` sums the settled
    /// (post-curvature-flow) density; `volume_correct_pipeline` rescales
    /// the settled result by their ratio. Both are single-element (4-byte)
    /// buffers, NEVER grown with `surface_res` (a global scalar per phase,
    /// not a per-cell field).
    post_total_reduce_pipeline: wgpu::ComputePipeline,
    post_total_reduce_bgl: wgpu::BindGroupLayout,
    volume_correct_pipeline: wgpu::ComputePipeline,
    volume_correct_bgl: wgpu::BindGroupLayout,
    pre_total_atomic_buf: wgpu::Buffer,
    post_total_atomic_buf: wgpu::Buffer,
    /// Real 2D thermal-diffusion PDE on the recovered temperature field
    /// (`curvature_flow.wgsl`'s "Pass 1c") -- `temp_avg_pipeline` divides by
    /// final settled density once, `temp_diffuse_pipeline` then runs one
    /// real heat-equation step. `surface_temp_b_buf` is the ping-pong
    /// partner: avg writes into it, diffuse reads it and writes the result
    /// back into `surface_temp_float_buf` (the buffer `fs_main` already
    /// reads), so no render-bind-group change was needed for this addition.
    temp_avg_pipeline: wgpu::ComputePipeline,
    temp_avg_bgl: wgpu::BindGroupLayout,
    temp_diffuse_pipeline: wgpu::ComputePipeline,
    temp_diffuse_bgl: wgpu::BindGroupLayout,
    surface_temp_b_buf: wgpu::Buffer,
    /// Real diffusion approximation to light transport (`curvature_flow.
    /// wgsl`'s "Pass 1e") -- see that entry point's own doc for the full
    /// real derivation. `light_phi_bufs` are the two ping-pong buffers (see
    /// their own doc in `buffers.rs`); `light_frame_index` alternates which
    /// one is "current" (read) vs "next" (write) each frame, the same real
    /// role `wave_frame_index` plays for the (second-order) wave field,
    /// just a simpler 2-way rotation for this first-order equation.
    light_diffuse_pipeline: wgpu::ComputePipeline,
    light_diffuse_bgl: wgpu::BindGroupLayout,
    light_phi_bufs: [wgpu::Buffer; 2],
    light_diffuse_params_buf: wgpu::Buffer,
    light_frame_index: u32,
    /// Ping-pong plain-f32 buffers. `CURVATURE_ITERATIONS` (even) means the
    /// settled result always lands in `surface_a` -- `render_surface_
    /// reconstruction` only ever reads that one, never `surface_b` directly.
    surface_a_buf: wgpu::Buffer,
    surface_b_buf: wgpu::Buffer,
    surface_params_buf: wgpu::Buffer,
    surface_render_params_buf: wgpu::Buffer,
    /// Resolution the surface buffers were allocated at (`grid_res *
    /// SURFACE_RES_MULTIPLIER`) -- `ensure_surface_capacity` reallocates all
    /// three buffers when a caller's `grid_res` needs a bigger one.
    surface_res: u32,

    /// N-material extension (see `curvature_flow.wgsl`'s own doc),
    /// single-phase path only: flat `surface_res² × MAX_RENDER_MATERIAL_
    /// SLOTS` per-cell mass array, same real technique as `grid_volume.
    /// wgsl`'s own `material_mass`. Grown LAZILY by `ensure_surface_
    /// material_mass_capacity`, called only when a caller opts in
    /// (`SurfaceReconstructionSource::material_mass_enabled`) -- unlike
    /// every other surface buffer above, NOT grown unconditionally by
    /// `ensure_surface_capacity`, since it's 16x the size of a single
    /// per-cell field and most callers never use it (real cost accounting
    /// in the plan this shipped from).
    surface_material_mass_buf: wgpu::Buffer,
    /// Yu & Turk neighbourhood moments, 6 scalars per PHYSICS-grid cell (not
    /// per surface cell) -- see `surface_moments_atomic`'s own shader doc.
    surface_moments_buf: wgpu::Buffer,
    /// Physics-grid resolution `surface_moments_buf` is currently sized for.
    surface_moments_res: u32,
    /// Resolution `surface_material_mass_buf` is currently sized for.
    /// Independent of `surface_res` itself -- 0 means still the
    /// constructor's 4-byte placeholder, never opted into.
    surface_material_mass_res: u32,

    /// Real, persistent (across frames, unlike everything else in this
    /// section) 2D wave-equation height field -- see `curvature_flow.wgsl`'s
    /// own "Pass 2b" doc. THREE physical buffers, not two: wgpu's usage-
    /// scope validator rejects binding the SAME buffer as both read-only
    /// and read_write within one dispatch, even when the actual access
    /// pattern is index-disjoint and logically hazard-free (confirmed via a
    /// real validation error when a 2-buffer aliasing scheme was tried
    /// first) -- a genuine leapfrog integrator only ever needs u(t) and
    /// u(t-dt) to read, but writing u(t+dt) needs a THIRD distinct slot to
    /// satisfy wgpu's conservative rule. `wave_frame_index` rotates which
    /// of the 3 buffers plays which of the 3 roles (current/previous/next)
    /// each call -- see that method's own doc for the exact rotation.
    wave_step_pipeline: wgpu::ComputePipeline,
    wave_step_bgl: wgpu::BindGroupLayout,
    wave_bufs: [wgpu::Buffer; 3],
    wave_params_buf: wgpu::Buffer,
    wave_frame_index: u32,
    /// Last frame's settled density (a copy of `surface_a_buf` taken right
    /// after each frame's wave step reads it), so `wave_step_main` can
    /// excite from a genuine TEMPORAL disturbance rather than a permanent
    /// spatial-edge artifact.
    wave_density_prev_buf: wgpu::Buffer,

    /// Whether `wave_density_prev_buf` has been seeded from a real settled
    /// density yet. False right after `Renderer::new`/a capacity grow (the
    /// buffer is a zero placeholder then) -- reading it as "previous
    /// density" on that first frame would read density appearing from
    /// nothing as false motion. Seeded (prev := now, force = 0) before the
    /// first `wave_step` dispatch instead; see that call site's own doc.
    wave_prev_seeded: bool,
    /// Phase B's own `wave_prev_seeded` -- see its own doc.
    phase_b_wave_prev_seeded: bool,

    /// Real, persistent (across frames) hysteresis visible/invisible state
    /// -- see `curvature_flow.wgsl`'s own "Pass 2c" doc. A SINGLE buffer
    /// (no rotation needed, unlike the wave field: this pass only ever
    /// reads/writes its OWN index, no neighbor stencil, so there's no
    /// wgpu usage-scope conflict to avoid).
    visibility_step_pipeline: wgpu::ComputePipeline,
    visibility_step_bgl: wgpu::BindGroupLayout,
    visibility_buf: wgpu::Buffer,
    visibility_params_buf: wgpu::Buffer,

    /// Real, persistent hysteresis color-band state -- see `curvature_
    /// flow.wgsl`'s own "Pass 2d" doc. Same single-buffer, no-rotation
    /// shape as `visibility_buf` (self-index only, downstream/one-way of
    /// the density field, never fed back into it).
    band_hysteresis_step_pipeline: wgpu::ComputePipeline,
    band_hysteresis_step_bgl: wgpu::BindGroupLayout,
    band_state_buf: wgpu::Buffer,
    band_hysteresis_params_buf: wgpu::Buffer,

    /// Real, persistent RAW splat density history for neighborhood-
    /// clamped temporal smoothing -- see `convert_atomic_to_float_main`'s
    /// own doc (Lottes 2011 / Karis 2014 TAA technique, adapted to a
    /// scalar field).
    raw_splat_history_buf: wgpu::Buffer,

    // ── Two-phase extension (see curvature_flow.wgsl's own doc) ────────────
    /// A second, fully independent set of splat/ping-pong buffers for
    /// "phase B" -- `render_surface_reconstruction_dual_phase` runs the
    /// SAME clear/splat/convert/iterate pipelines twice, once into phase A's
    /// existing `surface_*_buf` fields above and once into these, so each
    /// phase gets its own real, independently-smoothed surface (see the
    /// shader's own VOF/phase-fraction citation for why that's the correct
    /// choice, not a shared blended field).
    phase_b_atomic_buf: wgpu::Buffer,
    /// Phase B's own temperature atomic/float pair -- see `surface_temp_
    /// atomic_buf`'s own doc. Written every dual-phase frame for symmetry
    /// with phase A's scatter, but NOT read by `fs_main_dual_phase` (no
    /// binding for it there -- see that entry point's own doc for why).
    phase_b_temp_atomic_buf: wgpu::Buffer,
    phase_b_temp_float_buf: wgpu::Buffer,
    /// Phase B's own volume-preserving-correction totals -- see
    /// `pre_total_atomic_buf`'s own doc.
    phase_b_pre_total_atomic_buf: wgpu::Buffer,
    phase_b_post_total_atomic_buf: wgpu::Buffer,
    phase_b_a_buf: wgpu::Buffer,
    phase_b_b_buf: wgpu::Buffer,
    /// Phase B's own raw-splat history -- `surface_convert_bgl` now needs
    /// this 4th binding on EVERY caller.
    phase_b_raw_splat_history_buf: wgpu::Buffer,
    phase_b_params_buf: wgpu::Buffer,
    render_params_b_buf: wgpu::Buffer,
    /// Phase B's own wave/visibility/band state -- SAME real techniques as
    /// `wave_bufs`/`visibility_buf`/`band_state_buf` above (see
    /// `curvature_flow.wgsl`'s Pass 3b doc), just a second independent set
    /// since phase B is a fully separate density field. Phase A reuses the
    /// single-phase fields directly (`render_surface_reconstruction` and
    /// `render_surface_reconstruction_dual_phase` are never called the same
    /// frame, so sharing is safe) -- only phase B needs its own buffers.
    /// `wave_params_buf`/`visibility_params_buf`/`band_hysteresis_params_buf`
    /// are ALSO shared across both phases: they only carry `surface_res`/
    /// `mass_floor`, identical for both phases every frame.
    phase_b_wave_bufs: [wgpu::Buffer; 3],
    /// Phase B's own previous-density copy -- see `wave_density_prev_buf`'s
    /// own doc.
    phase_b_wave_density_prev_buf: wgpu::Buffer,
    phase_b_visibility_buf: wgpu::Buffer,
    phase_b_band_state_buf: wgpu::Buffer,
    surface_dual_render_pipeline: wgpu::RenderPipeline,
    surface_dual_render_bgl: wgpu::BindGroupLayout,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    sigma_a: [[f32; 3]; 16],
    /// Reduced scattering coefficient per material slot (single scalar -- see
    /// `OpticalTable`'s own doc for why this isn't per-channel).
    sigma_s: [f32; 16],
    /// Specular Fresnel base reflectance R0 per material slot (see `OpticalTable`'s
    /// own doc for the real-but-bounded caveat).
    specular_r0: [f32; 16],
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        max_particles: usize,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let cap = max_particles.max(1);

        let RenderBuffers {
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            render_config_buf,
            optical_table_buf,
            grid_volume_params_buf,
            grid_visibility_buf,
            grid_visibility_params_buf,
            surface_atomic_buf,
            surface_temp_atomic_buf,
            surface_temp_float_buf,
            pre_total_atomic_buf,
            post_total_atomic_buf,
            surface_temp_b_buf,
            surface_a_buf,
            surface_b_buf,
            surface_params_buf,
            surface_render_params_buf,
            surface_material_mass_buf,
            surface_moments_buf,
            phase_b_atomic_buf,
            phase_b_temp_atomic_buf,
            phase_b_temp_float_buf,
            phase_b_pre_total_atomic_buf,
            phase_b_post_total_atomic_buf,
            phase_b_a_buf,
            phase_b_b_buf,
            phase_b_raw_splat_history_buf,
            phase_b_params_buf,
            render_params_b_buf,
            phase_b_wave_bufs,
            phase_b_wave_density_prev_buf,
            phase_b_visibility_buf,
            phase_b_band_state_buf,
            wave_bufs,
            wave_params_buf,
            wave_density_prev_buf,
            visibility_buf,
            visibility_params_buf,
            band_state_buf,
            band_hysteresis_params_buf,
            raw_splat_history_buf,
            light_phi_bufs,
            light_diffuse_params_buf,
        } = RenderBuffers::new(device, cap);

        let (render_pipeline, render_bgl) = build_particle_pipeline(device, output_format);
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let (prep_pipeline, prep_bgl) = build_prep_pipeline(device);
        let (grid_volume_pipeline, grid_volume_bgl) =
            build_grid_volume_pipeline(device, output_format);
        let (grid_visibility_step_pipeline, grid_visibility_step_bgl) =
            build_grid_visibility_step_pipeline(device);
        let (surface_clear_pipeline, surface_clear_bgl) = build_surface_clear_pipeline(device);
        let (surface_splat_pipeline, surface_splat_bgl) = build_surface_splat_pipeline(device);
        let surface_moments_pipeline = build_surface_moments_pipeline(device, &surface_splat_bgl);
        let (surface_convert_pipeline, surface_convert_bgl) =
            build_surface_convert_pipeline(device);
        let (surface_iterate_pipeline, surface_iterate_bgl) =
            build_surface_iterate_pipeline(device);
        let (surface_render_pipeline, surface_render_bgl) =
            build_surface_render_pipeline(device, output_format);
        let (surface_dual_render_pipeline, surface_dual_render_bgl) =
            build_surface_dual_render_pipeline(device, output_format);
        let (wave_step_pipeline, wave_step_bgl) = build_wave_step_pipeline(device);
        let (visibility_step_pipeline, visibility_step_bgl) =
            build_visibility_step_pipeline(device);
        let (band_hysteresis_step_pipeline, band_hysteresis_step_bgl) =
            build_band_hysteresis_step_pipeline(device);
        let (temp_avg_pipeline, temp_avg_bgl) = build_temp_avg_pipeline(device);
        let (temp_diffuse_pipeline, temp_diffuse_bgl) = build_temp_diffuse_pipeline(device);
        let (light_diffuse_pipeline, light_diffuse_bgl) = build_light_diffuse_pipeline(device);
        let (post_total_reduce_pipeline, post_total_reduce_bgl) =
            build_post_total_reduce_pipeline(device);
        let (volume_correct_pipeline, volume_correct_bgl) = build_volume_correct_pipeline(device);

        Self {
            render_pipeline,
            render_bind_group,
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            max_particles: cap,
            prep_pipeline,
            prep_bgl,
            render_config_buf,
            grid_volume_pipeline,
            grid_volume_bgl,
            grid_volume_params_buf,
            grid_visibility_step_pipeline,
            grid_visibility_step_bgl,
            grid_visibility_buf,
            grid_visibility_params_buf,
            grid_visibility_res: 1,
            cached_ortho: (1.0, 0.0, 1.0, 0.0),
            cached_grid_res: 1,
            cached_glow_strength: 0.0,
            cached_light_pos: Vec2::ZERO,
            cached_shading_strength: 0.0,
            cached_light_reference_distance: 0.0,
            rigid_render: false,
            light_dir: (-0.5, 0.7),
            grid_reference_cell_mass: 1.0,
            curvature_iterations: CURVATURE_ITERATIONS,
            surface_res_multiplier: SURFACE_RES_MULTIPLIER,
            splat_width_cells: 1.0,
            edge_reference_depth: DEFAULT_EDGE_REFERENCE_DEPTH,
            // Off by default: `> 0.0` is the sole gate on both the moments
            // scatter dispatch (`surface_reconstruction.rs`) and the fit
            // itself (`splat_density_main`), so `0.0` makes the whole Yu &
            // Turk path -- moments buffer, scatter, gather, eigendecompose --
            // fully inert, byte-identical to before it existed.
            //
            // Re-enabled 2026-08-14 at full strength, ALONE -- the real
            // white-noise cause (GRAD_EPSILON instability in the later
            // curvature-flow smoothing pass, see that constant's own doc) is
            // now fixed and tested, and this is a single, isolated variable:
            // `BOUNDARY_TRUNCATION_FACTOR` stays at its reverted `1.0`
            // no-op, and `set_particle_spacing_cells` stays un-wired in the
            // demo -- neither of the two things that were stacked together
            // when this last regressed live is reintroduced here. This is
            // also independently named, not just by this session's own
            // Yu & Turk work: Sebastian Lague's "Coding Adventure: Rendering
            // Fluids" names anisotropic particle rendering as the direct fix
            // for the same "bubbly"/blobby surface symptom seen here, from a
            // fully separate screen-space fluid renderer.
            anisotropy_strength: 1.0,
            surface_clear_pipeline,
            surface_clear_bgl,
            surface_splat_pipeline,
            surface_moments_pipeline,
            surface_splat_bgl,
            surface_convert_pipeline,
            surface_convert_bgl,
            surface_iterate_pipeline,
            surface_iterate_bgl,
            surface_render_pipeline,
            surface_render_bgl,
            surface_atomic_buf,
            surface_temp_atomic_buf,
            surface_temp_float_buf,
            post_total_reduce_pipeline,
            post_total_reduce_bgl,
            volume_correct_pipeline,
            volume_correct_bgl,
            pre_total_atomic_buf,
            post_total_atomic_buf,
            temp_avg_pipeline,
            temp_avg_bgl,
            temp_diffuse_pipeline,
            temp_diffuse_bgl,
            light_diffuse_pipeline,
            light_diffuse_bgl,
            light_phi_bufs,
            light_diffuse_params_buf,
            light_frame_index: 0,
            surface_temp_b_buf,
            surface_a_buf,
            surface_b_buf,
            surface_params_buf,
            surface_render_params_buf,
            surface_res: 1,
            surface_material_mass_buf,
            surface_moments_buf,
            surface_material_mass_res: 0,
            surface_moments_res: 0,
            wave_step_pipeline,
            wave_step_bgl,
            wave_bufs,
            wave_params_buf,
            wave_frame_index: 0,
            wave_density_prev_buf,
            wave_prev_seeded: false,
            visibility_step_pipeline,
            visibility_step_bgl,
            visibility_buf,
            visibility_params_buf,
            band_hysteresis_step_pipeline,
            band_hysteresis_step_bgl,
            band_state_buf,
            band_hysteresis_params_buf,
            raw_splat_history_buf,
            phase_b_atomic_buf,
            phase_b_temp_atomic_buf,
            phase_b_temp_float_buf,
            phase_b_pre_total_atomic_buf,
            phase_b_post_total_atomic_buf,
            phase_b_a_buf,
            phase_b_b_buf,
            phase_b_raw_splat_history_buf,
            phase_b_params_buf,
            render_params_b_buf,
            phase_b_wave_bufs,
            phase_b_wave_density_prev_buf,
            phase_b_wave_prev_seeded: false,
            phase_b_visibility_buf,
            phase_b_band_state_buf,
            surface_dual_render_pipeline,
            surface_dual_render_bgl,
            optical_table_buf,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            sigma_a: [[0.3f32; 3]; 16],
            sigma_s: [0.0f32; 16],
            specular_r0: [0.0f32; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// Call at init and on every resize. Centers the view on the grid's own
    /// center `(grid_res/2, grid_res/2)` -- thin wrapper over
    /// `set_camera_centered` for every existing caller that never needed to
    /// pan. Unchanged behavior, byte-identical to before this existed.
    pub fn set_camera(
        &mut self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let gr = grid_res as f32;
        self.set_camera_centered(
            queue,
            grid_res,
            width,
            height,
            particle_scale,
            round_particles,
            Vec2::splat(gr * 0.5),
        );
    }

    /// Real generalization of `set_camera`: same letterboxed orthographic
    /// projection, but centered on an arbitrary `center` (grid coords)
    /// instead of always the grid's own center -- the real hook a caller
    /// needs for cursor-drag pan (compute a grid-space delta via
    /// `screen_to_grid`, accumulate it into `center` across frames).
    #[allow(clippy::too_many_arguments)]
    pub fn set_camera_centered(
        &mut self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
        center: Vec2,
    ) {
        let gr = grid_res as f32;
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let (sx, sy) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), 2.0 / gr)
        } else {
            (2.0 / gr, 2.0 * aspect / gr)
        };
        let (tx, ty) = (-sx * center.x, -sy * center.y);
        self.cached_ortho = (sx, tx, sy, ty);
        self.cached_grid_res = grid_res;
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                glow_strength: self.cached_glow_strength,
                light_pos: self.cached_light_pos.to_array(),
                shading_strength: self.cached_shading_strength,
                light_reference_distance: self.cached_light_reference_distance,
                _pad: [0.0; 1],
            }),
        );
    }

    /// Real, general, opt-in soft-glow strength for round particles -- see
    /// `render_particles.wgsl`'s own fragment-shader doc for the falloff
    /// formula. `0.0` (never calling this) is byte-identical to every
    /// existing caller's current look. Deliberately a SEPARATE, focused
    /// setter rather than another `set_camera_centered` parameter: glow is
    /// independent of position/zoom, so bundling it in would force every
    /// resize/pan/zoom call to also know the current glow value. A direct
    /// partial buffer write at `glow_strength`'s own byte offset (72) --
    /// cheap, doesn't need the full camera math re-run.
    pub fn set_glow_strength(&mut self, queue: &wgpu::Queue, glow_strength: f32) {
        self.cached_glow_strength = glow_strength;
        const GLOW_STRENGTH_BYTE_OFFSET: u64 = 72;
        queue.write_buffer(
            &self.camera_buffer,
            GLOW_STRENGTH_BYTE_OFFSET,
            bytemuck::bytes_of(&glow_strength),
        );
    }

    /// Real, general, opt-in billboard-sphere Lambertian shading for round
    /// particles -- see `render_particles.wgsl`'s own fragment-shader doc
    /// for the technique (hemisphere-normal reconstruction + Lambert's
    /// cosine law against the real direction to `position`, modulated by a
    /// real inverse-square intensity falloff). Pass the scene's own real
    /// light-source position (grid coords) each frame it moves -- e.g. a
    /// star's actual, physically-computed position, not a fixed art-
    /// direction light.
    ///
    /// `reference_distance` (grid units) is the distance at which a
    /// particle reads at intensity 1.0 -- pick a real, physically
    /// meaningful value for the scene (e.g. a planet's own real orbital
    /// distance from its star, matching the literal definition of "solar
    /// constant"), not an arbitrary tuning number. `0.0` disables the
    /// inverse-square falloff (direction-only shading, uniform intensity).
    ///
    /// `strength=0.0` (never calling this) is byte-identical to every
    /// existing caller's current flat-disc look; same partial-buffer-write
    /// pattern as `set_glow_strength`, for the same reason (shading is
    /// independent of camera position/zoom).
    pub fn set_light_source(
        &mut self,
        queue: &wgpu::Queue,
        position: Vec2,
        strength: f32,
        reference_distance: f32,
    ) {
        self.cached_light_pos = position;
        self.cached_shading_strength = strength;
        self.cached_light_reference_distance = reference_distance;
        const LIGHT_POS_BYTE_OFFSET: u64 = 76;
        queue.write_buffer(
            &self.camera_buffer,
            LIGHT_POS_BYTE_OFFSET,
            bytemuck::bytes_of(&[position.x, position.y, strength, reference_distance]),
        );
    }

    /// Real, general, opt-in: when a particle's `deformation_gradient`
    /// carries no real physical meaning for what it represents -- e.g. one
    /// MPM particle standing in for an entire rigid/point-mass body (a
    /// planet in an N-body scene) rather than a differential element of a
    /// deforming continuum -- its own local velocity-gradient-driven F
    /// still evolves every substep (ordinary MPM/APIC mechanics don't know
    /// the particle is "supposed to" stay rigid), and visualizing that F
    /// distorts what should read as a plain point/disc into a spuriously
    /// stretched or sheared quad. This does not indicate a physics bug in
    /// the body's real motion (governed by real N-body gravity, verified
    /// independently via conservation laws) -- it is `render_particles.
    /// wgsl`'s own quad-deformation feature applied somewhere it isn't
    /// meaningful.
    ///
    /// `true` renders every particle in this `Renderer` as an undeformed
    /// disc/quad (identity in place of F) -- correct for point-mass/rigid-
    /// body demos, wrong for a real deforming continuum (jelly, sand,
    /// fluid), where the deformation IS the real, meaningful signal. Default
    /// `false` is byte-identical to every existing caller.
    pub fn set_rigid_render(&mut self, rigid: bool) {
        self.rigid_render = rigid;
    }

    /// Exact inverse of `set_camera`'s own NDC projection -- the single
    /// source of truth for screen-pixel <-> grid-coordinate conversion, so a
    /// caller's cursor/click handling can't drift out of sync with what
    /// actually rendered. `set_camera` letterboxes/pillarboxes a non-square
    /// window to preserve the grid's aspect ratio, so a naive `screen_pos /
    /// window_size * grid_res` (what this replaces) silently offsets on any
    /// non-square window. `width`/`height` must match whatever was last
    /// passed to `set_camera`.
    pub fn screen_to_grid(
        &self,
        screen_x: f32,
        screen_y: f32,
        width: u32,
        height: u32,
    ) -> (f32, f32) {
        let ndc_x = (screen_x / width.max(1) as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / height.max(1) as f32) * 2.0;
        let (sx, tx, sy, ty) = self.cached_ortho;
        ((ndc_x - tx) / sx, (ndc_y - ty) / sy)
    }

    /// The same orthographic view-projection matrix `set_camera`/
    /// `set_camera_centered` just wrote to the GPU camera uniform, as a
    /// plain `Mat4` -- for a caller running a SECOND, independent render
    /// pass (e.g. [`super::TrailRenderer`]) that needs to stay pixel-
    /// aligned with this one without duplicating the projection math by
    /// hand. Matches `set_camera_centered`'s own `view_proj` layout exactly.
    pub fn view_proj(&self) -> glam::Mat4 {
        let (sx, tx, sy, ty) = self.cached_ortho;
        glam::Mat4::from_cols_array(&[
            sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
        ])
    }

    /// Real light direction for `render_grid_volume`/surface-reconstruction
    /// shading, sourced from wherever the caller's own real light lives --
    /// LP callers should pass `SimConfig::light_dir` (the SAME real value
    /// already driving `rod::Phototropism`), not invent a separate one.
    /// Replaces a value each fragment shader used to hardcode independently
    /// (and inconsistently with the sim's own real light direction).
    /// Set the mass of one fully-occupied grid cell, so the grid-volume and
    /// surface density thresholds mean "fraction of a full cell" rather than
    /// an absolute number -- see `grid_reference_cell_mass`'s own doc for the
    /// real bug this fixes. Pass the fluid's `rest_density` (grid units).
    /// Leaving it unset (1.0) preserves the previous behavior exactly.
    /// Set the curvature-flow smoothing pass count (clamped to an even value
    /// in `2..=64`). Higher = smoother, rounder surface at higher cost; lower
    /// = cheaper and more faceted. See `curvature_iterations`' own doc.
    pub fn set_curvature_iterations(&mut self, iterations: u32) {
        let clamped = iterations.clamp(2, 64);
        // Round DOWN to even -- odd would settle the result in `surface_b`
        // and silently render the un-final buffer.
        self.curvature_iterations = clamped & !1;
    }

    /// The curvature-flow smoothing pass count currently in use.
    pub fn curvature_iterations(&self) -> u32 {
        self.curvature_iterations
    }

    /// Set how much finer the surface-reconstruction grid is than the physics
    /// grid (clamped to `1..=12`). Cost scales with the SQUARE of this -- see
    /// `surface_res_multiplier`'s own doc.
    ///
    /// Forces the surface buffers to be reallocated on the next surface
    /// render, so it is safe to LOWER this too (the grow-only capacity check
    /// would otherwise keep the larger allocation and ignore the change).
    pub fn set_surface_res_multiplier(&mut self, multiplier: u32) {
        let clamped = multiplier.clamp(1, 12);
        if clamped == self.surface_res_multiplier {
            return;
        }
        self.surface_res_multiplier = clamped;
        // 0 makes `ensure_surface_capacity`'s `needed <= surface_res` check
        // fail unconditionally, so the buffers are rebuilt at the new size.
        self.surface_res = 0;
    }

    /// The surface-grid multiplier currently in use.
    pub fn surface_res_multiplier(&self) -> u32 {
        self.surface_res_multiplier
    }

    /// Set the per-particle surface splat width, in physics-grid cells
    /// (clamped to `0.05..=4.0`). `1.0` is the full MPM B-spline support and
    /// the default. See `splat_width_cells`' own doc -- this applies to ANY
    /// particle material, not just fluids.
    pub fn set_splat_width_cells(&mut self, width: f32) {
        if width.is_finite() {
            self.splat_width_cells = width.clamp(0.05, 4.0);
        }
    }

    /// The per-particle surface splat width currently in use, in cells.
    pub fn splat_width_cells(&self) -> f32 {
        self.splat_width_cells
    }

    /// Sets the splat width from the scene's real particle spacing, so the
    /// density estimate keeps a fixed statistical quality instead of a fixed
    /// size. Prefer this over [`Renderer::set_splat_width_cells`].
    ///
    /// A particle density field `ρ(x) = Σ mⱼ W(x − xⱼ)` is a sampled estimate,
    /// and its sampling noise is governed by the EFFECTIVE number of
    /// neighbours contributing to each evaluation:
    ///
    /// ```text
    /// N_eff = (Σw)² / Σw²  =  (1/Δ²) · (∫W dA)² / ∫W² dA  =  3.306 · (s/Δ)²
    /// ```
    ///
    /// with `s` the splat width, `Δ` the particle spacing in cells, `∫W dA = 1`
    /// by partition of unity and `∫W² dA = 0.3025`
    /// ([`BSPLINE_2D_SELF_OVERLAP`]). Solving for the width that reaches
    /// [`TARGET_EFFECTIVE_NEIGHBORS`] gives `s = Δ·√(N / 3.306) ≈ 2.46·Δ`.
    ///
    /// Why this matters beyond one scene: the noise is a property of the
    /// RATIO `s/Δ`, not of either alone, so a width hardcoded for one spacing
    /// is wrong at every other one. Deriving it means the same code holds at
    /// any scale -- grains of sand or planetary bodies -- and for any material,
    /// since only positions enter. It also explains the two behaviours already
    /// observed live: at `s = 1.0`, `Δ = 0.5` the estimate runs on `N_eff ≈
    /// 13`, well under the target, which is visible as grain; `s = 2.0` puts it
    /// at `N_eff ≈ 53`, which is not.
    ///
    /// The same `1/√N_eff` also governs the estimate's variation over TIME, so
    /// an under-sampled field does not merely look grainy -- it re-rolls that
    /// grain as particles drift, which is seen as flicker. Spatial grain and
    /// temporal flicker here are one defect, not two, and raising `N_eff`
    /// addresses both.
    ///
    /// Widening the kernel is also the CHEAP way to buy that: the splat loop is
    /// O(radius²) on one pass, whereas raising surface resolution multiplies
    /// every per-cell pass in the pipeline -- clear, convert, all twelve
    /// curvature-flow iterations, visibility, banding -- by the square of the
    /// resolution ratio. Both reduce grain; only one is affordable.
    ///
    /// Raising particle count instead reduces `Δ` and lets a NARROWER kernel
    /// hit the same target, which is the physically honest version of the same
    /// trade.
    pub fn set_particle_spacing_cells(&mut self, spacing_cells: f32) {
        if !spacing_cells.is_finite() || spacing_cells <= 0.0 {
            return;
        }
        let width = spacing_cells
            * (BOUNDARY_TRUNCATION_FACTOR * TARGET_EFFECTIVE_NEIGHBORS * BSPLINE_2D_SELF_OVERLAP)
                .sqrt();
        self.set_splat_width_cells(width);
    }

    /// Effective neighbour count the current splat width gives at `spacing_cells`
    /// -- the diagnostic behind [`Renderer::set_particle_spacing_cells`].
    /// Compare against [`TARGET_EFFECTIVE_NEIGHBORS`]; materially below it means
    /// visible density grain.
    pub fn effective_neighbors(&self, spacing_cells: f32) -> f32 {
        if spacing_cells <= 0.0 {
            return 0.0;
        }
        let ratio = self.splat_width_cells / spacing_cells;
        ratio * ratio / BSPLINE_2D_SELF_OVERLAP
    }

    pub fn set_grid_reference_cell_mass(&mut self, mass: f32) {
        if mass.is_finite() && mass > 0.0 {
            self.grid_reference_cell_mass = mass;
        }
    }

    /// Sets the edge-colour optical-depth floor, in dimensionless band units
    /// (multiples of `grid_reference_cell_mass`). See `edge_reference_depth`'s
    /// own doc for what moving it trades off in each direction.
    ///
    /// Upper bound is the band range itself: at or above it, the floor would
    /// win the `max` for every pixel and flatten the shading entirely, which
    /// is the failure this knob exists to avoid.
    pub fn set_edge_reference_depth(&mut self, depth: f32) {
        if depth.is_finite() {
            self.edge_reference_depth = depth.clamp(0.0, DEPTH_BANDS);
        }
    }

    /// The edge-colour optical-depth floor currently in use, in band units.
    pub fn edge_reference_depth(&self) -> f32 {
        self.edge_reference_depth
    }

    /// Sets the Yu & Turk neighbourhood-fitted splat anisotropy strength,
    /// clamped to `[0, 1]`. 0.0 restores the previous isotropic-for-fluids
    /// behaviour exactly. Applies to every material, not just fluids -- the
    /// fit reads particle positions only.
    pub fn set_anisotropy_strength(&mut self, strength: f32) {
        if strength.is_finite() {
            self.anisotropy_strength = strength.clamp(0.0, 1.0);
        }
    }

    /// The splat anisotropy strength currently in use.
    pub fn anisotropy_strength(&self) -> f32 {
        self.anisotropy_strength
    }

    pub fn set_light_dir(&mut self, x: f32, y: f32) {
        self.light_dir = (x, y);
    }

    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }
    pub fn set_vel_scale(&mut self, s: f32) {
        self.vel_scale = s;
    }

    /// Sets the Beer-Lambert absorption coefficient for `slot` AND uploads it to
    /// the GPU immediately, not just CPU-side state -- `render()`'s per-particle
    /// `particle_color()` path reads CPU state directly, but `render_grid_volume`'s
    /// GPU shader reads a GPU-resident buffer that would otherwise silently keep
    /// whatever was uploaded last, ignoring every material's real color in
    /// grid-volume mode. The redundant-write cost when setting several slots in a
    /// row is negligible (scene-setup-time only, never a per-frame path).
    pub fn set_optical_params(&mut self, queue: &wgpu::Queue, slot: usize, sigma_a: [f32; 3]) {
        self.sigma_a[slot % 16] = sigma_a;
        self.upload_optical_params(queue);
    }

    /// Reduced scattering coefficient for `slot` -- see `OpticalTable`'s doc for
    /// what this represents physically (real subsurface scattering, single-
    /// scattering approximation) and its real citation (Jacques 2013). Auto-
    /// uploads immediately -- see `set_optical_params`'s own doc for why.
    pub fn set_optical_scattering(&mut self, queue: &wgpu::Queue, slot: usize, sigma_s: f32) {
        self.sigma_s[slot % 16] = sigma_s;
        self.upload_optical_params(queue);
    }

    /// Specular Fresnel base reflectance R0 for `slot` -- see `OpticalTable`'s doc
    /// for the real-but-bounded caveat (constant near-normal reflectance, no
    /// surface-normal-dependent angle term). Auto-uploads immediately -- see
    /// `set_optical_params`'s own doc for why.
    pub fn set_specular_r0(&mut self, queue: &wgpu::Queue, slot: usize, r0: f32) {
        self.specular_r0[slot % 16] = r0;
        self.upload_optical_params(queue);
    }

    fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );
    }

    // ── GPU compute render path ────────────────────────────────────────────────

    /// Zero-readback GPU render. No `sync_particles_blocking()` needed.
    pub fn render_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        if particle_count == 0 {
            return;
        }
        self.ensure_capacity(device, particle_count);

        queue.write_buffer(
            &self.render_config_buf,
            0,
            bytemuck::bytes_of(&RenderConfig {
                mode: self.color_mode as u32,
                particle_count: particle_count as u32,
                vel_scale: self.vel_scale,
                _pad: 0,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
        );

        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prep_bg"),
            layout: &self.prep_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.storage_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.render_config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.optical_table_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_gpu"),
        });
        // Compute: fill the storage instance buffer from the particle buffer.
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prep_instances"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.prep_pipeline);
            cp.set_bind_group(0, &prep_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(PREP_WG), 1, 1);
        }
        // GPU->GPU copy into the vertex buffer (decouples storage and vertex roles).
        let bytes = (particle_count * mem::size_of::<InstanceData>()) as u64;
        enc.copy_buffer_to_buffer(&self.storage_instances, 0, &self.instance_buffer, 0, bytes);
        // Render: draw instanced quads from the vertex buffer.
        self.draw_pass(&mut enc, output_view, clear, particle_count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Grid-volume render path ────────────────────────────────────────────────

    // ── CPU render path ────────────────────────────────────────────────────────

    /// CPU-fill render for the SoA `Particles` store (CPU `Simulation`).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &Particles,
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles.iter() {
            let f = if self.rigid_render {
                Mat2::IDENTITY
            } else {
                p.deformation_gradient
            };
            self.scratch.push(InstanceData {
                deform_col0: f.x_axis.to_array(),
                deform_col1: f.y_axis.to_array(),
                position: p.x.to_array(),
                emission: blackbody_glow_factor(p.temperature),
                _pad: [0.0; 1],
                color: self.particle_color(&p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_soa"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    pub fn render_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[Particle],
        output_view: &wgpu::TextureView,
        clear: bool,
    ) {
        let count = particles.len();
        if count == 0 {
            return;
        }
        self.ensure_capacity(device, count);

        self.scratch.clear();
        for p in particles {
            let f = if self.rigid_render {
                Mat2::IDENTITY
            } else {
                p.deformation_gradient
            };
            self.scratch.push(InstanceData {
                deform_col0: f.x_axis.to_array(),
                deform_col1: f.y_axis.to_array(),
                position: p.x.to_array(),
                emission: blackbody_glow_factor(p.temperature),
                _pad: [0.0; 1],
                color: self.particle_color(p),
            });
        }
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.scratch),
        );

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_particles_cpu"),
        });
        self.draw_pass(&mut enc, output_view, clear, count);
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.max_particles {
            let size = (count * mem::size_of::<InstanceData>()) as u64;
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances"),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_instances_storage"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.max_particles = count;
        }
    }

    fn draw_pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        clear: bool,
        count: usize,
    ) {
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, &self.render_bind_group, &[]);
        rp.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rp.set_vertex_buffer(1, self.instance_buffer.slice(..));
        rp.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rp.draw_indexed(0..6, 0, 0..count as u32);
    }
}

// particle_color (the CPU-path per-particle color computation) is split into
// color.rs alongside the rest of the "Color helpers" section below -- see
// that file's own doc comment.
mod color;
use color::{blackbody_glow_factor, write_optical_table};

// Test suite split into its own file -- was ~150 of this file's ~930 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`.
#[cfg(test)]
mod tests;

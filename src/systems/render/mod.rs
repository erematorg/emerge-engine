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

use glam::Vec2;

use crate::particle::{Particle, Particles};
use crate::systems::gpu::MAX_RENDER_MATERIAL_SLOTS;

const RENDER_SHADER: &str = include_str!("shaders/render_particles.wgsl");
const SNAPSHOT_SHADER: &str = include_str!("shaders/snapshot_positions.wgsl");

// `blackbody.inc.wgsl` is prepended to every shader that renders thermal
// emission, so the particle, grid and surface paths share one implementation
// of `energy::radiation`'s colour instead of three drifting copies, and
// `cel_lighting.inc.wgsl` gives the grid and surface paths one copy of their
// cel-shaded lighting. `concat!` over `include_str!` keeps these `const`, with no runtime string
// work -- the same shared-source idea as the compute side's `.inc.wgsl`
// files, which concatenate at pipeline build time instead.
const PREP_SHADER: &str = concat!(
    include_str!("shaders/blackbody.inc.wgsl"),
    include_str!("shaders/radiative_transfer.inc.wgsl"),
    include_str!("shaders/prep_instances.wgsl")
);
const GRID_VOLUME_SHADER: &str = concat!(
    include_str!("shaders/blackbody.inc.wgsl"),
    include_str!("shaders/radiative_transfer.inc.wgsl"),
    include_str!("shaders/cel_lighting.inc.wgsl"),
    include_str!("shaders/grid_volume.wgsl")
);
const LIGHT_PASS_SHADER: &str = include_str!("shaders/light_pass.wgsl");
const CURVATURE_FLOW_SHADER: &str = concat!(
    include_str!("shaders/blackbody.inc.wgsl"),
    include_str!("shaders/radiative_transfer.inc.wgsl"),
    include_str!("shaders/cel_lighting.inc.wgsl"),
    include_str!("shaders/curvature_flow.wgsl")
);
/// Default thermal-emission exposure anchor, in kelvin -- the temperature
/// that renders at full brightness until a scene states its own (see
/// `Renderer::set_emission_reference_temperature`). 3000 K is roughly an
/// incandescent filament and the hot core of an open flame, so the engine's
/// existing fire/lava scenes land mid-range rather than black or blown out.
/// `blackbody.inc.wgsl` repeats this value for the zero-initialized-uniform
/// case; the two must stay equal.
const DEFAULT_EMISSION_REFERENCE_K: f32 = 3000.0;
const PREP_WG: u32 = 64;
const SURFACE_CLEAR_WG: u32 = 64;
const SURFACE_SPLAT_WG: u32 = 64;
/// Finer-than-physics-grid resolution multiplier (see `curvature_flow.wgsl`'s
/// own top doc). Cost scales with the SQUARE of this constant --
/// `(N*grid_res)^2` f32 cells through every pass (splat, convert,
/// curvature-iterate, thermal diffusion, wave, visibility, band-hysteresis).
const SURFACE_RES_MULTIPLIER: u32 = 6;
/// Fixed even iteration count, so the settled result always lands in
/// `surface_a` and the final bind group needs no runtime branch (see
/// `render_surface_reconstruction`). 12 is within van der Laan et al.
/// 2009's "several iterations per frame".
const CURVATURE_ITERATIONS: u32 = 12;
/// Number of flat depth bands `curvature_flow.wgsl` quantizes optical depth
/// into; must equal that file's file-scope `DEPTH_BANDS` (which `fs_main`,
/// `shade_phase` and `band_hysteresis_step_main` share) and
/// `grid_volume.wgsl`'s copy. WGSL and Rust cannot share the constant, so
/// these two are kept in sync by hand.
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
/// Set to `1.0` (no-op): at `2.0` (`splat_width_cells ≈ 1.74`, against ≈ 1.23
/// interior-only) a gap opened at the free surface, growing with
/// `surface_res_multiplier`, when stacked on other raised settings. Not
/// re-applied until that interaction is diagnosed on screen; see
/// `set_particle_spacing_cells`.
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

/// The largest mass difference, as a fraction of a full cell, between two
/// neighbouring cells across a straight free surface, for a splat kernel
/// spread over `splat_cells` cells per physics cell.
///
/// Across such a surface the density follows the cumulative of the
/// quadratic B-spline (`grid::kernel::axis_weights`), stretched by
/// `splat_cells`, so neighbouring cells differ by at most the kernel's mass
/// inside one cell centred on its peak: the integral of the B-spline over
/// `[-h, h]`, `h = 1 / (2 * splat_cells)`. At 1 (the physics grid) that is
/// 2/3, what the grid-volume path uses. Curvature flow moves
/// level sets by their curvature, zero along a straight edge, so the
/// surface pass does not lower this bound there.
fn free_surface_cell_step(splat_cells: f32) -> f32 {
    let h = 0.5 / splat_cells.max(1.0e-3);
    if h <= 0.5 {
        2.0 * (0.75 * h - h * h * h / 3.0)
    } else if h < 1.5 {
        2.0 / 3.0 + (1.0 - (1.5 - h).powi(3)) / 3.0
    } else {
        1.0
    }
}
/// Default [`Renderer::set_edge_reference_depth`].
///
/// The shader's doc gives a LOWER bound of one interior depth-band step
/// (`1.0 / DEPTH_BANDS`). It is correct as a bound and wrong as a default:
/// unpinning optical depth that far makes the per-particle density ripple
/// drive colour directly, and the surface renders as blue speckle over white
/// rather than a solid body. The ripple is pre-existing -- a high floor
/// flattens it out of sight -- and its fix is the neighbourhood-fitted splat
/// kernel in `render::anisotropy`, not this knob. Lowering this is worth
/// retrying once that lands.
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
    /// See `Particle::scalar_field`'s doc. Distinct wire value (6, not the next
    /// unused slot after ByActivation's implicit WGSL else-branch) so the GPU shader's
    /// existing fallback `else` can keep meaning ByActivation without renumbering it.
    ByScalarField = 6,
    /// A per-particle scalar the caller computes and hands in with
    /// `set_stress_field`, drawn through `heat` after `set_stress_scale`.
    /// Its intended field is `MaterialRegistry::von_mises_stress_field`, the
    /// von Mises equivalent of each particle's deviatoric stress: shear
    /// activity, which `ByVolume` (det(F)) misses and pressure no longer
    /// masks. CPU render path only for now (`Renderer::render`,
    /// which owns the `&Particles`/`MaterialRegistry` needed to compute
    /// it) -- see `Renderer::set_stress_field`'s doc for the real,
    /// disclosed reason the GPU shader path doesn't have this yet.
    ByStress = 7,
}

// GPU-side wire structs (InstanceData/CameraParams/RenderConfig/OpticalTable,
// GridVolumeParams/GridVolumeSource) live in gpu_types.rs -- see that file's doc.
mod gpu_types;
use gpu_types::{
    CameraParams, InstanceData, OpticalTable, PhysicalRenderParams, RenderConfig, SnapshotConfig,
};
pub use gpu_types::{
    DualPhaseSurfaceSource, GpuRenderParams, GridVolumeSource, SurfaceReconstructionSource,
};

// GPU buffer allocation (the RenderBuffers struct + its own constructor)
// lives in buffers.rs -- see that file's doc.
mod buffers;
use buffers::RenderBuffers;

// Grid-native volumetric render path (`render_grid_volume`) lives in
// grid_volume.rs -- see that file's doc.
mod grid_volume;

// Transmittance of the declared light through the matter, shared by the
// grid-volume and surface paths -- see that file's doc.
mod light_pass;
use light_pass::{LightPassSource, light_field_buffer};

// Buffers a CPU `Simulation` needs to use the grid-volume and surface paths,
// which read GPU-resident state -- see that file's doc.
mod cpu_bridge;
pub use cpu_bridge::CpuRenderBridge;

// Shared SI optical contract and analytic reference solutions.  Kept
// independent of any particular GPU path so particle, grid-volume, and
// reconstructed-surface rendering cannot silently choose different units.
pub mod optics;
pub use optics::{
    PhysicalRenderContract, PhysicalRenderContractError, PhysicalRenderContractParams,
};
// Attenuation is an energy law, not a rendering one; the renderer applies it
// but does not own it. Re-exported here so `render::` stays one import for a
// caller wiring up optics.
pub use crate::energy::radiation::{
    OpticalCoefficientsError, OpticalCoefficientsSi, beer_lambert_transmittance,
};

// The colour of hot matter is physics, not rendering: it comes from
// `energy::radiation` (Planck's law through the CIE 1931 observer), used by
// `color.rs` on the CPU path, and mirrored on the GPU by
// `shaders/blackbody.inc.wgsl`. Nothing in this module derives an emission
// colour of its own.

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
use gpu_types::{
    BandHysteresisParams, LightDiffuseParams, SurfaceParams, SurfaceRenderParams, VisibilityParams,
    WaveStepParams,
};
use pipelines::{
    build_band_hysteresis_step_pipeline, build_grid_peak_pipeline, build_grid_volume_pipeline,
    build_light_diffuse_pipeline, build_light_pass_pipelines, build_particle_pipeline,
    build_post_total_reduce_pipeline, build_prep_pipeline, build_snapshot_pipeline,
    build_surface_clear_pipeline, build_surface_convert_pipeline,
    build_surface_dual_render_pipeline, build_surface_iterate_pipeline,
    build_surface_moments_pipeline, build_surface_render_pipeline, build_surface_splat_pipeline,
    build_temp_avg_pipeline, build_temp_diffuse_pipeline, build_visibility_step_pipeline,
    build_volume_correct_pipeline, build_wave_step_pipeline,
};

// ── Renderer ──────────────────────────────────────────────────────────────────

pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer, // VERTEX | COPY_DST -- drawn as per-instance attributes
    storage_instances: wgpu::Buffer, // STORAGE | COPY_SRC -- compute write target (GPU path)
    /// Pre-step position snapshot for GPU render interpolation -- see
    /// `snapshot_particle_positions`'s doc.
    prev_positions_buf: wgpu::Buffer,
    snapshot_pipeline: wgpu::ComputePipeline,
    snapshot_bgl: wgpu::BindGroupLayout,
    snapshot_config_buf: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    max_particles: usize,

    prep_pipeline: wgpu::ComputePipeline,
    prep_bgl: wgpu::BindGroupLayout,
    render_config_buf: wgpu::Buffer,
    optical_table_buf: wgpu::Buffer,
    /// Shared SI scale/radiance uniform. Zero-initialized means no validated
    /// physical contract has been supplied; every physical GPU path binds the
    /// same buffer rather than carrying path-specific unit assumptions.
    physical_render_params_buf: wgpu::Buffer,
    physical_render_contract: Option<PhysicalRenderContract>,
    /// See `set_emission_reference_temperature`. The default matches
    /// `blackbody.inc.wgsl`'s own `BLACKBODY_DEFAULT_REFERENCE_K`, so the
    /// zero-initialized uniform above and this field agree before any scene
    /// states its own exposure.
    emission_reference_k: f32,

    grid_volume_pipeline: wgpu::RenderPipeline,
    grid_volume_bgl: wgpu::BindGroupLayout,
    grid_volume_params_buf: wgpu::Buffer,
    /// Local peak of the cell mass, the grid-native path's edge reference
    /// (`grid_volume.wgsl`'s `grid_peak_main`), at `grid_res`.
    grid_peak_pipeline: wgpu::ComputePipeline,
    grid_peak_bgl: wgpu::BindGroupLayout,
    grid_peak_buf: wgpu::Buffer,
    grid_peak_params_buf: wgpu::Buffer,
    /// Resolution `grid_peak_buf` is currently allocated at --
    /// `ensure_grid_peak_capacity` regrows it when a caller's
    /// `grid_res` exceeds this, same lazy-growth convention as `surface_res`.
    grid_peak_res: u32,
    /// The light pass (`light_pass.rs`): per-cell extinction from the grid or
    /// from the surface density, then the transmittance of the declared light
    /// to each physics-grid cell, at `light_pass_res`. Run only under a
    /// `PhysicalRenderContract`.
    light_extinction_grid_pipeline: wgpu::ComputePipeline,
    /// Per surface cell, then each grid cell's mean (`light_pass.wgsl`).
    light_extinction_surface_pipelines: [wgpu::ComputePipeline; 2],
    light_march_pipeline: wgpu::ComputePipeline,
    light_pass_bgl: wgpu::BindGroupLayout,
    light_pass_params_buf: wgpu::Buffer,
    /// The surface path's per-surface-cell extinction, at
    /// `light_surface_extinction_res`.
    light_surface_extinction_buf: wgpu::Buffer,
    light_surface_extinction_res: u32,
    light_extinction_buf: wgpu::Buffer,
    light_transmittance_buf: wgpu::Buffer,
    light_pass_res: u32,
    /// Cached ortho projection + grid_res (set by `set_camera`) -- lets
    /// `render_grid_volume` take just (device, queue, grid_buf, material_mass_buf,
    /// view, clear) instead of repeating width/height/grid_res, keeping it under
    /// clippy's argument-count lint.
    cached_ortho: (f32, f32, f32, f32),
    /// Light direction for `render_grid_volume`/`render_surface_
    /// reconstruction`(`_dual_phase`)'s Lambertian and specular shading, set
    /// via `set_light_dir` (e.g. from `SimConfig::light_dir`); defaults to the
    /// value those shaders used to hardcode.
    light_dir: (f32, f32),

    /// Mass of ONE fully-occupied grid cell in the caller's own units, used to
    /// scale the grid-volume/surface density thresholds. Defaults to `1.0`,
    /// which reproduces the old absolute thresholds.
    ///
    /// Absolute thresholds (e.g. `grid_volume.rs`'s `mass_floor = 0.15`)
    /// assumed cells of mass 0.5-4. Real water at 1 cm cells is
    /// `rho0 = 1000 kg/m^3 * dx^2 = 0.1` grid units, so a full cell weighed
    /// less than the floor and the fluid rendered as near empty while the
    /// particle mode showed it.
    ///
    /// Callers should pass their fluid's `rest_density * cell_area` (in grid
    /// units, cell_area = 1). Thresholds then mean "this fraction of a full
    /// cell", which is scale-free and correct for any density calibration.
    grid_reference_cell_mass: f32,

    /// Curvature-flow smoothing passes per frame, defaulting to
    /// `CURVATURE_ITERATIONS`. Settable via `set_curvature_iterations` so a
    /// caller can trade surface smoothness against cost instead of being stuck
    /// with one compile-time value: van der Laan et al. 2009 only prescribes
    /// "several iterations per frame", and the right number depends
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
    /// up. The right value depends on the scene and the perf budget,
    /// which is exactly why it should not be baked in.
    surface_res_multiplier: u32,

    /// Propagating-wave excitation strength for the Surface path's Pass 2b
    /// (the wave PDE `∂²h/∂t² = c²∇²h`, CFL-stable explicit scheme -- see
    /// `curvature_flow.wgsl`). Defaults to `0.0` (inert): only a material
    /// that behaves like a fluid propagates such waves, so derive it at scene
    /// setup from `registry.owns_deformation_volume_state(material_id)` (the
    /// test `sand_water_saturation.rs` uses to classify moisture sources),
    /// not per demo. `0.35` is `basic_fluids.rs`'s tuned value for fluids.
    wave_force_coeff: f32,

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
    /// Mass-weighted temperature: fixed-point atomic scatter target and the
    /// converted f32 buffer `fs_main` samples for blackbody emission (see
    /// `surface_temp_atomic` in the shader). Shared by phase A of the
    /// dual-phase path like `surface_atomic_buf`; phase B has its own pair.
    surface_temp_atomic_buf: wgpu::Buffer,
    surface_temp_float_buf: wgpu::Buffer,
    /// Volume-preserving correction (`curvature_flow.wgsl`'s Pass 1d):
    /// `pre_total_atomic_buf` accumulates the particle mass during the splat,
    /// `post_total_atomic_buf` sums the settled density, and
    /// `volume_correct_pipeline` rescales by their ratio. Single-element
    /// buffers (one scalar per phase), never grown with `surface_res`.
    post_total_reduce_pipeline: wgpu::ComputePipeline,
    post_total_reduce_bgl: wgpu::BindGroupLayout,
    volume_correct_pipeline: wgpu::ComputePipeline,
    volume_correct_bgl: wgpu::BindGroupLayout,
    pre_total_atomic_buf: wgpu::Buffer,
    post_total_atomic_buf: wgpu::Buffer,
    /// 2D heat equation on the recovered temperature field
    /// (`curvature_flow.wgsl`'s Pass 1c): `temp_avg_pipeline` divides by the
    /// settled density once, `temp_diffuse_pipeline` runs one step.
    /// `surface_temp_b_buf` is the ping-pong partner: avg writes it, diffuse
    /// reads it and writes back into `surface_temp_float_buf`, the buffer
    /// `fs_main` reads.
    temp_avg_pipeline: wgpu::ComputePipeline,
    temp_avg_bgl: wgpu::BindGroupLayout,
    temp_diffuse_pipeline: wgpu::ComputePipeline,
    temp_diffuse_bgl: wgpu::BindGroupLayout,
    surface_temp_b_buf: wgpu::Buffer,
    /// Diffusion approximation to light transport (`curvature_flow.wgsl`'s
    /// Pass 1e). `light_phi_bufs` are two ping-pong buffers (see
    /// `buffers.rs`); `light_frame_index` alternates read and write each
    /// frame, a 2-way rotation for this first-order equation where the wave
    /// field needs 3.
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

    /// N-material extension (see `curvature_flow.wgsl`'s doc),
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

    /// Persistent 2D wave-equation height field (`curvature_flow.wgsl`'s
    /// Pass 2b). Three buffers, not two: wgpu's usage-scope validation
    /// rejects one buffer bound read-only and read_write in the same
    /// dispatch even when access is index-disjoint, so writing u(t+dt) next
    /// to u(t) and u(t-dt) needs a third slot. `wave_frame_index` rotates the
    /// roles (current/previous/next) each call.
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

    /// Whether `wave_density_prev_buf` has been seeded from a settled
    /// density yet. False right after `Renderer::new`/a capacity grow (the
    /// buffer is a zero placeholder then) -- reading it as "previous
    /// density" on that first frame would read density appearing from
    /// nothing as false motion. Seeded (prev := now, force = 0) before the
    /// first `wave_step` dispatch instead; see that call site's doc.
    wave_prev_seeded: bool,
    /// Phase B's own `wave_prev_seeded` -- see its doc.
    phase_b_wave_prev_seeded: bool,

    /// Persistent hysteresis visible/invisible state (`curvature_flow.wgsl`'s
    /// Pass 2c). One buffer: the pass reads and writes only its own index, so
    /// there is no usage-scope conflict to rotate around.
    visibility_step_pipeline: wgpu::ComputePipeline,
    visibility_step_bgl: wgpu::BindGroupLayout,
    visibility_buf: wgpu::Buffer,
    visibility_params_buf: wgpu::Buffer,

    /// Persistent hysteresis color-band state (`curvature_flow.wgsl`'s Pass
    /// 2d): one buffer like `visibility_buf`, downstream of the density field
    /// and never fed back into it.
    band_hysteresis_step_pipeline: wgpu::ComputePipeline,
    band_hysteresis_step_bgl: wgpu::BindGroupLayout,
    band_state_buf: wgpu::Buffer,
    band_hysteresis_params_buf: wgpu::Buffer,

    /// Persistent raw splat density history for neighbourhood-clamped
    /// temporal smoothing (see `convert_atomic_to_float_main`; the Lottes
    /// 2011 / Karis 2014 TAA technique on a scalar field).
    raw_splat_history_buf: wgpu::Buffer,

    // ── Two-phase extension (see curvature_flow.wgsl's doc) ────────────
    /// A second, fully independent set of splat/ping-pong buffers for
    /// "phase B" -- `render_surface_reconstruction_dual_phase` runs the
    /// SAME clear/splat/convert/iterate pipelines twice, once into phase A's
    /// existing `surface_*_buf` fields above and once into these, so each
    /// phase gets its own independently-smoothed surface (see the
    /// shader's own VOF/phase-fraction citation for why that's the correct
    /// choice, not a shared blended field).
    phase_b_atomic_buf: wgpu::Buffer,
    /// Phase B's own temperature atomic/float pair -- see `surface_temp_
    /// atomic_buf`'s doc. Written every dual-phase frame for symmetry
    /// with phase A's scatter, but NOT read by `fs_main_dual_phase` (no
    /// binding for it there -- see that entry point's doc for why).
    phase_b_temp_atomic_buf: wgpu::Buffer,
    phase_b_temp_float_buf: wgpu::Buffer,
    /// Phase B's own volume-preserving-correction totals -- see
    /// `pre_total_atomic_buf`'s doc.
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
    /// doc.
    phase_b_wave_density_prev_buf: wgpu::Buffer,
    phase_b_visibility_buf: wgpu::Buffer,
    phase_b_band_state_buf: wgpu::Buffer,
    surface_dual_render_pipeline: wgpu::RenderPipeline,
    surface_dual_render_bgl: wgpu::BindGroupLayout,

    scratch: Vec<InstanceData>,
    color_mode: ColorMode,
    vel_scale: f32,
    /// Per-particle value for `ColorMode::ByStress`, computed once per frame
    /// by the caller (typically `MaterialRegistry::von_mises_stress_field`,
    /// which has the particles and the material dispatch `Renderer` does not
    /// depend on) and handed in via `set_stress_field`, like
    /// `sigma_a`/`sigma_s`/`specular_r0` for `ColorMode::ByPhysics`. Indexed
    /// in `render`'s particle order; reads 0.0 per particle if never set.
    stress_field: Vec<f32>,
    /// Display scale for `ColorMode::ByStress`: stress magnitudes span orders
    /// of magnitude across materials, so the caller sets it (like `vel_scale`
    /// for `ByVelocity`).
    stress_scale: f32,
    sigma_a: [[f32; 3]; 16],
    /// Reduced scattering coefficient per material slot (single scalar -- see
    /// `OpticalTable`'s doc for why this isn't per-channel).
    sigma_s: [f32; 16],
    /// Specular Fresnel base reflectance R0 per material slot (see `OpticalTable`'s
    /// doc for the real-but-bounded caveat).
    specular_r0: [f32; 16],
    /// Whether each slot's material holds its shape, a nonzero shear
    /// modulus (see `adopt_material_optics`). The grid-volume and surface
    /// views draw such a body with flat faces.
    holds_shape: [bool; 16],
    /// Refractive index of the dry solid/grain per material slot (see
    /// `set_refractive_index`: pore-fluid index matching darkens wet
    /// material, for any material and scalar field). `1.0`, air, means no
    /// contrast and leaves `particle_color` unchanged.
    refractive_index: [f32; 16],
    /// Per-material volumetric luminous source, `W/m^3` -- see
    /// `MaterialModel::luminous_emission_w_m3`. Zero everywhere until a
    /// material declares it, which is the normal case: almost nothing
    /// glows on its own.
    luminous_emission: [f32; 16],
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
            prev_positions_buf,
            snapshot_config_buf,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            render_config_buf,
            optical_table_buf,
            physical_render_params_buf,
            grid_volume_params_buf,
            grid_peak_buf,
            grid_peak_params_buf,
            light_pass_params_buf,
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
        let (snapshot_pipeline, snapshot_bgl) = build_snapshot_pipeline(device);
        let (grid_volume_pipeline, grid_volume_bgl) =
            build_grid_volume_pipeline(device, output_format);
        let (grid_peak_pipeline, grid_peak_bgl) = build_grid_peak_pipeline(device);
        let (
            light_extinction_grid_pipeline,
            light_extinction_surface_pipelines,
            light_march_pipeline,
            light_pass_bgl,
        ) = build_light_pass_pipelines(device);
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
            prev_positions_buf,
            snapshot_pipeline,
            snapshot_bgl,
            snapshot_config_buf,
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
            grid_peak_pipeline,
            grid_peak_bgl,
            grid_peak_buf,
            grid_peak_params_buf,
            grid_peak_res: 1,
            light_extinction_grid_pipeline,
            light_extinction_surface_pipelines,
            light_march_pipeline,
            light_pass_bgl,
            light_pass_params_buf,
            light_surface_extinction_buf: light_field_buffer(device, 1, "light_surface_extinction"),
            light_surface_extinction_res: 1,
            light_extinction_buf: light_field_buffer(device, 1, "light_extinction"),
            light_transmittance_buf: light_field_buffer(device, 1, "light_transmittance"),
            light_pass_res: 1,
            cached_ortho: (1.0, 0.0, 1.0, 0.0),
            light_dir: (-0.5, 0.7),
            grid_reference_cell_mass: 1.0,
            curvature_iterations: CURVATURE_ITERATIONS,
            surface_res_multiplier: SURFACE_RES_MULTIPLIER,
            wave_force_coeff: 0.0,
            splat_width_cells: 1.0,
            edge_reference_depth: DEFAULT_EDGE_REFERENCE_DEPTH,
            // Off by default: `> 0.0` is the sole gate on both the moments
            // scatter dispatch (`surface_reconstruction.rs`) and the fit
            // itself (`splat_density_main`), so `0.0` makes the whole Yu &
            // Turk path -- moments buffer, scatter, gather, eigendecompose --
            // fully inert.
            //
            // At full strength, alone: `BOUNDARY_TRUNCATION_FACTOR` stays at
            // 1.0 and `set_particle_spacing_cells` stays unwired in the demo,
            // the two changes that regressed together before. Anisotropic
            // particle rendering is also the fix Sebastian Lague's "Coding
            // Adventure: Rendering Fluids" names for the same blobby surface.
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
            physical_render_params_buf,
            physical_render_contract: None,
            emission_reference_k: DEFAULT_EMISSION_REFERENCE_K,
            scratch: Vec::with_capacity(cap),
            color_mode: ColorMode::ByMaterial,
            vel_scale: 0.05,
            stress_field: Vec::new(),
            stress_scale: 1.0,
            sigma_a: [[0.3f32; 3]; 16],
            sigma_s: [0.0f32; 16],
            specular_r0: [0.0f32; 16],
            holds_shape: [false; 16],
            refractive_index: [1.0f32; 16],
            luminous_emission: [0.0f32; 16],
        }
    }

    // ── Configuration ─────────────────────────────────────────────────────────

    /// The orthographic projection that frames the grid rectangle from `min`
    /// to `max`, in cells, in a `width` x `height` window: `(sx, tx, sy, ty)`,
    /// a grid point `(x, y)` landing at NDC `(x * sx + tx, y * sy + ty)`. The
    /// rectangle fills the window along whichever axis it reaches first,
    /// centred, with square pixels.
    ///
    /// Every camera goes through this one function: `set_camera` frames the
    /// whole grid with it, `set_camera_region` a part. A caller that maps a
    /// cursor without holding the renderer inverts the same numbers, so what
    /// was drawn and what a click reads back cannot drift apart.
    pub fn region_projection(
        min: Vec2,
        max: Vec2,
        width: u32,
        height: u32,
    ) -> (f32, f32, f32, f32) {
        let size = (max - min).max(Vec2::splat(1.0e-3));
        let centre = (min + max) * 0.5;
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        // NDC per cell along y; along x it is divided by the aspect so a
        // cell is as many pixels wide as it is tall. The rectangle must fit
        // both ways: 2 NDC across the height, 2 * aspect across the width.
        let s = (2.0 / size.y).min(2.0 * aspect / size.x);
        let (sx, sy) = (s / aspect, s);
        (sx, -sx * centre.x, sy, -sy * centre.y)
    }

    /// Call at init and on every resize.
    pub fn set_camera(
        &mut self,
        queue: &wgpu::Queue,
        grid_res: u32,
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let (sx, tx, sy, ty) =
            Self::region_projection(Vec2::ZERO, Vec2::splat(grid_res as f32), width, height);
        self.cached_ortho = (sx, tx, sy, ty);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                _pad: [0.0; 2],
            }),
        );
    }

    /// Frames a rectangle of the grid instead of the whole of it: the
    /// rectangle from `min` to `max`, in grid cells, fills the window along
    /// whichever axis it reaches first, centred, with square pixels.
    ///
    /// A scene whose material occupies a thin strip of a large grid is
    /// otherwise drawn small, because `set_camera` always shows every cell.
    /// This writes the same cached projection `set_camera` does, so
    /// `screen_to_grid` and `grid_to_screen` stay exact under it. It does not
    /// set the grid resolution `render_grid_volume` reads; that path needs
    /// `set_camera` to have run once.
    pub fn set_camera_region(
        &mut self,
        queue: &wgpu::Queue,
        region: (Vec2, Vec2),
        width: u32,
        height: u32,
        particle_scale: f32,
        round_particles: bool,
    ) {
        let (sx, tx, sy, ty) = Self::region_projection(region.0, region.1, width, height);
        self.cached_ortho = (sx, tx, sy, ty);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraParams {
                view_proj: [
                    sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 0.0, 1.0,
                ],
                particle_scale,
                round_particles: round_particles as u32,
                _pad: [0.0; 2],
            }),
        );
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

    /// Exact inverse of `screen_to_grid` -- grid coordinate -> screen pixel
    /// position, from the SAME cached projection `set_camera` last uploaded.
    /// For a caller drawing a screen-space overlay (a UI marker for a
    /// non-`Particle` object, say) at a world/grid position: use this
    /// instead of re-deriving the ortho formula, which can drift out of sync
    /// with what `set_camera` did. `width`/`height` must match whatever was
    /// last passed to `set_camera`.
    pub fn grid_to_screen(&self, grid_x: f32, grid_y: f32, width: u32, height: u32) -> (f32, f32) {
        let (sx, tx, sy, ty) = self.cached_ortho;
        let ndc_x = grid_x * sx + tx;
        let ndc_y = grid_y * sy + ty;
        let screen_x = (ndc_x + 1.0) * 0.5 * width.max(1) as f32;
        let screen_y = (1.0 - ndc_y) * 0.5 * height.max(1) as f32;
        (screen_x, screen_y)
    }

    /// A world/grid-space distance (e.g. an object's radius) -> screen
    /// pixels, from the same cached projection. `set_camera` always keeps
    /// grid cells isotropic (a circle in grid space renders as a circle,
    /// never an ellipse, regardless of window aspect), so the y-axis scale
    /// alone gives the correct pixels-per-grid-unit conversion for
    /// any direction.
    pub fn grid_distance_to_pixels(&self, distance: f32, height: u32) -> f32 {
        let (_, _, sy, _) = self.cached_ortho;
        distance * sy.abs() * 0.5 * height.max(1) as f32
    }

    /// `grid_to_screen`, converted into UI/LOGICAL points instead of
    /// physical pixels -- the units a UI toolkit (egui and friends) actually
    /// draws in. `width`/`height` are still the PHYSICAL size last passed to
    /// `set_camera` (that projection operates in physical space);
    /// `pixels_per_point` is the display's DPI scale factor (egui:
    /// `Context::pixels_per_point()`, 1.0 at 100% OS scaling, 1.25 at 125%,
    /// etc.). Dividing by it by hand is easy to forget and only shows at
    /// non-100% scaling, so the DPI-correct path is the one with the name.
    pub fn grid_to_screen_points(
        &self,
        grid_x: f32,
        grid_y: f32,
        width: u32,
        height: u32,
        pixels_per_point: f32,
    ) -> (f32, f32) {
        let (px, py) = self.grid_to_screen(grid_x, grid_y, width, height);
        let ppp = pixels_per_point.max(1.0e-6);
        (px / ppp, py / ppp)
    }

    /// `grid_distance_to_pixels`, converted into UI/LOGICAL points -- see
    /// `grid_to_screen_points`'s doc for the DPI reasoning this
    /// shares.
    pub fn grid_distance_to_points(
        &self,
        distance: f32,
        height: u32,
        pixels_per_point: f32,
    ) -> f32 {
        self.grid_distance_to_pixels(distance, height) / pixels_per_point.max(1.0e-6)
    }

    /// Light direction for `render_grid_volume`/surface-reconstruction
    /// shading; LP callers should pass `SimConfig::light_dir`, the value that
    /// drives `rod::Phototropism`, not a separate one.
    /// Set the mass of one fully-occupied grid cell, so the grid-volume and
    /// surface density thresholds mean "fraction of a full cell" rather than
    /// an absolute number -- see `grid_reference_cell_mass`. Pass the
    /// fluid's `rest_density` (grid units). Unset (1.0) keeps the absolute
    /// thresholds.
    /// Set the curvature-flow smoothing pass count (clamped to an even value
    /// in `2..=64`). Higher = smoother, rounder surface at higher cost; lower
    /// = cheaper and more faceted. See `curvature_iterations`' doc.
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
    /// `surface_res_multiplier`'s doc.
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

    /// Set the Surface path's propagating-wave excitation strength -- see
    /// `wave_force_coeff`'s doc for the mechanism and why this
    /// should be DERIVED from `MaterialModel::owns_deformation_volume_
    /// state()` at the call site, not hand-picked per scene. `0.0` (the
    /// default) is inert; `0.35` is this engine's own real fluid-tuned
    /// value (`basic_fluids.rs`).
    pub fn set_wave_force_coeff(&mut self, coeff: f32) {
        self.wave_force_coeff = coeff;
    }

    /// Set the per-particle surface splat width, in physics-grid cells
    /// (clamped to `0.05..=4.0`). `1.0` is the full MPM B-spline support and
    /// the default. See `splat_width_cells`' doc -- this applies to ANY
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
    /// doc for what moving it trades off in each direction.
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

    /// Installs the validated SI scale and radiance contract used by every
    /// `ByPhysics` render path.
    ///
    /// This is intentionally separate from [`Renderer::new`]: the renderer
    /// cannot infer a simulation's metres-per-cell or the out-of-plane
    /// thickness represented by a 2-D slice. Until this is called, the shared
    /// GPU uniform remains disabled and existing dimensionless rendering is
    /// explicitly legacy behavior.
    pub fn set_physical_render_contract(
        &mut self,
        queue: &wgpu::Queue,
        contract: PhysicalRenderContract,
    ) {
        let camera = contract.camera_direction();
        let light = contract.light_direction();
        let incident = contract.incident_radiance_w_m2_sr();
        let background = contract.background_radiance_w_m2_sr();
        let display_white = contract.display_white_radiance_w_m2_sr();
        let params = PhysicalRenderParams {
            spatial: [contract.dx_meters(), contract.slice_thickness_m(), 1.0, 0.0],
            incident_radiance: [incident[0], incident[1], incident[2], 0.0],
            background_radiance: [background[0], background[1], background[2], 0.0],
            display_white_radiance: [display_white[0], display_white[1], display_white[2], 0.0],
            camera_direction: [camera.x, camera.y, camera.z, 0.0],
            light_direction: [light.x, light.y, light.z, 0.0],
            emission: [self.emission_reference_k, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(
            &self.physical_render_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        self.physical_render_contract = Some(contract);
    }

    /// Pulls every material's own declared optical constants into the
    /// renderer, so a scene never types absorption coefficients by hand.
    ///
    /// This is the mechanism that replaces the painted palette: a material
    /// that declares measured constants (`MaterialModel::optical_properties`)
    /// gets its colour computed from them, and nothing about the renderer
    /// changes when a new material is added -- it simply works. A material
    /// that declares nothing is left untouched, keeping whatever the caller
    /// set, and falls back to the placeholder palette in `ByMaterial`.
    ///
    /// It also records which materials hold their shape: a nonzero shear
    /// modulus (`MaterialParams::mu`), so the body resists being sheared and
    /// keeps flat faces. The grid-volume and surface views then draw it
    /// flat: no rim shading from the density ramp at its edge, which is the
    /// reconstruction kernel's width and not a rounded surface, and no
    /// self-darkening down its own column, which the grid-volume view
    /// models as sunlight crossing a liquid. A liquid or gas (zero shear
    /// modulus) keeps both.
    ///
    /// Returns how many materials declared optics, so a caller can see at a
    /// glance how much of its scene is physically coloured and how much is
    /// still standing on a placeholder.
    pub fn adopt_material_optics(
        &mut self,
        queue: &wgpu::Queue,
        registry: &crate::materials::registry::MaterialRegistry,
    ) -> usize {
        let mut declared = 0;
        // Only the slots the registry actually has. Walking all
        // `MAX_RENDER_MATERIAL_SLOTS` asked it for material ids that were
        // never registered, which `MaterialRegistry::get`'s own
        // `debug_assert` exists to catch: every debug-build scene with
        // fewer than sixteen materials panicked here, and every release
        // build silently read material 0's optics into the empty slots and
        // counted them as declared.
        let registered = (registry.len()).min(MAX_RENDER_MATERIAL_SLOTS as usize);
        for slot in 0..registered {
            let material = registry.get(slot as u32);
            self.luminous_emission[slot] = material.luminous_emission_w_m3();
            self.holds_shape[slot] = material.params().mu > 0.0;
            if let Some(optics) = material.optical_properties() {
                self.sigma_a[slot] = optics.absorption_m_inv;
                self.sigma_s[slot] = optics.reduced_scattering_m_inv;
                declared += 1;
            }
        }
        self.upload_optical_params(queue);
        declared
    }

    /// Removes the SI contract, returning every physical path to the legacy
    /// dimensionless convention.
    ///
    /// Exists so a caller can compare the two side by side at runtime --
    /// `examples/gpu/basic_fluids_gpu.rs` cycles them with the O key. The
    /// per-material optical coefficients are NOT reset: they are expressed
    /// in `m^-1` for the contract path and in dimensionless units for the
    /// legacy one, so a caller switching back has to set the ones it wants.
    pub fn clear_physical_render_contract(&mut self, queue: &wgpu::Queue) {
        self.physical_render_contract = None;
        let params = PhysicalRenderParams {
            emission: [self.emission_reference_k, 0.0, 0.0, 0.0],
            ..bytemuck::Zeroable::zeroed()
        };
        queue.write_buffer(
            &self.physical_render_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// Sets the exposure anchor for thermal emission: the temperature whose
    /// blackbody renders at full brightness.
    ///
    /// Emission itself is not tunable -- its colour is Planck's law through
    /// the CIE observer (`energy::radiation`) and its brightness is
    /// Stefan-Boltzmann's `T^4`. What a scene still has to state is its
    /// exposure, exactly as a photographer does: a 1200 K ember and a 5772 K
    /// photosphere differ by a factor of 500 in radiance, and no single
    /// setting shows both. Pick the temperature the scene is "shot" for.
    ///
    /// A full [`PhysicalRenderContract`] makes this redundant: with a real
    /// display-white radiance in `W/(m^2 sr)`, the exposure is measured
    /// rather than chosen, and the shaders use that instead.
    pub fn set_emission_reference_temperature(&mut self, queue: &wgpu::Queue, kelvin: f32) {
        if !kelvin.is_finite() || kelvin <= 0.0 {
            return;
        }
        self.emission_reference_k = kelvin;
        match self.physical_render_contract {
            Some(contract) => self.set_physical_render_contract(queue, contract),
            None => {
                let params = PhysicalRenderParams {
                    emission: [kelvin, 0.0, 0.0, 0.0],
                    ..bytemuck::Zeroable::zeroed()
                };
                queue.write_buffer(
                    &self.physical_render_params_buf,
                    0,
                    bytemuck::bytes_of(&params),
                );
            }
        }
    }

    /// The thermal-emission exposure anchor currently in use, in kelvin.
    pub fn emission_reference_temperature(&self) -> f32 {
        self.emission_reference_k
    }

    /// Mean display-white radiance in `W/(m^2 sr)`, or 0 when no contract is
    /// in force. Render passes whose bind group carries the shared physical
    /// uniform read it there; the light-diffusion sweep's does not, so it
    /// receives this scalar instead (see `LightDiffuseParams`).
    pub(super) fn display_white_mean(&self) -> f32 {
        match self.physical_render_contract {
            Some(contract) => {
                let white = contract.display_white_radiance_w_m2_sr();
                (white[0] + white[1] + white[2]) / 3.0
            }
            None => 0.0,
        }
    }

    /// Returns the physical contract, or `None` while this renderer is still
    /// using the legacy dimensionless optical convention.
    pub fn physical_render_contract(&self) -> Option<PhysicalRenderContract> {
        self.physical_render_contract
    }

    fn clear_color(&self) -> wgpu::Color {
        if let Some(contract) = self.physical_render_contract {
            let background = contract.background_radiance_w_m2_sr();
            let display_white = contract.display_white_radiance_w_m2_sr();
            return wgpu::Color {
                r: (background[0] / display_white[0]).clamp(0.0, 1.0) as f64,
                g: (background[1] / display_white[1]).clamp(0.0, 1.0) as f64,
                b: (background[2] / display_white[2]).clamp(0.0, 1.0) as f64,
                a: 1.0,
            };
        }
        wgpu::Color {
            r: 0.05,
            g: 0.05,
            b: 0.08,
            a: 1.0,
        }
    }

    pub fn set_color_mode(&mut self, mode: ColorMode) {
        self.color_mode = mode;
    }
    pub fn set_vel_scale(&mut self, s: f32) {
        self.vel_scale = s;
    }

    /// Per-particle von Mises stress for `ColorMode::ByStress` (see
    /// `stress_field`): compute it once per frame with
    /// `MaterialRegistry::von_mises_stress_field(&particles)` and hand it in
    /// before `render`. A length mismatch is not an error: out-of-range reads
    /// give 0.0 in `particle_color`, so newly added particles just show
    /// inert colouring until the field is recomputed.
    pub fn set_stress_field(&mut self, values: Vec<f32>) {
        self.stress_field = values;
    }
    pub fn set_stress_scale(&mut self, s: f32) {
        self.stress_scale = s;
    }

    /// Legacy dimensionless optical input.
    ///
    /// This setter predates the renderer's SI contract, and its values are
    /// consumed without a guaranteed metre path length. New physical callers
    /// must use [`Renderer::set_optical_coefficients_si`]. It remains during
    /// staged migration so existing scenes do not silently change appearance.
    ///
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

    /// Stores real SI absorption and reduced-scattering coefficients without
    /// scene-side multiplication by `dx_meters`, particle spacing, or any
    /// renderer-specific depth surrogate.
    ///
    /// A validated [`PhysicalRenderContract`] is still required before these
    /// coefficients can produce dimensionally meaningful optical depth.
    pub fn set_optical_coefficients_si(
        &mut self,
        queue: &wgpu::Queue,
        slot: usize,
        coefficients: OpticalCoefficientsSi,
    ) {
        let slot = slot % 16;
        self.sigma_a[slot] = coefficients.absorption_m_inv;
        self.sigma_s[slot] = coefficients.reduced_scattering_m_inv;
        self.upload_optical_params(queue);
    }

    /// Reduced scattering coefficient for `slot` -- see `OpticalTable`'s doc for
    /// what this represents physically (real subsurface scattering, single-
    /// scattering approximation) and its citation (Jacques 2013). Auto-
    /// uploads immediately -- see `set_optical_params`'s doc for why.
    pub fn set_optical_scattering(&mut self, queue: &wgpu::Queue, slot: usize, sigma_s: f32) {
        self.sigma_s[slot % 16] = sigma_s;
        self.upload_optical_params(queue);
    }

    /// Specular Fresnel base reflectance R0 for `slot` -- see `OpticalTable`'s doc
    /// for the real-but-bounded caveat (constant near-normal reflectance, no
    /// surface-normal-dependent angle term). Auto-uploads immediately -- see
    /// `set_optical_params`'s doc for why.
    pub fn set_specular_r0(&mut self, queue: &wgpu::Queue, slot: usize, r0: f32) {
        self.specular_r0[slot % 16] = r0;
        self.upload_optical_params(queue);
    }

    /// Refractive index of `slot`'s dry solid/grain (quartz sand ~1.5),
    /// driving pore-fluid index-matching darkening in `ColorMode::ByPhysics`
    /// for any material and any scalar field wired to
    /// `Particle::scalar_field` (moisture or another saturating quantity).
    ///
    /// Wet porous materials darken because the pore fluid's index (water
    /// 1.33, Hecht "Optics") is closer to the grain's than air's (1.0), which
    /// reduces the mismatch that scatters light at each interface: measured
    /// on beach sand by Sadeghi et al. ("Measuring and Modeling the Effect of
    /// Surface Moisture on the Spectral Reflectance of Coastal Beach Sand",
    /// PMC4226492), treated for real-time rendering by Lagarde 2013 ("Water
    /// drop 3a: Physically based wet surfaces"). Scattering goes with the
    /// square of the mismatch, so the ratio at full saturation is
    /// `((n_solid - n_water) / (n_solid - n_air))^2`, interpolated by the
    /// particle's `scalar_field` in [0, 1] in `color.rs`'s `particle_color`.
    ///
    /// `1.0` (default, same as air) means no contrast. The GPU render path
    /// (`prep_instances.wgsl`) does not mirror this yet; CPU path only.
    pub fn set_refractive_index(&mut self, slot: usize, n: f32) {
        self.refractive_index[slot % 16] = n;
    }

    fn upload_optical_params(&self, queue: &wgpu::Queue) {
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
            &self.holds_shape,
        );
    }

    /// Snapshots the current GPU-resident particle positions for the render-
    /// interpolation path -- call once per render-frame's physics-step batch,
    /// BEFORE stepping (mirrors the CPU `basic_fluids.rs` `prev_x` convention:
    /// skip when `steps==0`, the last snapshot stays valid since nothing
    /// moved). Pass the resulting `FixedStepController::interpolation_alpha()`
    /// to `render_gpu`'s own `alpha` param afterward.
    ///
    /// GPU demos step on `FixedStepController`; without interpolating the
    /// leftover fractional step, uneven per-step cost showed as uneven
    /// position jumps ("sudden acceleration", see
    /// `RenderConfig::interp_alpha`). No CPU readback: a small compute pass
    /// copies `Particle::x` into a packed GPU buffer that
    /// `prep_instances.wgsl` blends against.
    pub fn snapshot_particle_positions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_buf: &wgpu::Buffer,
        particle_count: usize,
    ) {
        if particle_count == 0 {
            return;
        }
        self.ensure_capacity(device, particle_count);
        queue.write_buffer(
            &self.snapshot_config_buf,
            0,
            bytemuck::bytes_of(&SnapshotConfig {
                particle_count: particle_count as u32,
                _pad: [0; 3],
            }),
        );
        let snapshot_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snapshot_bg"),
            layout: &self.snapshot_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.prev_positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.snapshot_config_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("snapshot_positions"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("snapshot_positions"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.snapshot_pipeline);
            cp.set_bind_group(0, &snapshot_bg, &[]);
            cp.dispatch_workgroups((particle_count as u32).div_ceil(PREP_WG), 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    // ── GPU compute render path ────────────────────────────────────────────────

    /// Zero-readback GPU render. No `sync_particles_blocking()` needed.
    ///
    /// `params.interp_alpha`: pass `1.0` for the old, unblended behavior
    /// (render exactly the current GPU particle state), or a real
    /// `FixedStepController::interpolation_alpha()` reading -- combined with a
    /// prior same-frame `snapshot_particle_positions` call -- to smooth motion
    /// between fixed physics steps. See that method's doc.
    pub fn render_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: GpuRenderParams<'_>,
    ) {
        let GpuRenderParams {
            particle_buf,
            particle_count,
            output_view,
            clear,
            interp_alpha,
        } = params;
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
                interp_alpha,
            }),
        );
        write_optical_table(
            queue,
            &self.optical_table_buf,
            &self.sigma_a,
            &self.sigma_s,
            &self.specular_r0,
            &self.holds_shape,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.physical_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.prev_positions_buf.as_entire_binding(),
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
        for (i, p) in particles.iter().enumerate() {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(&p, i),
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
        for (i, p) in particles.iter().enumerate() {
            self.scratch.push(InstanceData {
                deform_col0: p.deformation_gradient.x_axis.to_array(),
                deform_col1: p.deformation_gradient.y_axis.to_array(),
                position: p.x.to_array(),
                _pad: [0.0; 2],
                color: self.particle_color(p, i),
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
            self.prev_positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render_prev_positions"),
                size: (count * mem::size_of::<[f32; 2]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE,
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
            wgpu::LoadOp::Clear(self.clear_color())
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
// that file's doc comment.
mod color;
use color::write_optical_table;

// Test suite split into its own file -- was ~150 of this file's ~930 lines,
// same pattern as `gpu/solver/device_lost_tests.rs`.
#[cfg(test)]
mod tests;

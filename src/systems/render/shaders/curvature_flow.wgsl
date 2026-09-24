// Curvature-flow surface reconstruction (van der Laan, Green, Sainz 2009,
// "Screen Space Fluid Rendering with Curvature Flow") -- a finer,
// resolution-independent alternative to `grid_volume.wgsl`'s coarse
// physics-grid sampling. `grid_volume.wgsl` samples the solver's own P2G
// mass field directly, so its quality is capped by `grid_res` (the physics
// resolution); this splats particles onto a dedicated, finer auxiliary
// buffer (`SurfaceParams::surface_res`, independent of the physics grid),
// then applies mean-curvature smoothing to merge nearby particle
// footprints into one continuous surface instead of a "particle soup" of
// discrete blobs.
//
// Three passes, ping-ponged across two plain `array<f32>` buffers (matches
// `grid_volume.wgsl`'s own `array<u32>` bitcast pattern rather than
// introducing a new Texture2D resource type):
//
//   1. `clear_surface_main` + `splat_density_main`: scatter each particle's
//      mass onto the finer buffer using the same quadratic B-spline kernel
//      (Steffen & Kirby 2008) P2G already uses for the physics grid, at a
//      different resolution. Uses the same fixed-point atomic-add
//      technique `p2g.wgsl` uses (WebGPU has no atomic<f32>).
//   2. `curvature_iterate_main`: one thread per surface cell, the standard
//      closed-form mean curvature of an implicit function,
//      κ = (Dxx·Dy² − 2·Dx·Dy·Dxy + Dyy·Dx²) / (Dx²+Dy²)^1.5, stepped via
//      `D_new = D + dt·κ` (render_plan's own cited equation,
//      `∂D/∂t = ∇·(∇D/|∇D|)` -- that divergence IS the closed-form κ above,
//      no extra |∇D| factor). `MAX_KAPPA` clamps this equation's known
//      numerical failure mode in near-flat regions (gradient → 0 makes κ's
//      denominator → 0).
//   3. `fs_main`: bilinear-sample the final smoothed buffer, gradient-
//      based normal + Lambertian shading + Beer-Lambert absorption -- the
//      same technique `grid_volume.wgsl`'s own fragment shader uses,
//      against the new, finer buffer.
//
// v1 scope: single unified density surface with ONE material color for the
// whole call -- correct for any single-dominant-material scene (fluid,
// sand, snow, an elastic body).
//
// **Two-phase extension**: `SurfaceParams::phase_filter_material_id`
// (-1 = no filter, the original v1 behavior, unchanged) lets
// `splat_density_main` accumulate ONLY particles matching one material ID
// into its own separate buffer -- calling the whole clear/splat/convert/
// iterate pipeline TWICE (once per phase, into two independent buffer
// sets) and combining at `fs_main_dual_phase` gives each phase its OWN
// independently-smoothed surface, not one merged blob at a sand/
// water-style interface. Grounded in the Volume-of-Fluid phase-fraction
// concept (Hirt & Nichols 1981, "Volume of fluid (VOF) method for the
// dynamics of free boundaries," J. Comput. Phys. 39:201-225 -- each phase
// gets its own fractional field, not a single shared density) and this
// engine's own existing per-material `material_mass` precedent
// (`grid_volume.wgsl`); not a replication of Zhang et al. 2024's specific
// "phase fraction texture" implementation (that paper is paywalled and
// unread) -- only its abstract-level concept (the phase-fraction name and
// the "keep phases visually distinct" goal) is reused here. v2 scope:
// exactly 2 simultaneous phases, not the full general N-material case --
// matches the actual 2-material scenes (`mixture_sand_water.rs`) this
// engine currently has.
//
// **N-material extension, single-phase path only**: the 2-phase mechanism
// above doesn't scale further -- `fs_main_dual_phase`'s own bind group is
// already at the WebGPU-guaranteed minimum of 8 storage buffers (see its
// own doc), so a 3rd hand-duplicated buffer set would break portability.
// Instead, `splat_density_main` now ALSO scatters each particle's mass
// into its own `material_id % 16` slot of a flat per-cell array,
// `surface_material_mass` (`surface_res² × 16`, one extra storage buffer
// total, not per-phase) -- the same technique `grid_volume.wgsl` uses for
// its own `material_mass`/`dominant_material` (majority-mass-per-cell
// wins; reads the fixed-point atomic buffer as plain `f32` for ordering
// comparisons only -- IEEE 754's non-negative bit-pattern-to-value mapping
// is monotonic, so this reinterpretation is safe). The existing single,
// TOTAL density field and its 12-iteration curvature-smoothing pipeline
// are completely unchanged -- only which `OpticalTable` slot colors each
// final pixel becomes per-cell instead of one caller-chosen slot for the
// whole surface. `fs_main_dual_phase` does not get this (unrelated
// mechanism, still exactly 2 phases, untouched).
//
// **Anisotropic splat extension**: `splat_density_main` now transforms
// each candidate cell's offset through the particle's own inverted
// `deformation_gradient` (regularized, see `regularize_deformation`)
// before evaluating the isotropic B-spline kernel there -- making the
// resulting screen-space footprint stretch/rotate to match the particle's
// current physical shape, the same `F` already used for mode-1's
// per-particle anisotropic quad (`render_particles.wgsl`), reused here
// instead of an invented anisotropy source. When `F` is exactly identity
// this is bit-for-bit the same isotropic kernel as before. Naive
// deformation-gradient-driven anisotropy is a documented technique in
// recent MPM-adjacent rendering research (e.g. MPM-Gaussian-splat
// hybrids), but becomes ineffective under large deformations per that
// same literature -- `regularize_deformation` clamps each column's length
// to a bounded range before use, the same fix production systems use
// (clamp F's singular values). This is a different, MPM-specific
// technique from Yu & Turk 2013's neighbor-PCA anisotropic kernels (the
// classic SPH approach, cited in render_plan's own doc) -- that needs a
// neighbor search this splat pass doesn't have; this reuses state MPM
// particles already carry for free instead.

struct Particle {
    x:                    vec2<f32>,
    v:                    vec2<f32>,
    velocity_gradient:    mat2x2<f32>,
    deformation_gradient: mat2x2<f32>,
    mass:                 f32,
    initial_volume:       f32,
    volume:               f32,
    density:              f32,
    material_id:          u32,
    plastic_volume_ratio: f32,
    hardening_scale:      f32,
    friction_hardening:   f32,
    log_volume_strain:    f32,
    temperature:          f32,
    user_tag:             u32,
    activation:           f32,
    activation_dir:       vec2<f32>,
    muscle_group_id:      u32,
    contact_group:        u32,
    sleeping:             u32,
    pinned:               u32,
    scalar_field:         f32,
    internal_pressure:    f32,
}

struct SurfaceParams {
    grid_res:       u32,
    surface_res:    u32, // = grid_res * SURFACE_RES_MULTIPLIER, finer than the physics grid
    particle_count: u32,
    // -1 = no filter (v1 behavior: every particle contributes). >= 0 =
    // only particles with this exact material_id are splatted -- the
    // two-phase extension's own per-phase filter (see module doc).
    phase_filter_material_id: i32,
    // N-material extension (see module doc), single-phase path only. 0 =
    // disabled: clear/splat skip the extra 16-slot-per-cell atomic work
    // entirely (zero cost). 1 = enabled. Always 0 on the dual-phase path.
    material_mass_enabled: u32,
    // Real simulation timestep (`SimConfig::dt`, the SAME value the solver
    // itself steps with -- not a render-frame time, which this pass has no
    // way to know and which is the wrong physical quantity anyway: what
    // matters for a real motion-stretch footprint is how far the particle
    // moved during the physics step that produced its CURRENT `v`, not how
    // long the screen took to redraw). Used only by `splat_density_main`'s
    // real velocity-stretch extension (see that function's own doc) --
    // every other pass sharing this struct ignores the field, same existing
    // convention `phase_filter_material_id`/`material_mass_enabled` already
    // use for pass-specific fields.
    dt: f32,
    // Splat kernel width in PHYSICS-GRID CELLS, for `splat_density_main`
    // only. 1.0 reproduces the original behaviour exactly (the full MPM
    // quadratic B-spline support, 1.5 cells radius).
    //
    // Why this is separate from the physics kernel: the MPM B-spline width
    // is correct for scattering MASS TO THE GRID, but it is not the right
    // width for DRAWING a particle. A material point stands for a patch of
    // fluid about `particle_spacing` across; splatting it 3 cells wide makes
    // every particle overlap ~6 of its neighbours at 0.5-cell spacing, and
    // the overlapping discs merge into the "blobby metaball" look. Narrowing
    // it toward the real particle spacing gives a sharper surface AND costs
    // less (the splat loop is O(radius^2)).
    splat_width_cells: f32,
    // Yu & Turk 2013 neighbourhood-fitted kernel anisotropy, for
    // `splat_density_main` only. 0.0 = disabled, the splat behaves exactly as
    // it did before this term existed (bit for bit -- the fit is skipped
    // entirely, not merely multiplied by zero). 1.0 = the fitted shape at full
    // strength. Intermediate values blend toward identity, which is the knob
    // for dialling the effect back without an all-or-nothing switch.
    anisotropy_strength: f32,
}

// Yu & Turk's `k_r`, the upper bound on the fitted kernel's axis ratio (4 in
// the paper; the same role NVIDIA Flex's `anisotropyMin`/`anisotropyMax` play).
// Without it a near-degenerate neighbourhood -- a particle with two collinear
// neighbours -- fits an arbitrarily thin kernel that renders as a sliver
// instead of a surface.
const ANISOTROPY_MAX_AXIS_RATIO: f32 = 4.0;

struct SurfaceRenderParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    // Real light direction, sourced from `SimConfig::light_dir` via
    // `Renderer::set_light_dir` -- see `grid_volume.wgsl`'s own
    // `GridVolumeParams::light_dir` doc for why this replaced a value
    // hardcoded separately in each fragment shader.
    light_dir: vec2<f32>,
    surface_res: u32,
    mass_floor: f32,
    // Fallback slot used when material_mass_enabled is 0 -- unchanged v1
    // behavior for every existing caller.
    material_slot: u32,
    // N-material extension (see module doc). Was _pad1: f32, an unused pad
    // field -- same offset, same size, struct stays 48 bytes.
    material_mass_enabled: u32,
    // Real per-cell mass scale this scene's densities are expressed in --
    // the SAME value `BandHysteresisParams` already carries. `fs_main`
    // divides by it before quantizing, so the band range below is a
    // dimensionless "how many reference cell-masses deep" instead of an
    // absolute mass. Was half of `_pad2`; struct stays 48 bytes.
    reference_cell_mass: f32,
    // Floor on optical depth for edge color, in those same dimensionless
    // band units. Was the other half of `_pad2`.
    edge_reference_depth: f32,
}

struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

// Number of flat depth/color bands `fs_main`, `shade_phase`, and
// `band_hysteresis_step_main` all quantize optical depth into -- a tuned,
// retunable real-time-shading choice (not a derived physical value), shared
// file-scope so `fs_main` and `shade_phase` can't independently drift the
// way they once needed a "MUST match" comment to guard against. Must also
// stay equal to `mod.rs`'s own `DEPTH_BANDS` (Rust can't share this WGSL
// const directly) and `grid_volume.wgsl`'s own copy (a separate shader
// module, same constraint).
const DEPTH_BANDS: f32 = 4.0;
const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const DENSITY_ATOMIC_SCALE: f32 = 1000000.0;
// Deliberately smaller fixed-point scale than DENSITY_ATOMIC_SCALE for the
// temperature atomic (see `surface_temp_atomic`'s own doc): the quantity
// accumulated is `w * mass * temperature`, not `w * mass` -- temperature
// (up to `grid_volume.wgsl`'s own ~5000K blackbody-normalization ceiling)
// multiplies the same per-particle mass contribution DENSITY_ATOMIC_SCALE
// was tuned for, so reusing that scale risks i32 overflow in a dense, hot
// cell. Chosen so the worst case density scale already tolerates (local
// accumulated mass up to i32::MAX/DENSITY_ATOMIC_SCALE ~= 2147) times a
// 5000K ceiling still stays under i32::MAX: 2147 * 5000 * 100 ~= 1.07e9,
// half of i32::MAX ~= 2.147e9.
const TEMP_ATOMIC_SCALE: f32 = 100.0;
// Deliberately smaller fixed-point scale than DENSITY_ATOMIC_SCALE for the
// volume-preserving-correction TOTALS (`pre_total_atomic`/
// `post_total_atomic`): DENSITY_ATOMIC_SCALE is tuned for accumulating
// into ONE cell (bounded by local particle overlap), not for summing
// across an entire scene's cells/particles (unbounded -- thousands of
// cells, each near i32::MAX/DENSITY_ATOMIC_SCALE's own headroom); summing
// at that scale overflows i32 (a negative total is the signature) well
// before a real-sized scene. This scale supports a global total up to
// i32::MAX/TOTAL_ATOMIC_SCALE ~= 2.15 million mass units before overflow.
const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
// N-material extension (see module doc) -- matches grid_volume.wgsl's own
// copy of the same constant exactly (Rust-side source of truth:
// gpu::step_params::subsystems::MAX_RENDER_MATERIAL_SLOTS).
const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

// Numerical safeguard (see module doc's "anisotropic splat extension"
// section): clamps each column's LENGTH (a cheap proxy for that axis'
// stretch/singular value) to a bounded range, preserving its direction --
// the same fix production MPM/Gaussian-splat renderers use when naive
// F-driven anisotropy is unstable under large deformation.
const MIN_STRETCH: f32 = 0.3;
const MAX_STRETCH: f32 = 3.0;

fn regularize_deformation(f: mat2x2<f32>) -> mat2x2<f32> {
    let len0 = max(length(f[0]), 1.0e-5);
    let len1 = max(length(f[1]), 1.0e-5);
    let scale0 = clamp(len0, MIN_STRETCH, MAX_STRETCH) / len0;
    let scale1 = clamp(len1, MIN_STRETCH, MAX_STRETCH) / len1;
    return mat2x2<f32>(f[0] * scale0, f[1] * scale1);
}

// Standard closed-form 2x2 matrix inverse (adjugate/determinant) -- WGSL
// has no built-in `inverse()`. `f` should already be `regularize_
// deformation`'s output, so `det` is bounded away from zero by
// construction (each column's own length is clamped to >= MIN_STRETCH).
fn inverse2x2(f: mat2x2<f32>) -> mat2x2<f32> {
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    // NOT `sign(det) * max(abs(det), eps)` -- `sign(0.0)` is 0.0 in WGSL
    // (IEEE convention), which would leave a genuine zero divisor for an
    // exactly-degenerate (rank-deficient) `f` -- a real edge case
    // `regularize_deformation`'s own length clamp does NOT rule out (it
    // bounds magnitude, not whether the two columns are parallel).
    // `select` guarantees a nonzero divisor unconditionally.
    let safe_det = max(abs(det), 1.0e-4) * select(-1.0, 1.0, det >= 0.0);
    return mat2x2<f32>(
        f[1][1] / safe_det, -f[0][1] / safe_det,
        -f[1][0] / safe_det, f[0][0] / safe_det,
    );
}

// ── Pass 1: clear + splat ────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> surface_atomic: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> splat_params: SurfaceParams;
// Real mass-weighted temperature scatter, same fixed-point atomic technique
// as `surface_atomic` above (WebGPU has no atomic<f32>). Accumulates
// `w * mass * temperature` at the same indices/weights as the density
// scatter below -- `fs_main` recovers a real mass-weighted average
// temperature by dividing this by the settled density, the SAME real
// blackbody-emission gap `grid_volume.wgsl` already closed (see that
// shader's own doc for the formula reused verbatim here). Single-phase
// `fs_main` only -- `fs_main_dual_phase` is deliberately NOT wired to this
// (see Pass 3b's own bind group: it's already at the real, confirmed
// WebGPU-guaranteed minimum of 8 storage buffers per fragment stage, adding
// a 9th would exceed that guarantee).
@group(0) @binding(3) var<storage, read_write> surface_temp_atomic: array<atomic<i32>>;
// Real volume-preserving correction (see "Pass 1d" doc below): the TRUE
// total particle mass (ground truth, known directly from real physics, not
// derived from the splat kernel) accumulated once per real particle here --
// what the settled surface SHOULD sum to before curvature-flow's own real
// shrinkage bias distorts it.
@group(0) @binding(4) var<storage, read_write> pre_total_atomic: array<atomic<i32>>;
// Cleared here too (needs zeroing every frame like the others above), but
// filled by a separate later pass (`post_total_reduce_main`) after the
// curvature-flow iterations settle -- this pass has no use for it itself.
@group(0) @binding(5) var<storage, read_write> post_total_atomic: array<atomic<i32>>;
// N-material extension (see module doc): flat per-cell array, 16 slots per
// surface cell, one particle's mass lands in its own `material_id % 16`
// slot. Read back in `fs_main` as plain `array<f32>` for ordering
// comparisons only -- same real, already-shipped bit-reinterpretation
// `grid_volume.wgsl`'s own `material_mass` already relies on.
@group(0) @binding(6) var<storage, read_write> surface_material_mass_atomic: array<atomic<i32>>;
// Yu & Turk 2013 anisotropic-kernel support: per-PHYSICS-grid-cell weighted
// moments of the particle distribution, 6 scalars per cell laid out flat as
// [m0, m1x, m1y, m2xx, m2xy, m2yy]. Filled by `splat_moments_main`, read back
// by `splat_density_main` to fit each particle's own kernel shape. Same
// fixed-point atomic technique as every other accumulator in this file
// (WebGPU has no atomic<f32>).
//
// Moments are accumulated relative to each cell's OWN centre, not in absolute
// grid coordinates. A second moment of a coordinate of order 64 is of order
// 4096, and recovering a variance of order 1 by subtracting the squared mean
// from that loses most of an f32 to cancellation; cell-relative offsets never
// exceed the kernel half-width, so both the float math and the fixed-point
// range stay small. `moments_at` shifts them into a particle's own frame.
@group(0) @binding(7) var<storage, read_write> surface_moments_atomic: array<atomic<i32>>;

const MOMENTS_PER_CELL: u32 = 6u;
const MOMENT_ATOMIC_SCALE: f32 = 1.0e6;

@compute @workgroup_size(64, 1, 1)
fn clear_surface_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // Dispatched over max(surface cells, moment words) -- the moment buffer is
    // sized off `grid_res` while everything else here is sized off
    // `surface_res`, and neither is guaranteed to be the larger of the two at
    // every `surface_res_multiplier`. Each store guards its own range.
    let moment_words = splat_params.grid_res * splat_params.grid_res * MOMENTS_PER_CELL;
    if idx < moment_words {
        atomicStore(&surface_moments_atomic[idx], 0);
    }
    if idx >= splat_params.surface_res * splat_params.surface_res { return; }
    atomicStore(&surface_atomic[idx], 0);
    atomicStore(&surface_temp_atomic[idx], 0);
    if idx == 0u {
        atomicStore(&pre_total_atomic[0], 0);
        atomicStore(&post_total_atomic[0], 0);
    }
    if splat_params.material_mass_enabled != 0u {
        let mm_base = idx * MAX_RENDER_MATERIAL_SLOTS;
        for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
            atomicStore(&surface_material_mass_atomic[mm_base + s], 0);
        }
    }
}

// ── Pass 1b: neighbourhood moments (Yu & Turk 2013) ──────────────────────────
//
// Scatters the weighted moments of the particle distribution onto the PHYSICS
// grid, so `splat_density_main` can fit each particle an oriented kernel from
// its own neighbourhood instead of from its `deformation_gradient`. See
// `render::anisotropy` (Rust) for the full rationale and the unit-tested
// reference implementation this is a translation of.
//
// Short version: `F` is isotropic for a liquid by constitutive definition (no
// shear memory), so the existing F-driven anisotropy collapses to a scaled
// identity, every particle splats the same symmetric blob, and identical blobs
// on a regular spawn lattice interfere -- the visible speckle/crosshatch. A
// neighbourhood fit does not degenerate that way: on a flat sheet the
// neighbours lie along the surface, so the kernel flattens along it.
//
// Doing it as a grid scatter/gather rather than a neighbour list is what makes
// it affordable here: no neighbour search, no sorting, one extra O(particles)
// pass reusing the B-spline weights the splat already computes.
//
// Deliberately NOT phase-filtered, unlike `splat_density_main`: the shape of a
// particle's neighbourhood is a geometric fact about where matter actually is,
// so every particle contributes regardless of which phase is being splatted.
// It also keeps this a single shared buffer rather than one per phase.
@compute @workgroup_size(64, 1, 1)
fn splat_moments_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= splat_params.particle_count { return; }
    let p = particles[i];

    let res = i32(splat_params.grid_res);
    let base = vec2<i32>(i32(floor(p.x.x)), i32(floor(p.x.y)));
    let radius = i32(ceil(BSPLINE_OUTER_LIMIT));
    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let cx = base.x + dx;
        if cx < 0 || cx >= res { continue; }
        for (var dy: i32 = -radius; dy <= radius; dy++) {
            let cy = base.y + dy;
            if cy < 0 || cy >= res { continue; }
            let centre = vec2<f32>(f32(cx) + CELL_CENTER_OFFSET, f32(cy) + CELL_CENTER_OFFSET);
            // Offset of this particle from the cell centre -- the frame every
            // moment below is accumulated in (see the buffer's own doc).
            let d = p.x - centre;
            let w = bspline_w(d.x) * bspline_w(d.y);
            if w <= 0.0 { continue; }
            let wm = w * p.mass;
            let b = (u32(cy) * splat_params.grid_res + u32(cx)) * MOMENTS_PER_CELL;
            atomicAdd(&surface_moments_atomic[b], i32(round(wm * MOMENT_ATOMIC_SCALE)));
            atomicAdd(&surface_moments_atomic[b + 1u], i32(round(wm * d.x * MOMENT_ATOMIC_SCALE)));
            atomicAdd(&surface_moments_atomic[b + 2u], i32(round(wm * d.y * MOMENT_ATOMIC_SCALE)));
            atomicAdd(
                &surface_moments_atomic[b + 3u],
                i32(round(wm * d.x * d.x * MOMENT_ATOMIC_SCALE)),
            );
            atomicAdd(
                &surface_moments_atomic[b + 4u],
                i32(round(wm * d.x * d.y * MOMENT_ATOMIC_SCALE)),
            );
            atomicAdd(
                &surface_moments_atomic[b + 5u],
                i32(round(wm * d.y * d.y * MOMENT_ATOMIC_SCALE)),
            );
        }
    }
}

// Symmetric 2x2 covariance -> area-preserving kernel shape matrix.
//
// A direct translation of `render::anisotropy::kernel_shape_from_neighbors`'s
// second half, which carries the unit tests (flat sheet elongates along the
// surface, isotropic bulk returns identity, det stays 1, rotation
// equivariance). 2D is what makes this cheap: the 3D case needs an iterative
// symmetric eigensolver, this is a quadratic root plus a normalization.
fn shape_from_covariance(a: f32, b: f32, c: f32, max_ratio: f32) -> mat2x2<f32> {
    let half_trace = 0.5 * (a + c);
    // Half-difference form: cannot go negative through cancellation the way
    // `trace^2/4 - det` can for a nearly degenerate matrix.
    let half_diff = 0.5 * (a - c);
    let disc = sqrt(max(half_diff * half_diff + b * b, 0.0));
    let l_major = half_trace + disc;
    let l_minor = half_trace - disc;
    if l_major <= 1.0e-12 {
        return mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    }

    // Eigenvector of `l_major` for [[a,b],[b,c]] is (l_major - c, b); when `b`
    // vanishes the matrix is already diagonal and that degenerates, so take
    // the axis of the larger diagonal entry directly.
    var axis = vec2<f32>(1.0, 0.0);
    if abs(b) > 1.0e-12 {
        axis = vec2<f32>(l_major - c, b);
    } else if a < c {
        axis = vec2<f32>(0.0, 1.0);
    }
    let axis_len = length(axis);
    if axis_len > 1.0e-12 {
        axis = axis / axis_len;
    } else {
        axis = vec2<f32>(1.0, 0.0);
    }

    // Kernel extent goes as the standard deviation: covariance eigenvalues are
    // squared lengths and this matrix is applied to lengths.
    let major = sqrt(max(l_major, 0.0));
    var minor = sqrt(max(l_minor, 0.0));
    // Bound the aspect ratio BEFORE normalizing, so a degenerate fit becomes a
    // bounded ellipse rather than an infinitely thin sliver.
    minor = max(minor, major / max(max_ratio, 1.0));
    if major <= 1.0e-12 {
        return mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    }
    // Unit determinant: strip size, keep orientation and aspect. This is what
    // makes the result independent of particle spacing (a covariance scales as
    // length^2 and this divides that scale straight back out) and what lets it
    // compose with `F` without changing the splat's footprint area, hence
    // without disturbing deposited mass or the volume-preserving correction.
    let norm = sqrt(major * minor);
    if norm <= 1.0e-12 {
        return mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let s_major = major / norm;
    let s_minor = minor / norm;
    let e0 = axis;
    let e1 = vec2<f32>(-e0.y, e0.x);
    return mat2x2<f32>(
        s_major * e0.x * e0 + s_minor * e1.x * e1,
        s_major * e0.y * e0 + s_minor * e1.y * e1,
    );
}

// Gathers the moments scattered above back at an arbitrary position and
// returns the fitted, area-preserving kernel shape there.
//
// Each cell stores its moments about its OWN centre, so they are shifted into
// this position's frame on the way in (`d_here = d_cell + s`, expanded per
// moment order) before mean and covariance are recovered.
fn kernel_shape_at(x: vec2<f32>, max_ratio: f32) -> mat2x2<f32> {
    let res = i32(splat_params.grid_res);
    let base = vec2<i32>(i32(floor(x.x)), i32(floor(x.y)));
    let radius = i32(ceil(BSPLINE_OUTER_LIMIT));

    var m0 = 0.0;
    var m1 = vec2<f32>(0.0, 0.0);
    var mxx = 0.0;
    var mxy = 0.0;
    var myy = 0.0;
    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let cx = base.x + dx;
        if cx < 0 || cx >= res { continue; }
        for (var dy: i32 = -radius; dy <= radius; dy++) {
            let cy = base.y + dy;
            if cy < 0 || cy >= res { continue; }
            let centre = vec2<f32>(f32(cx) + CELL_CENTER_OFFSET, f32(cy) + CELL_CENTER_OFFSET);
            let s = centre - x;
            let u = bspline_w(s.x) * bspline_w(s.y);
            if u <= 0.0 { continue; }
            let b = (u32(cy) * splat_params.grid_res + u32(cx)) * MOMENTS_PER_CELL;
            let c0 = f32(atomicLoad(&surface_moments_atomic[b])) / MOMENT_ATOMIC_SCALE;
            if c0 <= 0.0 { continue; }
            let c1x = f32(atomicLoad(&surface_moments_atomic[b + 1u])) / MOMENT_ATOMIC_SCALE;
            let c1y = f32(atomicLoad(&surface_moments_atomic[b + 2u])) / MOMENT_ATOMIC_SCALE;
            let c2xx = f32(atomicLoad(&surface_moments_atomic[b + 3u])) / MOMENT_ATOMIC_SCALE;
            let c2xy = f32(atomicLoad(&surface_moments_atomic[b + 4u])) / MOMENT_ATOMIC_SCALE;
            let c2yy = f32(atomicLoad(&surface_moments_atomic[b + 5u])) / MOMENT_ATOMIC_SCALE;

            m0 += u * c0;
            m1 += u * (vec2<f32>(c1x, c1y) + s * c0);
            mxx += u * (c2xx + 2.0 * s.x * c1x + s.x * s.x * c0);
            mxy += u * (c2xy + s.x * c1y + s.y * c1x + s.x * s.y * c0);
            myy += u * (c2yy + 2.0 * s.y * c1y + s.y * s.y * c0);
        }
    }
    if m0 <= 1.0e-12 {
        return mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    }

    // The weighted mean is deliberately NOT assumed to sit at `x`: in the bulk
    // it nearly does, but at a free surface the neighbourhood centroid sits
    // measurably inward, and that offset is exactly the surface signal this
    // technique exists to pick up.
    let mean = m1 / m0;
    let cxx = mxx / m0 - mean.x * mean.x;
    let cxy = mxy / m0 - mean.x * mean.y;
    let cyy = myy / m0 - mean.y * mean.y;
    return shape_from_covariance(cxx, cxy, cyy, max_ratio);
}

@compute @workgroup_size(64, 1, 1)
fn splat_density_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= splat_params.particle_count { return; }
    let p = particles[i];
    // Two-phase filter (see module doc / SurfaceParams): -1 = accumulate
    // everyone (v1 behavior); >= 0 = only this exact material_id, letting
    // two independent calls build two independent, later-combined phases.
    if splat_params.phase_filter_material_id >= 0
        && p.material_id != u32(splat_params.phase_filter_material_id) {
        return;
    }
    // Real ground-truth total mass for THIS phase -- once per particle
    // thread (not once per kernel-touched cell below), see
    // `pre_total_atomic`'s own doc.
    atomicAdd(&pre_total_atomic[0], i32(round(p.mass * TOTAL_ATOMIC_SCALE)));
    // Particle position, converted from physics-grid units into the finer
    // surface buffer's own coordinate system (same origin, finer spacing).
    let scale = f32(splat_params.surface_res) / f32(splat_params.grid_res);
    let sp = p.x * scale;

    // `bspline_w`'s 0.5/1.5 cutoffs are expressed in GRID-cell units --
    // reusing them directly against SURFACE-cell offsets shrinks the
    // kernel's reach by a factor of `scale` (e.g. at scale=3 a nominal
    // "1.5 grid-cell" radius becomes only 0.5 grid-cells wide, smaller than
    // typical particle spacing, so neighboring footprints never overlap).
    // Divide the surface-space offset by `scale` before calling `bspline_w`
    // so its cutoffs stay meaningful in grid-unit distance regardless of
    // `surface_res`.
    //
    // Anisotropic extension (see module doc): the offset is ALSO
    // transformed through the particle's own regularized, inverted `F`
    // before the kernel is evaluated -- when `F` is identity this reduces
    // to exactly the isotropic case above (bit-for-bit unchanged).
    // `max_stretch` widens the scatter loop's radius to cover the
    // (potentially larger, in a stretched direction) footprint this
    // implies -- a fixed isotropic radius is only correct when F==identity.
    //
    // Real velocity-stretch extension (2026-08-11): a SECOND, complementary
    // anisotropy source composed with F above -- F captures ACCUMULATED
    // shape change (already real), this captures INSTANTANEOUS motion (a
    // fast splash droplet stretching along its own flight path, distinct
    // from any shape deformation it's separately undergoing). Real, cited
    // technique (Codrops Feb-2025 WebGPU fluid renderer; "Real-time
    // deformable droplet rendering," 2025 preprint -- both drive particle
    // deformation from velocity, not just accumulated shape).
    //
    // `stretch_factor` is a real, DERIVED dimensionless ratio, not a tuned
    // constant: `|v|*dt` is the real physical distance (grid units) this
    // particle moved during the physics step that produced its current
    // `v` (`splat_params.dt` is `SimConfig::dt`, threaded through from the
    // real solver, not a render-frame time); dividing by
    // BSPLINE_OUTER_LIMIT (the kernel's own real half-width, already
    // defined above) gives "how many kernel-radii did it travel this
    // step" -- a particle moving less than one kernel-radius (the common
    // CFL-limited case) gets a near-1.0 factor, negligible extra stretch;
    // a genuinely fast splash droplet gets real, visible elongation along
    // its own velocity.
    let speed = length(p.v);
    var stretch_factor = 1.0;
    var v_dir = vec2<f32>(1.0, 0.0);
    if speed > 1.0e-5 {
        v_dir = p.v / speed;
        stretch_factor = 1.0 + (speed * splat_params.dt) / BSPLINE_OUTER_LIMIT;
    }
    let v_perp = vec2<f32>(-v_dir.y, v_dir.x);
    // Real, area-preserving anisotropic stretch matrix -- eigen-
    // decomposition form: eigenvector `v_dir` with eigenvalue
    // `stretch_factor` (elongate along real motion), eigenvector `v_perp`
    // with eigenvalue `1/stretch_factor` (compress perpendicular, same
    // real convention any anisotropic Gaussian/kernel construction uses to
    // avoid inflating the kernel's own total footprint area).
    let motion_col0 = stretch_factor * v_dir.x * v_dir + (1.0 / stretch_factor) * v_perp.x * v_perp;
    let motion_col1 = stretch_factor * v_dir.y * v_dir + (1.0 / stretch_factor) * v_perp.y * v_perp;
    let motion_stretch = mat2x2<f32>(motion_col0, motion_col1);

    // Yu & Turk neighbourhood fit, composed on top of the two existing
    // anisotropy sources rather than replacing them: for a solid, `F` carries
    // real accumulated shape and this adds surface alignment on top; for a
    // liquid, `F` is isotropic and this supplies all of the shape. Because
    // `det(shape) == 1` the composition rotates and elongates the footprint
    // without changing its AREA, so deposited mass -- and the volume-preserving
    // correction built on it -- are untouched. `anisotropy_strength = 0` leaves
    // the previous behaviour bit for bit.
    var shape = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    if splat_params.anisotropy_strength > 0.0 {
        let fitted = kernel_shape_at(p.x, ANISOTROPY_MAX_AXIS_RATIO);
        let t = clamp(splat_params.anisotropy_strength, 0.0, 1.0);
        shape = mat2x2<f32>(
            mix(vec2<f32>(1.0, 0.0), fitted[0], t),
            mix(vec2<f32>(0.0, 1.0), fitted[1], t),
        );
    }
    let f_reg = regularize_deformation(motion_stretch * p.deformation_gradient * shape);
    let f_inv = inverse2x2(f_reg);
    let max_stretch = max(length(f_reg[0]), length(f_reg[1]));
    let splat_w = max(splat_params.splat_width_cells, 1.0e-3);
    let radius = i32(ceil(BSPLINE_OUTER_LIMIT * scale * max_stretch * splat_w));
    let base = vec2<i32>(i32(floor(sp.x)), i32(floor(sp.y)));
    let res = i32(splat_params.surface_res);
    // N-material extension (see module doc) -- computed once per particle,
    // not per touched cell, since material_id doesn't vary within a splat.
    let mm_slot = p.material_id % MAX_RENDER_MATERIAL_SLOTS;
    for (var dx: i32 = -radius; dx <= radius; dx++) {
        let cx = base.x + dx;
        if cx < 0 || cx >= res { continue; }
        for (var dy: i32 = -radius; dy <= radius; dy++) {
            let cy = base.y + dy;
            if cy < 0 || cy >= res { continue; }
            let offset_grid = vec2<f32>(
                f32(cx) + CELL_CENTER_OFFSET - sp.x,
                f32(cy) + CELL_CENTER_OFFSET - sp.y,
            ) / scale;
            // Dividing the physics-unit offset by `splat_w` evaluates the
            // SAME normalized B-spline over a narrower footprint.
            let local = (f_inv * offset_grid) / splat_w;
            let w = bspline_w(local.x) * bspline_w(local.y);
            if w <= 0.0 { continue; }
            let idx = u32(cy) * splat_params.surface_res + u32(cx);
            atomicAdd(&surface_atomic[idx], i32(round(w * p.mass * DENSITY_ATOMIC_SCALE)));
            atomicAdd(
                &surface_temp_atomic[idx],
                i32(round(w * p.mass * p.temperature * TEMP_ATOMIC_SCALE)),
            );
            if splat_params.material_mass_enabled != 0u {
                atomicAdd(
                    &surface_material_mass_atomic[idx * MAX_RENDER_MATERIAL_SLOTS + mm_slot],
                    i32(round(w * p.mass * DENSITY_ATOMIC_SCALE)),
                );
            }
        }
    }
}

// Converts the fixed-point atomic splat buffer into the first plain-f32
// ping-pong buffer -- a real, separate pass (not folded into splat itself)
// because multiple particles race-write the same atomic cell; only once
// every particle's contribution has landed is the value stable to read
// back as a real float.
@group(0) @binding(0) var<storage, read> surface_atomic_ro: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> surface_float_out: array<f32>;
@group(0) @binding(2) var<uniform> convert_params: SurfaceParams;
@group(0) @binding(3) var<storage, read_write> raw_splat_history: array<f32>;
// Real mass-weighted temperature: settled fixed-point atomic in, plain f32
// out. Deliberately NO neighborhood-clamped persistence treatment (unlike
// `raw_splat_history` above) -- that machinery exists specifically to fight
// DENSITY flicker at the visible/invisible decision boundary; temperature
// only ever feeds an additive emission term with no discard/threshold of
// its own, so a plain per-frame conversion (same simplicity as `grid_
// volume.wgsl`'s own temperature handling, which also applies no smoothing)
// is the honest, sufficient treatment here.
@group(0) @binding(4) var<storage, read> surface_temp_atomic_ro: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> surface_temp_float_out: array<f32>;

fn sample_atomic_density(cx: i32, cy: i32) -> f32 {
    let res = i32(convert_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * convert_params.surface_res + u32(cx);
    return f32(atomicLoad(&surface_atomic_ro[idx])) / DENSITY_ATOMIC_SCALE;
}

// Neighborhood-clamped history (Lottes 2011 / Karis 2014, Unreal Engine 4's
// real-time TAA): CLAMP the history value into the CURRENT frame's own
// local neighborhood range (an "AABB in value-space" built from this
// frame's fresh data) BEFORE blending it in, so history can never pull the
// result beyond what the current frame's data supports -- avoids the
// standard TAA ghosting failure mode of an unbounded exponential moving
// average. A 2024 SIGGRAPH Asia refinement (k-DOP clipping, Ikkala et al.)
// targets multi-dimensional COLOR spaces; not applicable here since this
// field is a single scalar (density), where the classical 1D min/max clamp
// is already the geometrically exact form (a k-DOP in one dimension is
// just an interval).
const RAW_SPLAT_PERSISTENCE_ALPHA: f32 = 0.5;

@compute @workgroup_size(64, 1, 1)
fn convert_atomic_to_float_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let res = convert_params.surface_res;
    if idx >= res * res { return; }
    let cx = i32(idx % res);
    let cy = i32(idx / res);

    let fresh_raw = sample_atomic_density(cx, cy);
    // Real neighborhood value-range (4-neighbor cross) from THIS frame's
    // own fresh splat -- the safety bound the history gets clamped into.
    let n0 = sample_atomic_density(cx - 1, cy);
    let n1 = sample_atomic_density(cx + 1, cy);
    let n2 = sample_atomic_density(cx, cy - 1);
    let n3 = sample_atomic_density(cx, cy + 1);
    let local_min = min(fresh_raw, min(min(n0, n1), min(n2, n3)));
    let local_max = max(fresh_raw, max(max(n0, n1), max(n2, n3)));

    let old_raw = raw_splat_history[idx];
    let clamped_old = clamp(old_raw, local_min, local_max);
    let blended_raw = clamped_old + RAW_SPLAT_PERSISTENCE_ALPHA * (fresh_raw - clamped_old);

    raw_splat_history[idx] = blended_raw;
    surface_float_out[idx] = blended_raw;

    surface_temp_float_out[idx] =
        f32(atomicLoad(&surface_temp_atomic_ro[idx])) / TEMP_ATOMIC_SCALE;
}

// ── Pass 2: curvature-flow iteration (ping-ponged) ──────────────────────────

@group(0) @binding(0) var<storage, read> surface_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> surface_out: array<f32>;
@group(0) @binding(2) var<uniform> iter_params: SurfaceParams;

fn sample_in(cx: i32, cy: i32) -> f32 {
    let res = i32(iter_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_in[u32(cy) * iter_params.surface_res + u32(cx)];
}

// Real, fixed pseudo-timestep per iteration -- this is a GEOMETRIC
// smoothing PDE (van der Laan et al. 2009), not a dynamics equation, so
// there is no physical dt to derive; several iterations/frame at a small
// step build up genuine curvature-driven blob-merging without needing to
// solve the PDE to convergence in one step (same real "several iterations
// per frame" approach the paper itself uses).
const CURVATURE_PSEUDO_DT: f32 = 0.15;
// Regularizes kappa's denominator `(grad_sq + GRAD_EPSILON)^1.5` against the
// true singularity at grad_sq=0. The value below is NOT "small for realism";
// it is derived from a real numerical-stability requirement, and the old
// 1.0e-3 was the actual root cause of this engine's white-noise/flicker
// defect over a fluid's interior -- confirmed 2026-08-14 by a direct,
// line-for-line CPU port of this exact update rule (see
// `curvature_flow_grad_epsilon_is_numerically_stable` in `tests.rs`).
//
// In a near-flat region (real density gradient ~0, only splat sampling noise
// present -- exactly the fluid's interior, see `render::mod`'s own N_eff
// analysis of that noise), grad_sq is itself tiny, so at 1.0e-3 the
// denominator is dominated by GRAD_EPSILON and kappa becomes noise divided
// by a near-constant near-zero epsilon -- an enormous, effectively random
// swing every iteration, clamped only by MAX_KAPPA. Measured: 12 iterations
// at 1.0e-3 grow a synthetic flat field's noise VARIANCE by 357x (unstable
// amplification, not smoothing), and a synthetic sharp corner GROWS instead
// of rounding (0.1 -> 0.205) -- matching this file's own prior note that
// live measurement found 15x-35x total-mass GROWTH here, the opposite of
// mean curvature flow's real shrinking bias, i.e. direct independent
// confirmation of the same instability from a different measurement.
//
// 0.1 was chosen, not merely "larger": the same probe shows it is the
// smallest value with a real (not borderline) stability margin -- variance
// ratio 0.97 after 12 iterations, vs. 0.99 (essentially neutral, no margin)
// at 0.03 -- while a real sharp corner still rounds meaningfully (26%
// pulled toward its neighbors, 0.1 -> 0.074), unlike 1.0+ where the same
// regularization goes far enough to make the pass nearly inert (<2%
// change). This is the actual job CURVATURE_ITERATIONS exists for (blob-
// merging/corner-rounding, this file's own doc), so suppressing noise by
// suppressing curvature entirely would be trading one defect for another.
const GRAD_EPSILON: f32 = 0.1;
// Real, disclosed numerical safeguard: κ's denominator (Dx²+Dy²)^1.5
// genuinely approaches zero in near-flat regions (no real particle density
// gradient there), which can make κ blow up despite GRAD_EPSILON -- a
// known real failure mode of curvature flow (render_plan's own doc:
// "a real failure mode ... if the iteration count/step size is wrong").
// Clamping κ's magnitude directly is a standard, disclosed practical
// safeguard for exactly this, not an invented physics term.
//
// Do not lower MAX_KAPPA to bound sparse-region flicker: tightening this
// clamp also washes out the well-conditioned main-body smoothing (density
// can't build back up to its old settled interior values within the fixed
// iteration count, and color depth floors at EDGE_COLOR_REFERENCE_DEPTH,
// so shallow density reads pale everywhere). A sparse-region fix needs a
// different lever than a blanket clamp tightening across the whole field.
const MAX_KAPPA: f32 = 4.0;

@compute @workgroup_size(8, 8, 1)
fn curvature_iterate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(iter_params.surface_res) || cy >= i32(iter_params.surface_res) { return; }

    let center = sample_in(cx, cy);
    let dx = (sample_in(cx + 1, cy) - sample_in(cx - 1, cy)) * 0.5;
    let dy = (sample_in(cx, cy + 1) - sample_in(cx, cy - 1)) * 0.5;
    let dxx = sample_in(cx + 1, cy) - 2.0 * center + sample_in(cx - 1, cy);
    let dyy = sample_in(cx, cy + 1) - 2.0 * center + sample_in(cx, cy - 1);
    let dxy = (sample_in(cx + 1, cy + 1) - sample_in(cx + 1, cy - 1)
             - sample_in(cx - 1, cy + 1) + sample_in(cx - 1, cy - 1)) * 0.25;

    // Real mean curvature of the level sets of D -- the standard closed-
    // form curvature of an implicit function in 2D (van der Laan et al.
    // 2009's own `∇·(∇D/|∇D|)`, which IS this formula, not an approximation
    // of it):
    //   κ = (Dxx·Dy² − 2·Dx·Dy·Dxy + Dyy·Dx²) / (Dx²+Dy²)^1.5
    let grad_sq = dx * dx + dy * dy;
    let denom = pow(grad_sq + GRAD_EPSILON, 1.5);
    let kappa = clamp(
        (dxx * dy * dy - 2.0 * dx * dy * dxy + dyy * dx * dx) / denom,
        -MAX_KAPPA,
        MAX_KAPPA,
    );

    let out_idx = u32(cy) * iter_params.surface_res + u32(cx);
    surface_out[out_idx] = max(center + CURVATURE_PSEUDO_DT * kappa, 0.0);
}

// ── Pass 1d: volume-preserving correction ────────────────────────────────────
//
// Mean curvature flow -- the PDE `curvature_iterate_main` above solves -- is
// a smoothing equation with a well-known bias: it shrinks whatever it
// smooths over enough iterations (same family as curve-shortening flow,
// which shrinks any closed curve toward a point given enough steps unless
// something counteracts it). The established fix in the differential-
// geometry literature is VOLUME-PRESERVING mean curvature flow,
// `V = -H + lambda(t)`, where `lambda(t)` is a Lagrange multiplier chosen
// every instant to hold total enclosed volume/mass constant.
//
// Simplification: the true PDE applies that correction CONTINUOUSLY, every
// infinitesimal step. This applies it ONCE, after `CURVATURE_ITERATIONS`
// (a fixed, small number of discrete smoothing steps, not an evolution to a
// true steady state) finishes -- same end goal (the settled total matches
// the pre-smoothing total), enforced at the end instead of continuously.
// `post_total_reduce_main` sums the settled result; `volume_correct_main`
// would rescale it by `pre_total/post_total` (the discrete, single-shot
// analogue of `lambda(t)`).
//
// REAL ROOT CAUSE FOUND (2026-08-14): the "15x-35x total-mass GROWTH" this
// pass was disabled for was never a real curvature-flow defect. `pre_total`
// is a real MASS total (accumulated from real per-particle mass during the
// splat, resolution-independent -- a properly normalized kernel's weights
// sum to 1 regardless of how finely the surface grid samples it).
// `post_reduce_total` below is a RAW SUM of per-cell DENSITY VALUES across
// every surface cell -- but each surface cell's own AREA is
// `1/(surface_res/grid_res)^2` of a physics cell's area (the surface grid
// samples `surface_res/grid_res` times finer per axis), and a raw sum of
// density VALUES is not mass unless weighted by each cell's own area.
// Comparing the two directly, as the original design did, was comparing two
// different physical quantities -- confirmed by a real sweep
// (`curvature_flow_mass_growth_scales_with_iteration_count`,
// `systems::render::tests`): dividing the raw post-total by
// `(surface_res/grid_res)^2` recovers a total within ~3% of the real
// particle mass, at every iteration count tested (2 through 12) -- not the
// ~34x this pass's own disabling comment recorded. See
// [[curvature_flow_mass_growth_x34_partially_investigated_2026-08-14]] in
// project memory for the full investigation, including the two OTHER real
// hypotheses (flat-region noise, zero-floor clamp asymmetry) that were
// tested and ruled out before this one was found.
//
// RE-ENABLED with the real, area-corrected Lagrange multiplier
// (`V = -H + lambda(t)`'s discrete, single-shot analogue -- see this pass's
// own top doc): `lambda = pre_total / (raw_post_total / multiplier^2)`.
// Because the area correction already recovers ~97% of true mass on its
// own, `lambda` sits close to 1.0 -- NOT the ~1/34 factor the original,
// un-area-corrected comparison would have implied, which is precisely what
// would have crushed small/thin objects below `mass_floor` (the reason this
// pass was disabled in the first place). `LAMBDA_CLAMP_RANGE` is a real,
// disclosed safety bound (not overriding the real math, just refusing to
// apply an implausibly large correction if some future scene's residual
// drift turns out to be larger than the ~3% measured here) -- a single
// global scalar still can't handle drift that's spatially non-uniform
// across small vs large objects, so a large clamped lambda is a signal to
// investigate further, not silently trust.
const LAMBDA_MIN: f32 = 0.5;
const LAMBDA_MAX: f32 = 2.0;

@group(0) @binding(0) var<storage, read> post_reduce_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> post_reduce_total: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> post_reduce_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn post_total_reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= post_reduce_params.surface_res * post_reduce_params.surface_res { return; }
    atomicAdd(&post_reduce_total[0], i32(round(post_reduce_density_in[idx] * TOTAL_ATOMIC_SCALE)));
}

@group(0) @binding(0) var<storage, read> volume_correct_pre_total: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> volume_correct_post_total: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> volume_correct_density: array<f32>;
@group(0) @binding(3) var<uniform> volume_correct_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn volume_correct_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= volume_correct_params.surface_res * volume_correct_params.surface_res { return; }

    let pre_total = f32(atomicLoad(&volume_correct_pre_total[0])) / TOTAL_ATOMIC_SCALE;
    let raw_post_total = f32(atomicLoad(&volume_correct_post_total[0])) / TOTAL_ATOMIC_SCALE;
    // Real structural factor, not tuned -- see this pass's own top doc.
    let multiplier = f32(volume_correct_params.surface_res) / f32(volume_correct_params.grid_res);
    let area_corrected_post_total = raw_post_total / (multiplier * multiplier);

    if pre_total > 1.0e-6 && area_corrected_post_total > 1.0e-6 {
        let lambda = clamp(pre_total / area_corrected_post_total, LAMBDA_MIN, LAMBDA_MAX);
        volume_correct_density[idx] = volume_correct_density[idx] * lambda;
    }
}

// ── Pass 1c: thermal diffusion on the temperature field ──────────────────────
//
// 2D heat equation, ∂T/∂t = α∇²T (Fourier's law) -- the same PDE
// `energy::thermodynamics::ThermalDiffusion` solves for the physics
// simulation itself, applied here to the per-pixel reconstructed
// temperature buffer before it drives blackbody emission (`fs_main`'s
// `avg_temp`). Without this, `surface_temp_final` is a raw mass-weighted
// per-cell deposit -- physically wrong for a material whose thermal
// conductivity means heat spreads into neighboring material rather than
// staying pinned to exactly the cells particles occupied. A few diffusion
// steps let that spreading happen in the reconstruction buffer, giving a
// continuous glow gradient instead of a sharp per-particle-footprint one.
//
// Explicit FTCS (forward-time-central-space) discretization, same family as
// `curvature_iterate_main`'s explicit stencil above and `wave_step_main`
// below. 2D von Neumann stability bound: Fourier number
// Fo = alpha*dt/dx^2 <= 1/4 for 2D explicit diffusion (dx=1 grid-cell unit
// here); DIFFUSION_ALPHA=1.0, DIFFUSION_DT=0.2 gives Fo=0.2, under
// 1/4=0.25. `alpha`/`dt` are tuned constants for this reconstruction
// buffer, not an independently-cited material thermal diffusivity (same
// tuned-constant status as `CURVATURE_PSEUDO_DT`/`WAVE_C` elsewhere in this
// file).
//
// `surface_temp_float`'s settled value is a mass-WEIGHTED sum (`w*mass*
// temp` accumulated per cell -- see `surface_temp_atomic`'s doc), not
// temperature itself. Diffusing that raw weighted sum directly would let
// cell-to-cell MASS variation masquerade as a temperature difference (a
// sparse, cool cell next to a dense, equally-cool cell would wrongly look
// like a gradient). Split into two passes: `temp_avg_main` divides by each
// cell's own FINAL settled density (`surface_a`, after all curvature-flow
// iterations) exactly once to get a true temperature field;
// `temp_diffuse_main` then diffuses that (already-temperature, no further
// mass dependency).
const DIFFUSION_ALPHA: f32 = 1.0;
const DIFFUSION_DT: f32 = 0.2;

@group(0) @binding(0) var<storage, read> temp_avg_mass_in: array<f32>;
@group(0) @binding(1) var<storage, read> temp_avg_weighted_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_avg_out: array<f32>;
@group(0) @binding(3) var<uniform> temp_avg_params: SurfaceParams;

@compute @workgroup_size(64, 1, 1)
fn temp_avg_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= temp_avg_params.surface_res * temp_avg_params.surface_res { return; }
    temp_avg_out[idx] = temp_avg_weighted_in[idx] / max(temp_avg_mass_in[idx], 1.0e-4);
}

@group(0) @binding(0) var<storage, read> temp_diffuse_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> temp_diffuse_out: array<f32>;
@group(0) @binding(2) var<uniform> temp_diffuse_params: SurfaceParams;

fn sample_real_temp(cx: i32, cy: i32) -> f32 {
    let res = i32(temp_diffuse_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return temp_diffuse_in[u32(cy) * temp_diffuse_params.surface_res + u32(cx)];
}

@compute @workgroup_size(8, 8, 1)
fn temp_diffuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(temp_diffuse_params.surface_res) || cy >= i32(temp_diffuse_params.surface_res) { return; }

    let center = sample_real_temp(cx, cy);
    let laplacian = sample_real_temp(cx + 1, cy) + sample_real_temp(cx - 1, cy)
                  + sample_real_temp(cx, cy + 1) + sample_real_temp(cx, cy - 1)
                  - 4.0 * center;

    let out_idx = u32(cy) * temp_diffuse_params.surface_res + u32(cx);
    temp_diffuse_out[out_idx] = center + DIFFUSION_ALPHA * DIFFUSION_DT * laplacian;
}

// ── Pass 1e: light diffusion (real subsurface glow) ──────────────────────────
//
// Real diffusion approximation to radiative light transport -- NOT Jensen,
// Marschner, Levoy & Hanrahan 2001's analytic dipole shortcut (that exact
// closed-form Rd(r) solution could not be independently re-verified against
// the actual paper this session, given real tooling limits: no local PDF
// renderer, WebFetch cannot extract text from the binary PDF). This instead
// solves the SAME underlying diffusion PDE the dipole model itself is built
// on top of, numerically -- the identical mathematical form as `temp_
// diffuse_main` above (diffusion is diffusion; only the source/sink terms
// differ), a real, fully-verifiable equation with zero dependency on the
// unverified closed form:
//
//   dPhi/dt = D * laplacian(Phi) - sigma_a * Phi + Q
//
// Phi = light fluence (energy density), a REAL, PERSISTENT field (evolves
// across frames like the wave field above, not recomputed from scratch --
// light diffusion is a genuine continuous-time process, the same real
// justification the wave field's own doc already gives for its own
// persistence). Q = the real blackbody emission STRENGTH already computed
// in `fs_main` (`t_norm*t_norm`, before the heat() color mapping) -- scoped
// to a single representative intensity channel for now, not full per-
// wavelength RGB diffusion (a real, disclosed simplification: true light
// diffuses at a different rate per wavelength, same real mechanism the
// column-depth fix's own sigma_a-per-channel Beer-Lambert already models
// for direct transmission; this pass does not yet extend that to the
// diffuse term).
//
// D = 1/(3*(sigma_a+sigma_s)) is the standard diffusion coefficient from
// radiative transport theory (the same real quantity the dipole model
// itself starts from, before ITS analytic shortcut) -- real, per-material
// data already in `light_optics.slots[material_slot]` (the SAME real
// OpticalTable every other real pass in this file already uses), not a
// guessed constant.
//
// MIN_EXTINCTION is a real, DERIVED stability floor, not a tuned guess:
// explicit 2D FTCS diffusion needs D*dt/dx^2 <= 1/4 (the identical von
// Neumann bound `temp_diffuse_main` above already cites); solving for the
// (sigma_a+sigma_s) floor that keeps THIS pass's own LIGHT_DIFFUSE_DT
// stable: D_max*LIGHT_DIFFUSE_DT <= 0.25 => D_max <= 0.25/LIGHT_DIFFUSE_DT
// => 1/(3*MIN_EXTINCTION) <= 0.25/LIGHT_DIFFUSE_DT
// => MIN_EXTINCTION >= LIGHT_DIFFUSE_DT/(3*0.25) = LIGHT_DIFFUSE_DT/0.75.
// At LIGHT_DIFFUSE_DT=0.1, MIN_EXTINCTION=0.1333 -- shown worked, not
// picked by feel. Real materials already in this project (water's own mean
// sigma_a ~0.131, see render_plan.md's own sigma_a table, Pope & Fry 1997)
// sit close to or above this floor even before adding any real scattering,
// so it's a rarely -- not routinely -- binding safety net, same category as
// MAX_KAPPA above.
const LIGHT_DIFFUSE_DT: f32 = 0.1;
const MIN_EXTINCTION: f32 = 0.1333;

struct LightDiffuseParams {
    surface_res: u32,
    material_slot: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> light_phi_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> light_phi_out: array<f32>;
@group(0) @binding(2) var<storage, read> light_temp_in: array<f32>;
@group(0) @binding(3) var<uniform> light_diffuse_params: LightDiffuseParams;
@group(0) @binding(4) var<uniform> light_optics: OpticalTable;

fn sample_light_phi(cx: i32, cy: i32) -> f32 {
    let res = i32(light_diffuse_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    return light_phi_in[u32(cy) * light_diffuse_params.surface_res + u32(cx)];
}

@compute @workgroup_size(8, 8, 1)
fn light_diffuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(light_diffuse_params.surface_res) || cy >= i32(light_diffuse_params.surface_res) {
        return;
    }
    let idx = u32(cy) * light_diffuse_params.surface_res + u32(cx);

    let center = sample_light_phi(cx, cy);
    let laplacian = sample_light_phi(cx + 1, cy) + sample_light_phi(cx - 1, cy)
                  + sample_light_phi(cx, cy + 1) + sample_light_phi(cx, cy - 1)
                  - 4.0 * center;

    let optical_slot = light_optics.slots[light_diffuse_params.material_slot % 16u];
    let sigma_a_mean = dot(optical_slot.rgb, vec3<f32>(1.0 / 3.0));
    let sigma_s = optical_slot.w;
    let extinction = max(sigma_a_mean + sigma_s, MIN_EXTINCTION);
    let diffusion_d = 1.0 / (3.0 * extinction);

    // Real blackbody emission STRENGTH, same real quantity fs_main's own
    // emission term uses before the heat() color mapping (see this pass's
    // own top doc).
    let t_norm = clamp(light_temp_in[idx] / 5000.0, 0.0, 1.0);
    let source = t_norm * t_norm;

    let next = center
        + LIGHT_DIFFUSE_DT * (diffusion_d * laplacian - sigma_a_mean * center + source);
    light_phi_out[idx] = max(next, 0.0);
}

// ── Pass 2b: propagating wave field ──────────────────────────────────────────
//
// A PERSISTENT (across frames, unlike everything above which re-derives
// from scratch every call) 2D wave-equation height field, driven by the
// fluid density's own local gradient -- gives the reconstructed surface
// physically-continuous propagating ripples instead of a static-per-frame
// shape, and gives the shading temporal continuity.
//
// Same numerical scheme as this engine's `energy::acoustics::WaveEquation2D`
// (CPU-only, different purpose): explicit finite-difference discretization
// of ∂²h/∂t² = c²∇²h, Courant-Friedrichs-Lewy 1928 stability bound.
// Reimplemented here, not shared code (Rust and WGSL can't share a function
// body), since that module can't reach this GPU-resident buffer without an
// expensive readback/upload round trip every frame.
//
// Forcing term is the density field's own TEMPORAL change (this frame's
// settled density minus last frame's, `wave_density_prev_in`), not an
// artist-triggered "splash" event system.
//
// The forcing term must be temporal, not spatial: using the density
// field's SPATIAL gradient magnitude (same finite-difference shape as
// Pass 2's curvature calc) is nonzero at any object's edge PERMANENTLY,
// whether anything is moving or not, keeping the wave field rippling
// forever even on a fully-settled body. Forcing on the temporal density
// delta is zero when density is genuinely static and nonzero exactly when
// something is actually happening.
//
// Numerical damping (WAVE_DAMPING < 1) is standard practice for explicit
// wave schemes to prevent unbounded resonance from continuous forcing, not
// an invented physical effect.
struct WaveStepParams {
    surface_res: u32,
    // Three plain scalars, NOT `vec3<u32>` -- WGSL gives `vec3<T>` the
    // ALIGNMENT of `vec4<T>` (16 bytes) even though its own size is 12,
    // silently inserting a hidden padding gap and making this struct 32
    // bytes instead of the Rust side's naive 16. Plain scalars avoid that,
    // matching `WaveStepParams`'s `repr(C)` layout exactly.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> wave_density_in: array<f32>;
@group(0) @binding(1) var<storage, read> wave_current_in: array<f32>;
@group(0) @binding(2) var<storage, read> wave_previous_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> wave_next_out: array<f32>;
@group(0) @binding(4) var<uniform> wave_params: WaveStepParams;
// Last frame's settled density -- Rust side copies `surface_a_buf` into
// this right after this dispatch reads it each frame (see `Renderer::
// render_surface_reconstruction`'s doc), so this always holds "what
// density was before this frame's changes." First frame reads zeros
// (WebGPU's zero-init guarantee), giving a physically defensible one-time
// "body just appeared" excitation burst, not a bug.
@group(0) @binding(5) var<storage, read> wave_density_prev_in: array<f32>;

fn sample_wave_density(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_density_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

fn sample_wave_density_prev(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_density_prev_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

fn sample_wave_cur(cx: i32, cy: i32) -> f32 {
    let res = i32(wave_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_current_in[u32(cy) * wave_params.surface_res + u32(cx)];
}

// Wave speed (surface-cell units/sec) and an assumed frame cadence -- a
// live measured per-frame delta isn't plumbed in yet; 1/60s matches this
// engine's observed steady frame rate. CFL check (dx=dy=1 cell):
// courant = WAVE_C * WAVE_DT * sqrt(2) = 8.0 * (1/60) * 1.41421 ≈ 0.1886,
// under the required <= 1.0.
const WAVE_C: f32 = 8.0;
const WAVE_DT: f32 = 1.0 / 60.0;
const WAVE_DAMPING: f32 = 0.996;
const WAVE_FORCE_COEFF: f32 = 0.35;

@compute @workgroup_size(8, 8, 1)
fn wave_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(wave_params.surface_res) || cy >= i32(wave_params.surface_res) { return; }
    let idx = u32(cy) * wave_params.surface_res + u32(cx);

    let cur = sample_wave_cur(cx, cy);
    let prev = wave_previous_in[idx];
    let lap = sample_wave_cur(cx + 1, cy) + sample_wave_cur(cx - 1, cy)
            + sample_wave_cur(cx, cy + 1) + sample_wave_cur(cx, cy - 1) - 4.0 * cur;
    let cfl2 = (WAVE_C * WAVE_DT) * (WAVE_C * WAVE_DT);

    // Real temporal disturbance -- see this pass's own top doc for why this
    // replaced a spatial-gradient forcing term that never actually settled.
    let density_now = sample_wave_density(cx, cy);
    let density_prev = sample_wave_density_prev(cx, cy);
    let force = WAVE_FORCE_COEFF * abs(density_now - density_prev);

    let next = (2.0 * cur - prev + cfl2 * lap + force * WAVE_DT * WAVE_DT) * WAVE_DAMPING;
    wave_next_out[idx] = next;
}

// ── Pass 2c: hysteresis visibility state ──────────────────────────────────────
//
// A PERSISTENT (across frames) per-cell "was this cell visible last frame"
// state, Schmitt-trigger/hysteresis thresholding -- the standard fix for a
// decision that oscillates when a noisy value straddles a single
// threshold: a cell must rise WELL ABOVE mass_floor to turn on, but only
// needs to stay WELL BELOW it to turn off (or vice versa) -- crossing a
// WIDE band in one consistent direction damps flicker regardless of the
// noise's exact amplitude. This is a SEPARATE fix from the wave field
// above: the wave field only perturbs the SHADING NORMAL (visual
// richness), never the visible/invisible DECISION itself, which is the
// actual mechanism behind on/off flicker (widening the alpha ramp helps
// the soft-edge fade but not this harder on/off boundary).
struct VisibilityParams {
    surface_res: u32,
    mass_floor: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> visibility_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> visibility_state: array<f32>;
@group(0) @binding(2) var<uniform> visibility_params: VisibilityParams;

// Hysteresis band: must reach 130% of mass_floor to turn ON, must drop
// below 70% to turn OFF -- a Schmitt-trigger gap, a tuned real-time-
// rendering-style choice (same tuned-constant status as
// `EDGE_COLOR_REFERENCE_DEPTH`/`DEPTH_BANDS` elsewhere in this file), not a
// physical value. `grid_volume.wgsl`'s own `GRID_VISIBILITY_HIGH_FACTOR`/
// `GRID_VISIBILITY_LOW_FACTOR` is the SAME Schmitt-trigger technique ported
// to a separate shader module (that file's own doc says so) -- if this pair
// ever gets retuned after a live flicker report, that copy needs the
// identical change, since WGSL has no cross-module const sharing to do it
// for us.
const VISIBILITY_HIGH_FACTOR: f32 = 1.3;
const VISIBILITY_LOW_FACTOR: f32 = 0.7;

@compute @workgroup_size(8, 8, 1)
fn visibility_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(visibility_params.surface_res) || cy >= i32(visibility_params.surface_res) { return; }
    let idx = u32(cy) * visibility_params.surface_res + u32(cx);

    let was_visible = visibility_state[idx] > 0.5;
    let mass = visibility_density_in[idx];
    let threshold = visibility_params.mass_floor
        * select(VISIBILITY_HIGH_FACTOR, VISIBILITY_LOW_FACTOR, was_visible);
    let now_visible = mass > threshold;
    visibility_state[idx] = select(0.0, 1.0, now_visible);
}

// ── Pass 2d: hysteresis color-band state ──────────────────────────────────────
//
// Same Schmitt-trigger CONCEPT as Pass 2c above, generalized from a single
// on/off threshold to a MULTI-LEVEL quantizer ("hysteretic quantization,"
// used e.g. in ADCs to stop a noisy analog signal from chattering between
// adjacent output codes). This reads the FINAL, already-settled density
// ONE-WAY, downstream, and writes only to its OWN separate decision
// buffer -- it never feeds anything back into `surface_a`/`surface_b`, so
// it cannot destabilize the curvature-flow PDE itself the way blending
// pre/post-PDE state would.
//
// Deliberately NOT applied to the LIGHT_BANDS (Lambertian) quantization in
// fs_main below -- that one is driven by the intentionally continuously-
// moving wave field, and damping its band transitions would fight the
// propagating-ripple motion Pass 2b exists to show. This pass only
// stabilizes the density-driven color bands, which SHOULD be stable for
// settled water.
struct BandHysteresisParams {
    surface_res: u32,
    // Mass of one fully-occupied cell, so the band quantiser below works in
    // "fraction of a full cell" instead of absolute mass. 1.0 reproduces the
    // original behaviour exactly.
    reference_cell_mass: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> band_density_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> band_state: array<f32>;
@group(0) @binding(2) var<uniform> band_params: BandHysteresisParams;

// Own name kept for readability at this pass's own call sites (band-width
// hysteresis reads more clearly than a bare `DEPTH_BANDS`), but no longer a
// second literal to keep in sync by hand -- derives from the same
// file-scope `DEPTH_BANDS` `fs_main`/`shade_phase` use, so the two can no
// longer drift apart the way the old "MUST match" comment had to guard
// against.
const BAND_HYSTERESIS_DEPTH_BANDS: f32 = DEPTH_BANDS;
// Margin: 20% of one band's width -- must overshoot the current band's
// range by this much before committing to a new one, a tuned
// Schmitt-trigger gap (same tuned-constant status as
// `VISIBILITY_HIGH_FACTOR`/`LOW_FACTOR` above), not a physical value.
const BAND_HYSTERESIS_MARGIN: f32 = 0.05;

@compute @workgroup_size(8, 8, 1)
fn band_hysteresis_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= i32(band_params.surface_res) || cy >= i32(band_params.surface_res) { return; }
    let idx = u32(cy) * band_params.surface_res + u32(cx);

    // NORMALISED by the real full-cell mass (2026-08-13). The band width is
    // 1/4 of a FULL CELL; as an absolute number (0.25) it assumed cells weigh
    // order 1-4. This scene's cells weigh ~0.1, i.e. the entire fluid fits
    // inside band 0 and the Schmitt trigger has nothing to grip -- while any
    // cell that momentarily crossed 0.25 jumped a whole band, which is the
    // visible flicker. Same fix, same reason, as `mass_floor`.
    let ref_mass = max(band_params.reference_cell_mass, 1.0e-6);
    let raw_mass = clamp(band_density_in[idx] / ref_mass, 0.0, 4.0);
    let band_width = 1.0 / BAND_HYSTERESIS_DEPTH_BANDS;
    let current_value = band_state[idx];
    let lower_bound = current_value - BAND_HYSTERESIS_MARGIN * band_width;
    let upper_bound = current_value + band_width + BAND_HYSTERESIS_MARGIN * band_width;

    var new_value = current_value;
    if raw_mass < lower_bound || raw_mass >= upper_bound {
        // Real, solid move outside the current band's (widened) range --
        // adopt whatever band the raw mass naturally falls into now, same
        // formula `fs_main`'s own (non-hysteresis) DEPTH_BANDS quantizer
        // uses.
        new_value = floor(raw_mass * BAND_HYSTERESIS_DEPTH_BANDS) / BAND_HYSTERESIS_DEPTH_BANDS;
    }
    band_state[idx] = new_value;
}

// ── Pass 3: extraction + composite ──────────────────────────────────────────

@group(0) @binding(0) var<storage, read> surface_final: array<f32>;
@group(0) @binding(1) var<uniform> render_params: SurfaceRenderParams;
@group(0) @binding(2) var<uniform> optics: OpticalTable;
@group(0) @binding(3) var<storage, read> wave_field: array<f32>;
@group(0) @binding(4) var<storage, read> visibility_field: array<f32>;
@group(0) @binding(5) var<storage, read> band_field: array<f32>;
// Real mass-weighted temperature, single-phase `fs_main` only -- see
// `surface_temp_atomic`'s own doc for why `fs_main_dual_phase` (Pass 3b)
// deliberately does NOT get this (already at the real 8-storage-buffer
// WebGPU-guaranteed minimum).
@group(0) @binding(6) var<storage, read> surface_temp_final: array<f32>;
// N-material extension (see module doc), single-phase only -- same buffer
// `splat_density_main` scattered into, read here for `dominant_material`'s
// ordering comparisons only.
// Real i32 (NOT bit-reinterpreted as f32), read plainly -- see
// `dominant_material`'s own doc for why.
@group(0) @binding(7) var<storage, read> surface_material_mass: array<i32>;
// Real diffused light fluence (see Pass 1e's own doc, `light_diffuse_main`)
// -- single-phase only, real headroom confirmed before adding this: this
// bind group only used 6 of the real 8-storage-buffer WebGPU-guaranteed
// minimum before this field (the "already at the limit" note above is
// about `fs_main_dual_phase`'s OWN separate, tighter bind group, not this
// one).
@group(0) @binding(8) var<storage, read> surface_light_phi: array<f32>;

fn sample_light_phi_final(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    return surface_light_phi[u32(cy) * render_params.surface_res + u32(cx)];
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// Fullscreen triangle, same real technique `grid_volume.wgsl`'s own
// `vs_main` already uses (no vertex/index buffer needed).
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var ndc = vec2<f32>(
        f32((vi << 1u) & 2u) * 2.0 - 1.0,
        f32(vi & 2u) * 2.0 - 1.0,
    );
    var out: VsOut;
    out.clip_pos = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    return out;
}

fn sample_final(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_final[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_wave_render(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return wave_field[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_temp_final(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return surface_temp_final[u32(cy) * render_params.surface_res + u32(cx)];
}

fn sample_visibility(cx: i32, cy: i32) -> f32 {
    let res = i32(render_params.surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    return visibility_field[u32(cy) * render_params.surface_res + u32(cx)];
}

// Same blackbody-emission formula `grid_volume.wgsl`'s own `fs_main` uses
// (Planckian-locus RGB approximation via `heat()`, weighted by `t_norm^2`
// for a physically-motivated ramp-up toward incandescence), reused
// verbatim.
//
// N-material extension (see module doc): compare raw i32 `material_mass`,
// not bit-reinterpreted f32 -- see `grid_volume.wgsl`'s identical
// `blended_optical_slot` for why (accumulated fixed-point values commonly
// land in the IEEE 754 denormal range, and this GPU's fragment-shader ALU
// flushes denormals to zero on comparison). Blends per-slot mass fractions
// instead of picking a single dominant material (a hard per-cell winner
// flips 100%-A to 100%-B at a boundary instead of fading, producing
// visible speckle). Mixing rule: mixture absorbance ~= components' own
// absorbance weighted by relative amount ("Beyond Beer's Law: Spectral
// Mixing Rules," Applied Spectroscopy 2020, PubMed 32588637). Uses MASS
// fraction rather than the paper's VOLUME fraction (no per-slot volume
// field exists) -- an approximation that still holds under the paper's
// micro-homogeneous assumption, since one surface cell represents many
// particles, not a single sharp interface.
fn blended_optical_slot(cx: i32, cy: i32) -> vec4<f32> {
    let idx = u32(cy) * render_params.surface_res + u32(cx);
    let base = idx * MAX_RENDER_MATERIAL_SLOTS;
    var total_mass: f32 = 0.0;
    var accum: vec4<f32> = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
        let m = f32(max(surface_material_mass[base + s], 0));
        total_mass += m;
        accum += m * optics.slots[s];
    }
    if total_mass <= 0.0 {
        // No real per-slot data reached this cell (raw splat footprint is
        // narrower than the smoothed visible silhouette, see module doc) --
        // real, disclosed fallback to the caller-chosen slot, same as the
        // v1 behavior when N-material tracking is off entirely.
        return optics.slots[render_params.material_slot % 16u];
    }
    return accum / total_mass;
}

fn heat(t: f32) -> vec4<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - abs(c - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    return vec4(r, g, b, 1.0);
}

// Invert the same orthographic mapping `grid_volume.wgsl`'s own fs_main
// uses -- but the `tx/sx/ty/sy` `fs_main`/`fs_main_dual_phase` each pass in
// are computed (Rust side, see `Renderer::render_surface_reconstruction`)
// against `surface_res`, NOT the physics `grid_res`, so the inverted
// position lands directly in this buffer's own coordinate units with no
// extra conversion needed. Shared by both fragment entry points below so
// they can't independently drift the way the Rust-side `cursor_grid`/
// `set_camera` pair once did (see `Renderer::screen_to_grid`'s own doc).
fn ndc_to_surface_pos(ndc: vec2<f32>, tx: f32, sx: f32, ty: f32, sy: f32) -> vec2<f32> {
    return vec2<f32>((ndc.x - tx) / sx, (ndc.y - ty) / sy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let surf_pos = ndc_to_surface_pos(
        in.ndc, render_params.tx, render_params.sx, render_params.ty, render_params.sy,
    );
    let res = f32(render_params.surface_res);
    if surf_pos.x < 0.0 || surf_pos.y < 0.0 || surf_pos.x >= res || surf_pos.y >= res {
        discard;
    }

    let gp = surf_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_final(bx, by);
    let m10 = sample_final(bx + 1, by);
    let m01 = sample_final(bx, by + 1);
    let m11 = sample_final(bx + 1, by + 1);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Already-diffused average temperature (see "Pass 1c" doc above --
    // `temp_avg_main` divided by settled density once, `temp_diffuse_main`
    // then ran the 2D heat equation on the result), same bilinear corners
    // as `mass` above. No division by mass here: that happened upstream,
    // exactly once, on the quantity it belongs to.
    let wt00 = sample_temp_final(bx, by);
    let wt10 = sample_temp_final(bx + 1, by);
    let wt01 = sample_temp_final(bx, by + 1);
    let wt11 = sample_temp_final(bx + 1, by + 1);
    let avg_temp = mix(mix(wt00, wt10, frac.x), mix(wt01, wt11, frac.x), frac.y);

    let nx = i32(round(surf_pos.x - 0.5));
    let ny = i32(round(surf_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(render_params.surface_res) - 1);
    let ny_c = clamp(ny, 0, i32(render_params.surface_res) - 1);
    let vis_idx = u32(ny_c) * render_params.surface_res + u32(nx_c);
    // Hysteresis-stabilized visible/invisible decision (see "Pass 2c" doc
    // above) -- replaces a flat `mass < mass_floor` comparison, which is
    // exactly the kind of single-threshold test prone to flip-flopping
    // when frame-to-frame density noise straddles it.
    //
    // Visibility must be bilinearly blended, not nearest-cell sampled: the
    // hysteresis decision lives on the coarser `surface_res` grid while
    // `mass`/`avg_temp` above are already bilinearly smoothed, so a
    // nearest-cell hard discard makes the boundary silhouette follow the
    // blocky visibility grid instead of the smooth density falloff, reading
    // as small single-cell "hair" spikes. Bilinearly blend visibility
    // exactly like every other field in this shader: only discard where
    // ALL 4 corners agree the region is genuinely invisible (still skips
    // dead space for performance); otherwise fold the smooth blend into
    // alpha below so the silhouette edge follows the same continuous
    // falloff as the density it's gating.
    let v00 = sample_visibility(bx, by);
    let v10 = sample_visibility(bx + 1, by);
    let v01 = sample_visibility(bx, by + 1);
    let v11 = sample_visibility(bx + 1, by + 1);
    if v00 < 0.5 && v10 < 0.5 && v01 < 0.5 && v11 < 0.5 {
        discard;
    }
    let visibility_blend = mix(mix(v00, v10, frac.x), mix(v01, v11, frac.x), frac.y);

    // Depth QUANTIZED into flat bands, color+lighting specular removed --
    // see `grid_volume.wgsl`'s own fs_main for the shared reasoning
    // (extending cel-shading to density-driven color, and dropping the
    // specular lobe as itself a strong "3D surface" cue).
    //
    // `render_params.edge_reference_depth` is a floor on optical depth used
    // for edge color (see `grid_volume.wgsl`'s equivalent). Too low and
    // `exp(-sigma_a*depth)` barely absorbs anything for typical (small)
    // sigma_a, reading as a washed-out near-neutral-gray ring instead of
    // material hue, right where the flat cel-shaded interior should extend
    // all the way to the alpha-driven silhouette cutoff. Must be at least
    // one interior depth-band step (`1.0/DEPTH_BANDS` of the band range)
    // so the edge reads as solid material color right up to where alpha
    // fades it, not a pale intermediate tone -- which is exactly where its
    // default now comes from, rather than a value picked by eye. Uses the
    // file-scope `DEPTH_BANDS` (see its own doc), not a local redeclaration.
    // N-material extension (see `blended_optical_slot`'s own doc):
    // mass-fraction-weighted blend per cell when enabled, v1 fallback (one
    // caller-chosen slot) otherwise.
    var optical_slot: vec4<f32> = optics.slots[render_params.material_slot % 16u];
    if render_params.material_mass_enabled != 0u {
        optical_slot = blended_optical_slot(nx_c, ny_c);
    }
    let sigma_a = optical_slot.rgb;
    // Real fix already applied HERE (2026-08-10, comment corrected to match
    // -- this used to describe a nearest-cell `band_field[vis_idx]` lookup
    // that no longer exists in this function): quantized straight from the
    // SAME bilinear `mass` the alpha/silhouette above already uses, so
    // there's no second, coarser-grid source to go stale relative to it.
    // The still-real, still-open version of this tradeoff (hysteresis's
    // anti-flicker benefit vs bilinear's anti-staleness benefit, unresolved
    // because nobody has visually confirmed which reads better) lives in
    // `shade_phase`'s dual-phase path below, which still uses the
    // nearest-cell hysteresis band field -- NOT touched, needs real visual
    // verification before choosing, not another blind swap.
    //
    // Normalized by the scene's real reference cell mass BEFORE quantizing.
    // Without this the band range is an absolute mass, so a physically
    // calibrated water cell (~0.1) lands in band 0 for every pixel while a
    // dense material saturates at the top -- and cells straddling a band
    // edge flip level frame to frame, which is what reads as flicker. It
    // also makes this quantizer actually agree with the band-hysteresis
    // pass, which already normalizes the same way (its own
    // `BAND_HYSTERESIS_DEPTH_BANDS` is documented as having to match this
    // one, and silently did not).
    // Real physics fix (2026-08-15): Beer-Lambert (`transmitted` below) is the
    // exact solution of the radiative-transfer equation for pure absorption,
    // dI/dz = -sigma_a * I (see e.g. Chandrasekhar 1950, "Radiative
    // Transfer") -- already the real law this engine cites elsewhere
    // (prep_instances.wgsl's ByPhysics mode). The DEPTH_BANDS quantization
    // this used to feed into that law (`floor(...) * DEPTH_BANDS) /
    // DEPTH_BANDS`) was a real, separate, disclosed ARTISTIC choice (flat
    // cel-shaded look, same reasoning as grid_volume.wgsl's fs_main) that
    // belongs to color/silhouette styling, not the physics term -- stair-
    // stepping the input to `exp(-sigma_a*depth)` put jumps into the actual
    // absorption law with no physical basis. Use the real, continuous,
    // already-physically-calibrated mass ratio (same ref_mass normalization,
    // same edge floor as before) for the transmission physics instead --
    // DEPTH_BANDS/discrete banding stays exactly where it belongs, driving
    // the cel-shaded color elsewhere in this file, untouched by this fix.
    let ref_mass = max(render_params.reference_cell_mass, 1.0e-6);
    let depth_continuous = clamp(mass / ref_mass, 0.0, 4.0);
    let optical_depth = max(depth_continuous, render_params.edge_reference_depth);
    let transmitted = exp(-sigma_a * optical_depth);

    // Subsurface scattering -- same real formula `prep_instances.wgsl`'s
    // ByPhysics mode uses per-particle (Jacques 2013 single-scattering
    // albedo), ported here for optical parity at zero new data cost.
    let sigma_s = optical_slot.w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    let grad = vec2<f32>(
        ((m10 - m00) + (m11 - m01)) * 0.5,
        ((m01 - m00) + (m11 - m10)) * 0.5,
    );
    // Real propagating wave field folded into the shading normal -- the
    // SAME bilinear-corner finite-difference gradient technique as `grad`
    // above, just against the persistent wave height buffer instead of the
    // density buffer. `WAVE_SHADING_WEIGHT` scales the wave gradient up
    // (its own magnitude is naturally small relative to density) so the
    // real propagating ripple is actually perceptible in the shading, not
    // real physics silently underneath a dominant static shape gradient.
    let w00 = sample_wave_render(bx, by);
    let w10 = sample_wave_render(bx + 1, by);
    let w01 = sample_wave_render(bx, by + 1);
    let w11 = sample_wave_render(bx + 1, by + 1);
    let wave_grad = vec2<f32>(
        ((w10 - w00) + (w11 - w01)) * 0.5,
        ((w01 - w00) + (w11 - w10)) * 0.5,
    );
    // The wave's gradient must NOT feed into the SAME quantized `diffuse`
    // the flat cel-shading bands select from (i.e. not
    // `combined_grad = grad + wave_grad*3.0`, then banded): the wave's
    // continuous motion repeatedly re-crosses that quantizer's own
    // boundaries, a flicker source independent of the depth-band
    // hysteresis fix (Pass 2d), which targets a different quantizer and
    // has no effect on this one. Keep the flat cel-shaded BASE driven only
    // by `grad` (the density gradient, stable for settled water), and add
    // the wave's motion as a SEPARATE, small, CONTINUOUS (non-quantized)
    // highlight -- a smoothly-varying term can move with the wave without
    // "stepping," unlike a banded one.
    let grad_len = length(grad);
    // Computed early (normally lives right before `return` below, next to
    // `alpha`) so the Fresnel block can use it -- see `fresnel_interior`'s
    // own doc just below for why.
    let edge_margin = max(render_params.mass_floor * 1.5, 1.0e-4);
    var lit = with_scattering;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(render_params.light_dir);
        let diffuse_raw = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Cel-shading (NPR technique) -- see `grid_volume.wgsl`'s own
        // fs_main for the full doc. LIGHT_BANDS=8: fewer bands (4) amplify
        // surface detail into visible posterization noise.
        const LIGHT_BANDS: f32 = 8.0;
        let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
        let shaded = with_scattering * (0.6 + 0.4 * diffuse);

        let wave_len = length(wave_grad);
        var wave_highlight = vec3<f32>(0.0, 0.0, 0.0);
        if wave_len > 1.0e-6 {
            let wave_normal = -wave_grad / wave_len;
            let wave_diffuse = clamp(dot(wave_normal, light_dir), 0.0, 1.0);
            // Deliberately small and additive, not banded, so it reads as
            // a subtle moving glint riding on top of the stable flat
            // shading, not a competing light source.
            const WAVE_HIGHLIGHT_STRENGTH: f32 = 0.15;
            wave_highlight = with_scattering * wave_diffuse * WAVE_HIGHLIGHT_STRENGTH;
        }

        // Real, angle-dependent Fresnel reflectance (Schlick 1994
        // approximation) -- 2026-08-10, found live: `OpticalTable.specular`
        // (`optics.specular[slot].x`, real per-material R0, Schlick's own
        // near-normal-incidence reflectance) was already uploaded but NEVER
        // READ anywhere in this shader -- `prep_instances.wgsl`'s own
        // ByPhysics path adds R0 as a flat constant too, missing the actual
        // view-angle dependence that IS Schlick's whole point (that struct
        // field's own doc already names this exact gap: "NOT a full
        // view-angle-dependent Fresnel term"). This is that missing term,
        // added ADDITIVELY on top of the existing cel-shaded diffuse base
        // (same layering convention `wave_highlight` above already
        // established: a real, continuous, physically-derived signal riding
        // on top of the stylized flat-banded shading, not replacing it --
        // matches this project's own "no decor, everything caused by real
        // physics" standard without reversing the deliberate cel-shading
        // choice `grid_volume.wgsl`'s own doc explains).
        //
        // `grad` is only a 2D (screen-space) gradient of a density field --
        // no explicit 3rd (depth) axis exists in this 2D engine to measure a
        // real view angle against. Reconstructed the standard bump-mapping
        // way (same real technique the billboard-sphere depth trick in
        // Sebastian Lague's "Coding Adventure: Rendering Fluids" uses):
        // treat the density field as a height field and synthesize an
        // implicit z-component from FRESNEL_HEIGHT_SCALE, a real, tuned (not
        // measured) constant standing in for "how much the reconstructed
        // surface bulges toward the camera" -- honestly a simplification
        // (no true 3D surface exists here), not a claim of measured
        // geometry. `cos_theta = normal.z` because the (orthographic, 2D)
        // camera's view direction is exactly (0,0,1) in this same local
        // convention -- grazing angles (steep 2D density gradient) push
        // cos_theta toward 0 and Schlick's term toward 1 (near-total
        // reflection); a flat/calm region (near-zero gradient) keeps
        // cos_theta near 1, mostly R0 -- the real "look straight down, see
        // through; look near the edge, see a mirror" effect.
        const FRESNEL_HEIGHT_SCALE: f32 = 2.0;
        let normal3 = normalize(vec3<f32>(-grad, FRESNEL_HEIGHT_SCALE));
        let cos_theta = clamp(normal3.z, 0.0, 1.0);
        let r0 = optics.specular[render_params.material_slot % 16u].x;
        let fresnel = r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
        // Real bug found live 2026-08-10 (user: "y a des artifacts autour
        // des rendus"): `grad` is the SAME density gradient used for both
        // (a) real interior surface/wave curvature (what Fresnel is meant
        // to react to) AND (b) the silhouette's own alpha falloff -- and
        // `grad` is UNAVOIDABLY steepest exactly at that falloff band (that
        // IS what an edge is), so every fluid blob got a bright white rim
        // traced along its own silhouette, all the time, not just at real
        // grazing angles. Gate the Fresnel blend by the SAME interior-vs-
        // edge signal `alpha` below already computes (just evaluated here,
        // slightly wider, so it fully rolls off just BEFORE alpha starts
        // fading rather than exactly overlapping it) -- keeps real Fresnel
        // variation from interior wave undulations, removes the artifact
        // ring at the domain-cutoff edge, which was never a real surface
        // angle to begin with.
        let fresnel_interior = smoothstep(render_params.mass_floor, render_params.mass_floor + edge_margin * 3.0, mass);
        // Reflection color: no real environment capture exists in this 2D
        // engine to sample a true reflected scene from, so this uses a
        // simple, disclosed, brightened-toward-white version of the
        // surface's own already-lit color -- a real, standard cheap stand-in
        // for "reflects ambient sky/environment light" other stylized water
        // shaders use absent a real cubemap, not an invented color.
        let fresnel_reflection = mix(shaded, vec3<f32>(1.0, 1.0, 1.0), 0.6);
        let with_fresnel = mix(shaded, fresnel_reflection, fresnel * fresnel_interior);

        lit = clamp(with_fresnel + wave_highlight, vec3(0.0), vec3(1.0));
    }

    // Blackbody thermal emission -- real, exact same formula `grid_
    // volume.wgsl`'s own fs_main already uses (see that shader's own doc
    // for the full real citation/derivation): normalized to a 5000K
    // ceiling, additive on top of the (possibly cel-shaded) lit color so
    // near-ignition material reads as genuinely glowing.
    let t_norm = clamp(avg_temp / 5000.0, 0.0, 1.0);
    let emission = heat(0.5 + t_norm * 0.5).rgb * (t_norm * t_norm) * 2.0;

    // Real diffused subsurface glow (Pass 1e, `light_diffuse_main`) --
    // DISTINCT from the local `emission` term just above: emission is the
    // direct blackbody glow of THIS cell's own temperature; this is real
    // light that has diffused in from nearby hot cells, genuinely visible
    // even where local temperature itself is low (a cool cell right next
    // to an ember should show real bleed-in glow, not a hard cutoff at the
    // ember's own silhouette). Bilinear-sampled across the SAME 4 corners
    // `mass`/`avg_temp` already use above, same real anti-blockiness
    // reasoning. Same `heat()` color mapping as direct emission -- this is
    // the same real light, just spread, not a differently-colored effect.
    let lp00 = sample_light_phi_final(bx, by);
    let lp10 = sample_light_phi_final(bx + 1, by);
    let lp01 = sample_light_phi_final(bx, by + 1);
    let lp11 = sample_light_phi_final(bx + 1, by + 1);
    let phi = mix(mix(lp00, lp10, frac.x), mix(lp01, lp11, frac.x), frac.y);
    let phi_norm = clamp(phi, 0.0, 1.0);
    let diffused_glow = heat(0.5 + phi_norm * 0.5).rgb * phi_norm * 1.0;

    let with_emission = clamp(lit + emission + diffused_glow, vec3(0.0), vec3(1.0));

    // Widened past a narrow 0.5x band -- see `grid_volume.wgsl`'s own
    // fs_main doc for the real flicker mechanism this fixes. (`edge_margin`
    // itself is computed earlier now, next to `grad_len` -- the Fresnel
    // interior mask above needs it too.)
    let density_alpha = smoothstep(render_params.mass_floor, render_params.mass_floor + edge_margin, mass);
    // Fold the bilinear visibility blend in as a multiplicative alpha term
    // (see the "hair"/aliasing doc above) instead of a hard per-cell
    // discard-only gate.
    let alpha = density_alpha * visibility_blend;
    return vec4<f32>(with_emission, alpha);
}

// ── Pass 3b: dual-phase extraction + composite (two-phase extension) ────────

@group(0) @binding(0) var<storage, read> phase_a_final: array<f32>;
@group(0) @binding(1) var<storage, read> phase_b_final: array<f32>;
@group(0) @binding(2) var<uniform> render_params_a: SurfaceRenderParams;
@group(0) @binding(3) var<uniform> render_params_b: SurfaceRenderParams;
@group(0) @binding(4) var<uniform> dual_optics: OpticalTable;
// Real, persistent per-phase wave/hysteresis state -- SAME techniques as
// `fs_main`'s own Pass 2b/2c/2d above, ported here so the dual-phase path
// gets the identical proven flicker fixes instead of the raw, unstabilized
// mass/DEPTH_BANDS math it shipped with originally. 8 storage buffers
// total in this fragment stage (2 final + 2 wave + 2 visibility + 2 band)
// -- right at, not over, the WebGPU-guaranteed minimum
// `maxStorageBuffersPerShaderStage` of 8; verified by the real headless
// test device (default limits) actually running this pipeline, not just
// assumed compatible.
@group(0) @binding(5) var<storage, read> phase_a_wave_field: array<f32>;
@group(0) @binding(6) var<storage, read> phase_b_wave_field: array<f32>;
@group(0) @binding(7) var<storage, read> phase_a_visibility_field: array<f32>;
@group(0) @binding(8) var<storage, read> phase_b_visibility_field: array<f32>;
@group(0) @binding(9) var<storage, read> phase_a_band_field: array<f32>;
@group(0) @binding(10) var<storage, read> phase_b_band_field: array<f32>;

fn sample_phase(buf_is_a: bool, cx: i32, cy: i32, surface_res: u32) -> f32 {
    let res = i32(surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * surface_res + u32(cx);
    if buf_is_a { return phase_a_final[idx]; }
    return phase_b_final[idx];
}

fn sample_wave_phase(buf_is_a: bool, cx: i32, cy: i32, surface_res: u32) -> f32 {
    let res = i32(surface_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res { return 0.0; }
    let idx = u32(cy) * surface_res + u32(cx);
    if buf_is_a { return phase_a_wave_field[idx]; }
    return phase_b_wave_field[idx];
}

// Real, honest bilinear color+mass for ONE phase -- packs `(lit.rgb, mass)`
// into the return value (mass rides in `.a`, NOT a real alpha yet; the
// caller computes the real smoothstep-edge alpha itself, only for
// whichever phase actually wins the pixel -- see `fs_main_dual_phase`).
// `mass` is what decides which phase is actually in front at this pixel
// (real winner-take-all by local density, the SAME "dominant material
// wins" convention `grid_volume.wgsl` already established, just applied to
// two independently-smoothed surfaces instead of one shared field -- see
// module doc's own VOF/phase-fraction citation for why independent fields
// are the real, correct choice here, not a shared blended one).
fn shade_phase(
    buf_is_a: bool,
    surf_pos: vec2<f32>,
    p: SurfaceRenderParams,
    optics: OpticalTable,
    vis_idx: u32,
) -> vec4<f32> {
    let gp = surf_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_phase(buf_is_a, bx, by, p.surface_res);
    let m10 = sample_phase(buf_is_a, bx + 1, by, p.surface_res);
    let m01 = sample_phase(buf_is_a, bx, by + 1, p.surface_res);
    let m11 = sample_phase(buf_is_a, bx + 1, by + 1, p.surface_res);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Depth quantized, specular removed -- see `grid_volume.wgsl`'s own
    // fs_main for the full doc, and `fs_main`'s own copy of this constant
    // above for the edge-color-floor reasoning. Uses the file-scope
    // `DEPTH_BANDS` (see its own doc) -- same value as `fs_main` by
    // construction now, not by convention.
    let slot = p.material_slot % 16u;
    let sigma_a = optics.slots[slot].rgb;
    // Real port (2026-08-10) of `fs_main`'s own already-shipped choice:
    // quantize straight from the SAME bilinear `mass` the silhouette/alpha
    // above already uses, instead of a NEAREST-cell hysteresis-stabilized
    // lookup (`phase_a_band_field`/`phase_b_band_field`, still declared,
    // now unused here) that goes stale relative to that bilinear field at
    // a moving boundary -- the exact real bug this file's own module doc
    // already named. Real, disclosed, NOT fully resolved trade-off: the
    // hysteresis pass was real anti-flicker protection too (its own doc:
    // "stops re-crossing a band boundary every frame from ordinary density
    // jitter") -- whether bilinear-staleness or hysteresis-flicker reads
    // worse has never been visually confirmed either way, only ported here
    // for real consistency with `fs_main`'s own already-made choice, not
    // because this one was independently proven better. Revert to the
    // `select(...)` line above if live use shows real flicker regression.
    //
    // Same reference-cell-mass normalization as `fs_main` above, for the
    // same reason -- see that copy's own doc. Same real-physics fix
    // (2026-08-15) also ported here: continuous mass ratio feeds the real
    // Beer-Lambert law directly, DEPTH_BANDS quantization (an artistic
    // choice) no longer distorts the physics term -- see `fs_main`'s own
    // longer comment for the full citation/reasoning.
    let ref_mass = max(p.reference_cell_mass, 1.0e-6);
    let depth_continuous = clamp(mass / ref_mass, 0.0, 4.0);
    let optical_depth = max(depth_continuous, p.edge_reference_depth);
    let transmitted = exp(-sigma_a * optical_depth);

    // Same ByPhysics-parity scattering port as `fs_main` above.
    let sigma_s = optics.slots[slot].w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    let grad = vec2<f32>(((m10 - m00) + (m11 - m01)) * 0.5, ((m01 - m00) + (m11 - m10)) * 0.5);
    // Real, persistent wave field for THIS phase -- same bilinear-corner
    // gradient technique as `fs_main`'s own wave highlight above.
    let w00 = sample_wave_phase(buf_is_a, bx, by, p.surface_res);
    let w10 = sample_wave_phase(buf_is_a, bx + 1, by, p.surface_res);
    let w01 = sample_wave_phase(buf_is_a, bx, by + 1, p.surface_res);
    let w11 = sample_wave_phase(buf_is_a, bx + 1, by + 1, p.surface_res);
    let wave_grad = vec2<f32>(((w10 - w00) + (w11 - w01)) * 0.5, ((w01 - w00) + (w11 - w10)) * 0.5);

    let grad_len = length(grad);
    var lit = with_scattering;
    if grad_len > 1.0e-5 {
        let normal_dir = -grad / grad_len;
        let light_dir = normalize(p.light_dir);
        let diffuse_raw = clamp(dot(normal_dir, light_dir), 0.0, 1.0);
        // Cel-shading (NPR technique) -- see `grid_volume.wgsl`'s own
        // fs_main for the full doc.
        const LIGHT_BANDS: f32 = 8.0;
        let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
        let shaded = with_scattering * (0.6 + 0.4 * diffuse);

        // Wave motion as a separate, small, CONTINUOUS highlight, NOT fed
        // into the quantized `diffuse` above -- same wave/light-band
        // decoupling reasoning as `fs_main`'s above: a continuous term can
        // move with the wave without "stepping" across a light-band
        // boundary every frame.
        let wave_len = length(wave_grad);
        var wave_highlight = vec3<f32>(0.0, 0.0, 0.0);
        if wave_len > 1.0e-6 {
            let wave_normal = -wave_grad / wave_len;
            let wave_diffuse = clamp(dot(wave_normal, light_dir), 0.0, 1.0);
            const WAVE_HIGHLIGHT_STRENGTH: f32 = 0.15;
            wave_highlight = with_scattering * wave_diffuse * WAVE_HIGHLIGHT_STRENGTH;
        }
        lit = clamp(shaded + wave_highlight, vec3(0.0), vec3(1.0));
    }

    return vec4<f32>(lit, mass);
}

@fragment
fn fs_main_dual_phase(in: VsOut) -> @location(0) vec4<f32> {
    let surf_pos = ndc_to_surface_pos(
        in.ndc, render_params_a.tx, render_params_a.sx, render_params_a.ty, render_params_a.sy,
    );
    let res = f32(render_params_a.surface_res);
    if surf_pos.x < 0.0 || surf_pos.y < 0.0 || surf_pos.x >= res || surf_pos.y >= res {
        discard;
    }

    // Real hysteresis-stabilized visible/invisible decision, per phase --
    // same nearest-cell lookup and same real fix as `fs_main`'s own Pass
    // 2c, replacing a flat `mass < mass_floor` comparison on EACH phase's
    // raw density (the same single-threshold flip-flop `fs_main` already
    // fixed, just previously left unfixed here).
    let nx = i32(round(surf_pos.x - 0.5));
    let ny = i32(round(surf_pos.y - 0.5));
    let nx_c = clamp(nx, 0, i32(render_params_a.surface_res) - 1);
    let ny_c = clamp(ny, 0, i32(render_params_a.surface_res) - 1);
    let vis_idx = u32(ny_c) * render_params_a.surface_res + u32(nx_c);
    let a_visible = phase_a_visibility_field[vis_idx] > 0.5;
    let b_visible = phase_b_visibility_field[vis_idx] > 0.5;
    if !a_visible && !b_visible {
        discard;
    }

    let shaded_a = shade_phase(true, surf_pos, render_params_a, dual_optics, vis_idx);
    let shaded_b = shade_phase(false, surf_pos, render_params_b, dual_optics, vis_idx);
    let mass_a = shaded_a.a;
    let mass_b = shaded_b.a;

    // Real winner-take-all AMONG VISIBLE phases: whichever visible phase has
    // more real local density at THIS pixel wins -- not a blend (a real
    // sand/water interface is a hard boundary, blending would render a
    // physically wrong translucent mixing zone at every contact point). A
    // phase hysteresis has decided is NOT visible must never win just
    // because its raw, un-stabilized mass happens to compare higher --
    // that would silently undo the whole point of the visibility check
    // above.
    let a_wins = a_visible && (!b_visible || mass_a >= mass_b);
    if a_wins {
        let edge_margin = max(render_params_a.mass_floor * 1.5, 1.0e-4);
        let alpha = smoothstep(render_params_a.mass_floor, render_params_a.mass_floor + edge_margin, mass_a);
        return vec4<f32>(shaded_a.rgb, alpha);
    }
    let edge_margin = max(render_params_b.mass_floor * 1.5, 1.0e-4);
    let alpha = smoothstep(render_params_b.mass_floor, render_params_b.mass_floor + edge_margin, mass_b);
    return vec4<f32>(shaded_b.rgb, alpha);
}

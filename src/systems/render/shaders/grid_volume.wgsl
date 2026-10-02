// MPM-native grid-volume rendering -- samples the solver's own P2G mass field
// directly instead of drawing one instanced splat per particle, so adjacent
// cells blend into one continuous shape instead of a cloud of discrete dots.
//
// Per-cell material colour from `GpuSimulation::attach_grid_material_render_gpu`'s
// opt-in per-material mass accumulator (`material_mass`): a mass-fraction-
// weighted blend (see `material_accum_at`), bilinear across the same 4 corner
// cells as the density and alpha (`sample_mass`), so mixed-material
// boundaries fade instead of stepping. Slot 0 for every cell when
// `material_mass_enabled` is 0.

const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;

// Smooth heat map: t ∈ [0, 1] → blue → cyan → green → yellow → red. Same
// formula as `prep_instances.wgsl`'s own `heat()` -- duplicated, not shared,
// since these compile as separate shader modules with no common-include
// mechanism in this codebase; keep both in sync if either changes.
fn heat(t: f32) -> vec4<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - abs(c - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    return vec4(r, g, b, 1.0);
}

struct GridVolumeParams {
    sx: f32,
    tx: f32,
    sy: f32,
    ty: f32,
    // Light direction from `SimConfig::light_dir` via
    // `Renderer::set_light_dir`, the value `rod::Phototropism` uses.
    light_dir: vec2<f32>,
    grid_res: u32,
    // A cell lighter than this is open air for the legacy column-depth term
    // (`accumulate_column_depth`), its only reader.
    mass_floor: f32,
    material_mass_enabled: u32,
    reference_cell_mass: f32,
    // Steepest mass difference between neighbouring cells across a straight
    // free surface (Rust's `free_surface_cell_step`), for `cel_lambert`.
    free_surface_step: f32,
    _pad2: f32,
}

struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

struct PhysicalRenderParams {
    spatial: vec4<f32>,
    incident_radiance: vec4<f32>,
    background_radiance: vec4<f32>,
    display_white_radiance: vec4<f32>,
    camera_direction: vec4<f32>,
    light_direction: vec4<f32>,
    // x = thermal-emission exposure anchor in kelvin, 0 = unset. See
    // `Renderer::set_emission_reference_temperature`. y/z/w reserved.
    emission: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> grid_int: array<u32>;
@group(0) @binding(1) var<uniform> params: GridVolumeParams;
@group(0) @binding(2) var<uniform> optics: OpticalTable;
// Read as i32, not reinterpreted as f32 (see `dominant_material`).
@group(0) @binding(3) var<storage, read> material_mass: array<i32>;
// Local peak of the cell mass (`grid_peak_main`), for the edge.
@group(0) @binding(4) var<storage, read> grid_peak_field: array<f32>;
@group(0) @binding(5) var<uniform> physical: PhysicalRenderParams;
// Per-cell transmittance of the declared light (`light_pass.wgsl`), read by
// the SI branch only.
@group(0) @binding(6) var<storage, read> light_transmittance_field: array<vec4<f32>>;

// ── Where matter ends on screen ─────────────────────────────────────────────
//
// The P2G mass is the matter's mass spread by the quadratic B-spline, a
// symmetric kernel of unit integral, so across a straight boundary it follows
// the kernel's cumulative and crosses half the mass the matter holds inside
// exactly at the boundary. `fs_main` draws the edge there: where the bilinear
// mass equals half the local peak, the largest cell mass within the kernel's
// reach. A material denser or lighter than the scene's reference finds its
// own edge, and matter thinner than the kernel (a lone particle, whose peak is
// a fraction of a full cell) still shows, at the kernel's size: the grid
// resolves nothing finer.
//
// The premise is matter of near-uniform density with a sharp boundary, a
// liquid or a solid. A gas has no free surface: its density varies
// continuously, and this contour cuts it into blobs wherever it falls under
// half its neighbourhood's peak. It should be drawn by its own transmittance
// instead (issue #60).
//
// The reach: one cell inside a straight boundary the field already holds
// 1 - 1/48 of the full mass (the B-spline's tail beyond one cell), and every
// bilinear corner of a pixel on the edge has a cell at least one cell inside
// the matter within two cells of it. The peak found is then within 1/48 of
// the full value, which moves the edge by under 0.02 cell.
struct GridPeakParams {
    grid_res: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> grid_peak_density_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> grid_peak_out: array<f32>;
@group(0) @binding(2) var<uniform> grid_peak_params: GridPeakParams;

const PEAK_REACH_CELLS: i32 = 2;

@compute @workgroup_size(8, 8, 1)
fn grid_peak_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = i32(grid_peak_params.grid_res);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= res || cy >= res { return; }
    var peak = 0.0;
    for (var dy = -PEAK_REACH_CELLS; dy <= PEAK_REACH_CELLS; dy++) {
        for (var dx = -PEAK_REACH_CELLS; dx <= PEAK_REACH_CELLS; dx++) {
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && y >= 0 && x < res && y < res {
                peak = max(peak, bitcast<f32>(grid_peak_density_in[u32(y * res + x) * 4u + 2u]));
            }
        }
    }
    grid_peak_out[u32(cy * res + cx)] = peak;
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

// Fullscreen triangle (no vertex/index buffer needed) -- covers the whole
// clip-space quad with 3 vertices via the classic oversized-triangle trick.
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

// Mass at grid cell (cx, cy), 0.0 outside the domain (the out-of-bounds
// convention of CPU `Grid::velocity_at`), so bilinear sampling fades to "no
// matter" at the domain edge without a border special case.
fn sample_mass(cx: i32, cy: i32) -> f32 {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    let idx = u32(cy) * params.grid_res + u32(cx);
    return bitcast<f32>(grid_int[idx * 4u + 2u]);
}

// Local peak at grid cell (cx, cy) (`grid_peak_main`), 0.0 outside the domain
// like `sample_mass`.
fn sample_peak(cx: i32, cy: i32) -> f32 {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    return grid_peak_field[u32(cy) * params.grid_res + u32(cx)];
}

// Vertical mass integral above this cell up to open air (see `optical_depth`
// in `fs_main`: sunlight attenuated with depth, Pope & Fry 1997). Marches
// toward increasing y (up, against gravity) one cell at a time; stops at
// MAX_COLUMN_SAMPLES (a GPU cost cap) or after 2 consecutive near-empty cells
// (the free surface, robust to one noisy near-zero read).
//
// MAX_COLUMN_SAMPLES = 56: the fluid demos (`basic_fluids.rs`/`_gui`/`_gpu`,
// one scene) use a 52-cell-deep column (`box_size: IVec2::new(14, 52)`), plus a
// margin (40 would truncate them). Cost at grid_res 64 with the column filled
// to y = 54: +742 us per frame over the local-density-only path (569.5 ->
// 1311.5 us, `diag_grid_volume_render_timing`), ~3.5% of the demo's
// ~20.8 ms frame.
const MAX_COLUMN_SAMPLES: i32 = 56;

fn accumulate_column_depth(cx: i32, cy: i32) -> f32 {
    var depth: f32 = 0.0;
    var consecutive_empty: i32 = 0;
    for (var i: i32 = 0; i < MAX_COLUMN_SAMPLES; i++) {
        let m = sample_mass(cx, cy + i);
        if m < params.mass_floor {
            consecutive_empty += 1;
            if consecutive_empty >= 2 {
                break;
            }
        } else {
            consecutive_empty = 0;
        }
        depth += m;
    }
    return depth;
}

// Light transmittance at cell (cx, cy); 1 (unattenuated) outside the
// domain, where no matter stands between the cell and the light.
fn light_at(cx: i32, cy: i32) -> vec3<f32> {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return vec3<f32>(1.0);
    }
    return light_transmittance_field[u32(cy) * params.grid_res + u32(cx)].rgb;
}

// Mass-weighted temperature at grid cell (cx, cy): sum(mass_p *
// temperature_p) over the particles scattered into it (the P2G convention of
// `ThermalDiffusion`). Divided by the bilinearly sampled mass elsewhere; kept
// unnormalized here so it interpolates correctly (interpolating averages
// would over-weight low-mass neighbours). 0.0 out of bounds, like
// `sample_mass`.
fn sample_weighted_temp(cx: i32, cy: i32) -> f32 {
    let res = i32(params.grid_res);
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    let idx = u32(cy) * params.grid_res + u32(cx);
    return bitcast<f32>(grid_int[idx * 4u + 0u]);
}

// Compare raw i32 `material_mass`, not bit-reinterpreted f32: accumulated
// fixed-point values commonly land in the IEEE 754 denormal range, and this
// GPU's fragment-shader ALU flushes denormals to zero on comparison, which
// silently breaks a bit-reinterpreted read.
//
// Blends per-slot mass fractions instead of picking a single dominant
// material (a hard per-cell winner flips 100%-A to 100%-B at a boundary
// instead of fading, producing visible speckle). Mixing rule: mixture
// absorbance ~= components' own absorbance weighted by relative amount
// ("Beyond Beer's Law: Spectral Mixing Rules," Applied Spectroscopy 2020,
// PubMed 32588637). Uses MASS fraction rather than the paper's VOLUME
// fraction (no per-slot volume field exists) -- an approximation that still
// holds under the paper's micro-homogeneous assumption, since one grid cell
// represents many particles, not a single sharp interface.
//
// Returns the raw (accum, total_mass) pair so `fs_main` blends 4 neighbour
// cells before one division, like `mass`/`weighted_temp`. Blending 4
// normalized colours would pull toward slot 0 wherever an empty corner falls
// back to the default; with raw sums an empty corner contributes (0, 0) and
// carries no weight.
struct MaterialAccum {
    accum: vec4<f32>,
    // Mass-weighted Fresnel R0, blended the same way absorption is: a cell
    // holding sand and water reflects like the mixture, not like whichever
    // of the two happens to weigh more.
    specular_accum: f32,
    // Mass-weighted shape-holding flag (`OpticalTable::specular[slot].y`),
    // blended the same way.
    holds_shape_accum: f32,
    total_mass: f32,
}

fn material_accum_at(cx: i32, cy: i32) -> MaterialAccum {
    let idx = u32(cy) * params.grid_res + u32(cx);
    let base = idx * MAX_RENDER_MATERIAL_SLOTS;
    var total_mass: f32 = 0.0;
    var accum: vec4<f32> = vec4<f32>(0.0);
    var specular_accum: f32 = 0.0;
    var holds_shape_accum: f32 = 0.0;
    for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
        let m = f32(max(material_mass[base + s], 0));
        total_mass += m;
        accum += m * optics.slots[s];
        specular_accum += m * optics.specular[s].x;
        holds_shape_accum += m * optics.specular[s].y;
    }
    return MaterialAccum(accum, specular_accum, holds_shape_accum, total_mass);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Invert the same orthographic mapping Renderer::set_camera uses:
    // clip = grid_pos * (sx, sy) + (tx, ty)  =>  grid_pos = (clip - t) / s
    // Same formula as `curvature_flow.wgsl`'s own `ndc_to_surface_pos` -- a
    // separate shader module, so WGSL has no way to share that function
    // here; keep this inverse in sync by hand if either one changes.
    let grid_pos = vec2<f32>(
        (in.ndc.x - params.tx) / params.sx,
        (in.ndc.y - params.ty) / params.sy,
    );
    let res = f32(params.grid_res);
    if grid_pos.x < 0.0 || grid_pos.y < 0.0 || grid_pos.x >= res || grid_pos.y >= res {
        discard;
    }

    // Bilinear sample against cell CENTERS (P2G's own convention: cell i's center
    // sits at grid position i+0.5, see p2g.wgsl's CELL_CENTER_OFFSET) -- smooths
    // the blocky nearest-cell look into a continuous density falloff at edges,
    // real MPM-render technique (same idea screen-space fluid rendering's
    // bilateral smoothing pass achieves, simpler since this is grid-native data
    // already, not a reconstructed depth buffer).
    let gp = grid_pos - vec2<f32>(0.5, 0.5);
    let base_cell = floor(gp);
    let frac = gp - base_cell;
    let bx = i32(base_cell.x);
    let by = i32(base_cell.y);

    let m00 = sample_mass(bx, by);
    let m10 = sample_mass(bx + 1, by);
    let m01 = sample_mass(bx, by + 1);
    let m11 = sample_mass(bx + 1, by + 1);
    let mass = mix(mix(m00, m10, frac.x), mix(m01, m11, frac.x), frac.y);

    // Same bilinear pattern as mass, for the mass-weighted temperature channel.
    let wt00 = sample_weighted_temp(bx, by);
    let wt10 = sample_weighted_temp(bx + 1, by);
    let wt01 = sample_weighted_temp(bx, by + 1);
    let wt11 = sample_weighted_temp(bx + 1, by + 1);
    let weighted_temp = mix(mix(wt00, wt10, frac.x), mix(wt01, wt11, frac.x), frac.y);
    // Average temperature: weighted sum over mass, both bilinear over the same
    // 4 cells, floored against near-zero mass at sparse edges.
    let avg_temp = weighted_temp / max(mass, 1.0e-4);

    // Where the matter ends (see "Where matter ends on screen" above): the
    // bilinear mass against half the local peak, over the same four cells.
    let peak = mix(
        mix(sample_peak(bx, by), sample_peak(bx + 1, by), frac.x),
        mix(sample_peak(bx, by + 1), sample_peak(bx + 1, by + 1), frac.x),
        frac.y,
    );
    // The share of the local peak held here, less one half: zero on the
    // edge, -1/2 where there is no matter at all. (Measured as a difference
    // of masses instead, it fades to zero without crossing it where both the
    // mass and the peak fade out, and the anti-aliasing below drew a line
    // there.)
    let inside = mass / max(peak, 1.0e-12) - 0.5;
    // How fast `inside` changes across this pixel: the edge below is
    // anti-aliased over it. A pixel wholly outside the matter (alpha 0) stops
    // here, before the shading it would not show.
    let edge_width = max(fwidth(inside), 1.0e-9);
    if peak <= 0.0 || inside <= -0.5 * edge_width {
        discard;
    }

    // Mass-fraction-weighted material blend, bilinear over the same 4 corners
    // as `mass`/`weighted_temp` (`bx`/`by`/`frac`, see `material_accum_at`).
    // Slot 0 only when all 4 corners are empty, as with material tracking off.
    var optical_slot: vec4<f32> = optics.slots[0];
    var specular_r0: f32 = optics.specular[0].x;
    var holds_shape: f32 = optics.specular[0].y;
    if params.material_mass_enabled != 0u {
        let ma00 = material_accum_at(bx, by);
        let ma10 = material_accum_at(bx + 1, by);
        let ma01 = material_accum_at(bx, by + 1);
        let ma11 = material_accum_at(bx + 1, by + 1);
        let accum_blend = mix(
            mix(ma00.accum, ma10.accum, frac.x),
            mix(ma01.accum, ma11.accum, frac.x),
            frac.y,
        );
        let mass_blend = mix(
            mix(ma00.total_mass, ma10.total_mass, frac.x),
            mix(ma01.total_mass, ma11.total_mass, frac.x),
            frac.y,
        );
        let specular_blend = mix(
            mix(ma00.specular_accum, ma10.specular_accum, frac.x),
            mix(ma01.specular_accum, ma11.specular_accum, frac.x),
            frac.y,
        );
        let holds_shape_blend = mix(
            mix(ma00.holds_shape_accum, ma10.holds_shape_accum, frac.x),
            mix(ma01.holds_shape_accum, ma11.holds_shape_accum, frac.x),
            frac.y,
        );
        if mass_blend > 0.0 {
            optical_slot = accum_blend / mass_blend;
            specular_r0 = specular_blend / mass_blend;
            holds_shape = holds_shape_blend / mass_blend;
        }
    }

    // Beer-Lambert absorption, same formula ByPhysics uses per-particle
    // (prep_instances.wgsl) but evaluated once per pixel against the grid's
    // own (now bilinear-smoothed) mass instead of a single particle's J.
    // Higher mass -> denser -> more absorption, giving a soft density-based
    // falloff at the shape's own edge instead of a hard per-particle silhouette.
    //
    // Color depth is floored at EDGE_COLOR_REFERENCE_DEPTH, separately from the
    // edge alpha below, which still uses the true raw mass -- without this
    // split, the band where the mass falls off renders as a near-white halo
    // before reaching full alpha.
    //
    // Depth QUANTIZED into flat bands (same real cel-shading technique the
    // lighting below already uses, extended here to the density-driven
    // color itself): a continuously-varying optical depth is what makes
    // this read as a smooth, photoreal 3D gradient (the "blobby" look
    // reported directly) even after the lighting itself was banded --
    // color needs the same treatment for a Celeste/Rain-World-style
    // flat-region look. Alpha (the edge) deliberately still uses the RAW,
    // unbanded `mass` just below -- banding the edge too would read as
    // blocky/pixel-stair-stepped (the "Minecraft" look explicitly rejected
    // as too limited), not the flat-color-with-smooth-edge look aimed for.
    //
    // Floor on optical depth used for edge color (shares this formula with
    // `curvature_flow.wgsl`). Too low and `exp(-sigma_a*depth)` barely
    // absorbs anything for typical (small) sigma_a, reading as a washed-out
    // near-neutral-gray ring instead of material color. Must be at least
    // one interior depth-band step (`1.0/DEPTH_BANDS`) so the edge reads as
    // solid material color right up to where alpha fades it.
    const EDGE_COLOR_REFERENCE_DEPTH: f32 = 3.0;
    // A separate shader module from `curvature_flow.wgsl` -- WGSL has no
    // cross-module const sharing, so this literal MUST be kept equal to
    // that file's own (file-scope) `DEPTH_BANDS` by hand.
    const DEPTH_BANDS: f32 = 4.0;
    let sigma_a = optical_slot.rgb;

    // Column-depth attenuation: with local density alone (clamped to [0,4]) a
    // 1-cell puddle and a 20-cell lake read the same. In this 2D side view the
    // darkening comes from sunlight crossing the water column down to this
    // point (Pope & Fry 1997, the source of the water sigma_a table), not from
    // a camera ray. `accumulate_column_depth` marches up to the free surface
    // summing mass, discretizing tau = integral sigma_a*rho ds along y.
    //
    // Matter that holds its shape (`holds_shape`, see `cel_lambert`) skips
    // it: the column models light crossing a liquid, and on a solid body it
    // shaded the body like a lit cylinder, dark at the bottom whatever its
    // shape. Measured on a still square and disc: top-to-bottom gradient
    // with it, one flat tone without it.
    //
    // Both depths count full cells, mass over `reference_cell_mass`, as the
    // visibility floor and the SI branch's `relative_density` do: in raw mass
    // a scene of 1 m cells (about 900 per cell) saturated every column black,
    // and water at about 0.1 per cell barely darkened with depth.
    let ref_mass = max(params.reference_cell_mass, 1.0e-12);
    let column_depth = accumulate_column_depth(bx, by) / ref_mass;
    let depth_banded = floor(clamp(mass / ref_mass, 0.0, 4.0) * DEPTH_BANDS) / DEPTH_BANDS;
    let column_banded = floor(clamp(column_depth, 0.0, 16.0) * DEPTH_BANDS) / DEPTH_BANDS
        * (1.0 - clamp(holds_shape, 0.0, 1.0));
    let optical_depth = max(max(depth_banded, column_banded), EDGE_COLOR_REFERENCE_DEPTH);
    let transmitted = exp(-sigma_a * optical_depth);

    // Subsurface scattering + Fresnel specular: same real formula
    // `prep_instances.wgsl`'s ByPhysics mode already uses per-particle (see
    // that file's doc for the Jacques 2013 / Schlick 1994 grounding) --
    // ported here so the grid-native and curvature-flow render paths reach
    // the same optical detail ByPhysics already had, at zero new cost (both
    // terms only need `optics`, already bound, and the `optical_depth`
    // already computed above).
    let sigma_s = optical_slot.w;
    let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1.0e-4));
    let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
    let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));

    // Blackbody thermal emission -- the same shared `blackbody.inc.wgsl`
    // every render path now calls, so a hot cell and a hot particle are the
    // same colour by construction rather than by keeping two copies in sync.
    // This path needs `sample_weighted_temp` above to have a temperature at
    // all: the buffer only ever scattered mass until that was added.
    let emission = blackbody_emission(
        avg_temp,
        physical.spatial.z,
        physical.display_white_radiance.rgb,
        physical.emission.x,
    );

    // Density gradient from the bilinear corners (standard volume-rendering
    // technique), pointing toward increasing mass; `cel_lambert`
    // (`cel_lighting.inc.wgsl`) shades it, weighted by its size against
    // `free_surface_step` (Rust's `free_surface_cell_step` on the physics
    // grid).
    //
    // No specular lobe here, deliberately: a continuous BRDF highlight
    // reads as "lit 3D surface," the opposite of the cel-shaded flat-2D
    // look -- flat cel-shaded 2D games don't render one on ordinary
    // terrain/water. `OpticalTable::specular` (R0) stays wired for
    // `prep_instances.wgsl`'s ByPhysics particle mode, which has no
    // gradient/normal to build a lobe from and is unrelated to this path.
    let grad = vec2<f32>(
        ((m10 - m00) + (m11 - m01)) * 0.5,
        ((m01 - m00) + (m11 - m10)) * 0.5,
    );
    let lit = clamp(
        with_scattering * cel_lambert(grad, params.light_dir, params.free_surface_step, holds_shape),
        vec3(0.0),
        vec3(1.0),
    );

    // Additive blackbody glow on top of the (possibly cel-shaded) lit color --
    // same additive placement `prep_instances.wgsl`'s ByPhysics mode uses
    // (`with_specular + emission`), so near-ignition material reads as
    // glowing rather than just a brighter base color.
    let with_emission = clamp(lit + emission, vec3(0.0), vec3(1.0));

    // The edge anti-aliased over one pixel: the share of this pixel on the
    // matter's side of `inside = 0`.
    let alpha = clamp(inside / edge_width + 0.5, 0.0, 1.0);
    if physical.spatial.z > 0.5 {
        // SI radiative transfer: absorption, single scattering and Fresnel
        // together (`radiative_transfer.inc.wgsl`). `cos_view` comes from the
        // density gradient used as the surface normal above, so a steep
        // gradient reflects and a flat one transmits.
        let relative_density = max(mass / max(params.reference_cell_mass, 1.0e-12), 0.0);
        let view_length_m = physical.spatial.y / max(abs(physical.camera_direction.z), 1.0e-6);
        let path_m = relative_density * view_length_m;
        let cos_view = 1.0 / sqrt(1.0 + dot(grad, grad));
        // The declared light as it reaches this point, after crossing the
        // matter on the light's side (`light_pass.wgsl`), bilinear over the
        // same four cells as `mass`.
        let light_reaching = mix(
            mix(light_at(bx, by), light_at(bx + 1, by), frac.x),
            mix(light_at(bx, by + 1), light_at(bx + 1, by + 1), frac.x),
            frac.y,
        );
        let radiance = slab_radiance(
            physical.background_radiance.rgb,
            physical.incident_radiance.rgb * light_reaching,
            sigma_a,
            sigma_s,
            path_m,
            specular_r0,
            cos_view,
        );
        let display_radiance = radiance / physical.display_white_radiance.rgb + emission;
        return vec4<f32>(clamp(display_radiance, vec3(0.0), vec3(1.0)), alpha);
    }
    return vec4<f32>(with_emission, alpha);
}

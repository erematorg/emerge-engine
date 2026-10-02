// ── Light transmittance: 2D volumetric shadows ──────────────────────────────
//
// The light pass of Hillaire's unified volumetric rendering ("Physically Based
// and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015 Advances in
// Real-Time Rendering) on the 2D slab: the declared light reaching a cell has
// crossed the matter between that cell and the domain edge on the light's
// side, so it arrives attenuated along that path. Two stages: the extinction
// stores each physics-grid cell's attenuation per metre of path, the
// mass-fraction blend of the slots' `penetration_attenuation_m_inv`
// (`energy::radiation`, computed on the CPU from each slot's measured
// optics) times `rho_rel` (`light_extinction_grid_main`, or for the surface
// `light_surface_cell_extinction_main` then `light_extinction_surface_main`);
// `light_march_main` sums it toward the light into a per-channel
// transmittance, which the SI branch
// of `grid_volume.wgsl`'s and `curvature_flow.wgsl`'s `fs_main` multiplies
// into the incident radiance. Only the SI branches read it: these are metres,
// and the legacy branches have no length unit (issue #57).
//
// Both render paths march the same physics-grid field, so they shadow alike.
// The grid volume reads the solver's own P2G mass; the curvature-flow surface
// averages its finer reconstructed density down to the physics grid first:
// marching at its own resolution costs the cube of the multiplier more
// (cells times steps), for shadow edges sharper than the solver resolves
// the matter that casts them.
//
// The scene is taken as a cross-section of a world uniform in depth, the
// side view of a 2D platformer: light comes from the scene's own sky and
// travels in the plane, not in through the slab's faces.
// `PhysicalRenderParams::light_direction` points toward the light; its
// in-plane part sets the march, and since the medium does not vary in
// depth, one cell step in the plane is a 3D path of `dx / |l_xy|`. A light
// with no in-plane component casts no in-plane shadow.

// Rust-side source of truth: gpu::step_params::subsystems::MAX_RENDER_MATERIAL_SLOTS.
const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;

struct PhysicalRenderParams {
    spatial: vec4<f32>,
    incident_radiance: vec4<f32>,
    background_radiance: vec4<f32>,
    display_white_radiance: vec4<f32>,
    camera_direction: vec4<f32>,
    light_direction: vec4<f32>,
    emission: vec4<f32>,
}

struct LightPassParams {
    // Side of the physics grid the light is marched on.
    res: u32,
    // Side of the density field the extinction is read from: `res` for the
    // grid volume, the surface buffer's own side for the curvature flow.
    source_res: u32,
    // Mass of one cell of matter at its reference density, in the source
    // field's units: `rho_rel = mass / reference_cell_mass`.
    reference_cell_mass: f32,
    material_mass_enabled: u32,
    // Slot whose optics a cell takes when per-material mass is off, or when
    // the cell carries none.
    fallback_slot: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Each slot's `penetration_attenuation_m_inv`, per metre (rgb).
    slot_attenuation: array<vec4<f32>, MAX_RENDER_MATERIAL_SLOTS>,
}

// Binding 0 is the density source, read by one extinction entry each: the
// P2G grid as raw words (mass at word 2 of each cell's four), or the
// reconstructed surface density, one f32 per surface cell.
@group(0) @binding(0) var<storage, read> light_grid_int: array<u32>;
@group(0) @binding(0) var<storage, read> light_surface_density: array<f32>;
// Per-slot mass of the same field, read as i32 (see `grid_volume.wgsl`'s
// `dominant_material`).
@group(0) @binding(1) var<storage, read> light_material_mass: array<i32>;
@group(0) @binding(2) var<uniform> light_pass_params: LightPassParams;
@group(0) @binding(3) var<uniform> light_physical: PhysicalRenderParams;
@group(0) @binding(4) var<storage, read_write> light_extinction: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> light_transmittance: array<vec4<f32>>;
// The surface path's per-surface-cell extinction, before its mean over each
// grid cell. Bound but unused on the grid path.
@group(0) @binding(6) var<storage, read_write> light_surface_extinction: array<vec4<f32>>;

// Optical depth past which the transmittance, exp(-9) < 1/8000, is below the
// last step of an 8-bit display channel: marching further changes no pixel.
const LIGHT_TAU_CUTOFF: f32 = 9.0;

// Attenuation of the matter in one source cell, whose per-slot masses start
// at `base` in `light_material_mass`: the mass-fraction blend of the slots'
// attenuation, or the fallback slot's own.
fn cell_attenuation(base: u32) -> vec3<f32> {
    var attenuation = light_pass_params.slot_attenuation[
        light_pass_params.fallback_slot % MAX_RENDER_MATERIAL_SLOTS
    ].rgb;
    if light_pass_params.material_mass_enabled != 0u {
        var total = 0.0;
        var accum = vec3<f32>(0.0);
        for (var s: u32 = 0u; s < MAX_RENDER_MATERIAL_SLOTS; s++) {
            let m = f32(max(light_material_mass[base + s], 0));
            total += m;
            accum += m * light_pass_params.slot_attenuation[s].rgb;
        }
        if total > 0.0 {
            attenuation = accum / total;
        }
    }
    return attenuation;
}

fn relative_density(mass: f32) -> f32 {
    return max(mass, 0.0) / max(light_pass_params.reference_cell_mass, 1.0e-12);
}

@compute @workgroup_size(8, 8, 1)
fn light_extinction_grid_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = light_pass_params.res;
    if gid.x >= res || gid.y >= res { return; }
    let idx = gid.y * res + gid.x;
    let rho_rel = relative_density(bitcast<f32>(light_grid_int[idx * 4u + 2u]));
    light_extinction[idx] = vec4<f32>(cell_attenuation(idx * MAX_RENDER_MATERIAL_SLOTS) * rho_rel, 0.0);
}

// The surface path in two steps, each one thread per output cell: every
// surface cell's own extinction, then each physics-grid cell's mean over the
// surface cells whose centres fall inside it. Extinction is linear in
// density, so the mean is what a ray crossing the whole grid cell sees.
@compute @workgroup_size(8, 8, 1)
fn light_surface_cell_extinction_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source_res = light_pass_params.source_res;
    if gid.x >= source_res || gid.y >= source_res { return; }
    let sidx = gid.y * source_res + gid.x;
    let rho_rel = relative_density(light_surface_density[sidx]);
    var extinction = vec3<f32>(0.0);
    if rho_rel > 0.0 {
        extinction = cell_attenuation(sidx * MAX_RENDER_MATERIAL_SLOTS) * rho_rel;
    }
    light_surface_extinction[sidx] = vec4<f32>(extinction, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn light_extinction_surface_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = light_pass_params.res;
    if gid.x >= res || gid.y >= res { return; }
    let source_res = light_pass_params.source_res;
    // Surface cell i has its centre at (i + 0.5) / r in grid cells, inside
    // grid cell g for i in [g r - 0.5, (g + 1) r - 0.5).
    let r = f32(source_res) / f32(res);
    let x0 = u32(max(ceil(f32(gid.x) * r - 0.5), 0.0));
    let x1 = max(min(u32(max(ceil(f32(gid.x + 1u) * r - 0.5), 0.0)), source_res), x0);
    let y0 = u32(max(ceil(f32(gid.y) * r - 0.5), 0.0));
    let y1 = max(min(u32(max(ceil(f32(gid.y + 1u) * r - 0.5), 0.0)), source_res), y0);
    var sum = vec3<f32>(0.0);
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            sum += light_surface_extinction[y * source_res + x].rgb;
        }
    }
    let count = max((x1 - x0) * (y1 - y0), 1u);
    light_extinction[gid.y * res + gid.x] = vec4<f32>(sum / f32(count), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn light_march_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = light_pass_params.res;
    if gid.x >= res || gid.y >= res { return; }
    let idx = gid.y * res + gid.x;
    let l_xy = light_physical.light_direction.xy;
    let in_plane = length(l_xy);
    if light_physical.spatial.z < 0.5 || in_plane < 1.0e-6 {
        light_transmittance[idx] = vec4<f32>(1.0);
        return;
    }
    let step = l_xy / in_plane;
    let path_per_step_m = light_physical.spatial.x / in_plane;
    // Start one step upwind: the cell's own thickness along the view is
    // already in `slab_radiance`'s scattered term.
    var p = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5) + step;
    var tau = vec3<f32>(0.0);
    let max_steps = i32(2u * res);
    for (var k: i32 = 0; k < max_steps; k++) {
        let c = vec2<i32>(floor(p));
        if c.x < 0 || c.y < 0 || c.x >= i32(res) || c.y >= i32(res) { break; }
        tau += light_extinction[u32(c.y) * res + u32(c.x)].rgb * path_per_step_m;
        if min(tau.r, min(tau.g, tau.b)) > LIGHT_TAU_CUTOFF { break; }
        p += step;
    }
    light_transmittance[idx] = vec4<f32>(exp(-tau), 1.0);
}

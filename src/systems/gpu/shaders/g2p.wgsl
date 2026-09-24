// G2P — gather grid velocity/momentum into particle velocity and APIC affine matrix C.
// One thread per particle. F update, plasticity, position advance: particles_update.wgsl.

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

struct Cell {
    momentum: vec2<f32>, // after grid_update this holds velocity, not momentum
    mass:     f32,
    _pad:     f32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    _pad0:              u32,
    _pad1:              u32,
    // True (nonzero) iff any particle anywhere has contact_group != 0 this frame --
    // repurposes the third pad slot, see GpuStepParams::contact_active's Rust doc. When
    // false, resolve_contact/gather_contact_points never ran this frame (skipped as
    // provable dead work), so resolved_grip_v/resolved_rest_v are NOT safe to read --
    // read the plain grid velocity directly instead, mirroring CPU's
    // Grid::has_contact_activity() gate in transfer.rs exactly.
    contact_active:     u32,
}

// GPU/CPU parity fix (2026-08-15) -- see Rust `MaterialParams`'s own doc.
// Byte-exact mirror of the same struct in p2g.wgsl/particles_update.wgsl
// (must all agree, they share one uniform buffer). Only
// `owns_deformation_volume_state` is actually read here -- every other
// field exists purely for layout parity.
struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32,
    compression_limit:       f32,
    stretch_limit:           f32,
    rest_density:            f32,
    eos_stiffness:           f32,
    eos_power:               f32,
    dynamic_viscosity:       f32,
    volume_ratio_min:        f32,
    volume_ratio_max:        f32,
    dp_h0:                   f32,
    dp_h1:                   f32,
    dp_h2:                   f32,
    dp_h3:                   f32,
    active_stress_coeff:     f32,
    hardening_modulus:       f32,
    thermal_viscosity_coeff: f32,
    thermal_expansion:       f32,
    pressure_floor:          f32,
    bulk_viscosity:          f32,
    surface_tension_coeff:   f32,
    cohesion_coeff:          f32,
    owns_deformation_volume_state: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const MAX_MATERIALS:        u32 = {{MAX_MATERIALS}}u;
const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const NUM_FLOOR:            f32 = 1e-6;

@group(0) @binding(0) var<storage, read_write> particles:   array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid:        array<Cell>;
@group(0) @binding(2) var<uniform>             materials:   array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>             step_params: StepParams;
// Multi-field contact (GPU port) — resolved velocities from resolve_contact_main, one
// per grid node, ALREADY defaulted to the ordinary total velocity everywhere a real
// contact-active field wasn't found (see resolve_contact.wgsl's resolve_cell doc) —
// safe to read unconditionally at every stencil node, mirroring CPU's
// grip_velocity_at/rest_velocity_at fallback exactly.
@group(1) @binding(17) var<storage, read_write> resolved_grip_v: array<vec2<f32>>;
@group(1) @binding(18) var<storage, read_write> resolved_rest_v: array<vec2<f32>>;

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

// Free-surface velocity extrapolation for an UNTOUCHED grid node -- mirrors
// CPU's `Grid::velocity_at_or_extrapolated` (`a236fef`, 2026-08-13). An
// untouched node's raw `cell.momentum` is a stale zero, not a real velocity
// -- gathering it reads as an artificial jump across a free-surface
// particle's own stencil, which LOOKS like stretching even in true free
// fall (where div(v) must be exactly 0). Verified live: free fall's own J
// went to exactly `[1.000,1.000]`, bit-for-bit matching this fix's own
// CPU-side measured result. Slip wall on the extrapolated value (matching
// CPU's `apply_slip_wall_velocity`) so a near-boundary cell doesn't feed
// back a velocity that ignores the wall.
fn extrapolated_boundary_velocity(
    particle_v: vec2<f32>,
    cx: i32,
    cy: i32,
    res: i32,
    gravity: vec2<f32>,
    dt: f32,
    boundary_thickness: u32,
) -> vec2<f32> {
    var v = particle_v + gravity * dt;
    let bt = i32(boundary_thickness);
    if cx < bt          && v.x < 0.0 { v.x = 0.0; }
    if cx >= res - bt   && v.x > 0.0 { v.x = 0.0; }
    if cy < bt          && v.y < 0.0 { v.y = 0.0; }
    if cy >= res - bt   && v.y > 0.0 { v.y = 0.0; }
    return v;
}

@compute @workgroup_size(64, 1, 1)
fn g2p_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_idx = gid.x;
    if p_idx >= step_params.particle_count { return; }

    let p   = particles[p_idx];
    let res = step_params.grid_res;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));

    // Sleeping particles are never gathered into — same as CPU, which excludes them
    // from G2P entirely, leaving v/velocity_gradient frozen at whatever they were when
    // they fell asleep. Exception: wake propagation — if a nearby cell shows REAL motion
    // this substep, this particle wakes and falls through to the full gather below,
    // getting a real G2P this same substep (matches CPU: wake_particle happens before
    // G2P runs).
    //
    // Checks velocity, not mass: P2G now scatters mass for every particle, awake or
    // asleep (sleeping particles still need to deposit support for neighbors resting on
    // them — see p2g.wgsl). So "mass nearby" is true almost everywhere near any particle
    // at all, sleeping or not, and can no longer distinguish real activity from a calm,
    // settled neighbor. grid.momentum holds actual velocity by this point (grid_update
    // already converted it) — a cell fed only by frozen, at-rest particles has velocity
    // near zero; one fed by a genuinely moving particle does not.
    if p.sleeping != 0u {
        var should_wake = false;
        for (var di: i32 = -1; di <= 1; di++) {
            for (var dj: i32 = -1; dj <= 1; dj++) {
                let cx = base.x + di;
                let cy = base.y + dj;
                if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) { continue; }
                let cell = grid[u32(cy) * res + u32(cx)];
                if cell.mass > NUM_FLOOR && length(cell.momentum) > step_params.sleep_threshold {
                    should_wake = true;
                }
            }
        }
        if !should_wake { return; }
        particles[p_idx].sleeping = 0u;
    }

    // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
    // velocity_gradient=0 instead of gathering from the grid -- mirrors
    // `gather_grid_to_particles`'s CPU behavior exactly (see transfer.rs). The
    // particle's own mass/stress still scattered into P2G normally (unconditional
    // there, same as sleeping particles), so it remains a real, immovable anchor
    // other bodies push against.
    if p.pinned != 0u {
        particles[p_idx].v                 = vec2<f32>(0.0);
        particles[p_idx].velocity_gradient = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        return;
    }

    var new_v       = vec2<f32>(0.0);
    var B_col0      = vec2<f32>(0.0);
    var B_col1      = vec2<f32>(0.0);
    var new_density = 0.0; // Σ w_i·m_i — grid-gathered density, avoids F-tracked drift

    // Multi-field contact (GPU port): a grip particle (contact_group != 0) gathers
    // from the resolved GRIP field; any other particle (the "rest" field, the default)
    // gathers from the resolved REST field -- exact port of CPU's
    // gather_grid_to_particles routing (transfer.rs), which reads
    // grid.grip_velocity_at/rest_velocity_at by the SAME contact_group check. Density
    // still comes from the ordinary total mass field (unaffected by which velocity
    // field a particle reads — mirrors CPU exactly, mass is never per-field).
    let is_grip = p.contact_group != 0u;
    // Global gate (mirrors CPU's Grid::has_contact_activity() check at
    // transfer.rs's gather_grid_to_particles call site exactly): when NO particle anywhere
    // uses contact_group this frame, resolve_contact/gather_contact_points were skipped
    // entirely (see contact_active's doc, step.rs), so resolved_grip_v/resolved_rest_v were
    // never populated -- reading them here would be reading stale/garbage data, not just an
    // unnecessary read. Falls back to the plain grid velocity in that case, same as CPU.
    let contact_active = step_params.contact_active != 0u;

    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let cx = base.x + di;
            let cy = base.y + dj;
            if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) { continue; }

            let cell_dist = vec2<f32>(f32(cx), f32(cy)) + vec2<f32>(CELL_CENTER_OFFSET) - p.x;
            let w = bspline_w(cell_dist.x) * bspline_w(cell_dist.y);

            let node_idx = u32(cy) * res + u32(cx);
            let cell   = grid[node_idx];
            let touched_v = select(
                cell.momentum,
                select(resolved_rest_v[node_idx], resolved_grip_v[node_idx], is_grip),
                contact_active,
            );
            // Free-surface velocity extrapolation for untouched nodes -- see
            // `extrapolated_boundary_velocity`'s own doc for the full account.
            let extrap_v = extrapolated_boundary_velocity(
                p.v, cx, cy, i32(res), step_params.gravity, step_params.dt,
                step_params.boundary_thickness,
            );
            let cell_v = select(extrap_v, touched_v, cell.mass > NUM_FLOOR);

            new_v       += w * cell_v;
            B_col0      += w * cell_v * cell_dist.x;
            B_col1      += w * cell_v * cell_dist.y;
            new_density += w * cell.mass;
        }
    }

    // Velocity clamp: !(spd <= limit) also catches NaN (NaN <= x = false).
    // Inf guard: if spd=Inf, inv=0, then Inf×0=NaN — zero out via select.
    let spd = length(new_v);
    if !(spd <= step_params.vel_limit) {
        let inv = step_params.vel_limit / spd;
        new_v = select(new_v * inv, vec2<f32>(0.0), !(inv > 0.0));
    }

    // C = B · D_inverse (APIC affine velocity gradient)
    // No C-clamp: CPU gather_grid_to_particles has none. Clamping C at 0.5*vel_limit
    // fires at natural impact velocities (C ~ 6v) and under-deforms F, killing elastic bounce.
    // The velocity clamp above already bounds the energy; CFL bounds the timestep.
    let C = mat2x2<f32>(B_col0, B_col1) * step_params.kernel_d_inverse;

    particles[p_idx].v                 = new_v;
    particles[p_idx].velocity_gradient = C;

    // GPU/CPU parity fix (2026-08-15): materials that own their own
    // deformation-derived volume state (today: strict fluids) skip this
    // raw kernel-mass gather entirely -- it's free-surface-biased and
    // unbounded (real, measured: water density drifting to [0.0116,0.358]
    // against a rest density of 0.1, well outside what the analytical,
    // J-clamp-derived formula in particles_update.wgsl could ever produce).
    // Mirrors CPU's `estimate_particle_volumes` (density.rs), which
    // `continue`s past exactly these particles for the identical reason.
    // Density/volume are left untouched here for them -- particles_update.wgsl
    // overwrites both, every substep, from the material's own clamped F.
    if materials[p.material_id].owns_deformation_volume_state == 0u {
        let density = max(new_density, NUM_FLOOR);
        particles[p_idx].density = density;
        particles[p_idx].volume  = p.mass / density;
    }
}

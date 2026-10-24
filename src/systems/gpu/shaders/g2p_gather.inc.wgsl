// G2P -- gather grid velocity into particle velocity and APIC affine matrix C,
// as a function the fused `g2p_update_main` (particles_update.wgsl) calls before
// the F update / plasticity / position advance and the force fields, all in one
// dispatch per particle. Formerly its own pass (`g2p.wgsl`); fusing the three
// per-particle passes saves two dispatches and two full particle load/stores
// per substep.
//
// Host module must declare `Particle`, `StepParams` (with `contact_active`),
// `MaterialParams`, `NUM_FLOOR`, and the `particles`/`materials`/`step_params`
// bindings.

struct Cell {
    momentum: vec2<f32>, // after grid_update this holds velocity, not momentum
    mass:     f32,
    _pad:     f32,
}

const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;

@group(0) @binding(1) var<storage, read_write> grid:        array<Cell>;
// Multi-field contact (GPU port) -- resolved velocities from resolve_contact_main, one
// per grid node, ALREADY defaulted to the ordinary total velocity everywhere a real
// contact-active field wasn't found (see resolve_contact.wgsl's resolve_cell doc) --
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

// Free-surface velocity extrapolation for an untouched grid node, mirroring
// CPU's `Grid::velocity_at_or_extrapolated`. An untouched node's raw
// `cell.momentum` is a stale zero, not a velocity; gathering it makes a
// free-surface particle's stencil look stretched even in free fall (where
// div(v) must be exactly 0; with extrapolation free fall's J stays at
// `[1.000,1.000]`, as on CPU). Slip wall on the extrapolated value (as CPU's
// `apply_slip_wall_velocity`) so a near-boundary cell does not feed back a
// velocity that ignores the wall.
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

// Not done: a workgroup-local copy of this workgroup's grid patch (loaded once
// cooperatively, read from shared memory by its 64 particles) measured no
// difference -- 112-120us without vs 115-125us with, on the dam break. The whole
// grid is 64KB at grid_res=64 and already sits in the GPU's L2; the extra
// barriers and the cooperative load cost as much as the saved global reads.

// Gathers into `*pp` (and writes the same fields to `particles[p_idx]`, exactly as
// the standalone pass did). Leaves both untouched for a sleeping particle that
// does not wake this substep.
fn g2p_gather(p_idx: u32, pp: ptr<function, Particle>) {
    let p   = *pp;
    let res = step_params.grid_res;
    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));

    // Sleeping particles are never gathered into -- same as CPU, which excludes them
    // from G2P entirely, leaving v/velocity_gradient frozen at whatever they were when
    // they fell asleep. Exception: wake propagation -- if a nearby cell shows motion
    // this substep, this particle wakes and falls through to the full gather below,
    // getting a G2P this same substep (matches CPU: wake_particle happens before
    // G2P runs).
    //
    // Checks velocity, not mass: P2G now scatters mass for every particle, awake or
    // asleep (sleeping particles still need to deposit support for neighbors resting on
    // them -- see p2g.wgsl). So "mass nearby" is true almost everywhere near any particle
    // at all, sleeping or not, and can no longer distinguish real activity from a calm,
    // settled neighbor. grid.momentum holds actual velocity by this point (grid_update
    // already converted it) -- a cell fed only by frozen, at-rest particles has velocity
    // near zero; one fed by a moving particle does not.
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
        (*pp).sleeping = 0u;
    }

    // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
    // velocity_gradient=0 instead of gathering from the grid -- mirrors
    // `gather_grid_to_particles`'s CPU behavior exactly (see transfer.rs). The
    // particle's own mass/stress still scattered into P2G normally (unconditional
    // there, same as sleeping particles), so it remains a immovable anchor
    // other bodies push against.
    if p.pinned != 0u {
        particles[p_idx].v                 = vec2<f32>(0.0);
        particles[p_idx].velocity_gradient = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        (*pp).v                 = vec2<f32>(0.0);
        (*pp).velocity_gradient = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        return;
    }

    var new_v       = vec2<f32>(0.0);
    var B_col0      = vec2<f32>(0.0);
    var B_col1      = vec2<f32>(0.0);
    var new_density = 0.0; // Σ w_i·m_i -- grid-gathered density, avoids F-tracked drift

    // Multi-field contact (GPU port): a grip particle (contact_group != 0) gathers
    // from the resolved GRIP field; any other particle (the "rest" field, the default)
    // gathers from the resolved REST field -- exact port of CPU's
    // gather_grid_to_particles routing (transfer.rs), which reads
    // grid.grip_velocity_at/rest_velocity_at by the SAME contact_group check. Density
    // still comes from the ordinary total mass field (unaffected by which velocity
    // field a particle reads -- mirrors CPU exactly, mass is never per-field).
    let is_grip = p.contact_group != 0u;
    // Global gate (mirrors CPU's Grid::has_contact_activity() check at
    // transfer.rs's gather_grid_to_particles call site exactly): when NO particle anywhere
    // uses contact_group this frame, resolve_contact/gather_contact_points were skipped
    // entirely (see contact_active's doc, step.rs), so resolved_grip_v/resolved_rest_v were
    // never populated -- reading them here would be reading stale/garbage data, not just an
    // unnecessary read. Falls back to the plain grid velocity in that case, same as CPU.
    let contact_active = step_params.contact_active != 0u;
    // A node below NUM_FLOOR mass counts as untouched here and is left out
    // of B; an axis needs >= 2 included offsets to give a derivative. A
    // wall (out-of-bounds) cell is a boundary value and counts. The
    // CPU has no such nodes: it gathers only nodes its own P2G created.
    var included_di = array<bool, 3>(false, false, false);
    var included_dj = array<bool, 3>(false, false, false);

    for (var di: i32 = -1; di <= 1; di++) {
        for (var dj: i32 = -1; dj <= 1; dj++) {
            let cx = base.x + di;
            let cy = base.y + dj;
            if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) {
                included_di[di + 1] = true;
                included_dj[dj + 1] = true;
                continue;
            }

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
            // `extrapolated_boundary_velocity`'s doc for the full account.
            let extrap_v = extrapolated_boundary_velocity(
                p.v, cx, cy, i32(res), step_params.gravity, substep_dt(),
                step_params.boundary_thickness,
            );
            let is_touched = cell.mass > NUM_FLOOR;
            let cell_v = select(extrap_v, touched_v, is_touched);

            // An extrapolated value is an assumed uniform stand-in with no
            // gradient: it counts toward `new_v` but not toward B. Plain path
            // only: a contact-active node's `touched_v` already falls back to
            // the total velocity in resolve_contact.wgsl.
            let excluded_from_gradient = !is_touched && !contact_active;

            new_v       += w * cell_v;
            if !excluded_from_gradient {
                included_di[di + 1] = true;
                included_dj[dj + 1] = true;
                B_col0 += w * cell_v * cell_dist.x;
                B_col1 += w * cell_v * cell_dist.y;
            }
            new_density += w * cell.mass;
        }
    }
    if (i32(included_di[0]) + i32(included_di[1]) + i32(included_di[2])) < 2 {
        B_col0 = vec2<f32>(0.0);
    }
    if (i32(included_dj[0]) + i32(included_dj[1]) + i32(included_dj[2])) < 2 {
        B_col1 = vec2<f32>(0.0);
    }

    // Velocity clamp: !(spd <= limit) also catches NaN (NaN <= x = false).
    // Inf guard: if spd=Inf, inv=0, then Inf×0=NaN -- zero out via select.
    let spd = length(new_v);
    if !(spd <= substep_vel_limit(substep_dt())) {
        let inv = substep_vel_limit(substep_dt()) / spd;
        new_v = select(new_v * inv, vec2<f32>(0.0), !(inv > 0.0));
    }

    // C = B · D_inverse (APIC affine velocity gradient). No clamp and no
    // relaxation, matching CPU `gather_grid_to_particles` exactly. Fluid
    // div(v) is the trace of C (see `trace2` in `particles_update.wgsl`); with
    // that, GPU and a CPU twin of the same scene agree within a few percent.
    let C = mat2x2<f32>(B_col0, B_col1) * step_params.kernel_d_inverse;

    particles[p_idx].v                 = new_v;
    particles[p_idx].velocity_gradient = C;
    (*pp).v                 = new_v;
    (*pp).velocity_gradient = C;

    // Materials that own their deformation-derived volume state (strict
    // fluids) skip this raw kernel-mass gather: it is free-surface-biased and
    // unbounded (measured: water density drifting to [0.0116,0.358] against a
    // rest density of 0.1, outside anything the J-clamped formula in
    // particles_update.wgsl can produce). Mirrors CPU's
    // `estimate_particle_volumes` (density.rs), which skips the same
    // particles. particles_update.wgsl overwrites their density and volume
    // every substep from the material's clamped F.
    if materials[p.material_id].owns_deformation_volume_state == 0u {
        let density = max(new_density, NUM_FLOOR);
        particles[p_idx].density = density;
        particles[p_idx].volume  = p.mass / density;
        (*pp).density = density;
        (*pp).volume  = p.mass / density;
    }
}

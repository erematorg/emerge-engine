// Grid update — momentum normalization, gravity, force fields, boundary enforcement.
// Runs between P2G and G2P. One thread per cell.

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
}

struct ForceFieldEntry {
    field_type:    u32,
    material_mask: u32,
    _pad0:         u32,
    _pad1:         u32,
    params01:      vec4<f32>,
    params45:      vec4<f32>,
}

struct ForceFieldsParams {
    count:   u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
    entries: array<ForceFieldEntry, 16>,
}

const MASS_FLOOR:         f32 = 1e-10;
const MASS_ATOMIC_SCALE:  f32 = 1000000.0;
const MOM_ATOMIC_SCALE:   f32 = 100000.0;
const CELL_CENTER_OFFSET: f32 = 0.5;
const FIELD_GRAVITY_WELL: u32 = 1u;
const FIELD_COULOMB:      u32 = 2u;
const MAX_FORCE_FIELDS:   u32 = {{MAX_FORCE_FIELDS}}u;
const FF_NUM_FLOOR:       f32 = 1e-10;

@group(0) @binding(1) var<storage, read_write> grid_int:    array<i32>;
@group(0) @binding(3) var<uniform>             step_params: StepParams;
@group(0) @binding(4) var<uniform>             force_fields: ForceFieldsParams;

// Smooth taper from 1 at switch_on to 0 at cutoff (cubic Hermite).
fn force_switch(dist: f32, cutoff: f32, switch_on: f32) -> f32 {
    if dist <= switch_on { return 1.0; }
    if dist >= cutoff    { return 0.0; }
    let t = (cutoff - dist) / (cutoff - switch_on);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(8, 8, 1)
fn grid_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cx = gid.x;
    let cy = gid.y;
    let res = step_params.grid_res;
    if cx >= res || cy >= res { return; }

    // Decode fixed-point i32 → float mass. Write it back as bitcast so g2p reads it as f32.
    let base4 = (cy * res + cx) * 4u;
    let mass  = f32(grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
    grid_int[base4 + 2u] = bitcast<i32>(mass);

    // Empty cells: gravity for stray particles, but enforce boundary slip so floor/wall
    // cells don't feed downward velocity into the G2P gather and over-compress blobs.
    if mass < MASS_FLOOR {
        var grav_vel = step_params.gravity * step_params.dt;
        let bt2 = step_params.boundary_thickness;
        if cx < bt2          && grav_vel.x < 0.0 { grav_vel.x = 0.0; }
        if cx >= res - bt2   && grav_vel.x > 0.0 { grav_vel.x = 0.0; }
        if cy < bt2          && grav_vel.y < 0.0 { grav_vel.y = 0.0; }
        if cy >= res - bt2   && grav_vel.y > 0.0 { grav_vel.y = 0.0; }
        grid_int[base4 + 0u] = bitcast<i32>(grav_vel.x);
        grid_int[base4 + 1u] = bitcast<i32>(grav_vel.y);
        return;
    }

    let mom_x = f32(grid_int[base4 + 0u]) / MOM_ATOMIC_SCALE;
    let mom_y = f32(grid_int[base4 + 1u]) / MOM_ATOMIC_SCALE;
    var vel   = vec2<f32>(mom_x, mom_y) / mass;

    vel += step_params.gravity * step_params.dt;

    // Apply cursor force fields in grid space (same substep as position advance — no lag).
    if force_fields.count > 0u {
        let cell_pos = vec2<f32>(f32(cx), f32(cy)) + vec2<f32>(CELL_CENTER_OFFSET);
        for (var fi: u32 = 0u; fi < force_fields.count && fi < MAX_FORCE_FIELDS; fi++) {
            let entry = force_fields.entries[fi];
            if entry.field_type == FIELD_GRAVITY_WELL {
                let src    = vec2<f32>(entry.params01.x, entry.params01.y);
                let gm     = entry.params01.z;
                let eps2   = entry.params01.w;
                let cutoff = entry.params45.z;
                let sw_on  = entry.params45.w;
                let r      = cell_pos - src;
                let r2     = dot(r, r);
                let r_len  = sqrt(r2);
                if cutoff <= 0.0 || r_len < cutoff {
                    let r2_soft = r2 + eps2;
                    let r3 = r2_soft * sqrt(r2_soft);
                    if r3 >= FF_NUM_FLOOR {
                        var acc = -(gm / r3) * r;
                        if cutoff > 0.0 { acc *= force_switch(r_len, cutoff, sw_on); }
                        vel += acc * step_params.dt;
                    }
                }
            } else if entry.field_type == FIELD_COULOMB {
                let src           = vec2<f32>(entry.params01.x, entry.params01.y);
                let charge_factor = entry.params01.z;
                let eps2          = entry.params01.w;
                let cutoff        = entry.params45.z;
                let sw_on         = entry.params45.w;
                let r             = cell_pos - src;
                let r2            = dot(r, r);
                let r_len         = sqrt(r2);
                if cutoff <= 0.0 || r_len < cutoff {
                    let r2_soft = r2 + eps2;
                    let r3 = r2_soft * sqrt(r2_soft);
                    if r3 >= FF_NUM_FLOOR {
                        var acc = (charge_factor / r3) * r;
                        if cutoff > 0.0 { acc *= force_switch(r_len, cutoff, sw_on); }
                        vel += acc * step_params.dt;
                    }
                }
            }
        }
    }

    // Slip boundary: zero inward normal velocity near each wall.
    let bt = step_params.boundary_thickness;
    if cx < bt          && vel.x < 0.0 { vel.x = 0.0; }
    if cx >= res - bt   && vel.x > 0.0 { vel.x = 0.0; }
    if cy < bt          && vel.y < 0.0 { vel.y = 0.0; }
    if cy >= res - bt   && vel.y > 0.0 { vel.y = 0.0; }

    // CFL clamp before G2P — bounds both particle velocity AND affine matrix C at the source.
    let spd = length(vel);
    if spd > step_params.vel_limit { vel *= step_params.vel_limit / spd; }

    // Write velocity as bitcast<i32>(f32) so g2p can read the same buffer as array<Cell>.
    grid_int[base4 + 0u] = bitcast<i32>(vel.x);
    grid_int[base4 + 1u] = bitcast<i32>(vel.y);
}

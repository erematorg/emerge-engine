// Grid update — momentum normalization, gravity, force fields, boundary enforcement.
// Runs between P2G and G2P.
//
// GPU sparse grid Phase 2 (see mpm_technique_survey memory note): dispatch one workgroup per
// active-block SLOT, exactly like grid_clear.wgsl's already-shipped Phase 1 pattern (same
// active_block_ids/active_block_ids_prev grace-period lists, same halo-expanded compaction
// from particle_sort.wgsl, so this reuses infrastructure already proven correctness-safe for
// kernel-stencil spillover across block boundaries — no new halo logic needed here). Was the
// last remaining pass with unconditional O(grid_res²) dispatch cost: P2G/G2P/particles_update
// are already particle-parallel (cost scales with particle count, not grid size), and
// grid_clear already got this treatment in Phase 1 — this was the one gap. Per-cell logic is
// completely unchanged; only which cells get visited changes (active blocks' real cell range,
// not the whole dense grid), via the same block-relative grid-stride loop grid_clear uses.

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
    _pad2:              u32,
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

// ASFLIP (GPU port, Fei et al. 2021) — see GpuAsflipParams' own Rust doc.
struct AsflipParams {
    blend:   f32,
    enabled: u32,
    _pad0:   u32,
    _pad1:   u32,
}

const MASS_FLOOR:         f32 = 1e-4;
const MASS_ATOMIC_SCALE:  f32 = 1000000.0;
const MOM_ATOMIC_SCALE:   f32 = 100000.0;
const CELL_CENTER_OFFSET: f32 = 0.5;
const FIELD_GRAVITY_WELL: u32 = 1u;
const FIELD_COULOMB:      u32 = 2u;
override MAX_FORCE_FIELDS: u32;
const FF_NUM_FLOOR:       f32 = 1e-10;

// override, not a hardcoded literal — must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM
// exactly, single Rust-side source of truth (src/gpu/mod.rs step_params module). Same
// convention as grid_clear.wgsl.
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² — array sizes can't be override-derived
const BLOCK_THREADS_PER_DIM: u32 = 16u;

@group(0) @binding(1)  var<storage, read_write> grid_int:               array<i32>;
@group(0) @binding(3)  var<uniform>             step_params:             StepParams;
@group(0) @binding(4)  var<uniform>             force_fields:            ForceFieldsParams;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:        array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:      atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;
// Multi-field contact (GPU port, first slice) — raw-int view of grip_grid, same
// fixed-point atomic convention as `grid_int` above. Decoded (fixed-point → real f32,
// bitcast back) alongside the main grid's own decode below — REAL BUG FOUND AND FIXED
// 2026-07-14: without this, `grip_grid` never had ANY decode step at all (unlike
// `grid`, which grid_update.wgsl already handles), so a raw reader (e.g. a test doing
// a readback) would reinterpret the still-fixed-point integer bit pattern as a
// nonsensical near-zero float. Caught by `gpu_contact_grip_scatter_and_point_cloud_are_correct`
// measuring ~0 total grip mass instead of the real scattered total.
@group(1) @binding(12) var<storage, read_write> grip_grid_int:          array<i32>;
// ASFLIP (GPU port) — shares group 3 with resource regrowth, see pipeline.rs's module
// doc comment for why (WebGPU's 4-bind-group baseline is already fully used).
@group(3) @binding(28) var<uniform>             asflip_params:           AsflipParams;
@group(3) @binding(29) var<storage, read_write> asflip_snapshot:         array<vec2<f32>>;

// Smooth taper from 1 at switch_on to 0 at cutoff (cubic Hermite).
fn force_switch(dist: f32, cutoff: f32, switch_on: f32) -> f32 {
    if dist <= switch_on { return 1.0; }
    if dist >= cutoff    { return 0.0; }
    let t = (cutoff - dist) / (cutoff - switch_on);
    return t * t * (3.0 - 2.0 * t);
}

// Unchanged from the pre-Phase-2 version — one cell's worth of momentum normalization,
// gravity, force fields, boundary enforcement, and CFL clamp. Only the CALLER (which cells
// get visited) changed.
fn update_cell(cx: u32, cy: u32, res: u32) {
    // Decode fixed-point i32 → float mass. Write it back as bitcast so g2p reads it as f32.
    let base4 = (cy * res + cx) * 4u;
    let mass  = f32(grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
    grid_int[base4 + 2u] = bitcast<i32>(mass);

    // Multi-field contact (GPU port, first slice) — same fixed-point decode for the
    // grip field, but WITHOUT gravity/boundary/CFL (those apply to the resolved grip
    // velocity later, in a future resolve_contact pass, exactly matching CPU's own
    // `resolve_contact`: `v_grip = grip_momentum/grip_mass + gravity*dt` is computed
    // there, not baked into the raw scattered momentum here). This is just the decode
    // step CPU never needs (its `ContactCell` fields are already real f32, never
    // fixed-point) but GPU's atomic scatter requires.
    let grip_mass = f32(grip_grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
    let grip_mom_x = f32(grip_grid_int[base4 + 0u]) / MOM_ATOMIC_SCALE;
    let grip_mom_y = f32(grip_grid_int[base4 + 1u]) / MOM_ATOMIC_SCALE;
    grip_grid_int[base4 + 2u] = bitcast<i32>(grip_mass);
    grip_grid_int[base4 + 0u] = bitcast<i32>(grip_mom_x);
    grip_grid_int[base4 + 1u] = bitcast<i32>(grip_mom_y);

    // Empty cells: gravity for stray particles, but enforce boundary slip so floor/wall
    // cells don't feed downward velocity into the G2P gather and over-compress blobs.
    if mass < MASS_FLOOR {
        // ASFLIP: an empty/untouched cell has no real pre-force velocity -- write zero,
        // matching CPU's Grid::pre_force_velocity_at fallback for an untouched cell
        // exactly (Grid::snapshot_velocities never inserts untouched cells at all; GPU's
        // buffer is dense, so writing zero here is the dense equivalent of "absent").
        if asflip_params.enabled != 0u {
            asflip_snapshot[cy * res + cx] = vec2<f32>(0.0);
        }
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

    // ASFLIP: snapshot the pre-force velocity right after momentum normalization,
    // before gravity/boundary/CFL-clamp below modify it -- the exact same instant CPU's
    // Grid::snapshot_velocities captures (see solver/step.rs's normalize_velocities ->
    // snapshot -> apply_gravity ordering). Real gate: `enabled == 0` (default) means
    // this write never happens, zero cost for every scene that never attaches ASFLIP.
    if asflip_params.enabled != 0u {
        asflip_snapshot[cy * res + cx] = vel;
    }

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

// Dispatch: (2 * NUM_BLOCKS, 1, 1) workgroups, every frame, fixed — identical convention to
// grid_clear_main. workgroup_id.x is a SLOT: slots 0..NUM_BLOCKS index THIS substep's
// active_block_ids, slots NUM_BLOCKS..2*NUM_BLOCKS index active_block_ids_prev (last
// substep's list) — the same one-substep grace period grid_clear uses, needed here for the
// same reason: a block that just stopped being active still needs ITS cells' velocity
// written consistently with what grid_clear just zeroed them to (an empty cell must still
// get the "gravity for stray particles" treatment if some nearby active particle's G2P
// stencil might sample it), not left with whatever stale i32 bits happened to be there.
@compute @workgroup_size(BLOCK_THREADS_PER_DIM, BLOCK_THREADS_PER_DIM, 1)
fn grid_update_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    var block: u32;
    if wg_id.x < NUM_BLOCKS {
        if wg_id.x >= atomicLoad(&active_block_count) { return; }
        block = active_block_ids[wg_id.x];
    } else {
        let slot = wg_id.x - NUM_BLOCKS;
        if slot >= active_block_count_prev { return; }
        block = active_block_ids_prev[slot];
        // REAL BUG FOUND AND FIXED 2026-07-12: unlike grid_clear (whose write is always the
        // SAME constant zero, so two workgroups racing on it are harmless in practice), a
        // block that's active BOTH this substep and last substep appears in BOTH lists --
        // grid_clear tolerates the resulting double-dispatch, but grid_update computes each
        // cell's velocity via several read-modify-write steps, and two workgroups doing that
        // concurrently on the same non-atomic `grid_int` cells is a genuine data race, not a
        // harmless duplicate. Confirmed via real GPU test regressions introduced by this
        // Phase 2 change (gpu_rankine_stable, gpu_lp_realistic_combined_stress -- "J
        // collapsed" -- and gpu_sleep_wakes_on_nearby_activity), at BOTH a small (32) and
        // large (256) grid_res, ruling out a block-size-margin explanation and pointing
        // straight at concurrent double-processing. Fix: a block already present in the
        // CURRENT list is skipped here -- its own current-list workgroup already handles it
        // correctly, so only a block that's PURELY in the grace-period list (deactivated
        // this substep) needs this branch at all.
        let current_count = atomicLoad(&active_block_count);
        for (var i: u32 = 0u; i < current_count; i++) {
            if active_block_ids[i] == block { return; }
        }
    }
    let res = step_params.grid_res;

    let block_size = (res + NUM_BLOCKS_PER_DIM - 1u) / NUM_BLOCKS_PER_DIM; // ceil div
    let block_x = block % NUM_BLOCKS_PER_DIM;
    let block_y = block / NUM_BLOCKS_PER_DIM;
    let x_start = block_x * block_size;
    let y_start = block_y * block_size;
    let x_end = min(x_start + block_size, res);
    let y_end = min(y_start + block_size, res);

    // Grid-stride loop, same reasoning as grid_clear_main: block_size can exceed
    // BLOCK_THREADS_PER_DIM at high grid_res.
    var y = y_start + lid.y;
    loop {
        if y >= y_end { break; }
        var x = x_start + lid.x;
        loop {
            if x >= x_end { break; }
            update_cell(x, y, res);
            x += BLOCK_THREADS_PER_DIM;
        }
        y += BLOCK_THREADS_PER_DIM;
    }
}

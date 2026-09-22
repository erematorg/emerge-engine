// Grid update -- momentum normalization, gravity, force fields, boundary enforcement.
// Runs between P2G and G2P.
//
// GPU sparse grid Phase 2: dispatch one workgroup per active-block SLOT, same pattern as
// grid_clear.wgsl's Phase 1 (same active_block_ids/active_block_ids_prev grace-period lists,
// same halo-expanded compaction from particle_sort.wgsl, so kernel-stencil spillover across
// block boundaries is already handled). Per-cell logic is unchanged; only which cells get
// visited changes (active blocks' cell range, not the whole dense grid), via the same
// block-relative grid-stride loop grid_clear uses.

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    contact_friction:              f32,
    grid_cell_size:              f32,
    contact_active:              u32,
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

// ASFLIP (GPU port, Fei et al. 2021) -- see GpuAsflipParams' own Rust doc.
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

// override, not a hardcoded literal -- must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM
// exactly, single Rust-side source of truth (src/gpu/mod.rs step_params module). Same
// convention as grid_clear.wgsl.
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² -- array sizes can't be override-derived
// Threads per workgroup side -- set at pipeline creation to min(16, cells per block
// side): at grid_res=64 a block is 4x4 cells, and a 16x16 workgroup left 240 of its
// 256 threads idle on every one of up to 512 dispatched workgroups. The grid-stride
// loops below still cover blocks larger than this.
override BLOCK_THREADS_PER_DIM: u32 = 16u;

@group(0) @binding(1)  var<storage, read_write> grid_int:               array<i32>;
@group(0) @binding(3)  var<uniform>             step_params:             StepParams;

// This substep's timestep and velocity cap, decided on the GPU at the end of the previous
// substep -- see `adaptive_cfl.wgsl`. `substep_dt()`/`vel_limit` are the CPU's
// frame-start values and are NOT authoritative any more (the GPU may only tighten them).
@group(2) @binding(37) var<storage, read_write> adaptive_dt: array<atomic<u32>, 5>;

// Cached per invocation: this is an atomic storage load, and reading it at every use
// site cost ~40% of a substep (measured: 0.29 -> 0.41ms per substep on the dam break).
var<private> substep_dt_cache: f32 = -1.0;

fn substep_dt() -> f32 {
    if substep_dt_cache < 0.0 {
        substep_dt_cache = bitcast<f32>(atomicLoad(&adaptive_dt[0]));
    }
    return substep_dt_cache;
}

fn substep_vel_limit(dt: f32) -> f32 {
    return step_params.grid_cell_size / max(dt, 1.0e-12);
}

@group(0) @binding(4)  var<uniform>             force_fields:            ForceFieldsParams;
// Raw per-block particle histogram for THIS substep (written by particle_sort_count
// right before this substep's compact; nothing in between rewrites it).
@group(0) @binding(6)  var<storage, read_write> block_counts:            array<atomic<u32>, NUM_BLOCKS>;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:        array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:      atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;
// Multi-field contact (GPU port) -- raw-int view of grip_grid, same fixed-point atomic
// convention as `grid_int` above. Must be decoded (fixed-point → f32, bitcast back)
// alongside the main grid's own decode below, or a raw reader (e.g. a readback) sees
// the still-fixed-point integer bit pattern reinterpreted as a nonsensical near-zero float.
@group(1) @binding(12) var<storage, read_write> grip_grid_int:          array<i32>;
// ASFLIP (GPU port) -- shares group 3 with resource regrowth, see pipeline.rs's module
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

// Unchanged from the pre-Phase-2 version -- one cell's worth of momentum normalization,
// gravity, force fields, boundary enforcement, and CFL clamp. Only the CALLER (which cells
// get visited) changed.
fn update_cell(cx: u32, cy: u32, res: u32) {
    // Main-grid mass/momentum are summed as real f32 bit patterns by p2g's
    // `atomic_add_f32_grid` (exact, no fixed-point quantum -- see that
    // function's own doc), so reading them is a plain bitcast.
    let base4 = (cy * res + cx) * 4u;
    let mass  = bitcast<f32>(grid_int[base4 + 2u]);

    // Multi-field contact (GPU port, first slice) -- same fixed-point decode for the
    // grip field, but WITHOUT gravity/boundary/CFL (those apply to the resolved grip
    // velocity later, in a future resolve_contact pass, exactly matching CPU's own
    // `resolve_contact`: `v_grip = grip_momentum/grip_mass + gravity*dt` is computed
    // there, not baked into the raw scattered momentum here). This is just the decode
    // step CPU never needs (its `ContactCell` fields are already real f32, never
    // fixed-point) but GPU's atomic scatter requires.
    // Only while contact is active -- see grid_clear.wgsl's matching gate.
    if step_params.contact_active != 0u {
        let grip_mass = f32(grip_grid_int[base4 + 2u]) / MASS_ATOMIC_SCALE;
        let grip_mom_x = f32(grip_grid_int[base4 + 0u]) / MOM_ATOMIC_SCALE;
        let grip_mom_y = f32(grip_grid_int[base4 + 1u]) / MOM_ATOMIC_SCALE;
        grip_grid_int[base4 + 2u] = bitcast<i32>(grip_mass);
        grip_grid_int[base4 + 0u] = bitcast<i32>(grip_mom_x);
        grip_grid_int[base4 + 1u] = bitcast<i32>(grip_mom_y);
    }

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
        var grav_vel = step_params.gravity * substep_dt();
        let bt2 = step_params.boundary_thickness;
        if cx < bt2          && grav_vel.x < 0.0 { grav_vel.x = 0.0; }
        if cx >= res - bt2   && grav_vel.x > 0.0 { grav_vel.x = 0.0; }
        if cy < bt2          && grav_vel.y < 0.0 { grav_vel.y = 0.0; }
        if cy >= res - bt2   && grav_vel.y > 0.0 { grav_vel.y = 0.0; }
        grid_int[base4 + 0u] = bitcast<i32>(grav_vel.x);
        grid_int[base4 + 1u] = bitcast<i32>(grav_vel.y);
        return;
    }

    let mom_x = bitcast<f32>(grid_int[base4 + 0u]);
    let mom_y = bitcast<f32>(grid_int[base4 + 1u]);
    var vel   = vec2<f32>(mom_x, mom_y) / mass;

    // ASFLIP: snapshot the pre-force velocity right after momentum normalization,
    // before gravity/boundary/CFL-clamp below modify it -- the exact same instant CPU's
    // Grid::snapshot_velocities captures (see solver/step.rs's normalize_velocities ->
    // snapshot -> apply_gravity ordering). Real gate: `enabled == 0` (default) means
    // this write never happens, zero cost for every scene that never attaches ASFLIP.
    if asflip_params.enabled != 0u {
        asflip_snapshot[cy * res + cx] = vel;
    }

    vel += step_params.gravity * substep_dt();

    // Apply cursor force fields in grid space (same substep as position advance -- no lag).
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
                        vel += acc * substep_dt();
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
                        vel += acc * substep_dt();
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

    // CFL clamp before G2P -- bounds both particle velocity AND affine matrix C at the source.
    let spd = length(vel);
    if spd > substep_vel_limit(substep_dt()) { vel *= substep_vel_limit(substep_dt()) / spd; }

    // Write velocity as bitcast<i32>(f32) so g2p can read the same buffer as array<Cell>.
    grid_int[base4 + 0u] = bitcast<i32>(vel.x);
    grid_int[base4 + 1u] = bitcast<i32>(vel.y);
}

// Same occupancy rule as particle_sort_compact_main, which built this substep's
// active_block_ids from these exact counts.
fn in_current_active_list(block: u32) -> bool {
    let bx = i32(block % NUM_BLOCKS_PER_DIM);
    let by = i32(block / NUM_BLOCKS_PER_DIM);
    for (var dy: i32 = -1; dy <= 1; dy++) {
        let ny = by + dy;
        if ny < 0 || ny >= i32(NUM_BLOCKS_PER_DIM) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nx = bx + dx;
            if nx < 0 || nx >= i32(NUM_BLOCKS_PER_DIM) { continue; }
            if atomicLoad(&block_counts[u32(ny) * NUM_BLOCKS_PER_DIM + u32(nx)]) > 0u {
                return true;
            }
        }
    }
    return false;
}

// Dispatch: (2 * NUM_BLOCKS, 1, 1) workgroups, every frame, fixed -- identical convention to
// grid_clear_main. workgroup_id.x is a SLOT: slots 0..NUM_BLOCKS index THIS substep's
// active_block_ids, slots NUM_BLOCKS..2*NUM_BLOCKS index active_block_ids_prev (last
// substep's list) -- the same one-substep grace period grid_clear uses, needed here for the
// same reason: a block that just stopped being active still needs ITS cells' velocity
// written consistently with what grid_clear just zeroed them to (an empty cell must still
// get the "gravity for stray particles" treatment if some nearby active particle's G2P
// stencil might sample it), not left with whatever stale i32 bits happened to be there.
@compute @workgroup_size(BLOCK_THREADS_PER_DIM, BLOCK_THREADS_PER_DIM, 1)
fn grid_update_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // The frame's time is already fully advanced -- this encoded substep is spare
    // capacity the CPU could not size exactly in advance (see adaptive_cfl.wgsl).
    if substep_dt() <= 0.0 { return; }
    var block: u32;
    if wg_id.x < NUM_BLOCKS {
        if wg_id.x >= atomicLoad(&active_block_count) { return; }
        block = active_block_ids[wg_id.x];
    } else {
        let slot = wg_id.x - NUM_BLOCKS;
        if slot >= active_block_count_prev { return; }
        block = active_block_ids_prev[slot];
        // Unlike grid_clear (whose write is always the same constant zero, so two
        // workgroups racing on it are harmless), grid_update computes each cell's velocity
        // via several read-modify-write steps -- two workgroups doing that concurrently on
        // the same non-atomic `grid_int` cells is a genuine data race. A block active BOTH
        // this substep and last substep appears in BOTH lists, so skip it here if it's
        // already in the CURRENT list -- its own current-list workgroup already handles it;
        // only a block PURELY in the grace-period list (deactivated this substep) needs
        // this branch.
        //
        // Membership in the current list is decided by particle_sort_compact_main's own
        // rule (the block or any of its 8 neighbours holds particles), re-evaluated from
        // the same histogram: 9 loads, where the previous linear search over the whole
        // current list cost O(active blocks) serial loads per grace workgroup --
        // measured at ~330us per substep on the 10k-particle vortex (~156 active blocks).
        if in_current_active_list(block) { return; }
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

// Grid clear — zero cells before each P2G pass.
//
// GPU sparse grid Phase 1 (see mpm_technique_survey memory note): dispatch one workgroup per
// POTENTIAL active-block slot (NUM_BLOCKS, fixed worst-case size — no indirect dispatch yet,
// that's Phase 3), with an early-return guard for slots beyond how many blocks are actually
// active this frame. The few workgroups that proceed clear only their own block's real cell
// range, not the whole grid_res² grid — this is where the actual win comes from. P2G,
// grid_update, and G2P are untouched by this phase; they still index the (still dense) grid
// buffer exactly as before. Only which cells get zeroed changes, never the value written once
// a cell is touched — same physics, less wasted work.
//
// Must run before P2G every substep so the atomic scatter starts from zero.
//
// Reference: Hu et al. 2018 §4, Algorithm 1 line 2 ("for each grid node: reset mass/momentum").

struct Cell {
    momentum: vec2<f32>,
    mass:     f32,
    _pad:     f32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:          f32,
    gravity:            vec2<f32>, // angled gravity — offset 16, 8-byte aligned ✓
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32, // 48 bytes — 16-byte aligned for uniform binding ✓
}

// override, not a hardcoded literal — must match particle_sort.wgsl's NUM_BLOCKS_PER_DIM
// exactly, single Rust-side source of truth (src/gpu/mod.rs step_params module).
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² — array sizes can't be override-derived
// Thread-grid covering one block, per workgroup — see the grid-stride loop below for why a
// fixed-size workgroup still correctly covers a block whose real cell range is larger.
const BLOCK_THREADS_PER_DIM: u32 = 16u;

@group(0) @binding(1)  var<storage, read_write> grid:                    array<Cell>;
@group(0) @binding(3)  var<uniform>             step_params:             StepParams;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:        array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:      atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;

// Dispatch: (2 * NUM_BLOCKS, 1, 1) workgroups, every frame, fixed — worst case (every block
// active, in both lists) never overflows. workgroup_id.x is a SLOT, not a block ID. Slots
// 0..NUM_BLOCKS index THIS substep's active_block_ids; slots NUM_BLOCKS..2*NUM_BLOCKS index
// active_block_ids_prev (LAST substep's list, offset by NUM_BLOCKS) — the one-substep grace
// period that guarantees a block which just stopped being active still gets cleared one more
// time. See active_block_swap_main in particle_sort.wgsl for why this exists. Most slots
// beyond their list's real count do nothing at all.
@compute @workgroup_size(BLOCK_THREADS_PER_DIM, BLOCK_THREADS_PER_DIM, 1)
fn grid_clear_main(
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
    }
    let res = step_params.grid_res;

    let block_size = (res + NUM_BLOCKS_PER_DIM - 1u) / NUM_BLOCKS_PER_DIM; // ceil div
    let block_x = block % NUM_BLOCKS_PER_DIM;
    let block_y = block / NUM_BLOCKS_PER_DIM;
    let x_start = block_x * block_size;
    let y_start = block_y * block_size;
    // Last block per row/column covers a smaller residual range when grid_res doesn't divide
    // evenly by NUM_BLOCKS_PER_DIM — clamp so no cell index ever reaches grid_res.
    let x_end = min(x_start + block_size, res);
    let y_end = min(y_start + block_size, res);

    // Grid-stride loop: block_size can exceed BLOCK_THREADS_PER_DIM at high grid_res (e.g.
    // grid_res=2048, NUM_BLOCKS_PER_DIM=16 → block_size=128 > 16 threads/dim), so each thread
    // clears every BLOCK_THREADS_PER_DIM'th cell in its block instead of assuming one cell
    // per thread covers the whole range.
    var y = y_start + lid.y;
    loop {
        if y >= y_end { break; }
        var x = x_start + lid.x;
        loop {
            if x >= x_end { break; }
            let idx = y * res + x;
            grid[idx].momentum = vec2<f32>(0.0, 0.0);
            grid[idx].mass     = 0.0;
            grid[idx]._pad     = 0.0;
            x += BLOCK_THREADS_PER_DIM;
        }
        y += BLOCK_THREADS_PER_DIM;
    }
}

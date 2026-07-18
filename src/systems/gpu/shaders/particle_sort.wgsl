// particle_sort — block-level counting sort of particles by spatial position.
//
// Produces sorted_particle_ids[gid.x] = original particle index, ordered so consecutive
// threads visit spatially-nearby particles. p2g and particles_update read
// particles[sorted_particle_ids[gid.x]] instead of particles[gid.x] directly — this improves
// grid memory access coherence during P2G scatter (nearby particles touch nearby grid cells,
// reducing cache-miss/atomic-contention diversity for concurrently-running threads).
//
// Reference: Gao, Wang, Wu, Pradhana, Sifakis, Yuksel, Jiang — "GPU Optimization of Material
// Point Methods" (SIGGRAPH Asia 2018) — histogram sort keyed on block index for coalesced
// P2G access. This is a simplified single-workgroup-scan variant (NUM_BLOCKS=256, bounded so
// the scan fits in one workgroup's shared memory — no cross-workgroup reduction needed).
//
// Five passes, run once per FRAME (not per substep) in sequence: clear -> count -> compact ->
// scan -> scatter. Each must complete before the next starts — wgpu inserts the necessary
// barriers automatically between compute dispatches that read/write the same buffer.
//
// "compact" (GPU sparse grid Phase 1 — see mpm_technique_survey memory note) reads the RAW
// per-block histogram built by count, before scan overwrites block_counts into an exclusive-
// prefix-sum scatter cursor. block_counts[b] > 0 right after count means block b has at least
// one particle; compact records which blocks those are into active_block_ids, consumed by
// grid_clear.wgsl to bound its real work to occupied blocks instead of the whole dense grid.

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
    _pad:                 u32,
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
    _pad2:              u32,
}

// override, not a hardcoded literal — single Rust-side source of truth in
// src/gpu/mod.rs's step_params module (NUM_BLOCKS_PER_DIM/NUM_BLOCKS), injected at pipeline
// creation. grid_clear.wgsl reads the same override — two independent hardcoded copies of
// this constant would be a silent-drift risk neither file's history has any reason to allow.
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u; // NUM_BLOCKS_PER_DIM² — array sizes can't be override-derived

@group(0) @binding(0) var<storage, read_write> particles:           array<Particle>;
@group(0) @binding(3) var<uniform>             step_params:         StepParams;
@group(0) @binding(5) var<storage, read_write> sorted_particle_ids: array<u32>;
@group(0) @binding(6)  var<storage, read_write> block_counts:           array<atomic<u32>, NUM_BLOCKS>;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:       array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:     atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;
// Multi-field contact (GPU port, first slice) — per-BLOCK point-cloud size, cleared
// here (not grid_clear.wgsl, which iterates per-CELL) since this is already a
// 256-wide, one-thread-per-block dispatch, the same granularity this data needs. See
// buffers.rs's `contact_point_counts` doc for the full rationale.
@group(1) @binding(14) var<storage, read_write> contact_point_counts:    array<atomic<u32>, NUM_BLOCKS>;

// Maps a particle's grid-cell position to one of NUM_BLOCKS coarse spatial buckets.
// Block size scales with grid_res so the same 256-bucket scan works at any resolution.
fn block_index(pos: vec2<f32>, grid_res: u32) -> u32 {
    let max_cell = grid_res - 1u;
    let cell_x = u32(clamp(pos.x, 0.0, f32(max_cell)));
    let cell_y = u32(clamp(pos.y, 0.0, f32(max_cell)));
    let block_size = (grid_res + NUM_BLOCKS_PER_DIM - 1u) / NUM_BLOCKS_PER_DIM; // ceil div
    let block_x = min(cell_x / block_size, NUM_BLOCKS_PER_DIM - 1u);
    let block_y = min(cell_y / block_size, NUM_BLOCKS_PER_DIM - 1u);
    return block_y * NUM_BLOCKS_PER_DIM + block_x;
}

// ── Pass 0: swap (GPU sparse grid Phase 1 — one-substep grace period) ────────
// One workgroup, NUM_BLOCKS threads — dispatched FIRST, before clear/count/compact. Copies
// THIS substep's about-to-be-stale active list into active_block_ids_prev/count_prev (the
// snapshot grid_clear will also clear, in addition to whatever's freshly compacted below),
// then resets active_block_count to 0 so compact starts from a clean slate.
//
// Real bug found via a long-running headless diagnostic (basic_sand_gpu blew up after ~1500
// frames, ~1-in-5 runs): a block that stops being active (a particle moves away) was never
// cleared again, since grid_clear only ever clears CURRENTLY active blocks — its last P2G
// contribution sat there permanently until some particle wandered back near it much later,
// at which point P2G's atomic ADD compounded onto the stale residual. Dense grid_clear never
// had this problem (it unconditionally zeroed every cell every substep regardless of
// activity). A first attempt at fixing this (accumulate-without-resetting, deduplicated via a
// presence bitmask, reset right after grid_clear in the SAME substep) was wrong: resetting in
// the same substep it was used in means the "previous" state is already gone by the time the
// NEXT substep's compact runs — zero actual grace period, just extra work. This is why: a
// genuine 1-substep grace period needs the LAST substep's active set to still exist when the
// NEXT substep's grid_clear runs, which requires two independent buffers (this substep's,
// and last substep's), not one buffer with a same-substep reset.
@compute @workgroup_size(256, 1, 1)
fn active_block_swap_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    active_block_ids_prev[lid.x] = active_block_ids[lid.x];
    if lid.x == 0u {
        active_block_count_prev = atomicLoad(&active_block_count);
        atomicStore(&active_block_count, 0u);
    }
}

// ── Pass 1: clear ─────────────────────────────────────────────────────────────
// One workgroup, NUM_BLOCKS threads — zeroes the per-block occupancy histogram before
// counting. active_block_ids/count themselves were already handled by the swap pass above.
@compute @workgroup_size(256, 1, 1)
fn particle_sort_clear_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    atomicStore(&block_counts[lid.x], 0u);
    // Multi-field contact (GPU port, first slice) — see contact_point_counts binding doc.
    atomicStore(&contact_point_counts[lid.x], 0u);
}

// ── Pass 2: count ─────────────────────────────────────────────────────────────
// One thread per particle — builds the per-block histogram.
@compute @workgroup_size(64, 1, 1)
fn particle_sort_count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    let b = block_index(particles[i].x, step_params.grid_res);
    atomicAdd(&block_counts[b], 1u);
}

// ── Pass 3: compact (GPU sparse grid Phase 1) ─────────────────────────────────
// One workgroup, NUM_BLOCKS threads — must run after count and before scan, since it reads
// the RAW histogram value (block_counts[b] > 0 means "block b has particles"). Scan (next
// pass) overwrites block_counts into an exclusive-prefix-sum scatter cursor — after that
// point the buffer no longer represents occupancy at all, so this can't be deferred or
// reordered relative to count/scan without breaking the only correct window to read it.
//
// A block is marked active if it OR ANY of its 8 neighbors has particles — not just itself.
// Real bug found via direct testing (gpu_sleep_freezes_settled_particles never settling):
// the quadratic B-spline P2G kernel scatters into a 3-cell-wide neighborhood around each
// particle, which routinely spills across a block boundary into an adjacent block whenever
// block_size (cells per block) is smaller than the kernel's reach — e.g. grid_res=32 with
// NUM_BLOCKS_PER_DIM=16 gives block_size=2, smaller than the 3-cell stencil, so nearly every
// particle's scatter crosses into a neighbor. If that neighbor isn't marked active,
// grid_clear never clears it while P2G keeps atomically scattering into it anyway —
// uncleared values accumulate substep over substep instead of resetting. Mirrors the kernel's
// own 3×3 reach at block granularity instead of cell granularity — correct at any
// grid_res/NUM_BLOCKS_PER_DIM ratio, not just ones where block_size happens to exceed 3.
//
// Starts from a clean slate every substep (active_block_swap_main already reset
// active_block_count to 0 and preserved the old list in active_block_ids_prev) — no
// deduplication needed here, this pass only ever describes THIS substep's occupancy.
// grid_clear separately processes active_block_ids_prev too, covering the one-substep grace
// period — see active_block_swap_main's doc comment for the full reasoning.
@compute @workgroup_size(256, 1, 1)
fn particle_sort_compact_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let b = lid.x;
    let bx = i32(b % NUM_BLOCKS_PER_DIM);
    let by = i32(b / NUM_BLOCKS_PER_DIM);
    var occupied = false;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        let ny = by + dy;
        if ny < 0 || ny >= i32(NUM_BLOCKS_PER_DIM) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nx = bx + dx;
            if nx < 0 || nx >= i32(NUM_BLOCKS_PER_DIM) { continue; }
            let neighbor = u32(ny) * NUM_BLOCKS_PER_DIM + u32(nx);
            if atomicLoad(&block_counts[neighbor]) > 0u { occupied = true; }
        }
    }
    if !occupied { return; }
    let slot = atomicAdd(&active_block_count, 1u);
    active_block_ids[slot] = b;
}

// ── Pass 3: scan ──────────────────────────────────────────────────────────────
// One workgroup, NUM_BLOCKS threads — Hillis-Steele inclusive scan in shared memory, then
// converts to exclusive prefix sum (the scatter starting offset for each block) and writes it
// back into block_counts, which doubles as the atomic scatter cursor in pass 4.
var<workgroup> scan_temp: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn particle_sort_scan_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let orig = atomicLoad(&block_counts[i]);
    scan_temp[i] = orig;
    workgroupBarrier();

    var offset: u32 = 1u;
    loop {
        if offset >= NUM_BLOCKS { break; }
        var val: u32 = 0u;
        if i >= offset {
            val = scan_temp[i - offset];
        }
        workgroupBarrier();
        scan_temp[i] = scan_temp[i] + val;
        workgroupBarrier();
        offset = offset << 1u;
    }

    // scan_temp[i] now holds the INCLUSIVE scan; exclusive = inclusive - this block's own count.
    let exclusive = scan_temp[i] - orig;
    atomicStore(&block_counts[i], exclusive);
}

// ── Pass 4: scatter ───────────────────────────────────────────────────────────
// One thread per particle — claims a unique slot within its block (atomicAdd on the now-
// exclusive-prefix-sum cursor) and writes its own original index there.
@compute @workgroup_size(64, 1, 1)
fn particle_sort_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    let b = block_index(particles[i].x, step_params.grid_res);
    let slot = atomicAdd(&block_counts[b], 1u);
    sorted_particle_ids[slot] = i;
}

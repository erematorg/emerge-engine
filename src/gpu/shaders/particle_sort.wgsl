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
// Four passes, run once per FRAME (not per substep) in sequence: clear -> count -> scan ->
// scatter. Each must complete before the next starts — wgpu inserts the necessary barriers
// automatically between compute dispatches that read/write the same buffer.

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
}

const NUM_BLOCKS_PER_DIM: u32 = 16u;
const NUM_BLOCKS:         u32 = 256u; // NUM_BLOCKS_PER_DIM²

@group(0) @binding(0) var<storage, read_write> particles:           array<Particle>;
@group(0) @binding(3) var<uniform>             step_params:         StepParams;
@group(0) @binding(5) var<storage, read_write> sorted_particle_ids: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_counts:        array<atomic<u32>, NUM_BLOCKS>;

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

// ── Pass 1: clear ─────────────────────────────────────────────────────────────
// One workgroup, NUM_BLOCKS threads — zeroes the histogram before counting.
@compute @workgroup_size(256, 1, 1)
fn particle_sort_clear_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    atomicStore(&block_counts[lid.x], 0u);
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

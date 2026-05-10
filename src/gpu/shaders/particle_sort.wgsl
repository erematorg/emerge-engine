// particle_sort — initialize sorted_particle_ids to identity permutation.
//
// One thread per particle. Writes sorted_particle_ids[i] = i.
//
// Architecture placeholder: the CPU sort in step_frame() already provides spatial
// ordering each frame by sorting particles_cpu by grid-cell key before upload.
// The identity permutation seeds that ordering for p2g and particles_update, which
// read particles[sorted_particle_ids[gid.x]] instead of particles[gid.x] directly.
//
// Future: replace this pass with a GPU counting sort or bitonic sort for
// per-substep reordering without a CPU roundtrip.

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
}

@group(0) @binding(3) var<uniform>             step_params:         StepParams;
@group(0) @binding(5) var<storage, read_write> sorted_particle_ids: array<u32>;

// Workgroup size MUST match WG_PARTICLES (= 64) in src/gpu/mod.rs.
@compute @workgroup_size(64, 1, 1)
fn particle_sort_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= step_params.particle_count { return; }
    sorted_particle_ids[i] = i;
}

// Grid update — momentum normalization, gravity, boundary enforcement.
// Runs between P2G and G2P. One thread per grid cell.
//
// Steps per cell:
//   1. Normalize: velocity = momentum / mass  (skip if mass ≈ 0)
//   2. Apply gravity: velocity.y += gravity * dt
//   3. Boundary: zero normal velocity within boundary_thickness cells of each wall
//
// Reference: Hu et al. 2018 §4, Algorithm 1 lines 6–9.
//
// TODO: implement

struct Cell {
    momentum: vec2<f32>,
    mass:     f32,
    _pad:     f32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    d_inverse:          f32,
    gravity:            f32,
    boundary_thickness: u32,
    _pad:               vec2<f32>,
}

@group(0) @binding(0) var<storage, read_write> grid:        array<Cell>;
@group(0) @binding(1) var<uniform>             step_params: StepParams;

// TODO: @compute @workgroup_size(8, 8, 1)
// fn grid_update_main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }

// Grid clear — zero all cells before each P2G pass.
// One thread per cell. Dispatch: (ceil(grid_res/8), ceil(grid_res/8), 1).
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
    vel_limit:          f32,       // 32 bytes — 16-byte aligned for uniform binding ✓
}

@group(0) @binding(1) var<storage, read_write> grid:        array<Cell>;
@group(0) @binding(3) var<uniform>             step_params: StepParams;

// Workgroup size MUST match WG_GRID (= 8) in src/gpu/mod.rs.
@compute @workgroup_size(8, 8, 1)
fn grid_clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let res = step_params.grid_res;
    if x >= res || y >= res { return; }
    let idx = y * res + x;
    grid[idx].momentum = vec2<f32>(0.0, 0.0);
    grid[idx].mass     = 0.0;
    grid[idx]._pad     = 0.0;
}

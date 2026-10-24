// GPU port of the Chorin-style incompressibility projection
// (`src/spacetime/grid/pressure.rs::project_fluid_incompressibility`, Bridson
// "Fluid Simulation for Computer Graphics" ch. 5): no acoustic CFL term (no
// Tait stiffness, `eos_stiffness = 0`) for a strict single-material fluid. The
// CPU solves the Poisson equation exactly by DCT (`dct.rs`); there is no GPU
// DCT, so this uses Jacobi iteration (Harris, "Fast Fluid Dynamics Simulation
// on the GPU," GPU Gems 2004, from Stam 1999 "Stable Fluids"). The GPU grid is
// dense (`array<Cell>`), so every pass walks the whole `grid_res x grid_res`
// domain with no padded bounding box.
//
// Rust decides the Jacobi sweep count; free-surface classification is
// recomputed every call, as on the CPU (`pressure.rs`'s `is_surface`).
//
// 3 passes, mirroring the CPU algorithm step for step:
//   1. fluid_pressure_setup_main    -- divergence RHS + free-surface classification,
//                                      pressure_a initialized to 0 (cold start,
//                                      the CPU refinement's seed when no better
//                                      guess exists).
//   2. fluid_pressure_jacobi_a_to_b_main / fluid_pressure_jacobi_b_to_a_main
//                                    -- one Jacobi sweep each, alternating
//                                       source and destination buffer (two
//                                       entry points instead of a runtime
//                                       ping-pong flag; Rust alternates the
//                                       pipeline). An even sweep count leaves
//                                       the answer in pressure_a.
//   3. fluid_pressure_correct_main   -- per-cell-mass momentum correction,
//                                       `a = -grad(p)/mass`, as the CPU's
//                                       final loop.

struct Cell {
    momentum: vec2<f32>,
    mass:     f32,
    _pad:     f32,
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

// Per-cell mass below which a cell is empty or a placeholder, the value and
// purpose of the CPU's `MIN_ABSOLUTE_MASS_FOR_CORRECTION`.
const MIN_ABSOLUTE_MASS: f32 = 1.0e-3;
// Relative free-surface threshold, as in the CPU's `pressure.rs`: a fraction
// of a representative fluid cell mass, not a tiny constant (which flickers
// at a splashing interface). The CPU's `mass_avg` would need a reduction
// pass here, so the reference is the material's rest cell mass, which Rust
// knows exactly and uploads once in `FluidPressureParams`
// (`REST_CELL_MASS_FRACTION` of it).
const REST_CELL_MASS_FRACTION: f32 = 0.3;

struct FluidPressureParams {
    // Reference fluid cell mass (`rest_density * spacing^2` in this material's
    // grid units), the value the material and spawn code use. Only for the
    // free-surface threshold, the CPU's `mass_avg * 0.3`.
    reference_cell_mass: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(1) var<storage, read_write> grid_int: array<i32>;
@group(0) @binding(3) var<uniform> step_params: StepParams;

@group(3) @binding(32) var<uniform> fluid_pressure_params: FluidPressureParams;
@group(3) @binding(33) var<storage, read_write> fp_divergence: array<f32>;
@group(3) @binding(34) var<storage, read_write> fp_pressure_a: array<f32>;
@group(3) @binding(35) var<storage, read_write> fp_pressure_b: array<f32>;
// 0u = interior/wall (Neumann), 1u = free surface (Dirichlet p=0). u32, not
// bool -- WGSL storage buffers don't support bool arrays.
@group(3) @binding(36) var<storage, read_write> fp_is_surface: array<u32>;

const MASS_ATOMIC_SCALE: f32 = 1000000.0;
const MOM_ATOMIC_SCALE:  f32 = 100000.0;

// Velocity at a cell, after `grid_update.wgsl`'s normalization: `grid_int`
// already holds bitcast<f32> velocity, not momentum, since `grid_update` runs
// first every substep. Out of bounds reads zero, like CPU `Grid::velocity_at`.
fn velocity_at(cx: i32, cy: i32, res: i32) -> vec2<f32> {
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return vec2<f32>(0.0);
    }
    let base4 = (u32(cy) * u32(res) + u32(cx)) * 4u;
    return vec2<f32>(
        bitcast<f32>(grid_int[base4 + 0u]),
        bitcast<f32>(grid_int[base4 + 1u]),
    );
}

fn mass_at(cx: i32, cy: i32, res: i32) -> f32 {
    if cx < 0 || cy < 0 || cx >= res || cy >= res {
        return 0.0;
    }
    let base4 = (u32(cy) * u32(res) + u32(cx)) * 4u;
    return bitcast<f32>(grid_int[base4 + 2u]);
}

@compute @workgroup_size(16, 16, 1)
fn fluid_pressure_setup_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = i32(step_params.grid_res);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= res || cy >= res {
        return;
    }
    let idx = u32(cy) * u32(res) + u32(cx);

    // Central-difference divergence, the shape of `pressure.rs`'s loop.
    let v_r = velocity_at(cx + 1, cy, res).x;
    let v_l = velocity_at(cx - 1, cy, res).x;
    let v_u = velocity_at(cx, cy + 1, res).y;
    let v_d = velocity_at(cx, cy - 1, res).y;
    fp_divergence[idx] = (v_r - v_l) * 0.5 + (v_u - v_d) * 0.5;

    // Free-surface classification as in `pressure.rs`: a fluid cell (mass above
    // threshold) next to a low-mass neighbour is free surface (Dirichlet p=0)
    // unless that neighbour is the domain wall (Neumann, `SlipBoundary`'s zero
    // flux): a low-mass wall cell means no particle has reached it, not air.
    let threshold = fluid_pressure_params.reference_cell_mass * REST_CELL_MASS_FRACTION;
    let own_mass = mass_at(cx, cy, res);
    var is_surface = 0u;
    if own_mass > threshold {
        let bt = i32(step_params.boundary_thickness);
        if (cx - 1 >= 0 && mass_at(cx - 1, cy, res) <= threshold && cx - 1 >= bt)
            || (cx + 1 < res && mass_at(cx + 1, cy, res) <= threshold && cx + 1 < res - bt)
            || (cy - 1 >= 0 && mass_at(cx, cy - 1, res) <= threshold && cy - 1 >= bt)
            || (cy + 1 < res && mass_at(cx, cy + 1, res) <= threshold && cy + 1 < res - bt) {
            is_surface = 1u;
        }
    }
    fp_is_surface[idx] = is_surface;
    fp_pressure_a[idx] = 0.0;
    fp_pressure_b[idx] = 0.0;
}

// Two real Jacobi sweep entry points below solve `Laplacian(p) = rhs / alpha`,
// `alpha = 1/mass` (the same constant-coefficient-per-cell form CPU's own
// Gauss-Seidel refinement solves, just Jacobi instead of Gauss-Seidel --
// Jacobi is the standard choice for a data-parallel GPU sweep, since
// every cell's new value depends only on the PREVIOUS sweep's neighbors,
// unlike Gauss-Seidel's in-place update, which would be a data race
// across parallel threads). Neumann at a true wall/padding edge (excluded
// from both the sum and the divisor, matching CPU's own convention exactly);
// Dirichlet (fixed 0, never updated) at a free-surface cell. WGSL has
// no first-class buffer-pointer parameters that could share one function
// body across two different storage bindings, so the two directions are
// two separate, otherwise-identical entry points instead of one parametrized
// function -- Rust alternates which one it dispatches, sweep to sweep.

@compute @workgroup_size(16, 16, 1)
fn fluid_pressure_jacobi_a_to_b_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = i32(step_params.grid_res);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= res || cy >= res {
        return;
    }
    let idx = u32(cy) * u32(res) + u32(cx);
    if fp_is_surface[idx] != 0u {
        fp_pressure_b[idx] = 0.0;
        return;
    }
    let own_mass = mass_at(cx, cy, res);
    if own_mass <= MIN_ABSOLUTE_MASS {
        fp_pressure_b[idx] = fp_pressure_a[idx];
        return;
    }
    let alpha = 1.0 / own_mass;
    var sum = 0.0;
    var count = 0.0;
    if cx - 1 >= 0 { sum += fp_pressure_a[idx - 1u]; count += 1.0; }
    if cx + 1 < res { sum += fp_pressure_a[idx + 1u]; count += 1.0; }
    if cy - 1 >= 0 { sum += fp_pressure_a[idx - u32(res)]; count += 1.0; }
    if cy + 1 < res { sum += fp_pressure_a[idx + u32(res)]; count += 1.0; }
    if count > 0.0 {
        fp_pressure_b[idx] = (sum - fp_divergence[idx] / alpha) / count;
    } else {
        fp_pressure_b[idx] = fp_pressure_a[idx];
    }
}

@compute @workgroup_size(16, 16, 1)
fn fluid_pressure_jacobi_b_to_a_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = i32(step_params.grid_res);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= res || cy >= res {
        return;
    }
    let idx = u32(cy) * u32(res) + u32(cx);
    if fp_is_surface[idx] != 0u {
        fp_pressure_a[idx] = 0.0;
        return;
    }
    let own_mass = mass_at(cx, cy, res);
    if own_mass <= MIN_ABSOLUTE_MASS {
        fp_pressure_a[idx] = fp_pressure_b[idx];
        return;
    }
    let alpha = 1.0 / own_mass;
    var sum = 0.0;
    var count = 0.0;
    if cx - 1 >= 0 { sum += fp_pressure_b[idx - 1u]; count += 1.0; }
    if cx + 1 < res { sum += fp_pressure_b[idx + 1u]; count += 1.0; }
    if cy - 1 >= 0 { sum += fp_pressure_b[idx - u32(res)]; count += 1.0; }
    if cy + 1 < res { sum += fp_pressure_b[idx + u32(res)]; count += 1.0; }
    if count > 0.0 {
        fp_pressure_a[idx] = (sum - fp_divergence[idx] / alpha) / count;
    } else {
        fp_pressure_a[idx] = fp_pressure_b[idx];
    }
}

// Per-cell-mass momentum correction, `a = -grad(p)/mass`, as the CPU's final
// loop in `pressure.rs`. Reads `fp_pressure_a`: the Rust dispatch loop must
// use an even sweep count so the answer ends there.
@compute @workgroup_size(16, 16, 1)
fn fluid_pressure_correct_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = i32(step_params.grid_res);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    if cx >= res || cy >= res {
        return;
    }
    let own_mass = mass_at(cx, cy, res);
    if own_mass <= MIN_ABSOLUTE_MASS {
        return;
    }
    let p_r = select(0.0, fp_pressure_a[u32(cy) * u32(res) + u32(cx + 1)], cx + 1 < res);
    let p_l = select(0.0, fp_pressure_a[u32(cy) * u32(res) + u32(cx - 1)], cx - 1 >= 0);
    let p_u = select(0.0, fp_pressure_a[u32(cy + 1) * u32(res) + u32(cx)], cy + 1 < res);
    let p_d = select(0.0, fp_pressure_a[u32(cy - 1) * u32(res) + u32(cx)], cy - 1 >= 0);
    let grad_p = vec2<f32>((p_r - p_l) * 0.5, (p_u - p_d) * 0.5);

    let base4 = (u32(cy) * u32(res) + u32(cx)) * 4u;
    var vel = vec2<f32>(
        bitcast<f32>(grid_int[base4 + 0u]),
        bitcast<f32>(grid_int[base4 + 1u]),
    );
    // Same real under-relaxation CPU uses even with an exact solve -- see
    // `pressure.rs`'s own `RELAXATION` constant doc for why 1.0 (the naive
    // "trust the solve fully" choice) is not safe for a live-updating
    // scene: the Poisson solve is exact for THIS substep's instantaneous
    // divergence only, not a converged steady state.
    const RELAXATION: f32 = 0.2;
    vel -= (grad_p / own_mass) * RELAXATION;
    grid_int[base4 + 0u] = bitcast<i32>(vel.x);
    grid_int[base4 + 1u] = bitcast<i32>(vel.y);
}

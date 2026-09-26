// Adaptive substep timestep, decided on the GPU.
//
// The CPU picks one substep size per FRAME, from the particle mirror as it was at the
// frame's start (`GpuSimulation::step_frame`'s scan). CPU stepping instead re-picks
// before every substep (`spacetime::solver::step`), so it tightens dt the instant a
// violent contact makes the state stiffer, while the GPU crossed the whole impact with
// the calm pre-impact dt. That gap is real and measured: at material_cfl 0.4 the GPU's
// vortex spikes J to 1.28 where the CPU, same scene and same coefficient, stays at 1.025.
//
// This pass closes it. Every substep, `g2p_update` folds each particle's own post-update
// CFL bound into `adaptive_dt[2]` (atomicMin on the f32 bit pattern -- positive floats
// order the same as their bits), and this pass turns that into the next substep's dt.
// It can only go BELOW the CPU's own frame-start choice (`dt_cap`), never above, so the
// GPU may tighten what the CPU approved but never loosen it.
//
// When the frame's time is spent, `adaptive_dt[0]` becomes 0 and every pass of the
// remaining encoded substeps returns immediately (the CPU encodes a fixed, slightly
// generous substep count -- it cannot know in advance how far the GPU will tighten).

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    contact_friction:   f32,
    grid_cell_size:     f32,
    contact_active:     u32,
    cfl_coefficient:    f32,
    material_cfl_coefficient: f32,
    min_dt:             f32,
    dt_cap:             f32,
}

@group(0) @binding(3) var<uniform>             step_params: StepParams;
@group(2) @binding(37) var<storage, read_write> adaptive_dt: array<atomic<u32>, 4>;

// Below this, a leftover slice of frame time is dropped rather than stepped: the same
// "honest dropped time" the CPU loop already reports when it runs out of substeps.
const DT_EPSILON: f32 = 1.0e-9;

@compute @workgroup_size(1, 1, 1)
fn cfl_commit_main() {
    let bound = bitcast<f32>(atomicLoad(&adaptive_dt[2]));
    let remaining = bitcast<f32>(atomicLoad(&adaptive_dt[1]));
    var next = min(bound, step_params.dt_cap);
    if !(next > 0.0) {
        next = step_params.dt_cap;
    }
    next = max(next, step_params.min_dt);
    next = min(next, remaining);
    if !(next > DT_EPSILON) {
        next = 0.0;
    }
    atomicStore(&adaptive_dt[0], bitcast<u32>(next));
    atomicStore(&adaptive_dt[1], bitcast<u32>(max(remaining - next, 0.0)));
    atomicStore(&adaptive_dt[2], bitcast<u32>(3.4e38));
    let advanced = bitcast<f32>(atomicLoad(&adaptive_dt[3])) + next;
    atomicStore(&adaptive_dt[3], bitcast<u32>(advanced));
}

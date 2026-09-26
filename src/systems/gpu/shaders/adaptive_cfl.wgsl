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
// `[3]` and `[4]` accumulate the time and the number of substeps actually executed,
// which the CPU reads back as the frame's real substep count and dropped time.

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
@group(2) @binding(37) var<storage, read_write> adaptive_dt: array<atomic<u32>, 5>;

// Below this, a leftover slice of frame time is dropped rather than stepped: the same
// "honest dropped time" the CPU loop already reports when it runs out of substeps.
const DT_EPSILON: f32 = 1.0e-9;

@compute @workgroup_size(1, 1, 1)
fn cfl_commit_main() {
    // `[0]` still holds the dt of the substep that just ran (0 if it was one of the
    // frame's spare encoded substeps), so what was really executed is counted here.
    let just_ran = bitcast<f32>(atomicLoad(&adaptive_dt[0]));
    if just_ran > 0.0 {
        let executed = bitcast<f32>(atomicLoad(&adaptive_dt[3])) + just_ran;
        atomicStore(&adaptive_dt[3], bitcast<u32>(executed));
        atomicAdd(&adaptive_dt[4], 1u);
    }
    let bound = bitcast<f32>(atomicLoad(&adaptive_dt[2]));
    let remaining = bitcast<f32>(atomicLoad(&adaptive_dt[1]));
    var next = min(bound, step_params.dt_cap);
    if !(next > 0.0) {
        next = step_params.dt_cap;
    }
    // No `max(next, min_dt)` here, deliberately. CPU's `cfl_bound` says it
    // in its own words, and has a test named after it
    // (`min_dt_never_raises_a_cfl_upper_bound`): "min_dt is intentionally
    // not a floor. Raising a material/acoustic CFL upper bound changes the
    // PDE integration; callers must substep, defer, or report inability to
    // meet their work budget instead." This shader used to raise it, which
    // is why a Bingham fluid whose own bound is 2e-4 ran at this scene's
    // 1e-3 min_dt: three substeps a frame against the CPU's eleven, five
    // times past its own CFL limit. A bound that really does fall below
    // min_dt now runs out of the frame's encoded substeps instead, and the
    // leftover time is reported as dropped, which is what the CPU does.
    next = min(next, remaining);
    if !(next > DT_EPSILON) {
        next = 0.0;
    }
    atomicStore(&adaptive_dt[0], bitcast<u32>(next));
    atomicStore(&adaptive_dt[1], bitcast<u32>(max(remaining - next, 0.0)));
    atomicStore(&adaptive_dt[2], bitcast<u32>(3.4e38));
}

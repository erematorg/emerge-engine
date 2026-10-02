// Force fields -- standalone pass, one thread per particle, used after the
// ASFLIP fused G2P. The per-particle logic lives in `force_fields_apply.inc.wgsl`,
// appended to this source at pipeline creation (every other scene runs the same
// logic inside the fused `g2p_update_main`, see `particles_update.wgsl`).

// ── Particle struct -- 128 bytes, matches repr(C) in src/matter/particle.rs ────────────────
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
    internal_pressure:    f32,  // total 128 bytes
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:          f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    contact_friction:              f32,
    grid_cell_size:              f32,
    contact_active:              u32,
}

@group(0) @binding(0) var<storage, read_write> particles:    array<Particle>;
@group(0) @binding(3) var<uniform>             step_params:  StepParams;

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


@compute @workgroup_size(64, 1, 1)
fn force_fields_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The frame's time is already fully advanced -- this encoded substep is spare
    // capacity the CPU could not size exactly in advance (see adaptive_cfl.wgsl).
    if substep_dt() <= 0.0 { return; }
    let p_idx = gid.x;
    if p_idx >= step_params.particle_count { return; }
    var p = particles[p_idx];
    if apply_force_fields(&p) {
        particles[p_idx] = p;
    }
}

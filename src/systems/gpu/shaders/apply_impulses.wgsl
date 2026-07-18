// apply_impulses — apply velocity impulses directly to GPU particle velocities.
//
// Called before particle_sort and physics substeps each frame. Reads the LIVE GPU
// particle positions and writes updated velocities in-place — no CPU mirror upload.
//
// This eliminates the stale-CPU-mirror artifact: previously apply_radial_impulse
// scanned CPU particles (potentially 2 frames stale due to async readback lag),
// modified them, and uploaded. The GPU would receive 2-frame-old positions with
// new velocities, causing visible particle jumps. Now the GPU reads its own current
// positions and applies the impulse correctly.
//
// mode 0 (radial): push/pull from center — `v += normalize(p - center) * strength * falloff`
// mode 1 (directional): fixed force vector — `v += force * falloff`

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

struct ImpulseEntry {
    center:   vec2<f32>,  // impulse origin in grid coords
    radius:   f32,         // influence radius in grid cells
    strength: f32,         // radial strength (signed — negative = pull)
    force:    vec2<f32>,  // directional force vector (mode 1 only)
    mode:     u32,         // 0 = radial, 1 = directional
    _pad:                 u32,
}

struct ImpulseParams {
    count:          u32,
    vel_limit:      f32,   // grid_cell_size / min_dt — hard cap per particle
    particle_count: u32,
    _pad:                 u32,
    entries:        array<ImpulseEntry, 16>,
}

@group(0) @binding(0) var<storage, read_write> particles:      array<Particle>;
@group(0) @binding(1) var<uniform>             impulse_params: ImpulseParams;

@compute @workgroup_size(64, 1, 1)
fn apply_impulses_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= impulse_params.particle_count { return; }

    var vel = particles[i].v;
    let pos = particles[i].x;
    var touched = false;

    for (var k: u32 = 0u; k < impulse_params.count; k++) {
        let e     = impulse_params.entries[k];
        let delta = pos - e.center;
        let d     = length(delta);
        if d > 0.0 && d < e.radius {
            let falloff = 1.0 - d / e.radius;
            if e.mode == 0u {
                vel += (delta / d) * e.strength * falloff;
            } else {
                vel += e.force * falloff;
            }
            touched = true;
        }
    }

    let spd = length(vel);
    if spd > impulse_params.vel_limit && spd > 0.0 {
        vel *= impulse_params.vel_limit / spd;
    }

    particles[i].v = vel;
    // Wake on genuine disturbance — without this, a sleeping particle inside an
    // impulse's radius gets a real velocity written but stays sleeping=1, so every
    // other pass (p2g/g2p/particles_update/force_fields) keeps skipping it: the
    // velocity sits inert (position never integrates) until it happens to wake on
    // its own via a neighbor's grid activity, then suddenly resumes motion using
    // this stale injected velocity — a surprising delayed "pop", not an immediate
    // push. Same wake condition as everywhere else: a real disturbance clears it.
    if touched && particles[i].sleeping != 0u {
        particles[i].sleeping = 0u;
    }
}

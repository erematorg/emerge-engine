// Force fields — non-uniform body forces applied after G2P. One thread per particle.
//
// Field types: 0=disabled, 1=GravityWell, 2=Coulomb, 3=AabbConfinement,
//              4=RadialConfinement, 5=UniformElectric, 6=Buoyancy.
// params layout per type is documented in GpuForceFieldEntry constructors (src/gpu/mod.rs).
// All positions in grid coordinates. material_mask bit i = material i affected.
//
// MAX_FORCE_FIELDS is a pipeline-overridable constant set from src/gpu/mod.rs at pipeline creation.

// ── Particle struct — 112 bytes, matches repr(C) in src/mechanics/particle.rs ────────────────
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
    sleeping:             u32,       // total 112 bytes
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
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}

// 48 bytes — matches GpuForceFieldEntry in src/gpu/mod.rs
struct ForceFieldEntry {
    field_type:    u32,
    material_mask: u32,  // bit i = material i affected; 0xFFFFFFFF = all
    _pad0:         u32,
    _pad1:         u32,
    params01:      vec4<f32>,  // params[0..3]
    params45:      vec4<f32>,  // params[4..7]
}

// 784 bytes — matches GpuForceFieldsParams in src/gpu/mod.rs
struct ForceFieldsParams {
    count:   u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
    entries: array<ForceFieldEntry, 16>,
}

// 80 bytes — matches GpuSleepWakeParams in src/gpu/mod.rs.
// Minimal hook for LP's future chunk system: force-sleep/force-wake by user_tag,
// independent of velocity. See sleep_tag/wake_tag doc comments on GpuSimulation.
// Tags packed 4-per-vec4<u32> (8 tags = 2 vec4s) — a flat array<u32,8> would need
// 16-byte-strided elements in uniform address space and waste 3x the space.
struct SleepWakeParams {
    sleep_count: u32,
    wake_count:  u32,
    _pad0:       u32,
    _pad1:       u32,
    sleep_tags:  array<vec4<u32>, 2>,
    wake_tags:   array<vec4<u32>, 2>,
}

// Reads tag index i (0..7) out of a packed array<vec4<u32>, 2>.
fn packed_tag(tags: array<vec4<u32>, 2>, i: u32) -> u32 {
    return tags[i / 4u][i % 4u];
}

// ── Named constants ───────────────────────────────────────────────────────────
override MAX_FORCE_FIELDS:       u32;
override MAX_SLEEP_WAKE_TAGS:    u32;
const FIELD_DISABLED:            u32 = 0u;
const FIELD_GRAVITY_WELL:        u32 = 1u;
const FIELD_COULOMB:             u32 = 2u;
const FIELD_AABB_CONFINEMENT:    u32 = 3u;
const FIELD_RADIAL_CONFINEMENT:  u32 = 4u;
const FIELD_UNIFORM_ELECTRIC:    u32 = 5u;
const FIELD_BUOYANCY:            u32 = 6u;
// Prevent divide-by-near-zero in softened potentials.
const FF_NUM_FLOOR:              f32 = 1e-10;
// Prevent a = F/m overflow when mass is negligible.
const FF_MASS_FLOOR:             f32 = 1e-10;
// material_mask sentinel: all materials affected.
const MASK_ALL:                  u32 = 0xFFFFFFFFu;

@group(0) @binding(0) var<storage, read_write> particles:    array<Particle>;
@group(0) @binding(3) var<uniform>             step_params:  StepParams;
@group(0) @binding(4) var<uniform>             force_fields: ForceFieldsParams;
@group(0) @binding(7) var<uniform>             sleep_wake:   SleepWakeParams;

// Returns true if entry applies to the given material.
fn material_matches(entry: ForceFieldEntry, material_id: u32) -> bool {
    if entry.material_mask == MASK_ALL { return true; }
    return (entry.material_mask & (1u << material_id)) != 0u;
}

// Cubic force-switch taper: returns 1 in [0, onset], tapers to 0 at cutoff.
// Mirrors GravityWellField / CoulombField FADE_ONSET_RATIO convention on CPU.
fn force_switch(dist: f32, cutoff: f32, switch_on: f32) -> f32 {
    if dist <= switch_on { return 1.0; }
    if dist >= cutoff    { return 0.0; }
    let t = (cutoff - dist) / (cutoff - switch_on);
    return t * t * (3.0 - 2.0 * t);  // smoothstep
}

@compute @workgroup_size(64, 1, 1)
fn force_fields_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_idx = gid.x;
    if p_idx >= step_params.particle_count { return; }

    var p = particles[p_idx];

    // Force-wake-by-tag: checked unconditionally (not just while currently sleeping),
    // because the sleep_wake uniform stays loaded for every substep in this step_frame()
    // call (uploaded once per frame — see GpuSimulation::step_frame), not just the one
    // substep where the caller's wake_tag() takes effect. A tagged particle hasn't been
    // given any real velocity by waking — it's still genuinely at rest — so without
    // exempting it from the natural sleep-scoring below for the WHOLE frame (not just the
    // substep where the flag flips), substep 2 onward would see near-zero velocity and
    // immediately re-sleep it, undoing the wake before step_frame() ever returns.
    // force_sleep_tag (checked last, below) still wins over this — a deliberate forced
    // sleep beats a forced wake, same "last write wins" rule as any other conflict here.
    var is_wake_tagged = false;
    for (var i: u32 = 0u; i < sleep_wake.wake_count && i < MAX_SLEEP_WAKE_TAGS; i++) {
        if p.user_tag == packed_tag(sleep_wake.wake_tags, i) { is_wake_tagged = true; }
    }
    if is_wake_tagged { p.sleeping = 0u; }

    // Still-sleeping particles get no field forces, no velocity clamp, and no
    // rescoring — frozen, same as CPU excluding them from the force-field loop
    // entirely (`for i in 0..self.active_count`). No writeback needed: nothing changes.
    if p.sleeping != 0u { return; }

    let dt = step_params.dt;

    // force_fields.count == 0u is handled by the loop condition below (i < count
    // is immediately false) — no need for an early return, since the velocity
    // clamp and sleep-scoring after this loop must always run regardless.
    for (var i: u32 = 0u; i < force_fields.count && i < MAX_FORCE_FIELDS; i++) {
        let entry = force_fields.entries[i];
        if entry.field_type == FIELD_DISABLED { continue; }
        // GravityWell (1) and Coulomb (2) are applied to grid velocities in grid_update.wgsl.
        // Applying them here too would double their effect — skip.
        if entry.field_type == FIELD_GRAVITY_WELL || entry.field_type == FIELD_COULOMB { continue; }
        if !material_matches(entry, p.material_id) { continue; }

        switch entry.field_type {
            case FIELD_GRAVITY_WELL: {
                // Plummer-softened point-mass: a = -G*M·r / (r²+ε²)^(3/2)
                // params01 = (src_x, src_y, G*M, softening²)
                // params45 = (_, _, cutoff, switch_on)
                let src = vec2<f32>(entry.params01.x, entry.params01.y);
                let gm = entry.params01.z;
                let eps2 = entry.params01.w;
                let cutoff  = entry.params45.z;
                let sw_on   = entry.params45.w;

                let r     = p.x - src;          // from source to particle
                let r2    = dot(r, r);
                let r_len = sqrt(r2);
                if cutoff > 0.0 && r_len >= cutoff { continue; }

                let r2_soft = r2 + eps2;
                let r3 = r2_soft * sqrt(r2_soft);   // (r²+ε²)^(3/2)
                if r3 < FF_NUM_FLOOR { continue; }

                var acc = -(gm / r3) * r;           // toward source
                if cutoff > 0.0 {
                    acc *= force_switch(r_len, cutoff, sw_on);
                }
                p.v += acc * dt;
            }
            case FIELD_COULOMB: {
                // Plummer-softened Coulomb: a = k·q_s·q_p·r / (r²+ε²)^(3/2)  (positive = repulsion)
                // params01 = (src_x, src_y, charge_factor, softening²)
                // params45 = (_, _, cutoff, switch_on)
                let src = vec2<f32>(entry.params01.x, entry.params01.y);
                let charge_factor = entry.params01.z;  // k * q_source * q_particle (signed)
                let eps2 = entry.params01.w;
                let cutoff  = entry.params45.z;
                let sw_on   = entry.params45.w;

                let inv_mass = select(0.0, 1.0 / p.mass, p.mass > FF_MASS_FLOOR);
                if inv_mass == 0.0 { continue; }

                let r     = p.x - src;
                let r2    = dot(r, r);
                let r_len = sqrt(r2);
                if cutoff > 0.0 && r_len >= cutoff { continue; }

                let r2_soft = r2 + eps2;
                let r3 = r2_soft * sqrt(r2_soft);
                if r3 < FF_NUM_FLOOR { continue; }

                var acc = (charge_factor * inv_mass / r3) * r;  // positive = away from source
                if cutoff > 0.0 {
                    acc *= force_switch(r_len, cutoff, sw_on);
                }
                p.v += acc * dt;
            }
            case FIELD_AABB_CONFINEMENT: {
                // Soft repulsion from AABB walls.
                // params01 = (min_x, min_y, max_x, max_y)
                // params45 = (stiffness, thickness, _, _)
                let min_c    = vec2<f32>(entry.params01.x, entry.params01.y);
                let max_c    = vec2<f32>(entry.params01.z, entry.params01.w);
                let stiff    = entry.params45.x;
                let thick    = entry.params45.y;
                var acc      = vec2<f32>(0.0, 0.0);

                let pen_lx = (min_c.x + thick) - p.x.x;
                let pen_rx = p.x.x - (max_c.x - thick);
                let pen_by = (min_c.y + thick) - p.x.y;
                let pen_ty = p.x.y - (max_c.y - thick);

                if pen_lx > 0.0 { acc.x += stiff * pen_lx; }   // push right
                if pen_rx > 0.0 { acc.x -= stiff * pen_rx; }   // push left
                if pen_by > 0.0 { acc.y += stiff * pen_by; }   // push up
                if pen_ty > 0.0 { acc.y -= stiff * pen_ty; }   // push down

                p.v += acc * dt;
            }
            case FIELD_RADIAL_CONFINEMENT: {
                // Soft inward repulsion outside (radius − thickness).
                // params01 = (cx, cy, radius, stiffness)
                // params45 = (thickness, _, _, _)
                let center  = vec2<f32>(entry.params01.x, entry.params01.y);
                let radius  = entry.params01.z;
                let stiff   = entry.params01.w;
                let thick   = entry.params45.x;

                let r_vec  = p.x - center;
                let dist   = length(r_vec);
                let onset  = radius - thick;

                if dist > onset && dist > FF_NUM_FLOOR {
                    let excess = dist - onset;
                    // inward acceleration proportional to penetration
                    let acc = -(stiff * excess / dist) * r_vec;
                    p.v += acc * dt;
                }
            }
            case FIELD_UNIFORM_ELECTRIC: {
                // Spatially uniform E field: a = q·E / m
                // params01 = (field_x, field_y, charge, _)
                let e_field  = vec2<f32>(entry.params01.x, entry.params01.y);
                let charge   = entry.params01.z;
                let inv_mass = select(0.0, 1.0 / p.mass, p.mass > FF_MASS_FLOOR);
                if inv_mass == 0.0 { continue; }
                p.v += (charge * inv_mass) * e_field * dt;
            }
            case FIELD_BUOYANCY: {
                // True Archimedes buoyancy: net a = g·(ρ_fluid/ρ₀_particle − 1) upward.
                // Formula: Δv = −g · (ρ_fluid/ρ₀) · dt. Solver already applies −g, so
                // net = −g + g·(ρ_fluid/ρ₀) = g·(ρ_fluid/ρ₀ − 1) — upward for lighter particles.
                // Uses rest density (m/V₀) not instantaneous density — prevents expansion runaway.
                // params01 = (gravity_x, gravity_y, fluid_density, _unused)
                let gravity_vec  = vec2<f32>(entry.params01.x, entry.params01.y);
                let fluid_rho    = entry.params01.z;
                let rest_rho     = p.mass / max(p.initial_volume, 1e-10);
                p.v += -gravity_vec * (fluid_rho / max(rest_rho, 1e-10)) * dt;
            }
            default: {}
        }
    }

    // Clamp velocity magnitude (same limit applied in g2p).
    let v_len = length(p.v);
    if v_len > step_params.vel_limit && v_len > FF_NUM_FLOOR {
        p.v = p.v * (step_params.vel_limit / v_len);
    }

    // Sleep scoring — mirrors src/solver/mod.rs (~lines 855-870) exactly, including the
    // lack of a same-substep re-sleep guard (CPU doesn't have one either). Runs last per
    // substep, after force fields, since a field can keep a particle that looks sleepy
    // awake. No-op whenever sleep_threshold <= 0.0 (the SimConfig default). Skipped this
    // substep if force-woken above — a force-woken particle hasn't been given any
    // velocity, so without this guard it would immediately re-sleep here, undoing the
    // wake before it's ever observed outside this function.
    if !is_wake_tagged && step_params.sleep_threshold > 0.0 && p.activation == 0.0
        && dot(p.v, p.v) < step_params.sleep_threshold * step_params.sleep_threshold {
        p.sleeping = 1u;
    }

    // Force-sleep-by-tag: independent of velocity/activation, checked last so it wins
    // over the velocity-based scoring above (a forced sleep is deliberate, not a guess).
    for (var i: u32 = 0u; i < sleep_wake.sleep_count && i < MAX_SLEEP_WAKE_TAGS; i++) {
        if p.user_tag == packed_tag(sleep_wake.sleep_tags, i) { p.sleeping = 1u; }
    }

    particles[p_idx] = p;
}

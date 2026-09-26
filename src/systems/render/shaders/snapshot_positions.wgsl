// Extracts each active particle's position into a tightly-packed buffer --
// the pre-step snapshot `prep_instances.wgsl` interpolates against for
// "Fix Your Timestep" (Gaffer 2004) render smoothing. Called once per
// render-frame's physics-step batch, BEFORE stepping (see
// `Renderer::snapshot_particle_positions`'s own doc) -- zero CPU readback,
// stays GPU-resident the whole way.
//
// Particle struct layout (128 bytes) must match src/matter/particle.rs
// exactly, same contract as `prep_instances.wgsl`.

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
    internal_pressure:    f32,
}

// NOT `vec3<u32>` for the padding: WGSL's uniform address-space layout rules
// give vec3 a 16-byte alignment, which silently inflates this struct's real
// GPU size to 32 bytes (16-byte aligned offset + its own 16-byte rounded
// size) even though the equivalent `repr(C)` Rust struct (u32 + [u32;3],
// all 4-byte-aligned) is only 16 bytes -- a real, confirmed wgpu validation
// panic ("size 16 where the shader expects 32") caught by actually running
// this, not a hypothetical. Three separate `u32` fields keep every member's
// alignment at 4 bytes, matching the Rust side exactly.
struct SnapshotConfig {
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> prev_positions: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             config: SnapshotConfig;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= config.particle_count { return; }
    prev_positions[id] = particles[id].x;
}

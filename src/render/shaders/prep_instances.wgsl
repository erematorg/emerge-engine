// Prepare instanced draw data from the MPM particle buffer.
//
// One thread per particle. Reads Particle[], writes InstanceData[].
// InstanceData feeds the render_particles vertex shader as per-instance attributes.
//
// Color modes (RenderConfig::mode):
//   0 = ByMaterial — material_id % 16 → fixed palette
//   1 = ByVelocity — |v| * vel_scale → blue→red heat map
//   2 = ByVolume   — det(F) → blue (compressed) / white (rest) / red (expanded)
//
// Particle struct layout (112 bytes) must match src/particle.rs exactly.

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
    _pad:                 u32,
}

// InstanceData layout (48 bytes) — must match MpmRenderer's VertexBufferLayout:
//   deform_col0: vec2<f32> @ offset  0
//   deform_col1: vec2<f32> @ offset  8
//   position:    vec2<f32> @ offset 16
//   _pad:        vec2<f32> @ offset 24
//   color:       vec4<f32> @ offset 32
struct InstanceData {
    deform_col0: vec2<f32>,
    deform_col1: vec2<f32>,
    position:    vec2<f32>,
    _pad:        vec2<f32>,
    color:       vec4<f32>,
}

struct RenderConfig {
    mode:           u32,
    particle_count: u32,
    vel_scale:      f32, // 1 / (max expected speed) — maps |v| to [0, 1]
    _pad:           u32,
}

@group(0) @binding(0) var<storage, read>       particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(2) var<uniform>             config:    RenderConfig;

// ── Color helpers ─────────────────────────────────────────────────────────────

// 16-slot material palette. material_id % 16 selects the color.
fn material_color(id: u32) -> vec4<f32> {
    switch id % 16u {
        case  0u: { return vec4(0.35, 0.65, 1.00, 1.0); } // blue   (fluid default)
        case  1u: { return vec4(0.90, 0.80, 0.30, 1.0); } // yellow (sand default)
        case  2u: { return vec4(0.80, 0.90, 1.00, 1.0); } // ice    (snow default)
        case  3u: { return vec4(0.50, 0.85, 0.50, 1.0); } // green  (elastic default)
        case  4u: { return vec4(1.00, 0.45, 0.20, 1.0); } // orange
        case  5u: { return vec4(0.85, 0.35, 0.35, 1.0); } // red
        case  6u: { return vec4(0.65, 0.40, 0.85, 1.0); } // purple
        case  7u: { return vec4(0.40, 0.85, 0.80, 1.0); } // teal
        case  8u: { return vec4(0.90, 0.60, 0.40, 1.0); } // peach
        case  9u: { return vec4(0.50, 0.50, 0.90, 1.0); } // indigo
        case 10u: { return vec4(0.70, 0.90, 0.40, 1.0); } // lime
        case 11u: { return vec4(1.00, 0.80, 0.20, 1.0); } // gold
        case 12u: { return vec4(0.85, 0.50, 0.75, 1.0); } // pink
        case 13u: { return vec4(0.40, 0.70, 0.50, 1.0); } // sage
        case 14u: { return vec4(0.60, 0.60, 0.60, 1.0); } // grey
        default:  { return vec4(1.00, 1.00, 1.00, 1.0); } // white
    }
}

// Smooth heat map: t ∈ [0, 1] → blue → cyan → green → yellow → red.
fn heat(t: f32) -> vec4<f32> {
    let c = clamp(t, 0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - abs(c - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    return vec4(r, g, b, 1.0);
}

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= config.particle_count { return; }

    let p = particles[id];
    let f = p.deformation_gradient;

    var color: vec4<f32>;
    if config.mode == 0u {
        // ByMaterial: palette lookup by material_id.
        color = material_color(p.material_id);
    } else if config.mode == 1u {
        // ByVelocity: |v| * vel_scale → heat map.
        color = heat(length(p.v) * config.vel_scale);
    } else {
        // ByVolume: J = det(F). J < 1 → compressed (blue), J > 1 → expanded (red).
        // Map: J=0 → t=0 (blue), J=1 → t=0.5 (green/white), J=2 → t=1 (red).
        let j = det2(f);
        color = heat(clamp(j * 0.5, 0.0, 1.0));
    }

    instances[id] = InstanceData(
        f[0],        // deform_col0 — F's x-axis
        f[1],        // deform_col1 — F's y-axis
        p.x,         // position in grid coords
        vec2(0.0),   // _pad
        color,
    );
}

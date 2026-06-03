// Instanced particle render — 2D.
// Vertex: deforms unit quad by F, projects via orthographic camera.
// Fragment: optional disc clip with soft edge.

struct Camera {
    view_proj:       mat4x4<f32>,
    particle_scale:  f32,
    round_particles: u32,
    _pad:            vec2<f32>,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexIn {
    @location(0) local_pos:   vec2<f32>, // unit quad corner [-0.5, 0.5]²
    @location(1) deform_col0: vec2<f32>, // F column 0  (per-instance)
    @location(2) deform_col1: vec2<f32>, // F column 1
    @location(3) position:    vec2<f32>, // particle grid position
    @location(4) color:       vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color:          vec4<f32>,
    @location(1) local_pos:      vec2<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let f        = mat2x2<f32>(in.deform_col0, in.deform_col1);
    let deformed = f * (in.local_pos * cam.particle_scale) + in.position;
    var out: VertexOut;
    out.clip_pos  = cam.view_proj * vec4(deformed, 0.0, 1.0);
    out.color     = in.color;
    out.local_pos = in.local_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if cam.round_particles != 0u {
        let d = length(in.local_pos);
        if d > 0.5 { discard; }
        let alpha = 1.0 - smoothstep(0.42, 0.5, d);
        return vec4(in.color.rgb, in.color.a * alpha);
    }
    return in.color;
}

// Orbit/motion trail — fading polyline per particle. Real, standard N-body
// visualization technique (the same "trailing line, older = dimmer" shape
// REBOUND's own OpenGL/js visualizers use for orbit paths) — separate,
// self-contained pipeline from `render_particles.wgsl`'s instanced quads, so
// this stays a general, opt-in capability usable by any demo, not something
// welded into the solar-system example.
//
// wgpu core has no native line-width control, so this draws 1px `LineStrip`
// segments — a disclosed simplification; a thicker ribbon would need a
// quad-per-segment approach instead, not built here.

struct Camera {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) age:      f32,      // 0 = newest point, 1 = oldest (trail tail)
    @location(2) color:    vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color:          vec4<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = cam.view_proj * vec4(in.position, 0.0, 1.0);
    // Real, simple quadratic fade -- newest point stays bright, tail fades
    // out well before the trail's own length limit so it doesn't end in a
    // hard visible cutoff.
    let alpha = pow(1.0 - in.age, 2.0);
    out.color = vec4(in.color.rgb, in.color.a * alpha);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}

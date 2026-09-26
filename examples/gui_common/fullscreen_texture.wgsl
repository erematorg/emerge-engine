// Minimal fullscreen-triangle texture blit -- the standard way to display
// any CPU- or compute-rasterized 2D field that isn't already MPM particle
// or grid data (see basic_energy.rs's own doc for why this scene needs its
// own render path instead of reusing render_particles/grid_volume/
// curvature_flow, all of which are genuinely MPM-specific).
//
// 3 vertices, no vertex buffer: a well-known trick (each vertex's clip
// position is derived purely from its index) that covers the whole screen
// with one triangle, cheaper than two triangles from a real quad buffer
// and with no seam.

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    // Real, standard fullscreen-triangle trick (was WRONG before -- the
    // previous formula produced an ordinary triangle inscribed in clip
    // space, not one large enough to cover the whole viewport, which is
    // exactly the "black triangle" bug found live 2026-08-27). The correct
    // version deliberately overshoots clip space on two sides so the
    // visible -1..1 square sits entirely inside the triangle's interior.
    let u = f32((i << 1u) & 2u);
    let v = f32(i & 2u);
    out.uv = vec2<f32>(u, v);
    out.clip_position = vec4<f32>(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}

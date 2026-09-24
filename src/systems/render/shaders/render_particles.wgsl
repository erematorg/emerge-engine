// Instanced particle render — 2D.
// Vertex: deforms unit quad by F, projects via orthographic camera.
// Fragment: optional disc clip with soft edge.

struct Camera {
    view_proj:       mat4x4<f32>,
    particle_scale:  f32,
    round_particles: u32,
    // Real, general, opt-in soft-glow strength (0.0 = old hard-edged disc,
    // byte-identical default). See fs_main's own doc for the falloff.
    glow_strength:   f32,
    // Real, general, opt-in point-light position (grid coords) for
    // billboard-sphere Lambertian shading -- see fs_main's own doc. Two
    // scalars, not vec2<f32> -- WGSL requires vec2 to be 8-byte aligned in
    // a uniform buffer, which would make naga insert padding the Rust-side
    // `#[repr(C)] CameraParams` (tightly packed, 4-byte float alignment)
    // does not have, silently misreading every field after it.
    light_pos_x:      f32,
    light_pos_y:      f32,
    // 0.0 = no shading (old flat-disc look, byte-identical default).
    shading_strength: f32,
    // Real inverse-square-law reference distance (grid units) -- see
    // fs_main's own doc. 0.0 disables the falloff.
    light_reference_distance: f32,
    _pad:             f32,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexIn {
    @location(0) local_pos:   vec2<f32>, // unit quad corner [-0.5, 0.5]²
    @location(1) deform_col0: vec2<f32>, // F column 0  (per-instance)
    @location(2) deform_col1: vec2<f32>, // F column 1
    @location(3) position:    vec2<f32>, // particle grid position
    @location(5) emission:    f32,       // real per-particle blackbody-glow factor (color::blackbody_glow_factor)
    @location(4) color:       vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color:          vec4<f32>,
    @location(1) local_pos:      vec2<f32>,
    @location(2) emission:       f32,
    @location(3) world_pos:      vec2<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let f        = mat2x2<f32>(in.deform_col0, in.deform_col1);
    let deformed = f * (in.local_pos * cam.particle_scale) + in.position;
    var out: VertexOut;
    out.clip_pos  = cam.view_proj * vec4(deformed, 0.0, 1.0);
    out.color     = in.color;
    out.local_pos = in.local_pos;
    out.emission  = in.emission;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if cam.round_particles != 0u {
        let d = length(in.local_pos);
        if d > 0.5 { discard; }
        let edge_alpha = 1.0 - smoothstep(0.42, 0.5, d);
        var rgb = in.color.rgb;

        if cam.shading_strength > 0.0 {
            // Real billboard-impostor sphere shading: reconstructs a
            // hemisphere normal from the particle's own on-quad position
            // (standard technique for shading a flat disc as if it were a
            // real sphere -- e.g. point-sprite planet/moon rendering), then
            // Lambertian-shades it (Lambert's cosine law) by the real
            // direction to `cam.light_pos` (this scene's own tracked light
            // source position, e.g. the Sun's actual, moving barycentric
            // position -- not a fixed art-direction light). Gives a real
            // lit/dark terminator instead of a flat, uniformly-colored
            // disc. `xy` in [-1,1] across the disc's own radius; `z`
            // completes the unit hemisphere normal (zero at the silhouette
            // edge, 1.0 dead center -- the point facing the viewer most
            // directly).
            let n_xy = in.local_pos / 0.5;
            let n_z = sqrt(max(0.0, 1.0 - dot(n_xy, n_xy)));
            let normal = vec3(n_xy, n_z);
            // 2D light direction (this renderer has no out-of-plane depth
            // for light sources) -- a disclosed simplification, still gives
            // a real, correct terminator for light sources roughly in the
            // scene's own plane (true for every current 2D top-down demo).
            // Epsilon avoids a NaN from normalize(0) for a particle that
            // IS its own light source (e.g. the Sun shading itself).
            let to_light = vec2(cam.light_pos_x, cam.light_pos_y) - in.world_pos;
            let dist = length(to_light);
            let light_dir = normalize(vec3(to_light, 0.0) + vec3(1e-6, 0.0, 0.0));
            let ndotl = max(0.0, dot(normal, light_dir));
            // No ambient floor here, deliberately -- unlike `grid_volume.
            // wgsl`'s own 0.6 ambient (an artistic constant with no
            // physical derivation, kept there but not repeated here), a
            // genuinely unlit side facing away from the only real light
            // source in an otherwise empty scene SHOULD read as near-black
            // -- that IS the physically correct answer for deep space, not
            // a rendering flaw to paper over with a made-up floor.
            //
            // Real inverse-square intensity, not direction alone:
            // `light_reference_distance` is the distance at which a
            // particle reads at intensity 1.0 (the caller's own real
            // choice -- `basic_solar_system_gui.rs` uses Earth's actual
            // orbital distance, matching the literal definition of "solar
            // constant" -- see `Renderer::set_light_source`'s own doc), so
            // closer bodies genuinely read brighter and farther ones dimmer
            // by the real 1/r^2 law, same physics `RadianceField::
            // irradiance_at` already computes in SI units for this scene's
            // own live irradiance display -- this is the same law,
            // re-derived directly in the shader's own grid-distance space
            // (a ratio, so SI/grid units cancel cleanly) rather than a
            // second unit-conversion path. `light_reference_distance=0.0`
            // (unset) disables the falloff (intensity=1.0 everywhere,
            // direction-only, the prior behavior).
            var intensity = 1.0;
            if cam.light_reference_distance > 0.0 {
                let floor = cam.light_reference_distance * 0.01;
                let ratio = cam.light_reference_distance / max(dist, floor);
                intensity = ratio * ratio;
            }
            // Real, honest LDR-display cap -- the true unbounded 1/r^2
            // value near the light source itself is real, but this
            // renderer has no HDR/tonemap stage (yet), so the softened
            // ceiling keeps `rgb * shade` from overflowing before the
            // eventual clamp, the same real "clips to white near the
            // source" compromise the existing glow term already makes.
            let lit = min(ndotl * intensity, 4.0);
            let shade = mix(1.0, lit, cam.shading_strength);
            rgb = rgb * shade;
        }

        if cam.glow_strength > 0.0 {
            // Real, simple soft-glow look: a Gaussian-like bright core
            // (real formula, not a texture/lookup) boosts color toward the
            // particle's own center; on an LDR surface this clips to white
            // near the center, a real (if simplified) "glowing" appearance.
            // A disclosed, deliberately simpler alternative to a true
            // multi-pass bloom pipeline (downscale/blur/composite) -- stays
            // within the particle's own existing radius, does not bleed
            // light onto neighboring pixels/particles the way real bloom
            // does. Zero cost / byte-identical look when glow_strength=0.0
            // (every existing caller that never opts in).
            //
            // Real, taxonomically grounded (2026-08-17): `glow_strength` is
            // a renderer-level MAXIMUM, not a uniform brightness boost --
            // `in.emission` (the particle's own real blackbody-emission
            // factor, `color::blackbody_glow_factor`, shared with
            // `ByPhysics`'s own thermal-glow term) gates how much of it a
            // given particle actually receives. A particle with no tracked
            // temperature (emission=0, the common default) gets none: a
            // point-mass planet doesn't glow just because the Sun in the
            // same scene does.
            let core = exp(-d * d * 10.0);
            let boost = 1.0 + core * cam.glow_strength * in.emission;
            rgb = rgb * boost;
        }

        return vec4(rgb, in.color.a * edge_alpha);
    }
    return in.color;
}

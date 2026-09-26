// Prepare instanced draw data from the MPM particle buffer.
//
// One thread per particle. Reads Particle[], writes InstanceData[].
// InstanceData feeds the render_particles vertex shader as per-instance attributes.
//
// Color modes (RenderConfig::mode):
//   0 = ByMaterial -- material_id % 16 → fixed palette
//   1 = ByVelocity -- |v| * vel_scale → blue→red heat map
//   2 = ByVolume   -- det(F) → blue (compressed) / white (rest) / red (expanded)
//
// Particle struct layout (128 bytes) must match src/matter/particle.rs exactly --
// any field/padding drift here silently misaligns every WGSL array index past 0.

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

// InstanceData layout (48 bytes) -- must match MpmRenderer's VertexBufferLayout:
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
    vel_scale:      f32,
    // Render-time position blend factor, "Fix Your Timestep" (Gaffer 2004) --
    // see the Rust-side `RenderConfig::interp_alpha` doc for the real
    // "sudden acceleration" symptom this closes.
    interp_alpha:   f32,
}

// Per-material optical absorption: σ_a [r, g, b, σ_s] × 16 slots.
// Beer-Lambert: transmitted_rgb = exp(-σ_a_rgb).
// High σ_a = strong absorption = dark / hue-shifted toward complementary color.
// .w = σ_s, reduced scattering coefficient (single scalar -- real tissue scattering
// is far less wavelength-dependent than absorption in the visible range, Jacques
// 2013 -- used for a real, bounded single-scattering-albedo subsurface approximation,
// not a full BSSRDF/diffusion simulation).
//
// specular[i].x = R0, Fresnel base reflectance (Schlick 1994 approximation) for
// material slot i. This renderer has no surface-normal estimation (particle
// instances, not a reconstructed/raytraced surface), so this is a constant
// near-normal-incidence reflectance, not a view-angle-dependent Fresnel term --
// a real, cited, but honestly bounded simplification.
struct OpticalTable {
    slots: array<vec4<f32>, 16>,
    specular: array<vec4<f32>, 16>,
}

// Shared validated SI contract. `spatial.z == 0` means the caller has not
// supplied physical scale yet and this path is still in legacy mode.
struct PhysicalRenderParams {
    spatial: vec4<f32>,
    incident_radiance: vec4<f32>,
    background_radiance: vec4<f32>,
    display_white_radiance: vec4<f32>,
    camera_direction: vec4<f32>,
    light_direction: vec4<f32>,
    // x = thermal-emission exposure anchor in kelvin, 0 = unset. See
    // `Renderer::set_emission_reference_temperature`. y/z/w reserved.
    emission: vec4<f32>,
}

@group(0) @binding(0) var<storage, read>       particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(2) var<uniform>             config:    RenderConfig;
@group(0) @binding(3) var<uniform>             optics:    OpticalTable;
@group(0) @binding(4) var<uniform>             physical:  PhysicalRenderParams;
// Pre-step position snapshot from `snapshot_positions.wgsl`, taken once per
// render-frame's physics-step batch (mirrors the CPU `prev_x` convention in
// `basic_fluids.rs`). Blended against the current `p.x` by `config.interp_alpha`
// below -- zero-readback, GPU-resident the whole way, no `sync_particles_blocking`.
@group(0) @binding(5) var<storage, read>       prev_positions: array<vec2<f32>>;

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
    } else if config.mode == 2u {
        // ByVolume: J = det(F). J < 1 → blue (compressed), J > 1 → red (expanded).
        let j = det2(f);
        color = heat(clamp(j * 0.5, 0.0, 1.0));
    } else if config.mode == 3u {
        // ByPhysics: Beer-Lambert absorption + real subsurface scattering
        // (single-scattering approximation) + Fresnel specular + blackbody thermal
        // emission.
        //
        // Absorption: white light transmitted through one particle layer.
        //   transmitted = exp(-σ_a)   [Beer-Lambert, depth=1 particle]
        //   J < 1 (compressed) → denser → deeper optical path → more absorption.
        let slot = p.material_id % 16u;
        let sigma_a = optics.slots[slot].rgb;
        let sigma_s = optics.slots[slot].w;
        let j = clamp(det2(p.deformation_gradient), 0.05, 4.0);
        let optical_depth = 1.0 / j; // compressed = denser = deeper path
        let transmitted = exp(-sigma_a * optical_depth);
        //
        // Subsurface scattering (single-scattering albedo approximation, real but
        // bounded -- see OpticalTable's own doc). Real single-scattering albedo:
        //   a = σ_s / (σ_s + σ_a)
        // Light lost to absorption alone would just leave the medium dark; real
        // scattering tissue instead looks brighter/softer than pure absorption
        // predicts, because scattered photons re-emerge diffusely rather than
        // being lost. Blend transmitted color toward a soft, desaturated glow
        // by the real albedo fraction, weighted by how deep light had to travel.
        let albedo = sigma_s / max(sigma_s + sigma_a, vec3(1e-4));
        let scatter_glow = vec3(1.0, 0.95, 0.9) * (1.0 - exp(-sigma_s * optical_depth));
        let with_scattering = mix(transmitted, scatter_glow, clamp(albedo, vec3(0.0), vec3(1.0)));
        //
        // Specular: constant near-normal Fresnel reflectance (Schlick 1994), real
        // but bounded -- see OpticalTable's own doc for why this isn't view-angle
        // dependent. Adds a small additive highlight, real magnitude (water R0~0.02).
        let r0 = optics.specular[slot].x;
        let with_specular = with_scattering + vec3(r0);
        //
        // Thermal emission: real blackbody, Planck's colour weighted by
        // Stefan-Boltzmann's T^4 -- see `blackbody.inc.wgsl`, and
        // `energy::radiation` for the physics it mirrors. Additive, because
        // an emitter's own light adds to whatever it transmits.
        let emission = blackbody_emission(
            p.temperature,
            physical.spatial.z,
            physical.display_white_radiance.rgb,
            physical.emission.x,
        );
        //
        if physical.spatial.z > 0.5 {
            // Real SI radiative transfer: absorption, single scattering and
            // Fresnel together (`radiative_transfer.inc.wgsl`). A particle
            // has no surface normal, so reflection is evaluated at normal
            // incidence -- the same disclosed limitation `OpticalTable`'s
            // own doc already states for this path, now at least anchored
            // to a real R0 instead of added flat.
            let view_length_m = physical.spatial.y / max(abs(physical.camera_direction.z), 1.0e-6);
            let path_m = (1.0 / j) * view_length_m;
            let radiance = slab_radiance(
                physical.background_radiance.rgb,
                physical.incident_radiance.rgb,
                sigma_a,
                sigma_s,
                path_m,
                r0,
                1.0,
            );
            let display_radiance = radiance / physical.display_white_radiance.rgb;
            color = vec4(clamp(display_radiance + emission, vec3(0.0), vec3(1.0)), 1.0);
        } else {
            color = vec4(clamp(with_specular + emission, vec3(0.0), vec3(1.0)), 1.0);
        }
    } else if config.mode == 4u {
        // ByThermal: emission alone, nothing else. Cold is black, an ember is
        // deep red, a flame orange, incandescence white, hotter still blue --
        // the real Planckian sequence, not a colour ramp.
        let thermal = blackbody_emission(
            p.temperature,
            physical.spatial.z,
            physical.display_white_radiance.rgb,
            physical.emission.x,
        );
        color = vec4(clamp(thermal, vec3(0.0), vec3(1.0)), 1.0);
    } else if config.mode == 6u {
        // ByScalarField: generic second carrier (resource/grass level, pheromone,
        // nutrients -- see Particle::scalar_field's own doc). Unlike temperature,
        // this field has no universal physical scale, so no normalization divisor --
        // callers own fields in roughly [0,1] (matches GpuResourceParams' own
        // logistic carrying-capacity convention).
        let s = clamp(p.scalar_field, 0.0, 1.0);
        color = heat(s);
    } else {
        // ByActivation: muscle activation [0,1] → cool (rest) → warm (firing).
        color = heat(clamp(p.activation, 0.0, 1.0) * 0.8);
    }

    // mix(prev, current, alpha): alpha=1.0 (no snapshot taken yet, or the
    // caller passed the "unblended" default) reduces to p.x exactly.
    let render_pos = mix(prev_positions[id], p.x, config.interp_alpha);

    instances[id] = InstanceData(
        f[0],        // deform_col0 -- F's x-axis
        f[1],        // deform_col1 -- F's y-axis
        render_pos,  // interpolated position in grid coords
        vec2(0.0),   // _pad
        color,
    );
}

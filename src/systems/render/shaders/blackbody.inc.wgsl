// Blackbody emission colour -- the GPU mirror of `energy::radiation::spectrum`.
//
// Prepended to `prep_instances.wgsl`, `grid_volume.wgsl` and
// `curvature_flow.wgsl` at pipeline build time (see `render/mod.rs`'s
// `concat!(include_str!(..))` constants), so all three paths render a hot
// particle, a hot cell and a hot surface the same colour.
//
// A fragment shader cannot integrate Planck's law against the CIE observer
// per pixel, so this evaluates Kim et al. 2002's closed-form Planckian locus
// instead. `spectrum::locus_fit_matches_planck_integration` is the test that
// keeps the two within 0.03 per channel over 1667-9000 K; every coefficient
// below is the exact counterpart of one in
// `spectrum::blackbody_linear_srgb_locus_fit`, and the two must change
// together.
//
// What this replaced: `heat(0.5 + T/5000 * 0.5) * (T/5000)^2`, a hand-drawn
// ramp that went green-to-red with rising temperature (real blackbodies go
// red-to-white-to-blue) and scaled brightness as T^2 (Stefan-Boltzmann says
// T^4).

// Stefan-Boltzmann constant -- W/(m^2 K^4). Same value as
// `energy::thermodynamics::transfer::STEFAN_BOLTZMANN`.
const BLACKBODY_STEFAN_BOLTZMANN: f32 = 5.6703744e-8;
// Kim et al. 2002's fit is defined over this range. Below it a blackbody's
// chromaticity has nearly stopped moving while its radiance has already
// collapsed under T^4, so clamping the hue costs far less than the
// brightness term already does.
const BLACKBODY_LOCUS_MIN_K: f32 = 1667.0;
const BLACKBODY_LOCUS_MAX_K: f32 = 25000.0;
// Exposure headroom cap. Not a physical limit: it bounds what the light
// diffusion pass can be fed by a runaway temperature, the way any real
// sensor saturates rather than returning an unbounded value.
const BLACKBODY_MAX_EXPOSURE: f32 = 16.0;
// Used when the shared physical uniform is still zero-initialized, i.e. no
// scene has stated its exposure. Must equal `render/mod.rs`'s own
// `DEFAULT_EMISSION_REFERENCE_K`.
const BLACKBODY_DEFAULT_REFERENCE_K: f32 = 3000.0;

// CIE xy chromaticity to linear sRGB (IEC 61966-2-1, D65), normalized so the
// brightest channel is 1. Out-of-gamut chromaticities desaturate toward
// white instead of clipping one channel, which would shift the hue silently.
fn blackbody_chromaticity_to_srgb(x: f32, y: f32) -> vec3<f32> {
    if y <= 1.0e-6 {
        return vec3(1.0);
    }
    // Unit-luminance XYZ for this chromaticity: Y is 1 by construction.
    let big_x = x / y;
    let big_y = 1.0;
    let big_z = (1.0 - x - y) / y;
    var rgb = vec3(
        3.2406 * big_x - 1.5372 * big_y - 0.4986 * big_z,
        -0.9689 * big_x + 1.8758 * big_y + 0.0415 * big_z,
        0.0557 * big_x - 0.2040 * big_y + 1.0570 * big_z,
    );
    let lowest = min(rgb.r, min(rgb.g, rgb.b));
    if lowest < 0.0 {
        rgb -= vec3(lowest);
    }
    let highest = max(rgb.r, max(rgb.g, rgb.b));
    if highest <= 0.0 {
        return vec3(0.0);
    }
    return rgb / highest;
}

// Exposure-normalized colour of a blackbody at `temperature_k`.
fn blackbody_srgb(temperature_k: f32) -> vec3<f32> {
    let t = clamp(temperature_k, BLACKBODY_LOCUS_MIN_K, BLACKBODY_LOCUS_MAX_K);
    let inv = 1.0 / t;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;

    var x: f32;
    if t <= 4000.0 {
        x = -0.2661239e9 * inv3 - 0.2343589e6 * inv2 + 0.8776956e3 * inv + 0.179910;
    } else {
        x = -3.0258469e9 * inv3 + 2.1070379e6 * inv2 + 0.2226347e3 * inv + 0.240390;
    }

    let x2 = x * x;
    let x3 = x2 * x;
    var y: f32;
    if t <= 2222.0 {
        y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
    } else if t <= 4000.0 {
        y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
    } else {
        y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    }

    return blackbody_chromaticity_to_srgb(x, y);
}

// How bright this blackbody is, relative to whatever the scene exposes for.
// Stefan-Boltzmann: total emitted radiance goes as T^4, so a ratio of
// temperatures raised to the fourth IS the exposure ratio.
//
// Two ways to know what "full brightness" means, both real:
//
//  - A `PhysicalRenderContract` states the display's white in W/(m^2 sr).
//    Then a Lambertian emitter's own radiance, `L = sigma T^4 / pi`, divided
//    by that white, is the answer outright -- the exposure is measured.
//    Only the white's mean is used; a deliberately non-neutral display white
//    would also tint emission, which this does not model.
//  - Otherwise the scene names a reference temperature and that renders at
//    1.0, the way a photographer picks an exposure.
//
// Arguments rather than direct uniform reads: the three shaders that include
// this file name that uniform differently.
// `display_white_mean` of 0 means no contract, so use `reference_k`.
// Passes whose bind group does not carry the shared physical uniform (the
// light-diffusion sweep) call this directly with the two scalars they do
// carry.
fn blackbody_exposure_scalar(
    temperature_k: f32,
    display_white_mean: f32,
    reference_k: f32,
) -> f32 {
    if temperature_k <= 0.0 {
        return 0.0;
    }
    if display_white_mean > 0.0 {
        let t2 = temperature_k * temperature_k;
        let radiance = BLACKBODY_STEFAN_BOLTZMANN * t2 * t2 / 3.14159265;
        return min(radiance / display_white_mean, BLACKBODY_MAX_EXPOSURE);
    }
    let reference = select(reference_k, BLACKBODY_DEFAULT_REFERENCE_K, reference_k <= 0.0);
    let ratio = temperature_k / max(reference, 1.0);
    let ratio2 = ratio * ratio;
    return min(ratio2 * ratio2, BLACKBODY_MAX_EXPOSURE);
}

fn blackbody_exposure(
    temperature_k: f32,
    contract_enabled: f32,
    display_white: vec3<f32>,
    reference_k: f32,
) -> f32 {
    let mean = select(
        0.0,
        max(dot(display_white, vec3(1.0 / 3.0)), 1.0e-12),
        contract_enabled > 0.5,
    );
    return blackbody_exposure_scalar(temperature_k, mean, reference_k);
}

// Thermal emission: Planck's colour times Stefan-Boltzmann's brightness.
fn blackbody_emission(
    temperature_k: f32,
    contract_enabled: f32,
    display_white: vec3<f32>,
    reference_k: f32,
) -> vec3<f32> {
    let exposure = blackbody_exposure(temperature_k, contract_enabled, display_white, reference_k);
    if exposure <= 0.0 {
        return vec3(0.0);
    }
    return blackbody_srgb(temperature_k) * exposure;
}

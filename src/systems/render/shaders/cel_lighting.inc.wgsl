// Cel-shaded Lambertian lighting of a 2D density field, shared by the
// grid-volume and curvature-flow paths (both prepend this file), so their
// three call sites cannot drift apart.

// Number of flat bands N.L is quantized into (toon shading), to match this
// engine's non-photoreal 2D style instead of reading as photoreal 3D
// lighting, while still driven by the real density gradient. Fewer bands (4)
// amplify a fluid surface's genuine small-scale curvature detail into
// visible posterization noise; 8 stays clean.
const LIGHT_BANDS: f32 = 8.0;

// Brightness factor for a pixel whose density gradient is `grad`: an ambient
// floor (0.6) that keeps the shape visible on the unlit side, plus a banded
// Lambertian term (0.4 * N.L), with the outward normal along -grad.
//
// The normal carries only the gradient's direction, so on its own a
// sub-percent density ripple inside a body would be shaded as fully as a
// real edge, pointing wherever the ripple does. The Lambertian term is
// therefore weighted by the gradient's size against `free_surface_step`, the
// steepest a straight free surface gives in the caller's grid (Rust's
// `free_surface_cell_step`): full shading at an edge, next to none for a
// ripple. With no gradient the factor is 1.
//
// `holds_shape` is the fraction of the matter here that holds its shape
// (`OpticalTable::specular[slot].y`, a nonzero shear modulus). Such a body
// has flat faces: its density ramp at the edge is the reconstruction
// kernel's width, not a rounded surface, so shading it would draw a bevel
// the body does not have. Its share of the edge shading is removed.
fn cel_lambert(
    grad: vec2<f32>,
    light_dir: vec2<f32>,
    free_surface_step: f32,
    holds_shape: f32,
) -> f32 {
    let grad_len = length(grad);
    if grad_len <= 1.0e-5 {
        return 1.0;
    }
    let diffuse_raw = clamp(dot(-grad / grad_len, normalize(light_dir)), 0.0, 1.0);
    let diffuse = floor(diffuse_raw * LIGHT_BANDS) / LIGHT_BANDS;
    let surface_weight = clamp(grad_len / max(free_surface_step, 1.0e-12), 0.0, 1.0)
        * (1.0 - clamp(holds_shape, 0.0, 1.0));
    return mix(1.0, 0.6 + 0.4 * diffuse, surface_weight);
}

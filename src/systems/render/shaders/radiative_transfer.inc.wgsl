// Radiative transfer through a slab of matter -- the GPU side of
// `energy::radiation`'s attenuation and Fresnel laws.
//
// Prepended alongside `blackbody.inc.wgsl` to every shader that renders
// matter under a `PhysicalRenderContract`, so the particle, grid and surface
// paths agree by construction.
//
// What this replaced: a contract branch that computed transmission only.
// Absorption was measured and honest, but the scattering and reflection
// terms of the legacy dimensionless path were simply dropped, so enabling
// real SI optics cost a scene its subsurface glow and its specular. Those
// terms exist here in SI, driven by the contract's own declared incident
// radiance rather than by a hand-picked tint.

// Schlick's angular Fresnel approximation. Mirror of
// `energy::radiation::fresnel::schlick_reflectance` -- see that function for
// the citation and for why the approximation is exact at both ends.
fn fresnel_schlick(r0: f32, cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, 0.0, 1.0);
    let m = 1.0 - c;
    let m2 = m * m;
    return clamp(r0 + (1.0 - r0) * m2 * m2 * m, 0.0, 1.0);
}

// Radiance leaving a slab of absorbing, scattering medium, in whatever unit
// `background` and `incident` are given in.
//
// Three real terms, in the order light meets them:
//
//  - Fresnel reflection at the front face. What reflects never enters, so
//    it carries the incident radiance straight back and the two interior
//    terms are weighted by `1 - R`.
//  - Direct transmission of whatever is behind the slab, attenuated by the
//    full extinction `sigma_t = sigma_a + sigma_s`. Scattering removes light
//    from the direct beam just as absorption does, which is why the
//    exponent is not `sigma_a` alone.
//  - Single scattering. Of everything the slab took out of the beam,
//    `sigma_s / sigma_t` -- the single-scattering albedo -- was scattered
//    rather than absorbed, and some of it leaves toward the viewer. Lit by
//    the incident radiance, so the glow takes the colour of the actual light
//    source. (Jacques, "Optical properties of biological tissues: a review",
//    Phys. Med. Biol. 58, 2013, already the citation for the dimensionless
//    path this generalises.)
//
// Disclosed simplification, unchanged from the dimensionless path: single
// scattering with an isotropic phase function and no multiple-scattering
// term, so a very dense, highly scattering medium (thick cloud, milk) is
// under-lit. Nothing here is fitted -- correcting it means adding the
// diffusion term, not a constant.
fn slab_radiance(
    background: vec3<f32>,
    incident: vec3<f32>,
    sigma_a: vec3<f32>,
    sigma_s: f32,
    path_m: f32,
    r0: f32,
    cos_view: f32,
) -> vec3<f32> {
    let sigma_t = sigma_a + vec3(max(sigma_s, 0.0));
    let transmitted = exp(-sigma_t * max(path_m, 0.0));
    let albedo = vec3(max(sigma_s, 0.0)) / max(sigma_t, vec3(1.0e-9));
    let scattered = incident * albedo * (vec3(1.0) - transmitted);
    let reflectance = fresnel_schlick(r0, cos_view);
    return (background * transmitted + scattered) * (1.0 - reflectance)
        + incident * reflectance;
}

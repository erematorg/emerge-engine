//! CPU-path color computation for `Renderer` -- split out of `mod.rs` (was its
//! own already-marked "Color helpers (CPU path)" section plus `particle_color`,
//! together ~110 of the file's ~930 lines). Mirrors `prep_instances.wgsl`'s
//! ByPhysics branch exactly -- see that shader for the real citations/
//! derivation of each term (Beer-Lambert absorption, single-scattering-albedo
//! subsurface approximation, Schlick Fresnel specular, blackbody emission).

use glam::Mat2;

use super::{ColorMode, OpticalTable, Renderer};
use crate::energy::radiation::{
    blackbody_linear_srgb_locus_fit, blackbody_radiance_w_m2_sr, slab_radiance,
};
use crate::particle::Particle;

/// Mirrors `blackbody.inc.wgsl`'s `BLACKBODY_MAX_EXPOSURE`; the two must stay
/// equal or CPU and GPU emission diverge above the ceiling.
const BLACKBODY_MAX_EXPOSURE: f32 = 16.0;

impl Renderer {
    /// Thermal emission: Planck's colour (`energy::radiation`) weighted by
    /// Stefan-Boltzmann's `T^4` exposure. The CPU mirror of
    /// `blackbody.inc.wgsl`'s `blackbody_emission`, including its exposure
    /// ceiling, so both paths saturate at the same place.
    pub(super) fn blackbody_emission(&self, temperature_k: f32) -> [f32; 3] {
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return [0.0; 3];
        }
        let exposure = match self.physical_render_contract {
            Some(_) => {
                blackbody_radiance_w_m2_sr(temperature_k) / self.display_white_mean().max(1.0e-12)
            }
            None => {
                let ratio = temperature_k / self.emission_reference_temperature().max(1.0);
                ratio * ratio * ratio * ratio
            }
        }
        .min(BLACKBODY_MAX_EXPOSURE);
        blackbody_linear_srgb_locus_fit(temperature_k).map(|channel| channel * exposure)
    }

    pub(super) fn particle_color(&self, p: &Particle, i: usize) -> [f32; 4] {
        match self.color_mode {
            ColorMode::ByMaterial => material_palette(p.material_id),
            ColorMode::ByVelocity => heat(p.v.length() * self.vel_scale),
            ColorMode::ByVolume => heat(det2(p.deformation_gradient) * 0.5),
            ColorMode::ByPhysics => {
                // Mirrors prep_instances.wgsl's ByPhysics branch exactly -- see that
                // shader's comments for the real citations/derivation of each term
                // (Beer-Lambert absorption, single-scattering-albedo subsurface
                // approximation, Schlick Fresnel specular, blackbody emission).
                let slot = p.material_id as usize % 16;
                let sigma = self.sigma_a[slot];
                // Real pore-fluid index-matching darkening -- see
                // `Renderer::set_refractive_index`'s own doc for the
                // mechanism/citations. Generic: driven by whatever this
                // particle's OWN `scalar_field` holds (moisture, or any
                // other saturating quantity a scene wires up there), not a
                // sand-specific special case. `refractive_index[slot]==1.0`
                // (default) makes `contrast_dry==0`, guarded below to skip
                // entirely -- byte-identical color for every material that
                // never opts in.
                let sigma_s = {
                    let base = self.sigma_s[slot];
                    let n_solid = self.refractive_index[slot];
                    let contrast_dry = (n_solid - 1.0).abs();
                    if contrast_dry > 1.0e-4 {
                        const N_WATER: f32 = 1.33; // Hecht, "Optics" -- standard reference
                        let saturation = p.scalar_field.clamp(0.0, 1.0);
                        let n_fluid = 1.0 + saturation * (N_WATER - 1.0);
                        let contrast_wet = (n_solid - n_fluid).abs();
                        base * (contrast_wet / contrast_dry).powi(2)
                    } else {
                        base
                    }
                };
                let j = det2(p.deformation_gradient).clamp(0.05, 4.0);
                if let Some(contract) = self.physical_render_contract {
                    // Full SI radiative transfer, the same law
                    // `radiative_transfer.inc.wgsl` runs on the GPU. A
                    // particle carries no surface normal, so Fresnel is
                    // evaluated at normal incidence (`cos_view = 1`).
                    let path_m = (1.0 / j) * contract.view_thickness_meters()
                        / contract.camera_direction().z.abs().max(1.0e-6);
                    let radiance = slab_radiance(
                        contract.background_radiance_w_m2_sr(),
                        contract.incident_radiance_w_m2_sr(),
                        sigma,
                        sigma_s,
                        path_m,
                        self.specular_r0[slot],
                        1.0,
                    );
                    let display_white = contract.display_white_radiance_w_m2_sr();
                    let emission = self.blackbody_emission(p.temperature);
                    return [
                        (radiance[0] / display_white[0] + emission[0]).clamp(0.0, 1.0),
                        (radiance[1] / display_white[1] + emission[1]).clamp(0.0, 1.0),
                        (radiance[2] / display_white[2] + emission[2]).clamp(0.0, 1.0),
                        1.0,
                    ];
                }
                let od = 1.0 / j;
                let transmitted = [
                    (-sigma[0] * od).exp(),
                    (-sigma[1] * od).exp(),
                    (-sigma[2] * od).exp(),
                ];
                let scatter_glow = [1.0f32, 0.95, 0.9];
                let with_scattering: Vec<f32> = (0..3)
                    .map(|c| {
                        let albedo = (sigma_s / (sigma_s + sigma[c]).max(1e-4)).clamp(0.0, 1.0);
                        let glow = scatter_glow[c] * (1.0 - (-sigma_s * od).exp());
                        transmitted[c] * (1.0 - albedo) + glow * albedo
                    })
                    .collect();
                let r0 = self.specular_r0[slot];
                let emission = self.blackbody_emission(p.temperature);
                [
                    (with_scattering[0] + r0 + emission[0]).min(1.0),
                    (with_scattering[1] + r0 + emission[1]).min(1.0),
                    (with_scattering[2] + r0 + emission[2]).min(1.0),
                    1.0,
                ]
            }
            ColorMode::ByThermal => {
                let [r, g, b] = self.blackbody_emission(p.temperature);
                [r.min(1.0), g.min(1.0), b.min(1.0), 1.0]
            }
            ColorMode::ByActivation => heat(p.activation.clamp(0.0, 1.0) * 0.8),
            ColorMode::ByScalarField => heat(p.scalar_field.clamp(0.0, 1.0)),
            ColorMode::ByStress => {
                let sigma_vm = self.stress_field.get(i).copied().unwrap_or(0.0);
                heat(sigma_vm * self.stress_scale)
            }
        }
    }
}

pub(super) fn write_optical_table(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    sigma_a: &[[f32; 3]; 16],
    sigma_s: &[f32; 16],
    specular_r0: &[f32; 16],
) {
    let mut table = OpticalTable {
        slots: [[0.0; 4]; 16],
        specular: [[0.0; 4]; 16],
    };
    for (i, s) in sigma_a.iter().enumerate() {
        table.slots[i] = [s[0], s[1], s[2], sigma_s[i]];
        table.specular[i] = [specular_r0[i], 0.0, 0.0, 0.0];
    }
    queue.write_buffer(buf, 0, bytemuck::bytes_of(&table));
}

fn det2(f: Mat2) -> f32 {
    f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x
}

fn heat(t: f32) -> [f32; 4] {
    let c = t.clamp(0.0, 1.0);
    let r = smoothstep(0.5, 0.75, c);
    let g = 1.0 - (c - 0.5).abs() * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, c);
    [r, g, b, 1.0]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn material_palette(id: u32) -> [f32; 4] {
    match id % 16 {
        0 => [0.35, 0.65, 1.00, 1.0],
        1 => [0.90, 0.80, 0.30, 1.0],
        2 => [0.80, 0.90, 1.00, 1.0],
        3 => [0.50, 0.85, 0.50, 1.0],
        4 => [1.00, 0.45, 0.20, 1.0],
        5 => [0.85, 0.35, 0.35, 1.0],
        6 => [0.65, 0.40, 0.85, 1.0],
        7 => [0.40, 0.85, 0.80, 1.0],
        8 => [0.90, 0.60, 0.40, 1.0],
        9 => [0.50, 0.50, 0.90, 1.0],
        10 => [0.70, 0.90, 0.40, 1.0],
        11 => [1.00, 0.80, 0.20, 1.0],
        12 => [0.85, 0.50, 0.75, 1.0],
        13 => [0.40, 0.70, 0.50, 1.0],
        14 => [0.60, 0.60, 0.60, 1.0],
        _ => [1.00, 1.00, 1.00, 1.0],
    }
}

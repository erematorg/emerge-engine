//! CPU-path color computation for `Renderer` -- split out of `mod.rs` (was its
//! own already-marked "Color helpers (CPU path)" section plus `particle_color`,
//! together ~110 of the file's ~930 lines). Mirrors `prep_instances.wgsl`'s
//! ByPhysics branch exactly -- see that shader for the real citations/
//! derivation of each term (Beer-Lambert absorption, single-scattering-albedo
//! subsurface approximation, Schlick Fresnel specular, blackbody emission).

use glam::Mat2;

use super::{ColorMode, OpticalTable, Renderer};
use crate::particle::Particle;

impl Renderer {
    pub(super) fn particle_color(&self, p: &Particle) -> [f32; 4] {
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
                let sigma_s = self.sigma_s[slot];
                let j = det2(p.deformation_gradient).clamp(0.05, 4.0);
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
                let t = (p.temperature / 5000.0).clamp(0.0, 1.0);
                let glow = t * t * 2.0;
                let [er, eg, eb, _] = heat(0.5 + t * 0.5);
                [
                    (with_scattering[0] + r0 + er * glow).min(1.0),
                    (with_scattering[1] + r0 + eg * glow).min(1.0),
                    (with_scattering[2] + r0 + eb * glow).min(1.0),
                    1.0,
                ]
            }
            ColorMode::ByThermal => {
                let t = (p.temperature / 1500.0).clamp(0.0, 1.0);
                let [r, g, b, _] = heat(t);
                [
                    r * (0.1 + t * 0.9),
                    g * (0.1 + t * 0.9),
                    b * (0.1 + t * 0.9),
                    1.0,
                ]
            }
            ColorMode::ByActivation => heat(p.activation.clamp(0.0, 1.0) * 0.8),
            ColorMode::ByScalarField => heat(p.scalar_field.clamp(0.0, 1.0)),
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

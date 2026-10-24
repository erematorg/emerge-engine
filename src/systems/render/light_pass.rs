//! The light pass (`shaders/light_pass.wgsl`): the transmittance of the
//! declared light to each physics-grid cell, after the matter between that
//! cell and the light. The grid-volume and curvature-flow surface paths both
//! run it before drawing, each from its own density field, and both read the
//! result in their SI branch.

use super::Renderer;
use super::gpu_types::LightPassParams;
use crate::energy::radiation::penetration_attenuation_m_inv;

/// A per-cell `vec4<f32>` field at `res x res`, as the light pass reads and
/// writes.
pub(super) fn light_field_buffer(device: &wgpu::Device, res: u32, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(res) * u64::from(res) * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// The density field the light pass reads the matter from.
pub(super) enum LightPassSource<'a> {
    /// The solver's P2G grid (mass at word 2 of each cell's four), at the
    /// march resolution, with its per-slot mass when tracked.
    Grid {
        grid: &'a wgpu::Buffer,
        material_mass: &'a wgpu::Buffer,
        material_mass_enabled: bool,
    },
    /// The curvature-flow surface density, one f32 per cell at `surface_res`
    /// over the same domain, averaged down to the physics grid. `material_slot`
    /// is the slot whose optics a cell takes when it carries no per-slot mass.
    Surface {
        density: &'a wgpu::Buffer,
        material_mass: &'a wgpu::Buffer,
        material_mass_enabled: bool,
        surface_res: u32,
        material_slot: u32,
    },
}

impl Renderer {
    /// Grows the light-pass fields to `res x res`. Call before creating a
    /// bind group that reads `light_transmittance_buf`: growing replaces it.
    pub(super) fn ensure_light_pass_capacity(&mut self, device: &wgpu::Device, res: u32) {
        if res > self.light_pass_res {
            self.light_extinction_buf = light_field_buffer(device, res, "light_extinction");
            self.light_transmittance_buf = light_field_buffer(device, res, "light_transmittance");
            self.light_pass_res = res;
        }
    }

    /// Grows the surface path's per-surface-cell extinction to
    /// `surface_res x surface_res`. Call before `encode_light_pass` with a
    /// `LightPassSource::Surface`.
    pub(super) fn ensure_light_surface_capacity(
        &mut self,
        device: &wgpu::Device,
        surface_res: u32,
    ) {
        if surface_res > self.light_surface_extinction_res {
            self.light_surface_extinction_buf =
                light_field_buffer(device, surface_res, "light_surface_extinction");
            self.light_surface_extinction_res = surface_res;
        }
    }

    /// Encodes the light pass for a `res x res` physics grid into `enc`,
    /// filling `light_transmittance_buf`; `ensure_light_pass_capacity` (and,
    /// for a surface source, `ensure_light_surface_capacity`) must have grown
    /// the fields first. SI rendering only: the legacy branches never read
    /// the transmittance, and their depths have no length unit to march in,
    /// so without a `PhysicalRenderContract` nothing is encoded.
    pub(super) fn encode_light_pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        res: u32,
        source: LightPassSource,
    ) {
        if self.physical_render_contract.is_none() {
            return;
        }
        debug_assert!(
            res <= self.light_pass_res,
            "light pass fields not grown to {res}"
        );
        let slot_attenuation = std::array::from_fn(|slot| {
            let [r, g, b] = penetration_attenuation_m_inv(self.sigma_a[slot], self.sigma_s[slot]);
            [r, g, b, 0.0]
        });
        let (density, material_mass, params, surface_res) = match source {
            LightPassSource::Grid {
                grid,
                material_mass,
                material_mass_enabled,
            } => (
                grid,
                material_mass,
                LightPassParams {
                    res,
                    source_res: res,
                    reference_cell_mass: self.grid_reference_cell_mass,
                    material_mass_enabled: material_mass_enabled as u32,
                    fallback_slot: 0,
                    _pad: [0; 3],
                    slot_attenuation,
                },
                None,
            ),
            LightPassSource::Surface {
                density,
                material_mass,
                material_mass_enabled,
                surface_res,
                material_slot,
            } => {
                debug_assert!(
                    surface_res <= self.light_surface_extinction_res,
                    "surface light field not grown to {surface_res}"
                );
                (
                    density,
                    material_mass,
                    LightPassParams {
                        res,
                        source_res: surface_res,
                        reference_cell_mass: self.grid_reference_cell_mass,
                        material_mass_enabled: material_mass_enabled as u32,
                        fallback_slot: material_slot,
                        _pad: [0; 3],
                        slot_attenuation,
                    },
                    Some(surface_res),
                )
            }
        };
        queue.write_buffer(&self.light_pass_params_buf, 0, bytemuck::bytes_of(&params));
        let light_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_pass_bg"),
            layout: &self.light_pass_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: density.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: material_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.light_pass_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.physical_render_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.light_extinction_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.light_transmittance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.light_surface_extinction_buf.as_entire_binding(),
                },
            ],
        });
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("light_pass"),
            timestamp_writes: None,
        });
        cp.set_bind_group(0, &light_bg, &[]);
        let groups = |side: u32| side.div_ceil(8);
        match surface_res {
            None => {
                cp.set_pipeline(&self.light_extinction_grid_pipeline);
                cp.dispatch_workgroups(groups(res), groups(res), 1);
            }
            Some(surface_res) => {
                let [per_surface_cell, per_grid_cell] = &self.light_extinction_surface_pipelines;
                cp.set_pipeline(per_surface_cell);
                cp.dispatch_workgroups(groups(surface_res), groups(surface_res), 1);
                cp.set_pipeline(per_grid_cell);
                cp.dispatch_workgroups(groups(res), groups(res), 1);
            }
        }
        cp.set_pipeline(&self.light_march_pipeline);
        cp.dispatch_workgroups(groups(res), groups(res), 1);
    }
}

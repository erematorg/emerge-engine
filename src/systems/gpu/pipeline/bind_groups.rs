//! Bind-group construction for `SimPipelines` -- split out of `pipeline.rs`
//! (was ~200 of its ~850 lines), same reasoning as the `buffers.rs`/
//! `readback.rs` split: pipeline/layout CONSTRUCTION stays in `pipeline.rs`,
//! per-substep bind-group building lives here.

use super::GpuBuffers;
use super::SimPipelines;

impl SimPipelines {
    /// Build a bind group for the apply_impulses pass (particles + impulse_params).
    /// Created on-demand each cursor frame — cheap, no GPU work.
    pub fn make_impulse_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_impulse_bind_group"),
            layout: &self.impulse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.impulse_params.as_entire_binding(),
                },
            ],
        })
    }

    /// Build a bind group for one substep using the given step_params buffer slot.
    /// Cheap — wgpu bind groups are descriptor tables, not data copies.
    pub fn make_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
        step_params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.materials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: step_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.force_fields_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.sorted_particle_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.block_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.sleep_wake_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.active_block_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.active_block_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.active_block_ids_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.active_block_count_prev.as_entire_binding(),
                },
            ],
        })
    }

    /// Build the group-1 (contact subsystem) bind group. Unlike `make_bind_group`, this
    /// takes no `step_params` slot and, like every buffer it originally bound, is
    /// particle-count-independent — `spawn_region` reallocating `buffers.particles`
    /// never invalidates it. One exception since `material_mass` joined this group
    /// (bind-group economy, see its own binding comment): that buffer IS replaced once,
    /// lazily, on first `attach_grid_material_render_gpu` call, so this bind group must
    /// be rebuilt then too — mirrors `attach_asflip_gpu` rebuilding `resource_bind_group`
    /// for the exact same reason. See the module doc comment on the bind-group-layout
    /// split for why this is a separate group at all.
    pub fn make_contact_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_contact_bind_group"),
            layout: &self.contact_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.grip_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.contact_points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: buffers.contact_point_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: buffers.contact_debug_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: buffers.contact_debug_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: buffers.resolved_grip_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: buffers.resolved_rest_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: buffers.grip_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 30,
                    resource: buffers.material_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 31,
                    resource: buffers.material_mass_params.as_entire_binding(),
                },
            ],
        })
    }

    /// Build the group-2 (thermal subsystem) bind group. Same "built once, never
    /// rebuilt" shape as `make_contact_bind_group` -- none of these buffers are
    /// particle-count-scaled (all fixed `grid_res²`-sized), so `spawn_region`
    /// reallocating `buffers.particles` never invalidates it.
    pub fn make_thermal_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_thermal_bind_group"),
            layout: &self.thermal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: buffers.thermal_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: buffers.thermal_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: buffers.thermal_temp_old.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: buffers.thermal_work.as_entire_binding(),
                },
            ],
        })
    }

    /// Build the group-3 (resource regrowth + ASFLIP) bind group. Same "built once,
    /// never rebuilt" shape as `make_thermal_bind_group` -- none of these buffers are
    /// particle-count-scaled. Carries ASFLIP's 2 bindings (28-29) alongside resource
    /// regrowth purely for bind-group-count economy -- see the module doc comment on
    /// `SimPipelines::new`'s Group 3 entry for why the two unrelated subsystems share
    /// a group.
    pub fn make_resource_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm_resource_bind_group"),
            layout: &self.resource_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: buffers.resource_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: buffers.resource_mass.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 26,
                    resource: buffers.resource_phi_old.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 27,
                    resource: buffers.resource_work.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 28,
                    resource: buffers.asflip_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 29,
                    resource: buffers.asflip_snapshot.as_entire_binding(),
                },
            ],
        })
    }
}

//! GPU force-field entries -- split out of `step_params.rs`, see that module's
//! own doc comment for the full file map.

/// Maximum number of active GPU force-field entries per frame.
/// Must match `MAX_FORCE_FIELDS` in `force_fields.wgsl`.
pub const MAX_FORCE_FIELDS: usize = 16;

/// Field-type discriminants -- match `FIELD_*` constants in `force_fields.wgsl`.
pub mod field_type {
    pub const DISABLED: u32 = 0;
    pub const GRAVITY_WELL: u32 = 1;
    pub const COULOMB: u32 = 2;
    pub const AABB_CONFINEMENT: u32 = 3;
    pub const RADIAL_CONFINEMENT: u32 = 4;
    pub const UNIFORM_ELECTRIC: u32 = 5;
    pub const BUOYANCY: u32 = 6;
    pub const LINEAR_DRAG: u32 = 7;
    pub const SPATIAL_DRAG_CYLINDER: u32 = 8;
}

/// One GPU force-field entry -- 48 bytes, 16-byte aligned.
/// Matches `struct FieldEntry` in `force_fields.wgsl` exactly (size-asserted).
/// Use the named constructors instead of filling `params` manually.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFieldEntry {
    pub field_type: u32,
    pub material_mask: u32,
    pub _pad: [u32; 2],
    pub params: [f32; 8],
}

const _: () = assert!(core::mem::size_of::<GpuFieldEntry>() == 48);

impl GpuFieldEntry {
    /// material_mask value for a field that affects all materials.
    pub const ALL_MATERIALS: u32 = 0xFFFF_FFFF;

    /// Plummer-softened point-mass gravity: a = −G·M·r / (r²+ε²)^(3/2).
    ///
    /// - `gm`: gravitational_constant × source_mass (positive = attractive)
    /// - `softening_sq`: Plummer ε² (prevents singularity at r=0)
    /// - `cutoff`: hard cutoff distance (0.0 = no cutoff)
    /// - `switch_on`: force-switch onset (< cutoff; force tapers from `switch_on` to `cutoff`)
    pub fn gravity_well(
        pos: glam::Vec2,
        gm: f32,
        softening_sq: f32,
        cutoff: f32,
        switch_on: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = pos.x;
        p[1] = pos.y;
        p[2] = gm;
        p[3] = softening_sq;
        p[6] = cutoff;
        p[7] = switch_on;
        Self {
            field_type: field_type::GRAVITY_WELL,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Plummer-softened Coulomb interaction for one (source, material) pair.
    ///
    /// - `charge_factor`: k × q_source × q_particle (signed; positive = repulsion)
    /// - `softening_sq`: Plummer ε²
    /// - `material_id`: which material's particles are affected (bitmask = 1 << id)
    /// - `cutoff` / `switch_on`: same as `gravity_well`
    pub fn coulomb(
        pos: glam::Vec2,
        charge_factor: f32,
        softening_sq: f32,
        material_id: u32,
        cutoff: f32,
        switch_on: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = pos.x;
        p[1] = pos.y;
        p[2] = charge_factor;
        p[3] = softening_sq;
        p[6] = cutoff;
        p[7] = switch_on;
        Self {
            field_type: field_type::COULOMB,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Soft repulsive walls of an axis-aligned bounding box.
    ///
    /// Particles that penetrate within `thickness` cells of any wall get a
    /// restoring acceleration proportional to penetration depth × `stiffness`.
    pub fn aabb_confinement(
        min: glam::Vec2,
        max: glam::Vec2,
        stiffness: f32,
        thickness: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = min.x;
        p[1] = min.y;
        p[2] = max.x;
        p[3] = max.y;
        p[4] = stiffness;
        p[5] = thickness;
        Self {
            field_type: field_type::AABB_CONFINEMENT,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Soft inward repulsion outside a radial shell.
    ///
    /// Particles beyond `radius − thickness` receive an inward acceleration
    /// proportional to excess penetration × `stiffness`.
    pub fn radial_confinement(
        center: glam::Vec2,
        radius: f32,
        stiffness: f32,
        thickness: f32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = center.x;
        p[1] = center.y;
        p[2] = radius;
        p[3] = stiffness;
        p[4] = thickness;
        Self {
            field_type: field_type::RADIAL_CONFINEMENT,
            material_mask: Self::ALL_MATERIALS,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Spatially-constant electric field: a = q · E / m.
    ///
    /// - `field`: E-field vector (simulation units -- force per unit charge)
    /// - `charge`: per-particle charge for `material_id` (same units as the Coulomb constant)
    /// - `material_id`: only particles of this material are affected
    pub fn uniform_electric(field: glam::Vec2, charge: f32, material_id: u32) -> Self {
        let mut p = [0f32; 8];
        p[0] = field.x;
        p[1] = field.y;
        p[2] = charge;
        Self {
            field_type: field_type::UNIFORM_ELECTRIC,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Archimedes buoyancy for particles of `material_id` floating in a denser fluid.
    ///
    /// - `gravity`: must match `SimConfig::gravity` (solver gravity, including sign)
    /// - `fluid_density_grid`: surrounding fluid's rest_density in grid units
    ///   (`ρ_SI · dx_m²` -- same value as `NewtonianFluidMaterial::rest_density`, no
    ///   extra `/dt_s²` factor)
    /// - `material_id`: only particles of this material receive the buoyancy force
    ///
    /// Uses particle rest density (`mass / initial_volume`) not instantaneous density,
    /// preventing the expansion-buoyancy runaway where expanded fluid appears falsely light.
    /// Applies `Δv = −gravity · (fluid_density / ρ₀_particle − 1) · dt` each substep.
    pub fn buoyancy(gravity: glam::Vec2, fluid_density_grid: f32, material_id: u32) -> Self {
        let mut p = [0f32; 8];
        p[0] = gravity.x;
        p[1] = gravity.y;
        p[2] = fluid_density_grid;
        p[3] = 1.0e-4; // min_density floor -- mirrors BuoyancyField::new default
        Self {
            field_type: field_type::BUOYANCY,
            material_mask: 1 << material_id,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Linear drag toward a target/ambient flow velocity: a = k·(v_target − v_particle) --
    /// see `LinearDragField`'s (CPU) doc comment for the real physics (Stokes drag /
    /// Rayleigh friction) this mirrors exactly. River current, wind-blown sand, any scene
    /// needing sustained directional flow instead of gravity settling into a static
    /// pool/pile.
    ///
    /// - `target_velocity`: the ambient flow velocity particles relax toward
    /// - `drag_coefficient`: relaxation rate k (1/time); decay timescale is 1/k
    /// - `material_mask`: general bitmask (`1 << material_id`, OR together for several,
    ///   or `Self::ALL_MATERIALS`) -- NOT a single `material_id` like most other
    ///   constructors here, matching `LinearDragField`'s own CPU-side parameter exactly
    ///   for real CPU/GPU parity.
    pub fn linear_drag(
        target_velocity: glam::Vec2,
        drag_coefficient: f32,
        material_mask: u32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = target_velocity.x;
        p[1] = target_velocity.y;
        p[2] = drag_coefficient;
        Self {
            field_type: field_type::LINEAR_DRAG,
            material_mask,
            _pad: [0; 2],
            params: p,
        }
    }

    /// Spatially-varying wind/current drag: same `a = k·(target(x) − v)` mechanism as
    /// `linear_drag`, but `target` is sampled from the real, exact closed-form solution
    /// for 2D potential flow around a circular cylinder (uniform stream + doublet
    /// superposition -- see CPU's `SpatialDragField`/its test module doc for the derivation
    /// and citations). WGSL has no function pointers, so unlike CPU's generic
    /// `target_velocity_fn: fn(Vec2) -> Vec2`, this GPU port bakes this ONE specific
    /// analytic formula into its own field-type case in `force_fields.wgsl` -- the real,
    /// disclosed trade-off of porting a fn-pointer-based mechanism to a shader.
    ///
    /// - `cylinder_center`: the flow singularity's position, in grid coordinates
    /// - `free_stream_u`: undisturbed flow speed far from the cylinder (+X direction)
    /// - `radius`: cylinder radius `a` (the doublet strength is `U·a²`)
    /// - `drag_coefficient`: same relaxation rate `k` as `linear_drag`
    pub fn spatial_drag_potential_flow_cylinder(
        cylinder_center: glam::Vec2,
        free_stream_u: f32,
        radius: f32,
        drag_coefficient: f32,
        material_mask: u32,
    ) -> Self {
        let mut p = [0f32; 8];
        p[0] = cylinder_center.x;
        p[1] = cylinder_center.y;
        p[2] = free_stream_u;
        p[3] = radius;
        p[4] = drag_coefficient;
        Self {
            field_type: field_type::SPATIAL_DRAG_CYLINDER,
            material_mask,
            _pad: [0; 2],
            params: p,
        }
    }
}

/// Uniform buffer containing all active GPU force-field entries -- 784 bytes.
/// Matches `struct FieldsParams` in `force_fields.wgsl` exactly (size-asserted).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFieldsParams {
    pub count: u32,
    pub _pad: [u32; 3],
    pub entries: [GpuFieldEntry; MAX_FORCE_FIELDS],
}

const _: () = assert!(core::mem::size_of::<GpuFieldsParams>() == 784);

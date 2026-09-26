//! Opt-in per-substep subsystem params: thermal diffusion, resource
//! reaction-diffusion, ASFLIP, and material-mass render accumulation. Each
//! follows the same `enabled`/`disabled()` gate convention -- 0 skips every
//! pass for that subsystem entirely, zero cost, byte-identical to before the
//! subsystem existed. Split out of `step_params.rs`, see that module's own doc
//! comment for the full file map.

/// Grid-based Fourier heat diffusion -- GPU mirror of `ThermalDiffusion`/`ThermalConfig`
/// (`src/energy/thermodynamics/diffusion.rs`). Implements the same real PDE:
/// `∂T/∂t = α·∇²T` (Fourier's law) plus Newton cooling `dT/dt = −k_c·(T−ambient)`.
/// `dt` itself is NOT stored here -- the thermal pass reads `step_params.dt` (group 0)
/// directly, since substep `dt` is already the single source of truth uploaded there
/// every substep; duplicating it here would risk the two going out of sync.
/// `enabled == 0` skips all 4 thermal passes entirely (see `contact_active`'s identical
/// gate-when-unused pattern) -- every scene that never attaches thermal pays nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuThermalParams {
    /// Thermal diffusivity α = k / (c_p · dx²), grid-units²/s -- see
    /// `ThermalConfig::alpha_grid`'s own doc for the real derivation/units.
    pub alpha: f32,
    /// Ambient/boundary temperature -- empty cells and Newton cooling both relax toward this.
    pub ambient: f32,
    /// Newton cooling rate k_c, 1/s. 0.0 = no cooling (adiabatic walls).
    pub cooling_rate: f32,
    /// 0 = no thermal system attached (default, every existing scene) -- skips all 4
    /// thermal passes. 1 = attached and active.
    pub enabled: u32,
}

impl GpuThermalParams {
    pub fn disabled() -> Self {
        Self {
            alpha: 0.0,
            ambient: 0.0,
            cooling_rate: 0.0,
            enabled: 0,
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuThermalParams>() == 16);

/// Generic reaction-diffusion resource field -- GPU mirror of `ScalarDiffusionField`
/// (`src/energy/thermodynamics/scalar_field.rs`), specialized to the one source term
/// its own CPU test module uses: logistic growth (Verhulst 1838, `dφ/dt = r·φ·(1−φ/K)`).
/// Same PDE shape as `GpuThermalParams` (scatter -> normalize -> Laplacian+reaction ->
/// gather), but its own separate group/buffers and carrier field (`particle.scalar_field`,
/// not `particle.temperature`) -- composes freely with `GpuThermalParams` in the same
/// scene since the two no longer share a carrier field.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuResourceParams {
    /// Diffusivity D, grid-units²/s -- spatial spread rate.
    pub diffusivity: f32,
    /// Value assigned to empty cells (no particle mass) and domain boundaries.
    pub ambient: f32,
    /// Logistic growth rate r, 1/s.
    pub resource_r: f32,
    /// Logistic carrying capacity K.
    pub resource_k: f32,
    /// 0 = no resource system attached (default) -- skips all 4 passes entirely.
    pub enabled: u32,
    pub _pad: [u32; 3],
}

impl GpuResourceParams {
    pub fn disabled() -> Self {
        Self {
            diffusivity: 0.0,
            ambient: 0.0,
            resource_r: 0.0,
            resource_k: 0.0,
            enabled: 0,
            _pad: [0; 3],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuResourceParams>() == 32);

/// ASFLIP (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the Material Point
/// Method: A Scheme for Easier Separation and Less Dissipation") -- GPU mirror of
/// `SimConfig::asflip_blend`. `enabled == 0` (the default, every existing scene) makes
/// `grid_update.wgsl` skip the pre-force velocity snapshot write entirely and the fused
/// `g2p_asflip_fused.wgsl` pass never gets dispatched (see `SubstepGates::asflip_active`)
/// -- zero cost, byte-identical behavior to before ASFLIP existed, matching every other
/// opt-in GPU subsystem's own gate.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAsflipParams {
    /// Blend factor [0, 1] -- see `SimConfig::asflip_blend`'s own doc for the real
    /// derivation and the ~0.97 reference value (`nepluno/pyasflip`).
    pub blend: f32,
    /// 0 = ASFLIP disabled (default) -- skips the snapshot write and the fused G2P+
    /// position pass, falling back to the ordinary split g2p/particles_update passes
    /// unchanged. 1 = attached and active.
    pub enabled: u32,
    pub _pad: [u32; 2],
}

impl GpuAsflipParams {
    pub fn disabled() -> Self {
        Self {
            blend: 0.0,
            enabled: 0,
            _pad: [0; 2],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GpuAsflipParams>() == 16);

/// Number of per-cell material-mass render slots -- matches `render::OpticalTable`'s
/// own real 16-slot cap exactly (that's the actual bottleneck on how many materials
/// can be visually distinguished anyway, independent of `MAX_MATERIAL_SLOTS`'s larger
/// 64-material solver cap). `material_id >= 16` collides into slot `material_id % 16`,
/// same convention `Renderer::set_optical_params` already uses.
pub const MAX_RENDER_MATERIAL_SLOTS: u32 = 16;

/// Opt-in per-cell per-material mass accumulator for `ColorMode::GridVolume`'s
/// material-aware coloring (see `grid_volume.wgsl`'s own doc). 0 = disabled
/// (default) -- P2G skips the extra atomic scatter entirely, zero cost, byte-
/// identical to before this existed, same gate convention as `GpuAsflipParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterialMassParams {
    pub enabled: u32,
    pub _pad: [u32; 3],
}

impl GpuMaterialMassParams {
    pub fn disabled() -> Self {
        Self {
            enabled: 0,
            _pad: [0; 3],
        }
    }
}

/// GPU mirror of `fluid_pressure.wgsl`'s own `FluidPressureParams` struct --
/// field order and types must match exactly (WGSL uniform buffers use the
/// same std140-style layout rules bytemuck's `Pod` derive already assumes
/// elsewhere in this file). Real GPU port of the CPU-proven Chorin-style
/// incompressibility pressure projection, see that shader's own module doc.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFluidPressureParams {
    /// Real reference fluid cell mass (`rest_density * spacing^2`, this
    /// material's own grid units) -- Rust already knows this exactly from
    /// the material/spawn setup, avoiding a GPU-side reduction pass purely
    /// to recover what the CPU equivalent (`pressure.rs`'s own `mass_avg`)
    /// computes analytically. Used only for the free-surface classification
    /// threshold (`reference_cell_mass * 0.3`, matching CPU's own
    /// `mass_avg * 0.3` convention).
    pub reference_cell_mass: f32,
    pub _pad: [f32; 3],
}

const _: () = assert!(core::mem::size_of::<GpuMaterialMassParams>() == 16);

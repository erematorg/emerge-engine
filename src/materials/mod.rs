pub mod bingham;
pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod nacc;
pub mod params;
pub mod rankine;
pub mod registry;
pub mod sand;
pub mod sand_mui;
pub mod snow;
pub(crate) mod svd;
pub mod utils;
pub mod viscoelastic;
pub mod von_mises;

pub use bingham::BinghamFluidMaterial;
pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use nacc::NaccMaterial;
pub use params::MaterialParams;
pub use rankine::RankineMaterial;
pub use registry::{MAX_MATERIAL_SLOTS, MaterialRegistry};
pub use sand::SandMaterial;
pub use sand_mui::SandMuIMaterial;
pub use snow::SnowMaterial;
pub use utils::{
    elastic_wave_dt, gravity_to_grid, lame_from_si, lame_from_young, polar_decomposition_2d,
};
pub use viscoelastic::ViscoelasticMaterial;
pub use von_mises::VonMisesMaterial;

use glam::Mat2;

use crate::particle::{Particle, Particles};

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles — do not change them.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstitutiveModel {
    Fallback = 0,
    Fluid = 1,            // Weakly-compressible Newtonian fluid, Tait EOS
    NeoHookean = 2,       // Neo-Hookean hyperelastic (jelly, soft solids)
    Corotated = 3,        // Corotated linear elastic (stiffer baseline)
    Snow = 4,             // Corotated + SVD plasticity (Stomakhin 2013)
    DruckerPrager = 5,    // Corotated elastic + DP yield surface (sand, soil, rock)
    VonMises = 6,         // J2 perfect plasticity — ductile flow, no hardening (lava, metal, clay)
    Rankine = 7,          // Tensile cutoff + exponential softening — brittle rock, bone, ice
    DruckerPragerMuI = 8, // Rate-dependent DP — µ(I) rheology, granular flow
    Viscoelastic = 9,     // Kelvin-Voigt: NeoHookean elastic + viscous dashpot in parallel
    Nacc = 10,            // Non-Associated Cam-Clay — wet soil, clay, bio tissue under compression
}

pub trait MaterialModel: Send + Sync + core::fmt::Debug {
    /// Which constitutive law this material implements.
    /// Used by the GPU shader to select the correct stress branch per particle.
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fallback
    }
    // Returns the Kirchhoff-like stress used by the transfer kernel.
    // The kernel applies geometry/time factors (dt, kernel_d_inverse, cell_dist, weight).
    fn kirchhoff_stress(&self, _particles: &Particles, _i: usize) -> Mat2 {
        Mat2::ZERO
    }

    // Returns the particle volume used in the stress contribution.
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn timestep_bound(
        &self,
        _particles: &Particles,
        _i: usize,
        _cell_width: f32,
        _material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        f32::INFINITY
    }

    fn update_particle(&self, _particles: &mut Particles, _i: usize, _dt: f32) {}

    /// Seed per-particle plastic state at spawn time.
    ///
    /// Called once per particle immediately after position/volume assignment.
    /// Default: no-op (elastic materials need no initial plastic state).
    /// Override for materials that have a non-zero neutral accumulator (e.g. sand).
    fn init_particle(&self, _particle: &mut Particle) {}

    /// Whether `update_particle` does real work on the CPU.
    ///
    /// Return `false` if plasticity is fully handled on GPU (default).
    /// Return `true` for CPU-only plasticity paths — the GPU solver uses this to
    /// decide whether to download particles and run the CPU pass each frame.
    fn needs_cpu_update(&self) -> bool {
        false
    }

    /// Whether particles of this material require a per-substep density recompute.
    ///
    /// Fluid EOS materials (Newtonian, Bingham) need up-to-date density each substep
    /// because their pressure is a function of current ρ. Elastic/plastic materials
    /// do not — density is derived from J at the end of update_particle.
    /// Default: false. Override in fluid models.
    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Scaling coefficient for activation-driven deviatoric stress.
    ///
    /// When non-zero, the per-particle `activation` field (0.0–1.0) modulates the
    /// deviatoric component of the Kirchhoff stress. This is the engine-level hook for
    /// active matter: muscles, motile cells, contractile tissue.
    ///
    /// Physics: τ_total = τ_elastic + activation × coeff × I  (contractile active pressure)
    /// Default: 0.0 — activation has no effect on passive materials.
    fn activation_scale(&self) -> f32 {
        0.0
    }

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

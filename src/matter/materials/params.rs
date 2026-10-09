/// Flat material parameters for a single registered material slot.
///
/// Layout is a union -- only the fields relevant to the constitutive model are filled;
/// all others are zero. `model` is the `ConstitutiveModel` discriminant and is always set.
///
/// 112 bytes, 16-byte aligned -- directly uploadable to a GPU uniform buffer as
/// `array<MaterialParams, N>` indexed by `particle.material_id`. `_pad` is explicit,
/// always-zeroed reserved space (bytemuck `Pod` forbids implicit padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialParams {
    /// ConstitutiveModel discriminant. The shader reads this first to select the stress branch.
    pub model: u32,

    // --- Elastic (Neo-Hookean, Corotated, Snow) ---
    /// First Lamé parameter λ -- controls bulk-like volumetric stiffness.
    pub lambda: f32,
    /// Second Lamé parameter µ -- controls shear stiffness.
    pub mu: f32,

    // --- Snow plasticity (Stomakhin 2013) ---
    /// Hardening exponent ξ. Scales stiffness as h = exp(ξ(1−Jp)).
    /// VonMises: repurposed as `yield_stress` (union layout).
    pub hardening_exponent: f32,
    /// Snow: compression limit θ_c -- singular values below (1−θ_c) are clamped.
    /// DP (Sand): repurposed as Reynolds dilatancy angle ψ (radians).
    ///            δεᵥᵖ = sin(ψ)·dq per plastic step. 0.0 = no dilation.
    pub compression_limit: f32,
    /// Stretch limit θ_s. Singular values above (1+θ_s) are clamped.
    pub stretch_limit: f32,

    // --- Fluid (Tait EOS + Newtonian viscosity) ---
    /// Reference density ρ₀ at rest. Tait EOS pressure is zero when ρ = ρ₀.
    /// Solids built from SI fill it too, for multi-field contact to size a
    /// particle as `mass / rest_density` (0 = unknown, contact then uses
    /// `initial_volume`); no solid stress branch reads it.
    pub rest_density: f32,
    /// Bulk modulus k in the Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. Use γ ≈ 7 for near-incompressible water.
    pub eos_power: f32,
    /// Dynamic viscosity coefficient µ_f. Scales deviatoric stress.
    pub dynamic_viscosity: f32,

    /// Snow/DP: lower bound on plastic volume ratio Jp. Prevents h from exploding under compression.
    pub volume_ratio_min: f32,
    /// Snow/DP: upper bound on plastic volume ratio Jp.
    pub volume_ratio_max: f32,

    // --- Drucker-Prager friction-angle hardening (Klar 2016 §4) ---
    /// Initial friction angle φ₀ (radians). Dry sand ≈ 35° = 0.611 rad.
    pub dp_h0: f32,
    /// Friction hardening sensitivity. Scales the angle increase with accumulated plastic strain.
    pub dp_h1: f32,
    /// Friction hardening decay rate. Controls how quickly hardening saturates.
    pub dp_h2: f32,
    /// Residual friction angle φ_r (radians). ≈ 10° = 0.175 rad.
    pub dp_h3: f32,

    // --- Active matter ---
    /// Deviatoric stress scale for `Particle::activation`.
    ///
    /// τ = τ_elastic + activation × active_stress_coeff × dev(τ_elastic)
    ///
    /// Physically motivated by active stress in polar/nematic fluids
    /// (Marchetti et al. 2013, Rev. Mod. Phys. 85:1143); this specific
    /// decomposition is an MPM adaptation, not verbatim from that paper.
    /// 0.0 = passive (default). Positive = contractile (muscle-like).
    pub active_stress_coeff: f32,

    // --- VonMises linear isotropic hardening ---
    /// Linear hardening modulus H for VonMises plasticity.
    /// Effective yield stress: σ_Y(κ) = yield_stress + H·κ, where κ = accumulated plastic strain.
    /// 0.0 (default) = perfect plasticity (no hardening).
    pub hardening_modulus: f32,

    // --- Thermal coupling ---
    /// Fluid: effective viscosity µ_eff = µ₀ · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = no temperature dependence (default). Positive = thins with heat (lava-like).
    pub thermal_viscosity_coeff: f32,

    /// Elastic/corotated: Lamé params scaled by (1 + thermal_expansion · T).
    /// Negative = thermal softening (typical). Positive = thermal stiffening (rare).
    /// 0.0 = no temperature dependence (default).
    pub thermal_expansion: f32,

    // --- Granular-fluid / fluid extended (GPU-synced) ---
    /// Lower bound on EOS pressure, in grid units: the tension a fluid
    /// carries before it is taken to have cavitated. Read by every fluid,
    /// not only GranularFluid -- `p2g.wgsl`'s `kirchhoff` applies it inside
    /// the `model == 1u` branch, the same clamp the CPU laws apply, so the
    /// two paths share one number and cannot drift apart.
    ///
    /// Its SI source is each family's own measured coefficient:
    /// `BinghamProps::cavitation_pressure_pa` for a yield-stress fluid,
    /// `NewtonianFluidMaterial::from_physical`'s dissolved-gas figure for a
    /// liquid. Zero is not a neutral default here, it states that the fluid
    /// carries no tension at all.
    pub pressure_floor: f32,
    /// Fluid: bulk (second) viscosity ζ -- adds ζ·(∇·v)·I to Cauchy stress.
    /// 0.0 = off (Stokes assumption).
    pub bulk_viscosity: f32,
    /// Bingham-only regularisation cutoff for the shear rate. This shares a
    /// union slot because curvature-based surface tension is deliberately not
    /// implemented without an interface reconstruction.
    pub critical_shear_rate: f32,
    /// Snow: cohesion -- τ += c·Jp·(J−1)·J·I when Jp<1 and J>1.
    /// Resists elastic expansion in plastically compacted snow. Stable (no feedback loop).
    /// 0.0 = disabled (Stomakhin default). ~200–800 for wet/packed snow.
    /// Repurposed from padding; zero for all other materials.
    pub cohesion_coeff: f32,

    /// Mirrors `MaterialModel::owns_deformation_volume_state()` (1 = true,
    /// 0 = false). CPU's `estimate_particle_volumes`
    /// (`src/spacetime/solver/density.rs`) skips the raw kernel-mass
    /// density/volume gather for any material returning true (today:
    /// `NewtonianFluidMaterial`/`BinghamFluidMaterial`) and derives density
    /// and volume analytically from the material's clamped deformation
    /// gradient; this flag gives `g2p.wgsl` the same gate. Without it, the
    /// free-surface-biased kernel gather set water density to [0.0116, 0.358]
    /// against a rest density of 0.1 (`basic_fluids_gpu.rs`), outside the
    /// ~[0.05, 0.2] the clamped analytic formula allows
    /// (`.max(min_density).min(2*rest_density)`), and the spurious excursion
    /// fed the CFL acoustic term (∝ density^(eos_power-1)), pinning the
    /// substep count at its cap. Set via
    /// `self.owns_deformation_volume_state() as u32` in each material's
    /// `params()`, a per-material flag, so any material overriding that trait
    /// method gets the GPU behavior automatically.
    pub owns_deformation_volume_state: u32,
    /// Specific heat capacity `c_p`, J/(kg*K). 0 means the material has not
    /// declared one.
    ///
    /// A property of the matter, so it belongs here rather than in whichever
    /// solver happens to want it. `ThermalDiffusion` carries its own
    /// scene-wide `heat_capacity` for the diffusion stencil; this is the
    /// per-material value, and it is what lets dissipated work become a real
    /// temperature rise on the particle that absorbed it
    /// (`energy::thermodynamics::frictional_heating`).
    pub specific_heat_j_kg_k: f32,
    /// Explicit, always-zeroed reserved space: keeps the struct at a 16-byte
    /// multiple (112) with room for future flags.
    pub _pad: [u32; 2],
}

const _: () = assert!(core::mem::size_of::<MaterialParams>() == 112);

/// Flat material parameters for a single registered material slot.
///
/// Layout is a union — only the fields relevant to the constitutive model are filled;
/// all others are zero. `model` is the `ConstitutiveModel` discriminant and is always set.
///
/// 48 bytes, 16-byte aligned — directly uploadable to a GPU uniform buffer as
/// `array<MaterialParams, N>` indexed by `particle.material_id`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialParams {
    /// ConstitutiveModel discriminant. The shader reads this first to select the stress branch.
    pub model: u32,

    // --- Elastic (Neo-Hookean, Corotated, Snow) ---
    /// First Lamé parameter λ — controls bulk-like volumetric stiffness.
    pub lambda: f32,
    /// Second Lamé parameter µ — controls shear stiffness.
    pub mu: f32,

    // --- Snow plasticity (Stomakhin 2013 notation) ---
    /// Hardening coefficient ξ. Scales stiffness as h = exp(ξ(1−Jp)).
    pub hardening_xi: f32,
    /// Critical compression threshold θ_c. Singular values below (1−θ_c) are clamped.
    pub theta_compression: f32,
    /// Critical stretch threshold θ_s. Singular values above (1+θ_s) are clamped.
    pub theta_stretch: f32,

    // --- Fluid (Tait EOS + Newtonian viscosity) ---
    /// Reference density ρ₀ at rest. Tait EOS pressure is zero when ρ = ρ₀.
    pub rest_density: f32,
    /// Bulk modulus k in the Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. Use γ ≈ 7 for near-incompressible water.
    pub eos_power: f32,
    /// Dynamic viscosity coefficient µ_f. Scales deviatoric stress.
    pub dynamic_viscosity: f32,

    pub _pad0: f32,
    pub _pad1: f32, // 48 bytes total — 16-byte aligned
}

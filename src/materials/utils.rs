use glam::{Mat2, Vec2};

/// Floor applied to singular values before taking log — prevents ln(0).
/// All material `update_particle` implementations clamp σᵢ above this value.
pub(crate) const LOG_CLAMP: f32 = 1e-10;

/// Floor applied to det(F) = J before using it for volume/density.
/// Prevents divide-by-zero in stress and density computations.
/// All materials clamp J ≥ MIN_J after projection.
pub(crate) const MIN_J: f32 = 1e-6;

/// Compute Hencky (logarithmic) strains from SVD singular values.
///
/// ε_i = ln(|σ_i|), clamped above 1e-10 to avoid ln(0).
/// The absolute value preserves Hencky strain magnitudes when `svd2()` encodes
/// an inversion via a signed second singular value.
/// Used identically by VonMisesMaterial, SandMaterial, RankineMaterial.
#[inline(always)]
pub(crate) fn hencky_strains(sigma: Vec2) -> Vec2 {
    let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
    Vec2::new(sigma.x.ln(), sigma.y.ln())
}

/// Reconstruct a 2×2 deformation gradient from SVD factors and (possibly updated) singular values.
///
/// F = U · diag(sigma) · Vᵀ
#[inline(always)]
pub(crate) fn reconstruct_f(u: Mat2, sigma: Vec2, vt: Mat2) -> Mat2 {
    u * Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y)) * vt
}

/// Convert 2D principal Kirchhoff stresses back to Hencky strains (inverse of corotated elastic).
///
/// For corotated/Hencky elastic: τᵢ = (2µ+λ)·εᵢ + λ·ε_j  →  system inversion.
/// Inverse: ε = A⁻¹·τ where det(A) = 4µ(µ+λ).
#[inline(always)]
pub(crate) fn stress_to_hencky(tau: Vec2, lambda: f32, mu: f32) -> Vec2 {
    let det = 4.0 * mu * (mu + lambda);
    let a = 2.0 * mu + lambda;
    Vec2::new(
        (a * tau.x - lambda * tau.y) / det,
        (a * tau.y - lambda * tau.x) / det,
    )
}

/// 2D polar decomposition: returns the rotation R such that F = R·S.
///
/// Uses the analytical formula for 2×2 matrices (no SVD needed):
///   x = F₀₀+F₁₁, y = F₁₀−F₀₁, norm = √(x²+y²), R = [[x,−y],[y,x]]/norm
/// Returns Mat2::IDENTITY when F is near-singular (norm ≤ ε).
/// Used identically by CorotatedMaterial, SnowMaterial, VonMisesMaterial, SandMaterial.
pub fn polar_decomposition_2d(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    }
}

/// CFL timestep bound from elastic longitudinal wave speed c_P = √((λ+2µ)·h / ρ).
///
/// `hardening` = 1.0 for materials without hardening (elastic, sand, von Mises).
/// `hardening` = particle.hardening_scale for corotated/snow (stiffness grows on compression).
/// Returns f32::INFINITY when the material has zero or negative stiffness.
pub fn elastic_wave_dt(
    lambda: f32,
    mu: f32,
    hardening: f32,
    density: f32,
    min_density: f32,
    cell_width: f32,
    material_cfl: f32,
) -> f32 {
    let rho = density.max(min_density);
    let modulus = ((lambda + 2.0 * mu) * hardening).max(0.0);
    if modulus <= f32::EPSILON {
        return f32::INFINITY;
    }
    let c = (modulus / rho).sqrt();
    if c <= f32::EPSILON {
        return f32::INFINITY;
    }
    material_cfl * cell_width / c
}

/// Convert Young's modulus E and Poisson's ratio ν to Lamé parameters (λ, µ).
///
/// Valid for ν ∈ (−1, 0.5), E > 0. Matches the API used by sparkl and wgsparkl:
///   `ElasticCoefficients::from_young_modulus(E, nu)`
///
/// # Canonical values (from published MPM papers)
///
/// | Material       | E           | ν    | Source                     |
/// |----------------|-------------|------|----------------------------|
/// | sand (demo)    | 1.0×10⁵     | 0.20 | Klar 2016, sparkl basic2   |
/// | snow           | 1.4×10⁵     | 0.20 | Stomakhin 2013, MPM2D ref  |
/// | soft elastic   | 5.0×10⁶     | 0.20 | wgsparkl elasticity2       |
/// | soft tissue    | 1.0×10³     | 0.45 | typical MPM bio            |
///
/// Note: these are in whatever units your grid uses (not necessarily SI).
/// At emerge's default `grid_cell_size = 1.0`, use values that give
/// `sqrt((λ+2µ)/ρ) ≈ 10–60 cells/s` for interactive framerates.
pub fn lame_from_young(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    debug_assert!(young_modulus > 0.0, "Young's modulus must be positive");
    debug_assert!(
        poisson_ratio > -1.0 && poisson_ratio < 0.5,
        "Poisson's ratio must be in (-1, 0.5)"
    );
    let lambda =
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    let mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    (lambda, mu)
}

/// Convert SI Young's modulus and Poisson's ratio to emerge grid-unit Lamé parameters.
///
/// In the solver, velocity is in cells/s and stress is applied as:
///   `f_particle = vol_solver * sigma_solver * kernel`
/// The correct non-dimensionalization gives:
///   `λ_grid = λ_SI · dt² / (ρ₀ · dx²)`
///
/// Pair with `SolverConfig::earth()` and set `config.particle_mass =
/// rest_density_kg_m3 * (spacing * dx_meters).powi(2)` for a fully IRL-calibrated sim.
///
/// # Example — soft tissue (E ≈ 5 kPa, ν = 0.45, ρ = 1000 kg/m³, 1 cm/cell)
/// ```rust,no_run
/// use emerge::lame_from_si;
/// let (lambda, mu) = lame_from_si(5_000.0, 0.45, 1000.0, 0.01, 0.1);
/// // lambda ≈ 1552, mu ≈ 172 — ready for NeoHookeanMaterial or ViscoelasticMaterial
/// ```
pub fn lame_from_si(
    young_modulus_pa: f32,
    poisson_ratio: f32,
    rest_density_kg_m3: f32,
    dx_meters: f32,
    dt_seconds: f32,
) -> (f32, f32) {
    let (lambda_si, mu_si) = lame_from_young(young_modulus_pa, poisson_ratio);
    let scale = dt_seconds * dt_seconds / (rest_density_kg_m3 * dx_meters * dx_meters);
    (lambda_si * scale, mu_si * scale)
}

/// Convert SI gravity (m/s²) to solver units (grid cells / s²).
///
/// In the solver `v += gravity * sub_dt` where sub_dt is in real seconds,
/// so gravity must be in [cells/s²] = g_SI / dx_meters.
/// The `dt_seconds` parameter is unused but kept for API compatibility.
///
/// # Example — Earth gravity at 1 cm/cell
/// ```rust,no_run
/// use emerge::gravity_to_grid;
/// use glam::Vec2;
/// let g = gravity_to_grid(Vec2::new(0.0, -9.81), 0.01, 0.1);
/// // g ≈ Vec2::new(0.0, -981.0) cells/s²
/// ```
pub fn gravity_to_grid(g_si: glam::Vec2, dx_meters: f32, _dt_seconds: f32) -> glam::Vec2 {
    g_si / dx_meters
}


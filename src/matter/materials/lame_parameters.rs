//! Lamé parameters `lambda` and `mu`: the two constants of isotropic linear
//! elasticity, from the measured Young's modulus and Poisson's ratio.
//!
//! Source: the constitutive theory of isotropic elastic solids (Gurtin, Fried &
//! Anand 2010, part X); `lambda = E nu / ((1 + nu)(1 - 2 nu))`,
//! `mu = E / (2 (1 + nu))`.

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

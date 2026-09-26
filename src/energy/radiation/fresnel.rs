//! Fresnel reflectance: how much light bounces off an interface instead of
//! entering it.
//!
//! These are Maxwell's boundary conditions at a dielectric interface, reduced
//! to the unpolarized case. They live here rather than in
//! `energy::electromagnetics` for two reasons: this is light *transport*,
//! which is what `radiation` owns, and `electromagnetics` is behind the
//! `experimental` feature, so a renderer that depended on it would only have
//! correct reflectance in some builds. The refractive index itself is the
//! electromagnetic quantity, and `electromagnetics::MaterialProperties`
//! derives it from permittivity and permeability when that feature is on.
//!
//! Reference: C. Schlick, "An Inexpensive BRDF Model for Physically-based
//! Rendering", Computer Graphics Forum 13(3), 1994 -- the angular
//! approximation. The normal-incidence value it is anchored to is exact,
//! straight from the Fresnel equations.

/// Reflectance at normal incidence between two dielectrics.
///
/// `R0 = ((n1 - n2) / (n1 + n2))^2`, exact for unpolarized light hitting a
/// smooth interface head-on. Air to water (1.000 -> 1.333) gives 0.0204, and
/// air to window glass (1.000 -> 1.52) gives 0.0426 -- both standard values,
/// which is how a caller can tell this is wired correctly.
pub fn fresnel_r0_dielectric(n_outside: f32, n_inside: f32) -> f32 {
    if !n_outside.is_finite() || !n_inside.is_finite() {
        return 0.0;
    }
    let sum = n_outside + n_inside;
    if sum.abs() < 1.0e-12 {
        return 0.0;
    }
    let ratio = (n_outside - n_inside) / sum;
    ratio * ratio
}

/// Schlick's angular approximation to the Fresnel reflectance.
///
/// `R(theta) = R0 + (1 - R0) * (1 - cos theta)^5`, where `theta` is measured
/// from the surface normal. Exact at normal incidence, exact at grazing
/// (both go to 1), and within ~1% of the full Fresnel equations between --
/// which is why it is the standard choice in real-time rendering rather than
/// a shortcut taken here.
pub fn schlick_reflectance(r0: f32, cos_theta: f32) -> f32 {
    let cos_theta = cos_theta.clamp(0.0, 1.0);
    let one_minus = 1.0 - cos_theta;
    let pow5 = one_minus * one_minus * one_minus * one_minus * one_minus;
    (r0 + (1.0 - r0) * pow5).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published normal-incidence reflectances. Water at 2% and glass at 4%
    /// are numbers any optics text states, so reproducing them checks the
    /// formula against the outside world rather than against itself.
    #[test]
    fn normal_incidence_matches_published_values() {
        let water = fresnel_r0_dielectric(1.0, 1.333);
        assert!(
            (water - 0.0204).abs() < 0.001,
            "air/water R0 came out at {water:.4}, expected ~0.0204"
        );
        let glass = fresnel_r0_dielectric(1.0, 1.52);
        assert!(
            (glass - 0.0426).abs() < 0.001,
            "air/glass R0 came out at {glass:.4}, expected ~0.0426"
        );
    }

    /// Reflectance does not care which side of the interface you approach
    /// from -- the formula is symmetric in `n1` and `n2`.
    #[test]
    fn reflectance_is_symmetric_across_the_interface() {
        let entering = fresnel_r0_dielectric(1.0, 1.333);
        let leaving = fresnel_r0_dielectric(1.333, 1.0);
        assert!((entering - leaving).abs() < 1.0e-6);
    }

    /// Why a lake is a mirror at the far shore and clear at your feet:
    /// reflectance rises monotonically from R0 to 1 as the view flattens.
    #[test]
    fn reflectance_rises_from_r0_to_one_at_grazing() {
        let r0 = fresnel_r0_dielectric(1.0, 1.333);
        assert!((schlick_reflectance(r0, 1.0) - r0).abs() < 1.0e-6);
        assert!(schlick_reflectance(r0, 0.0) > 0.999);

        let mut previous = -1.0;
        for step in 0..=10u8 {
            let cos_theta = 1.0 - f32::from(step) / 10.0;
            let reflectance = schlick_reflectance(r0, cos_theta);
            assert!(
                reflectance > previous,
                "cos {cos_theta:.1} gave {reflectance:.4}, not above {previous:.4}"
            );
            previous = reflectance;
        }
    }
}

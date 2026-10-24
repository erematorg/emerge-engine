//! Cosserat (micropolar) granular kinematics: grain-scale rolling
//! resistance. Under plain Coulomb/DP friction a sand collapse never stops by
//! itself; the angle-of-repose literature attributes arrest to rolling
//! friction, not damping, and a scalar sliding-friction model has no
//! rolling.
//!
//! # Status: kinematics + elastic constitutive relation only
//! Not coupled into `DruckerPragerMaterial`'s stress and yield: a full
//! Cosserat coupling needs a second grid-level balance (angular momentum and
//! couple stress, `div(m) + e:σ = 0`), and a local relaxation in its place
//! would be a shortcut, not that balance. Implemented here: the
//! micro-curvature kinematics and the elastic couple-stress relation,
//! checked against the reference. The coupling is future work.
//!
//! # Citation
//! de Borst, R., Sabet, S.A. and Hageman, T. (2022), "Non-associated
//! Cosserat plasticity", International Journal of Mechanical Sciences,
//! 230, 107535, <https://doi.org/10.1016/j.ijmecsci.2022.107535> (open
//! access, CC-BY-NC-ND). Foundational theory: Cosserat & Cosserat (1909);
//! shear-band regularization application: de Borst (1991).
//!
//! # Equations (2D)
//! A Cosserat continuum carries a micro-rotation ω_c (a scalar in 2D,
//! rotation about the out-of-plane axis) not slaved to the antisymmetric
//! part of the velocity gradient. Its gradient is the micro-curvature:
//! ```text
//! κ = ∇ω_c        (κ_x = ∂ω_c/∂x, κ_y = ∂ω_c/∂y)
//! ```
//! work-conjugate to a couple-stress vector `m` (moment per unit area, the 2D
//! reduction of the 3D couple-stress tensor). The paper's planar reduction of
//! the elastic relation (eq. 36-38, after the out-of-plane terms cancel) is a
//! proportionality through an internal length `l` (grain size) and a coupling
//! modulus `alpha`:
//! ```text
//! m = alpha * l^2 * κ
//! ```
//! The paper puts the predicted shear-band width at roughly 15-20 times `l`,
//! so `l` is set by the grain diameter (the convention of
//! `GranularFluidityField`'s `grain_diameter_m`), not tuned.

/// Central-difference micro-curvature (`κ = ∇ω_c`) from a grid-scattered
/// micro-rotation field, with the column-major indexing (`idx =
/// x*grid_res+y`) of `energy::thermodynamics::stencil::laplacian_step`, so it
/// can reuse the same scatter/gather scaffolding.
///
/// Second-order central difference, `∂ω/∂x ≈ (ω(x+1,y) − ω(x−1,y)) / (2·dx)`,
/// one-sided at the domain edge (curvature has no ambient value to assume,
/// unlike temperature in `laplacian_step`'s Dirichlet boundary).
pub fn micro_curvature_2d(
    grid_omega: &[f32],
    grid_res: usize,
    x: usize,
    y: usize,
    dx: f32,
) -> glam::Vec2 {
    let c = x * grid_res + y;
    let kappa_x = if x > 0 && x + 1 < grid_res {
        (grid_omega[c + grid_res] - grid_omega[c - grid_res]) / (2.0 * dx)
    } else if x + 1 < grid_res {
        (grid_omega[c + grid_res] - grid_omega[c]) / dx
    } else if x > 0 {
        (grid_omega[c] - grid_omega[c - grid_res]) / dx
    } else {
        0.0
    };
    let kappa_y = if y > 0 && y + 1 < grid_res {
        (grid_omega[c + 1] - grid_omega[c - 1]) / (2.0 * dx)
    } else if y + 1 < grid_res {
        (grid_omega[c + 1] - grid_omega[c]) / dx
    } else if y > 0 {
        (grid_omega[c] - grid_omega[c - 1]) / dx
    } else {
        0.0
    };
    glam::Vec2::new(kappa_x, kappa_y)
}

/// Elastic couple-stress relation (de Borst, Sabet & Hageman 2022, planar
/// reduction of eq. 36-38): `m = alpha * l^2 * κ`.
///
/// `length_scale_m` is the internal length `l`, physically the grain
/// diameter (as `GranularFluidityConfig::grain_diameter_m`), not a fitting
/// knob. `coupling_modulus` is `alpha`, in stress units, so `m` comes out in
/// stress·length, a moment per unit area.
pub fn elastic_couple_stress_2d(
    curvature: glam::Vec2,
    coupling_modulus: f32,
    length_scale_m: f32,
) -> glam::Vec2 {
    coupling_modulus * length_scale_m * length_scale_m * curvature
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    /// For ω_c(x,y) = 2x + 3y (a plane), central differences reproduce the
    /// gradient (2, 3) exactly at every interior point (exact for degree 1).
    #[test]
    fn curvature_matches_analytic_gradient_of_a_linear_field() {
        const RES: usize = 8;
        const DX: f32 = 0.5;
        let mut grid = vec![0.0f32; RES * RES];
        for x in 0..RES {
            for y in 0..RES {
                let world_x = x as f32 * DX;
                let world_y = y as f32 * DX;
                grid[x * RES + y] = 2.0 * world_x + 3.0 * world_y;
            }
        }
        for x in 1..RES - 1 {
            for y in 1..RES - 1 {
                let kappa = micro_curvature_2d(&grid, RES, x, y, DX);
                assert!(
                    (kappa - Vec2::new(2.0, 3.0)).length() < 1e-4,
                    "at ({x},{y}): kappa={kappa:?}, expected (2.0, 3.0)"
                );
            }
        }
    }

    /// A uniform (constant) micro-rotation field has zero curvature
    /// everywhere -- necessary check: no differential rotation, no
    /// micro-curvature, matching κ=∇ω_c's own definition directly.
    #[test]
    fn uniform_field_has_zero_curvature() {
        const RES: usize = 6;
        let grid = vec![1.2345f32; RES * RES];
        for x in 0..RES {
            for y in 0..RES {
                let kappa = micro_curvature_2d(&grid, RES, x, y, 0.1);
                assert!(kappa.length() < 1e-6, "at ({x},{y}): kappa={kappa:?}");
            }
        }
    }

    /// Checks the cited formula: m = alpha * l^2 * κ.
    #[test]
    fn elastic_couple_stress_matches_the_cited_formula_directly() {
        let kappa = Vec2::new(0.02, -0.015);
        let alpha = 5.0e6f32; // real-magnitude modulus, Pa
        let l = 0.3e-3f32; // real grain diameter, m (same value NGF uses)
        let m = elastic_couple_stress_2d(kappa, alpha, l);
        let expected = alpha * l * l * kappa;
        assert!((m - expected).length() < 1e-9 * expected.length().max(1.0));
    }

    /// Zero curvature must give zero couple-stress -- a necessary
    /// property of the linear elastic relation (no differential rotation
    /// between neighboring material points means no torque resisting it).
    #[test]
    fn zero_curvature_gives_zero_couple_stress() {
        let m = elastic_couple_stress_2d(Vec2::ZERO, 5.0e6, 0.3e-3);
        assert_eq!(m, Vec2::ZERO);
    }

    /// Scale check: at a 0.3 mm grain and a modulus of the order of the sand
    /// presets' Young's modulus (~1e5-1e7 Pa), a large but plausible curvature
    /// (~1 rad/m) gives a couple stress far below the Cauchy stress scale
    /// (~alpha): couple stresses set the localization width, not the bulk
    /// stress, as the paper frames them.
    #[test]
    fn couple_stress_is_a_small_correction_at_real_grain_scale() {
        let kappa = Vec2::new(1.0, 0.0);
        let alpha = 1.0e6f32;
        let l = 0.3e-3f32;
        let m = elastic_couple_stress_2d(kappa, alpha, l);
        assert!(
            m.length() < alpha * 1e-3,
            "couple-stress {m:?} should be a small correction relative to alpha={alpha}"
        );
    }
}

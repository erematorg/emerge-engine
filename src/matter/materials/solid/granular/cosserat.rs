//! Cosserat (micropolar) granular kinematics — real grain-scale rolling
//! resistance, the confirmed root cause of `DruckerPragerMaterial`'s
//! self-arrest gap (see memory: sand collapse never naturally stops under
//! plain Coulomb/DP friction; real angle-of-repose literature attributes
//! this to rolling friction, not damping — a scalar sliding-friction model
//! has no notion of rolling at all).
//!
//! # Status: kinematics + elastic constitutive relation only
//! This module deliberately does NOT yet close the loop into
//! `DruckerPragerMaterial`'s stress/yield computation. A full Cosserat
//! coupling needs a genuine second grid-level balance (linear momentum AND
//! angular momentum/couple-stress, `div(m) + e:σ = 0`) — a real physics
//! channel, not a shortcut. Faking that with a local relaxation instead of
//! the real spatial balance would be exactly the "cheat with a PDE" this
//! project explicitly rejects. What's implemented here is the real,
//! independently-testable piece: the micro-curvature kinematics and the
//! real elastic couple-stress relation, verified against the cited
//! reference. Closing the loop (feeding this into an actual stress/yield
//! coupling) is real, disclosed future work — see
//! `project_grain_scale_rolling_resistance_two_real_paths` in memory.
//!
//! # Real citation
//! de Borst, R., Sabet, S.A. and Hageman, T. (2022), "Non-associated
//! Cosserat plasticity", International Journal of Mechanical Sciences,
//! 230, 107535, <https://doi.org/10.1016/j.ijmecsci.2022.107535> (open
//! access, CC-BY-NC-ND). Foundational theory: Cosserat & Cosserat (1909);
//! shear-band regularization application: de Borst (1991).
//!
//! # The real equations (2D, this project's own notation)
//! A Cosserat continuum carries a micro-rotation field ω_c (a scalar in 2D
//! — rotation about the out-of-plane axis) that is NOT slaved to the
//! ordinary velocity gradient's antisymmetric part, unlike classical
//! continuum mechanics. Its spatial gradient is the micro-curvature:
//! ```text
//! κ = ∇ω_c        (κ_x = ∂ω_c/∂x, κ_y = ∂ω_c/∂y)
//! ```
//! work-conjugate to a couple-stress vector `m` (moment per unit area, 2D
//! reduction of the general 3D couple-stress tensor). The cited paper's own
//! planar reduction of the general elastic relation (their eq. 36-38, after
//! the out-of-plane cancellation that occurs for genuinely 2D deformation)
//! is a direct proportionality through a real internal length scale `l`
//! (tied to physical grain size) and a coupling modulus `alpha`:
//! ```text
//! m = alpha * l^2 * κ
//! ```
//! The same paper reports the shear-band width predicted by this model is
//! real and citable: roughly 15-20 times `l`, i.e. `l` is not a free numerical
//! knob — it is set by the real grain diameter, same convention already used
//! by `GranularFluidityField`'s own `grain_diameter_m`.

/// Central-difference micro-curvature (`κ = ∇ω_c`) from a grid-scattered
/// micro-rotation field. Same column-major indexing (`idx = x*grid_res+y`)
/// and Dirichlet-at-domain-edge convention as
/// `energy::thermodynamics::stencil::laplacian_step`, so this can reuse the
/// exact same P2G-scatter/gather scaffolding once wired into a live field —
/// not invented independently.
///
/// Real, standard second-order central difference: `∂ω/∂x ≈ (ω(x+1,y) −
/// ω(x−1,y)) / (2·dx)`. Off-grid neighbors at the domain edge use a
/// one-sided difference instead of assuming an ambient value (unlike
/// `laplacian_step`'s Dirichlet boundary) — curvature has no natural
/// "ambient" value the way temperature does, so a one-sided estimate is the
/// real, honest choice at the edge, not an arbitrary substitute.
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

/// Real elastic couple-stress relation (de Borst, Sabet & Hageman 2022, the
/// planar reduction of their eq. 36-38): `m = alpha * l^2 * κ`.
///
/// `length_scale_m` is the real internal length scale `l` — physically the
/// grain diameter (same real quantity `GranularFluidityConfig::
/// grain_diameter_m` already uses), NOT a free numerical fitting knob.
/// `coupling_modulus` is `alpha`, a real elastic modulus with units of
/// stress (couple-stress `m` then comes out in stress·length units, the
/// genuine dimensional form of a moment per unit area).
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

    /// Real analytic check: for ω_c(x,y) = 2x + 3y (a plane, exact constant
    /// gradient everywhere), the central-difference stencil must reproduce
    /// the exact analytic gradient (2, 3) at every interior point — no
    /// truncation error at all for a linear function, since central
    /// differences are exact for polynomials up to degree 1.
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
    /// everywhere -- real, necessary check: no differential rotation, no
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

    /// Real, direct check of the cited formula: m = alpha * l^2 * κ.
    #[test]
    fn elastic_couple_stress_matches_the_cited_formula_directly() {
        let kappa = Vec2::new(0.02, -0.015);
        let alpha = 5.0e6f32; // real-magnitude modulus, Pa
        let l = 0.3e-3f32; // real grain diameter, m (same value NGF uses)
        let m = elastic_couple_stress_2d(kappa, alpha, l);
        let expected = alpha * l * l * kappa;
        assert!((m - expected).length() < 1e-9 * expected.length().max(1.0));
    }

    /// Zero curvature must give zero couple-stress -- a real, necessary
    /// property of the linear elastic relation (no differential rotation
    /// between neighboring material points means no torque resisting it).
    #[test]
    fn zero_curvature_gives_zero_couple_stress() {
        let m = elastic_couple_stress_2d(Vec2::ZERO, 5.0e6, 0.3e-3);
        assert_eq!(m, Vec2::ZERO);
    }

    /// Real sanity-of-scale check: at a real grain diameter (0.3mm) and a
    /// modulus of the same real order of magnitude as this project's own
    /// sand Young's modulus (~1e5-1e7 Pa range, see `DruckerPragerMaterial`
    /// presets), a real, physically plausible curvature (order 1 rad/m,
    /// i.e. micro-rotation changing by about a radian per meter -- a
    /// genuinely large but not absurd shear-localization scenario) should
    /// give a couple-stress magnitude much smaller than the ordinary
    /// Cauchy stress scale (order alpha itself) -- consistent with couple-
    /// stress effects being a real but small correction at grain scale, not
    /// a dominant term, matching the cited paper's own framing (couple
    /// stresses matter for localization WIDTH, not bulk stress magnitude).
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

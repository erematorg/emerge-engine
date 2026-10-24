//! Rankine tensile damage: how much a brittle material has cracked once its
//! largest principal stress passes its tensile strength.
//!
//! Source: continuum damage mechanics, Lemaitre & Chaboche, *Mechanics of
//! Solid Materials* (1990), for the residual strength floor below.

use glam::{Mat2, Vec2};

use crate::matter::kinematics::hencky_strain::hencky_strains;
use crate::tensor_algebra::svd::svd2;

/// Floor on Rankine's exponentially-softened effective tensile strength, as a
/// fraction of the virgin `tensile_strength`. Without this floor, `t_eff` decays
/// toward zero as damage grows, so ANY sustained cyclic stress eventually exceeds
/// it every step by a growing margin -- an unbounded damage ratchet with no
/// resting state. Real quasi-brittle/ductile damage models retain nonzero residual capacity
/// after yield rather than decaying to zero (Lemaitre & Chaboche, "Mechanics of
/// Solid Materials," 1990 -- continuum damage mechanics caps effective
/// stiffness/strength at a small nonzero residual specifically to keep the
/// damage variable bounded under sustained loading). 5% is a conservative
/// residual: still lets damage climb substantially before saturating, but
/// guarantees a stress level cyclic loading can eventually stay under.
pub(crate) const RANKINE_MIN_RESIDUAL_TENSILE_FRACTION: f32 = 0.05;

/// Damage value at which `t_eff` has already reached its residual floor -- i.e.
/// `tensile_strength * exp(-softening_rate * d) == tensile_strength * RESIDUAL_FRACTION`,
/// solved for d. Past this point, MORE damage would not lower `t_eff` any further
/// (it's already floored), so there is nothing left for the number to usefully
/// track: sustained cyclic loading that exceeds even the residual strength would
/// otherwise still accumulate damage forever, linearly, once the floor was hit --
/// the floor alone stops the exponential runaway but not an unbounded linear
/// climb. Clamping `prior_damage` at this saturation point is a no-op on future
/// stress behavior (t_eff there is identical to t_eff at any higher damage value)
/// and gives the accumulator a rupture point instead of an arbitrary cutoff.
/// `softening_rate <= 0` means no softening at all (hard cutoff, doc'd on
/// `RankineMaterial::softening_rate`) -- saturation point is +infinity, i.e. no cap.
#[inline]
pub(crate) fn rankine_damage_saturation_point(softening_rate: f32) -> f32 {
    if softening_rate <= 0.0 {
        f32::INFINITY
    } else {
        -RANKINE_MIN_RESIDUAL_TENSILE_FRACTION.ln() / softening_rate
    }
}

/// Convert 2D principal Kirchhoff stresses back to Hencky strains (inverse of corotated elastic).
///
/// For corotated/Hencky elastic: τᵢ = (2µ+λ)·εᵢ + λ·ε_j  →  system inversion.
/// Inverse: ε = A⁻¹·τ where det(A) = 4µ(µ+λ).
#[inline]
pub(crate) fn stress_to_hencky(tau: Vec2, lambda: f32, mu: f32) -> Vec2 {
    let det = 4.0 * mu * (mu + lambda);
    let a = 2.0 * mu + lambda;
    Vec2::new(
        (a * tau.x - lambda * tau.y) / det,
        (a * tau.y - lambda * tau.x) / det,
    )
}

/// Rankine (maximum principal stress) tensile-damage estimate -- the same failure
/// criterion `RankineMaterial` uses internally (max principal Kirchhoff stress vs.
/// an exponentially-softening tensile threshold), exposed as a standalone read-only
/// analysis function so ANY material can track real structural damage without
/// adopting Rankine's full constitutive/return-mapping model as its own stress
/// response. Does not modify `deformation_gradient` -- purely observational, safe
/// to call alongside a material's own (unrelated) `update_particle`, e.g. a
/// muscle-actuated `NeoHookeanMaterial` that still needs a damage/health
/// signal without giving up its own stress model.
///
/// `lambda`/`mu` should be the SAME Lamé parameters the calling material already
/// uses for its own elastic response -- this reads the strain state via the
/// same Hencky-strain path Rankine's own `update_particle` computes from, just
/// without writing the projected state back into `F`.
pub fn rankine_damage_estimate(
    deformation_gradient: Mat2,
    lambda: f32,
    mu: f32,
    tensile_strength: f32,
    softening_rate: f32,
    prior_damage: f32,
) -> f32 {
    let (_, sigma, _) = svd2(deformation_gradient);
    let eps = hencky_strains(sigma);
    let a = 2.0 * mu + lambda;
    let tau = Vec2::new(a * eps.x + lambda * eps.y, lambda * eps.x + a * eps.y);

    let t_eff = (tensile_strength * (-softening_rate * prior_damage).exp())
        .max(tensile_strength * RANKINE_MIN_RESIDUAL_TENSILE_FRACTION);
    let tau_proj = Vec2::new(tau.x.min(t_eff), tau.y.min(t_eff));
    if tau_proj == tau {
        return prior_damage; // within the tensile limit, no new damage
    }

    let eps_proj = stress_to_hencky(tau_proj, lambda, mu);
    (prior_damage + (eps - eps_proj).length()).min(rankine_damage_saturation_point(softening_rate))
}

#[cfg(test)]
mod rankine_damage_estimate_tests {
    use super::*;

    #[test]
    fn no_damage_within_tensile_limit() {
        let f = Mat2::from_cols(Vec2::new(1.01, 0.0), Vec2::new(0.0, 1.0));
        let damage = rankine_damage_estimate(f, 1000.0, 1000.0, 1.0e6, 1.0, 0.0);
        assert_eq!(
            damage, 0.0,
            "tiny strain must stay under a huge tensile threshold"
        );
    }

    #[test]
    fn damage_accumulates_past_tensile_limit() {
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let damage = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, 0.0);
        assert!(
            damage > 0.0,
            "large tensile strain must accumulate real damage"
        );
    }

    #[test]
    fn damage_never_decreases_across_repeated_overload() {
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let mut damage = 0.0;
        for _ in 0..5 {
            let next = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, damage);
            assert!(
                next >= damage,
                "damage must be monotonically non-decreasing"
            );
            damage = next;
        }
        assert!(damage > 0.0);
    }

    #[test]
    fn does_not_mutate_deformation_gradient() {
        // Purely observational -- caller's F is untouched, function only reads it.
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let f_before = f;
        let _ = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, 0.0);
        assert_eq!(f, f_before);
    }
}

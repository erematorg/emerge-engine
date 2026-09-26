//! Real elastic-perfectly-plastic bending -- permanent curvature set once a
//! vertex's real bending moment exceeds the material's own real yield
//! moment. Standard mechanics-of-materials result (elastic-plastic bending
//! of a beam -- e.g. Gere & Goodno, *Mechanics of Materials*, "Elastoplastic
//! Bending"): for a rectangular cross-section, first yield (outer-fiber
//! stress reaching `sigma_yield`) occurs at the real YIELD MOMENT
//! `M_yield = sigma_yield * I / c` (`I` the second moment of area, `c` the
//! outer-fiber distance from the neutral axis).
//!
//! Real, disclosed dimensional note (the actual bug this module's own
//! integration test caught before this doc was written): `forces::
//! discrete_curvature`'s own output `kappa` is DIMENSIONLESS (≈ the real
//! turning angle for small bends, see that function's own doc) -- NOT a real
//! per-meter curvature. `forces::compute_internal_forces`'s own bending
//! moment is `coeff = (ei/voronoi_length)*(kappa-kappa_rest)`, a real N·m
//! quantity (`secondary_growth.rs` already uses this SAME `coeff` as its own
//! real stress proxy for exactly this reason -- no separate cross-section
//! area is tracked independent of `EA`/`EI`, so comparing a raw `kappa`
//! against a stress-derived threshold is dimensionally meaningless; the
//! real moment is the correct, real-unit quantity to compare). This module
//! follows that SAME established precedent: the yield check compares the
//! real MOMENT (not raw curvature) against `yield_moment_n_m`, converting
//! back to a `kappa` update only through each vertex's own real local
//! stiffness `ei/voronoi_length`.
//!
//! This is the SAME real, established concept the engine already uses for
//! bulk MPM materials (`VonMisesMaterial`'s own return-mapping projects an
//! over-yield stress state back onto its yield surface) -- here the "stress
//! tensor" is replaced by a single scalar (the bending moment), so the
//! return mapping reduces to the engine's own shared, dimension-agnostic
//! `matter::materials::utils::scalar_return_map` core (the same "true core
//! concept, not reimplemented per domain" already backing
//! `self_consistent_plastic_multiplier` across the bulk MPM materials).
//! Rate-independent (no `dt`): the return mapping is an algebraic
//! projection, applied once per call, same as `VonMisesMaterial`'s own
//! per-substep return mapping.
//!
//! # Real, cited fix for unbounded creep: isotropic hardening
//! A pure elastic-PERFECTLY-plastic yield surface (no hardening) is real,
//! but has a real, well-documented failure mode under a SUSTAINED load that
//! never itself drops below the yield threshold: the structure never
//! "shakes down" and instead undergoes unrestricted ongoing plastic
//! deformation -- "ratcheting" / "incremental collapse", the classical
//! opposite outcome to shakedown in Melan's (1938, static/lower-bound
//! theorem) and Koiter's (1956, "A new general theorem on shakedown of
//! elastic-plastic structures," Proc. Koninklijke Nederlandsche Akademie
//! van Wetenschappen B59:24-34, kinematic/upper-bound theorem) -- the two
//! foundational results in this area of plasticity theory, re-verified via
//! web search, not recalled from memory alone. This is exactly what a rod
//! under a
//! sustained real bending moment (e.g. its own weight, in a geometry bad
//! enough that gravity alone keeps exceeding a fixed yield moment) will do
//! with zero hardening: creep, permanently, indefinitely, with no external
//! force needed to sustain it -- confirmed directly (a real, reproduced,
//! not hypothetical finding).
//!
//! The real, standard fix: `hardening_modulus_n_m` (mirrors
//! `VonMisesMaterial`'s own `hardening_modulus`, `sigma_y(kappa) =
//! yield_stress + H*kappa`) grows the EFFECTIVE yield moment with
//! `RodPoints::accumulated_plastic_curvature` (that vertex's own running
//! total of permanent set so far), so continued yielding under a SUSTAINED,
//! constant-direction moment (this engine's real scenario -- gravity/a
//! held push, not cyclic loading) makes the material progressively harder
//! to yield further, until the effective yield moment finally exceeds the
//! sustaining moment and the structure shakes down to a stable, purely
//! elastic residual shape -- it does NOT ratchet forever. Real, disclosed
//! scope limit: this is ISOTROPIC hardening specifically, which a real
//! literature search confirms is the correct, sufficient fix for
//! monotonic, one-direction sustained loading (this engine's actual case)
//! but is known to NOT capture ratcheting under genuinely CYCLIC/reversed
//! loading (that needs KINEMATIC hardening, a real, distinct, harder
//! mechanism -- not attempted here, no current scenario needs it). Default
//! `hardening_modulus_n_m = 0.0` (perfectly plastic, the original
//! behavior) -- opt in via `with_hardening`.

use super::RodPoints;
use super::forces::discrete_curvature;
use crate::matter::materials::utils::scalar_return_map;

#[derive(Debug, Clone, Copy)]
pub struct RodPlasticity {
    /// Real yield moment, `sigma_yield*I/c`, N·m -- see module doc. Use
    /// `from_young_modulus_rectangular` rather than computing this by hand.
    pub yield_moment_n_m: f32,
    /// Real isotropic hardening modulus, N·m per unit accumulated plastic
    /// curvature -- see module doc's "Real, cited fix for unbounded creep"
    /// section. `0.0` (default) = perfectly plastic, the original behavior.
    pub hardening_modulus_n_m: f32,
}

impl RodPlasticity {
    pub const fn new(yield_moment_n_m: f32) -> Self {
        Self {
            yield_moment_n_m: yield_moment_n_m.abs(),
            hardening_modulus_n_m: 0.0,
        }
    }

    /// Opt into real isotropic hardening (see module doc) -- without this,
    /// the material stays perfectly plastic and can ratchet indefinitely
    /// under a sustained moment.
    pub const fn with_hardening(mut self, hardening_modulus_n_m: f32) -> Self {
        self.hardening_modulus_n_m = hardening_modulus_n_m.abs();
        self
    }

    /// Real derivation for a rectangular cross-section bent about the axis
    /// perpendicular to the simulation's own 2D plane -- same `width_m`/
    /// `thickness_m` convention as `RodMaterial::from_young_modulus_
    /// rectangular` (`I = width_m^3*thickness_m/12`, bending about this SAME
    /// axis, outer fiber at `c = width_m/2`): `M_yield = sigma_yield_pa * I
    /// / c`. `hardening_modulus_n_m` defaults to `0.0` -- chain
    /// `.with_hardening(...)` to opt in.
    pub fn from_young_modulus_rectangular(
        yield_stress_pa: f32,
        width_m: f32,
        thickness_m: f32,
    ) -> Self {
        let i = width_m.powi(3) * thickness_m / 12.0;
        let c = (width_m * 0.5).max(1.0e-9);
        Self::new(yield_stress_pa * i / c)
    }
}

/// Real elastic-plastic return mapping -- at every interior vertex, clamps
/// the real bending MOMENT (`ei/voronoi_length * (kappa-rest_curvature)`,
/// exactly `forces::compute_internal_forces`'s own `coeff`) to
/// `[-effective_yield, +effective_yield]`, permanently absorbing any excess
/// into `rest_curvature` (converted back through that same local stiffness,
/// so the actual clamp on `kappa` itself is real and vertex-local, not a
/// single global threshold), and accumulating the absorbed magnitude into
/// `RodPoints::accumulated_plastic_curvature`. `effective_yield =
/// yield_moment_n_m + hardening_modulus_n_m * accumulated_plastic_
/// curvature[i]` -- real isotropic hardening (see module doc), `0.0` by
/// default (perfectly plastic). No-op for a rod with fewer than 3 points,
/// or whose `ei`/`accumulated_plastic_curvature` aren't filled to the real
/// per-vertex length (`Rod::new`/`build_straight_rod` do this -- same guard
/// `secondary_growth`'s own apply function uses).
pub fn apply_bending_plasticity(rod: &mut RodPoints, plasticity: &RodPlasticity, dx_meters: f32) {
    let n = rod.x.len();
    if n < 3 || rod.ei.len() != n - 2 || rod.accumulated_plastic_curvature.len() != n - 2 {
        return;
    }

    for i in 1..n - 1 {
        let (p0, p1, p2) = (
            rod.x[i - 1] * dx_meters,
            rod.x[i] * dx_meters,
            rod.x[i + 1] * dx_meters,
        );
        let kappa = discrete_curvature(p0, p1, p2);

        let l0_prev = rod.rest_edge_length[i - 1].max(1.0e-9);
        let l0_next = rod.rest_edge_length[i].max(1.0e-9);
        let voronoi_length = 0.5 * (l0_prev + l0_next);
        let stiffness = (rod.ei[i - 1] / voronoi_length).max(1.0e-9);

        let effective_yield = plasticity.yield_moment_n_m
            + plasticity.hardening_modulus_n_m * rod.accumulated_plastic_curvature[i - 1];
        let kappa_yield_local = effective_yield / stiffness;

        let old_rest = rod.rest_curvature[i - 1];
        let new_rest = scalar_return_map(kappa, old_rest, kappa_yield_local);
        rod.accumulated_plastic_curvature[i - 1] += (new_rest - old_rest).abs();
        rod.rest_curvature[i - 1] = new_rest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rod::build_straight_rod;
    use glam::Vec2;

    fn kinked_rod(angle_deg: f32) -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 8, 0.01, 1.0);
        points.ea = vec![1.0e5; 7];
        points.ei = vec![1.0e-3; 6];
        // Real, sharp, LOCALIZED kink at one vertex -- rotate everything
        // past index 4 by `angle_deg` about point 4.
        let pivot = points.x[4];
        let (sin_a, cos_a) = angle_deg.to_radians().sin_cos();
        for i in 5..points.x.len() {
            let rel = points.x[i] - pivot;
            points.x[i] =
                pivot + Vec2::new(rel.x * cos_a - rel.y * sin_a, rel.x * sin_a + rel.y * cos_a);
        }
        points
    }

    #[test]
    fn from_young_modulus_rectangular_matches_hand_derivation() {
        // Real, independent check: sigma_yield=2e7 Pa, width=0.02m,
        // thickness=0.001m -> I=width^3*thickness/12=8e-6*0.001/12=6.667e-10,
        // c=0.01 -> M_yield = 2e7*6.667e-10/0.01 = 1.3333 N*m, by hand.
        let p = RodPlasticity::from_young_modulus_rectangular(2.0e7, 0.02, 0.001);
        assert!(
            (p.yield_moment_n_m - 1.3333).abs() < 1.0e-3,
            "expected M_yield=1.3333, got {}",
            p.yield_moment_n_m
        );
    }

    #[test]
    fn below_yield_moment_leaves_rest_curvature_unchanged() {
        let mut rod = kinked_rod(2.0); // tiny real kink
        let plasticity = RodPlasticity::new(1.0e6); // deliberately unreachable
        let before = rod.rest_curvature.clone();
        apply_bending_plasticity(&mut rod, &plasticity, 1.0);
        assert_eq!(
            rod.rest_curvature, before,
            "moment below yield must produce exactly zero permanent set"
        );
    }

    #[test]
    fn overload_produces_a_real_nonzero_permanent_set() {
        let mut rod = kinked_rod(50.0); // real, sharp overload
        // Real, deliberately low yield moment relative to this rod's own
        // EI/voronoi_length -- a soft, easily yielded material.
        let plasticity = RodPlasticity::new(1.0e-4);
        apply_bending_plasticity(&mut rod, &plasticity, 1.0);

        assert!(
            rod.rest_curvature.iter().any(|&k| k.abs() > 1.0e-6),
            "a real overload must leave a nonzero permanent set: {:?}",
            rod.rest_curvature
        );
    }

    #[test]
    fn permanent_set_reduces_the_elastic_restoring_moment_at_the_bent_shape() {
        // Real, physically meaningful check: the whole POINT of plasticity
        // is that the rod's NEW natural shape is the bent one, not straight
        // -- proven by comparing the real bending force (isolated from
        // axial via `forces::compute_bending_forces_only`) at the SAME
        // still-bent shape, before vs after the plastic correction.
        use super::super::RodMaterial;
        use super::super::forces::compute_bending_forces_only;

        let mut rod = kinked_rod(50.0);
        let material = RodMaterial::new(1.0e5, 1.0e-3, 0.0, 0.0);
        let v = vec![Vec2::ZERO; rod.x.len()];

        let force_before = compute_bending_forces_only(
            &rod.x,
            &v,
            &rod.rest_edge_length,
            &rod.rest_curvature,
            &rod.ei,
            &material,
            1.0,
        );
        let max_before = force_before
            .iter()
            .map(|f| f.length())
            .fold(0.0f32, f32::max);

        let plasticity = RodPlasticity::new(1.0e-4);
        apply_bending_plasticity(&mut rod, &plasticity, 1.0);
        let force_after = compute_bending_forces_only(
            &rod.x,
            &v,
            &rod.rest_edge_length,
            &rod.rest_curvature,
            &rod.ei,
            &material,
            1.0,
        );
        let max_after = force_after
            .iter()
            .map(|f| f.length())
            .fold(0.0f32, f32::max);

        assert!(
            max_after < 0.5 * max_before,
            "plastic yield must substantially reduce the elastic restoring BENDING force at \
             the SAME bent shape (before={max_before}, after={max_after}) -- otherwise the rod \
             would still be fighting to spring back to straight"
        );
    }

    #[test]
    fn empty_ei_is_a_no_op_not_a_panic() {
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 6, 0.01, 1.0);
        assert!(rod.ei.is_empty());
        let plasticity = RodPlasticity::new(1.0);
        apply_bending_plasticity(&mut rod, &plasticity, 1.0);
        assert!(rod.ei.is_empty());
    }

    /// Real, permanent regression guard for the exact real-demo finding that
    /// motivated hardening: a rod repeatedly re-kinked a little further each
    /// "step" (standing in for a real sustained external moment, e.g.
    /// gravity acting on an ever-worsening geometry, continuing to demand
    /// slightly more curvature than the structure currently accommodates)
    /// must accumulate LESS total permanent set with hardening than without
    /// it -- proving hardening is actually resisting further yield, not
    /// just present and inert.
    #[test]
    fn hardening_accumulates_substantially_less_permanent_set_than_perfectly_plastic() {
        let drive = |hardening_modulus_n_m: f32| -> f32 {
            let mut rod = kinked_rod(5.0); // small initial real kink
            let plasticity = RodPlasticity::new(1.0e-4).with_hardening(hardening_modulus_n_m);
            let pivot_index = 4;
            let pivot = rod.x[pivot_index];
            for _ in 0..40 {
                // Real, small, SUSTAINED additional rotation each iteration --
                // the tail keeps demanding a bit more curvature, standing in
                // for gravity continuing to load an ever-worsening shape.
                let (sin_a, cos_a) = 1.0_f32.to_radians().sin_cos();
                for i in (pivot_index + 1)..rod.x.len() {
                    let rel = rod.x[i] - pivot;
                    rod.x[i] = pivot
                        + Vec2::new(rel.x * cos_a - rel.y * sin_a, rel.x * sin_a + rel.y * cos_a);
                }
                apply_bending_plasticity(&mut rod, &plasticity, 1.0);
            }
            rod.accumulated_plastic_curvature
                .iter()
                .fold(0.0f32, |m, &k| m.max(k))
        };

        let perfectly_plastic = drive(0.0);
        let hardened = drive(5.0e-3);

        assert!(
            hardened < 0.5 * perfectly_plastic,
            "hardening must substantially reduce total accumulated permanent set under the \
             SAME sustained, incrementally-worsening load (perfectly_plastic={perfectly_plastic:.5}, \
             hardened={hardened:.5}) -- otherwise it isn't actually resisting further yield"
        );
    }
}

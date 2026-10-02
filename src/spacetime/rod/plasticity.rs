//! Elastic-perfectly-plastic bending: permanent curvature set once a
//! vertex's bending moment exceeds the yield moment. For a rectangular
//! section, first yield (outer-fibre stress at `sigma_yield`) occurs at
//! `M_yield = sigma_yield * I / c` (`I` the second moment of area, `c` the
//! outer-fibre distance from the neutral axis; Gere & Goodno, *Mechanics of
//! Materials*, "Elastoplastic Bending").
//!
//! `forces::discrete_curvature` returns a dimensionless `kappa` (about the
//! turning angle for small bends), not a curvature per metre, so the yield
//! check compares the moment `coeff = (ei/voronoi_length)*(kappa-kappa_rest)`
//! (N·m, from `forces::compute_internal_forces`, the stress proxy
//! `secondary_growth.rs` uses too) with `yield_moment_n_m`, converting back
//! to `kappa` through each vertex's `ei/voronoi_length`.
//!
//! The return mapping of `VonMisesMaterial` with a scalar in place of the
//! stress tensor, through `matter::materials::utils::scalar_return_map`
//! (the core behind `self_consistent_plastic_multiplier`). Rate-independent
//! (no `dt`): an algebraic projection per call.
//!
//! # Isotropic hardening against unbounded creep
//! Perfect plasticity under a sustained load that stays above yield never
//! shakes down: it ratchets, deforming without end (incremental collapse,
//! the opposite of shakedown in Melan's 1938 static theorem and Koiter's
//! 1956 kinematic theorem, "A new general theorem on shakedown of
//! elastic-plastic structures," Proc. Koninklijke Nederlandsche Akademie van
//! Wetenschappen B59:24-34). A rod whose own weight keeps exceeding a fixed
//! yield moment creeps without end.
//!
//! `hardening_modulus_n_m` (as `VonMisesMaterial`'s `hardening_modulus`,
//! `sigma_y(kappa) = yield_stress + H*kappa`) raises the effective yield
//! moment with `RodPoints::accumulated_plastic_curvature`, so under a
//! sustained one-direction moment (gravity, a held push) yielding gets
//! harder until the yield moment exceeds the load and the rod shakes down to
//! an elastic residual shape. Isotropic hardening covers monotonic loading;
//! ratcheting under cyclic, reversed loading needs kinematic hardening, not
//! implemented. Default `0.0` (perfectly plastic); opt in with
//! `with_hardening`.

use super::RodPoints;
use super::forces::discrete_curvature;
use crate::matter::materials::utils::scalar_return_map;

#[derive(Debug, Clone, Copy)]
pub struct RodPlasticity {
    /// Yield moment, `sigma_yield*I/c`, N·m (see module doc). Use
    /// `from_young_modulus_rectangular` rather than computing it by hand.
    pub yield_moment_n_m: f32,
    /// Isotropic hardening modulus, N·m per unit accumulated plastic curvature
    /// (see the module doc). `0.0` (default) = perfectly plastic.
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

    /// Yield moment for a rectangular section bent about the axis
    /// perpendicular to the plane, with `RodMaterial::from_young_modulus_
    /// rectangular`'s `width_m`/`thickness_m` (`I = width_m^3*thickness_m/12`,
    /// outer fibre at `c = width_m/2`): `M_yield = sigma_yield_pa * I / c`.
    /// `hardening_modulus_n_m` is `0.0`; chain `.with_hardening(...)` to opt in.
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

/// Elastic-plastic return mapping: at every interior vertex, clamps the
/// bending moment (`ei/voronoi_length * (kappa-rest_curvature)`,
/// `forces::compute_internal_forces`'s `coeff`) to `[-effective_yield,
/// +effective_yield]`, moves the excess into `rest_curvature` through the
/// same local stiffness, and adds its magnitude to
/// `RodPoints::accumulated_plastic_curvature`. `effective_yield =
/// yield_moment_n_m + hardening_modulus_n_m * accumulated_plastic_
/// curvature[i]` (see the module doc). No-op for fewer than 3 points, or when
/// `ei`/`accumulated_plastic_curvature` are not filled per vertex
/// (`Rod::new`/`build_straight_rod` fill them), like `secondary_growth`.
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
        // A sharp kink at one vertex: everything past index 4 rotated by
        // `angle_deg` about point 4.
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
        // By hand: sigma_yield = 2e7 Pa, width = 0.02 m, thickness = 0.001 m ->
        // I = width^3*thickness/12 = 6.667e-10, c = 0.01 ->
        // M_yield = 2e7*6.667e-10/0.01 = 1.3333 N*m.
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
        // A yield moment low against this rod's EI/voronoi_length: a soft,
        // easily yielded material.
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
        // After plastic correction the bent shape is the rod's new natural
        // shape: the bending force (`forces::compute_bending_forces_only`,
        // without axial) at the same bent shape drops.
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

    /// A rod re-kinked a little further each step (standing in for a
    /// sustained moment on a worsening shape) accumulates less permanent set
    /// with hardening than without it.
    #[test]
    fn hardening_accumulates_substantially_less_permanent_set_than_perfectly_plastic() {
        let drive = |hardening_modulus_n_m: f32| -> f32 {
            let mut rod = kinked_rod(5.0); // small initial real kink
            let plasticity = RodPlasticity::new(1.0e-4).with_hardening(hardening_modulus_n_m);
            let pivot_index = 4;
            let pivot = rod.x[pivot_index];
            for _ in 0..40 {
                // A small additional rotation each iteration: the tail keeps
                // demanding a bit more curvature.
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

//! Stress-driven secondary growth (thigmomorphogenesis): a stem's bending
//! stiffness (`RodPoints::ei`, per vertex) grows over time in response to
//! the bending moment acting at that vertex, not on a fixed schedule. Jaffe
//! 1973, "Thigmomorphogenesis: The response of plant growth and development
//! to mechanical stimulation", *Planta* 114(2):143–157: mechanical loading
//! (wind sway, a push) changes plant growth, not only elastic response. The
//! direction of the change (the cambium adds wood where mechanical stress is
//! locally high, moving the structure toward uniform stress) is Mattheck &
//! Kübler 1995's "axiom of uniform stress" (*Wood -- The Internal
//! Optimization of Trees*), also covered in Niklas 1992, *Plant
//! Biomechanics*.
//!
//! The rod does not track a cross-section area separate from `EA`/`EI`, so
//! there is no stress in Pa to compare with an allowable stress. The bending
//! moment (`forces::compute_internal_forces`'s `coeff =
//! (ei/voronoi_length)*(kappa-kappa_rest)`, N·m) is the driving signal
//! instead: at a fixed cross-section bending stress is a direct function of
//! moment, `sigma = M*c/I`, so comparing moments avoids one more unknown.
//! Axial force drives `EA`'s growth the same way.
//!
//! ODE, analogous to `growth::Growth`'s logistic law but keyed on stress
//! excess rather than a carrying capacity:
//! `d(ei)/dt = bending_rate * max(0, |M| - M_threshold)`, `d(ea)/dt =
//! axial_rate * max(0, |F| - F_threshold)`. The rate and threshold constants
//! are illustrative (like `Gravitropism`'s), not fitted to a species.
//!
//! **Mass update**: wood deposition also adds mass at that cross-section.
//! `EA = E*A` with `E` constant gives `d(area)/area == d(ea)/ea`, so that
//! fraction is applied to `RodPoints::linear_density_kg_per_m` and the two
//! endpoint masses of the growing edge. Keyed on `ea`'s growth only, not
//! `ei`'s: `EA` is linear in cross-sectional area, while `EI ~ width^3`
//! mixes in which dimension grows; using `ea` avoids counting the same wood
//! twice through two independently tuned rates.

use super::RodPoints;
use super::forces::discrete_curvature;

#[derive(Debug, Clone, Copy)]
pub struct SecondaryGrowth {
    /// Bending stiffness growth rate per unit moment excess, N·m²/(N·m·s) =
    /// m/s. Illustrative, not species-calibrated (see module doc).
    pub bending_rate: f32,
    /// Bending moment below which no stiffening occurs, N·m (the real
    /// "allowable stress" threshold, expressed in moment terms -- see
    /// module doc).
    pub bending_moment_threshold_n_m: f32,
    /// Axial stiffness growth rate per unit force excess, N/(N·s) = 1/s.
    pub axial_rate: f32,
    /// Axial force below which no stiffening occurs, N.
    pub axial_force_threshold_n: f32,
}

impl SecondaryGrowth {
    pub const fn new(
        bending_rate: f32,
        bending_moment_threshold_n_m: f32,
        axial_rate: f32,
        axial_force_threshold_n: f32,
    ) -> Self {
        Self {
            bending_rate,
            bending_moment_threshold_n_m,
            axial_rate,
            axial_force_threshold_n,
        }
    }
}

/// Evolves `rod.ei`/`rod.ea` toward a more uniform-stress state (Mattheck &
/// Kübler 1995) by growing stiffness wherever the current bending
/// moment/axial force exceeds `growth`'s own threshold. No-op for a rod
/// with fewer than 2 points. Requires `rod.ea`/`rod.ei` to already be
/// filled to full length (`Rod::new` does this -- see Phase 1's doc);
/// a rod with empty `ea`/`ei` is skipped entirely rather than panicking,
/// since there is nowhere real to store the growth.
pub fn apply_secondary_growth(
    rod: &mut RodPoints,
    growth: &SecondaryGrowth,
    dx_meters: f32,
    dt: f32,
) {
    let n = rod.x.len();
    if n < 2 || rod.ea.len() != n - 1 {
        return;
    }

    // ── Axial: real force magnitude per edge, same law as forces.rs's f_stretch ──
    for i in 0..n - 1 {
        let edge = (rod.x[i + 1] - rod.x[i]) * dx_meters;
        let l = edge.length().max(1.0e-9);
        let l0 = rod.rest_edge_length[i].max(1.0e-9);
        let f_stretch = (rod.ea[i] * (l - l0) / l0).abs();
        let excess = (f_stretch - growth.axial_force_threshold_n).max(0.0);
        let d_ea = growth.axial_rate * excess * dt;
        if d_ea > 0.0 && rod.ea[i] > 1.0e-9 {
            // Wood-deposition mass update: EA = E*A with E (Young's modulus)
            // constant, so d(area)/area == d(EA)/EA. Mass at fixed length and
            // material density scales like area, so this edge's linear density
            // and the kilograms at its two endpoints grow by the same fraction
            // that drives the stiffness growth below.
            let frac = d_ea / rod.ea[i];
            let old_edge_mass = rod.linear_density_kg_per_m[i] * l0;
            let d_mass = old_edge_mass * frac;
            rod.linear_density_kg_per_m[i] *= 1.0 + frac;
            rod.mass[i] += 0.5 * d_mass;
            rod.mass[i + 1] += 0.5 * d_mass;
        }
        rod.ea[i] += d_ea;
    }

    // ── Bending: real moment magnitude per vertex, same law as forces.rs's coeff ──
    if n >= 3 && rod.ei.len() == n - 2 {
        for i in 1..n - 1 {
            let (p0, p1, p2) = (
                rod.x[i - 1] * dx_meters,
                rod.x[i] * dx_meters,
                rod.x[i + 1] * dx_meters,
            );
            let kappa = discrete_curvature(p0, p1, p2);
            let kappa_rest = rod.rest_curvature[i - 1];

            let l0_prev = rod.rest_edge_length[i - 1].max(1.0e-9);
            let l0_next = rod.rest_edge_length[i].max(1.0e-9);
            let voronoi_length = 0.5 * (l0_prev + l0_next);

            let moment = ((rod.ei[i - 1] / voronoi_length) * (kappa - kappa_rest)).abs();
            let excess = (moment - growth.bending_moment_threshold_n_m).max(0.0);
            rod.ei[i - 1] += growth.bending_rate * excess * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rod::build_straight_rod;
    use glam::Vec2;

    fn bent_rod() -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 6, 0.01, 1.0);
        points.ea = vec![1.0e5; 5];
        points.ei = vec![1.0; 4];
        // Bend the tip half sideways -- nonzero curvature at every
        // interior vertex, so there's a moment to stiffen against.
        for i in 3..points.x.len() {
            points.x[i].x += (i - 2) as f32 * 0.3;
        }
        points
    }

    #[test]
    fn vertex_under_real_bending_stiffens_while_a_straight_one_does_not() {
        let mut bent = bent_rod();
        let mut straight =
            build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 6, 0.01, 1.0);
        straight.ea = vec![1.0e5; 5];
        straight.ei = vec![1.0; 4];

        let growth = SecondaryGrowth::new(1.0, 0.0, 0.0, 1.0e9);
        for _ in 0..100 {
            apply_secondary_growth(&mut bent, &growth, 1.0, 1.0);
            apply_secondary_growth(&mut straight, &growth, 1.0, 1.0);
        }

        let bent_grew = bent.ei.iter().any(|&ei| ei > 1.5);
        let straight_unchanged = straight.ei.iter().all(|&ei| (ei - 1.0).abs() < 1.0e-6);
        assert!(
            bent_grew,
            "a vertex under real, sustained bending moment must stiffen: ei={:?}",
            bent.ei
        );
        assert!(
            straight_unchanged,
            "a straight rod has zero bending moment and must not stiffen: ei={:?}",
            straight.ei
        );
    }

    fn stretched_rod() -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 4, 0.01, 1.0);
        points.ea = vec![1.0e5; 3];
        points.ei = vec![1.0; 2];
        // Stretch every edge 20% beyond its own rest length -- nonzero
        // axial force to grow against.
        for i in 1..points.x.len() {
            let dir = (points.x[i] - points.x[i - 1]).normalize_or_zero();
            points.x[i] = points.x[i - 1] + dir * (points.rest_edge_length[i - 1] * 1.2);
        }
        points
    }

    #[test]
    fn sustained_axial_stress_adds_real_mass_while_unstressed_rod_gains_none() {
        let mut stretched = stretched_rod();
        let mut slack = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 4, 0.01, 1.0);
        slack.ea = vec![1.0e5; 3];
        slack.ei = vec![1.0; 2];

        let mass_before_stretched: f32 = stretched.mass.iter().sum();
        let mass_before_slack: f32 = slack.mass.iter().sum();

        // bending_rate=0 -- isolate the axial/mass path from bending entirely.
        let growth = SecondaryGrowth::new(0.0, 1.0e9, 2.0, 0.0);
        for _ in 0..200 {
            apply_secondary_growth(&mut stretched, &growth, 1.0, 0.01);
            apply_secondary_growth(&mut slack, &growth, 1.0, 0.01);
        }

        let mass_after_stretched: f32 = stretched.mass.iter().sum();
        let mass_after_slack: f32 = slack.mass.iter().sum();

        assert!(
            mass_after_stretched > mass_before_stretched * 1.01,
            "a rod under real, sustained axial stress must gain real new mass \
             (wood deposition), not just stiffness: before={mass_before_stretched} \
             after={mass_after_stretched}"
        );
        assert!(
            (mass_after_slack - mass_before_slack).abs() < 1.0e-9,
            "an unstressed rod must gain exactly zero mass: before={mass_before_slack} \
             after={mass_after_slack}"
        );
    }

    #[test]
    fn mass_growth_fraction_matches_ea_growth_fraction_exactly() {
        // Direct precision check of the documented derivation (EA=E*A, E
        // const => d(area)/area == d(ea)/ea, mass scales the same way) --
        // not just "some mass appeared somewhere." Measured at point 0
        // specifically: a true rod ENDPOINT touches only edge 0, so its
        // mass is not diluted by a neighboring edge's own static (here,
        // zero-growth) contribution the way an interior point's lumped mass
        // would be.
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 3, 0.01, 1.0);
        rod.ea = vec![1.0e5; 2];
        rod.ei = vec![1.0; 1];
        let l0 = rod.rest_edge_length[0];
        let dir = (rod.x[1] - rod.x[0]).normalize_or_zero();
        rod.x[1] = rod.x[0] + dir * (l0 * 1.2); // stretch edge 0 20%; edge 1 untouched

        let ea_before = rod.ea[0];
        let mass_before = rod.mass[0];

        let growth = SecondaryGrowth::new(0.0, 1.0e9, 3.0, 0.0);
        apply_secondary_growth(&mut rod, &growth, 1.0, 0.01);

        let ea_after = rod.ea[0];
        let mass_after = rod.mass[0];
        let ea_frac = (ea_after - ea_before) / ea_before;
        let mass_frac = (mass_after - mass_before) / mass_before;

        assert!(
            (ea_frac - mass_frac).abs() < 1.0e-5,
            "mass must grow by the SAME fraction as ea, not a different or \
             invented one: ea_frac={ea_frac:.6} mass_frac={mass_frac:.6}"
        );
    }

    #[test]
    fn below_threshold_produces_zero_growth() {
        let mut rod = bent_rod();
        // Threshold set far above any moment this rod can produce.
        let growth = SecondaryGrowth::new(1.0, 1.0e6, 1.0, 1.0e6);
        let before = rod.ei.clone();
        for _ in 0..50 {
            apply_secondary_growth(&mut rod, &growth, 1.0, 1.0);
        }
        assert_eq!(
            rod.ei, before,
            "moment/force below threshold must produce exactly zero growth"
        );
    }

    #[test]
    fn empty_ea_ei_is_a_no_op_not_a_panic() {
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 6, 0.01, 1.0);
        assert!(rod.ea.is_empty());
        let growth = SecondaryGrowth::new(1.0, 0.0, 1.0, 0.0);
        apply_secondary_growth(&mut rod, &growth, 1.0, 1.0);
        assert!(rod.ea.is_empty());
    }
}

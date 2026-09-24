//! A standalone, self-contained discrete-grain population -- rigid circular
//! bodies (2D: position + planar spin, matching this engine's own 2D
//! convention throughout) integrated via real semi-implicit Euler, with
//! contacts resolved by `contact_law`'s real, cited force law.
//!
//! Deliberately NOT coupled to the shared MPM grid yet (see `spacetime::grains`
//! module doc's own status note) -- this is the "prove the piece in isolation"
//! stage, mirroring exactly how this session's Cosserat work proved its own
//! kinematics/field math standalone before any grid wiring. Grid coupling
//! and the packing-fraction oracle deciding where grains are needed are
//! separate, later phases.
//!
//! Real, disclosed simplification for CPU-first correctness (per this
//! project's own standing "CPU correctness first, GPU port second" rule):
//! contacts are tracked as a flat, growable `Vec<ActiveContact>`, rebuilt
//! each substep via brute-force O(n^2) neighbor detection, not the
//! fixed-size-per-grain bounded array a real GPU port would need (matching
//! GeoTaichi's own real `cplist` precedent, which uses exactly this
//! flat-contact-list SHAPE, just with GPU-parallelization-driven fixed
//! capacity -- the same real technique, not a different one). Brute-force
//! neighbor detection is the correct, simple choice for THIS population's
//! expected scale (a thin enrichment layer, not the whole domain's particle
//! count) -- revisit only if a real profiling number shows it's the
//! bottleneck, per this project's own "measure before optimizing" rule.

use glam::Vec2;

use crate::matter::materials::solid::granular::grain_contact_law::{
    ContactLawConfig, ContactSpring, resolve_contact_pair,
};
use crate::matter::particle::Grain;

/// One currently-active contact pair, with its own persistent elastic
/// spring history. `i < j` always (canonical ordering — avoids storing the
/// same pair twice or ever comparing a grain against itself).
#[derive(Clone, Copy, Debug)]
struct ActiveContact {
    i: usize,
    j: usize,
    spring: ContactSpring,
}

/// A standalone discrete-grain population. See module doc for real, disclosed
/// scope (not grid-coupled yet, brute-force neighbor detection).
pub struct GrainPopulation {
    pub grains: Vec<Grain>,
    contacts: Vec<ActiveContact>,
    pub config: ContactLawConfig,
}

impl GrainPopulation {
    pub const fn new(grains: Vec<Grain>, config: ContactLawConfig) -> Self {
        Self {
            grains,
            contacts: Vec::new(),
            config,
        }
    }

    /// Real number of currently-resolved contacts -- diagnostic/test use.
    pub const fn active_contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// TEMP DIAGNOSTIC (2026-08-03 scale-residual investigation): real
    /// per-grain active contact count this substep -- used to correlate
    /// contact count (coordination number) with per-grain energy growth in
    /// the long-horizon column-collapse scale investigation. Remove once
    /// the investigation concludes.
    pub fn contact_count_per_grain(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.grains.len()];
        for c in &self.contacts {
            counts[c.i] += 1;
            counts[c.j] += 1;
        }
        counts
    }

    /// Detects contacts (carrying over persistent spring history for pairs
    /// that were ALREADY in contact last substep, matched by index -- a
    /// genuinely new pair starts with a fresh, zeroed spring, real DEM
    /// convention per `contact_law`'s own doc) and resolves every pair's
    /// force/moment, returning per-grain net contact force and torque.
    /// Pure computation, no integration -- separated from `step` so grid
    /// coupling (`grains::coupling`) can apply these forces AFTER a
    /// grid-gathered velocity, exactly mirroring how `rod::coupling`'s own
    /// internal forces apply after `gather_grid_to_rod`, not instead of it.
    pub fn resolve_contact_forces(&mut self, dt: f32) -> (Vec<Vec2>, Vec<f32>) {
        let n = self.grains.len();
        let mut forces = vec![Vec2::ZERO; n];
        let mut torques = vec![0.0f32; n];

        let mut new_contacts: Vec<ActiveContact> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let gi = self.grains[i].contact_state();
                let gj = self.grains[j].contact_state();
                // Cheap reject before the real contact-law geometry check --
                // avoids allocating/looking up spring history for pairs that
                // are nowhere near each other.
                let max_dist = gi.radius + gj.radius;
                if (gj.x - gi.x).length_squared() > max_dist * max_dist {
                    continue;
                }
                let mut spring = self
                    .contacts
                    .iter()
                    .find(|c| c.i == i && c.j == j)
                    .map(|c| c.spring)
                    .unwrap_or_default();
                if let Some(resolution) =
                    resolve_contact_pair(&gi, &gj, &mut spring, &self.config, dt)
                {
                    let force_on_j = resolution.normal_force * (gj.x - gi.x).normalize()
                        + resolution.tangential_force;
                    forces[j] += force_on_j;
                    forces[i] -= force_on_j;
                    // Rolling moment acts as a real action-reaction pair on
                    // spin, same convention as the linear force above (see
                    // `ContactResolution::rolling_moment`'s own doc).
                    torques[j] += resolution.rolling_moment;
                    torques[i] -= resolution.rolling_moment;
                    // Real, SEPARATE torque source: the tangential force
                    // itself acts at the contact point, offset from each
                    // grain's own center by its own radius -- NOT an
                    // action-reaction pair with a shared sign flip like the
                    // two terms above, because the moment ARM differs
                    // per grain (i.radius vs j.radius) even though the
                    // underlying force is the same -- see
                    // `ContactResolution::friction_torque_on_i/j`'s own doc.
                    // Missing this term was a real, confirmed bug (found via
                    // `diag_max_speed_reached_during_collapse`: max_spin_ever
                    // measured at EXACTLY 0.0 across 2000 real steps of
                    // otherwise-dynamic contact) -- without it, grains can
                    // only ever slide against each other, never actually
                    // start rolling from sliding contact at all.
                    torques[i] += resolution.friction_torque_on_i;
                    torques[j] += resolution.friction_torque_on_j;
                    new_contacts.push(ActiveContact { i, j, spring });
                }
            }
        }
        self.contacts = new_contacts;
        (forces, torques)
    }

    /// One real, standalone semi-implicit Euler substep (gravity + contact
    /// forces integrated directly into velocity/spin, then position
    /// integrated from the new velocity) -- for a `GrainPopulation` NOT
    /// coupled to the shared MPM grid (real, standard, more stable than
    /// explicit-Euler for stiff contact springs, same rationale this
    /// engine's own MLS-MPM P2G/G2P cycle already follows for the grid
    /// velocity update). Grid-coupled use goes through `grains::coupling`
    /// instead, which applies `resolve_contact_forces`'s output after a
    /// grid-gathered velocity rather than through this method.
    pub fn step(&mut self, gravity: Vec2, dt: f32) {
        let (forces, torques) = self.resolve_contact_forces(dt);
        for (idx, grain) in self.grains.iter_mut().enumerate() {
            let accel = gravity + forces[idx] / grain.mass;
            grain.v += accel * dt;
            let angular_accel = torques[idx] / grain.moment_of_inertia();
            grain.spin += angular_accel * dt;
            grain.orientation += grain.spin * dt;
            grain.x += grain.v * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ContactLawConfig {
        ContactLawConfig {
            normal_stiffness: 1.0e5,
            tangential_stiffness: 0.8e5,
            rolling_stiffness: 5.0e3,
            normal_damping: 50.0,
            tangential_damping: 50.0,
            rolling_damping: 50.0,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }

    #[test]
    fn single_grain_free_falls_under_gravity_exactly() {
        let mut pop = GrainPopulation::new(vec![Grain::new(Vec2::ZERO, 0.5, 1.0)], config());
        let gravity = Vec2::new(0.0, -9.8);
        let dt = 0.001;
        for _ in 0..100 {
            pop.step(gravity, dt);
        }
        let expected_v = gravity.y * dt * 100.0;
        assert!(
            (pop.grains[0].v.y - expected_v).abs() < 1e-3,
            "v={} expected={}",
            pop.grains[0].v.y,
            expected_v
        );
        assert_eq!(pop.active_contact_count(), 0);
    }

    #[test]
    fn light_grain_resting_on_a_pinned_floor_reaches_the_real_predicted_equilibrium_overlap() {
        // Two real mistakes fixed here from an earlier version of this test:
        // (1) a spatially-uniform force applied to BOTH grains equally
        //     produces ZERO relative acceleration between them -- elementary
        //     mechanics (equivalence principle: gravity accelerates
        //     everything identically regardless of mass), not a property of
        //     this contact code. A huge MASS alone does not pin a body
        //     against gravity -- it still free-falls at the same rate, just
        //     reacts less to CONTACT forces. A real fixed floor needs an
        //     actual position anchor (real precedent: `Particle::pinned`'s
        //     own Dirichlet-boundary convention elsewhere in this engine),
        //     approximated here by re-clamping the floor grain's state after
        //     every step -- a real, standard test technique, not a hack
        //     specific to this bug.
        // (2) using a huge position offset (1e6) alongside an expected
        //     SIGNAL of order 1e-4 completely loses f32 precision (~7
        //     significant digits) -- real numerical-conditioning mistake,
        //     not a contact-law bug. Kept both grains at well-conditioned,
        //     order-1 coordinates instead.
        //
        // Real, precise, closed-form prediction at equilibrium: kn*overlap =
        // m*g (spring force balances weight) -> overlap = m*g/kn.
        let cfg = config();
        let m = 1.0;
        let g = 9.8;
        let expected_overlap = m * g / cfg.normal_stiffness;
        let floor_anchor = Vec2::new(0.0, -1.0);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, m),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            cfg,
        );
        let gravity = Vec2::new(0.0, -g);
        let dt = 0.0002;
        let mut max_speed: f32 = 0.0;
        for _ in 0..20_000 {
            pop.step(gravity, dt);
            // Pin the floor grain: real Dirichlet-anchor technique, not a
            // contact-law shortcut (see the comment above).
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
            max_speed = max_speed.max(pop.grains[0].v.length());
        }
        assert!(
            max_speed < 50.0,
            "light grain never settled, exploded instead: max_speed={max_speed}"
        );
        let dist = (pop.grains[0].x - pop.grains[1].x).length();
        let overlap = 1.0 - dist;
        assert!(
            (overlap - expected_overlap).abs() < expected_overlap * 0.5,
            "expected overlap near {expected_overlap} (kn*overlap=m*g), got {overlap}"
        );
    }

    #[test]
    fn small_pile_under_gravity_settles_without_exploding() {
        // Real, minimal "does this actually work as a pile" sanity check --
        // not the full long-horizon repose-angle verification (a separate,
        // later, dedicated test matching this session's own established
        // discipline), just confirming a handful of grains dropped under
        // gravity onto a floor settle into a bounded, finite configuration
        // instead of diverging.
        let mut grains = Vec::new();
        for row in 0..3 {
            for col in 0..4 {
                let x = col as f32 * 1.05 + (row % 2) as f32 * 0.5;
                let y = row as f32 * 1.0 + 3.0;
                grains.push(Grain::new(Vec2::new(x, y), 0.5, 1.0));
            }
        }
        // A large, heavy "floor" grain well below everything else.
        let mut floor = Grain::new(Vec2::new(2.0, -50.0), 50.0, 1.0e8);
        floor.mass = 1.0e8;
        grains.push(floor);

        let mut pop = GrainPopulation::new(grains, config());
        let gravity = Vec2::new(0.0, -9.8);
        let dt = 0.0002;
        for _ in 0..5000 {
            pop.step(gravity, dt);
        }
        for (idx, g) in pop.grains.iter().enumerate() {
            assert!(
                g.x.is_finite() && g.v.is_finite() && g.spin.is_finite(),
                "grain {idx} diverged: x={:?} v={:?} spin={}",
                g.x,
                g.v,
                g.spin
            );
            assert!(
                g.v.length() < 100.0,
                "grain {idx} velocity exploded: {:?}",
                g.v
            );
        }
    }

    #[test]
    fn two_free_grains_total_mechanical_energy_never_grows_without_an_external_driver() {
        // Real, CORRECTED physical invariant (2026-08-03). The original
        // version of this test asserted raw KINETIC energy alone could not
        // grow past its initial value -- empirically measured as a 64%
        // "violation" (ke0=4.0, max_ke=6.571). A full energy-breakdown
        // instrumentation (see `spacetime::grains` session notes) traced
        // this to a real, non-buggy cause: this test's own initial
        // condition spawns the two grains ALREADY overlapping by 1% of
        // radius (radii sum 1.0, separation 0.99), which means the contact
        // starts with substantial PRELOADED elastic potential energy in the
        // normal spring -- PE_n0 = 0.5*kn*overlap0^2 = 5.0, actually MORE
        // than the initial kinetic energy itself (KE0=4.0). As that real
        // compressed spring naturally pushes the two free grains apart --
        // ordinary, correct physics for ANY spring-dashpot contact model,
        // not a bug, exactly like releasing a compressed spring between two
        // free masses -- it legitimately converts stored PE into KE, which
        // the old invariant misread as an energy-conservation violation.
        //
        // Direct instrumentation confirmed this is legitimate: total
        // mechanical energy (KE + normal-spring PE + tangential-spring PE +
        // cumulative dissipated energy) stayed within 0.58% of its initial
        // value across the entire 2,000,000-step run (peak relative
        // overshoot 0.5786% at step 18229) -- real, bounded, expected
        // semi-implicit-Euler numerical error for a stiff damped spring
        // (matches this force law's own hand-derived power balance:
        // d(KE+PE)/dt = -normal_damping*v_n^2 - tangential_damping*v_t^2 <=
        // 0), nowhere close to a genuine 64% energy injection. A separate
        // check at 10x finer dt (which any REAL discretization bug should
        // shrink under, not grow under) instead showed the KE-alone
        // "violation" tracks physical PE release consistently at matching
        // physical time, further confirming this is not a discretization
        // artifact of contact_law.rs/population.rs.
        //
        // The real, physically correct invariant for a purely dissipative
        // (damped) contact system with zero external driver -- one that may
        // start pre-loaded with elastic energy, exactly like every real
        // pair of touching grains in a settled pile -- is that TOTAL
        // mechanical energy (KE + energy stored in the contact springs) is
        // monotonically non-increasing, NOT raw KE alone, which is free to
        // rise as preloaded spring PE legitimately converts into it.
        let mut cfg = config();
        cfg.friction = 1.0e6; // cap should never engage except right at separation
        cfg.rolling_stiffness = 0.0; // isolate normal+tangential only
        cfg.rolling_friction = 0.0;
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.0), 0.5, 1.0),
                Grain::new(Vec2::new(0.99, 0.0), 0.5, 1.0),
            ],
            cfg,
        );
        pop.grains[0].v = Vec2::new(0.0, 2.0);
        pop.grains[1].v = Vec2::new(0.0, -2.0);

        // Real total mechanical energy: KE (translational + rotational)
        // plus whatever elastic PE is currently stored in the active
        // contact's normal and tangential springs -- the actual
        // conserved-minus-dissipated quantity for this force law.
        let mechanical_energy = |pop: &GrainPopulation| -> f32 {
            let ke: f32 = pop
                .grains
                .iter()
                .map(|g| {
                    0.5 * g.mass * g.v.length_squared()
                        + 0.5 * g.moment_of_inertia() * g.spin * g.spin
                })
                .sum();
            let gi = pop.grains[0];
            let gj = pop.grains[1];
            let overlap = (gi.radius + gj.radius - (gj.x - gi.x).length()).max(0.0);
            let normal_pe = 0.5 * pop.config.normal_stiffness * overlap * overlap;
            let spring = pop
                .contacts
                .iter()
                .find(|c| c.i == 0 && c.j == 1)
                .map(|c| c.spring)
                .unwrap_or_default();
            let tangential_pe =
                0.5 * pop.config.tangential_stiffness * spring.tangential.length_squared();
            ke + normal_pe + tangential_pe
        };

        let e0 = mechanical_energy(&pop);
        let mut max_e = e0;
        let dt = 0.0000001;
        for step in 0..2_000_000 {
            pop.step(Vec2::ZERO, dt);
            let e = mechanical_energy(&pop);
            max_e = max_e.max(e);
            assert!(
                pop.grains
                    .iter()
                    .all(|g| g.v.is_finite() && g.spin.is_finite()),
                "diverged at step {step}"
            );
        }
        // Real measured peak this session: 0.5786% overshoot. 1% gives
        // real headroom for bounded explicit-Euler numerical error while
        // still catching genuine energy injection (a real bug here would
        // look like the old invariant's measured 64% "violation").
        assert!(
            max_e <= e0 * 1.01,
            "total mechanical energy grew without an external driver: e0={e0:.6} max_e={max_e:.6} \
             -- with zero external driver and only dissipative (damped) contact forces, KE plus \
             energy stored in the contact springs must be monotonically non-increasing (up to \
             bounded semi-implicit-Euler numerical error), even though raw KE alone is free to \
             rise as preloaded spring PE legitimately converts into it"
        );
    }

    #[test]
    fn sliding_grain_on_a_pinned_floor_converges_toward_rolling_not_away_from_it() {
        // Real, direct physical check for the friction-induced-rolling
        // torque (`ContactResolution::friction_torque_on_i/j`): a grain
        // given a real sliding velocity along a fixed floor must evolve
        // TOWARD "rolling without slipping" (the contact-point tangential
        // slip speed |v_t| decaying over time as spin builds up to match).
        // Confirms the torque's sign/formula is genuinely correct in
        // isolation (verified: slip decays smoothly 3.0 -> 0.9 while real
        // contact stays engaged). A real, SEPARATE full-column collapse
        // test (`tests/grains_repose_angle.rs`) showed unbounded growth
        // instead -- this test proves that's NOT a sign error in the core
        // force law; the real cause is elsewhere (many-body/repeated-
        // contact dynamics specific to that scene, not this pairwise law).
        // Real, fixed overlap (1% of radius) from the start, ZERO gravity --
        // isolates purely the sliding-friction-induces-rolling question,
        // removing the confound of an earlier version of this test (gravity
        // continuously growing the overlap over time, meaning the contact
        // barely engaged at all for the first several thousand steps while
        // a real gap was still closing -- a real, separate effect that
        // muddied this specific measurement, not itself a bug).
        let floor_anchor = Vec2::new(0.0, -0.495);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop.grains[0].v = Vec2::new(3.0, 0.0); // real sliding velocity, no spin yet
        let gravity = Vec2::ZERO;
        let dt = 0.000001;

        let slip_speed = |pop: &GrainPopulation| -> f32 {
            let g = &pop.grains[0];
            let floor = &pop.grains[1];
            let n = (g.x - floor.x).normalize();
            let t = Vec2::new(-n.y, n.x);
            let v_rel = g.v - floor.v;
            (v_rel.dot(t) - (g.radius * g.spin + floor.radius * floor.spin)).abs()
        };

        // Measured over the window real contact stays genuinely engaged
        // (checked directly: this pair separates vertically -- the normal
        // spring's own bounce -- a bit after step ~4500, at which point
        // velocity/spin freeze and any further "slip" reading is pure
        // separated-body geometric drift, not real contact physics; this
        // window is entirely within the real, engaged-contact regime).
        let slip_early = slip_speed(&pop);
        for _ in 0..4000 {
            pop.step(gravity, dt);
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
        }
        let slip_late = slip_speed(&pop);

        assert!(
            pop.grains[0].v.is_finite() && pop.grains[0].spin.is_finite(),
            "diverged: v={:?} spin={}",
            pop.grains[0].v,
            pop.grains[0].spin
        );
        assert!(
            slip_late < slip_early,
            "expected slip speed to DECAY toward rolling (early={slip_early:.4} late={slip_late:.4}) \
             -- growth here means the friction torque is signed wrong, actively driving slip instead of resisting it"
        );
    }

    /// Real, direct numeric proof that `Grain::orientation` (added
    /// 2026-08-03 specifically so a grain's real rolling has something to
    /// render) actually accumulates a genuine rotation, not just a nonzero
    /// `spin` that never gets integrated anywhere. Same real scenario as
    /// `sliding_grain_on_a_pinned_floor_converges_toward_rolling_not_away_
    /// from_it` above (a grain sliding on a pinned floor, real friction-
    /// induced spin-up) -- reused rather than invented fresh, since that
    /// scenario already independently proves the underlying spin dynamics
    /// are correct; this test's ONLY new claim is that `orientation`
    /// faithfully tracks `integral(spin dt)`.
    #[test]
    fn grain_orientation_genuinely_accumulates_real_rotation_while_rolling() {
        let floor_anchor = Vec2::new(0.0, -0.495);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop.grains[0].v = Vec2::new(3.0, 0.0);
        let gravity = Vec2::ZERO;
        let dt = 0.000001;

        assert_eq!(
            pop.grains[0].orientation, 0.0,
            "must start at zero rotation"
        );
        println!(
            "step=0 orientation={:.6} rad spin={:.4} rad/s",
            pop.grains[0].orientation, pop.grains[0].spin
        );
        for step in 1..=4000 {
            pop.step(gravity, dt);
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
            if step % 500 == 0 {
                println!(
                    "step={step} orientation={:.6} rad ({:.2} deg) spin={:.4} rad/s",
                    pop.grains[0].orientation,
                    pop.grains[0].orientation.to_degrees(),
                    pop.grains[0].spin
                );
            }
        }
        let final_orientation = pop.grains[0].orientation;
        // Real, corrected threshold (2026-08-04): the ORIGINAL >0.5 rad bound
        // here was simply wrong -- at dt=1e-6s, 4000 steps is only 4ms of
        // real simulated time, and spin ramps from 0 up to ~-2.4 rad/s over
        // that same window, so the real, correct integral is on the order
        // of -0.005 rad (roughly avg_spin * duration), not >0.5 rad. Caught
        // by actually running this test rather than assuming the threshold
        // was right -- exactly the kind of "prove it, don't assume it"
        // check this session's own standing discipline requires. The real
        // claim this test makes is or nonzero, correctly-signed, non-NaN
        // rotation consistent with the spin history -- verified precisely
        // by the independent trapezoidal cross-check below, not by an
        // arbitrary magnitude bound.
        assert!(
            final_orientation.is_finite() && final_orientation != 0.0,
            "expected real, nonzero accumulated rotation from 4000 steps of \
             real friction-induced spin-up, got {final_orientation}"
        );
        // Cross-check: `orientation` must match the real numerical integral
        // of `spin`, not just be "some nonzero number" -- reruns the exact
        // same physics while independently trapezoidal-integrating spin by
        // hand, then compares against the engine's own bookkeeping.
        let mut pop2 = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop2.grains[0].v = Vec2::new(3.0, 0.0);
        let mut independent_integral = 0.0f32;
        for _ in 0..4000 {
            let spin_before = pop2.grains[0].spin;
            pop2.step(gravity, dt);
            pop2.grains[1].x = floor_anchor;
            pop2.grains[1].v = Vec2::ZERO;
            pop2.grains[1].spin = 0.0;
            let spin_after = pop2.grains[0].spin;
            independent_integral += 0.5 * (spin_before + spin_after) * dt;
        }
        let rel_err = (pop2.grains[0].orientation - independent_integral).abs()
            / independent_integral.abs().max(1e-6);
        println!(
            "engine orientation={:.6} independent trapezoidal integral={:.6} rel_err={:.4}",
            pop2.grains[0].orientation, independent_integral, rel_err
        );
        assert!(
            rel_err < 0.01,
            "engine's own orientation bookkeeping doesn't match an independent integral of spin: \
             engine={:.6} independent={:.6}",
            pop2.grains[0].orientation,
            independent_integral
        );
    }
}

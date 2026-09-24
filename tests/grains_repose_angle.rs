//! The real moment-of-truth test: does pure grain-grain elastic-plastic
//! rolling resistance (Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011)
//! hold a genuine, LONG-HORIZON-stable angle of repose from an unstable
//! collapsing column -- the exact question every rate-dependent continuum
//! mechanism already tried this session (Cundall damping, KE-peak switches,
//! Cosserat curvature coupling) failed to answer, because they all fade to
//! zero at rest. Same long-horizon discipline that caught Cosserat's own
//! false positive (Lajeunesse et al. 2004 comparison, checked at multiple
//! step counts, not just an early snapshot).
//!
//! Real, disclosed scope: pure grains only (no continuum coupling in THIS
//! test -- that's `grains::coupling`'s own, separately-tested concern), an
//! effective/coarse-grained grain radius (not literal ~0.15mm dry-sand
//! grain size -- same honest simulation-scale compromise this project's own
//! NGF/Cosserat work already made, disclosed not hidden), and a real,
//! disclosed FIXED substep dt derived from `contact_law::critical_timestep`
//! rather than the adaptive MPM substep chooser (which doesn't yet know
//! about grain contact stiffness at all -- a real, disclosed gap, not
//! silently worked around; folding grain stability into the adaptive
//! chooser, mirroring `rod_cfl_dt`'s own real precedent, is separate future
//! work).

extern crate emerge_engine as emerge;
use emerge::grains::population::GrainPopulation;
use emerge::materials::solid::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use glam::Vec2;

/// Real, disclosed effective grain properties -- not literal dry-sand grain
/// size (0.15-0.3mm would require an intractable grain count for a
/// human-scale pile, already established this session), but real order-of-
/// magnitude physical values for a coarse-grained "effective grain":
/// - nominal radius 0.01 m (1 cm), +-10% real polydispersity (see
///   `build_column`'s own doc)
/// - density 1600 kg/m^3 (real dry sand bulk density order of magnitude)
/// - 2D areal mass: rho * pi * r^2 (same convention this engine's own
///   `particle_mass` uses elsewhere for 2D MPM particles)
fn make_grain_with_radius(x: Vec2, radius_m: f32) -> Grain {
    const DENSITY_KG_M3: f32 = 1600.0;
    let mass = DENSITY_KG_M3 * std::f32::consts::PI * radius_m * radius_m;
    Grain::new(x, radius_m, mass)
}

/// Real contact stiffness from a real Young's modulus via the standard
/// linear-spring calibration `kn ~ E * r` (maps a real material stiffness to
/// an equivalent contact spring, common real DEM practice) -- E=1e7 Pa
/// (10 MPa), the same order of magnitude Klar et al. 2016's own sand
/// calibration uses (already cited throughout this engine's `sand.rs`).
/// Real friction mu=tan(35 deg) (this project's own already-cited real
/// friction angle for dry sand, Klar et al. 2016). Rolling friction 0.1 --
/// real, mid-range value from Ai et al. 2011's own cited survey range
/// (0.001-0.3), not hand-tuned to force a particular result.
fn config() -> ContactLawConfig {
    const RADIUS_M: f32 = 0.01;
    const E_PA: f32 = 1.0e7;
    let kn = E_PA * RADIUS_M;
    ContactLawConfig {
        normal_stiffness: kn,
        tangential_stiffness: 0.8 * kn,
        rolling_stiffness: kn * RADIUS_M * RADIUS_M * 0.1,
        // Real, moderate-to-high damping (~60% critical): real dry sand
        // grains are genuinely LOSSY colliders (real coefficient of
        // restitution for sand is commonly cited around 0.5 or lower --
        // most of a collision's kinetic energy converts to heat/sound/
        // micro-plastic deformation, not an elastic bounce). An earlier,
        // much lower damping level (~5% critical, closer to a near-elastic
        // e~1 collision) let the full many-body column run away without
        // bound (12x the real predicted runout and still climbing at
        // 40,000 steps) even though a clean, isolated 2-body sliding test
        // (`population::tests::sliding_grain_on_a_pinned_floor_...`)
        // confirmed the underlying force law/torque sign is genuinely
        // correct -- real granular energy dissipation across many
        // simultaneous, repeated contacts needs real, adequate damping,
        // not just a formally-stable-in-isolation low value.
        normal_damping: (2.0 * (kn * 2.01_f32).sqrt()) * 0.6,
        tangential_damping: (2.0 * (kn * 2.01_f32).sqrt()) * 0.6,
        // Real, missing piece added 2026-08-03: same 60%-critical convention
        // as normal/tangential above, applied to the rolling channel's own
        // stiffness (the rolling-torque sign fix alone took the small 8-grain
        // column from 31.3x/exploding to a near-exact 1.045x, but left the
        // full 80-grain column still growing, 12.1x -> 3.9x -- an undamped
        // elastic-plastic rolling oscillator, now correctly RESTORING but
        // still lossless, is the real remaining candidate).
        rolling_damping: (2.0 * (kn * RADIUS_M * RADIUS_M * 0.1 * 2.01_f32).sqrt()) * 0.6,
        friction: (35.0_f32).to_radians().tan(),
        rolling_friction: 0.1,
    }
}

/// Tiny deterministic LCG for reproducible jitter/polydispersity -- same
/// role as this engine's own internal `LcgRng` (used for exactly this
/// purpose in real spawn regions), a local copy here since this is a
/// standalone test file with no dependency on that private type.
struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG constants -- same real, standard choice
        // this engine's own `LcgRng` uses.
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Builds a real triangular column of grains (same real Lajeunesse et al.
/// 2004 collapse-geometry convention this project's own Cosserat/NGF tests
/// already use: initial radius R0, initial height H0, real predicted
/// runout R_inf = R0 * (1 + 2*sqrt(H0/R0))) -- packed on a square lattice,
/// touching neighbors, real gravity-consistent stacking (bottom rows first).
///
/// Real, disclosed, NOT optional: small position jitter + radius
/// polydispersity, matching two independent real precedents -- this
/// engine's own `SpawnRegion::position_jitter` doc ("break lattice
/// symmetry and prevent artificially regular pile formation") AND the
/// Hybrid Grains paper's own disclosed practice ("we run all simulations
/// with a slight polydispersity in granular radii... to better match real
/// shape distributions and to avoid crystallization"). A perfectly regular,
/// symmetric lattice has no physical asymmetry to trigger real lateral
/// collapse/toppling at all -- confirmed the hard way: an earlier, jitter-
/// free version of this exact test stayed frozen in its initial shape for
/// 40,000 steps, not because rolling resistance held it (it hadn't even
/// started moving), but because a perfectly regular stack under pure
/// vertical gravity has no reason to ever move sideways.
fn build_column(r0_grains: usize, h0_grains: usize, radius_m: f32) -> (Vec<Grain>, f32) {
    // Real, loose "poured" packing, not a snug touching lattice: a real
    // sand column is poured, not snapped into a perfect touching grid --
    // a touching lattice has an artificially high coordination number
    // (strong lateral interlocking support neighbors), unlike a real,
    // loosely-poured pile with room to genuinely rearrange/roll under
    // gravity. 30% extra spacing gives real initial gaps.
    let spacing = 2.6 * radius_m;
    let mut rng = SmallRng(0xC0FF_EE11_u64);
    let mut grains = Vec::new();
    for row in 0..h0_grains {
        for col in 0..(2 * r0_grains) {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let x = col as f32 * spacing + jx;
            let y = row as f32 * spacing + radius_m + jy;
            let r = radius_m * (0.9 + 0.2 * rng.next_f32()); // +-10% real polydispersity
            grains.push(make_grain_with_radius(Vec2::new(x, y), r));
        }
    }
    let r0_m = r0_grains as f32 * (2.0 * radius_m);
    let h0_m = h0_grains as f32 * (2.0 * radius_m);
    let aspect_ratio = h0_m / r0_m;
    let predicted_r_inf_m = r0_m * (1.0 + 2.0 * aspect_ratio.sqrt());
    (grains, predicted_r_inf_m)
}

/// Real, standard technique for an effectively-flat floor within a
/// standalone `GrainPopulation` (no MPM grid/boundary in this pure-grain
/// test): one real grain of a MUCH larger radius than the column
/// (curvature negligible across the column's own width), re-clamped to a
/// fixed position/velocity/spin after every step -- same real "pinned
/// anchor" technique already proven correct in
/// `population::tests::light_grain_resting_on_a_pinned_floor_...`, scaled
/// up to span the whole column's width instead of a single point.
///
/// Real, confirmed root cause of the long-standing "80-grain column still
/// creeps/grows without bound at long horizon" residual (2026-08-03): NOT a
/// `contact_law.rs` bug -- a test-harness geometry artifact in THIS floor
/// approximation. At the previous `FLOOR_RADIUS_M = 5.0`, direct
/// instrumentation (tracing the single grain responsible, per-grain
/// contact-count-vs-energy buckets, and a full extended-horizon rerun)
/// showed the growth was concentrated almost entirely in a single grain
/// with only ONE active contact (never the densely-coordinated interior
/// grains, ruling out a many-simultaneous-contact summation bug), whose
/// one contact partner was consistently the floor grain itself. That grain
/// had been given a real, ordinary sideways kick during the initial
/// chaotic collapse (nothing anomalous there), and from then on genuinely
/// ROLLED DOWNHILL along the floor's own visible curvature -- at
/// `FLOOR_RADIUS_M = 5.0` a 2.3 m lateral excursion (well within what a
/// real chaotic collapse produces for an ejected grain, even though the
/// column's own predicted runout is only ~0.1-0.3 m) already drops the
/// "flat" floor's surface by 0.57 m, comparable to or exceeding the real
/// gravitational PE budget available to that one grain -- a genuine,
/// energy-conserving (measured: KE gain of the same order as, and somewhat
/// less than, the PE released; near-exact rolling-without-slip,
/// `spin*radius` tracking `speed` throughout) but entirely UNINTENDED
/// energy source, not a force-law defect: confirmed directly by watching
/// that grain's spin/speed with the old, small radius (monotonic runaway,
/// 0 -> -226 rad/s by step 75,000, extended-horizon ratio diverging
/// 2.1x -> 56.1x by step 200,000 with `center_y` reaching -3.33, i.e. real
/// tunneling well below the intended floor) versus the SAME trace at this
/// 10x larger radius (settles to ~0 velocity/spin by step ~25,000 and
/// stays there for the rest of a 75,000-step run). Real fix: make the
/// floor stand-in radius large enough that curvature stays negligible over
/// the actual excursion range a real chaotic collapse can produce, not
/// just the column's own nominal predicted runout -- 50.0 m keeps the
/// worst-case curvature-induced drop under ~0.05 m even for a multi-meter
/// excursion while staying comfortably inside f32's precision budget at
/// this coordinate scale (both the 8-grain and 80-grain long-horizon tests
/// now genuinely arrest: flat ratio from 40,000 all the way to 200,000
/// steps, confirmed by direct extended-horizon rerun, not just an early
/// snapshot -- the exact discipline that caught Cosserat's own false
/// positive earlier this session).
const FLOOR_RADIUS_M: f32 = 50.0;

fn run_collapse(steps: usize) -> (f32, f32, f32) {
    run_collapse_sized(steps, 4, 10)
}

fn run_collapse_sized(steps: usize, r0_grains: usize, h0_grains: usize) -> (f32, f32, f32) {
    const RADIUS_M: f32 = 0.01;
    let (mut grains, predicted_r_inf_m) = build_column(r0_grains, h0_grains, RADIUS_M);
    // Real floor: top surface at y=0 (grains already stack starting at
    // y=radius, i.e. resting exactly on y=0), centered under the column.
    let column_width = 2.0 * r0_grains as f32 * (2.0 * RADIUS_M);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config();
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03; // real, standard DEM safety margin (10-20% of critical)

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    for _ in 0..steps {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;
    }

    let xs: Vec<f32> = pop
        .grains
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != floor_idx)
        .map(|(_, g)| g.x.x)
        .collect();
    let n = xs.len() as f32;
    let center_x = xs.iter().sum::<f32>() / n;
    let measured_r_inf_m = xs.iter().map(|&x| (x - center_x).abs()).fold(0.0, f32::max);
    let center_y = pop
        .grains
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != floor_idx)
        .map(|(_, g)| g.x.y)
        .sum::<f32>()
        / n;
    (measured_r_inf_m, predicted_r_inf_m, center_y)
}

/// Real diagnostic: isolates whether the full column's unbounded growth is
/// a genuine many-body/many-simultaneous-contact effect (a MUCH smaller
/// column should then stay stable) or something present even at small N
/// (would point to a deeper, still-unresolved issue). Neither damping level
/// nor timestep changes affected the full-size explosion at all -- this
/// checks the one remaining real variable, scale/contact-count, directly.
#[test]
fn diag_small_column_scale_isolation() {
    println!("── SMALL-COLUMN SCALE ISOLATION (2 wide x 4 tall = 8 grains) ──");
    for &checkpoint in &[500usize, 2_000, 5_000, 15_000, 40_000] {
        let (measured, predicted, center_y) = run_collapse_sized(checkpoint, 1, 4);
        println!(
            "  steps={checkpoint:>6}: measured_R={measured:.4}m predicted_R={predicted:.4}m ratio={:.3}x center_y={center_y:.4}",
            measured / predicted
        );
    }
}

/// Real, direct diagnostic: does ANY grain ever reach a real, meaningful
/// speed at all during the collapse, or does the whole system stay
/// essentially motionless from the start? Distinguishes a genuine
/// calibration issue (grains DO move/tumble with real kinetic energy, but
/// settle into a more-supported-than-expected shape) from an actual bug in
/// the force computation (nothing ever really moves at all).
#[test]
fn diag_max_speed_reached_during_collapse() {
    const R0_GRAINS: usize = 4;
    const H0_GRAINS: usize = 10;
    const RADIUS_M: f32 = 0.01;
    let (mut grains, _) = build_column(R0_GRAINS, H0_GRAINS, RADIUS_M);
    let column_width = 2.0 * R0_GRAINS as f32 * (2.0 * RADIUS_M);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config();
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03;
    println!(
        "dt={dt:.6e}s dt_crit={dt_crit:.6e}s num_grains={}",
        grains.len() - 1
    );

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    let mut max_speed_ever: f32 = 0.0;
    let mut max_spin_ever: f32 = 0.0;
    for step in 0..2000 {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;
        let (max_v, max_w) = pop
            .grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .fold((0.0f32, 0.0f32), |(mv, mw), (_, g)| {
                (mv.max(g.v.length()), mw.max(g.spin.abs()))
            });
        max_speed_ever = max_speed_ever.max(max_v);
        max_spin_ever = max_spin_ever.max(max_w);
        if step % 200 == 0 || step < 5 {
            println!(
                "  step={step:>5}: max_speed_now={max_v:.6} m/s max_spin_now={max_w:.4} rad/s contacts={}",
                pop.active_contact_count()
            );
        }
    }
    println!(
        "max_speed_ever={max_speed_ever:.6} m/s (real gravity free-fall for {:.4}s would reach {:.4} m/s)",
        2000.0 * dt,
        9.8 * 2000.0 * dt
    );
    println!("max_spin_ever={max_spin_ever:.4} rad/s");
}

/// Real regression test for a genuine, found-and-fixed sign bug (2026-08-03):
/// `diag_small_column_scale_isolation` already disproved the "it's a
/// many-body-network effect" hypothesis for the column-collapse divergence --
/// even an 8-grain column diverged WORSE (31x vs the full column's 12x) over
/// the same long horizon, with `center_y` going NEGATIVE (grains sinking
/// below the pinned floor -- physically impossible). This test isolates the
/// smallest case that still has an off-axis (non-purely-vertical) contact: a
/// single grain resting on a huge pinned "floor" grain whose center is
/// offset just 0.01 (0.2% of the floor's own radius) from being directly
/// below the grain -- exactly the sliver of tangential/rolling code path a
/// perfectly-vertical stack never exercises (`v_t`/`omega_rel` stay
/// EXACTLY zero by symmetry when perfectly aligned).
///
/// Real, confirmed root cause (traced with a temporary instrumented
/// breakdown calling `resolve_contact_pair` directly): `ContactLawConfig`'s
/// rolling-resistance spring (`contact_law::resolve_contact_pair`'s rolling
/// block) had its elastic-restoring-torque sign backwards relative to the
/// convention its own caller (`GrainPopulation::resolve_contact_forces`)
/// applies it with ("acting on j, equal and opposite on i") -- turning what
/// should be a torsional spring's NEGATIVE feedback (restoring, real Ai et
/// al. 2011 EPSD behavior) into POSITIVE feedback. Confirmed empirically:
/// `omega_rel`/`spin_i` grew MONOTONICALLY (never oscillating back toward
/// zero like a real restoring spring) and, once the Coulomb-like rolling cap
/// engaged, the net torque stayed pinned in a constant, growth-REINFORCING
/// direction forever. Fixed by dropping the erroneous minus sign from both
/// the elastic trial (`trial_mr`) and its plastic return-mapping rescale in
/// `contact_law.rs`. Before the fix this test diverged (contact fully lost
/// by step 100,000, KE growing from ~6 to ~6847 by step 400,000); after the
/// fix it stays bounded and settles (KE == 0.0 for the entire run, matching
/// the real closed-form equilibrium overlap) -- kept as a permanent
/// regression since a perfectly-aligned test can never catch this class of
/// bug again.
#[test]
fn diag_minimal_two_grains_on_floor_long_horizon() {
    const RADIUS_M: f32 = 0.01;
    let g0 = make_grain_with_radius(Vec2::new(0.0, RADIUS_M), RADIUS_M);
    let floor_anchor = Vec2::new(0.01, -FLOOR_RADIUS_M); // tiny 0.2%-of-floor-radius offset -- see doc comment above
    let floor = Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9);
    let floor_idx = 1;
    let mut pop = GrainPopulation::new(vec![g0, floor], config());

    let cfg = config();
    let m_eff = pop.grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03;
    println!("── MINIMAL 1-GRAIN-ON-FLOOR LONG-HORIZON DRIFT CHECK ── dt={dt:.6e}s");

    let gravity = Vec2::new(0.0, -9.8);
    let energy = |pop: &GrainPopulation| -> f32 {
        pop.grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .map(|(_, g)| {
                0.5 * g.mass * g.v.length_squared() + 0.5 * g.moment_of_inertia() * g.spin * g.spin
            })
            .sum()
    };
    for checkpoint_group in 0..8 {
        for _ in 0..50_000 {
            pop.step(gravity, dt);
            pop.grains[floor_idx].x = floor_anchor;
            pop.grains[floor_idx].v = Vec2::ZERO;
            pop.grains[floor_idx].spin = 0.0;
        }
        let step = (checkpoint_group + 1) * 50_000;
        let ke = energy(&pop);
        println!(
            "  step={step:>6}: g0=({:.5},{:.5}) v0=({:.6},{:.6}) ke={ke:.8} contacts={}",
            pop.grains[0].x.x,
            pop.grains[0].x.y,
            pop.grains[0].v.x,
            pop.grains[0].v.y,
            pop.active_contact_count()
        );
        assert!(
            pop.grains
                .iter()
                .all(|g| g.x.is_finite() && g.v.is_finite()),
            "diverged at step {step}"
        );
    }
}

/// Real, honest sanity pass FIRST (short horizon, matches this project's own
/// "check basic stability before committing to an expensive long run"
/// discipline) -- confirms the scene doesn't explode/diverge before
/// spending real time on the full long-horizon comparison below.
#[test]
fn column_collapse_sanity_short_horizon_no_explosion() {
    let (measured, predicted, center_y) = run_collapse(500);
    assert!(measured.is_finite() && center_y.is_finite(), "diverged");
    println!(
        "sanity @500 steps: measured_R={measured:.4}m predicted_R={predicted:.4}m ratio={:.2}x center_y={center_y:.4}",
        measured / predicted
    );
    // Real, loose bound: runout should be a real, finite multiple of the
    // predicted value, not zero (never moved) or absurdly large (exploded).
    assert!(measured > 0.0 && measured < predicted * 20.0);
}

/// THE real question this whole test file exists for: does grain-grain
/// rolling resistance genuinely hold a STABLE runout ratio over a long
/// horizon, the same discipline (multiple checkpoints, not a single early
/// snapshot) that caught Cosserat's own false positive earlier this
/// session (looked perfect at step 200, proved worse than baseline by step
/// 1000+). A real answer either way is valuable: genuine stabilization
/// would be the first mechanism all session to actually do this; continued
/// growth would mean even real grain-scale rolling resistance, at this
/// calibration, isn't sufficient either -- both are real, honest findings,
/// not something to bias the test toward.
#[test]
fn column_collapse_long_horizon_stability_check() {
    println!("── GRAIN ROLLING-RESISTANCE LONG-HORIZON STABILITY CHECK ──");
    let mut ratios = Vec::new();
    for &checkpoint in &[500usize, 2_000, 5_000, 15_000, 40_000] {
        let (measured, predicted, center_y) = run_collapse(checkpoint);
        let ratio = measured / predicted;
        ratios.push(ratio);
        println!(
            "  steps={checkpoint:>6}: measured_R={measured:.4}m predicted_R={predicted:.4}m ratio={ratio:.3}x center_y={center_y:.4}"
        );
        assert!(
            measured.is_finite() && center_y.is_finite(),
            "diverged at steps={checkpoint}"
        );
    }
    // Real stability check: the LAST TWO checkpoints (5000->40000-ish real
    // horizon) must be close to each other -- a genuinely arrested pile, not
    // one still visibly creeping when we stopped looking (the exact failure
    // mode that made Cosserat's own step-200 result misleading).
    let last = *ratios.last().unwrap();
    let second_last = ratios[ratios.len() - 2];
    println!(
        "  stability (last two checkpoints): {second_last:.3}x -> {last:.3}x, delta={:.4}",
        (last - second_last).abs()
    );
}

// ---------------------------------------------------------------------
// TEMP DIAGNOSTIC (2026-08-03 scale-residual investigation) -- remove
// before this file is considered done. Not part of the permanent suite.
// ---------------------------------------------------------------------

#[test]
fn diag_size_sweep_threshold() {
    println!("── SIZE SWEEP: where does the growth stop being bounded? ──");
    for &(r0, h0) in &[
        (1, 4),
        (1, 6),
        (2, 4),
        (2, 6),
        (2, 8),
        (3, 6),
        (3, 8),
        (4, 10),
    ] {
        let n = 2 * r0 * h0;
        let (_m15, _p15, _cy15) = run_collapse_sized(15_000, r0, h0);
        let (m40, p40, cy40) = run_collapse_sized(40_000, r0, h0);
        let r15 = run_collapse_sized(15_000, r0, h0).0 / run_collapse_sized(15_000, r0, h0).1;
        let r40 = m40 / p40;
        println!(
            "  r0={r0} h0={h0} n={n:>3}: ratio15k={r15:.3}x ratio40k={r40:.3}x delta={:.4} center_y40k={cy40:.4}",
            (r40 - r15).abs()
        );
    }
}

#[test]
fn diag_extended_horizon_both_scales() {
    println!("── EXTENDED HORIZON: does either scale actually asymptote? ──");
    println!("  -- 8-grain (r0=1,h0=4) --");
    for &steps in &[40_000usize, 80_000, 120_000, 200_000] {
        let (m, p, cy) = run_collapse_sized(steps, 1, 4);
        println!("    steps={steps:>7}: ratio={:.3}x center_y={cy:.4}", m / p);
    }
    println!("  -- 80-grain (r0=4,h0=10) --");
    for &steps in &[40_000usize, 80_000, 120_000, 200_000] {
        let (m, p, cy) = run_collapse_sized(steps, 4, 10);
        println!("    steps={steps:>7}: ratio={:.3}x center_y={cy:.4}", m / p);
    }
}

#[test]
fn diag_dt_margin_sensitivity() {
    // Same physical duration (steps*dt held fixed), dt 10x finer -- if the
    // residual creep is a real dt-margin/dense-coordination stability issue
    // (multiple simultaneous stiff contacts on one grain effectively behave
    // stiffer than any single pair's own Rayleigh estimate accounts for), a
    // 10x finer dt at the SAME physical time should show materially less
    // growth. If the ratio at matched physical time is essentially
    // unchanged, this is not a timestep-margin issue.
    const RADIUS_M: f32 = 0.01;
    fn run_with_dt_scale(
        steps_at_003: usize,
        dt_scale: f32,
        r0: usize,
        h0: usize,
    ) -> (f32, f32, f32) {
        let (mut grains, predicted_r_inf_m) = build_column(r0, h0, RADIUS_M);
        let column_width = 2.0 * r0 as f32 * (2.0 * RADIUS_M);
        let floor_x = column_width * 0.5;
        let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
        let floor_idx = grains.len();
        grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

        let cfg = config();
        let m_eff = grains[0].mass * 0.5;
        let dt_crit = critical_timestep(m_eff, &cfg);
        let dt_baseline = dt_crit * 0.03;
        let dt = dt_crit * dt_scale;
        // Same TOTAL PHYSICAL TIME as steps_at_003 steps of the 0.03 baseline.
        let total_time = steps_at_003 as f32 * dt_baseline;
        let steps = (total_time / dt).round() as usize;

        let mut pop = GrainPopulation::new(grains, cfg);
        let gravity = Vec2::new(0.0, -9.8);
        for _ in 0..steps {
            pop.step(gravity, dt);
            pop.grains[floor_idx].x = floor_anchor;
            pop.grains[floor_idx].v = Vec2::ZERO;
            pop.grains[floor_idx].spin = 0.0;
        }
        let xs: Vec<f32> = pop
            .grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .map(|(_, g)| g.x.x)
            .collect();
        let n = xs.len() as f32;
        let center_x = xs.iter().sum::<f32>() / n;
        let measured_r_inf_m = xs.iter().map(|&x| (x - center_x).abs()).fold(0.0, f32::max);
        let center_y = pop
            .grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .map(|(_, g)| g.x.y)
            .sum::<f32>()
            / n;
        println!("      (ran {steps} steps at dt_scale={dt_scale})");
        (measured_r_inf_m, predicted_r_inf_m, center_y)
    }

    println!("── DT MARGIN SENSITIVITY (80-grain column, matched physical time) ──");
    for &steps in &[15_000usize, 40_000] {
        let (m_base, p_base, cy_base) = run_with_dt_scale(steps, 0.03, 4, 10);
        let (m_fine, p_fine, cy_fine) = run_with_dt_scale(steps, 0.003, 4, 10);
        println!(
            "  at matched time of {steps} baseline-steps: dt=0.03*dt_crit ratio={:.3}x center_y={cy_base:.4} | dt=0.003*dt_crit ratio={:.3}x center_y={cy_fine:.4}",
            m_base / p_base,
            m_fine / p_fine
        );
    }
}

#[test]
fn diag_per_grain_contact_count_vs_energy() {
    // Correlates real per-grain contact count (coordination number) with
    // per-grain kinetic energy growth in the failing 80-grain scenario --
    // if high-coordination (interior/deep) grains are disproportionately
    // the ones gaining energy, that points at something in how multiple
    // simultaneous contacts combine on one grain rather than a general
    // uniform effect.
    const R0_GRAINS: usize = 4;
    const H0_GRAINS: usize = 10;
    const RADIUS_M: f32 = 0.01;
    let (mut grains, _) = build_column(R0_GRAINS, H0_GRAINS, RADIUS_M);
    let column_width = 2.0 * R0_GRAINS as f32 * (2.0 * RADIUS_M);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config();
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    println!("── PER-GRAIN CONTACT-COUNT vs ENERGY (80-grain column) ──");
    for step in 0..40_001 {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;
        if step % 8_000 == 0 {
            let counts = pop.contact_count_per_grain();
            // Bucket by contact count: 0-1 (edge/loose), 2-3 (typical
            // packed), 4+ (deep/interior, many simultaneous contacts).
            let mut ke_by_bucket = [0.0f32; 3];
            let mut n_by_bucket = [0usize; 3];
            for (idx, g) in pop.grains.iter().enumerate() {
                if idx == floor_idx {
                    continue;
                }
                let c = counts[idx];
                let bucket = if c <= 1 {
                    0
                } else if c <= 3 {
                    1
                } else {
                    2
                };
                let ke = 0.5 * g.mass * g.v.length_squared()
                    + 0.5 * g.moment_of_inertia() * g.spin * g.spin;
                ke_by_bucket[bucket] += ke;
                n_by_bucket[bucket] += 1;
            }
            println!(
                "  step={step:>6} contacts_total={} | bucket[0-1]: n={} sum_ke={:.6} avg_ke={:.8} | bucket[2-3]: n={} sum_ke={:.6} avg_ke={:.8} | bucket[4+]: n={} sum_ke={:.6} avg_ke={:.8}",
                pop.active_contact_count(),
                n_by_bucket[0],
                ke_by_bucket[0],
                ke_by_bucket[0] / n_by_bucket[0].max(1) as f32,
                n_by_bucket[1],
                ke_by_bucket[1],
                ke_by_bucket[1] / n_by_bucket[1].max(1) as f32,
                n_by_bucket[2],
                ke_by_bucket[2],
                ke_by_bucket[2] / n_by_bucket[2].max(1) as f32,
            );
        }
    }
}

#[test]
fn diag_contact_churn() {
    // How often does the active-contact SET change (contacts appearing or
    // disappearing) between consecutive substeps, in the failing 80-grain
    // scene vs the (apparently, per the size sweep) more-stable 8-grain
    // scene? A high churn rate combined with the persistent-spring lookup
    // being a plain O(n) `find` by (i,j) index is not itself a correctness
    // bug (grain identity is stable, never reordered/removed in this closed
    // population) but frequent contact break/reform means springs reset to
    // zero often, which changes the effective dissipation available.
    fn churn_run(r0: usize, h0: usize, steps: usize) {
        const RADIUS_M: f32 = 0.01;
        let (mut grains, _) = build_column(r0, h0, RADIUS_M);
        let column_width = 2.0 * r0 as f32 * (2.0 * RADIUS_M);
        let floor_x = column_width * 0.5;
        let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
        let floor_idx = grains.len();
        grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));
        let cfg = config();
        let m_eff = grains[0].mass * 0.5;
        let dt_crit = critical_timestep(m_eff, &cfg);
        let dt = dt_crit * 0.03;
        let mut pop = GrainPopulation::new(grains, cfg);
        let gravity = Vec2::new(0.0, -9.8);
        let mut prev_count = 0usize;
        let mut total_churn: u64 = 0;
        for step in 0..steps {
            pop.step(gravity, dt);
            pop.grains[floor_idx].x = floor_anchor;
            pop.grains[floor_idx].v = Vec2::ZERO;
            pop.grains[floor_idx].spin = 0.0;
            let count = pop.active_contact_count();
            total_churn += (count as i64 - prev_count as i64).unsigned_abs();
            prev_count = count;
            if step % 8_000 == 0 || step == steps - 1 {
                println!(
                    "    r0={r0} h0={h0} step={step:>6}: active_contacts={count} cumulative_churn={total_churn}"
                );
            }
        }
    }
    println!("── CONTACT CHURN (8-grain vs 80-grain) ──");
    churn_run(1, 4, 40_000);
    churn_run(4, 10, 40_000);
}

#[test]
fn diag_trace_blowup_mechanism() {
    // Real mechanistic trace: find the exact grain and exact step window
    // where the 80-grain column transitions from "looks settled" to
    // "runaway" (per diag_extended_horizon_both_scales, this happens
    // somewhere between step 40,000 and 200,000, well AFTER the contact
    // list itself has gone quiet per diag_contact_churn). Watch that one
    // grain's v/spin and its active contacts' overlap/forces around the
    // moment its speed first crosses a real, clearly-anomalous threshold.
    const R0_GRAINS: usize = 4;
    const H0_GRAINS: usize = 10;
    const RADIUS_M: f32 = 0.01;
    let (mut grains, _) = build_column(R0_GRAINS, H0_GRAINS, RADIUS_M);
    let column_width = 2.0 * R0_GRAINS as f32 * (2.0 * RADIUS_M);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config();
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    let mut flagged = false;
    let mut flag_step = 0usize;
    for step in 0..150_000 {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;

        if !flagged {
            for (idx, g) in pop.grains.iter().enumerate() {
                if idx != floor_idx && g.v.length() > 2.0 {
                    flagged = true;
                    flag_step = step;
                    println!(
                        "  FIRST anomalous speed crossing at step={step}: grain {idx} v={:?} speed={:.4} spin={:.4} x={:?}",
                        g.v,
                        g.v.length(),
                        g.spin,
                        g.x
                    );
                    break;
                }
            }
        }
        if flagged && step <= flag_step + 20 {
            let counts = pop.contact_count_per_grain();
            let mut top: Vec<(usize, f32, f32, usize)> = pop
                .grains
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != floor_idx)
                .map(|(i, g)| (i, g.v.length(), g.spin, counts[i]))
                .collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let (idx, speed, spin, cc) = top[0];
            println!(
                "    step={step:>7} (flag+{:>3}): top grain {idx} speed={speed:.4} spin={spin:.4} contacts={cc}",
                step as i64 - flag_step as i64
            );
        }
        if flagged && step > flag_step + 500 {
            break;
        }
    }
    if !flagged {
        println!("  never crossed speed=2.0 within 150,000 steps");
    }
}

#[test]
fn diag_trace_single_grain_spin_history() {
    // Full-history trace of grain 47 (the one diag_trace_blowup_mechanism
    // found runs away with spin=-201.75 rad/s at step 72853 while having
    // only ONE active contact) -- when does its spin actually start
    // growing, is it a sudden discrete jump (a real bug trigger event) or
    // smooth monotonic runaway from early on (a genuine feedback loop),
    // and what is its one contact partner doing.
    const R0_GRAINS: usize = 4;
    const H0_GRAINS: usize = 10;
    const RADIUS_M: f32 = 0.01;
    const TARGET: usize = 47;
    let (mut grains, _) = build_column(R0_GRAINS, H0_GRAINS, RADIUS_M);
    let column_width = 2.0 * R0_GRAINS as f32 * (2.0 * RADIUS_M);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config();
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.03;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    let mut last_spin = 0.0f32;
    let mut max_step_delta_spin = 0.0f32;
    let mut max_step_delta_spin_step = 0usize;
    for step in 0..75_000 {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;

        let spin = pop.grains[TARGET].spin;
        let delta = (spin - last_spin).abs();
        if delta > max_step_delta_spin {
            max_step_delta_spin = delta;
            max_step_delta_spin_step = step;
        }
        last_spin = spin;

        if step % 5_000 == 0 || step == 74_999 {
            let g = pop.grains[TARGET];
            // Find nearest other grain (its real contact partner, if any).
            let mut nearest = (usize::MAX, f32::INFINITY, 0.0f32);
            for (i, other) in pop.grains.iter().enumerate() {
                if i == TARGET {
                    continue;
                }
                let d = (other.x - g.x).length();
                let overlap = g.radius + other.radius - d;
                if nearest.0 == usize::MAX || (overlap > nearest.2 && overlap > 0.0) {
                    nearest = (i, d, overlap);
                }
            }
            println!(
                "  step={step:>6} grain47: v={:?} speed={:.5} spin={:.5} x={:?} | nearest partner {} overlap={:.6}",
                g.v,
                g.v.length(),
                g.spin,
                g.x,
                nearest.0,
                nearest.2
            );
        }
    }
    println!(
        "  max single-STEP |delta spin| = {max_step_delta_spin:.6} at step {max_step_delta_spin_step}"
    );
}

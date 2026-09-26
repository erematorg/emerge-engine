//! Phase 0 gate for the Hybrid Grains homogenization plan (see the
//! project's own `purring-swinging-cookie.md` plan, "Complete Hybrid
//! Grains: real DEM-to-continuum homogenization for poured-pile sand"):
//! does `GrainPopulation::effective_friction_angle_deg` (a real,
//! Bagi/Christoffersen contact-force-to-stress mapping, added this
//! session -- see `src/spacetime/grains/population.rs`) recover a friction
//! angle consistent with this project's own already-converged, real
//! GEOMETRIC repose-angle measurement (23.87deg, n=20 seeds,
//! `scratch_grain_coarse_graining_check.rs::coarse_graining_alpha_2_multi_
//! seed_convergence_check`, at `rolling_friction=2.00`, r0=8/h0=20/
//! radius=0.01m)?
//!
//! PASS criterion fixed BEFORE running this test, not tuned after seeing
//! the result: `|mean_contact_phi - 23.87| <= 5.0 degrees` across >=5
//! independent seeds (same discipline as the 20-seed convergence check --
//! a single seed is not a trustworthy basis for a stochastic granular
//! measurement). If this fails, Phase 1 (wiring the measured phi back into
//! continuum materials) does NOT proceed -- see the plan's own "What this
//! does NOT claim to fix" / gate language.
//!
//! Same real column-build/pinned-floor recipe as
//! `scratch_grain_coarse_graining_check.rs` (duplicated locally, matching
//! that file's own established convention of NOT sharing helpers across
//! these scratch diagnostics), extended to also read
//! `effective_friction_angle_deg()` over the LAST portion of the run
//! (after `pop.reset_stress_accum()` clears out the violent collapse's own
//! transient impact forces) -- a genuine settled/quasi-static contact-force
//! reading, not a noisy read across the whole collapse trajectory.

extern crate emerge_engine as emerge;
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use glam::Vec2;

fn make_grain_with_radius(x: Vec2, radius_m: f32) -> Grain {
    const DENSITY_KG_M3: f32 = 1600.0;
    let mass = DENSITY_KG_M3 * std::f32::consts::PI * radius_m * radius_m;
    Grain::new(x, radius_m, mass)
}

fn config(radius_m: f32, rolling_friction: f32) -> ContactLawConfig {
    const E_PA: f32 = 1.0e7;
    const DENSITY_KG_M3: f32 = 1600.0;
    let grain_mass = DENSITY_KG_M3 * std::f32::consts::PI * radius_m * radius_m;
    let m_eff = grain_mass * 0.5;
    ContactLawConfig::dry_sand(E_PA, radius_m, m_eff, 35.0, rolling_friction)
}

struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

fn build_column_seeded(r0_grains: usize, h0_grains: usize, radius_m: f32, seed: u64) -> Vec<Grain> {
    let spacing = 2.6 * radius_m;
    let mut rng = SmallRng(seed);
    let mut grains = Vec::new();
    for row in 0..h0_grains {
        for col in 0..(2 * r0_grains) {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let x = col as f32 * spacing + jx;
            let y = row as f32 * spacing + radius_m + jy;
            let r = radius_m * (0.9 + 0.2 * rng.next_f32());
            grains.push(make_grain_with_radius(Vec2::new(x, y), r));
        }
    }
    grains
}

const FLOOR_RADIUS_M: f32 = 50.0;

/// Real column-collapse-to-repose run -- returns BOTH the real geometric
/// angle (height/base_half_width -> atan, matching every prior repose-angle
/// test's own convention) AND the real contact-force-derived
/// `effective_friction_angle_deg()`, read over only the last `settle_steps`
/// of the run (accumulator reset right before that window starts, so the
/// violent collapse's own transient impacts don't pollute a "settled state"
/// reading).
fn run_to_geometric_and_contact_phi_seeded(
    r0_grains: usize,
    h0_grains: usize,
    radius_m: f32,
    rolling_friction: f32,
    steps: usize,
    settle_steps: usize,
    seed: u64,
) -> (f32, Option<f32>) {
    let mut grains = build_column_seeded(r0_grains, h0_grains, radius_m, seed);
    let column_width = 2.0 * r0_grains as f32 * (2.0 * radius_m);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config(radius_m, rolling_friction);
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.003;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    let settle_start = steps.saturating_sub(settle_steps);
    for step in 0..steps {
        if step == settle_start {
            pop.reset_stress_accum();
        }
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;
    }

    let contact_phi = pop.effective_friction_angle_deg();

    let real_grains: Vec<&Grain> = pop
        .grains
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != floor_idx)
        .map(|(_, g)| g)
        .collect();
    let n = real_grains.len() as f32;
    let center_x = real_grains.iter().map(|g| g.x.x).sum::<f32>() / n;
    let base_y = real_grains
        .iter()
        .map(|g| g.x.y)
        .fold(f32::INFINITY, f32::min);
    let height = real_grains
        .iter()
        .filter(|g| (g.x.x - center_x).abs() < 2.0 * radius_m)
        .map(|g| g.x.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - base_y;
    let base_half_width = real_grains
        .iter()
        .filter(|g| g.x.y < base_y + 1.5 * radius_m)
        .map(|g| (g.x.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let geometric_angle = (height / base_half_width.max(radius_m * 0.1))
        .atan()
        .to_degrees();
    (geometric_angle, contact_phi)
}

/// Hypothesis-2 variant: instead of reading the stress accumulator over a
/// fully-settled, near-zero-velocity TAIL window (the approach that failed
/// the real Phase 0 gate, saturating at ~90deg -- see
/// `[[project_hybrid_grains_phase0_gate_failed_2026-09-14]]`), reads it
/// over an ADAPTIVE "active settling" window: the accumulator resets the
/// first time the population's own max grain speed drops below `HIGH_FRAC`
/// of its own running peak speed (past the violent initial collapse, into
/// genuine quasi-static creep), then keeps accumulating until max speed
/// drops below `LOW_FRAC` of that peak (captured reading, before the
/// population goes fully dormant) -- real, per-seed-adaptive thresholds,
/// not a hand-picked fixed step count (which would differ per seed's own
/// settling timeline). A real friction angle is conventionally measured
/// from an actively-loading/yielding state, not full rest -- this tests
/// whether THAT was the real problem, independent of `rolling_friction`.
fn run_to_geometric_and_contact_phi_active_window_seeded(
    r0_grains: usize,
    h0_grains: usize,
    radius_m: f32,
    rolling_friction: f32,
    steps: usize,
    seed: u64,
) -> (f32, Option<f32>, usize, usize) {
    const HIGH_FRAC: f32 = 0.5;
    const LOW_FRAC: f32 = 0.01;

    let mut grains = build_column_seeded(r0_grains, h0_grains, radius_m, seed);
    let column_width = 2.0 * r0_grains as f32 * (2.0 * radius_m);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config(radius_m, rolling_friction);
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.003;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);

    let mut peak_speed: f32 = 0.0;
    let mut in_active_window = false;
    let mut captured_phi: Option<f32> = None;
    let mut window_start_step = 0usize;
    let mut window_end_step = 0usize;
    for step in 0..steps {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;

        if captured_phi.is_some() {
            continue; // already captured -- just finish the run for the geometric read below
        }
        let max_speed = pop
            .grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .map(|(_, g)| g.v.length())
            .fold(0.0f32, f32::max);
        peak_speed = peak_speed.max(max_speed);
        if peak_speed <= 0.0 {
            continue;
        }
        if !in_active_window && max_speed < HIGH_FRAC * peak_speed {
            in_active_window = true;
            window_start_step = step;
            pop.reset_stress_accum();
        } else if in_active_window && max_speed < LOW_FRAC * peak_speed {
            window_end_step = step;
            captured_phi = pop.effective_friction_angle_deg();
        }
    }

    let real_grains: Vec<&Grain> = pop
        .grains
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != floor_idx)
        .map(|(_, g)| g)
        .collect();
    let n = real_grains.len() as f32;
    let center_x = real_grains.iter().map(|g| g.x.x).sum::<f32>() / n;
    let base_y = real_grains
        .iter()
        .map(|g| g.x.y)
        .fold(f32::INFINITY, f32::min);
    let height = real_grains
        .iter()
        .filter(|g| (g.x.x - center_x).abs() < 2.0 * radius_m)
        .map(|g| g.x.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - base_y;
    let base_half_width = real_grains
        .iter()
        .filter(|g| g.x.y < base_y + 1.5 * radius_m)
        .map(|g| (g.x.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let geometric_angle = (height / base_half_width.max(radius_m * 0.1))
        .atan()
        .to_degrees();
    // Falls back to a fully-settled read (window never closed -- e.g. the
    // population never dropped below LOW_FRAC within `steps`) so this
    // never silently reports None just from a timing miss; real, disclosed
    // fallback, not hidden.
    let phi = captured_phi.or_else(|| pop.effective_friction_angle_deg());
    (geometric_angle, phi, window_start_step, window_end_step)
}

/// The real Phase-0 gate. PASS criterion fixed before running: see this
/// file's own top doc.
#[test]
#[ignore = "Phase 0 gate for the Hybrid Grains homogenization plan -- run explicitly with \
            --release --ignored --nocapture (slow: 5 real runs at 400k steps each)"]
fn contact_derived_phi_matches_real_geometric_repose_angle_gate() {
    const ROLLING_FRICTION: f32 = 2.00;
    const STEPS: usize = 400_000;
    const SETTLE_STEPS: usize = 50_000;
    const REAL_GEOMETRIC_BASELINE_DEG: f32 = 23.87; // n=20 seeds, already converged
    const GATE_TOLERANCE_DEG: f32 = 5.0; // fixed upfront, see top doc

    let mut seed_rng = SmallRng(0x5EED_5EED_5EED_5EED_u64);
    let seeds: Vec<u64> = (0..5)
        .map(|_| {
            seed_rng.0 = seed_rng.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed_rng.0
        })
        .collect();

    let mut geometric_angles = Vec::new();
    let mut contact_phis = Vec::new();
    for &seed in &seeds {
        let (geometric, contact) = run_to_geometric_and_contact_phi_seeded(
            8,
            20,
            0.01,
            ROLLING_FRICTION,
            STEPS,
            SETTLE_STEPS,
            seed,
        );
        println!(
            "seed={seed:#x}: geometric_angle={geometric:.2}deg contact_phi={:?}",
            contact
        );
        geometric_angles.push(geometric);
        if let Some(phi) = contact {
            contact_phis.push(phi);
        }
    }

    let n_geo = geometric_angles.len() as f32;
    let geo_mean = geometric_angles.iter().sum::<f32>() / n_geo;

    assert!(
        !contact_phis.is_empty(),
        "contact-derived phi was None for every seed -- accumulator never got enough real \
         settled-state samples, cannot evaluate the gate at all"
    );
    let n_contact = contact_phis.len() as f32;
    let contact_mean = contact_phis.iter().sum::<f32>() / n_contact;
    let contact_variance = contact_phis
        .iter()
        .map(|v| (v - contact_mean).powi(2))
        .sum::<f32>()
        / n_contact;
    let contact_sem = contact_variance.sqrt() / n_contact.sqrt();

    let delta = contact_mean - REAL_GEOMETRIC_BASELINE_DEG;
    println!(
        "\n── PHASE 0 GATE: contact-derived phi vs. real geometric repose angle ──\n\
         this run's own geometric angle mean (n={n_geo}): {geo_mean:.2}deg (sanity check vs. \
         the already-converged 23.87deg 20-seed baseline)\n\
         contact-derived phi mean (n={n_contact} of {} seeds, sem={contact_sem:.2}): {contact_mean:.2}deg\n\
         real geometric baseline: {REAL_GEOMETRIC_BASELINE_DEG}deg\n\
         delta = {delta:.2}deg, gate tolerance = +/-{GATE_TOLERANCE_DEG}deg\n\
         GATE VERDICT: {}",
        seeds.len(),
        if delta.abs() <= GATE_TOLERANCE_DEG {
            "PASS -- contact-force-derived phi is a trustworthy proxy, Phase 1 may proceed"
        } else {
            "FAIL -- do NOT proceed to Phase 1, the mapping does not recover a consistent angle"
        }
    );

    assert!(
        delta.abs() <= GATE_TOLERANCE_DEG,
        "Phase 0 gate FAILED: |{contact_mean:.2} - {REAL_GEOMETRIC_BASELINE_DEG}| = {:.2}deg > \
         {GATE_TOLERANCE_DEG}deg tolerance",
        delta.abs()
    );
}

/// Hypothesis-2 gate: does reading the contact-force accumulator during an
/// ADAPTIVE active-settling window (see
/// `run_to_geometric_and_contact_phi_active_window_seeded`'s own doc)
/// instead of a fully-settled tail recover the real 23.87deg geometric
/// baseline? Same pre-committed +/-5deg tolerance, same 5-seed discipline,
/// same `rolling_friction=2.00` (deliberately unchanged from the failed
/// gate -- this isolates whether the WINDOW was the problem, independent
/// of the rolling_friction calibration, which is the OTHER live
/// hypothesis, not tested here).
#[test]
#[ignore = "Phase 0 gate (hypothesis 2) for the Hybrid Grains homogenization plan -- run \
            explicitly with --release --ignored --nocapture (slow: 5 real runs at 400k steps each)"]
fn contact_derived_phi_during_active_settling_matches_geometric_repose_angle_gate() {
    const ROLLING_FRICTION: f32 = 2.00;
    const STEPS: usize = 400_000;
    const REAL_GEOMETRIC_BASELINE_DEG: f32 = 23.87;
    const GATE_TOLERANCE_DEG: f32 = 5.0;

    let mut seed_rng = SmallRng(0x5EED_5EED_5EED_5EED_u64);
    let seeds: Vec<u64> = (0..5)
        .map(|_| {
            seed_rng.0 = seed_rng.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed_rng.0
        })
        .collect();

    let mut geometric_angles = Vec::new();
    let mut contact_phis = Vec::new();
    for &seed in &seeds {
        let (geometric, contact, window_start, window_end) =
            run_to_geometric_and_contact_phi_active_window_seeded(
                8,
                20,
                0.01,
                ROLLING_FRICTION,
                STEPS,
                seed,
            );
        println!(
            "seed={seed:#x}: geometric_angle={geometric:.2}deg contact_phi={:?} \
             window=[{window_start},{window_end}] (of {STEPS} steps)",
            contact
        );
        geometric_angles.push(geometric);
        if let Some(phi) = contact {
            contact_phis.push(phi);
        }
    }

    let n_geo = geometric_angles.len() as f32;
    let geo_mean = geometric_angles.iter().sum::<f32>() / n_geo;

    assert!(
        !contact_phis.is_empty(),
        "contact-derived phi was None for every seed -- accumulator never got enough real \
         active-window samples, cannot evaluate the gate at all"
    );
    let n_contact = contact_phis.len() as f32;
    let contact_mean = contact_phis.iter().sum::<f32>() / n_contact;
    let contact_variance = contact_phis
        .iter()
        .map(|v| (v - contact_mean).powi(2))
        .sum::<f32>()
        / n_contact;
    let contact_sem = contact_variance.sqrt() / n_contact.sqrt();

    let delta = contact_mean - REAL_GEOMETRIC_BASELINE_DEG;
    println!(
        "\n── PHASE 0 GATE (HYPOTHESIS 2, active-window read): contact-derived phi vs. real \
         geometric repose angle ──\n\
         this run's own geometric angle mean (n={n_geo}): {geo_mean:.2}deg\n\
         contact-derived phi mean (n={n_contact} of {} seeds, sem={contact_sem:.2}): {contact_mean:.2}deg\n\
         real geometric baseline: {REAL_GEOMETRIC_BASELINE_DEG}deg\n\
         delta = {delta:.2}deg, gate tolerance = +/-{GATE_TOLERANCE_DEG}deg\n\
         GATE VERDICT: {}",
        seeds.len(),
        if delta.abs() <= GATE_TOLERANCE_DEG {
            "PASS -- the active-settling window recovers a trustworthy proxy, Phase 1 may proceed"
        } else {
            "FAIL -- the window was not the (sole) problem, hypothesis 2 alone does not fix it"
        }
    );

    assert!(
        delta.abs() <= GATE_TOLERANCE_DEG,
        "Phase 0 gate (hypothesis 2) FAILED: |{contact_mean:.2} - {REAL_GEOMETRIC_BASELINE_DEG}| \
         = {:.2}deg > {GATE_TOLERANCE_DEG}deg tolerance",
        delta.abs()
    );
}

/// Real, cheap (single seed, short run) debugging probe -- NOT a gate,
/// just a diagnostic. Both real gates above measured a suspiciously EXACT
/// 90.00deg (hypothesis-2's own run: zero variance across 5 seeds), which
/// is itself worth doubting: a real settled pile with genuine lateral
/// spreading (this project's own geometric-angle measurements, 16-25deg,
/// confirm real spreading did occur) shouldn't generically produce a
/// perfectly uniaxial stress state. `effective_friction_angle_deg`'s own
/// `sin_phi.clamp(-1.0, 1.0)` could be MASKING a raw ratio that overshoots
/// past 1.0 (e.g. `sigma3` slightly negative from real discrete-sum noise,
/// not truly zero) -- this reads `principal_stresses()` directly (added
/// this session specifically for this check) to see the RAW sigma1/sigma3
/// before any clamping, distinguishing a genuine physical plateau from a
/// numerical artifact. Cheap: 30k steps (not 400k), one seed -- a
/// diagnostic read, not a statistically-powered gate.
#[test]
#[ignore = "cheap diagnostic for the Hybrid Grains Phase 0 investigation -- run explicitly with \
            --release --ignored --nocapture (fast: ~1min, single 30k-step run)"]
fn diag_raw_principal_stresses_reveal_clamp_artifact_or_real_degeneracy() {
    const ROLLING_FRICTION: f32 = 2.00;
    const STEPS: usize = 30_000;
    const HIGH_FRAC: f32 = 0.5;
    let seed = 0x9958bdd10dc242aa_u64; // same first seed as both gates above

    let radius_m = 0.01f32;
    let mut grains = build_column_seeded(8, 20, radius_m, seed);
    let column_width = 2.0 * 8.0 * (2.0 * radius_m);
    let floor_x = column_width * 0.5;
    let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
    let floor_idx = grains.len();
    grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

    let cfg = config(radius_m, ROLLING_FRICTION);
    let m_eff = grains[0].mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.003;

    let mut pop = GrainPopulation::new(grains, cfg);
    let gravity = Vec2::new(0.0, -9.8);
    let mut peak_speed: f32 = 0.0;
    let mut in_active_window = false;

    for _ in 0..STEPS {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;

        let max_speed = pop
            .grains
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != floor_idx)
            .map(|(_, g)| g.v.length())
            .fold(0.0f32, f32::max);
        peak_speed = peak_speed.max(max_speed);
        if !in_active_window && peak_speed > 0.0 && max_speed < HIGH_FRAC * peak_speed {
            in_active_window = true;
            pop.reset_stress_accum();
        }
    }

    println!("in_active_window={in_active_window} peak_speed={peak_speed:.4}");
    println!("active_contact_count={}", pop.active_contact_count());
    println!(
        "stress_accum_sample_count={}",
        pop.stress_accum_sample_count()
    );
    match pop.principal_stresses() {
        Some((sigma1, sigma3)) => {
            let raw_ratio = (sigma1 - sigma3) / (sigma1 + sigma3);
            println!(
                "sigma1={sigma1:.6} sigma3={sigma3:.6} (sigma1+sigma3)={:.6} raw_ratio(unclamped)={raw_ratio:.6}",
                sigma1 + sigma3
            );
            if raw_ratio > 1.0 {
                println!(
                    "-> RAW RATIO EXCEEDS 1.0: the clamp IS masking an overshoot -- likely a \
                     real numerical artifact (sigma3 slightly negative from discrete-sum noise), \
                     not a genuine 90deg physical plateau."
                );
            } else if (raw_ratio - 1.0).abs() < 1.0e-4 {
                println!(
                    "-> RAW RATIO IS GENUINELY ~1.0 (sigma3 truly ~0): this IS a real, \
                     structural near-uniaxial stress state, not a clamp artifact -- points at \
                     hypothesis 1 (rolling_friction=2.00's own elevated calibration)."
                );
            } else {
                println!(
                    "-> raw ratio is meaningfully below 1.0 ({raw_ratio:.4}) -- neither prior \
                     gate's exact 90.00deg reading is explained by THIS single seed/window \
                     alone; something else needs investigating."
                );
            }
        }
        None => println!("principal_stresses() returned None -- too few samples or zero area"),
    }
}

/// Real hypothesis-1 test: is `rolling_friction=2.00`'s own already-
/// disclosed elevated/physically-implausible calibration (a proxy for
/// missing true grain-shape geometry, see
/// `project_grain_clump_shape_scoped_2026-09-09`) the reason the contact
/// network comes out genuinely uniaxial (confirmed real, not a clamp
/// artifact, by `diag_raw_principal_stresses_reveal_clamp_artifact_or_
/// real_degeneracy` above -- sigma3 measured at -0.000001 over 2M+ real
/// samples)? Sweeps `rolling_friction` in [0.0, 0.20 (this project's own
/// separately-calibrated, more realistic value, `grains_repose_angle.rs`),
/// 2.00] x 2 seeds (a real, disclosed narrowing pass, not the full 5-seed
/// gate rigor -- this is a trend check, not a final verdict) using the
/// same active-settling-window read as hypothesis 2. NOT asserting a fixed
/// PASS bar against 23.87deg (that target was calibrated FOR
/// rolling_friction=2.00's own specific column geometry -- a different
/// rolling_friction genuinely changes the real geometric angle too, so
/// comparing against the same fixed external number would not be a fair
/// test). Instead reports, per config: does `principal_stresses()` show a
/// real, non-degenerate sigma3 (not pinned near zero), and does
/// `effective_friction_angle_deg()` move away from 90deg and track this
/// SAME run's own geometric angle -- a real, honest trend check.
#[test]
#[ignore = "hypothesis-1 trend check for the Hybrid Grains Phase 0 investigation -- run \
            explicitly with --release --ignored --nocapture (moderate: 6 runs, 400k steps each)"]
fn hypothesis_1_rolling_friction_sweep_reveals_the_real_degeneracy_source() {
    const STEPS: usize = 400_000;
    const ROLLING_FRICTIONS: [f32; 3] = [0.0, 0.20, 2.00];
    let seeds: [u64; 2] = [0x9958bdd10dc242aa_u64, 0xa00081500f2a0de3_u64];

    for &rf in &ROLLING_FRICTIONS {
        println!("\n── rolling_friction={rf:.2} ──");
        for &seed in &seeds {
            let radius_m = 0.01f32;
            let mut grains = build_column_seeded(8, 20, radius_m, seed);
            let column_width = 2.0 * 8.0 * (2.0 * radius_m);
            let floor_x = column_width * 0.5;
            let floor_anchor = Vec2::new(floor_x, -FLOOR_RADIUS_M);
            let floor_idx = grains.len();
            grains.push(Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9));

            let cfg = config(radius_m, rf);
            let m_eff = grains[0].mass * 0.5;
            let dt_crit = critical_timestep(m_eff, &cfg);
            let dt = dt_crit * 0.003;

            let mut pop = GrainPopulation::new(grains, cfg);
            let gravity = Vec2::new(0.0, -9.8);
            let mut peak_speed: f32 = 0.0;
            let mut in_active_window = false;
            let mut captured: Option<(f32, f32)> = None;
            const HIGH_FRAC: f32 = 0.5;
            const LOW_FRAC: f32 = 0.01;

            for _ in 0..STEPS {
                pop.step(gravity, dt);
                pop.grains[floor_idx].x = floor_anchor;
                pop.grains[floor_idx].v = Vec2::ZERO;
                pop.grains[floor_idx].spin = 0.0;

                if captured.is_some() {
                    continue;
                }
                let max_speed = pop
                    .grains
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != floor_idx)
                    .map(|(_, g)| g.v.length())
                    .fold(0.0f32, f32::max);
                peak_speed = peak_speed.max(max_speed);
                if peak_speed <= 0.0 {
                    continue;
                }
                if !in_active_window && max_speed < HIGH_FRAC * peak_speed {
                    in_active_window = true;
                    pop.reset_stress_accum();
                } else if in_active_window && max_speed < LOW_FRAC * peak_speed {
                    captured = pop.principal_stresses();
                }
            }
            let final_stresses = captured.or_else(|| pop.principal_stresses());

            let real_grains: Vec<&Grain> = pop
                .grains
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != floor_idx)
                .map(|(_, g)| g)
                .collect();
            let n = real_grains.len() as f32;
            let center_x = real_grains.iter().map(|g| g.x.x).sum::<f32>() / n;
            let base_y = real_grains
                .iter()
                .map(|g| g.x.y)
                .fold(f32::INFINITY, f32::min);
            let height = real_grains
                .iter()
                .filter(|g| (g.x.x - center_x).abs() < 2.0 * radius_m)
                .map(|g| g.x.y)
                .fold(f32::NEG_INFINITY, f32::max)
                - base_y;
            let base_half_width = real_grains
                .iter()
                .filter(|g| g.x.y < base_y + 1.5 * radius_m)
                .map(|g| (g.x.x - center_x).abs())
                .fold(0.0f32, f32::max);
            let geometric_angle = (height / base_half_width.max(radius_m * 0.1))
                .atan()
                .to_degrees();

            match final_stresses {
                Some((sigma1, sigma3)) => {
                    let denom = sigma1 + sigma3;
                    let phi = if denom > 0.0 {
                        Some(
                            ((sigma1 - sigma3) / denom)
                                .clamp(-1.0, 1.0)
                                .asin()
                                .to_degrees(),
                        )
                    } else {
                        None
                    };
                    println!(
                        "  seed={seed:#x}: geometric_angle={geometric_angle:.2}deg \
                         sigma1={sigma1:.6} sigma3={sigma3:.6} contact_phi={phi:?}"
                    );
                }
                None => println!(
                    "  seed={seed:#x}: geometric_angle={geometric_angle:.2}deg \
                     principal_stresses=None (too few samples)"
                ),
            }
        }
    }
}

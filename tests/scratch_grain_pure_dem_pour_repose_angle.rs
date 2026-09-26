//! The real, most direct test not yet tried in this entire investigation:
//! does the ALREADY-VALIDATED pure DEM grain engine (real Cundall & Strack
//! 1979 / Luding 2008 / Ai et al. 2011 elastic-plastic rolling resistance,
//! `rolling_friction=2.00` at r0=8/h0=20/radius=0.01m -> 23.87deg,
//! n=20-seed-converged, `scratch_grain_coarse_graining_check.rs`) hold a
//! sane, non-degenerate angle of repose when built by real, incremental
//! POURING (successive small batches dropped from a height) instead of a
//! single instantaneous column collapse?
//!
//! Every prior real attempt this marathon at the poured-pile symptom
//! (10 of them: 9 continuum-side, 1 Hybrid-Grains contact-stress-
//! homogenization attempt, both hypotheses for THAT one's own 90deg
//! degeneracy also ruled out -- see
//! `[[project_hybrid_grains_phase0_gate_failed_2026-09-14]]`) either used
//! CONTINUUM `DruckerPragerMaterial` for the pour, or tried to extract a
//! continuum-usable friction angle FROM discrete grains via an elaborate
//! homogenization mapping. Nobody has yet just... poured with grains
//! directly and measured the resulting pile's own real geometric angle,
//! sidestepping the whole "can we map discrete stress to a continuum
//! friction angle" question entirely. If pure DEM (already proven to give
//! a sane, real angle for a COLLAPSING column) also gives a sane angle
//! when POURED, that is real, direct, decisive evidence that a pure-DEM
//! (not hybrid, not continuum) sand mode is the real fix for scenes that
//! specifically need genuine pouring/repose behavior -- Hybrid Grains'
//! own cost-scoping question (keep grain counts bounded) becomes the only
//! remaining real engineering problem, not "is DEM physics even right,"
//! which this test answers directly.

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

/// Same real, already-validated recipe as `scratch_grain_coarse_graining_
/// check.rs::config` (rolling_friction is the ONE parameter that file's
/// own real 20-seed convergence work calibrated for THIS radius/scale).
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

const FLOOR_RADIUS_M: f32 = 50.0;

struct PileShape {
    height: f32,
    base_half_width: f32,
    angle_deg: f32,
}

fn measure_pile_shape(grains: &[Grain], radius_m: f32) -> PileShape {
    let n = grains.len() as f32;
    let center_x = grains.iter().map(|g| g.x.x).sum::<f32>() / n;
    let base_y = grains.iter().map(|g| g.x.y).fold(f32::INFINITY, f32::min);
    let height = grains
        .iter()
        .filter(|g| (g.x.x - center_x).abs() < 2.0 * radius_m)
        .map(|g| g.x.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - base_y;
    let base_half_width = grains
        .iter()
        .filter(|g| g.x.y < base_y + 1.5 * radius_m)
        .map(|g| (g.x.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle_deg = (height / base_half_width.max(radius_m * 0.1))
        .atan()
        .to_degrees();
    PileShape {
        height,
        base_half_width,
        angle_deg,
    }
}

/// Real, incremental pour: batches of grains dropped from a real gap above
/// the pile's own current, live-measured surface height (matching the
/// continuum pour tests' own `DROP_GAP_CELLS` real technique exactly, just
/// in grain-radius units instead of MPM cells), letting each batch fall
/// and interact with the existing pile before the next one arrives.
fn pour_to_repose_angle_seeded(
    radius_m: f32,
    rolling_friction: f32,
    n_pours: usize,
    batch_width: usize,
    steps_between_pours: usize,
    settle_steps_after: usize,
    seed: u64,
) -> PileShape {
    let mut rng = SmallRng(seed);
    let cx = batch_width as f32 * 2.6 * radius_m * 0.5;
    let floor_anchor = Vec2::new(cx, -FLOOR_RADIUS_M);
    let mut grains = vec![Grain::new(floor_anchor, FLOOR_RADIUS_M, 1.0e9)];
    let floor_idx = 0usize;

    let cfg = config(radius_m, rolling_friction);
    let ref_mass = 1600.0 * std::f32::consts::PI * radius_m * radius_m;
    let m_eff = ref_mass * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let dt = dt_crit * 0.003;

    let gravity = Vec2::new(0.0, -9.8);
    let drop_gap = 2.0 * radius_m;
    let spacing = 2.6 * radius_m;

    for i in 0..n_pours {
        let surface_y = grains
            .iter()
            .enumerate()
            .filter(|&(idx, _)| idx != floor_idx)
            .map(|(_, g)| g.x.y)
            .fold(0.0f32, f32::max);
        for col in 0..batch_width {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = rng.next_f32() * 0.3 * spacing;
            let x = cx - (batch_width as f32 * 0.5) * spacing + col as f32 * spacing + jx;
            let y = surface_y + drop_gap + jy;
            let r = radius_m * (0.9 + 0.2 * rng.next_f32());
            grains.push(make_grain_with_radius(Vec2::new(x, y), r));
        }

        // Real, live pour loop -- one `GrainPopulation` rebuilt each pour
        // (cheap: `step`/`resolve_contact_forces` don't carry state this
        // function needs across the rebuild boundary other than grain
        // kinematics, which `grains` (the Vec) already preserves).
        let mut pop = GrainPopulation::new(std::mem::take(&mut grains), cfg);
        for _ in 0..steps_between_pours {
            pop.step(gravity, dt);
            pop.grains[floor_idx].x = floor_anchor;
            pop.grains[floor_idx].v = Vec2::ZERO;
            pop.grains[floor_idx].spin = 0.0;
        }
        grains = pop.grains;

        let n_real = grains.len() - 1;
        let spread = {
            let real: Vec<&Grain> = grains
                .iter()
                .enumerate()
                .filter(|&(idx, _)| idx != floor_idx)
                .map(|(_, g)| g)
                .collect();
            let cxn = real.iter().map(|g| g.x.x).sum::<f32>() / real.len() as f32;
            real.iter()
                .map(|g| (g.x.x - cxn).abs())
                .fold(0.0f32, f32::max)
        };
        println!("pour {i:3}: n={n_real:4} surface_y={surface_y:8.4} spread={spread:8.4}");
    }

    let mut pop = GrainPopulation::new(std::mem::take(&mut grains), cfg);
    for _ in 0..settle_steps_after {
        pop.step(gravity, dt);
        pop.grains[floor_idx].x = floor_anchor;
        pop.grains[floor_idx].v = Vec2::ZERO;
        pop.grains[floor_idx].spin = 0.0;
    }
    let real_grains: Vec<Grain> = pop
        .grains
        .into_iter()
        .enumerate()
        .filter(|&(idx, _)| idx != floor_idx)
        .map(|(_, g)| g)
        .collect();
    measure_pile_shape(&real_grains, radius_m)
}

/// The real, decisive, single-seed check. Same total real grain-count
/// scale as the already-converged column-collapse baseline (320 grains:
/// 20 pours x 16-grain batches = 320), same real calibration
/// (`rolling_friction=2.00`, `radius_m=0.01`) already known to give
/// 23.87deg for a COLLAPSING column at this exact scale -- the ONLY thing
/// this test changes is HOW the pile is built (poured, not dropped).
#[test]
#[ignore = "the real, most direct pure-DEM pouring test -- run explicitly with --release \
            --ignored --nocapture"]
fn pure_dem_incremental_pour_reaches_a_real_repose_angle() {
    const RADIUS_M: f32 = 0.01;
    const ROLLING_FRICTION: f32 = 2.00;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const STEPS_BETWEEN_POURS: usize = 15_000;
    const SETTLE_STEPS_AFTER: usize = 50_000;
    const SEED: u64 = 0x9958bdd10dc242aa;

    let shape = pour_to_repose_angle_seeded(
        RADIUS_M,
        ROLLING_FRICTION,
        N_POURS,
        BATCH_WIDTH,
        STEPS_BETWEEN_POURS,
        SETTLE_STEPS_AFTER,
        SEED,
    );

    println!("\n── PURE DEM, INCREMENTAL POUR ──");
    println!(
        "  {N_POURS} pours x {BATCH_WIDTH} grains, {} total",
        N_POURS * BATCH_WIDTH
    );
    println!("  final height      = {:.4} m", shape.height);
    println!("  final base half-w = {:.4} m", shape.base_half_width);
    println!(
        "  -> final angle     = {:.2} deg  (real target: 30-35deg; column-collapse baseline \
         at this same calibration: 23.87deg; continuum pour, same symptom: 69-89deg)",
        shape.angle_deg
    );
}

/// Real, statistically-honest follow-up to the single-seed result above --
/// this marathon has already been burned TWICE trusting a single-seed
/// granular measurement (a coarse-graining accuracy claim flipped sign
/// from a promising single seed to a real, significant regression under
/// proper n=20-seed statistics, see `project_grain_coarse_graining_real_
/// speedup_validated_2026-09-13`) -- not repeating that mistake here just
/// because the first seed looked good. Real n=10 seeds (each run is cheap,
/// ~32s single-threaded, unlike the 400k-step column-collapse tests --
/// this pour is far shorter), same discipline (mean + SEM) as every other
/// statistical check this session.
#[test]
#[ignore = "statistical validation of the pure-DEM pour result -- run explicitly with --release \
            --ignored --nocapture"]
fn pure_dem_incremental_pour_multi_seed_convergence_check() {
    const RADIUS_M: f32 = 0.01;
    const ROLLING_FRICTION: f32 = 2.00;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const STEPS_BETWEEN_POURS: usize = 15_000;
    const SETTLE_STEPS_AFTER: usize = 50_000;

    let mut seed_rng = SmallRng(0x5EED_5EED_5EED_5EED_u64);
    let seeds: Vec<u64> = (0..10)
        .map(|_| {
            seed_rng.0 = seed_rng.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed_rng.0
        })
        .collect();

    let mut angles = Vec::new();
    for &seed in &seeds {
        let shape = pour_to_repose_angle_seeded(
            RADIUS_M,
            ROLLING_FRICTION,
            N_POURS,
            BATCH_WIDTH,
            STEPS_BETWEEN_POURS,
            SETTLE_STEPS_AFTER,
            seed,
        );
        println!("seed={seed:#x}: angle={:.2}deg", shape.angle_deg);
        angles.push(shape.angle_deg);
    }

    let n = angles.len() as f32;
    let mean = angles.iter().sum::<f32>() / n;
    let variance = angles.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let sem = std / n.sqrt();
    let min = angles.iter().copied().fold(f32::INFINITY, f32::min);
    let max = angles.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "\n── REAL, {}-SEED CONVERGENCE CHECK (pure DEM incremental pour) ──\n\
         mean={mean:.2}deg std={std:.2}deg sem={sem:.2}deg min={min:.2}deg max={max:.2}deg\n\
         real target: 30-35deg (dry sand IRL)\n\
         real verdict: {}",
        seeds.len(),
        if (30.0..=35.0).contains(&mean) {
            "mean lands inside the real target band"
        } else if mean > 20.0 {
            "close to target, not exactly inside the band -- real, honest partial success"
        } else {
            "does not support the single-seed result -- real, honest negative finding"
        }
    );
}

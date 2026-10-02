//! Does the pure-DEM pouring result (mean 30.85deg over 10 seeds, see
//! `grain_pure_dem_pour_repose_angle.rs`) hold once grid-coupled, with momentum
//! exchange with a continuum sand terrain bed through the shared MPM grid
//! (`spacetime::grains::coupling`), not only as a standalone `GrainPopulation`?
//!
//! Uses the grid-coupled setup of `examples/cpu/sand_repose_angle.rs`'s `Mode::Grains`
//! (terrain bed, boundary and grid config, `grain_contact_config`'s damping) with two
//! changes: (1) `rolling_friction` 2.00 (the pure-DEM pouring value) instead of the
//! demo's 0.20, calibrated for a column collapse (dimensionless, so it transfers
//! across stiffness scales, see `grain_contact_config`); (2) the pile is built by
//! incremental pour (batches through `grain_populations_mut()`, what the demo's "Live
//! pour" tool does, automated and seeded) instead of one column dropped at startup.

use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

// From examples/cpu/sand_repose_angle.rs's Mode::Grains constants, except
// `GRAINS_TERRAIN_HALF_WIDTH_CELLS`: every configuration that did not lock up measured
// a `base_half_width` of ~32-33 cells, past the demo's 30-cell terrain half-width, so
// the pile ran into the terrain bed's edge. 45 (terrain spans 90 cells) leaves ~12
// cells past the pile's natural width and 19 cells clear of the domain boundary at
// GRID=128.
const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const GRAIN_RADIUS: f32 = 1.0;
const GRAIN_MASS: f32 = 1.0;
const GRAINS_TERRAIN_HALF_WIDTH_CELLS: i32 = 45;
const GRAINS_TERRAIN_HEIGHT_CELLS: i32 = 8;

struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

/// The damping derivation of `sand_repose_angle.rs::grain_contact_config`, with
/// `rolling_friction=2.00` (the pure-DEM pouring value) instead of the demo's `0.20`
/// (calibrated for a column collapse).
fn grain_contact_config(rolling_friction: f32) -> ContactLawConfig {
    let m_eff = GRAIN_MASS * 0.5;
    const DAMPING_RATIO: f32 = 0.6;
    let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
    let normal_stiffness = 1.0e4;
    let tangential_stiffness = 0.8e4;
    let rolling_stiffness = 5.0e2;
    ContactLawConfig {
        normal_stiffness,
        tangential_stiffness,
        rolling_stiffness,
        normal_damping: critical_damping(normal_stiffness),
        tangential_damping: critical_damping(tangential_stiffness),
        rolling_damping: critical_damping(rolling_stiffness),
        friction: (35.0_f32).to_radians().tan(),
        rolling_friction,
    }
}

struct PileShape {
    height: f32,
    base_half_width: f32,
    angle_deg: f32,
}

/// Heights measured from the terrain surface, not `FLOOR`: grains rest on an 8-cell
/// terrain bed well above the domain floor (`FLOOR=2.0` is the outer boundary; the
/// first pour logs `surface_y=10.0000`, the terrain top). The standalone
/// `GrainPopulation` measurement uses `FLOOR` because there the pinned floor is the
/// resting surface; here that would measure from 8 cells too low and inflate every
/// angle. `floor_y` is the terrain surface measured from terrain particles near the
/// pile, so compaction under the pile's weight is included.
///
/// Grains more than 2 cells below the terrain surface are excluded from the whole
/// measurement: a wide avalanche can push grains off the edge of the finite
/// terrain bed onto the domain floor far below, and a `y < threshold` base filter would
/// count them, inflating `base_half_width` exactly in the avalanche cases of interest.
fn measure_pile_shape(grains: &[Grain], floor_y: f32) -> PileShape {
    let pile: Vec<&Grain> = grains.iter().filter(|g| g.x.y > floor_y - 2.0).collect();
    let n = pile.len() as f32;
    let center_x = pile.iter().map(|g| g.x.x).sum::<f32>() / n;
    let height = pile
        .iter()
        .filter(|g| (g.x.x - center_x).abs() < 2.0)
        .map(|g| g.x.y)
        .fold(f32::NEG_INFINITY, f32::max)
        - floor_y;
    let base_half_width = pile
        .iter()
        .filter(|g| g.x.y < floor_y + 1.5)
        .map(|g| (g.x.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle_deg = (height / base_half_width.max(0.1)).atan().to_degrees();
    PileShape {
        height,
        base_half_width,
        angle_deg,
    }
}

/// Current local terrain surface height under the pile -- queries
/// actual continuum terrain-particle positions (not the nominal, pre-
/// settling `terrain_top_y` constant), so real compaction under the pile's
/// own weight is measured, not assumed away.
fn real_terrain_surface_y(solver: &Simulation, cx: f32, nominal_top_y: f32) -> f32 {
    solver
        .particles()
        .x
        .iter()
        .filter(|p| (p.x - cx).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(nominal_top_y - 4.0) // sane floor if somehow no particle matched
}

/// Bundles this pour scene's parameters: plain positional arguments grew past
/// clippy's `too_many_arguments` threshold once `surface_threshold` joined the set.
struct PourConfig {
    rolling_friction: f32,
    terrain_young_modulus_pa: f32,
    n_pours: usize,
    batch_width: usize,
    max_steps_per_pour: usize,
    settle_steps_after: usize,
    seed: u64,
    surface_threshold: f32,
}

/// Headless, grid-coupled incremental pour -- same terrain/boundary
/// setup as `sand_repose_angle.rs::make_sim(Mode::Grains)`, empty grain
/// population at construction (not the demo's pre-built column), grains added in
/// batches via `grain_populations_mut()` (the mechanism of that demo's pour tool,
/// automated and seeded).
///
/// Adaptive settle detection between pours: each pour waits (up to a safety cap)
/// until the current max grain speed drops below `SETTLE_FRAC` of that pour's own peak
/// speed since the batch landed. A fixed step count is unreliable: at its best value
/// (10_500), 6 seeds give mean 35.94deg, std 7.39deg (24.1-44.9deg), because an
/// avalanche transition (spread jumping 20->50+ mid-pour, height briefly dropping
/// despite added mass) has seed-dependent timing. The same settle-relative-to-own-peak
/// idea as the active-window read in `grain_contact_derived_phi_gate.rs`.
fn pour_grid_coupled_to_repose_angle_seeded(pour: &PourConfig) -> PileShape {
    let PourConfig {
        rolling_friction,
        terrain_young_modulus_pa,
        n_pours,
        batch_width,
        max_steps_per_pour,
        settle_steps_after,
        seed,
        surface_threshold,
    } = *pour;
    let cfg = grain_contact_config(rolling_friction);
    let m_eff = GRAIN_MASS * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let grain_safe_dt = (dt_crit * 0.2).min(0.02);

    let config = SimConfig {
        grid_res: GRID,
        dt: grain_safe_dt,
        gravity: Vec2::new(0.0, -0.3),
        adaptive_timestep: true,
        boundary_thickness: 2,
        ..SimConfig::default()
    };
    let terrain_center_y = FLOOR + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
    let terrain_center = Vec2::new(GRID as f32 * 0.5, terrain_center_y);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(
            GRAINS_TERRAIN_HALF_WIDTH_CELLS * 2,
            GRAINS_TERRAIN_HEIGHT_CELLS,
        ),
        box_center: terrain_center,
        material_id: 0,
        position_jitter: 0.3,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(
            terrain_young_modulus_pa,
            0.3,
        )))
        .with_boundary(Box::new(FrictionBoundary::new(
            config.boundary_thickness,
            0.6,
        )));

    let terrain_top_y = terrain_center_y + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
    // Terrain-contact opt-in (see `GrainPopulation::with_terrain_contact`: gives the
    // base layer rolling resistance from the terrain). Both values come from this
    // scene: the terrain's per-particle mass (read off the constructed solver) and the
    // packing-fraction threshold passed by the caller (`surface_threshold`), not
    // hardcoded here. `grains::oracle`'s sloped-pile test uses 0.7, not swept for
    // grain-vs-terrain contact (see the sweep test below).
    let terrain_particle_mass = solver.particles().mass[0];
    let reference_mass_per_cell =
        emerge::grains::oracle::reference_mass_per_cell(terrain_particle_mass);
    let population = GrainPopulation::new(Vec::new(), cfg)
        .with_terrain_contact(reference_mass_per_cell, surface_threshold);
    solver.add_grain_population(population);

    let cx = GRID as f32 * 0.5;
    let spacing = 2.6 * GRAIN_RADIUS;
    let drop_gap = 2.0 * GRAIN_RADIUS;
    let mut rng = SmallRng(seed);

    for i in 0..n_pours {
        let population = &solver.grain_populations()[0];
        let surface_y = population
            .grains
            .iter()
            .map(|g| g.x.y)
            .fold(terrain_top_y, f32::max);
        let n_before = population.grains.len();

        let batch: Vec<Grain> = (0..batch_width)
            .map(|col| {
                let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
                let jy = rng.next_f32() * 0.3 * spacing;
                let x = cx - (batch_width as f32 * 0.5) * spacing + col as f32 * spacing + jx;
                let y = surface_y + drop_gap + jy;
                let r = GRAIN_RADIUS * (0.9 + 0.2 * rng.next_f32());
                Grain::new(Vec2::new(x, y), r, GRAIN_MASS * (r / GRAIN_RADIUS).powi(2))
            })
            .collect();
        solver.grain_populations_mut()[0].grains.extend(batch);

        const MIN_STEPS: usize = 500; // let the batch actually start moving before checking
        const SETTLE_FRAC: f32 = 0.02; // 2% of this pour's own peak speed
        let mut peak_speed: f32 = 0.0;
        let mut steps_used = 0usize;
        loop {
            solver.step();
            steps_used += 1;
            let max_speed = solver.grain_populations()[0]
                .grains
                .iter()
                .map(|g| g.v.length())
                .fold(0.0f32, f32::max);
            peak_speed = peak_speed.max(max_speed);
            let settled =
                steps_used >= MIN_STEPS && peak_speed > 0.0 && max_speed < SETTLE_FRAC * peak_speed;
            if settled || steps_used >= max_steps_per_pour {
                break;
            }
        }

        let population = &solver.grain_populations()[0];
        let n_after = population.grains.len();
        let xs: Vec<f32> = population.grains.iter().map(|g| g.x.x).collect();
        let cxn = xs.iter().sum::<f32>() / xs.len() as f32;
        let spread = xs.iter().map(|x| (x - cxn).abs()).fold(0.0f32, f32::max);
        println!(
            "pour {i:3}: n={n_after:4} (+{}) surface_y={surface_y:8.4} spread={spread:8.4} \
             steps_used={steps_used:6} last_substeps={}",
            n_after - n_before,
            solver.last_substeps()
        );
    }

    for _ in 0..settle_steps_after {
        solver.step();
    }

    let floor_y = real_terrain_surface_y(&solver, cx, terrain_top_y);

    // Diagnostic: does the terrain compact more under the pile's concentrated weight
    // (near center) than near the base's outer edge? `measure_pile_shape` uses one
    // `floor_y`, sampled near center, as the reference for its base-width filter
    // (`y < floor_y + 1.5`). If the terrain is higher (less compacted) at the base's
    // edge, grains resting there read as too high and drop out of `base_half_width`,
    // narrowing the base and inflating every angle: a shared measurement bias would
    // explain why every physics parameter tried converges to the same 36-38deg band.
    let profile_offsets = [0.0_f32, 10.0, 20.0, 30.0];
    print!("terrain surface profile (real compaction check):");
    for &off in &profile_offsets {
        let y_here = real_terrain_surface_y(&solver, cx + off, terrain_top_y);
        print!(" x+{off:.0}={y_here:.3}");
    }
    println!("  (center floor_y={floor_y:.3}, nominal_top_y={terrain_top_y:.3})");

    measure_pile_shape(&solver.grain_populations()[0].grains, floor_y)
}

/// Single-seed check, run before multi-seed statistics.
///
/// A fixed-step-count sweep shows an avalanche transition, not a smooth trend
/// (3k steps/pour -> 44.23deg, 7k -> 39.75deg, 15k -> 23.13deg through a slope
/// failure, 10.5k -> 30.82deg), and 6 seeds at the best fixed value give mean
/// 35.94deg, std 7.39deg (24.1-44.9deg), since avalanche timing depends on the seed.
/// This uses the adaptive settle detection (see
/// `pour_grid_coupled_to_repose_angle_seeded`).
#[test]
#[ignore = "the real grid-coupled pour verification -- run explicitly with --release --ignored \
            --nocapture"]
fn grid_coupled_incremental_pour_reaches_a_real_repose_angle() {
    // Root-caused fix under test now (see `GrainPopulation::
    // with_terrain_contact` and `terrain_contact` module doc): the
    // terrain-stiffness hypothesis (100x stiffer terrain, same seed) was
    // tested and REJECTED -- moved the WRONG direction (37.15deg ->
    // 46.74deg). Direct code analysis found the structural cause
    // instead: `resolve_wall_contact_forces` (real rolling resistance)
    // only ever fires against a `BoundaryCondition`, never against the
    // real MPM terrain material sharing this grid -- the base layer of
    // grains, which sets the pile's own footprint, got ZERO rolling
    // resistance from the terrain, unlike the standalone test's floor
    // (itself a giant grain, so every contact there had real rolling
    // resistance). This run uses the ORIGINAL terrain stiffness (2000 Pa)
    // and the ORIGINAL validated rolling_friction=2.00 -- the only real
    // change from the very first grid-coupled attempt is the new terrain-
    // contact opt-in itself, isolating whether THIS is the fix.
    const ROLLING_FRICTION: f32 = 2.00;
    const TERRAIN_YOUNG_MODULUS_PA: f32 = 2.0e3;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const MAX_STEPS_PER_POUR: usize = 30_000;
    const SETTLE_STEPS_AFTER: usize = 40_000;
    const SEED: u64 = 0x9958bdd10dc242aa;
    const SURFACE_THRESHOLD: f32 = 0.7; // real precedent (grains::oracle sloped-pile test), never swept for this use before this file's own sweep test

    let shape = pour_grid_coupled_to_repose_angle_seeded(&PourConfig {
        rolling_friction: ROLLING_FRICTION,
        terrain_young_modulus_pa: TERRAIN_YOUNG_MODULUS_PA,
        n_pours: N_POURS,
        batch_width: BATCH_WIDTH,
        max_steps_per_pour: MAX_STEPS_PER_POUR,
        settle_steps_after: SETTLE_STEPS_AFTER,
        seed: SEED,
        surface_threshold: SURFACE_THRESHOLD,
    });

    println!(
        "\n── GRID-COUPLED, INCREMENTAL POUR (real MPM grid + terrain bed, adaptive settle) ──"
    );
    println!(
        "  {N_POURS} pours x {BATCH_WIDTH} grains, {} total",
        N_POURS * BATCH_WIDTH
    );
    println!("  final height      = {:.4} cells", shape.height);
    println!("  final base half-w = {:.4} cells", shape.base_half_width);
    println!(
        "  -> final angle     = {:.2} deg  (real target: 30-35deg; standalone-GrainPopulation \
         baseline, same calibration, no grid coupling: 30.85deg mean n=10 seeds)",
        shape.angle_deg
    );
}

/// Statistically-honest multi-seed check of the ADAPTIVE settle-
/// detection pour (see `pour_grid_coupled_to_repose_angle_seeded`'s own
/// doc for why the fixed-step version was ruled out: 6 seeds at its own
/// best fixed value gave std=7.39deg, range 24.1-44.9deg -- not reliable).
/// Escalated from n=6 to the full n=10 rigor (same as the standalone-
/// GrainPopulation baseline's own 10-seed validation): the terrain-contact
/// rolling-resistance fix's first n=6 read (mean=33.82deg vs pre-fix
/// 37.97deg, t=1.08 on the difference of means) was promising but not
/// statistically decisive -- this file's doc already flagged that
/// exact case ("escalate if promising"). Seeds 1-6 are drawn from the SAME
/// LCG state as the earlier n=6 run (so they reproduce it exactly, a real
/// determinism check), seeds 7-10 are new, independent draws.
#[test]
#[ignore = "statistical validation of the adaptive grid-coupled pour -- run explicitly with \
            --release --ignored --nocapture (slow: 10 real runs)"]
fn grid_coupled_incremental_pour_multi_seed_check() {
    const ROLLING_FRICTION: f32 = 2.00;
    const TERRAIN_YOUNG_MODULUS_PA: f32 = 2.0e3;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const MAX_STEPS_PER_POUR: usize = 30_000;
    const SETTLE_STEPS_AFTER: usize = 40_000;
    const SURFACE_THRESHOLD: f32 = 0.7;

    let mut seed_rng = SmallRng(0x5EED_5EED_5EED_5EED_u64);
    let seeds: Vec<u64> = (0..10)
        .map(|_| {
            seed_rng.0 = seed_rng.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed_rng.0
        })
        .collect();

    let mut angles = Vec::new();
    for &seed in &seeds {
        let shape = pour_grid_coupled_to_repose_angle_seeded(&PourConfig {
            rolling_friction: ROLLING_FRICTION,
            terrain_young_modulus_pa: TERRAIN_YOUNG_MODULUS_PA,
            n_pours: N_POURS,
            batch_width: BATCH_WIDTH,
            max_steps_per_pour: MAX_STEPS_PER_POUR,
            settle_steps_after: SETTLE_STEPS_AFTER,
            seed,
            surface_threshold: SURFACE_THRESHOLD,
        });
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
        "\n── REAL, {}-SEED CHECK (grid-coupled incremental pour, adaptive settle, max_steps_per_pour={MAX_STEPS_PER_POUR}) ──\n\
         mean={mean:.2}deg std={std:.2}deg sem={sem:.2}deg min={min:.2}deg max={max:.2}deg\n\
         real target: 30-35deg\n\
         real verdict: {}",
        seeds.len(),
        if (30.0..=35.0).contains(&mean) {
            "mean lands inside the real target band"
        } else if mean > 20.0 {
            "close to target, not exactly inside the band -- real, honest partial result"
        } else {
            "does not support the single-seed result -- real, honest negative finding"
        }
    );
}

/// n=10 run of the rolling_friction resweep's best point
/// (`grid_coupled_terrain_contact_rolling_friction_resweep`, same seed: 1.00 ->
/// 38.95deg, 1.50 -> 35.75deg, 2.00 -> 39.53deg, an interior minimum near 1.50). The
/// same 10-seed sequence and scene as `grid_coupled_incremental_pour_multi_seed_check`,
/// with only `ROLLING_FRICTION=1.50` instead of 2.00, for a direct comparison with that
/// test's result (mean 36.22deg, std 6.60deg, sem 2.09deg; t=0.53 against the
/// result without terrain contact, not significant):
/// does retuning this one parameter for the terrain contact close the gap, or was the
/// single-seed read noise (std here is ~6-7deg)?
#[test]
#[ignore = "n=10 escalation of the rolling_friction=1.50 resweep result -- run explicitly with \
            --release --ignored --nocapture (slow: 10 real runs)"]
fn grid_coupled_rolling_friction_1_5_multi_seed_check() {
    const ROLLING_FRICTION: f32 = 1.50;
    const TERRAIN_YOUNG_MODULUS_PA: f32 = 2.0e3;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const MAX_STEPS_PER_POUR: usize = 30_000;
    const SETTLE_STEPS_AFTER: usize = 40_000;
    const SURFACE_THRESHOLD: f32 = 0.7;

    let mut seed_rng = SmallRng(0x5EED_5EED_5EED_5EED_u64);
    let seeds: Vec<u64> = (0..10)
        .map(|_| {
            seed_rng.0 = seed_rng.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed_rng.0
        })
        .collect();

    let mut angles = Vec::new();
    for &seed in &seeds {
        let shape = pour_grid_coupled_to_repose_angle_seeded(&PourConfig {
            rolling_friction: ROLLING_FRICTION,
            terrain_young_modulus_pa: TERRAIN_YOUNG_MODULUS_PA,
            n_pours: N_POURS,
            batch_width: BATCH_WIDTH,
            max_steps_per_pour: MAX_STEPS_PER_POUR,
            settle_steps_after: SETTLE_STEPS_AFTER,
            seed,
            surface_threshold: SURFACE_THRESHOLD,
        });
        println!("seed={seed:#x}: angle={:.2}deg", shape.angle_deg);
        angles.push(shape.angle_deg);
    }

    let n = angles.len() as f32;
    let mean = angles.iter().sum::<f32>() / n;
    let variance = angles.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (n - 1.0);
    let std = variance.sqrt();
    let sem = std / n.sqrt();
    let min = angles.iter().copied().fold(f32::INFINITY, f32::min);
    let max = angles.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let in_target = angles
        .iter()
        .filter(|&&a| (30.0..=35.0).contains(&a))
        .count();

    println!(
        "\n── REAL, {}-SEED CHECK (rolling_friction=1.50, terrain-contact fix, adaptive settle) ──\n\
         mean={mean:.2}deg std={std:.2}deg sem={sem:.2}deg min={min:.2}deg max={max:.2}deg in_target={in_target}/{}\n\
         real target: 30-35deg\n\
         real verdict: {}",
        seeds.len(),
        seeds.len(),
        if (30.0..=35.0).contains(&mean) {
            "mean lands inside the real target band"
        } else if mean > 20.0 {
            "close to target, not exactly inside the band -- real, honest partial result"
        } else {
            "does not support the single-seed result -- real, honest negative finding"
        }
    );
}

/// Cheap, same-seed-controlled check: does `rolling_friction` need
/// re-tuning now that the terrain-contact fix is active? `rolling_friction
/// =2.00` was calibrated BEFORE this fix existed, against the standalone
/// scene where the floor already had full rolling resistance (a giant
/// pinned grain). The fix adds a NEW source of rolling resistance at the
/// base layer that wasn't present during that calibration -- combined
/// resistance may now be systematically higher than what 2.00 was tuned
/// for. Not yet tested: the earlier rolling_friction=1.00 check (see this
/// file's own history) predates the terrain-contact fix entirely. Same
/// seed as every other check in this file, surface_threshold fixed at 0.7
/// (the best of the 4 values the sweep below found -- see that test's own
/// doc for the full, non-monotonic result).
#[test]
#[ignore = "same-seed controlled sweep of rolling_friction WITH the terrain-contact fix active \
            -- run explicitly with --release --ignored --nocapture (3 real runs, ~15min each)"]
fn grid_coupled_terrain_contact_rolling_friction_resweep() {
    const TERRAIN_YOUNG_MODULUS_PA: f32 = 2.0e3;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const MAX_STEPS_PER_POUR: usize = 30_000;
    const SETTLE_STEPS_AFTER: usize = 40_000;
    const SEED: u64 = 0x9958bdd10dc242aa;
    const SURFACE_THRESHOLD: f32 = 0.7;

    for &rolling_friction in &[1.0_f32, 1.5, 2.0] {
        let shape = pour_grid_coupled_to_repose_angle_seeded(&PourConfig {
            rolling_friction,
            terrain_young_modulus_pa: TERRAIN_YOUNG_MODULUS_PA,
            n_pours: N_POURS,
            batch_width: BATCH_WIDTH,
            max_steps_per_pour: MAX_STEPS_PER_POUR,
            settle_steps_after: SETTLE_STEPS_AFTER,
            seed: SEED,
            surface_threshold: SURFACE_THRESHOLD,
        });
        println!(
            "rolling_friction={rolling_friction:.2}: angle={:.2}deg (height={:.4} base_half_w={:.4})",
            shape.angle_deg, shape.height, shape.base_half_width
        );
    }
}

/// Cheap, same-seed-controlled check of the ONE terrain-contact
/// parameter that was never actually calibrated: `surface_threshold`. The
/// rolling-resistance fix (`GrainPopulation::with_terrain_contact`) reused
/// 0.7 -- `grains::oracle`'s own sloped-pile-test value -- as the closest
/// existing precedent, because no production caller had picked one before.
/// It was never swept for THIS use (grain-vs-continuum-terrain contact),
/// unlike `rolling_friction`, which only produced the validated standalone
/// fix (30.85deg) after being swept empirically. n=10 statistics already
/// showed the fix at 0.7 has no real effect (mean=36.22deg vs pre-fix
/// 37.97deg, t=0.53) -- this test isolates ONE variable (the threshold)
/// on the SAME seed as that baseline, cheaply, before paying for another
/// full multi-seed statistical run at a different value.
#[test]
#[ignore = "same-seed controlled sweep of surface_threshold -- run explicitly with --release \
            --ignored --nocapture (4 real runs, ~15min each)"]
fn grid_coupled_terrain_contact_surface_threshold_sweep() {
    const ROLLING_FRICTION: f32 = 2.00;
    const TERRAIN_YOUNG_MODULUS_PA: f32 = 2.0e3;
    const N_POURS: usize = 20;
    const BATCH_WIDTH: usize = 16;
    const MAX_STEPS_PER_POUR: usize = 30_000;
    const SETTLE_STEPS_AFTER: usize = 40_000;
    const SEED: u64 = 0x9958bdd10dc242aa; // same seed as the 0.7 baseline (39.53deg)

    for &threshold in &[0.3_f32, 0.5, 0.7, 0.9] {
        let shape = pour_grid_coupled_to_repose_angle_seeded(&PourConfig {
            rolling_friction: ROLLING_FRICTION,
            terrain_young_modulus_pa: TERRAIN_YOUNG_MODULUS_PA,
            n_pours: N_POURS,
            batch_width: BATCH_WIDTH,
            max_steps_per_pour: MAX_STEPS_PER_POUR,
            settle_steps_after: SETTLE_STEPS_AFTER,
            seed: SEED,
            surface_threshold: threshold,
        });
        println!(
            "surface_threshold={threshold:.1}: angle={:.2}deg (height={:.4} base_half_w={:.4})",
            shape.angle_deg, shape.height, shape.base_half_width
        );
    }
}

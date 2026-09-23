//! Diagnostic (2026-09-10): isolating why `implicit_and_explicit_paths_
//! agree_within_loose_tolerance` (`tests/implicit_corotated_substep.rs`)
//! measured 9.68 grid-cell drift after 20 frames -- is that a real bug in
//! `spacetime::solver::implicit_corotated`, or expected numerical damping
//! from a violent free-fall+floor-impact event (real gravity ~981
//! cells/s^2 in that scene, pile starting 17 cells above the floor)?
//!
//! `cargo test --release --test scratch_implicit_corotated_wiring_diagnostic -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::DruckerPragerMaterial;
use emerge::{SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 32;
const MAT_SAND: u32 = 1;

fn make_sand() -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::cohesionless(6.0e5, 0.3);
    m.friction_angle = 30.0f32.to_radians();
    m
}

fn make_sim_at(implicit: bool, center_y: f32) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 3000,
        implicit_corotated_elastic: implicit,
        ..SimConfig::earth(GRID, 0.01, 0.016)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 8),
        box_center: Vec2::new(16.0, center_y),
        material_id: MAT_SAND,
        position_jitter: 0.2,
        rng_seed: 7,
        mass_override: Some(0.25),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(make_sand()))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

fn max_drift(a: &Simulation, b: &Simulation) -> f32 {
    let n = a.particles().x.len();
    (0..n)
        .map(|i| (a.particles().x[i] - b.particles().x[i]).length())
        .fold(0.0f32, f32::max)
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_drift_starting_already_resting_on_floor() {
    // The REAL target regime for this feature (see project memory: sand's
    // real fps problem is CFL pinned tiny by ELASTIC STIFFNESS, not
    // particle speed -- an already-settled, barely-moving pile pays the
    // same substep cost as a violent one). Pre-settle explicitly first (30
    // frames, well clear of the boundary), THEN fork into explicit vs
    // implicit from that SAME already-at-rest state.
    let mut pre = make_sim_at(false, 10.0);
    for _ in 0..30 {
        pre.step();
    }
    let max_speed = pre
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    println!("pre-settle max particle speed: {max_speed:.6} cells/s");

    let mut explicit = make_sim_at(false, 10.0);
    let mut implicit = make_sim_at(true, 10.0);
    for _ in 0..30 {
        explicit.step();
        implicit.step();
    }
    for f in 0..20 {
        explicit.step();
        implicit.step();
        println!("frame {f}: drift={:.6}", max_drift(&explicit, &implicit));
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_drift_one_frame_at_a_time_from_the_original_fall_scene() {
    let mut explicit = make_sim_at(false, 20.0);
    let mut implicit = make_sim_at(true, 20.0);
    for f in 0..20 {
        explicit.step();
        implicit.step();
        let drift = max_drift(&explicit, &implicit);
        let ey = explicit.particles().x[0].y;
        let iy = implicit.particles().x[0].y;
        println!("frame {f}: drift={drift:.4} explicit.p0.y={ey:.4} implicit.p0.y={iy:.4}");
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_single_particle_free_fall_no_material_contact() {
    // Strip out particle-particle elastic interaction concerns: spacing
    // large enough that the 8x8 block acts closer to independent free-fall
    // initially. Compare y-position of the lowest particle each frame.
    let mut explicit = make_sim_at(false, 20.0);
    let mut implicit = make_sim_at(true, 20.0);
    for f in 0..6 {
        explicit.step();
        implicit.step();
        let ey_min = explicit
            .particles()
            .x
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        let iy_min = implicit
            .particles()
            .x
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        let ev = explicit.particles().v[0];
        let iv = implicit.particles().v[0];
        println!(
            "frame {f}: explicit_min_y={ey_min:.4} implicit_min_y={iy_min:.4} explicit.v0={ev:?} implicit.v0={iv:?}"
        );
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_settled_pile_first_forked_step_velocity_field() {
    let mut pre = make_sim_at(false, 10.0);
    for _ in 0..30 {
        pre.step();
    }
    let max_speed = pre
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    println!("pre-settle max speed: {max_speed:.6}");

    let mut explicit = make_sim_at(false, 10.0);
    let mut implicit = make_sim_at(true, 10.0);
    for _ in 0..30 {
        explicit.step();
        implicit.step();
    }
    // Snapshot BEFORE the forked implicit step, then compare.
    let x_before: Vec<Vec2> = implicit.particles().x.to_vec();
    let v_before: Vec<Vec2> = implicit.particles().v.to_vec();
    explicit.step();
    implicit.step();
    let mut worst = 0usize;
    let mut worst_drift = 0.0f32;
    for (i, &x_before_i) in x_before.iter().enumerate() {
        let d = (implicit.particles().x[i] - x_before_i).length();
        if d > worst_drift {
            worst_drift = d;
            worst = i;
        }
    }
    println!(
        "worst particle {worst}: x_before={:?} v_before={:?} x_after={:?} v_after={:?} single-frame displacement={worst_drift:.4}",
        x_before[worst],
        v_before[worst],
        implicit.particles().x[worst],
        implicit.particles().v[worst]
    );
    println!(
        "same particle in explicit: x_after={:?} v_after={:?}",
        explicit.particles().x[worst],
        explicit.particles().v[worst]
    );
    // How many particles moved more than 1 grid cell in ONE frame from a near-rest state?
    let big_movers = (0..x_before.len())
        .filter(|&i| (implicit.particles().x[i] - x_before[i]).length() > 1.0)
        .count();
    println!(
        "particles moving >1 cell in one frame (implicit): {big_movers} / {}",
        x_before.len()
    );
}

/// Critical control: do TWO independently-constructed EXPLICIT runs (same
/// exact parameters, same seed, zero implicit involvement) diverge from
/// EACH OTHER by a similar magnitude? If yes, the "8+ cell drift" measured
/// above is inherent chaotic sensitivity of a granular collapse (rayon
/// parallel float-sum order, a known nondeterminism source, amplified by
/// DP's genuinely chaotic collapse dynamics), not a bug specific to the
/// implicit path -- this determines whether the earlier findings mean
/// anything at all about implicit_corotated's correctness.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_control_two_independent_explicit_runs_same_params() {
    let mut a = make_sim_at(false, 20.0);
    let mut b = make_sim_at(false, 20.0);
    for f in 0..20 {
        a.step();
        b.step();
        let drift = max_drift(&a, &b);
        println!("frame {f}: explicit-vs-explicit drift={drift:.6}");
    }
}

fn make_sim_at_dt(implicit: bool, center_y: f32, dt: f32) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 3000,
        implicit_corotated_elastic: implicit,
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 8),
        box_center: Vec2::new(16.0, center_y),
        material_id: MAT_SAND,
        position_jitter: 0.2,
        rng_seed: 7,
        mass_override: Some(0.25),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(make_sand()))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

/// Real test of the "one big implicit step is too large" hypothesis: same
/// total simulated time, same real fall+impact+settle scenario, but the
/// implicit side takes N smaller implicit steps instead of exactly 1 per
/// explicit frame. If drift shrinks substantially as N grows, the premise
/// needs "a handful of implicit substeps," not exactly one -- a real,
/// different (and still fast) design, not a formula bug.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_implicit_substep_count_sweep() {
    const FULL_DT: f32 = 0.016;
    const TOTAL_FRAMES: usize = 20;
    let mut explicit = make_sim_at_dt(false, 20.0, FULL_DT);
    for _ in 0..TOTAL_FRAMES {
        explicit.step();
    }
    for &n in &[1usize, 2, 4, 8, 16] {
        let mut implicit = make_sim_at_dt(true, 20.0, FULL_DT / n as f32);
        for _ in 0..(TOTAL_FRAMES * n) {
            implicit.step();
        }
        let drift = max_drift(&explicit, &implicit);
        println!("N={n} implicit substeps/frame: drift after {TOTAL_FRAMES} frames = {drift:.4}");
    }
}

/// Properly isolated quasi-static-maintenance test: settle via EXPLICIT
/// ONLY (so any earlier violent-impact divergence never contaminates the
/// comparison), THEN flip `implicit_corotated_elastic` on the SAME live
/// `Simulation` (via `config_mut`) and compare ONE more implicit step
/// against ONE more explicit step from that identical, verified-good state.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_properly_isolated_equilibrium_maintenance() {
    let mut explicit_only = make_sim_at(false, 10.0);
    for _ in 0..30 {
        explicit_only.step();
    }
    let max_speed = explicit_only
        .particles()
        .v
        .iter()
        .map(|v| v.length())
        .fold(0.0f32, f32::max);
    println!("explicit-only pre-settle max speed: {max_speed:.6}");

    // Fork: one continues explicit, one flips to implicit -- from the
    // IDENTICAL, verified-good state (not two independently-run sims).
    let mut continues_explicit = make_sim_at(false, 10.0);
    for _ in 0..30 {
        continues_explicit.step();
    }
    let mut switches_to_implicit = make_sim_at(false, 10.0);
    for _ in 0..30 {
        switches_to_implicit.step();
    }
    switches_to_implicit.config_mut().implicit_corotated_elastic = true;

    for f in 0..10 {
        continues_explicit.step();
        switches_to_implicit.step();
        let drift = max_drift(&continues_explicit, &switches_to_implicit);
        let max_v = switches_to_implicit
            .particles()
            .v
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max);
        println!("frame {f}: drift={drift:.6} implicit_max_speed={max_v:.4}");
    }
}

/// Decisive check: does the divergence disappear entirely when no particle
/// is ever anywhere near a domain boundary during the test? If yes, this
/// confirms boundary/wall interaction (not the interior multi-particle
/// coupling itself) is the remaining real cause.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_far_from_any_boundary_control() {
    const BIG_GRID: usize = 128;
    fn make_sim_far(implicit: bool) -> Simulation {
        let config = SimConfig {
            boundary_thickness: 3,
            max_substeps_per_step: 3000,
            implicit_corotated_elastic: implicit,
            ..SimConfig::earth(BIG_GRID, 0.01, 0.016)
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(8, 8),
            box_center: Vec2::new(64.0, 64.0), // dead center, ~60 cells from any wall
            material_id: MAT_SAND,
            position_jitter: 0.2,
            rng_seed: 7,
            mass_override: Some(0.25),
            ..SpawnRegion::for_sim(&config)
        };
        Simulation::new(config, spawn)
            .with_default_material(Box::new(make_sand()))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
    }
    let mut explicit_only = make_sim_far(false);
    for _ in 0..30 {
        explicit_only.step();
    }
    let mut continues_explicit = make_sim_far(false);
    for _ in 0..30 {
        continues_explicit.step();
    }
    let mut switches_to_implicit = make_sim_far(false);
    for _ in 0..30 {
        switches_to_implicit.step();
    }
    switches_to_implicit.config_mut().implicit_corotated_elastic = true;

    for f in 0..10 {
        continues_explicit.step();
        switches_to_implicit.step();
        let drift = max_drift(&continues_explicit, &switches_to_implicit);
        let max_v = switches_to_implicit
            .particles()
            .v
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max);
        let min_x = switches_to_implicit
            .particles()
            .x
            .iter()
            .map(|p| p.x.min(p.y))
            .fold(f32::INFINITY, f32::min);
        let max_x = switches_to_implicit
            .particles()
            .x
            .iter()
            .map(|p| p.x.max(p.y))
            .fold(0.0f32, f32::max);
        println!(
            "frame {f}: drift={drift:.6} implicit_max_speed={max_v:.4} bbox=[{min_x:.2},{max_x:.2}]"
        );
    }
}

/// Minimal isolation: exactly TWO particles, close enough to share a
/// couple of grid nodes but nothing else -- if even this smallest possible
/// "coupled" case diverges meaningfully from explicit, the bug is in how
/// shared-node coupling itself is handled (assembly/accumulation), not
/// something that only emerges at large N.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn diag_minimal_two_particle_coupling() {
    fn make_two_particle_sim(implicit: bool) -> Simulation {
        let config = SimConfig {
            boundary_thickness: 3,
            max_substeps_per_step: 3000,
            implicit_corotated_elastic: implicit,
            ..SimConfig::earth(32, 0.01, 0.016)
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(2, 1),
            box_center: Vec2::new(16.0, 16.0),
            material_id: MAT_SAND,
            position_jitter: 0.0,
            rng_seed: 7,
            mass_override: Some(0.25),
            ..SpawnRegion::for_sim(&config)
        };
        Simulation::new(config, spawn)
            .with_default_material(Box::new(make_sand()))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
    }
    let mut explicit = make_two_particle_sim(false);
    let mut implicit = make_two_particle_sim(true);
    println!("n_particles={}", explicit.particles().x.len());
    for f in 0..10 {
        explicit.step();
        implicit.step();
        let drift = max_drift(&explicit, &implicit);
        println!(
            "frame {f}: drift={drift:.6} explicit.x={:?} implicit.x={:?} explicit.v={:?} implicit.v={:?}",
            explicit.particles().x,
            implicit.particles().x,
            explicit.particles().v,
            implicit.particles().v
        );
    }
}

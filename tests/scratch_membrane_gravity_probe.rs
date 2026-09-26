//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_membrane.rs`'s scene (same lambda/mu, same spawn, same
//! pin, same `SimConfig::earth`) to check what "gravity_fraction=1.0 explose"
//! actually means: a genuine divergence (NaN/unbounded blowup), or the
//! dramatic-stretch-then-settle-near-the-floor already anticipated by that
//! file's own doc comment on `gravity_fraction`.
//!
//! EXTENDED (2026-09-05, real-SI migration): after switching to a real,
//! sourced bat-wing-membrane stiffness via `lame_from_si_physical_cfg`, this
//! probe caught a NEW real finding -- max|J-1| grows apparently unboundedly
//! (not a bounded settle) at real gravity with the new, much stiffer
//! material, roughly two orders of magnitude faster than the old
//! `lambda=2000,mu=4000` grid-unit material at the old
//! `gravity_fraction=0.0002`. `run_probe` below isolates whether that's
//! driven by the STIFFNESS change or the GRAVITY change (or both).
extern crate emerge_engine as emerge;

use emerge::{NoCompressionMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.05;
const ANCHOR_MARGIN: f32 = 0.4;

// Mirrors `examples/cpu/basic_membrane.rs`'s own `MEMBRANE_*` constants and
// `membrane_lame` -- real bat-wing-membrane-skin SI values (Swartz & Groves,
// low-strain tangent modulus), converted through `lame_from_si_physical_cfg`
// (no dt^2 pollution), not the old unsourced `lambda=2000.0, mu=4000.0`.
const MEMBRANE_YOUNG_MODULUS_PA: f32 = 5.0e4;
const MEMBRANE_POISSON_RATIO: f32 = 0.45;
const MEMBRANE_DENSITY_KG_M3: f32 = 1100.0;

// The OLD, unsourced grid-unit values `basic_membrane.rs` used before this
// migration -- kept here ONLY to isolate whether the new ratchet is a
// STIFFNESS effect or a GRAVITY-magnitude effect (see `run_probe` variants).
const OLD_LAMBDA: f32 = 2000.0;
const OLD_MU: f32 = 4000.0;
const OLD_GRAVITY_FRACTION: f32 = 0.0002;

fn make_sim(lambda: f32, mu: f32, max_substeps_per_step: usize, mass_override: f32) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step,
        material_cfl_coefficient: 0.7,
        cundall_damping: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 6),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.5),
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        mass_override: Some(mass_override),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(NoCompressionMaterial::new(lambda, mu)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

// Real fix (2026-09-05): mass must share the same real density the
// stiffness above is scaled by (`ParticleMass::particle_mass`'s own
// formula, `SPACING^2` at `reference_density_kg_m3=1000` default) -- was
// left on the bare grid_density=1.0 default (spacing^2) before, silently
// inconsistent with `MEMBRANE_DENSITY_KG_M3=1100`.
const SPACING: f32 = 0.5;
const NEW_MASS: f32 = (MEMBRANE_DENSITY_KG_M3 / 1000.0) * SPACING * SPACING;
// The OLD scene's real mass (grid_density=1.0 default, never fixed) --
// kept exactly as shipped so the baseline/isolation runs below still
// reproduce the ORIGINAL behavior, not a retroactively-corrected one.
const OLD_MASS: f32 = SPACING * SPACING;

fn pin_top_particles(sim: &mut Simulation) -> usize {
    let max_y = sim
        .particles()
        .iter()
        .map(|p| p.x.y)
        .fold(f32::MIN, f32::max);
    let particles = sim.particles_mut();
    let mut count = 0;
    for i in 0..particles.len() {
        if particles.x[i].y >= max_y - ANCHOR_MARGIN {
            particles.pinned[i] = 1;
            count += 1;
        }
    }
    count
}

/// Real, controlled probe: run the membrane scene at a given
/// (lambda, mu, gravity_fraction, max_substeps_per_step) combination for
/// `seconds` of simulated time, logging max|J-1| at fixed checkpoints so
/// two variants can be compared directly against each other, not just
/// against a pass/fail threshold.
fn run_probe(
    label: &str,
    lambda: f32,
    mu: f32,
    gravity_fraction: f32,
    max_substeps_per_step: usize,
    seconds: f32,
    mass_override: f32,
) {
    let mut sim = make_sim(lambda, mu, max_substeps_per_step, mass_override);
    let real_gravity = sim.config().gravity;
    let anchored = pin_top_particles(&mut sim);
    sim.set_gravity(real_gravity * gravity_fraction);
    let initial_count = sim.particles().len();
    let n_frames = (seconds / DT) as u32;
    let checkpoint = (n_frames / 20).max(1);

    println!(
        "[{label}] lambda={lambda} mu={mu} gravity_fraction={gravity_fraction} \
         max_substeps_per_step={max_substeps_per_step} anchored={anchored}"
    );
    for frame in 0..n_frames {
        sim.step();
        let dropped = sim.diagnostics_snapshot().sim_time_dropped;
        if dropped > 0.0 {
            println!("[{label}] frame={frame} DROPPED SIM TIME dropped={dropped}");
        }
        if frame.is_multiple_of(checkpoint) || frame == n_frames - 1 {
            let particles = sim.particles();
            let n = particles.len();
            let min_y = particles.iter().map(|p| p.x.y).fold(f32::MAX, f32::min);
            let max_y = particles.iter().map(|p| p.x.y).fold(f32::MIN, f32::max);
            let max_speed = particles
                .iter()
                .map(|p| p.v.length())
                .fold(0.0f32, f32::max);
            let max_j_dev = particles
                .iter()
                .map(|p| (p.deformation_gradient.determinant() - 1.0).abs())
                .fold(0.0f32, f32::max);
            let any_nonfinite = particles.iter().any(|p| {
                !p.x.is_finite() || !p.v.is_finite() || !p.deformation_gradient.is_finite()
            });
            println!(
                "[{label}] frame={frame} t={:.2} n={n} bbox_y=[{min_y:.3},{max_y:.3}] \
                 max_speed={max_speed:.4} max_|J-1|={max_j_dev:.5} any_nonfinite={any_nonfinite}",
                frame as f32 * DT,
            );
            assert!(!any_nonfinite, "[{label}] genuine NaN/Inf at frame {frame}");
            assert_eq!(n, initial_count, "[{label}] particle count changed");
        }
    }
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn membrane_scene_at_full_gravity_fraction() {
    let si_config = SimConfig::earth(GRID, 0.01, DT);
    let (lambda, mu) = si_config.lame_from_si_physical_cfg(
        MEMBRANE_YOUNG_MODULUS_PA,
        MEMBRANE_POISSON_RATIO,
        MEMBRANE_DENSITY_KG_M3,
    );
    run_probe(
        "new_stiffness+full_gravity",
        lambda,
        mu,
        1.0,
        256,
        600.0,
        NEW_MASS,
    );
}

/// Isolates the STIFFNESS change: new real SI stiffness, but at the OLD
/// near-zero gravity fraction (near-static, minimal velocity-gradient
/// activity). If the ratchet is a stiffness/substep-count effect, it
/// should still show up here even though gravity barely moves anything.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn membrane_scene_new_stiffness_old_gravity() {
    let si_config = SimConfig::earth(GRID, 0.01, DT);
    let (lambda, mu) = si_config.lame_from_si_physical_cfg(
        MEMBRANE_YOUNG_MODULUS_PA,
        MEMBRANE_POISSON_RATIO,
        MEMBRANE_DENSITY_KG_M3,
    );
    run_probe(
        "new_stiffness+old_gravity",
        lambda,
        mu,
        OLD_GRAVITY_FRACTION,
        256,
        120.0,
        NEW_MASS,
    );
}

/// Isolates the GRAVITY change: the OLD, soft, unsourced grid-unit
/// stiffness, but at full real gravity. If the ratchet is a
/// gravity-magnitude effect, it should show up here even with the old,
/// numerically gentle stiffness.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn membrane_scene_old_stiffness_full_gravity() {
    run_probe(
        "old_stiffness+full_gravity",
        OLD_LAMBDA,
        OLD_MU,
        1.0,
        256,
        120.0,
        OLD_MASS,
    );
}

/// Baseline: the ORIGINAL scene exactly as it shipped (old stiffness, old
/// near-zero gravity fraction) -- the real number the file's own comment
/// (0.001884 @ 600s at cundall=0.0) should reproduce.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn membrane_scene_old_stiffness_old_gravity_baseline() {
    run_probe(
        "old_stiffness+old_gravity",
        OLD_LAMBDA,
        OLD_MU,
        OLD_GRAVITY_FRACTION,
        32,
        120.0,
        OLD_MASS,
    );
}

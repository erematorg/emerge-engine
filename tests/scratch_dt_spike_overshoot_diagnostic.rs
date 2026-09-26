//! Item 6 (60fps stability / "dt-related sudden-acceleration" issue) --
//! FIRST real diagnostic, not a guess. User has flagged this live before,
//! never previously instrumented (see `project_6_item_pre_fajv_plan_2026-09-15.md`
//! item 6: "NOT STARTED... real difficulty unknown").
//!
//! Hypothesis under test: `choose_substep_dt` (`src/spacetime/solver/cfl.rs`)
//! picks dt from THIS substep's own pre-force particle velocities/stresses
//! (confirmed by reading the source: `last_max_speed` is lagged but used
//! ONLY for the fluid near-wall Mach gate, NOT the general velocity/material
//! CFL bound -- that scan is fresh every substep). So a genuine spike can
//! only come from a FORCE applied within the substep (gravity + contact +
//! constitutive stress) driving velocity past what the pre-force state's own
//! CFL bound assumed -- the standard "explicit contact/collision timestep"
//! problem, not a stale-data bug. This test measures whether that actually
//! happens, and how large, on a real hard-impact scene -- for a material
//! (`DruckerPragerMaterial`/sand) that has NO post-hoc retry/rollback safety
//! net at all (confirmed via grep: `owns_deformation_volume_state` is only
//! overridden by fluid materials; sand uses the trait default `false`, so
//! `do_substep_with_retry`'s fast path always takes the plain
//! `self.do_substep(requested_dt)` branch for sand, unconditionally,
//! regardless of `fluid_step_retry_enabled`).
extern crate emerge_engine as emerge;

use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

/// Real post-substep CFL-safety check, reusing the EXACT same formula
/// `SimConfig::cfl_coefficient`'s own doc gives (`dt <= cfl_coefficient *
/// cell_width / max_speed`), just evaluated AFTER the step instead of
/// before -- if a substep's actual resulting speed needed a smaller dt than
/// the one it was actually integrated with, `ratio > 1.0` proves the
/// substep briefly violated its own stability margin.
fn cfl_safe_speed(config: &SimConfig, dt: f32) -> f32 {
    config.cfl_coefficient * config.grid_cell_size / dt
}

fn max_particle_speed(sim: &Simulation) -> f32 {
    sim.particles()
        .iter()
        .map(|p| p.v.length())
        .fold(0.0f32, f32::max)
}

#[test]
fn sand_hard_impact_dt_overshoot_diagnostic() {
    const GRID: usize = 64;
    const FLOOR: f32 = 2.0;
    let gravity = Vec2::new(0.0, -9.81);
    let config = SimConfig {
        max_substeps_per_step: 2000,
        ..SimConfig::standard(GRID, 0.02, gravity)
    };

    let side = 6i32;
    let drop_height = 20.0;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(side, side),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + drop_height),
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(5429.0, 0.357)))
        .with_boundary(Box::new(SlipBoundary::new(2)));

    let mut max_ratio = 0.0f32;
    let mut max_ratio_step = 0usize;
    let mut violation_count = 0usize;
    let mut total_steps = 0usize;
    // Tolerance: a small overshoot right at the impact instant is expected
    // (explicit force-then-check, standard). This flags a REAL problem only
    // once ratio clears a real margin above 1.0 -- 1.0 itself is the exact
    // theoretical CFL limit, so anything over is already, by definition, a
    // step that would be unstable if sustained.
    const REPORT_THRESHOLD: f32 = 1.0;

    // Second, independently-tested hypothesis: not a kinematic-overshoot bug,
    // but a real "frame-spike" -- CFL correctly demands many more substeps
    // during a violent event, so a single `step()` call's WALL-CLOCK cost
    // spikes even though the physics itself stays admissible (matches the
    // real, already-documented `basic_snow_gpu` regression from 2026-07-30:
    // a substep-count explosion collapsing fps 60->13-16). If real demos
    // gate substeps behind a `FixedStepController`'s `max_substeps_per_frame`,
    // a frame this expensive either stalls (visible pause) or, once it
    // finally completes, the renderer's next poll shows a position delta from
    // MANY accumulated substeps at once -- reading as a sudden jump/
    // acceleration to a human watching, even though no single substep here
    // was itself unstable.
    let mut baseline_us = 0u64;
    let mut baseline_substeps = 0usize;
    let mut max_step_us = 0u64;
    let mut max_step_us_at = 0usize;
    let mut max_substeps = 0usize;
    let mut max_substeps_at = 0usize;

    for step in 0..250 {
        let t = std::time::Instant::now();
        solver.step_n(1);
        let step_us = t.elapsed().as_micros() as u64;
        total_steps += 1;
        let substeps = solver.last_substeps();
        if step == 0 {
            baseline_us = step_us;
            baseline_substeps = substeps;
        }
        if step_us > max_step_us {
            max_step_us = step_us;
            max_step_us_at = step;
        }
        if substeps > max_substeps {
            max_substeps = substeps;
            max_substeps_at = step;
        }
        let dt = solver.effective_dt();
        let after = max_particle_speed(&solver);
        let safe = cfl_safe_speed(solver.config(), dt);
        let ratio = after / safe;
        if ratio > REPORT_THRESHOLD {
            violation_count += 1;
        }
        if ratio > max_ratio {
            max_ratio = ratio;
            max_ratio_step = step;
        }
    }

    println!(
        "\n── SAND HARD-IMPACT dt/CFL OVERSHOOT DIAGNOSTIC ──\n\
         total_steps={total_steps} violations(ratio>1.0)={violation_count} \
         max_ratio={max_ratio:.4} at step={max_ratio_step}\n\
         (ratio = actual post-substep max speed / cfl_coefficient*cell_width/dt_used; \
         ratio>1.0 means that substep's own result already exceeded the stability \
         margin the dt it was integrated with was supposed to guarantee)\n\
         ── FRAME-SPIKE (substep-count/wall-clock) CHECK ──\n\
         step0(baseline): {baseline_us}us, {baseline_substeps} substeps\n\
         worst wall-clock: step={max_step_us_at} {max_step_us}us ({:.1}x baseline)\n\
         worst substep count: step={max_substeps_at} {max_substeps} substeps ({:.1}x baseline)",
        max_step_us as f64 / baseline_us.max(1) as f64,
        max_substeps as f64 / baseline_substeps.max(1) as f64,
    );

    for p in solver.particles().iter() {
        assert!(
            p.x.is_finite() && p.v.is_finite(),
            "sand particle went non-finite during impact: x={:?} v={:?}",
            p.x,
            p.v
        );
    }
}

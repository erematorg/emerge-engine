//! Diagnostic for a "sudden acceleration" seen at 60 fps: does a substep outrun its
//! own CFL bound?
//!
//! Hypothesis: `choose_substep_dt` (`src/spacetime/solver/cfl.rs`) picks dt from the
//! substep's own pre-force particle velocities and stresses (`last_max_speed` is lagged
//! but used only for the fluid near-wall Mach gate; the general velocity and material
//! CFL scan is fresh every substep). So a spike can only come from a force applied
//! within the substep (gravity + contact + constitutive stress) driving velocity past
//! what the pre-force CFL bound assumed, the standard explicit contact/collision
//! timestep problem, not stale data. This measures whether and how much that happens
//! on a hard-impact scene, for sand (`DruckerPragerMaterial`), which has no
//! retry/rollback: only fluid materials override `owns_deformation_volume_state`, so
//! `do_substep_with_retry` takes the plain `self.do_substep(requested_dt)` branch for
//! sand whatever `fluid_step_retry_enabled` says.

use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

/// Post-substep CFL check with the formula of `SimConfig::cfl_coefficient`'s doc
/// (`dt <= cfl_coefficient * cell_width / max_speed`), evaluated after the step instead
/// of before: `ratio > 1.0` means the substep's resulting speed needed a smaller dt than
/// it was integrated with, a brief violation of its stability margin.
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
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
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
    // (explicit force-then-check, standard). This flags a problem only
    // once ratio clears a margin above 1.0 -- 1.0 itself is the exact
    // theoretical CFL limit, so anything over is already, by definition, a
    // step that would be unstable if sustained.
    const REPORT_THRESHOLD: f32 = 1.0;

    // Second hypothesis: not a kinematic overshoot but a frame spike. CFL correctly
    // demands many more substeps during a violent event, so one `step()` call's
    // wall-clock cost spikes while the physics stays admissible (as when a substep
    // explosion took `basic_snow_gpu` from 60 to 13-16 fps). Behind a
    // `FixedStepController`'s `max_substeps_per_frame`, such a frame either stalls
    // (a visible pause) or, once done, the renderer shows the position change of many
    // accumulated substeps at once, which reads as a sudden jump though no substep was
    // unstable.
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

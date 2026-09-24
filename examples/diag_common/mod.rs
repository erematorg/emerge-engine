//! Shared by `diag_step_timing_breakdown.rs` and
//! `diag_pressure_projection_timing.rs` -- both independently ran the same
//! "step N times, accumulate `StepTiming`, print the per-phase breakdown"
//! loop before this extraction (2026-08-15). Included via
//! `#[path = "diag_common/mod.rs"] mod diag_common;`, not a crate
//! dependency -- example binaries in this project don't share a lib target.
extern crate emerge_engine as emerge;

use emerge::Simulation;
use emerge::diagnostics::StepTiming;

/// Steps `solver` `n` times, accumulates `StepTiming`, and prints the same
/// per-phase averages + `unaccounted_us` breakdown every diagnostic in this
/// family already printed independently. `print_progress_every` optionally
/// prints one line per that many frames (`total_us`/`pressure_us`/
/// `retry_snapshot_us`) as it goes -- `None` prints only the final summary.
pub fn run_and_report_timing(
    solver: &mut Simulation,
    n: usize,
    print_progress_every: Option<usize>,
) {
    let particle_count = solver.particles().len();
    let mut totals = StepTiming::default();
    let wall_start = std::time::Instant::now();
    for i in 0..n {
        solver.step();
        let t = solver.diagnostics_snapshot().timing;
        totals.p2g_us += t.p2g_us;
        totals.grid_update_us += t.grid_update_us;
        totals.g2p_us += t.g2p_us;
        totals.fields_us += t.fields_us;
        totals.thermal_us += t.thermal_us;
        totals.cfl_us += t.cfl_us;
        totals.spatial_hash_us += t.spatial_hash_us;
        totals.phase_sleep_us += t.phase_sleep_us;
        totals.project_us += t.project_us;
        totals.density_us += t.density_us;
        totals.pressure_us += t.pressure_us;
        totals.retry_snapshot_us += t.retry_snapshot_us;
        totals.total_us += t.total_us;
        if let Some(every) = print_progress_every
            && i % every == 0
        {
            println!(
                "  frame {i}: total_us={} pressure_us={} retry_snapshot_us={}",
                t.total_us, t.pressure_us, t.retry_snapshot_us
            );
        }
    }
    let wall_elapsed = wall_start.elapsed();

    println!("particle_count={particle_count}");
    println!(
        "wall_time={:.2}s  ({:.2}fps over {n} frames)",
        wall_elapsed.as_secs_f64(),
        n as f64 / wall_elapsed.as_secs_f64()
    );
    println!("avg total_us={:.1}", totals.total_us as f64 / n as f64);
    println!("avg p2g_us={:.1}", totals.p2g_us as f64 / n as f64);
    println!(
        "avg grid_update_us={:.1}",
        totals.grid_update_us as f64 / n as f64
    );
    println!("avg g2p_us={:.1}", totals.g2p_us as f64 / n as f64);
    println!("avg fields_us={:.1}", totals.fields_us as f64 / n as f64);
    println!("avg thermal_us={:.1}", totals.thermal_us as f64 / n as f64);
    println!("avg cfl_us={:.1}", totals.cfl_us as f64 / n as f64);
    println!(
        "avg spatial_hash_us={:.1}",
        totals.spatial_hash_us as f64 / n as f64
    );
    println!(
        "avg phase_sleep_us={:.1}",
        totals.phase_sleep_us as f64 / n as f64
    );
    println!("avg project_us={:.1}", totals.project_us as f64 / n as f64);
    println!("avg density_us={:.1}", totals.density_us as f64 / n as f64);
    println!(
        "avg pressure_us={:.1}  ({:.1}% of total)",
        totals.pressure_us as f64 / n as f64,
        100.0 * totals.pressure_us as f64 / totals.total_us as f64
    );
    println!(
        "avg retry_snapshot_us={:.1}  ({:.1}% of total)",
        totals.retry_snapshot_us as f64 / n as f64,
        100.0 * totals.retry_snapshot_us as f64 / totals.total_us as f64
    );
    // pressure_us is a SUBSET of grid_update_us (see StepTiming's own doc),
    // NOT additive -- excluded here to avoid double-counting.
    let accounted = totals.p2g_us
        + totals.grid_update_us
        + totals.g2p_us
        + totals.fields_us
        + totals.thermal_us
        + totals.cfl_us
        + totals.spatial_hash_us
        + totals.phase_sleep_us
        + totals.project_us
        + totals.density_us
        + totals.retry_snapshot_us;
    println!(
        "unaccounted_us={:.1} ({:.1}% of total)",
        (totals.total_us - accounted) as f64 / n as f64,
        100.0 * (totals.total_us - accounted) as f64 / totals.total_us as f64
    );
}

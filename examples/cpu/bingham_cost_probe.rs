//! Where does a yield-stress scene's frame time actually go?
//!
//! Prints the engine's own per-phase timing record for the `basic_bingham`
//! scene, so a performance claim about it rests on the measured breakdown
//! rather than on a guess about which phase dominates.
//!
//!   cargo run --release --example bingham_cost_probe
extern crate emerge_engine as emerge;

use emerge::Simulation;
use emerge::{
    BinghamFluidMaterial, BinghamProps, FrictionBoundary, FromSI, SimConfig, SpawnRegion,
};
use glam::{IVec2, Vec2};

/// The demo's grid, 160 cells, overridable with `BINGHAM_PROBE_GRID` to
/// price a different tank: the same three columns at the same places, more
/// or fewer empty cells around them. The grid is sparse, so the question is
/// whether empty cells cost.
///
/// This probe keeps its own copy of the scene rather than including
/// `bingham_slump_scene.rs`, because it builds the materials itself to
/// offer `BINGHAM_PROBE_ELASTIC`; the geometry, floor and constants below
/// must match that file.
fn grid() -> usize {
    std::env::var("BINGHAM_PROBE_GRID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160)
}
const DX_M: f32 = 0.002;
/// Simulated seconds advanced per frame, overridable so the frame-rate
/// against playback-speed trade can be swept rather than asserted.
fn step_seconds() -> f32 {
    std::env::var("BINGHAM_PROBE_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.002)
}
const RHO: f32 = 1000.0;
const ETA: f32 = 0.5;
const FLOOR: f32 = 2.0;
const COLUMN: IVec2 = IVec2::new(10, 20);
const YIELD_STRAIN: f32 = 0.05;
const YIELDS: [f32; 3] = [2.0, 60.0, 1200.0];
const COLUMN_X: [f32; 3] = [57.0, 125.0, 148.0];

/// `BINGHAM_PROBE_ELASTIC=0` measures the purely viscous branch instead, so
/// the cost of the elastoviscoplastic branch's own SVD work is attributable
/// rather than assumed.
fn elastic_scale() -> f32 {
    match std::env::var("BINGHAM_PROBE_ELASTIC").as_deref() {
        Ok("0") => 0.0,
        _ => 1.0,
    }
}

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(grid(), DX_M, step_seconds())
    };
    let v_max = (2.0 * 9.81 * COLUMN.y as f32 * DX_M).sqrt();
    let bulk_modulus = RHO * (10.0 * v_max).powi(2);

    let props = |tau0: f32| BinghamProps {
        rho_kg_m3: RHO,
        eta_pa_s: ETA,
        bulk_modulus_pa: bulk_modulus,
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN * elastic_scale(),
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR + COLUMN.y as f32 * 0.5),
            material_id: slot as u32,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(YIELDS[slot]), &config)
    };

    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props(YIELDS[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(FrictionBoundary::new(
            config.boundary_thickness,
            1.0,
        )));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));

    println!("particles={}", sim.particles().len());
    let mut acc = [0u64; 10];
    let mut substeps = 0usize;
    let mut peak_speed_cells_s = 0.0f32;
    // 120 by default so the averages below stay comparable with the table
    // in `basic_bingham`'s header. Raise it to see past the collapse.
    let frames: usize = std::env::var("BINGHAM_PROBE_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    // Each frame's own wall time and substep count. A mean over a second is
    // what the demo's panel shows, and it hides exactly the long frames a
    // viewer sees as a stutter, because the demo advances a FIXED slice of
    // simulated time per frame: a slow frame is a frame where the motion
    // on screen visibly slows down.
    let mut frame_ms = Vec::with_capacity(frames);
    let mut frame_substeps = Vec::with_capacity(frames);
    let wall = std::time::Instant::now();
    for _ in 0..frames {
        let frame_start = std::time::Instant::now();
        sim.step();
        frame_ms.push(frame_start.elapsed().as_secs_f64() * 1000.0);
        let s = sim.diagnostics_snapshot();
        frame_substeps.push(s.substeps_last_step);
        let t = s.timing;
        substeps += s.substeps_last_step;
        peak_speed_cells_s = peak_speed_cells_s.max(s.max_particle_speed);
        for (slot, value) in [
            t.p2g_us,
            t.grid_update_us,
            t.g2p_us,
            t.cfl_us,
            t.spatial_hash_us,
            t.phase_sleep_us,
            t.project_us,
            t.density_us,
            t.retry_snapshot_us,
            t.total_us,
        ]
        .into_iter()
        .enumerate()
        {
            acc[slot] += value;
        }
    }
    let elapsed_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let names = [
        "p2g",
        "grid_update",
        "g2p",
        "cfl",
        "spatial_hash",
        "phase_sleep",
        "project",
        "density",
        "retry_snapshot",
        "TOTAL",
    ];
    let total = acc[9].max(1) as f64;
    println!(
        "{frames} frames in {elapsed_ms:.1} ms -> {:.2} ms/frame, {:.1} fps, {:.1} substeps/frame",
        elapsed_ms / frames as f64,
        1000.0 * frames as f64 / elapsed_ms,
        substeps as f64 / frames as f64
    );
    for (name, value) in names.iter().zip(acc.iter()) {
        println!(
            "  {name:<15} {:8.2} ms/frame  {:5.1}%",
            *value as f64 / 1000.0 / frames as f64,
            100.0 * *value as f64 / total
        );
    }

    // The spread, per phase. The first half second of simulated time is
    // the collapse, where everything moves; after it the columns sit.
    let dt = step_seconds();
    let split = ((0.5 / dt).round() as usize).min(frames);
    println!("frame time spread, physics only (no rendering), per phase:");
    println!(
        "  phase        frames    p50 ms   p95 ms   p99 ms   max ms   >16.7 ms  >33.3 ms   substeps min-max"
    );
    for (label, range) in [("collapse", 0..split), ("settled", split..frames)] {
        if range.is_empty() {
            continue;
        }
        let mut t: Vec<f64> = frame_ms[range.clone()].to_vec();
        t.sort_by(f64::total_cmp);
        let pct = |p: f64| t[((t.len() - 1) as f64 * p).round() as usize];
        let subs = &frame_substeps[range.clone()];
        println!(
            "  {label:<10} {:>7}   {:>7.2}  {:>7.2}  {:>7.2}  {:>7.2}   {:>8}  {:>8}   {:>5}-{}",
            t.len(),
            pct(0.50),
            pct(0.95),
            pct(0.99),
            t[t.len() - 1],
            t.iter().filter(|&&x| x > 1000.0 / 60.0).count(),
            t.iter().filter(|&&x| x > 1000.0 / 30.0).count(),
            subs.iter().min().unwrap_or(&0),
            subs.iter().max().unwrap_or(&0)
        );
    }
    println!(
        "peak particle speed over the run: {:.3} cells/s = {:.3} m/s  (the acoustic derating assumed {:.3} m/s)",
        peak_speed_cells_s,
        peak_speed_cells_s * DX_M,
        v_max
    );
    let accounted: u64 = acc[..9].iter().sum();
    println!(
        "  {:<15} {:8.2} ms/frame  {:5.1}%",
        "unaccounted",
        (acc[9].saturating_sub(accounted)) as f64 / 1000.0 / frames as f64,
        100.0 * acc[9].saturating_sub(accounted) as f64 / total
    );
}

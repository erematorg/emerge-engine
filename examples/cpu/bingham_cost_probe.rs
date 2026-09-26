//! Where does a yield-stress scene's frame time actually go?
//!
//! Prints the engine's own per-phase timing record for the `basic_bingham`
//! scene, so a performance claim about it rests on the measured breakdown
//! rather than on a guess about which phase dominates.
//!
//!   cargo run --release --example bingham_cost_probe
extern crate emerge_engine as emerge;

use emerge::Simulation;
use emerge::{BinghamFluidMaterial, BinghamProps, FromSI, SimConfig, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
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
const COLUMN: IVec2 = IVec2::new(4, 20);
const YIELD_STRAIN: f32 = 0.05;
const YIELDS: [f32; 3] = [2.0, 60.0, 400.0];
const COLUMN_X: [f32; 3] = [12.0, 32.0, 52.0];

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
        ..SimConfig::earth(GRID, DX_M, step_seconds())
    };
    let v_max = (2.0 * 9.81 * COLUMN.y as f32 * DX_M).sqrt();
    let bulk_modulus = RHO * (10.0 * v_max).powi(2);

    let props = |tau0: f32| BinghamProps {
        rho_kg_m3: RHO,
        eta_pa_s: ETA,
        bulk_modulus_pa: bulk_modulus,
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN * elastic_scale(),
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR + COLUMN.y as f32 * 0.5),
            material_id: slot as u32,
            precompute_initial_volumes: true,
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
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));

    println!("particles={}", sim.particles().len());
    let mut acc = [0u64; 10];
    let mut substeps = 0usize;
    let mut peak_speed_cells_s = 0.0f32;
    const FRAMES: usize = 120;
    let wall = std::time::Instant::now();
    for _ in 0..FRAMES {
        sim.step();
        let s = sim.diagnostics_snapshot();
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
        "{FRAMES} frames in {elapsed_ms:.1} ms -> {:.2} ms/frame, {:.1} fps, {:.1} substeps/frame",
        elapsed_ms / FRAMES as f64,
        1000.0 * FRAMES as f64 / elapsed_ms,
        substeps as f64 / FRAMES as f64
    );
    for (name, value) in names.iter().zip(acc.iter()) {
        println!(
            "  {name:<15} {:8.2} ms/frame  {:5.1}%",
            *value as f64 / 1000.0 / FRAMES as f64,
            100.0 * *value as f64 / total
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
        (acc[9].saturating_sub(accounted)) as f64 / 1000.0 / FRAMES as f64,
        100.0 * acc[9].saturating_sub(accounted) as f64 / total
    );
}

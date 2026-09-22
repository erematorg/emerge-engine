//! What frame time can the `soil_horizons` scene actually afford?
//!
//! Rebuilds the scene headless (same four horizons, spawns and materials)
//! and, for a given simulated time per frame, reports how many substeps the
//! CFL condition really asks for, at rest and under the scene's strongest
//! press, and what a frame costs. The substep budget is generous here so the
//! need is measured rather than cut off.
//!
//!   cargo run --release --example soil_horizons_cost_probe
//!   SOIL_PROBE_DT=0.02 cargo run --release --example soil_horizons_cost_probe
extern crate emerge_engine as emerge;

use emerge::{
    DruckerPragerMaterial, GranularFluidMaterial, NaccMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

fn main() {
    let dt: f32 = std::env::var("SOIL_PROBE_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    let config = SimConfig {
        max_substeps_per_step: 1024,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(64, 0.01, dt)
    };
    let mut sim = Simulation::empty(config)
        .with_material(0, Box::new(DruckerPragerMaterial::low_friction(600.0, 0.3)))
        .with_material(
            1,
            Box::new(GranularFluidMaterial::saturated_loam(1200.0, 0.3)),
        )
        .with_material(2, Box::new(NaccMaterial::kaolin(1800.0, 0.3)))
        .with_material(3, Box::new(DruckerPragerMaterial::dilatant(2400.0, 0.3)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    // (id, thickness, density ratio, bottom), bottom up, as in the scene.
    for (material_id, thickness, density_ratio, y_bottom) in [
        (3, 16.0_f32, 1.8, 2.0),
        (2, 14.0, 1.5, 18.0),
        (1, 8.0, 1.2, 32.0),
        (0, 2.0, 0.2, 40.0),
    ] {
        let _ = sim.add_body(SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(48, thickness.round().max(1.0) as i32),
            box_center: Vec2::new(32.0, y_bottom + thickness * 0.5),
            material_id,
            precompute_initial_volumes: true,
            mass_override: Some(density_ratio),
            ..SpawnRegion::for_sim(&config)
        });
    }

    // Five simulated seconds at rest, then two under the scene's maximum
    // press (force 200, radius 3, at the surface centre).
    let rest_frames = (5.0 / dt).round() as usize;
    let press_frames = (2.0 / dt).round() as usize;
    let phase = |label: &str, frames: usize, press: bool, sim: &mut Simulation| {
        let (mut substeps, mut worst, mut dropped) = (0usize, 0usize, 0.0f32);
        let wall = std::time::Instant::now();
        for _ in 0..frames {
            if press {
                sim.apply_impulse(Vec2::new(32.0, 42.0), 3.0, Vec2::new(0.0, -200.0 * dt));
            }
            sim.step();
            let s = sim.diagnostics_snapshot();
            substeps += s.substeps_last_step;
            worst = worst.max(s.substeps_last_step);
            dropped += s.sim_time_dropped;
        }
        let ms = wall.elapsed().as_secs_f64() * 1000.0 / frames as f64;
        println!(
            "dt={dt} {label}: {:.1} substeps/frame (worst {worst}), {ms:.1} ms/frame, {:.0} fps, dropped {dropped:.4}",
            substeps as f64 / frames as f64,
            1000.0 / ms
        );
    };
    phase("rest", rest_frames, false, &mut sim);
    phase("max press", press_frames, true, &mut sim);
    let finite = sim
        .particles()
        .iter()
        .all(|p| p.x.is_finite() && p.v.is_finite());
    println!("dt={dt} all particles finite: {finite}");
}

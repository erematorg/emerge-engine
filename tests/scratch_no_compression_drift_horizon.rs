//! Does an anchored body's volume drift settle or keep growing? Same scene
//! as `no_compression_hanging_body_has_no_passive_volume_ratchet` in
//! `tests/solver.rs`, run longer, printing every 1000 steps the mean and max
//! |J - 1|, J by band of distance below the anchor, and the free body's
//! centre of mass. Variants through the environment: DRIFT_STEPS,
//! DRIFT_GRAVITY (fraction of g), DRIFT_DT (frame time), DRIFT_MATERIAL
//! (`neohookean` for an ordinary elastic body), DRIFT_PIN=0 (rest on the
//! floor instead of hanging). Results are in KNOWN_LIMITATIONS.md.
extern crate emerge_engine as emerge;

use emerge::{
    MaterialModel, NeoHookeanMaterial, NoCompressionMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn no_compression_drift_over_a_long_horizon() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let gravity_fraction = env("DRIFT_GRAVITY", 0.0002);
    let frame_s = env("DRIFT_DT", 0.05);
    let steps = env("DRIFT_STEPS", 30_000.0) as usize;
    let material: Box<dyn MaterialModel> = match std::env::var("DRIFT_MATERIAL").as_deref() {
        Ok("neohookean") => Box::new(NeoHookeanMaterial::new(2000.0, 4000.0)),
        _ => Box::new(NoCompressionMaterial::new(2000.0, 4000.0)),
    };
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 32,
        material_cfl_coefficient: 0.7,
        cundall_damping: 0.0,
        ..SimConfig::earth(64, 0.01, frame_s)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 6),
        box_center: Vec2::splat(32.0),
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(material)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let max_y = sim
        .particles()
        .iter()
        .map(|p| p.x.y)
        .fold(f32::MIN, f32::max);
    let pin = std::env::var("DRIFT_PIN").as_deref() != Ok("0");
    let particles = sim.particles_mut();
    for i in 0..particles.len() {
        if pin && particles.x[i].y >= max_y - 0.4 {
            particles.pinned[i] = 1;
        }
    }
    sim.set_gravity(config.gravity * gravity_fraction);
    let anchor_y = max_y;
    let mut prev_com = f32::NAN;
    for block in 1..=steps / 1000 {
        sim.step_n(1000);
        let (mut max_dev, mut mean_j) = (0.0_f32, 0.0_f32);
        // Mean J per band of distance below the anchor line, in cells.
        let (mut band_j, mut band_n) = ([0.0_f32; 6], [0u32; 6]);
        let (mut com_num, mut mass_free, mut mom_y) = (0.0_f32, 0.0_f32, 0.0_f32);
        for p in sim.particles().iter() {
            let j = p.deformation_gradient.determinant();
            max_dev = max_dev.max((j - 1.0).abs());
            mean_j += j;
            let band = (((anchor_y - p.x.y) / 1.0).floor().max(0.0) as usize).min(5);
            band_j[band] += j;
            band_n[band] += 1;
            if p.pinned == 0 {
                com_num += p.mass * p.x.y;
                mass_free += p.mass;
                mom_y += p.mass * p.v.y;
            }
        }
        mean_j /= sim.particles().len() as f32;
        let com = com_num / mass_free;
        let bands: Vec<String> = (0..6)
            .map(|b| format!("{:+.5}", band_j[b] / band_n[b].max(1) as f32 - 1.0))
            .collect();
        let min_y = sim
            .particles()
            .iter()
            .map(|p| p.x.y)
            .fold(f32::MAX, f32::min);
        println!(
            "steps={:6} max|J-1|={max_dev:.6} mean(J)-1={:+.6} min_y={min_y:.4} com_y={com:.5} dcom={:+.6} v_com={:+.3e} J-1 by band below anchor [0-1,1-2,2-3,3-4,4-5,5+ cells]={}",
            block * 1000,
            mean_j - 1.0,
            com - prev_com,
            mom_y / mass_free,
            bands.join(" ")
        );
        prev_com = com;
    }
}

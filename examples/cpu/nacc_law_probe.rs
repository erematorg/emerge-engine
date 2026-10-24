//! Headless replicas of the two shipped scenes that use `NaccMaterial`
//! (`soil_horizons`, `permafrost`), same materials, spawns and frame times,
//! printing what a change to the Cam-Clay hardening law would move: the
//! clay's height, its plastic volume ratio, its preconsolidation pressure
//! against the pressure it carries, and the frame cost.
//!
//!   cargo run --release --example nacc_law_probe
extern crate emerge_engine as emerge;

use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    DruckerPragerMaterial, GranularFluidMaterial, MaterialModel, NaccMaterial, SimConfig,
    Simulation, SlipBoundary, SpawnRegion, WithLatentHeat,
};
use glam::{IVec2, Vec2};

/// Summary of every particle carrying `material_id`, read through `nacc`.
fn report(label: &str, sim: &Simulation, material_id: u32, nacc: &NaccMaterial) {
    let parts = sim.particles();
    let (mut lo, mut hi, mut n) = (f32::MAX, f32::MIN, 0u32);
    let (mut plastic_ratio, mut p0_sum, mut p_sum, mut finite) = (0.0f32, 0.0f32, 0.0f32, true);
    for i in 0..parts.len() {
        if parts.material_id[i] != material_id {
            continue;
        }
        let x = parts.x[i];
        finite &= x.is_finite() && parts.v[i].is_finite();
        lo = lo.min(x.y);
        hi = hi.max(x.y);
        let alpha = parts.log_volume_strain[i];
        plastic_ratio += alpha.exp();
        p0_sum += nacc.preconsolidation_pressure(alpha);
        let tau = nacc.kirchhoff_stress(parts, i);
        let j = parts.deformation_gradient[i].determinant().max(1.0e-6);
        p_sum += -(tau.x_axis.x + tau.y_axis.y) * 0.5 / j;
        n += 1;
    }
    let n_f = n.max(1) as f32;
    println!(
        "{label}: n={n} height={:.2} cells plastic_volume_ratio={:.4} p0_mean={:.4e} p_mean={:.4e} p0/p={:.3} finite={finite}",
        hi - lo,
        plastic_ratio / n_f,
        p0_sum / n_f,
        p_sum / n_f,
        p0_sum / p_sum.max(f32::MIN_POSITIVE)
    );
}

fn soil_horizons() {
    const GRID: usize = 64;
    const B_ID: u32 = 2;
    let config = SimConfig {
        max_substeps_per_step: 16,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, 0.01)
    };
    let mut b_horizon = NaccMaterial::kaolin(1800.0, 0.3);
    // Same own-weight preconsolidation the scene now gives it.
    let areal = |ratio: f32| ratio / 0.25;
    let sigma_v = 0.3 * (areal(0.2) * 2.0 + areal(1.2) * 8.0 + areal(1.5) * 7.0);
    b_horizon.initial_preconsolidation = sigma_v * (2.0 - 0.436) * 0.5;
    let mut sim = Simulation::empty(config)
        .with_material(0, Box::new(DruckerPragerMaterial::low_friction(600.0, 0.3)))
        .with_material(
            1,
            Box::new(GranularFluidMaterial::saturated_loam(1200.0, 0.3)),
        )
        .with_material(B_ID, Box::new(b_horizon))
        .with_material(3, Box::new(DruckerPragerMaterial::dilatant(2400.0, 0.3)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    // (id, thickness, density ratio, bottom), bottom up, as in the scene.
    let horizons = [
        (3, 16.0, 1.8, 2.0),
        (B_ID, 14.0, 1.5, 18.0),
        (1, 8.0, 1.2, 32.0),
        (0, 2.0, 0.2, 40.0),
    ];
    for (material_id, thickness, density_ratio, y_bottom) in horizons {
        let _ = sim.add_body(SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(48, thickness as i32),
            box_center: Vec2::new(32.0, y_bottom + thickness * 0.5),
            material_id,
            mass_override: Some(density_ratio),
            ..SpawnRegion::for_sim(&config)
        });
    }
    report("soil_horizons B, t=0", &sim, B_ID, &b_horizon);
    let wall = std::time::Instant::now();
    const FRAMES: usize = 500;
    for _ in 0..FRAMES {
        sim.step();
    }
    let ms = wall.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
    report("soil_horizons B, t=5s", &sim, B_ID, &b_horizon);
    println!("soil_horizons: {ms:.2} ms/frame over {FRAMES} frames");
}

fn permafrost(frozen_start: bool) {
    const FROZEN_ID: u32 = 0;
    const THAWED_ID: u32 = 1;
    const DT: f32 = 0.02;
    let config = SimConfig {
        max_substeps_per_step: 64,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(64, 0.01, DT)
    };
    let ambient = if frozen_start { 260.0 } else { 280.0 };
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 2.2,
            heat_capacity: 2000.0,
            density: 1800.0,
            ambient,
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );
    let sigma_v = 0.3 * 20.0 * 0.5;
    let preconsolidation = sigma_v * (2.0 - 0.436) * 0.5;
    let mut frozen = NaccMaterial::kaolin(18000.0 * 250.0, 0.3);
    frozen.initial_preconsolidation = preconsolidation;
    let mut thawed = NaccMaterial::kaolin(18000.0, 0.3);
    thawed.initial_preconsolidation = preconsolidation;
    let mut sim = Simulation::empty(config)
        .with_material(FROZEN_ID, Box::new(frozen))
        .with_material(THAWED_ID, Box::new(WithLatentHeat::new(thawed, 334.0)))
        .with_thermal(thermal)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            if p.material_id == FROZEN_ID && p.temperature > 273.15 {
                Some(THAWED_ID)
            } else if p.material_id == THAWED_ID && p.temperature < 273.15 {
                Some(FROZEN_ID)
            } else {
                None
            }
        });
    let id = if frozen_start { FROZEN_ID } else { THAWED_ID };
    let nacc = if frozen_start { frozen } else { thawed };
    let _ = sim.add_body(SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(32, 20),
        box_center: Vec2::new(32.0, 12.0),
        material_id: id,
        ..SpawnRegion::for_sim(&config)
    });
    for t in sim.particles_mut().temperature.iter_mut() {
        *t = ambient;
    }
    let label = if frozen_start {
        "permafrost frozen"
    } else {
        "permafrost thawed"
    };
    let wall = std::time::Instant::now();
    const SETTLE: usize = 250;
    for _ in 0..SETTLE {
        sim.step();
    }
    let ms = wall.elapsed().as_secs_f64() * 1000.0 / SETTLE as f64;
    report(&format!("{label}, settled t=5s"), &sim, id, &nacc);
    let before: Vec<Vec2> = sim.particles().x.clone();
    // The scene's strike at its lowest force setting, at the block's top centre.
    sim.apply_impulse(Vec2::new(32.0, 21.0), 3.0, Vec2::new(0.0, -10.0 * DT));
    for _ in 0..30 {
        sim.step();
    }
    let moved = before
        .iter()
        .zip(sim.particles().x.iter())
        .map(|(b, a)| (*a - *b).length())
        .fold(0.0f32, f32::max);
    report(&format!("{label}, after strike"), &sim, id, &nacc);
    println!("{label}: {ms:.2} ms/frame settling, max displacement after strike {moved:.4} cells");
}

fn main() {
    soil_horizons();
    permafrost(true);
    permafrost(false);
}

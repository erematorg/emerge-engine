//! How much of the particles' motion f32 rounds away, on a scene that
//! settles slowly: the sand pile of `accuracy.rs`'s
//! `sand_angle_of_repose_is_physical` (the Bingham deposit's probe lives in
//! `bingham_deposit_state.rs`, which already builds that scene).
//! Each line reports what f32 rounding does to the increments over the
//! frame's mean substep (see `PositionResolution` for each share). Probes,
//! no criterion.

use emerge::diagnostics::position_resolution;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

fn report(label: &str, sim: &Simulation) {
    let r = position_resolution(
        sim.particles().x.iter().copied(),
        sim.particles().v.iter().copied(),
        sim.config().dt / sim.last_substeps().max(1) as f32,
    );
    println!(
        "{label}: {} moving of {}, frozen {:.4}, displacement lost {:.4}, lost {:.4}, coarse {:.4}, substep {:.3e} s, largest |x| {:.1}",
        r.moving,
        sim.particles().len(),
        r.frozen,
        r.displacement_lost,
        r.lost,
        r.coarse,
        sim.config().dt / sim.last_substeps().max(1) as f32,
        sim.particles()
            .x
            .iter()
            .fold(0.0f32, |m, x| m.max(x.x.abs()).max(x.y.abs()))
    );
}

#[test]
#[ignore = "probe: run with --ignored --nocapture"]
fn probe_sand_pile_position_resolution() {
    const GRID: usize = 64;
    const DT: f32 = 0.1;
    const FLOOR: f32 = 2.0;
    let config = SimConfig {
        max_substeps_per_step: 64,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + 8.0),
        material_id: 0,
        ..SpawnRegion::for_sim(&config)
    };
    let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    let mut sim = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
    for step in 1..=1500 {
        sim.step();
        if step % 250 == 0 {
            report(&format!("sand pile step {step}"), &sim);
        }
    }
}

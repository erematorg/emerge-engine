//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/trophic_predation_demo.rs`'s scene (no GUI, no predation
//! logic) to verify the real SI migration (soft-tissue E=500 Pa/nu=0.45/
//! rho=1000, zero gravity) is stable at a real substep budget.
extern crate emerge_engine as emerge;

use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PREY_ID: u32 = 0;
const PREDATOR_ID: u32 = 1;
const EATEN_ID: u32 = 2;

const BODY_YOUNG_MODULUS_PA: f32 = 500.0;
const BODY_POISSON_RATIO: f32 = 0.45;
const BODY_DENSITY_KG_M3: f32 = 1000.0;

fn make_sim(max_substeps_per_step: usize) -> Simulation {
    let config = SimConfig {
        gravity: Vec2::ZERO,
        max_substeps_per_step,
        ..SimConfig::standard(GRID, DT, Vec2::ZERO)
    };
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        BODY_YOUNG_MODULUS_PA,
        BODY_POISSON_RATIO,
        BODY_DENSITY_KG_M3,
    );
    println!("[trophic-probe] lambda={lambda} mu={mu}");
    const SPACING: f32 = 0.6;
    let mass_grid = (BODY_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;
    let prey_spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(40, 40),
        box_center: Vec2::new(32.0, 32.0),
        material_id: PREY_ID,
        initial_velocity_scale: 0.0,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let predator_spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(32.0, 32.0),
        material_id: PREDATOR_ID,
        initial_velocity_scale: 0.0,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, prey_spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(lambda, mu)))
        .with_material(PREDATOR_ID, Box::new(NeoHookeanMaterial::new(lambda, mu)))
        .with_material(EATEN_ID, Box::new(NeoHookeanMaterial::new(lambda, mu)));
    let _ = sim.add_body(predator_spawn);
    sim
}

fn run_probe(label: &str, max_substeps_per_step: usize, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step);
    for step in 1..=steps {
        sim.step();
        if step.is_multiple_of(steps / 20) || step == 1 || step == steps {
            let snap = sim.diagnostics_snapshot();
            println!(
                "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] cfl={:.4} \
                 non_finite={} mass_err={:.2e} mom_err={:.2e} time_dropped={:.4}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.cfl_number,
                snap.non_finite_particle_values,
                snap.relative_mass_error,
                snap.relative_momentum_error,
                snap.sim_time_dropped,
            );
            assert_eq!(
                snap.non_finite_particle_values, 0,
                "[{label}] NaN/Inf at step {step}"
            );
        }
    }
    let snap = sim.diagnostics_snapshot();
    assert!(
        snap.sim_time_dropped < 1.0e-6,
        "[{label}] sim_time_dropped={} -- max_substeps_per_step too low",
        snap.sim_time_dropped
    );
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn trophic_predation_real_si_stiffness_substep_sweep() {
    run_probe("substeps=20000", 20_000, 300);
}

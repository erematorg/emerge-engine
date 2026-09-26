//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_snow.rs`'s real-SI migration (Stomakhin 2013 canonical
//! E=1.4e5/nu=0.2, rho=200 kg/m3). Checks real substep need and stability.
extern crate emerge_engine as emerge;

use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_PACKED: u32 = 1;
const MAT_SHATTER: u32 = 2;
const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
const SNOW_YOUNG_MODULUS_PA: f32 = 1.4e5;
const SNOW_POISSON_RATIO: f32 = 0.2;
const SNOW_DENSITY_KG_M3: f32 = 200.0;

fn make_sim(max_substeps_per_step: usize) -> Simulation {
    let config = SimConfig {
        max_substeps_per_step,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SNOW_YOUNG_MODULUS_PA,
        SNOW_POISSON_RATIO,
        SNOW_DENSITY_KG_M3,
    );
    let mass_grid = (SNOW_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(58, 58),
        rng_seed: 7,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(StomakhinMaterial::new(
            lambda, mu, 7.0, 0.025, 0.0075, 0.6, 20.0,
        )))
        .with_material(
            MAT_PACKED,
            Box::new(
                StomakhinMaterial::new(lambda, mu, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0),
            ),
        )
        .with_material(
            MAT_SHATTER,
            Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    solver.retain_particles(|p| {
        (p.x - BALL_A).length() <= BALL_R || (p.x - BALL_B).length() <= BALL_R
    });
    solver.particles_mut().for_each_mut(|p| {
        let dir = if (p.x - BALL_A).length() <= BALL_R {
            Vec2::new(1.0, 0.0)
        } else {
            Vec2::new(-1.0, 0.0)
        };
        p.v = dir * 15.0;
    });
    solver
}

fn run_probe(label: &str, max_substeps_per_step: usize, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step);
    println!("[{label}] max_substeps_per_step={max_substeps_per_step}");
    for step in 1..=steps {
        sim.step();
        let dropped = sim.diagnostics_snapshot().sim_time_dropped;
        if dropped > 0.0 {
            println!("[{label}] step={step} DROPPED SIM TIME dropped={dropped}");
        }
        if step.is_multiple_of((steps / 10).max(1)) || step == 1 {
            let snap = sim.diagnostics_snapshot();
            println!(
                "[{label}] step={step} t={:.2} sub={} cfl={:.4} non_finite={} mass_err={:.2e} mom_err={:.2e}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.cfl_number,
                snap.non_finite_particle_values,
                snap.relative_mass_error,
                snap.relative_momentum_error,
            );
            assert_eq!(
                snap.non_finite_particle_values, 0,
                "[{label}] NaN/Inf at step {step}"
            );
        }
    }
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_snow_real_si_collision() {
    run_probe("real_si", 10000, 30);
}

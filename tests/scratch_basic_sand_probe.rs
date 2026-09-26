//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_sand.rs`'s real-SI migration (real dry sand,
//! E=15 MPa/nu=0.3/rho=1600, same reference already verified for
//! sand_ngf_collapse.rs). Checks stability at real gravity, not just the
//! file's own tiny 0.001 default.
extern crate emerge_engine as emerge;

use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const SAND_YOUNG_MODULUS_PA: f32 = 15.0e6;
const SAND_POISSON_RATIO: f32 = 0.3;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim(max_substeps_per_step: usize, gravity_fraction: f32) -> Simulation {
    let mut config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    config.gravity *= gravity_fraction;
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    let mass_grid = (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = |c: Vec2, mat, seed| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: seed,
        position_jitter: 0.5,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11))
        .with_default_material(Box::new(make_sand(lambda, mu, 20.0)))
        .with_material(MAT_DENSE, Box::new(make_sand(lambda, mu, 40.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22));
    solver
}

fn run_probe(label: &str, max_substeps_per_step: usize, gravity_fraction: f32, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step, gravity_fraction);
    println!(
        "[{label}] max_substeps_per_step={max_substeps_per_step} gravity_fraction={gravity_fraction}"
    );
    for step in 1..=steps {
        sim.step();
        let dropped = sim.diagnostics_snapshot().sim_time_dropped;
        if dropped > 0.0 {
            println!("[{label}] step={step} DROPPED SIM TIME dropped={dropped}");
        }
        if step.is_multiple_of((steps / 10).max(1)) || step == 1 {
            let snap = sim.diagnostics_snapshot();
            let max_speed = sim
                .particles()
                .v
                .iter()
                .map(|v| v.length())
                .fold(0.0f32, f32::max);
            println!(
                "[{label}] step={step} t={:.2} sub={} cfl={:.4} max_speed={max_speed:.3} \
                 non_finite={} mass_err={:.2e} mom_err={:.2e}",
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
fn basic_sand_real_si_at_default_gravity_fraction() {
    run_probe("default_gravity_0.001", 2000, 0.001, 100);
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_sand_real_si_at_full_gravity() {
    run_probe("full_gravity_1.0", 3000, 1.0, 30);
}

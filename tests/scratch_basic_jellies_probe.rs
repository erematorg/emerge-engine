//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_jellies.rs`'s scene (no GUI) to verify the real SI
//! migration (soft-tissue E=500 Pa/nu=0.45/rho=1000, real Earth gravity)
//! is stable at a real substep budget.
extern crate emerge_engine as emerge;

use emerge::{
    CorotatedMaterial, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    ViscoelasticMaterial,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_NEO: u32 = 0;
const MAT_COR: u32 = 1;
const MAT_VIS: u32 = 2;

const JELLY_YOUNG_MODULUS_PA: f32 = 500.0;
const JELLY_POISSON_RATIO: f32 = 0.45;
const JELLY_DENSITY_KG_M3: f32 = 1000.0;
const JELLY_VISCOSITY_PA_S: f32 = 1.0;

fn make_sim(max_substeps_per_step: usize, gravity_fraction: f32, drop_y: f32) -> Simulation {
    let mut config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    config.gravity *= gravity_fraction;
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        JELLY_YOUNG_MODULUS_PA,
        JELLY_POISSON_RATIO,
        JELLY_DENSITY_KG_M3,
    );
    let visc = config.visc_from_si_physical(JELLY_VISCOSITY_PA_S, JELLY_DENSITY_KG_M3);
    println!("[jellies-probe] lambda={lambda} mu={mu} visc={visc}");
    let spawn = |c: Vec2, mat| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn(Vec2::new(14.0, drop_y), MAT_NEO))
        .with_default_material(Box::new(NeoHookeanMaterial::new(lambda, mu)))
        .with_material(MAT_COR, Box::new(CorotatedMaterial::new(lambda, mu)))
        .with_material(
            MAT_VIS,
            Box::new(ViscoelasticMaterial::new(lambda, mu, visc)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(32.0, drop_y), MAT_COR));
    let _ = solver.add_body(spawn(Vec2::new(50.0, drop_y), MAT_VIS));
    solver
}

fn run_probe(label: &str, drop_y: f32) {
    let mut sim = make_sim(20000, 1.0, drop_y);
    for step in 1..=60u64 {
        sim.step();
        if step.is_multiple_of(10) || step == 1 {
            let snap = sim.diagnostics_snapshot();
            let min_y = sim
                .particles()
                .x
                .iter()
                .map(|p| p.y)
                .fold(f32::MAX, f32::min);
            let particles = sim.particles();
            let n = particles.len();
            let near_min_j: Vec<usize> = (0..n)
                .filter(|&i| particles.deformation_gradient[i].determinant() < 1.0e-3)
                .collect();
            println!(
                "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] cfl={:.4} min_y={min_y:.2} \
                 non_finite={} mass_err={:.2e} mom_err={:.2e} n_near_min_j={}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.cfl_number,
                snap.non_finite_particle_values,
                snap.relative_mass_error,
                snap.relative_momentum_error,
                near_min_j.len(),
            );
            assert_eq!(
                snap.non_finite_particle_values, 0,
                "[{label}] NaN/Inf at step {step}"
            );
        }
    }
}

fn run_probe_long(label: &str, drop_y: f32, steps: u64) {
    let mut sim = make_sim(20000, 1.0, drop_y);
    for step in 1..=steps {
        sim.step();
        if step.is_multiple_of(steps / 20) || step == 1 || step == steps {
            let snap = sim.diagnostics_snapshot();
            let particles = sim.particles();
            let n = particles.len();
            let near_min_j = (0..n)
                .filter(|&i| particles.deformation_gradient[i].determinant() < 1.0e-3)
                .count();
            println!(
                "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] non_finite={} n_near_min_j={near_min_j}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.non_finite_particle_values,
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
fn basic_jellies_real_si_stiffness_drop_y15_long_horizon() {
    run_probe_long("drop_y=15_long", 15.0, 3000);
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_jellies_real_si_stiffness_drop_height_sweep() {
    // Original scene drop height (50) inverted ~370 Corotated particles.
    // Sweep down to find where a real, correctly-stiff soft-tissue material
    // stops producing a hard enough impact to invert the linearized
    // corotational model.
    run_probe("drop_y=50_original", 50.0);
    run_probe("drop_y=25", 25.0);
    run_probe("drop_y=15", 15.0);
    run_probe("drop_y=10", 10.0);
}

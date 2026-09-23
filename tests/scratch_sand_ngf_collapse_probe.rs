//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/sand_ngf_collapse.rs`'s scene (no GUI/GPU) to measure the
//! real substep/stability consequence of migrating its Lame conversion from
//! `lame_from_si_cfg` (old, dt^2-polluted) to `lame_from_si_physical_cfg`
//! (new, dt-independent) -- same discipline as the membrane migration probe.
extern crate emerge_engine as emerge;

use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 96;
const CELL_M: f32 = 0.01;
const DT_S: f32 = 0.01;
const YOUNG_MODULUS_PA: f32 = 15.0e6;
const POISSON_RATIO: f32 = 0.3;
const BULK_DENSITY_KG_M3: f32 = 1600.0;
const FRICTION_DEG: f32 = 35.0;
const R0_CELLS: f32 = 4.0;
const H0_CELLS: f32 = 16.0;
const FLOOR_CELLS: f32 = 5.0;

fn make_sim(max_substeps_per_step: usize, use_physical: bool) -> Simulation {
    let config = SimConfig {
        max_substeps_per_step,
        ..SimConfig::earth(GRID, CELL_M, DT_S)
    };
    // Real fix (2026-09-05): mass now shares BULK_DENSITY_KG_M3 with the
    // stiffness conversion (was left on grid_density=1.0 default) --
    // matches the same fix applied to the real example.
    let mass_grid = if use_physical {
        (BULK_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5
    } else {
        0.5 * 0.5
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + 8.0),
        material_id: 0,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let (lambda, mu) = if use_physical {
        config.lame_from_si_physical_cfg(YOUNG_MODULUS_PA, POISSON_RATIO, BULK_DENSITY_KG_M3)
    } else {
        config.lame_from_si_cfg(YOUNG_MODULUS_PA, POISSON_RATIO, BULK_DENSITY_KG_M3)
    };
    let sand = DruckerPragerMaterial {
        friction_angle: FRICTION_DEG.to_radians(),
        dilatancy_angle: 0.0,
        ngf_enabled: false,
        ..DruckerPragerMaterial::new(lambda, mu)
    };
    Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)))
}

fn measured_spread_cells(sim: &Simulation) -> f32 {
    let xs = &sim.particles().x;
    let n = xs.len() as f32;
    if n == 0.0 {
        return 0.0;
    }
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    xs.iter()
        .map(|p| (p.x - center_x).abs())
        .fold(0.0f32, f32::max)
}

fn predicted_r_inf_cells() -> f32 {
    let aspect = H0_CELLS / R0_CELLS;
    R0_CELLS * (1.0 + 2.0 * aspect.sqrt())
}

fn run_probe(label: &str, max_substeps_per_step: usize, use_physical: bool, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step, use_physical);
    let r_inf = predicted_r_inf_cells();
    println!(
        "[{label}] lambda_mu_source={} max_substeps_per_step={max_substeps_per_step} r_inf_predicted={r_inf:.2}",
        if use_physical {
            "physical"
        } else {
            "old_buggy"
        }
    );
    for step in 1..=steps {
        sim.step();
        if step.is_multiple_of(steps / 10) || step == steps {
            let snap = sim.diagnostics_snapshot();
            let spread = measured_spread_cells(&sim);
            println!(
                "[{label}] step={step} t={:.3} sub={} J=[{:.4},{:.4}] cfl={:.4} \
                 spread={spread:.3} ratio={:.3}x non_finite={} mass_err={:.2e} mom_err={:.2e}",
                step as f32 * DT_S,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.cfl_number,
                spread / r_inf,
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
fn sand_ngf_collapse_old_vs_new_conversion_at_original_substep_budget() {
    // The file's own shipped budget (4000) -- was it enough for the OLD
    // (much softer, dt^2-shrunk) stiffness, and is it still enough for the
    // NEW, correct, much stiffer one?
    run_probe("old_conversion@4000", 4000, false, 300);
    run_probe("new_conversion@4000", 4000, true, 300);
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn sand_ngf_collapse_new_conversion_raised_budget() {
    run_probe("new_conversion@40000", 40000, true, 300);
}

//! One spawn contract: a body's initial volume must not depend on how it
//! was added.
//!
//! `Simulation::new` and `add_body` are two ways to put the same body in
//! the same world. They used to disagree: `add_body` always measured the
//! initial volume from the particles' own packing, while `Simulation::new`
//! only did so when `precompute_initial_volumes` was set, and that flag
//! defaulted to off. A scene that left it alone therefore gave its FIRST
//! body an arbitrary `default_initial_volume` and every later body the
//! measured one, 3.1 times smaller. Initial volume multiplies stress
//! directly, so those bodies were not the same material.
//!
//! Three checks, because each of the others alone missed something:
//!
//! - the two paths must agree, whatever the scene asked for;
//! - they must agree for a material that sets its own initial volume too,
//!   which the elastic check above cannot see (see
//!   `a_fluid_body_is_the_same_whichever_path_adds_it`);
//! - the volume they agree on must be the physically right one, which a
//!   column at rest proves by carrying its own weight.
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, MaterialModel, NeoHookeanMaterial, SimConfig,
    Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 48;
const DX_M: f32 = 0.01;
const SPACING: f32 = 0.5;
const COLUMN: IVec2 = IVec2::new(6, 12);
const RHO_KG_M3: f32 = 1000.0;
const YOUNG_PA: f32 = 2.0e5;
const POISSON: f32 = 0.3;

fn config() -> SimConfig {
    SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 128,
        ..SimConfig::earth(GRID, DX_M, 0.0005)
    }
}

/// Two identical columns, one spawned by each path, in one world.
fn two_columns() -> (Simulation, usize) {
    let config = config();
    let material = NeoHookeanMaterial::from_young_modulus(YOUNG_PA, POISSON);
    let mass = RHO_KG_M3 * (SPACING * DX_M).powi(2);
    let spawn = |x: f32| SpawnRegion {
        spacing: SPACING,
        box_size: COLUMN,
        box_center: Vec2::new(x, 3.0 + COLUMN.y as f32 * 0.5),
        material_id: 0,
        mass_override: Some(mass),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn(14.0))
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let first = sim.particles().len();
    let _ = sim.add_body(spawn(34.0));
    (sim, first)
}

fn mean_initial_volume(sim: &Simulation, range: std::ops::Range<usize>) -> f32 {
    let p = sim.particles();
    let n = range.len() as f32;
    range.map(|i| p.initial_volume[i]).sum::<f32>() / n
}

/// The two paths must produce the same body.
#[test]
fn both_spawn_paths_give_a_body_the_same_initial_volume() {
    let (sim, first) = two_columns();
    let total = sim.particles().len();
    let a = mean_initial_volume(&sim, 0..first);
    let b = mean_initial_volume(&sim, first..total);
    let gap = (a - b).abs() / a.max(b);
    assert!(
        gap < 1.0e-3,
        "the same body spawned two ways must carry the same initial volume: \
         Simulation::new gave {a}, add_body gave {b}, a factor of {:.2}",
        a.max(b) / a.min(b)
    );
}

/// The same contract for a material that sets its OWN initial volume.
///
/// The elastic check above passed while this was broken, and could not have
/// failed. Both paths estimate a new body's volume from its packing, then
/// run the material's `init_particle`, and an elastic law's `init_particle`
/// leaves the volume alone, so the estimate stands in both and they agree.
/// A fluid's `init_particle` sets `mass / rest_density` instead, and
/// `add_body` used to run it BEFORE the estimate while `Simulation::new`
/// ran it after. So in `add_body` the estimate had the last word, and at a
/// free surface the estimate inflates a particle's volume up to 2.56 times.
///
/// Measured before the fix (`tests/scratch_bingham_column_volume_loss.rs`):
/// the same column carried a mean 0.250000 through `Simulation::new` and
/// 0.280036 through `add_body`, worst particle 0.640000, and three
/// identical columns in one world ended at mean J 0.99907, 0.94304 and
/// 0.94441 depending only on which path had added them.
///
/// It has to be a yield-stress fluid WITH a storage modulus. A Newtonian
/// fluid and a purely viscous Bingham one both declare they own their
/// deformation volume, so the estimate skips them entirely and the order
/// never mattered for them.
#[test]
fn a_fluid_body_is_the_same_whichever_path_adds_it() {
    let config = config();
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: 0.5,
        bulk_modulus_pa: 78_480.0,
        yield_stress_pa: 60.0,
        shear_modulus_pa: 60.0 / 0.05,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let spawn = |x: f32| {
        SpawnRegion {
            spacing: SPACING,
            box_size: COLUMN,
            box_center: Vec2::new(x, 3.0 + COLUMN.y as f32 * 0.5),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };
    let mut sim = Simulation::new(config, spawn(14.0))
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let first = sim.particles().len();
    let _ = sim.add_body(spawn(34.0));
    let total = sim.particles().len();

    let spread = |range: std::ops::Range<usize>| {
        let p = sim.particles();
        range
            .map(|i| p.initial_volume[i])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            })
    };
    let a = mean_initial_volume(&sim, 0..first);
    let b = mean_initial_volume(&sim, first..total);
    let (a_lo, a_hi) = spread(0..first);
    let (b_lo, b_hi) = spread(first..total);
    // The mean alone would pass a body whose edges are inflated and whose
    // interior is shrunk to compensate, so the range must match as well.
    assert!(
        (a - b).abs() / a.max(b) < 1.0e-3,
        "a fluid body must carry the same initial volume whichever path added it: Simulation::new gave a mean of {a}, add_body {b}"
    );
    assert!(
        (a_hi - b_hi).abs() < 1.0e-6 && (a_lo - b_lo).abs() < 1.0e-6,
        "a fluid body's initial volume must span the same range whichever path added it: Simulation::new gave {a_lo} to {a_hi}, add_body {b_lo} to {b_hi}"
    );
}

/// And the volume they agree on has to be the real one. A column at rest
/// carries its own weight: the vertical stress at depth h is rho*g*h, which
/// only comes out right if each particle's initial volume is the volume it
/// actually occupies -- initial volume multiplies stress directly, so a V0
/// three times too large is a column three times too stiff.
///
/// The bar is 35 % because this column is a coarse discretisation of a
/// continuum, not because 14 % was accepted on faith. The same physical
/// column, 6 by 12 cm, was run at three cell sizes
/// (`tests/scratch_column_convergence.rs`):
///
/// ```text
///   cell size   particles   measured    analytic   off by
///    1.00 cm       288      -760.2 Pa   -881.3     -13.7 %
///    0.50 cm      1152      -820.0 Pa   -880.9      -6.9 %
///    0.25 cm      4608      -849.3 Pa   -880.5      -3.5 %
/// ```
///
/// Halving the cells halves the error, twice over: first order, which is
/// what a discretisation error looks like and what a residual defect does
/// not. This test stays coarse on purpose, to catch a factor of three in
/// a second rather than to measure convergence.
#[test]
fn a_settled_column_carries_its_own_weight() {
    let (mut sim, first) = two_columns();
    let g = sim.config().gravity.length();
    let material = NeoHookeanMaterial::from_young_modulus(YOUNG_PA, POISSON);
    for _ in 0..2000 {
        sim.step();
    }
    // Nothing damps this column, so it rings about its equilibrium for
    // ever: a single instant reads anywhere between zero and twice the
    // answer. The time average over a long window IS the static state,
    // which is what the analytic value describes.
    let (mut measured, mut expected, mut samples) = (0.0f64, 0.0f64, 0u32);
    for _ in 0..2000 {
        sim.step();
        let (a, b, n) = column_stress(&sim, &material, first, g);
        if n > 0 {
            measured += a;
            expected += b;
            samples += 1;
        }
    }
    let measured = measured / f64::from(samples);
    let expected = expected / f64::from(samples);
    println!(
        "settled column, averaged over {samples} steps: mean vertical stress {measured:.1} Pa against rho*g*h = {expected:.1} Pa"
    );
    let error = ((measured - expected) / expected).abs();
    assert!(
        error < 0.35,
        "a column at rest must carry its own weight: measured {measured:.1} Pa against \
         the analytic {expected:.1} Pa, {:.0} % off",
        100.0 * error
    );
}

/// Mean vertical Cauchy stress in the column's lower half, in pascals,
/// next to what its own weight says it should be.
fn column_stress(
    sim: &Simulation,
    material: &NeoHookeanMaterial,
    first: usize,
    g: f32,
) -> (f64, f64, u32) {
    let p = sim.particles();
    let top = (0..first).map(|i| p.x[i].y).fold(f32::MIN, f32::max);
    let (mut measured, mut expected, mut n) = (0.0f64, 0.0f64, 0u32);
    for i in 0..first {
        let depth_cells = top - p.x[i].y;
        if depth_cells < COLUMN.y as f32 * 0.5 {
            continue;
        }
        let depth_m = depth_cells * DX_M;
        let j = p.deformation_gradient[i].determinant().max(1.0e-6);
        let sigma_yy = material.kirchhoff_stress(p, i).y_axis.y / j;
        // Grid stress IS pascals here, with no conversion: this engine's
        // own units give `rho_grid * g_grid * h_cells = (rho dx^2)(g/dx)
        // (h/dx) = rho g h`, so the two sides are already the same number.
        // Converting the stress by the MASS factor (`rho dx^2`) reads 91 %
        // low, which is what this test said before the derivation was
        // written down.
        measured += f64::from(sigma_yy);
        expected += f64::from(-RHO_KG_M3 * g * DX_M * depth_m);
        n += 1;
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    (measured / f64::from(n), expected / f64::from(n), n)
}

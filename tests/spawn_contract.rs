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
//! Two checks, because one alone would not have caught it:
//!
//! - the two paths must agree, whatever the scene asked for;
//! - the volume they agree on must be the physically right one, which a
//!   column at rest proves by carrying its own weight.
extern crate emerge_engine as emerge;

use emerge::{MaterialModel, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
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

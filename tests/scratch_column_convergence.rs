//! Is the settled column's 14 % gap discretisation or something left?
//!
//! `tests/spawn_contract.rs` checks that a column at rest carries its own
//! weight, and reads 756 Pa against the analytic 881. That bar is wide
//! enough to hide a real defect, so this runs the SAME physical column,
//! 6 by 12 cm of the same material under the same gravity, at three cell
//! sizes. A discretisation error shrinks as the cells do; anything else
//! stays put.
extern crate emerge_engine as emerge;

use emerge::{MaterialModel, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const RHO_KG_M3: f32 = 1000.0;
const YOUNG_PA: f32 = 2.0e5;
const POISSON: f32 = 0.3;
const SPACING: f32 = 0.5;

fn column_error(grid: usize, dx_m: f32, cells: IVec2, steps: usize) -> (f64, f64, u32) {
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 256,
        ..SimConfig::earth(grid, dx_m, 0.0005)
    };
    let material = NeoHookeanMaterial::from_young_modulus(YOUNG_PA, POISSON);
    let mass = RHO_KG_M3 * (SPACING * dx_m).powi(2);
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: cells,
        box_center: Vec2::new(grid as f32 * 0.4, 3.0 + cells.y as f32 * 0.5),
        material_id: 0,
        mass_override: Some(mass),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let g = sim.config().gravity.length();
    let n_particles = sim.particles().len();
    for _ in 0..steps {
        sim.step();
    }
    let (mut measured, mut expected, mut samples) = (0.0f64, 0.0f64, 0u32);
    for _ in 0..steps {
        sim.step();
        let p = sim.particles();
        let top = (0..n_particles).map(|i| p.x[i].y).fold(f32::MIN, f32::max);
        let (mut m, mut e, mut n) = (0.0f64, 0.0f64, 0u32);
        for i in 0..n_particles {
            let depth_cells = top - p.x[i].y;
            if depth_cells < cells.y as f32 * 0.5 {
                continue;
            }
            let j = p.deformation_gradient[i].determinant().max(1.0e-6);
            m += f64::from(material.kirchhoff_stress(p, i).y_axis.y / j);
            e += f64::from(-RHO_KG_M3 * g * dx_m * (depth_cells * dx_m));
            n += 1;
        }
        if n > 0 {
            measured += m / f64::from(n);
            expected += e / f64::from(n);
            samples += 1;
        }
    }
    (
        measured / f64::from(samples),
        expected / f64::from(samples),
        n_particles as u32,
    )
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_settled_column_converges_as_the_cells_shrink() {
    for (grid, dx, cells, steps) in [
        (48usize, 0.01f32, IVec2::new(6, 12), 2000usize),
        (96, 0.005, IVec2::new(12, 24), 3000),
        (192, 0.0025, IVec2::new(24, 48), 4000),
    ] {
        let (measured, expected, n) = column_error(grid, dx, cells, steps);
        println!(
            "dx={dx:.4} m, {} x {} cells, {n} particles: measured {measured:8.1} Pa, analytic {expected:8.1} Pa, {:+.1} % off",
            cells.x,
            cells.y,
            100.0 * (measured - expected) / expected
        );
    }
}

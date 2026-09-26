//! Is the "twist" visible in the slump demo's columns physics, or an
//! artefact?
//!
//! A yield-stress column first yields where the shear stress is largest,
//! at its base, and keeps a rigid plug above, so a column bent near the
//! floor and straight on top is what the physics predicts. What it does
//! NOT predict is a lean to one side: the scene is mirror-symmetric, so
//! every column must collapse symmetrically about its own axis. The test is
//! that each column's centre of mass does not drift sideways, that its net
//! angular momentum stays near zero, and that its deposit ends as far on
//! the left of the axis as on the right.
//!
//! One thing breaks the symmetry before anything runs. `initialize_particles`
//! lays the lattice from the box's lower corner in steps of `spacing`, so a
//! 10-cell box centred at x = 32 holds particles from 27.0 to 36.5 and the
//! body's own axis is 31.75, a quarter cell left of the box centre and of
//! the tank's axis. And 31.75 is not a mirror line of the grid at all, whose
//! nodes sit at the integers. So even an exact solver could not keep that
//! body symmetric. Each column is therefore run twice: once spawned as the
//! demo spawns it, and once with the box moved a quarter cell right so the
//! body's axis lands on x = 32, a mirror line of the grid and of the tank.
//! Then the demo's own three-column arrangement, where the left deposit
//! reaches the middle one, to see what the contact adds.
//!
//!   RAYON_NUM_THREADS=1 cargo test --profile quick --all-features --test scratch_bingham_column_symmetry -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

// The slump demo's own scene, constant for constant.
const GRID: usize = 64;
const DX_M: f32 = 0.002;
const RHO_KG_M3: f32 = 1000.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
const FLOOR_CELLS: f32 = 2.0;
const YIELDS_PA: [f32; 3] = [2.0, 60.0, 1200.0];
const DEMO_X: [f32; 3] = [12.0, 32.0, 52.0];

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The demo's bulk modulus, from its own derivation: sound speed ten times
/// the free-fall speed over the column's height, `K = rho c^2`.
fn bulk_modulus_pa() -> f32 {
    let v_max = (2.0 * 9.81 * COLUMN_CELLS.y as f32 * DX_M).sqrt();
    RHO_KG_M3 * (10.0 * v_max).powi(2)
}

fn props(tau0: f32) -> BinghamProps {
    BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: bulk_modulus_pa(),
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    }
}

fn config(dt: f32) -> SimConfig {
    SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    }
}

fn column(config: &SimConfig, x: f32, slot: u32, tau0: f32) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(x, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
        material_id: slot,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(config)
    }
    .mass_from(&props(tau0), config)
}

/// Mirror measures of one body, selected by material slot.
struct Mirror {
    /// Centre of mass, in cells.
    x_cm: f64,
    /// How far the body reaches on each side of `axis`, in cells.
    left: f64,
    right: f64,
    /// Net angular momentum about the centre of mass over the total one
    /// would have if every particle turned the same way: 0 for a body whose
    /// motion is mirror-symmetric, 1 for a body spinning as one.
    spin: f64,
    /// The same total, to tell a quiet body (where `spin` means nothing)
    /// from a moving one.
    activity: f64,
}

fn mirror(sim: &Simulation, slot: u32, axis: f64) -> Mirror {
    let p = sim.particles();
    let idx: Vec<usize> = (0..p.len()).filter(|&i| p.material_id[i] == slot).collect();
    let mass: f64 = idx.iter().map(|&i| f64::from(p.mass[i])).sum();
    let x_cm = idx
        .iter()
        .map(|&i| f64::from(p.mass[i]) * f64::from(p.x[i].x))
        .sum::<f64>()
        / mass;
    let y_cm = idx
        .iter()
        .map(|&i| f64::from(p.mass[i]) * f64::from(p.x[i].y))
        .sum::<f64>()
        / mass;
    let (mut net, mut total) = (0.0f64, 0.0f64);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for &i in &idx {
        let rx = f64::from(p.x[i].x) - x_cm;
        let ry = f64::from(p.x[i].y) - y_cm;
        let (vx, vy) = (f64::from(p.v[i].x), f64::from(p.v[i].y));
        let m = f64::from(p.mass[i]);
        net += m * (rx * vy - ry * vx);
        total += m * (rx * rx + ry * ry).sqrt() * (vx * vx + vy * vy).sqrt();
        lo = lo.min(f64::from(p.x[i].x));
        hi = hi.max(f64::from(p.x[i].x));
    }
    Mirror {
        x_cm,
        left: axis - lo,
        right: hi - axis,
        spin: if total > 0.0 { net / total } else { 0.0 },
        activity: total,
    }
}

/// Worst sideways drift of the centre of mass, and worst spin while the
/// body is still genuinely moving, over the run; then the final extents.
struct Report {
    drift_max_mm: f64,
    drift_end_mm: f64,
    spin_max: f64,
    left_mm: f64,
    right_mm: f64,
}

fn run(sim: &mut Simulation, slot: u32, axis: f64, seconds: f32, dt: f32) -> Report {
    let start = mirror(sim, slot, axis).x_cm;
    let frames = (seconds / dt).round() as usize;
    let (mut drift_max, mut spin_max, mut peak_activity) = (0.0f64, 0.0f64, 0.0f64);
    let mut last = mirror(sim, slot, axis);
    for _ in 0..frames {
        sim.step();
        last = mirror(sim, slot, axis);
        drift_max = drift_max.max((last.x_cm - start).abs());
        peak_activity = peak_activity.max(last.activity);
        // A settled body's spin ratio is the ratio of two numbers near
        // zero, so it only counts while the body still carries at least a
        // hundredth of its peak motion.
        if last.activity > 0.01 * peak_activity {
            spin_max = spin_max.max(last.spin.abs());
        }
    }
    let mm = f64::from(DX_M) * 1000.0;
    Report {
        drift_max_mm: drift_max * mm,
        drift_end_mm: (last.x_cm - start) * mm,
        spin_max,
        left_mm: last.left * mm,
        right_mm: last.right * mm,
    }
}

fn print(label: &str, r: &Report) {
    println!(
        "  {label:<34} {:>8.4} {:>9.4}   {:>7.4}   {:>7.2} {:>7.2} {:>+8.3}",
        r.drift_max_mm,
        r.drift_end_mm,
        r.spin_max,
        r.left_mm,
        r.right_mm,
        r.right_mm - r.left_mm
    );
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn a_lone_column_collapses_symmetrically() {
    let seconds = env("SYMMETRY_SECONDS", 3.0);
    let dt = env("SYMMETRY_DT", 0.001);
    println!("{seconds} s at {dt} s a frame, demo scene constants, each column alone in the tank");
    println!(
        "  column, spawn                      drift mm   end mm    spin max   left mm right mm  R-L mm"
    );
    for tau0 in YIELDS_PA {
        // As the demo spawns it: box centred on 32, body axis at 31.75.
        // Then a quarter cell right: body axis on 32, the grid's and the
        // tank's mirror line.
        for (label, box_x) in [
            ("demo spawn, axis 31.75", 32.0f32),
            ("mirror spawn, axis 32.00", 32.25),
        ] {
            let config = config(dt);
            let spawn = column(&config, box_x, 0, tau0);
            let mut sim = Simulation::new(config, spawn)
                .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
                    &props(tau0),
                    &config,
                )))
                .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
            let axis = mirror(&sim, 0, 0.0).x_cm;
            let r = run(&mut sim, 0, axis, seconds, dt);
            print(&format!("{tau0:>6} Pa, {label}"), &r);
        }
    }

    // The demo's own arrangement: three columns in one tank, the left
    // deposit reaching the middle one. Same measures, per column, about
    // each body's own starting axis.
    println!("the demo's three columns together, same measures per column");
    let config = config(dt);
    let mut sim = Simulation::new(config, column(&config, DEMO_X[0], 0, YIELDS_PA[0]))
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props(YIELDS_PA[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS_PA[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(YIELDS_PA[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(column(&config, DEMO_X[1], 1, YIELDS_PA[1]));
    let _ = sim.add_body(column(&config, DEMO_X[2], 2, YIELDS_PA[2]));
    let axes: Vec<f64> = (0..3u32).map(|s| mirror(&sim, s, 0.0).x_cm).collect();
    let starts = axes.clone();
    let frames = (seconds / dt).round() as usize;
    let mut drift_max = [0.0f64; 3];
    for _ in 0..frames {
        sim.step();
        for s in 0..3 {
            let m = mirror(&sim, s as u32, axes[s]);
            drift_max[s] = drift_max[s].max((m.x_cm - starts[s]).abs());
        }
    }
    let mm = f64::from(DX_M) * 1000.0;
    for s in 0..3 {
        let m = mirror(&sim, s as u32, axes[s]);
        println!(
            "  {:>6} Pa, together                 {:>8.4} {:>9.4}   {:>7}   {:>7.2} {:>7.2} {:>+8.3}",
            YIELDS_PA[s],
            drift_max[s] * mm,
            (m.x_cm - starts[s]) * mm,
            "-",
            m.left * mm,
            m.right * mm,
            (m.right - m.left) * mm
        );
    }
}

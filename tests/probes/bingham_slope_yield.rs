//! Does a yield-stress layer on a slope hold below its yield thickness and
//! flow above it?
//!
//! An infinite layer of thickness h on a slope theta carries a basal shear
//! stress rho g h sin(theta); a Bingham material is at rest while that is
//! below its yield stress, so it flows only above
//!   h_c = tau_0 / (rho g sin(theta)).
//! The layer is the slump demo's 60 Pa material (eta 0.5 Pa s, storage
//! modulus tau_0 / 0.05, bulk modulus 78 480 Pa, `bingham_slump_scene.rs`),
//! 20 mm thick and 25 thicknesses long between two walls, so the middle
//! third, which is watched, is about eight thicknesses from either end. The
//! slope is chosen so h / h_c = `BINGHAM_SLOPE_RATIO`; gravity is ramped
//! from zero over 2 s along (1 - cos) so the elastic part below yield is
//! not rung by the loading, then held.
//!
//! The base is a substrate of pinned particles, which constrains its grid
//! nodes: a no-slip plane. The demos' `FrictionBoundary` floor (Coulomb,
//! mu = 1, at tan(theta) = 0.30) gives nearly the same layer
//! (`BINGHAM_SLOPE_FLOOR=friction`): at 0.85 h_c both rest after 0.33 mm,
//! and at 0.95 h_c, 2 s after the ramp, the rows from the base move 2.7,
//! 4.7 and 5.4 mm on the friction floor against 2.5, 4.4 and 5.0 mm on the
//! pinned base. The bottom rows lag the same way on a base that cannot
//! slide, so that lag is the shear in the bottom cell, not the floor giving
//! way.
//!
//! Measured, 2 mm cells (10 over the thickness), middle third, 4 s after the
//! ramp:
//!
//! ```text
//!   h / h_c   displacement   still moving at
//!   0.85      0.3 mm         rest
//!   0.90      1.0 mm         rest
//!   0.95      6.4 mm         0.5 mm/s, slowing
//!   1.05      24.5 mm        2.6 mm/s, slowing as the walled layer drains
//! ```
//!
//! The law is right in the bulk, but the onset comes about 10 % early: at
//! the no-slip base the bottom cell carries more shear than the layer
//! above it does. At 0.90 h_c, shear over tau_0 by particle row from the
//! base reads 1.01, 1.01, 0.64, 0.64, 0.63 where the infinite layer gives
//! 0.88, 0.83, 0.79, 0.74, 0.70: the same total over the five rows (0.785
//! against 0.787), shifted into the bottom cell, which reaches the yield
//! stress first and lets the layer creep on it. 1 mm cells do not help (at
//! 0.95 h_c, 8.7 mm against 6.4 mm), so this is not resolution.
//! Not fixed.
//!
//!   BINGHAM_SLOPE_RATIO=0.95 cargo test --test probes bingham_slope_yield:: -- --ignored --nocapture
//!
//! Knobs: BINGHAM_SLOPE_RATIO (0.95), BINGHAM_SLOPE_DX (m, 0.002),
//! BINGHAM_SLOPE_CELLS (10), BINGHAM_SLOPE_HOLD (s, 4),
//! BINGHAM_SLOPE_FLOOR (`pinned`, or `friction` for the demos' floor).

use emerge::{
    BinghamFluidMaterial, BinghamProps, FrictionBoundary, FromSI, MaterialModel, SimConfig,
    Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn yield_stress_layer_holds_below_its_yield_thickness() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let ratio = env("BINGHAM_SLOPE_RATIO", 0.95);
    let dx = env("BINGHAM_SLOPE_DX", 0.002);
    let cells = env("BINGHAM_SLOPE_CELLS", 10.0) as i32;
    let hold = env("BINGHAM_SLOPE_HOLD", 4.0);
    let pinned_base = std::env::var("BINGHAM_SLOPE_FLOOR").as_deref() != Ok("friction");
    const RHO: f32 = 1000.0;
    const G: f32 = 9.81;
    const TAU0: f32 = 60.0;
    const RAMP_S: f32 = 2.0;
    const FRAME_S: f32 = 0.005;

    let grid = (cells as usize * 25).next_power_of_two();
    let h_m = cells as f32 * dx;
    let sin_theta = ratio * TAU0 / (RHO * G * h_m);
    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 4096,
        ..SimConfig::earth(grid, dx, FRAME_S)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO,
        eta_pa_s: 0.5,
        bulk_modulus_pa: 78_480.0,
        yield_stress_pa: TAU0,
        shear_modulus_pa: TAU0 / 0.05,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let material = BinghamFluidMaterial::from_physical(&props, &config);
    let wall = config.boundary_thickness as f32;
    let length = grid as i32 - 2 * config.boundary_thickness as i32;
    let region = |height: i32, bottom: f32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(length, height),
            box_center: Vec2::new(grid as f32 * 0.5, bottom + height as f32 * 0.5),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };
    // Pinned base: three cells of pinned particles; their topmost row pins
    // grid nodes one cell above it, where the layer starts.
    let base = if pinned_base { wall + 4.0 } else { wall };
    let mut sim = if pinned_base {
        let mut sim = Simulation::new(config, region(3, wall))
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(FrictionBoundary::new(
                config.boundary_thickness,
                1.0,
            )));
        let p = sim.particles_mut();
        for i in 0..p.len() {
            p.pinned[i] = 1;
        }
        let _ = sim.add_body(region(cells, base));
        sim
    } else {
        Simulation::new(config, region(cells, base))
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(FrictionBoundary::new(
                config.boundary_thickness,
                1.0,
            )))
    };
    let layer: Vec<usize> = (0..sim.particles().len())
        .filter(|&i| sim.particles().pinned[i] == 0)
        .collect();

    let g_full = Vec2::new(sin_theta, -sin_theta.asin().cos()) * (G / dx);
    let frames = |seconds: f32| (seconds / FRAME_S).round() as usize;
    for f in 0..frames(RAMP_S) {
        let s = (f + 1) as f32 / frames(RAMP_S) as f32;
        sim.set_gravity(g_full * (0.5 - 0.5 * (std::f32::consts::PI * s).cos()));
        sim.step();
    }
    sim.set_gravity(g_full);
    println!(
        "h/h_c {ratio}: slope {:.2} deg, layer {:.1} mm, h_c {:.2} mm, {} base, {} cells over the thickness",
        sin_theta.asin().to_degrees(),
        h_m * 1e3,
        TAU0 / (RHO * G * sin_theta) * 1e3,
        if pinned_base { "pinned" } else { "friction" },
        cells
    );

    // Shear by particle row in the middle third when the ramp ends, against
    // the infinite layer's rho g sin(theta) (surface - y).
    let (lo, hi) = (grid as f32 / 3.0, 2.0 * grid as f32 / 3.0);
    let middle: Vec<usize> = layer
        .iter()
        .copied()
        .filter(|&i| (lo..hi).contains(&sim.particles().x[i].x))
        .collect();
    let rows = cells as usize * 2;
    let row_of = |y: f32| ((y - base) / 0.5).round().max(0.0) as usize;
    let (mut shear, mut count) = (vec![0.0f64; rows], vec![0usize; rows]);
    for &i in &middle {
        let r = row_of(sim.particles().x[i].y);
        if r < rows {
            let tau = material.kirchhoff_stress(sim.particles(), i);
            shear[r] += f64::from(0.5 * (tau.x_axis.y + tau.y_axis.x)).abs()
                / f64::from(material.yield_stress);
            count[r] += 1;
        }
    }
    let row_line: Vec<String> = (0..rows.min(6))
        .map(|r| {
            let depth = cells as f32 - 0.5 * r as f32 - 0.25;
            format!(
                "{:.2} ({:.2})",
                shear[r] / count[r].max(1) as f64,
                ratio * depth / cells as f32
            )
        })
        .collect();
    println!(
        "  shear / tau_0 by row from the base (infinite layer): {}",
        row_line.join(" ")
    );

    let start: Vec<f32> = middle.iter().map(|&i| sim.particles().x[i].x).collect();
    let start_y: Vec<f32> = middle.iter().map(|&i| sim.particles().x[i].y).collect();
    let mean_shift = |sim: &Simulation| -> f64 {
        middle
            .iter()
            .zip(&start)
            .map(|(&i, &x0)| f64::from(sim.particles().x[i].x - x0))
            .sum::<f64>()
            / middle.len() as f64
            * f64::from(dx)
            * 1e3
    };
    for second in 1..=(hold.round() as usize) {
        for _ in 0..frames(1.0) {
            sim.step();
        }
        let speed = middle
            .iter()
            .map(|&i| f64::from(sim.particles().v[i].x))
            .sum::<f64>()
            / middle.len() as f64
            * f64::from(dx);
        println!(
            "  {second} s after the ramp: middle third moved {:.2} mm, speed {:.2e} m/s",
            mean_shift(&sim),
            speed
        );
    }

    // Across the thickness: a layer sliding on its floor moves its bottom
    // row as far as the rest; a sheared one leaves it behind.
    let (mut moved, mut n) = (vec![0.0f64; rows], vec![0usize; rows]);
    for ((&i, &x0), &y) in middle.iter().zip(&start).zip(&start_y) {
        let r = row_of(y);
        if r < rows {
            moved[r] += f64::from(sim.particles().x[i].x - x0) * f64::from(dx) * 1e3;
            n[r] += 1;
        }
    }
    let line: Vec<String> = (0..rows.min(4))
        .map(|r| format!("{:.2}", moved[r] / n[r].max(1) as f64))
        .collect();
    println!(
        "  moved (mm) by row from the base: {} ... surface {:.2}",
        line.join(" "),
        moved[rows - 1] / n[rows - 1].max(1) as f64
    );
}

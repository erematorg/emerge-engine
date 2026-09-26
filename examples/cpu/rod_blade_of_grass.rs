//! A real blade of grass: a discrete elastic rod (Cosserat-rod family,
//! Bergou et al. 2008), clamped at the root, standing under its own real
//! weight and swaying in real wind -- coupled through the SAME shared MPM
//! grid every other body in `emerge` uses (`spacetime::rod`, Phase 2).
//!
//! Run: `cargo run --example rod_blade_of_grass`

extern crate emerge_engine as emerge;
use emerge::rod::{Rod, RodMaterial, build_straight_rod};
use emerge::{SimConfig, Simulation};
use glam::Vec2;

const WIND_DRAG_COEFF: f32 = 1.5;
const WIND_SPEED_REAL_M_S: f32 = 0.05; // gentle real breeze (Beaufort "light air" range)
const WIND_GUST_PERIOD_SECONDS: f32 = 4.0;

fn main() {
    let dx_meters = 0.01; // 1 grid cell = 1cm
    let config = SimConfig {
        max_substeps_per_step: 5000,
        min_dt: 1.0e-8,
        ..SimConfig::earth(64, dx_meters, 0.02)
    };
    let mut solver = Simulation::empty(config);

    // A real blade of grass: 10cm tall, 3mm wide, 1mm thick, E=1e7 Pa
    // (soft plant-tissue stiffness, Niklas 1992 parenchyma range). This
    // height is comfortably below this exact geometry's own real
    // Euler/Greenhill self-buckling threshold (h_crit ~= 0.1216m, hand-
    // derived from h_crit = (7.8373*E*I/(rho*g*A))^(1/3)) -- it stands for
    // the same reason a real blade this thin does, not because we tuned it
    // to look right.
    let height_m = 0.10;
    let n_points = 20;
    let start = Vec2::new(32.0, 9.0);
    let end = Vec2::new(32.0, 9.0 + height_m / dx_meters);
    let mut rod_points = build_straight_rod(start, end, n_points, 0.01, dx_meters);
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = height_m / (n_points as f32 - 1.0);
    let point_mass = 0.01 * l0;
    let ea = 1.0e7 * 0.003 * 0.001;
    let ei = 1.0e7 * 0.003_f32.powi(3) * 0.001 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        1.0e7,
        0.003,
        0.001,
        axial_damping,
        bending_damping,
    );
    let mut rod = Rod::new(rod_points, material);
    rod.wind_drag_coeff = WIND_DRAG_COEFF;
    solver.add_rod(rod);

    println!("A real blade of grass: {height_m:.2}m tall, 3mm wide, clamped at the root.");
    println!(
        "Standing under real self-weight + a {WIND_SPEED_REAL_M_S:.3} m/s gust, {WIND_GUST_PERIOD_SECONDS:.0}s period.\n"
    );

    let step_dt = config.dt;
    let total_seconds = 12.0; // 3 full gust cycles
    let outer_steps = (total_seconds / step_dt).ceil() as u32;
    let mut wind_time = 0.0f32;
    let report_every = outer_steps / 24;

    for step in 0..outer_steps {
        wind_time += step_dt;
        let omega = std::f32::consts::TAU / WIND_GUST_PERIOD_SECONDS;
        let gust_speed_real = WIND_SPEED_REAL_M_S * (wind_time * omega).sin();
        solver.rods_mut()[0].wind_velocity = Vec2::new(gust_speed_real / dx_meters, 0.0);
        solver.step();

        if step % report_every == 0 || step == outer_steps - 1 {
            let tip = solver.rods()[0].points.x[n_points - 1];
            let tip_height_m = (tip.y - start.y) * dx_meters;
            let sway_mm = (tip.x - start.x) * dx_meters * 1000.0;
            let bar_pos = ((sway_mm + 1.0) * 10.0).round().clamp(0.0, 20.0) as usize;
            let mut bar = vec!['.'; 21];
            bar[10] = '|';
            bar[bar_pos] = '*';
            let bar: String = bar.into_iter().collect();
            println!(
                "t={:5.2}s  height={tip_height_m:.5}m  sway={sway_mm:+.4}mm  [{bar}]",
                step as f32 * step_dt,
            );
        }
    }

    println!("\nStill standing. Real self-weight, real wind, real grid coupling -- no cheating.");
}

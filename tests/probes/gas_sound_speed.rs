//! How fast does sound travel through the engine's air, in a running
//! simulation?
//!
//! `IdealGasMaterial` is isentropic, `p = p0 (rho / rho0)^gamma`, so its
//! continuum sound speed is `sqrt(gamma R T)`: 342.24 m/s for dry air
//! (R = 287.05 J/(kg K), gamma = 7/5) at 18.3 C. This fills a box of that
//! air wall to wall, gives it a small Gaussian x-velocity pulse (Mach
//! 1e-3), and times the right-going half between two fixed stations inside
//! the box: the lag that maximises the cross-correlation of the two
//! stations' column-averaged velocity, refined by a parabola through the
//! peak. The recording stops before anything reflected off the far wall
//! can come back to the second station.
//!
//! Wall to wall in both directions on purpose: with a free top, the layer
//! is a waveguide (gauge pressure makes the free surface a pressure-release
//! boundary over a rigid floor) whose cutoff keeps the pulse's long
//! wavelengths at the source. The box is sealed (`SlipBoundary::sealed`):
//! its air has no ambient air behind the walls to give way to.
//!
//! Measured at 18.3 C, 256 cells, earth gravity, stations 69 to 80 cells
//! apart:
//!
//! ```text
//!   dx       pulse half-width   lattice     c (m/s)   against sqrt(gamma R T)
//!   1 cm     8 cells            regular     341.47    -0.22 %
//!   0.5 cm   8 cells            regular     341.53    -0.21 %
//!   1 cm     8 cells            jitter 0.3  339.10    -0.92 %
//!   1 cm     4 cells            regular     340.25    -0.58 %
//!   1 cm     4 cells            jitter 0.3  338.57    -1.07 %
//! ```
//!
//! The narrower pulse and the jittered lattice read slower: that is the
//! method's numerical dispersion, not the gas law (the same 8-cell pulse
//! at 20 C reads -0.23 % again, against 343.23 m/s).
//!
//! Without gravity the first row reads 341.45 m/s in the sealed box, so
//! gravity does not move it. With one-sided walls (`SlipBoundary::new`,
//! `GAS_WALL=open`) it reads 341.86 with gravity and 341.63 without: the
//! closer value those walls gave depended on gravity, so it was not a
//! better measurement.
//!
//! The same air measured outdoors: Berg and Courtney, "Echo-based
//! measurement of the speed of sound" (arXiv:1102.2664), balloon pops
//! echoed off a wall 45.72 m away, 344.41(24) m/s at 18.3 C with a dew
//! point of 16.1 C at 360 m elevation. Their dry-air ideal-gas value is
//! 342.2 m/s and their humidity- and altitude-adjusted prediction 343.4 m/s:
//! the engine's air is dry, so most of the 0.9 % between 341.5 and 344.4 is
//! the humidity it does not model.
//!
//!   cargo test --test probes gas_sound_speed:: -- --ignored --nocapture
//!
//! Knobs: GAS_DX (m, 0.01), GAS_PULSE_CELLS (8), GAS_JITTER (0),
//! GAS_T_K (291.45), GAS_GRID (256), GAS_GRAVITY_G (1, in units of earth
//! gravity), GAS_WALL (`sealed`, or `open` for one-sided walls).
//!
//! `a_rarefaction_reflects_off_a_sealed_wall_as_a_rarefaction` below times
//! the other half of the same kind of pulse against a wall.

use emerge::{IdealGasMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Mat2, Vec2};

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn sound_crosses_the_engines_air_at_the_ideal_gas_speed() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let dx = env("GAS_DX", 0.01);
    let width_cells = env("GAS_PULSE_CELLS", 8.0);
    let jitter = env("GAS_JITTER", 0.0);
    let t_k = env("GAS_T_K", 291.45);
    let grid = env("GAS_GRID", 256.0) as usize;
    let gravity_g = env("GAS_GRAVITY_G", 1.0);
    let sealed = std::env::var("GAS_WALL").as_deref() != Ok("open");
    const SPACING: f32 = 0.5;
    // Air at one standard atmosphere and this temperature: rho = p / (R T).
    let rho = 101_325.0 / (287.05 * t_k);
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 100_000,
        min_dt: 1.0e-7,
        ..SimConfig::earth(grid, dx, 2.0e-5)
    };
    let config = SimConfig {
        gravity: config.gravity * gravity_g,
        ..config
    };
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::splat(grid as i32 - 6),
        box_center: Vec2::splat(grid as f32 * 0.5),
        material_id: 0,
        mass_override: Some(rho * (SPACING * dx).powi(2)),
        initial_velocity_scale: 0.0,
        position_jitter: jitter,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(IdealGasMaterial::air(rho, t_k, &config)))
        .with_boundary(Box::new(if sealed {
            SlipBoundary::sealed(config.boundary_thickness)
        } else {
            SlipBoundary::new(config.boundary_thickness)
        }));

    let c_law = (1.4f32 * 287.05 * t_k).sqrt();
    let x0 = (grid as f32 * 0.47).round();
    let amplitude = 1.0e-3 * c_law / dx; // Mach 1e-3, in cells per second
    {
        let p = sim.particles_mut();
        for i in 0..p.len() {
            let d = (p.x[i].x - x0) / width_cells;
            let g = (-d * d).exp();
            p.v[i] = Vec2::new(amplitude * g, 0.0);
            // The APIC affine field of the same pulse, du/dx.
            let dudx = amplitude * g * (-2.0 * d / width_cells);
            p.velocity_gradient[i] = Mat2::from_cols(Vec2::new(dudx, 0.0), Vec2::ZERO);
        }
    }

    let wall = grid as f32 - 3.0;
    let station_a = (x0 + 3.0 * width_cells) as usize;
    let station_b = station_a
        + 80.min((wall as usize).saturating_sub(station_a + 4 * width_cells as usize + 8));
    // Long enough for the pulse to clear station B, and over before its
    // reflection off the far wall can come back to B.
    let seconds = |cells: f32| f64::from(cells * dx / c_law);
    let t_clear = 1.3 * seconds(station_b as f32 - x0 + 6.0 * width_cells);
    let t_reflect = seconds(2.0 * wall - station_b as f32 - x0 - 3.0 * width_cells);
    let t_end = t_clear.min(t_reflect);

    let column_mean = |sim: &Simulation, column: usize| -> f64 {
        let p = sim.particles();
        let (mut sum, mut n) = (0.0f64, 0usize);
        for i in 0..p.len() {
            let c = p.x[i].x as usize;
            if c + 1 >= column && c <= column + 1 {
                sum += f64::from(p.v[i].x);
                n += 1;
            }
        }
        if n > 0 { sum / n as f64 } else { 0.0 }
    };
    let (mut series_a, mut series_b) = (Vec::new(), Vec::new());
    let mut t = 0.0f64;
    while t <= t_end {
        sim.step();
        t += f64::from(config.dt);
        series_a.push(column_mean(&sim, station_a));
        series_b.push(column_mean(&sim, station_b));
    }

    let len = series_a.len();
    let xcorr = |lag: usize| -> f64 {
        (0..len - lag)
            .map(|k| series_a[k] * series_b[k + lag])
            .sum()
    };
    let best = (1..len - 2)
        .max_by(|&a, &b| xcorr(a).total_cmp(&xcorr(b)))
        .expect("a recording longer than three frames");
    let (below, at, above) = (xcorr(best - 1), xcorr(best), xcorr(best + 1));
    let refined = best as f64 + 0.5 * (below - above) / (below - 2.0 * at + above);
    let travel_s = refined * f64::from(config.dt);
    let c = f64::from((station_b - station_a) as f32 * dx) / travel_s;
    println!(
        "dx {dx} m, pulse half-width {width_cells} cells, jitter {jitter}, T {t_k} K, {} particles",
        sim.particles().len()
    );
    println!(
        "stations {station_a} and {station_b}: travel {:.4} ms, c {c:.2} m/s against sqrt(gamma R T) {c_law:.2} m/s ({:+.2} %)",
        travel_s * 1e3,
        (c / f64::from(c_law) - 1.0) * 100.0
    );
}

/// Does a rarefaction come back off the box wall as a rarefaction?
///
/// A Gaussian x-velocity pulse with no pressure perturbation splits into a
/// right-going compression and a left-going rarefaction. Off a rigid closed
/// wall the rarefaction reflects with its pressure sign kept (reflection
/// coefficient +1, the fluid velocity reversed); off a pressure-release
/// surface the sign flips (-1). The column-mean density perturbation is
/// read at a station 19 cells from the left wall, 18 cells from the pulse,
/// so the reflection reaches it after a 56-cell path. The reference run
/// (`free`) times the same rarefaction over the same 56 cells with no wall
/// in the way, so the coefficient leaves out what the travel itself costs.
/// No gravity, so the only motion is the pulse.
///
/// Measured, 128-cell sealed box of air at 18.3 C, 1 cm cells, 6-cell pulse
/// half-width, Mach 1e-3:
///
/// ```text
///   GAS_WALL    walls                     rho'/rho0 at the station
///   sealed      SlipBoundary::sealed      -3.00e-4 (reflected)
///   open        SlipBoundary::new         +2.44e-4 (reflected)
///   free        sealed, no wall on path   -3.23e-4 (after 56 cells)
///   free-open   open, no wall on path     -2.92e-4 (after 56 cells)
/// ```
///
/// Each wall against the reference with the same side walls: the sealed
/// wall reflects with +0.93 (-3.00 / -3.23), the open one with -0.84
/// (+2.44 / -2.92).
///
/// The one-sided wall lets gas below ambient pressure pull off it, so it
/// reflects like a free surface; it also costs a wave running along the top
/// and bottom walls 10 % over 56 cells, the rarefaction leaving them as it
/// passes.
///
///   GAS_WALL=sealed cargo test --test probes gas_sound_speed::a_rarefaction -- --ignored --nocapture
///
/// Knobs: GAS_WALL (`sealed`, `open`, `free` for the reference with sealed
/// side walls, `free-open`).
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn a_rarefaction_reflects_off_a_sealed_wall_as_a_rarefaction() {
    let wall_kind = std::env::var("GAS_WALL").unwrap_or_else(|_| "sealed".into());
    let reference = wall_kind.starts_with("free");
    let sealed = wall_kind == "sealed" || wall_kind == "free";
    const DX: f32 = 0.01;
    const T_K: f32 = 291.45;
    const WIDTH_CELLS: f32 = 6.0;
    const SPACING: f32 = 0.5;
    const MACH: f32 = 1.0e-3;
    // Pulse at 40 and station at 22 in a 128-cell box with walls at 3: a
    // 56-cell path to the wall and back. The reference box is wide enough
    // for the same 56-cell path with nothing on it.
    let (grid, x0, station) = if reference {
        (256usize, 140.0f32, 84usize)
    } else {
        (128usize, 40.0f32, 22usize)
    };
    let rho = 101_325.0 / (287.05 * T_K);
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 100_000,
        min_dt: 1.0e-7,
        gravity: Vec2::ZERO,
        ..SimConfig::earth(grid, DX, 2.0e-5)
    };
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::splat(grid as i32 - 6),
        box_center: Vec2::splat(grid as f32 * 0.5),
        material_id: 0,
        mass_override: Some(rho * (SPACING * DX).powi(2)),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let wall = if sealed {
        SlipBoundary::sealed(config.boundary_thickness)
    } else {
        SlipBoundary::new(config.boundary_thickness)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(IdealGasMaterial::air(rho, T_K, &config)))
        .with_boundary(Box::new(wall));

    let c_law = (1.4f32 * 287.05 * T_K).sqrt();
    let amplitude = MACH * c_law / DX;
    {
        let p = sim.particles_mut();
        for i in 0..p.len() {
            let d = (p.x[i].x - x0) / WIDTH_CELLS;
            let g = (-d * d).exp();
            p.v[i] = Vec2::new(amplitude * g, 0.0);
            let dudx = amplitude * g * (-2.0 * d / WIDTH_CELLS);
            p.velocity_gradient[i] = Mat2::from_cols(Vec2::new(dudx, 0.0), Vec2::ZERO);
        }
    }
    let cells_per_s = c_law / DX;
    let left = config.boundary_thickness as f32;
    let path = if reference {
        x0 - station as f32
    } else {
        (x0 - left) + (station as f32 - left)
    };
    let t_arrive = f64::from(path / cells_per_s);
    let window = f64::from(2.0 * WIDTH_CELLS / cells_per_s);
    let t_end = t_arrive + 2.0 * window;

    let column_density = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut sum, mut n) = (0.0f64, 0usize);
        for i in 0..p.len() {
            let c = p.x[i].x as usize;
            if c + 1 >= station && c <= station + 1 {
                sum += f64::from(p.initial_volume[i] / p.volume[i]) - 1.0;
                n += 1;
            }
        }
        sum / n as f64
    };
    // The largest density perturbation inside the arrival window.
    let (mut t, mut peak) = (0.0f64, 0.0f64);
    while t <= t_end {
        sim.step();
        t += f64::from(config.dt);
        if (t - t_arrive).abs() < window {
            let r = column_density(&sim);
            if r.abs() > peak.abs() {
                peak = r;
            }
        }
    }
    println!(
        "{wall_kind}: rho'/rho0 {peak:+.3e} at the station after a {path}-cell path ({:.4} ms)",
        t_arrive * 1e3
    );
}

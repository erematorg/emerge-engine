//! Elastic waves travel at their speed whatever the material's density: a
//! solid built from SI properties (`from_physical`, `mass_from`) must carry
//! its stiffness and its mass in the same units on both paths, the
//! explicit one and the total Lagrangian one.
//!
//! The scene: a bar 2 cells thick, its lateral faces free, whose left half
//! starts at +V and right half at rest (a Riemann problem); the
//! compressive front reaches a probe 10 cells right of the middle at
//! `0.1 m / c`, timed where the probe's mean velocity crosses V/4.
//!
//! The reference speed, derived before measuring: a plane-strain bar with
//! free lateral faces (`sigma_yy = 0`, `eps_zz = 0`) carries a long
//! longitudinal wave at `c_bar = sqrt(E / ((1 - nu^2) rho))`, not the P
//! wave's `c_P = sqrt((lambda + 2 mu) / rho)`. Derived from isotropic
//! Hooke's law: `eps_zz = 0` gives `sigma_zz = nu (sigma_xx + sigma_yy)`,
//! then `sigma_yy = 0` gives `eps_xx = (1 - nu^2) sigma_xx / E`, so the
//! axial modulus is `E / (1 - nu^2)`. `c_bar / c_P = sqrt((1 - 2 nu) /
//! (1 - nu)^2)`, 0.9035 at `nu = 0.3`. At `nu = 0` the two coincide and the
//! plane wave `u_x(x - c t)` meets the free faces exactly (`sigma_yy =
//! lambda eps_xx = 0`), at every wavelength: the clean check. At `nu =
//! 0.3` the bar speed is the long-wave limit only; the front's shorter
//! wavelengths (the bar is 2 cm thick, the probe 10 cm away) travel
//! slower, a dispersion inside the tolerance (measured below).
//!
//! Tolerance: 5 percent on `c / c_bar`, 2 percent on the `1/sqrt(rho)`
//! scaling, both set before the density fix and unchanged since. Measured
//! before the fix: 1700 kg/m^3 ran at 0.77 of the speed, wave speeds
//! scaling as `1/rho` instead of `1/sqrt(rho)`.
//!
//! Frames of 2.5e-5 s: the front arrives after about 8e-4 s (1000 kg/m^3)
//! to 1.4e-3 s (2700), 32 to 56 frames, the crossing's linear
//! interpolation inside one frame then a small part of the arrival time
//! (the total Lagrangian path's step is capped at the frame).

extern crate emerge_engine as emerge;

use emerge::materials::physical_props::{Elastic, Fluid};
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const DX_M: f32 = 0.01;
const FRAME_S: f32 = 2.5e-5;
const PROBE_CELLS: f32 = 10.0;

fn solid(rho: f32, nu: f32) -> Elastic {
    Elastic {
        e_pa: 1.125e7,
        nu,
        rho_kg_m3: rho,
    }
}

/// The plane-strain bar speed of `props`, `sqrt(E / ((1 - nu^2) rho))`,
/// m/s (module doc).
fn bar_speed(props: &Elastic) -> f64 {
    let (e, nu, rho) = (
        f64::from(props.e_pa),
        f64::from(props.nu),
        f64::from(props.rho_kg_m3),
    );
    (e / ((1.0 - nu * nu) * rho)).sqrt()
}

/// The front's arrival time at the probe, s.
fn arrival(total_lagrangian: bool, props: &Elastic) -> f64 {
    let mut config = SimConfig::earth(64, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    if total_lagrangian {
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    }
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(props, &config)))
        .0;
    let middle = 32.0f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(48, 2))
            .at(Vec2::new(middle, 32.0))
            .spacing(0.5)
            .material(id)
            .mass_from(props, &config),
    );
    // 1 cm/s, in cells/s.
    let speed = 0.01 / DX_M;
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        if x.x < middle {
            *v = Vec2::new(speed, 0.0);
        }
    }
    let probe = middle + PROBE_CELLS;
    let probe_speed = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut sum, mut count) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            if (p.x[i].x - probe).abs() < 0.5 && p.mass[i] > 0.0 {
                sum += f64::from(p.v[i].x);
                count += 1.0;
            }
        }
        sum / count
    };
    let threshold = 0.25 * f64::from(speed);
    let mut previous = probe_speed(&sim);
    for frame in 1..=240 {
        sim.step();
        let now = probe_speed(&sim);
        if now >= threshold {
            let fraction = (threshold - previous) / (now - previous);
            return (f64::from(frame) - 1.0 + fraction) * f64::from(FRAME_S);
        }
        previous = now;
    }
    panic!("the front never reached the probe in 240 frames");
}

/// Each density's front speed against `c_bar` (within 5 percent), and
/// against the 1000 kg/m^3 front scaled by `sqrt(1000 / rho)` (within 2
/// percent: the front's own detection error is common to every density
/// and cancels there; the 2 percent, stated before the fix, allows its
/// variation with the step count), at `nu = 0.3` and at `nu = 0`.
fn check(total_lagrangian: bool) {
    let mut worst = 0.0f64;
    let mut worst_scaling = 0.0f64;
    for nu in [0.3f32, 0.0] {
        let mut reference = None;
        for rho in [1000.0f32, 1700.0, 2700.0] {
            let props = solid(rho, nu);
            let measured = f64::from(PROBE_CELLS * DX_M) / arrival(total_lagrangian, &props);
            let ratio = measured / bar_speed(&props);
            let at_1000 = *reference.get_or_insert(measured);
            let scaling = measured / (at_1000 * (1000.0 / f64::from(rho)).sqrt());
            println!(
                "total Lagrangian {total_lagrangian}, nu {nu}, rho {rho} kg/m^3: front {measured:.2} m/s, c_bar {:.2} m/s, ratio {ratio:.4}, against the 1000 kg/m^3 front scaled by 1/sqrt(rho) {scaling:.4}",
                bar_speed(&props)
            );
            worst = worst.max((ratio - 1.0).abs());
            worst_scaling = worst_scaling.max((scaling - 1.0).abs());
        }
    }
    assert!(worst <= 0.05, "wave speed off by {worst:.4} of c_bar");
    assert!(
        worst_scaling <= 0.02,
        "wave speed off its 1/sqrt(rho) scaling by {worst_scaling:.4}"
    );
}

#[test]
fn elastic_waves_travel_at_their_speed_on_the_explicit_path() {
    check(false);
}

#[test]
fn elastic_waves_travel_at_their_speed_on_the_total_lagrangian_path() {
    check(true);
}

/// A fluid's sound speed, `sqrt(K / rho)` (the Tait equation's slope at
/// rest), the same contract for a liquid: its pressure and its mass in the
/// same units at any density. The column fills the domain from floor to
/// ceiling: a free-surface strip carries no one-dimensional compression
/// wave, its pressure leaking out through the free surfaces (measured: the
/// probe never reached V/4). Measured before the density fix: 0.78 and
/// 0.62 of `sqrt(K/rho)` at 1700 and 2700 kg/m^3.
fn fluid_arrival(rho: f32) -> f64 {
    let props = Fluid {
        rho_kg_m3: rho,
        eta_pa_s: 1e-3,
        bulk_modulus_pa: 1e6,
        yield_stress_pa: None,
    };
    let mut config = SimConfig::earth(64, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(props.material(&config)).0;
    let middle = 32.0f32;
    let _column = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(48, 60))
            .at(Vec2::new(middle, 32.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let speed = 0.01 / DX_M;
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        if x.x < middle {
            *v = Vec2::new(speed, 0.0);
        }
    }
    let probe = middle + PROBE_CELLS;
    let probe_speed = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut sum, mut count) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            if (p.x[i].x - probe).abs() < 0.5 && (p.x[i].y - 32.0).abs() < 10.0 {
                sum += f64::from(p.v[i].x);
                count += 1.0;
            }
        }
        sum / count
    };
    let threshold = 0.25 * f64::from(speed);
    let mut previous = probe_speed(&sim);
    for frame in 1..=400 {
        sim.step();
        let now = probe_speed(&sim);
        if now >= threshold {
            let fraction = (threshold - previous) / (now - previous);
            return (f64::from(frame) - 1.0 + fraction) * f64::from(FRAME_S);
        }
        previous = now;
    }
    panic!("the front never reached the probe in 400 frames");
}

#[test]
fn sound_travels_at_its_speed_in_a_fluid() {
    let (mut worst, mut worst_scaling) = (0.0f64, 0.0f64);
    let mut reference = None;
    for rho in [1000.0f32, 1700.0, 2700.0] {
        let measured = f64::from(PROBE_CELLS * DX_M) / fluid_arrival(rho);
        let exact = (1e6 / f64::from(rho)).sqrt();
        let at_1000 = *reference.get_or_insert(measured);
        let scaling = measured / (at_1000 * (1000.0 / f64::from(rho)).sqrt());
        println!(
            "fluid, rho {rho} kg/m^3: front {measured:.3} m/s, sqrt(K/rho) {exact:.3} m/s, ratio {:.4}, scaling {scaling:.4}",
            measured / exact
        );
        worst = worst.max((measured / exact - 1.0).abs());
        worst_scaling = worst_scaling.max((scaling - 1.0).abs());
    }
    assert!(worst <= 0.05, "sound speed off by {worst:.4}");
    assert!(
        worst_scaling <= 0.02,
        "sound speed off its 1/sqrt(rho) scaling by {worst_scaling:.4}"
    );
}

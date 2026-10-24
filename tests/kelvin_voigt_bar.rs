//! A Kelvin-Voigt solid on the total Lagrangian path, measured against
//! exact answers in SI.
//!
//! The law: the corotated elastic solid with the Newtonian dashpot in
//! parallel, `tau_v = J (2 eta dev(D) + zeta tr(D) I)` (plane strain from
//! 3D, `materials::viscous_stress`). On the step: `S_D = 2 eta dev(Z/dt) +
//! zeta tr(Z/dt) I`, whose work over a step is the dissipation, turned into
//! heat.
//!
//! Criteria fixed before measuring:
//! (a) every frame's energy (kinetic, stored, gravity's) plus the heat made
//!     equals the start's to round-off (the dissipation leaves the solids'
//!     energy exactly as heat), and the dissipation is positive;
//! (b) a free-free bar's first axial mode decays at the exact rate: with
//!     `nu = 0` and `zeta = 2 eta / 3` the state is exactly uniaxial
//!     (`lambda = 0`, the dashpot's lateral stress zero), `sigma_xx = E eps +
//!     2 eta eps_dot`, so the modal damping ratio is `eta omega / E` at the
//!     mode's own angular frequency: measured within 5 percent;
//! (c) with no viscosity the path is unchanged (the other wired tests);
//! and the mode's frequency is the bar's own, `pi c / L`, within 5 percent.

extern crate emerge_engine as emerge;

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const DX_M: f32 = 0.01;
const E_PA: f32 = 1.0e5;
const RHO: f32 = 1000.0;
/// Shear viscosity, Pa s: a damping ratio near 0.05 on the first mode.
const ETA_PA_S: f32 = 50.0;
const LENGTH_CELLS: i32 = 32;
const FRAME_S: f32 = 2.0e-3;

#[test]
fn a_kelvin_voigt_bar_rings_down_at_the_exact_rate_and_its_energy_becomes_heat() {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    let eta = if std::env::var_os("EMERGE_KV_NO_VISCOSITY").is_some() {
        0.0
    } else {
        ETA_PA_S
    };
    material.elastic_viscosity = config.visc_from_si(eta);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * eta / 3.0);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let x0 = 8.0f32;
    let l = LENGTH_CELLS as f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(x0 + 0.5 * l, 24.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    if std::env::var_os("EMERGE_KV_TRACE").is_some() {
        let p = sim.particles();
        let mean: f64 =
            p.initial_volume.iter().map(|&v| f64::from(v)).sum::<f64>() / p.len() as f64;
        let (lo, hi) = p
            .initial_volume
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        println!("initial volume: mean {mean:.4} cells^2 (lattice 0.25), min {lo:.4}, max {hi:.4}");
    }
    // The first free-free mode's velocity shape, 1 cm/s (a strain near
    // 1e-3). `EMERGE_KV_SPEED` sets another, m/s.
    let amplitude_m_s: f32 = std::env::var("EMERGE_KV_SPEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = Vec2::new(
            (amplitude_m_s / DX_M) * (std::f32::consts::PI * (x.x - x0) / l).cos(),
            0.0,
        );
    }
    let amplitude = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let shape = f64::from((std::f32::consts::PI * (p.x[i].x - x0) / l).cos());
            num += f64::from(p.mass[i]) * f64::from(p.v[i].x) * shape;
            den += f64::from(p.mass[i]) * shape * shape;
        }
        num / den
    };
    // The modal amplitude every frame; its peaks (parabola through the
    // three samples around each) give the damped period and the
    // logarithmic decrement.
    let mut samples = vec![amplitude(&sim)];
    let mut worst_balance = 0.0f64;
    let mut dissipated = 0.0f64;
    // With `EMERGE_REPLAY_DIR`, a replay of the run (positions m, speeds
    // m/s, energy, heat and balance J/m per frame) written there, for
    // viewing, no check.
    let mut replay: Vec<String> = Vec::new();
    // Three periods of the first mode (64 ms each at 2 ms frames).
    for index in 0..100 {
        let clock = std::time::Instant::now();
        sim.step();
        let wall = clock.elapsed().as_secs_f64();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        assert!(frame.energy.is_finite());
        worst_balance = worst_balance.max(frame.balance.abs() / frame.start_energy.abs());
        dissipated += frame.dissipated;
        samples.push(amplitude(&sim));
        if std::env::var_os("EMERGE_REPLAY_DIR").is_some() {
            let dx = f64::from(sim.config().dx_meters);
            let p = sim.particles();
            let positions: Vec<String> =
                p.x.iter()
                    .map(|q| format!("[{:.6},{:.6}]", f64::from(q.x) * dx, f64::from(q.y) * dx))
                    .collect();
            let speeds: Vec<String> =
                p.v.iter()
                    .map(|v| format!("{:.6}", f64::from(v.length()) * dx))
                    .collect();
            replay.push(format!(
                "{{\"time\":{:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"energy\":{:.9e},\"heat\":{:.9e},\"balance\":{:.3e}}}",
                f64::from(sim.config().dt) * (index + 1) as f64,
                positions.join(","),
                speeds.join(","),
                frame.energy,
                frame.heat,
                frame.balance
            ));
        }
        if std::env::var_os("EMERGE_KV_TRACE").is_some() {
            println!(
                "frame ({wall:.2} s): stagnated {} steps {} rejected {} (unconverged {}) newton {} gmres {} at floor {} balance {:+.2e} dissipated {:.3e}",
                frame.stagnated,
                frame.steps,
                frame.rejected,
                frame.rejected_unconverged,
                frame.newton,
                frame.gmres,
                frame.at_floor,
                frame.balance,
                frame.dissipated
            );
        }
    }
    if let Some(dir) = std::env::var_os("EMERGE_REPLAY_DIR") {
        let header = format!(
            "{{\"scene\":\"kelvin_voigt_bar\",\"dx_m\":{},\"frame_s\":{},\"eta_pa_s\":{eta},\"path\":\"total Lagrangian, Kelvin-Voigt dashpot\",\"precomputed\":true}}",
            sim.config().dx_meters,
            sim.config().dt
        );
        let path = std::path::Path::new(&dir).join("kelvin_voigt_bar.json");
        std::fs::write(
            &path,
            format!(
                "{{\"header\":{header},\"frames\":[
{}
]}}
",
                replay.join(
                    ",
"
                )
            ),
        )
        .expect("the replay directory is writable");
        println!("replay written: {}", path.display());
    }
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 1..samples.len() - 1 {
        let (a, b, c) = (samples[k - 1].abs(), samples[k].abs(), samples[k + 1].abs());
        if b >= a && b > c {
            let curvature = a - 2.0 * b + c;
            let shift = if curvature != 0.0 {
                0.5 * (a - c) / curvature
            } else {
                0.0
            };
            let height = b - 0.25 * (a - c) * shift;
            peaks.push(((k as f64 + shift) * f64::from(FRAME_S), height));
        }
    }
    assert!(peaks.len() >= 5, "too few peaks: {peaks:?}");
    // |v| peaks every half period: over 4 of them, two periods.
    let (t0, a0) = peaks[0];
    let (t4, a4) = peaks[4];
    let period = (t4 - t0) / 2.0;
    let omega = 2.0 * std::f64::consts::PI / period;
    let decrement = (a0 / a4).ln() / 2.0;
    let zeta = decrement
        / (4.0 * std::f64::consts::PI * std::f64::consts::PI + decrement * decrement).sqrt();
    let exact = f64::from(ETA_PA_S) * omega / f64::from(E_PA);
    let omega_bar =
        std::f64::consts::PI * (f64::from(E_PA) / f64::from(RHO)).sqrt() / f64::from(l * DX_M);
    println!(
        "Kelvin-Voigt bar: damped period {period:.5} s (omega {omega:.2} rad/s, the bar's pi c / L {omega_bar:.2}), damping ratio {zeta:.5}, exact eta omega / E {exact:.5}, ratio {:.4}; worst balance {worst_balance:.2e} of the start's energy; dissipated {dissipated:.4e} J/m, heat {:.4e} J/m",
        zeta / exact,
        frame.heat
    );
    assert!(worst_balance < 1e-8, "energy balance {worst_balance:e}");
    assert!(dissipated > 0.0, "no dissipation");
    assert!(
        (zeta / exact - 1.0).abs() < 0.05,
        "damping ratio {zeta} against {exact}"
    );
    // And the mode's own frequency against the bar's, `pi c / L` with `c =
    // sqrt(E / rho)` (uniaxial strain, `nu = 0`), within the same 5 percent:
    // the reference area `V0 = m / rho` (issue #41) puts it there, the
    // packing estimate had it 18 percent high.
    assert!(
        (omega / omega_bar - 1.0).abs() < 0.05,
        "first mode {omega} rad/s against pi c / L {omega_bar}"
    );
}

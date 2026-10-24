//! Liquid water on the path (`LiquidWaterMaterial`, the barotropic law):
//! its scenes against exact solutions and the probes that measure them.
//!
//! - The lowest pressure mode of a free square block, `omega = c pi
//!   sqrt(2) / L`, `c` the record's speed of sound (`probe_acoustic_
//!   frequencies`). Measured: 0.990637 of it at 4 cm cells, 0.996875 at 2
//!   cm, 0.999019 at 1 cm, 1.000141 extrapolated (observed order 1.54).
//! - The hydrostatic column in a tank filled wall to wall, started at the
//!   exact nonlinear equilibrium (`probe_water_column_frames`). Measured:
//!   it does not hold. The walls hold one mean gap per patch of two cells
//!   (`total_lagrangian::weighted_gap`); inside a patch each side face
//!   tilts with zero mean (-1.83 to +1.80 mm after one frame, mean -1.9e-6
//!   m at 4 cm cells), a motion a solid pays for in shear energy and a
//!   liquid does not, and gravity drives the column through the walls
//!   (26 mm beyond them after five frames), the energy ledger closed. Open:
//!   a wall discretisation that holds the pointwise condition for a body
//!   with no shear stiffness.
//!
//! The hydrostatic start, derived: in reference coordinates (`Y` from the
//! floor, `H0` the depth), `F = diag(1, a(Y))` gives `P_yy = K (a - 1)` and
//! `P_xx = K a (a - 1)`, a function of `Y` alone. Equilibrium `dP_yy/dY =
//! rho0 g` with the free top `P_yy(H0) = 0` gives `a = 1 - rho0 g (H0 - Y)
//! / K` and the pressure `p = K (1 - J) = rho0 g (H0 - Y)` exactly (not
//! linearised); the side walls push with `-P_xx > 0` and slip needs no
//! shear. The positions `y = Y - rho0 g (H0 Y - Y^2 / 2) / K`. Set on the
//! path's `f64` state: the strain, about 1e-6, is below the particles'
//! `f32` resolution of `F`.

use super::*;
use crate::spacetime::total_lagrangian::ElasticLaw;
use crate::{LiquidWaterMaterial, SimConfig};
use glam::IVec2;

/// 20 C, K.
const T_WATER: f64 = 293.15;
/// The tank's inner width and the water's depth, m.
const WIDTH_M: f64 = 0.32;
const DEPTH_M: f64 = 0.12;
/// The free block's side, m.
const BLOCK_M: f64 = 0.12;

/// Water at `T_WATER` filling the tank wall to wall at cell `dx`, frames
/// of `frame_s`; the floor's height, m.
fn tank(dx: f32, frame_s: f32) -> (Simulation, f64) {
    let cells = |m: f64| (m / f64::from(dx)).round() as i32;
    let mut config = SimConfig::earth(cells(WIDTH_M) as usize + 4, dx, frame_s);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let water = LiquidWaterMaterial::new(T_WATER, &config).expect("in range");
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(water)).0;
    let wall = config.boundary_thickness as f32;
    let (w, d) = (cells(WIDTH_M), cells(DEPTH_M));
    let _water = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(w, d))
            .at(Vec2::new(wall + 0.5 * w as f32, wall + 0.5 * d as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&water, &config),
    );
    // The wall planes' height, as `total_lagrangian_walls` places them.
    (sim, f64::from(wall) * f64::from(dx))
}

/// A free square block of water, side `BLOCK_M`, centred, no gravity.
fn free_block(dx: f32, frame_s: f32) -> Simulation {
    let side = (BLOCK_M / f64::from(dx)).round() as i32;
    let mut config = SimConfig::earth(side as usize + 8, dx, frame_s);
    config.gravity = Vec2::ZERO;
    // The explicit path's floor on its substeps (unused here) must not
    // exceed the frame.
    config.min_dt = config.min_dt.min(frame_s);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let water = LiquidWaterMaterial::new(T_WATER, &config).expect("in range");
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(water)).0;
    let centre = 0.5 * (side + 8) as f32;
    let _water = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::splat(side))
            .at(Vec2::splat(centre))
            .spacing(0.5)
            .material(id)
            .mass_from(&water, &config),
    );
    sim
}

/// Builds the path's state, then places it: every point (particles and
/// contact points) at `place(X)` with the particles' `F`, and the
/// particles at `velocity(X)` (the contact points at the velocity field
/// that gives, as the build starts them); the start's energies again.
fn start(
    sim: &mut Simulation,
    place: &dyn Fn(DVec2) -> (DVec2, DMat2),
    velocity: &dyn Fn(DVec2) -> DVec2,
) {
    let path = sim.config.total_lagrangian.expect("the path is on");
    let mut state = sim.build_total_lagrangian_state((&path.settings, false));
    let n = state.particles;
    for i in 0..state.x.len() {
        let (x, f) = place(state.reference[i]);
        state.x[i] = x;
        if i < n {
            state.f[i] = f;
            state.v[i] = velocity(state.reference[i]);
        }
    }
    let field = start_velocity_field(
        &sim.total_lagrangian_layout(),
        &state.reference,
        (&state.mass, &state.v0, &state.v),
        &path.settings,
    );
    state.v[n..].copy_from_slice(&field[n..]);
    state.start_momentum = (0..n).fold(DVec2::ZERO, |sum, p| sum + state.mass[p] * state.v[p]);
    state.start_energy = sim.total_lagrangian_energy(&state);
    let kinetic_and_stored: f64 = (0..n)
        .map(|p| {
            0.5 * state.mass[p] * state.v[p].length_squared()
                + state.v0[p] * state.solids[p].energy(state.f[p])
        })
        .sum();
    let mass: f64 = state.mass[..n].iter().sum();
    state.tolerance_energy = tolerance_energy(
        kinetic_and_stored,
        sim.total_lagrangian_gravity(),
        [(mass, &state.reference[n..])],
    );
    sim.total_lagrangian = Some(state);
}

/// The water's `rho0 = m / V0`, `K`, `g` (m/s^2, downward positive) and
/// its outline's highest reference point (m), from a state built and
/// dropped.
fn coefficients(sim: &mut Simulation) -> (f64, f64, f64, f64) {
    let path = sim.config.total_lagrangian.expect("the path is on");
    let state = sim.build_total_lagrangian_state((&path.settings, false));
    let rho0 = state.mass[0] / state.v0[0];
    let top = state.reference[state.particles..]
        .iter()
        .map(|q| q.y)
        .fold(f64::NEG_INFINITY, f64::max);
    (
        rho0,
        state.solids[0].bulk,
        -sim.total_lagrangian_gravity().y,
        top,
    )
}

/// The exact hydrostatic placement of the module doc, the floor at
/// `floor` m, the depth `h0` m.
fn hydrostatic(
    (rho0, k, g): (f64, f64, f64),
    (floor, h0): (f64, f64),
) -> impl Fn(DVec2) -> (DVec2, DMat2) {
    move |q: DVec2| {
        let y = q.y - floor;
        let a = 1.0 - rho0 * g * (h0 - y) / k;
        let drop = rho0 * g * (h0 * y - 0.5 * y * y) / k;
        (
            DVec2::new(q.x, q.y - drop),
            DMat2::from_diagonal(DVec2::new(1.0, a)),
        )
    }
}

/// One frame stepped: its balance over its bound (`allowed` accumulates
/// the frame's solve and contact bounds).
fn stepped(sim: &mut Simulation, allowed: &mut f64) -> f64 {
    sim.step();
    let f = sim.total_lagrangian_frame().expect("the path is on");
    *allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
    let count = sim.particles().len() as f64;
    let round_off = count
        * f64::EPSILON
        * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
    if std::env::var_os("EMERGE_WATER_TRACE").is_some() {
        println!(
            "  balance {:+.3e} allowed {:.3e} round-off {:.3e} residual work {:.3e} kinetic {:.6e} stored {:.6e} steps {} rejected {}",
            f.balance,
            *allowed,
            round_off,
            f.residual_work,
            f.parts[0],
            f.parts[1],
            f.steps,
            f.rejected
        );
    }
    f.balance.abs() / (*allowed + round_off)
}

/// Richardson from the 4, 2 and 1 cm values: the observed order and the
/// value extrapolated to `dx -> 0`.
fn extrapolated([q4, q2, q1]: [f64; 3]) -> (f64, f64) {
    let order = ((q2 - q4) / (q1 - q2)).log2();
    (order, q1 + (q1 - q2) / (2f64.powf(order) - 1.0))
}

/// The projection of the particles' velocities on `psi` at their
/// reference positions, `sum m v . psi / sum m |psi|^2`, each frame, and
/// the worst balance over its bound.
fn projections(
    sim: &mut Simulation,
    psi: &dyn Fn(DVec2) -> DVec2,
    frames: usize,
) -> (Vec<f64>, f64) {
    let (mut out, mut allowed, mut worst) = (Vec::new(), 0.0, 0.0f64);
    for _ in 0..frames {
        worst = worst.max(stepped(sim, &mut allowed));
        let state = sim.total_lagrangian.as_ref().expect("on");
        let (mut along, mut size) = (0.0, 0.0);
        for p in 0..state.particles {
            let shape = psi(state.reference[p]);
            along += state.mass[p] * state.v[p].dot(shape);
            size += state.mass[p] * shape.length_squared();
        }
        out.push(along / size);
    }
    (out, worst)
}

/// The angular frequency from the zero crossings of `a` sampled every
/// `dt` from `dt` on (linear interpolation), rad/s: `pi` per half period
/// between the first and the last crossing.
fn crossing_frequency(a: &[f64], dt: f64) -> f64 {
    let mut crossings = Vec::new();
    for k in 1..a.len() {
        if a[k - 1].signum() != a[k].signum() {
            let at = k as f64 + a[k - 1] / (a[k - 1] - a[k]);
            crossings.push(at * dt);
        }
    }
    assert!(
        crossings.len() >= 2,
        "fewer than two crossings: {crossings:?}"
    );
    let span = crossings[crossings.len() - 1] - crossings[0];
    std::f64::consts::PI * (crossings.len() - 1) as f64 / span
}

/// The free block's lowest pressure mode at cell `dx`: its measured
/// frequency over `c pi sqrt(2) / L`, the worst balance, the midpoint
/// rule's derived shift `-(omega dt)^2 / 12`.
fn acoustic(dx: f32) -> (f64, f64, f64) {
    const FRAME_S: f64 = 2e-6;
    let mut sim = free_block(dx, FRAME_S as f32);
    let path = sim.config.total_lagrangian.expect("on");
    let state = sim.build_total_lagrangian_state((&path.settings, false));
    // The block's lower-left corner: its outline's lowest point.
    let corner = state.reference[state.particles..]
        .iter()
        .fold(DVec2::INFINITY, |c, &q| c.min(q));
    let (rho0, k, _, _) = coefficients(&mut sim);
    let c = (k / rho0).sqrt();
    let wavenumber = std::f64::consts::PI / BLOCK_M;
    let omega = c * wavenumber * 2f64.sqrt();
    // `phi = sin(k X) sin(k Y)`, `v = A grad phi`, the largest strain
    // `A k^2 2 / omega` = 1e-5.
    let amplitude = 1e-5 * omega / (2.0 * wavenumber * wavenumber);
    let psi = move |q: DVec2| {
        let r = q - corner;
        DVec2::new(
            (wavenumber * r.x).cos() * (wavenumber * r.y).sin(),
            (wavenumber * r.x).sin() * (wavenumber * r.y).cos(),
        ) * wavenumber
    };
    start(&mut sim, &|q| (q, DMat2::IDENTITY), &|q| amplitude * psi(q));
    let period = std::f64::consts::TAU / omega;
    let frames = (2.0 * period / FRAME_S).ceil() as usize;
    let (a, worst) = projections(&mut sim, &psi, frames);
    let measured = crossing_frequency(&a, FRAME_S);
    (measured / omega, worst, -(omega * FRAME_S).powi(2) / 12.0)
}

/// The material on the path is the barotropic law with the record's
/// isentropic modulus, nothing else.
#[test]
fn the_path_carries_water_as_the_barotropic_law_of_its_record() {
    let (sim, _) = tank(0.04, 1.0 / 60.0);
    let water = LiquidWaterMaterial::new(T_WATER, sim.config()).expect("in range");
    let law = sim.total_lagrangian_material(sim.particles.material_id[0]);
    assert!(law.law == ElasticLaw::Barotropic, "{law:?}");
    assert_eq!(
        (law.bulk, law.mu, law.shear_viscosity, law.bulk_viscosity),
        (water.record.isentropic_bulk(), 0.0, 0.0, 0.0)
    );
    assert!(!law.is_relaxing());
}

/// The 4 cm column frame by frame from the exact start: each frame's
/// steps, rejections, Newton and GMRES iterations, contact passes and
/// wall time. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_water_column_frames() {
    cap::capped("probe_water_column_frames", 900, || {
        let (mut sim, floor) = tank(0.04, 1.0 / 60.0);
        let (rho0, k, g, top) = coefficients(&mut sim);
        start(
            &mut sim,
            &hydrostatic((rho0, k, g), (floor, top - floor)),
            &|_| DVec2::ZERO,
        );
        let planes = sim.total_lagrangian_walls().0;
        // Where the state is: the particles' mean drop and their extremes
        // against the walls, the contact points' deepest crossing, the
        // fastest particle.
        let geometry = |sim: &Simulation| {
            let state = sim.total_lagrangian.as_ref().expect("on");
            let n = state.particles;
            let drop = (0..n)
                .map(|p| state.reference[p].y - state.x[p].y)
                .sum::<f64>()
                / n as f64;
            let inside = |q: DVec2| {
                planes
                    .iter()
                    .map(|plane| plane.inward * (q[plane.axis] - plane.at))
                    .fold(f64::INFINITY, f64::min)
            };
            let particles = (0..n)
                .map(|p| inside(state.x[p]))
                .fold(f64::INFINITY, f64::min);
            let points = (n..state.x.len())
                .map(|p| inside(state.x[p]))
                .fold(f64::INFINITY, f64::min);
            let fastest = (0..n)
                .max_by(|&a, &b| state.v[a].length().total_cmp(&state.v[b].length()))
                .expect("particles");
            println!(
                "  mean drop {drop:.4e} m; nearest particle to a wall {particles:.4e} m, deepest contact point {points:.4e} m; fastest particle {:.4e} m/s at reference {:?}",
                state.v[fastest].length(),
                state.reference[fastest]
            );
            // Each wall's touching outline points: their reference
            // position along the wall and their gap, and the gaps' mean.
            for plane in &planes {
                let touching: Vec<(f64, f64)> = (n..state.x.len())
                    .filter(|&p| (state.reference[p][plane.axis] - plane.at).abs() < 1e-9)
                    .map(|p| {
                        (
                            state.reference[p][1 - plane.axis],
                            plane.inward * (state.x[p][plane.axis] - plane.at),
                        )
                    })
                    .collect();
                if touching.is_empty() {
                    continue;
                }
                let mean = touching.iter().map(|t| t.1).sum::<f64>() / touching.len() as f64;
                let listed: Vec<String> = touching
                    .iter()
                    .map(|(along, gap)| format!("{along:.3}:{gap:+.2e}"))
                    .collect();
                println!(
                    "  wall axis {} at {:.3}: mean gap {mean:+.3e} m; {}",
                    plane.axis,
                    plane.at,
                    listed.join(" ")
                );
            }
        };
        geometry(&sim);
        let mut allowed = 0.0;
        for frame in 1..=2 {
            let clock = std::time::Instant::now();
            let ratio = stepped(&mut sim, &mut allowed);
            geometry(&sim);
            let f = sim.total_lagrangian_frame().expect("on");
            let state = sim.total_lagrangian.as_ref().expect("on");
            println!(
                "frame {frame}: {:.2} s, steps {} rejected {} unconverged {} stagnated {} newton {} gmres {} passes {} contacts {} step length {:.3e} s, balance {ratio:.3} of its bound, kinetic {:.3e} J/m",
                clock.elapsed().as_secs_f64(),
                f.steps,
                f.rejected,
                f.unconverged,
                f.stagnated,
                f.newton,
                f.gmres,
                f.passes,
                f.contacts,
                state.step_length,
                f.parts[0]
            );
        }
    });
}

/// The free block's frequency and worst balance at the three cells,
/// printed, nothing held. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_acoustic_frequencies() {
    cap::capped("probe_acoustic_frequencies", 3600, || {
        let runs = [acoustic(0.04), acoustic(0.02), acoustic(0.01)];
        for (dx, (ratio, worst, shift)) in [4, 2, 1].iter().zip(runs) {
            println!(
                "free block at {dx} cm: omega over c pi sqrt(2) / L {ratio:.6} (the midpoint's shift {shift:+.2e}), worst balance {worst:.3} of its bound"
            );
        }
        let (order, ratio) = extrapolated(runs.map(|r| r.0));
        println!("extrapolated to dx -> 0: {ratio:.6} (order {order:.2})");
    });
}

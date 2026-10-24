//! A prototype for the real-time plan's vibration item: each impact
//! resolved by the step, the flight between impacts carried by the body's
//! exact modal oscillators. A probe: no engine change. It measures the
//! flight frames' cost and their error against the fine reference over a
//! whole bounce.
//!
//! In flight the motion splits into its rigid part and its vibration in
//! the body's own (corotated) frame:
//! - rigid: the centre of mass falls with `g` exactly; the rotation is the
//!   polar factor of `sum m (x - c) (X - C)^T`, turning at the spin that
//!   carries the angular momentum, `L / I`, held constant;
//! - vibration: the body-frame displacement and velocity projected on the
//!   elastic modes (`vibration::modes`), each advanced exactly, `z(t) =
//!   z(0) e^(i omega t)`, the undamped linear mode's own solution.
//!
//! Declared, left out: the vibration's small change of `I`, the body
//! frame's centrifugal and Coriolis forces, and the modes' nonlinear
//! exchange. The rebuilt state, `x = c + R (X - C + u_b)`, `v = v_c +
//! Omega perp(x - c) + R v_b`, `F = R (I + grad u_b)`, is the step's
//! state again, so the step takes over unchanged at the next contact; its
//! energy under the material's own law is reported against the start's.

use std::time::Instant;

use super::vibration::{
    FRAME, Modes, RIGID, Sample, modes, print_ledger, sample, scene, sensors, z,
};
use super::*;

/// One flight: its start time and its rigid and modal state there.
struct Flight {
    t: f64,
    centre: DVec2,
    velocity: DVec2,
    angle: f64,
    spin: f64,
    /// The reference centre of mass, `C`.
    origin: DVec2,
    /// Per mode, `z = omega q - i q'` in the body's frame (zero for the
    /// rigid ones).
    z: Vec<DVec2>,
}

/// The body's centre of mass, its velocity and its spin `L / I`.
fn rigid(state: &TotalLagrangianState) -> (DVec2, DVec2, f64) {
    let n = state.particles;
    let total: f64 = state.mass[..n].iter().sum();
    let (sum_x, sum_v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(a, b), p| {
        (
            a + state.mass[p] * state.x[p],
            b + state.mass[p] * state.v[p],
        )
    });
    let (centre, velocity) = (sum_x / total, sum_v / total);
    let (momentum, inertia) = (0..n).fold((0.0, 0.0), |(l, i), p| {
        let arm = state.x[p] - centre;
        (
            l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
            i + state.mass[p] * arm.length_squared(),
        )
    });
    (centre, velocity, momentum / inertia)
}

/// The flight state at `t`, and the largest rigid share left in the
/// body-frame motion (the split's own check, relative to the elastic).
fn split(state: &TotalLagrangianState, modes: &Modes, t: f64) -> (Flight, f64) {
    let n = state.particles;
    let total: f64 = state.mass[..n].iter().sum();
    let (centre, velocity, spin) = rigid(state);
    let origin = (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.reference[p]) / total;
    // Polar factor of `A = sum m (x - c) (X - C)^T`: in the plane, the
    // rotation by `atan2(A_10 - A_01, A_00 + A_11)`.
    let a = (0..n).fold(DMat2::ZERO, |a, p| {
        let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
        a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
    });
    let angle = (a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y);
    let back = DMat2::from_angle(-angle);
    let q: Vec<DVec2> = modes
        .at
        .iter()
        .map(|at| {
            (0..n).fold(DVec2::ZERO, |acc, p| {
                let arm = state.x[p] - centre;
                let u = back * arm - (state.reference[p] - origin);
                let w = back * (state.v[p] - velocity - spin * arm.perp());
                acc + state.mass[p] * DVec2::new(u.dot(at[p]), w.dot(at[p]))
            })
        })
        .collect();
    let elastic = q[RIGID..]
        .iter()
        .zip(&modes.omega[RIGID..])
        .map(|(c, &w)| z(w, *c).length())
        .fold(0.0, f64::max);
    let left =
        q[..RIGID].iter().map(|c| c.y.abs()).fold(0.0, f64::max) / elastic.max(f64::MIN_POSITIVE);
    let z_all = q
        .iter()
        .zip(&modes.omega)
        .enumerate()
        .map(|(i, (c, &w))| if i < RIGID { DVec2::ZERO } else { z(w, *c) })
        .collect();
    let flight = Flight {
        t,
        centre,
        velocity,
        angle,
        spin,
        origin,
        z: z_all,
    };
    (flight, left)
}

/// The flight advanced to `t`, written into the step's state.
fn rebuild(
    flight: &Flight,
    modes: &Modes,
    gravity: DVec2,
    t: f64,
    state: &mut TotalLagrangianState,
) {
    let tau = t - flight.t;
    let centre = flight.centre + flight.velocity * tau + 0.5 * gravity * tau * tau;
    let velocity = flight.velocity + gravity * tau;
    let turn = DMat2::from_angle(flight.angle + flight.spin * tau);
    let coordinates: Vec<DVec2> = flight
        .z
        .iter()
        .zip(&modes.omega)
        .map(|(z0, &w)| {
            let zt = DVec2::from_angle(w * tau).rotate(*z0);
            // `q = Re z / omega`, `q' = -Im z`.
            DVec2::new(if w > 0.0 { zt.x / w } else { 0.0 }, -zt.y)
        })
        .collect();
    for p in 0..state.x.len() {
        let (mut u, mut w) = (DVec2::ZERO, DVec2::ZERO);
        for (i, c) in coordinates.iter().enumerate().skip(RIGID) {
            u += modes.at[i][p] * c.x;
            w += modes.at[i][p] * c.y;
        }
        let arm = turn * (state.reference[p] - flight.origin + u);
        state.x[p] = centre + arm;
        state.v[p] = velocity + flight.spin * arm.perp() + turn * w;
        if p < state.particles {
            let g = coordinates
                .iter()
                .enumerate()
                .skip(RIGID)
                .fold(DMat2::ZERO, |g, (i, c)| g + modes.gradients[i][p] * c.x);
            state.f[p] = turn * (DMat2::IDENTITY + g);
        }
    }
}

/// The lowest gap of the body's outline over the walls.
fn lowest_gap(state: &TotalLagrangianState, planes: &[Plane]) -> f64 {
    state.outlines[0]
        .iter()
        .flat_map(|&q| {
            planes
                .iter()
                .map(move |plane| plane.inward * (state.x[q][plane.axis] - plane.at))
        })
        .fold(f64::INFINITY, f64::min)
}

/// What one frame of a run did: its time, whether it was flown modally,
/// its wall time, and its state's sample, rigid motion and energy.
struct Row {
    t: f64,
    modal: bool,
    wall: f64,
    sample: Sample,
    rigid: (DVec2, DVec2, f64),
    energy: f64,
}

fn row(
    sim: &Simulation,
    modes: &Modes,
    sensors: &[(usize, usize); 3],
    (t, modal, wall): (f64, bool, f64),
) -> Row {
    let state = sim.total_lagrangian.as_ref().expect("on the path");
    Row {
        t,
        modal,
        wall,
        sample: sample(sim, modes, sensors, t),
        rigid: rigid(state),
        energy: sim.total_lagrangian_energy(state) + state.heat,
    }
}

/// The prototype against the fine reference and the stepped coarse run,
/// over the first impact, the flight and the second impact.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_modal_flight() {
    super::cap::capped("probe_modal_flight", 1800, || {
        const FRAMES: usize = 18;
        let fine = (FRAME / 1024.0) as f32;
        let h = f64::from(fine);
        // (A) The fine reference over the whole run.
        let mut a = scene(fine, 1e-2);
        a.step();
        let state = a.total_lagrangian.as_ref().expect("on the path");
        let modes = modes(state);
        let sensors = sensors(state);
        let clock = Instant::now();
        let mut reference = vec![sample(&a, &modes, &sensors, h)];
        let mut contacts = vec![a.total_lagrangian_frame().expect("on the path").contacts];
        let mut at_frames = Vec::new();
        while reference.len() < 1024 * FRAMES {
            a.step();
            let t = (reference.len() + 1) as f64 * h;
            reference.push(sample(&a, &modes, &sensors, t));
            contacts.push(a.total_lagrangian_frame().expect("on the path").contacts);
            if reference.len() % 1024 == 0 {
                at_frames.push(row(&a, &modes, &sensors, (t, false, 0.0)));
            }
        }
        println!(
            "(A) fine: {} frames of {:.3} us, {:.0} s; the sensors' velocity against the field the modes rebuild: at most {:.1e} m/s",
            reference.len(),
            h * 1e6,
            clock.elapsed().as_secs_f64(),
            reference.iter().map(|s| s.off_field).fold(0.0, f64::max)
        );
        let onsets: Vec<f64> = (1..contacts.len())
            .filter(|&j| contacts[j] > 0 && contacts[j - 1] == 0)
            .map(|j| reference[j].t)
            .collect();
        let releases: Vec<f64> = (1..contacts.len())
            .filter(|&j| contacts[j] == 0 && contacts[j - 1] > 0)
            .map(|j| reference[j].t)
            .collect();
        println!(
            "  contact onsets, ms: {:?}; releases, ms: {:?}",
            onsets
                .iter()
                .map(|t| (t * 1e3 * 1e3).round() / 1e3)
                .collect::<Vec<_>>(),
            releases
                .iter()
                .map(|t| (t * 1e3 * 1e3).round() / 1e3)
                .collect::<Vec<_>>()
        );
        // (C) The stepped coarse run and (P) the prototype, frame by frame.
        let hframe = f64::from(FRAME as f32);
        let mut c = scene(FRAME as f32, 1e-2);
        let mut p = scene(FRAME as f32, 1e-2);
        let gravity = p.total_lagrangian_gravity();
        let planes = p.total_lagrangian_walls().0;
        let (mut stepped, mut flown) = (Vec::new(), Vec::new());
        let mut flight: Option<Flight> = None;
        let mut flights: Vec<(f64, f64, Vec<DVec2>)> = Vec::new();
        for k in 1..=FRAMES {
            let t = k as f64 * hframe;
            let clock = Instant::now();
            c.step();
            stepped.push(row(
                &c,
                &modes,
                &sensors,
                (t, false, clock.elapsed().as_secs_f64()),
            ));
            let clock = Instant::now();
            // Fly this frame if the flight stays clear of the walls to its end.
            let mut modal = false;
            if let Some(f) = &flight {
                let mut state = p.total_lagrangian.take().expect("on the path");
                let start = (state.x.clone(), state.v.clone(), state.f.clone());
                rebuild(f, &modes, gravity, t, &mut state);
                if lowest_gap(&state, &planes) > 0.0 {
                    modal = true;
                } else {
                    (state.x, state.v, state.f) = start;
                    let ended = flight.take().expect("flying");
                    flights.push((ended.t, t - hframe, ended.z));
                }
                p.total_lagrangian = Some(state);
            }
            if !modal {
                p.step();
                let state = p.total_lagrangian.as_ref().expect("on the path");
                let free = state.cache.as_ref().is_none_or(|c| c.contacts.is_empty());
                if free && lowest_gap(state, &planes) > 0.0 {
                    let (f, left) = split(state, &modes, t);
                    println!(
                        "(P) flight from {:.3} ms: rigid share left in the body frame {left:.1e}, spin {:.3e} rad/s",
                        t * 1e3,
                        f.spin
                    );
                    flight = Some(f);
                }
            }
            flown.push(row(
                &p,
                &modes,
                &sensors,
                (t, modal, clock.elapsed().as_secs_f64()),
            ));
        }
        if let Some(f) = flight.take() {
            flights.push((f.t, FRAMES as f64 * hframe, f.z));
        }
        // Frame by frame: rigid motion, vibration energy, energy balance.
        let start = flown[0].energy;
        println!(
            "frame | t ms | (A) y mm, vy m/s, vibration J/m | (C) y, vy, vibration, s | (P) y, vy, vibration, energy drift, s"
        );
        for k in 0..FRAMES {
            let (ra, rc, rp) = (&at_frames[k], &stepped[k], &flown[k]);
            let vibration = |r: &Row| r.sample.ledger[1] + r.sample.ledger[2] + r.sample.ledger[3];
            println!(
                "{:5} | {:6.2} | {:8.3} {:+.4} {:.4} | {:8.3} {:+.4} {:.4} {:6.2} | {:8.3} {:+.4} {:.4} {:+.1e} {:8.5} {}",
                k + 1,
                ra.t * 1e3,
                ra.rigid.0.y * 1e3,
                ra.rigid.1.y,
                vibration(ra),
                rc.rigid.0.y * 1e3,
                rc.rigid.1.y,
                vibration(rc),
                rc.wall,
                rp.rigid.0.y * 1e3,
                rp.rigid.1.y,
                vibration(rp),
                (rp.energy - start) / start,
                rp.wall,
                if rp.modal { "modal" } else { "stepped" }
            );
        }
        print_ledger(
            "(A) at the last frame",
            &at_frames[FRAMES - 1].sample.ledger,
        );
        print_ledger("(C) at the last frame", &stepped[FRAMES - 1].sample.ledger);
        print_ledger("(P) at the last frame", &flown[FRAMES - 1].sample.ledger);
        // The sensors over each flight, at the reference's own times: (P) from
        // its flight's start, and the same oscillators started from (A)'s own
        // state at that time, at the modes' frequencies and at the fine step's
        // rate (like for like).
        for (from, to, z0) in &flights {
            let j0 = reference
                .iter()
                .position(|s| s.t >= *from - 0.5 * h)
                .expect("in the reference");
            let window: Vec<&Sample> = reference[j0..].iter().take_while(|s| s.t <= *to).collect();
            for (s, &(point, axis)) in sensors.iter().enumerate() {
                let measure = |z_start: &[DVec2], rate: &dyn Fn(f64) -> f64| -> f64 {
                    let (mut error, mut size) = (0.0, 0.0);
                    for sample in &window {
                        let predicted: f64 = (RIGID..modes.omega.len())
                            .map(|i| {
                                let w = modes.omega[i];
                                let turned = DVec2::from_angle(rate(w) * (sample.t - from));
                                modes.at[i][point][axis] * -turned.rotate(z_start[i]).y
                            })
                            .sum();
                        error += (predicted - sample.sensors[s]).powi(2);
                        size += sample.sensors[s].powi(2);
                    }
                    (error / size).sqrt()
                };
                let own: Vec<DVec2> = reference[j0]
                    .q
                    .iter()
                    .zip(&modes.omega)
                    .map(|(c, &w)| z(w, *c))
                    .collect();
                let exact = |w: f64| w;
                let stepped_rate = |w: f64| 4.0 * (w * h / 4.0).atan() / h;
                println!(
                    "  flight {:.1}-{:.1} ms, sensor {s}: RMS error against (A) of (P) {:.2e} (like for like {:.2e}); of the oscillators from (A)'s own state {:.2e} (like for like {:.2e})",
                    from * 1e3,
                    to * 1e3,
                    measure(z0, &exact),
                    measure(z0, &stepped_rate),
                    measure(&own, &exact),
                    measure(&own, &stepped_rate)
                );
            }
        }
        // The flight frames' cost against the stepped ones.
        let cost = |rows: &[Row], modal: bool| -> (usize, f64) {
            let chosen: Vec<f64> = rows
                .iter()
                .filter(|r| r.modal == modal)
                .map(|r| r.wall)
                .collect();
            (
                chosen.len(),
                chosen.iter().sum::<f64>() / chosen.len().max(1) as f64,
            )
        };
        let ((n_modal, modal_wall), (n_stepped, stepped_wall)) =
            (cost(&flown, true), cost(&flown, false));
        println!(
            "cost per frame: (P) modal {n_modal} frames, {:.3} ms; (P) stepped {n_stepped} frames, {:.0} ms; (C) all stepped, {:.0} ms",
            modal_wall * 1e3,
            stepped_wall * 1e3,
            stepped.iter().map(|r| r.wall).sum::<f64>() / FRAMES as f64 * 1e3
        );
    });
}

/// The fine reference's run over the whole bounce at frame `fine`: the
/// rows at each real-time frame's end, and the contact onsets.
fn fine_run(
    fine: f32,
    modes: &Modes,
    sensors: &[(usize, usize); 3],
    frames: usize,
) -> (Vec<Row>, Vec<f64>) {
    let h = f64::from(fine);
    let per = (FRAME / h).round() as usize;
    let mut sim = scene(fine, 1e-2);
    let (mut rows, mut onsets, mut held) = (Vec::new(), Vec::new(), false);
    for j in 1..=per * frames {
        sim.step();
        let now = sim.total_lagrangian_frame().expect("on the path").contacts > 0;
        if now && !held {
            onsets.push(j as f64 * h);
        }
        held = now;
        if j % per == 0 {
            rows.push(row(&sim, modes, sensors, (j as f64 * h, false, 0.0)));
        }
    }
    (rows, onsets)
}

/// Whether the fine reference's second impact is converged (16 against 4
/// us frames over the whole bounce), and the modal flight started from the
/// fine reference's own state after the first impact, its second impact
/// stepped at the fine frame: the flight's error apart from the first
/// impact's.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_second_impact_convergence() {
    super::cap::capped("probe_second_impact_convergence", 1800, || {
        const FRAMES: usize = 18;
        let fine = (FRAME / 1024.0) as f32;
        let h = f64::from(fine);
        let mut a = scene(fine, 1e-2);
        a.step();
        let state = a.total_lagrangian.as_ref().expect("on the path");
        let modes = modes(state);
        let sensors = sensors(state);
        let clock = Instant::now();
        let (fine_rows, fine_onsets) = fine_run(fine, &modes, &sensors, FRAMES);
        println!(
            "(A) {:.3} us: {:.0} s",
            h * 1e6,
            clock.elapsed().as_secs_f64()
        );
        let clock = Instant::now();
        let (finer_rows, finer_onsets) = fine_run(fine / 4.0, &modes, &sensors, FRAMES);
        println!(
            "(A'') {:.3} us: {:.0} s",
            h * 0.25e6,
            clock.elapsed().as_secs_f64()
        );
        // (F) the fine reference to the first frame's end, then the modal
        // flight, then the fine frame again from the frame whose end crosses.
        let mut f = scene(fine, 1e-2);
        for _ in 0..1024 {
            f.step();
        }
        let gravity = f.total_lagrangian_gravity();
        let planes = f.total_lagrangian_walls().0;
        let hframe = f64::from(FRAME as f32);
        let (flight, left) = split(
            f.total_lagrangian.as_ref().expect("on the path"),
            &modes,
            1024.0 * h,
        );
        println!(
            "(F) flight from {:.3} ms, rigid share left {left:.1e}",
            flight.t * 1e3
        );
        let mut k = 1;
        loop {
            let mut state = f.total_lagrangian.take().expect("on the path");
            let start = (state.x.clone(), state.v.clone(), state.f.clone());
            rebuild(
                &flight,
                &modes,
                gravity,
                (k + 1) as f64 * hframe,
                &mut state,
            );
            let clear = lowest_gap(&state, &planes) > 0.0;
            if !clear {
                (state.x, state.v, state.f) = start;
            }
            f.total_lagrangian = Some(state);
            if !clear {
                break;
            }
            k += 1;
        }
        println!(
            "(F) modal to {:.3} ms, then stepped at {:.3} us",
            k as f64 * hframe * 1e3,
            h * 1e6
        );
        let mut flown_rows = Vec::new();
        let mut onset = None;
        for j in 1..=1024 * (FRAMES - k) {
            f.step();
            if onset.is_none() && f.total_lagrangian_frame().expect("on the path").contacts > 0 {
                onset = Some(k as f64 * hframe + j as f64 * h);
            }
            if j % 1024 == 0 {
                flown_rows.push(row(
                    &f,
                    &modes,
                    &sensors,
                    (k as f64 * hframe + j as f64 * h, false, 0.0),
                ));
            }
        }
        let second = |onsets: &[f64]| {
            onsets
                .iter()
                .copied()
                .find(|&t| t > 0.1)
                .unwrap_or(f64::NAN)
        };
        println!(
            "second impact's onset, ms: (A) {:.4}, (A'') {:.4}, (F) {:.4}",
            second(&fine_onsets) * 1e3,
            second(&finer_onsets) * 1e3,
            onset.unwrap_or(f64::NAN) * 1e3
        );
        let vibration = |r: &Row| r.sample.ledger[1] + r.sample.ledger[2] + r.sample.ledger[3];
        println!(
            "frame | (A) y mm, vy m/s, vibration J/m | (A'') y, vy, vibration | (F) y, vy, vibration"
        );
        for i in k..FRAMES {
            let (ra, rb) = (&fine_rows[i], &finer_rows[i]);
            let rf = &flown_rows[i - k];
            println!(
                "{:5} | {:8.3} {:+.4} {:.4} | {:8.3} {:+.4} {:.4} | {:8.3} {:+.4} {:.4}",
                i + 1,
                ra.rigid.0.y * 1e3,
                ra.rigid.1.y,
                vibration(ra),
                rb.rigid.0.y * 1e3,
                rb.rigid.1.y,
                vibration(rb),
                rf.rigid.0.y * 1e3,
                rf.rigid.1.y,
                vibration(rf)
            );
        }
        for i in [0, k - 1] {
            println!(
                "  frame {}: (A) y {:.3} mm vy {:+.4} vibration {:.4}; (A'') y {:.3} vy {:+.4} vibration {:.4}",
                i + 1,
                fine_rows[i].rigid.0.y * 1e3,
                fine_rows[i].rigid.1.y,
                vibration(&fine_rows[i]),
                finer_rows[i].rigid.0.y * 1e3,
                finer_rows[i].rigid.1.y,
                vibration(&finer_rows[i])
            );
        }
    });
}

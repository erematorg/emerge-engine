//! Whether a body's fast elastic vibration, its energy kept by a coarse
//! step but its phase aliased, can be recovered from the body's linear
//! modes.
//! A probe: it measures, and changes no answer of the path.
//!
//! The modes are the step's own: its grid on the reference configuration
//! (the extended B-splines included), its consistent mass, and the
//! corotated law's tangent at rest, `K phi = omega^2 M phi` with `phi^T M
//! phi = 1`, the mass-orthonormal basis `u = U q`, `U^T M U = I`, of
//! Barbic's thesis (CMU-CS-07-145, 2007, sec. 2.6). The grid stays on the
//! reference, so every displacement and velocity the step makes lies in
//! the basis, and the modal coordinates are exact projections: `q_i =
//! sum_p m_p (x_p - X_p) . phi_i(X_p)`, `q_i' = sum_p m_p v_p .
//! phi_i(X_p)`.

use std::f64::consts::{PI, TAU};
use std::time::Instant;

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::matter::materials::solid::corotated::stress::corotated_second_pk_tangent;
use crate::spacetime::total_lagrangian::{ElasticLaw, Grid};
use crate::tensor_algebra::dense::product;
use crate::tensor_algebra::symmetric_eigen::jacobi_eigen_symmetric_definite;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// A free body's rigid modes in the plane: two translations and the
/// linearised rotation.
pub(super) const RIGID: usize = 3;
/// The real-time frame, s.
pub(super) const FRAME: f64 = 1.0 / 60.0;
/// The band edge between audible and above, Hz: the conventional upper
/// limit of human hearing. Declared, not
/// sourced here: to be replaced by a receptor's bandwidth read from a
/// source.
pub(super) const AUDIBLE: f64 = 2.0e4;
/// A mode counts as excited from this share of the elastic energy at the
/// release: a reporting threshold, no physics.
const EXCITED: f64 = 1e-3;

/// E2's 8 x 4 cm clay block (`tests/realtime_profile.rs`), its bottom half
/// a cell above the floor and falling at E2's impact speed (`sqrt(2 g h)`
/// after its 10-cell drop), frame `frame` s, step tolerance `tolerance`.
pub(super) fn scene(frame: f32, tolerance: f64) -> Simulation {
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, frame);
    // The explicit path's substep floor, which this path never reads; a
    // frame below it is refused by the config's checks.
    config.min_dt = config.min_dt.min(frame);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(tolerance));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 0.5 + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let speed = (2.0 * config.gravity.length() * 10.0).sqrt();
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(0.0, -speed);
    }
    sim
}

/// The material gradient `dF = grad_X d` of the grid field `d` at point
/// `p` (as the step forms it).
pub(super) fn gradient(grid: &Grid, d: &[f64], p: usize) -> DMat2 {
    let (_, gu) = grid.read(d, p, 0);
    let (_, gv) = grid.read(d, p, 1);
    DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
}

/// `K d`: the internal force's derivative at rest, `sum_p V0_p dP : grad
/// N`, `dP = dS` there (`S = 0` at `F = I`), `dS` the corotated law's own
/// tangent with `dC = dF + dF^T`.
fn stiffness_times(grid: &Grid, d: &[f64], state: &TotalLagrangianState) -> Vec<f64> {
    let mut out = vec![0.0; grid.len];
    for p in 0..state.particles {
        let (v0, solid) = (state.v0[p], state.solids[p]);
        let g = gradient(grid, d, p);
        let ds = corotated_second_pk_tangent(
            DMat2::IDENTITY,
            g + g.transpose(),
            solid.bulk - solid.mu,
            solid.mu,
        );
        for a in 0..2 {
            let row = DVec2::new(ds.x_axis[a], ds.y_axis[a]);
            grid.visit(p, a, |k, _, grad| out[k] += v0 * row.dot(grad));
        }
    }
    out
}

/// The body's linear modes on the step's grid, ascending, the first
/// `RIGID` its rigid motions.
pub(super) struct Modes {
    /// Angular frequencies, rad/s (the rigid ones set to zero).
    pub(super) omega: Vec<f64>,
    /// Per mode, its value at every point of the state (particles, then
    /// outline points).
    pub(super) at: Vec<Vec<DVec2>>,
    /// Per mode, its material gradient at every particle.
    pub(super) gradients: Vec<Vec<DMat2>>,
}

/// The modes, by two-sided Jacobi on `M`, then on `M^-1/2 (K + sigma M)
/// M^-1/2`: the shift `sigma` makes the matrix definite, the case the
/// algorithm states, and is removed from the eigenvalues after. Prints the
/// size, the cost and the checks.
pub(super) fn modes(state: &TotalLagrangianState) -> Modes {
    let clock = Instant::now();
    let grid = &state.cache.as_ref().expect("built by the first frame").grid;
    assert!(
        state.solids[..state.particles]
            .iter()
            .all(|s| s.law == ElasticLaw::Corotated
                && s.shear_viscosity == 0.0
                && s.bulk_viscosity == 0.0),
        "the probe's stiffness is the undamped corotated law's"
    );
    let unknowns: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let n = unknowns.len();
    let (mut mass, mut stiffness) = (vec![0.0; n * n], vec![0.0; n * n]);
    for (j, &k) in unknowns.iter().enumerate() {
        let mut unit = vec![0.0; grid.len];
        unit[k] = 1.0;
        let m = grid.mass_times(&unit, &state.mass, 0.0);
        let s = stiffness_times(grid, &unit, state);
        for (i, &r) in unknowns.iter().enumerate() {
            mass[i * n + j] = m[r];
            stiffness[i * n + j] = s[r];
        }
    }
    let symmetric = |a: &mut [f64]| {
        for i in 0..n {
            for j in (i + 1)..n {
                let mean = 0.5 * (a[i * n + j] + a[j * n + i]);
                (a[i * n + j], a[j * n + i]) = (mean, mean);
            }
        }
    };
    symmetric(&mut mass);
    symmetric(&mut stiffness);
    let assembled = clock.elapsed().as_secs_f64();
    let tol = 1e-14;
    let (d, q) = jacobi_eigen_symmetric_definite(mass.clone(), n, tol);
    let (lightest, heaviest) = d
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(a, b), &e| (a.min(e), b.max(e)));
    assert!(lightest > 0.0, "the mass is definite: {lightest:e}");
    let root: Vec<f64> = (0..n * n)
        .map(|ij| {
            (0..n)
                .map(|e| q[ij / n * n + e] * q[ij % n * n + e] / d[e].sqrt())
                .sum()
        })
        .collect();
    let mut reduced = product(&product(&root, &stiffness, n), &root, n);
    symmetric(&mut reduced);
    let shift = (0..n).map(|i| reduced[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        reduced[i * n + i] += shift;
    }
    let (lambda, y) = jacobi_eigen_symmetric_definite(reduced, n, tol);
    let phi = product(&root, &y, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| lambda[a].total_cmp(&lambda[b]));
    let solved = clock.elapsed().as_secs_f64() - assembled;
    let squared: Vec<f64> = order.iter().map(|&e| lambda[e] - shift).collect();
    // Checks: mass-orthonormality and the eigen-residual, on the dense
    // matrices.
    let column = |e: usize| -> Vec<f64> { (0..n).map(|i| phi[i * n + e]).collect() };
    let times = |a: &[f64], x: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
            .collect()
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let (mut orthonormal, mut residual) = (0.0f64, 0.0f64);
    for (rank, &e) in order.iter().enumerate() {
        let x = column(e);
        let (mx, kx) = (times(&mass, &x), times(&stiffness, &x));
        for &f in &order {
            let expected = if f == e { 1.0 } else { 0.0 };
            orthonormal = orthonormal.max((dot(&column(f), &mx) - expected).abs());
        }
        if rank >= RIGID {
            let r: Vec<f64> = kx
                .iter()
                .zip(&mx)
                .map(|(k, m)| k - squared[rank] * m)
                .collect();
            residual = residual.max(dot(&r, &r).sqrt() / dot(&kx, &kx).sqrt());
        }
    }
    let full = |e: usize| -> Vec<f64> {
        let mut out = vec![0.0; grid.len];
        for (i, &k) in unknowns.iter().enumerate() {
            out[k] = phi[i * n + e];
        }
        out
    };
    // The first elastic mode's `phi^T K phi` from the law's own energy, by a
    // second difference at a strain of about 1e-4.
    let first = full(order[RIGID]);
    let largest = (0..state.particles)
        .map(|p| gradient(grid, &first, p).to_cols_array().map(f64::abs))
        .fold(0.0f64, |m, g| g.iter().fold(m, |m, &e| m.max(e)));
    let energy = |scale: f64| -> f64 {
        (0..state.particles)
            .map(|p| {
                let g = gradient(grid, &first, p) * scale;
                state.v0[p] * state.solids[p].energy(DMat2::IDENTITY + g)
            })
            .sum()
    };
    let h = 1e-4 / largest;
    let law = (energy(h) + energy(-h) - 2.0 * energy(0.0)) / (h * h);
    println!(
        "modes: {n} unknowns ({} faces reached), assembly {assembled:.2} s, the two Jacobi solves {solved:.2} s; mass eigenvalues {lightest:.3e} to {heaviest:.3e} kg/m",
        grid.len
    );
    println!(
        "  checks: |Phi^T M Phi - I| <= {orthonormal:.1e}; |K phi - w^2 M phi| / |K phi| <= {residual:.1e}; rigid w^2 / first elastic w^2: {:.1e} {:.1e} {:.1e}; first mode's w^2 from the law's energy / from K: {:.6}",
        squared[0] / squared[RIGID],
        squared[1] / squared[RIGID],
        squared[2] / squared[RIGID],
        law / squared[RIGID]
    );
    let omega: Vec<f64> = squared
        .iter()
        .enumerate()
        .map(|(rank, &w2)| {
            if rank < RIGID {
                0.0
            } else {
                w2.max(0.0).sqrt()
            }
        })
        .collect();
    let hz: Vec<String> = omega[RIGID..RIGID + 12]
        .iter()
        .map(|w| format!("{:.0}", w / TAU))
        .collect();
    println!(
        "  elastic modes, Hz: {} ... the highest {:.0}; at or above {AUDIBLE:.0} Hz: {}",
        hz.join(" "),
        omega[n - 1] / TAU,
        omega.iter().filter(|&&w| w / TAU >= AUDIBLE).count()
    );
    let (at, gradients) = order
        .iter()
        .map(|&e| {
            let field = full(e);
            let at = (0..state.x.len())
                .map(|p| DVec2::new(grid.read(&field, p, 0).0, grid.read(&field, p, 1).0))
                .collect();
            let gradients = (0..state.particles)
                .map(|p| gradient(grid, &field, p))
                .collect();
            (at, gradients)
        })
        .unzip();
    Modes {
        omega,
        at,
        gradients,
    }
}

/// The three surface sensors, outline points of the block: (point, axis)
/// for the top centre's and a top corner's vertical velocity and a side's
/// middle's horizontal one.
pub(super) fn sensors(state: &TotalLagrangianState) -> [(usize, usize); 3] {
    let outline = &state.outlines[0];
    let (lo, hi) = outline.iter().fold(
        (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)),
        |(lo, hi), &k| (lo.min(state.reference[k]), hi.max(state.reference[k])),
    );
    let nearest = |target: DVec2| -> usize {
        *outline
            .iter()
            .min_by(|&&a, &&b| {
                let da = (state.reference[a] - target).length();
                da.total_cmp(&(state.reference[b] - target).length())
            })
            .expect("an outline")
    };
    let centre = 0.5 * (lo + hi);
    [
        (nearest(DVec2::new(centre.x, hi.y)), 1),
        (nearest(hi), 1),
        (nearest(DVec2::new(hi.x, centre.y)), 0),
    ]
}

/// One state: its time, its modal coordinates `(q, q')`, its band ledger
/// (`ledger`), its sensors' elastic velocities (rigid part removed), and
/// how far the sensors' velocities lie from the field the modes rebuild.
pub(super) struct Sample {
    pub(super) t: f64,
    pub(super) q: Vec<DVec2>,
    pub(super) ledger: [f64; 6],
    pub(super) sensors: [f64; 3],
    pub(super) off_field: f64,
}

pub(super) fn sample(
    sim: &Simulation,
    modes: &Modes,
    sensors: &[(usize, usize); 3],
    t: f64,
) -> Sample {
    let state = sim.total_lagrangian.as_ref().expect("on the path");
    let q: Vec<DVec2> = modes
        .at
        .iter()
        .map(|at| {
            (0..state.particles).fold(DVec2::ZERO, |acc, p| {
                let (m, u) = (state.mass[p], state.x[p] - state.reference[p]);
                acc + m * DVec2::new(u.dot(at[p]), state.v[p].dot(at[p]))
            })
        })
        .collect();
    let along = |p: usize, axis: usize, modes_in: std::ops::Range<usize>| -> f64 {
        modes_in.map(|i| modes.at[i][p][axis] * q[i].y).sum()
    };
    let velocity = |p: usize, axis: usize| state.v[p][axis];
    let elastic = sensors.map(|(p, axis)| velocity(p, axis) - along(p, axis, 0..RIGID));
    let off_field = sensors
        .iter()
        .map(|&(p, axis)| (velocity(p, axis) - along(p, axis, 0..modes.omega.len())).abs())
        .fold(0.0, f64::max);
    let ledger = ledger(modes, state, &q);
    Sample {
        t,
        q,
        ledger,
        sensors: elastic,
        off_field,
    }
}

/// The band ledger, J/m: rigid kinetic; elastic below the frame's Nyquist;
/// from it to `AUDIBLE`; at or above `AUDIBLE`; the residual, `KE + stored`
/// less their sum (outside the linear basis); and `KE + stored`. Gravity's
/// potential belongs to the rigid motion and is left out of all of them.
pub(super) fn ledger(modes: &Modes, state: &TotalLagrangianState, q: &[DVec2]) -> [f64; 6] {
    let mut bands = [0.0; 6];
    for (rank, (&w, c)) in modes.omega.iter().zip(q).enumerate() {
        let f = w / TAU;
        let (band, e) = if rank < RIGID {
            (0, 0.5 * c.y * c.y)
        } else if f < 0.5 / FRAME {
            (1, 0.5 * (c.y * c.y + w * w * c.x * c.x))
        } else if f < AUDIBLE {
            (2, 0.5 * (c.y * c.y + w * w * c.x * c.x))
        } else {
            (3, 0.5 * (c.y * c.y + w * w * c.x * c.x))
        };
        bands[band] += e;
    }
    let mechanical: f64 = (0..state.particles)
        .map(|p| {
            0.5 * state.mass[p] * state.v[p].length_squared()
                + state.v0[p] * state.solids[p].energy(state.f[p])
        })
        .sum();
    bands[4] = mechanical - bands[..4].iter().sum::<f64>();
    bands[5] = mechanical;
    bands
}

pub(super) fn print_ledger(label: &str, l: &[f64; 6]) {
    let share = |e: f64| e / l[5];
    println!(
        "  {label}: KE + stored {:.6e} J/m; rigid {:.6}, elastic below the frame {:.2e}, audible {:.3e}, above {:.2e}, residual {:+.2e}",
        l[5],
        share(l[0]),
        share(l[1]),
        share(l[2]),
        share(l[3]),
        share(l[4])
    );
}

/// `z = omega q - i q'`: for `q = A cos(omega t + p)`, `z = A omega
/// e^(i (omega t + p))`, `|z|^2` twice the mode's energy.
pub(super) fn z(omega: f64, c: DVec2) -> DVec2 {
    DVec2::new(omega * c.x, -c.y)
}

/// The angle from `a` to `b`, in `(-pi, pi]`.
pub(super) fn turn(a: DVec2, b: DVec2) -> f64 {
    a.perp_dot(b).atan2(a.dot(b))
}

pub(super) fn wrap(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

/// The fine reference, the modal prediction from its release, the coarse
/// steps forked at the release, and the gate's coarse run, as
/// stated before the run (the module's doc).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_vibration_recovery() {
    super::cap::capped("probe_vibration_recovery", 1800, || {
        let fine = (FRAME / 1024.0) as f32;
        let h = f64::from(fine);
        let mut a = scene(fine, 1e-2);
        a.step();
        let state = a.total_lagrangian.as_ref().expect("on the path");
        let modes = modes(state);
        let sensors = sensors(state);
        let count = modes.omega.len();
        // (A) The fine reference: until 30 ms after the release and at least
        // three real-time frames.
        let clock = Instant::now();
        let mut record = vec![sample(&a, &modes, &sensors, h)];
        let (mut touched, mut release, mut fork) = (false, None::<usize>, None);
        let (mut steps, mut contact_frames, mut onset) = (0u64, 0u32, None::<f64>);
        let end = |r: usize| (r as f64 * h + 0.030).max(3.0 * FRAME);
        while release.is_none_or(|r| record.last().expect("sampled").t < end(r)) {
            a.step();
            let frame = a.total_lagrangian_frame().expect("on the path");
            steps += u64::from(frame.steps);
            let t = (record.len() + 1) as f64 * h;
            if frame.contacts > 0 {
                (touched, release) = (true, None);
                contact_frames += 1;
                onset.get_or_insert(t);
            } else if touched && release.is_none() {
                release = Some(record.len());
                fork = a.total_lagrangian.clone();
            }
            record.push(sample(&a, &modes, &sensors, t));
            assert!(t < 0.2, "no release");
        }
        let r = release.expect("released");
        let (start, last) = (&record[r], record.last().expect("sampled"));
        println!(
            "(A) fine: frame {:.3} us, {} frames, {steps} steps, {:.0} s; contact over {contact_frames} frames, release at {:.3} ms",
            h * 1e6,
            record.len(),
            clock.elapsed().as_secs_f64(),
            start.t * 1e3
        );
        print_ledger("(A) at the release", &start.ledger);
        print_ledger("(A) at the end", &last.ledger);
        let energy = |s: &Sample, i: usize| 0.5 * z(modes.omega[i], s.q[i]).length_squared();
        let elastic: f64 = (RIGID..count).map(|i| energy(start, i)).sum();
        let excited: Vec<usize> = (RIGID..count)
            .filter(|&i| energy(start, i) >= EXCITED * elastic)
            .collect();
        // Per excited mode: the reference's rate, least squares on its
        // unwrapped phase, and its amplitude's change.
        println!(
            "  excited modes (>= {EXCITED:e} of the elastic energy at release): {}",
            excited.len()
        );
        let mut rates = vec![0.0; count];
        for &i in &excited {
            let w = modes.omega[i];
            let mut phase = 0.0;
            let mut points = vec![(0.0, 0.0)];
            for j in (r + 1)..record.len() {
                phase += turn(z(w, record[j - 1].q[i]), z(w, record[j].q[i]));
                points.push((record[j].t - start.t, phase));
            }
            let mean =
                |f: fn(&(f64, f64)) -> f64| points.iter().map(f).sum::<f64>() / points.len() as f64;
            let (mt, mp) = (mean(|p| p.0), mean(|p| p.1));
            let slope = points.iter().map(|p| (p.0 - mt) * (p.1 - mp)).sum::<f64>()
                / points.iter().map(|p| (p.0 - mt).powi(2)).sum::<f64>();
            rates[i] = slope;
            let own = 4.0 * (w * h / 4.0).atan() / (w * h) - 1.0;
            println!(
                "    mode {i:3}: {:7.1} Hz, {:.4} of the elastic energy; reference rate {:+.2e} off the mode (the fine step's own shift {:+.2e}); amplitude end / release {:.5}",
                w / TAU,
                energy(start, i) / elastic,
                slope / w - 1.0,
                own,
                z(w, last.q[i]).length() / z(w, start.q[i]).length()
            );
        }
        // The beat of the two closest excited modes.
        let mut by_frequency = excited.clone();
        by_frequency.sort_by(|&a, &b| modes.omega[a].total_cmp(&modes.omega[b]));
        if let Some((i, j)) = by_frequency
            .windows(2)
            .map(|p| (p[0], p[1]))
            .min_by(|a, b| {
                let gap = |p: &(usize, usize)| modes.omega[p.1] - modes.omega[p.0];
                gap(a).total_cmp(&gap(b))
            })
        {
            println!(
                "  beat of the closest pair (modes {i}, {j}): modal {:.3} Hz, reference {:.3} Hz",
                (modes.omega[j] - modes.omega[i]) / TAU,
                (rates[j] - rates[i]) / TAU
            );
        }
        // The modal prediction from the release state at the sensors: each mode
        // at its own frequency (the linear system's exact solution), and at the
        // fine step's own rate `4 atan(omega h / 4) / h` (like for like: the
        // reference is that step's solution).
        println!(
            "  sensors' velocity against the field the modes rebuild: at most {:.1e} m/s over the run",
            record.iter().map(|s| s.off_field).fold(0.0, f64::max)
        );
        for (s, &(p, axis)) in sensors.iter().enumerate() {
            let rms = |window: f64, rate: &dyn Fn(f64) -> f64| -> (f64, f64) {
                let (mut error, mut size, mut count_in) = (0.0, 0.0, 0usize);
                for sample in record[r..].iter().filter(|s| s.t - start.t <= window) {
                    let predicted: f64 = (RIGID..count)
                        .map(|i| {
                            let w = modes.omega[i];
                            let turned = DVec2::from_angle(rate(w) * (sample.t - start.t));
                            modes.at[i][p][axis] * -turned.rotate(z(w, start.q[i])).y
                        })
                        .sum();
                    error += (predicted - sample.sensors[s]).powi(2);
                    size += sample.sensors[s].powi(2);
                    count_in += 1;
                }
                ((error / size).sqrt(), (size / count_in as f64).sqrt())
            };
            let exact = |w: f64| w;
            let stepped = |w: f64| 4.0 * (w * h / 4.0).atan() / h;
            let (short_exact, size) = rms(0.005, &exact);
            let (short_stepped, _) = rms(0.005, &stepped);
            let (long_exact, _) = rms(f64::INFINITY, &exact);
            let (long_stepped, _) = rms(f64::INFINITY, &stepped);
            println!(
                "  sensor {s} (point {p}, axis {axis}, elastic RMS {size:.3e} m/s over 5 ms): RMS error of the prediction at the modes' frequencies {short_exact:.2e} over 5 ms, {long_exact:.2e} over the window; at the fine step's rate {short_stepped:.2e}, {long_stepped:.2e}"
            );
        }
        // (A') Half and a quarter of (A)'s frame, to 1 ms after (A)'s release
        // (each run past its own release): is (A) converged through the
        // impact? Amplitudes compare directly (the step keeps each mode's
        // energy in flight); phases after removing the steps' own dispersion
        // since the contact's onset.
        let onset = onset.expect("touched");
        let at = &record[r + (0.001 / h).round() as usize];
        let elastic_of = |s: &Sample| s.ledger[1] + s.ledger[2] + s.ledger[3];
        let mut finer = Vec::new();
        for divisor in [2.0f32, 4.0] {
            let hd = f64::from(fine / divisor);
            let mut run = scene(fine / divisor, 1e-2);
            let (mut frames, mut touched, mut released) = (0usize, false, None::<f64>);
            while (frames as f64 + 0.5) * hd < at.t {
                run.step();
                frames += 1;
                if run.total_lagrangian_frame().expect("on the path").contacts > 0 {
                    (touched, released) = (true, None);
                } else if touched && released.is_none() {
                    released = Some(frames as f64 * hd);
                }
            }
            let s = sample(&run, &modes, &sensors, frames as f64 * hd);
            println!(
                "(A') frame {:.3} us: release at {:.3} ms ((A): {:.3} ms); at {:.3} ms ((A): {:.3} ms) its elastic energy over (A)'s {:.5}",
                hd * 1e6,
                released.expect("released") * 1e3,
                start.t * 1e3,
                s.t * 1e3,
                at.t * 1e3,
                elastic_of(&s) / elastic_of(at)
            );
            print_ledger("(A')", &s.ledger);
            finer.push((hd, s));
        }
        let mut weighted = vec![0.0; finer.len()];
        for &i in &excited {
            let w = modes.omega[i];
            let rate = |step: f64| 4.0 * (w * step / 4.0).atan() / step;
            let parts: Vec<String> = finer
                .iter()
                .zip(&mut weighted)
                .map(|((hd, s), sum)| {
                    let ratio = z(w, s.q[i]).length() / z(w, at.q[i]).length();
                    *sum += energy(at, i) / elastic * (ratio - 1.0).abs();
                    let phase = wrap(
                        turn(z(w, at.q[i]), z(w, s.q[i])) - (rate(*hd) - rate(h)) * (at.t - onset),
                    );
                    format!(
                        "{:.3} us: amplitude {ratio:.4}, phase {phase:+.4} rad",
                        hd * 1e6
                    )
                })
                .collect();
            println!("    mode {i:3} ({:6.0} Hz): {}", w / TAU, parts.join("; "));
        }
        println!(
            "  energy-weighted |amplitude ratio - 1| against (A): {}",
            finer
                .iter()
                .zip(&weighted)
                .map(|((hd, _), e)| format!("{:.3} us {e:.4}", hd * 1e6))
                .collect::<Vec<_>>()
                .join(", ")
        );
        // (B) Coarse fixed steps forked at the release: each excited mode's
        // rotation per kept step (two half steps) against the midpoint's.
        for divisor in [1.0, 10.0, 100.0] {
            let coarse = (FRAME / divisor) as f32;
            let hc = f64::from(coarse);
            let mut b = scene(coarse, 1e30);
            let mut state = fork.clone().expect("forked at the release");
            state.step_length = hc;
            b.total_lagrangian = Some(state);
            let mut previous = sample(&b, &modes, &sensors, start.t);
            let first = previous.q.clone();
            let mut turns = vec![Vec::new(); count];
            let (mut taken, mut rejected) = (0, 0);
            for k in 1..=10 {
                b.step();
                let frame = b.total_lagrangian_frame().expect("on the path");
                (taken, rejected) = (taken + frame.steps, rejected + frame.rejected);
                let now = sample(&b, &modes, &sensors, start.t + k as f64 * hc);
                for &i in &excited {
                    let w = modes.omega[i];
                    turns[i].push(turn(z(w, previous.q[i]), z(w, now.q[i])));
                }
                previous = now;
            }
            println!(
                "(B) step {:.4e} s forked at the release: {taken} steps, {rejected} rejected in 10 frames",
                hc
            );
            print_ledger("(B) after 10 steps", &previous.ledger);
            for &i in &excited {
                let w = modes.omega[i];
                let sum = turns[i]
                    .iter()
                    .fold(DVec2::ZERO, |acc, &t| acc + DVec2::from_angle(t));
                let mean = sum.y.atan2(sum.x);
                let spread = turns[i]
                    .iter()
                    .map(|&t| wrap(t - mean).abs())
                    .fold(0.0, f64::max);
                println!(
                    "    mode {i:3} (w h {:8.3}): rotation per step {:+.5} (spread {:.1e}), midpoint halves 4 atan(w h / 4) {:+.5}, exact w h {:+.5} (mod 2 pi); one step against halves {:+.4}; amplitude after / before {:.5}",
                    w * hc,
                    mean,
                    spread,
                    wrap(4.0 * (w * hc / 4.0).atan()),
                    wrap(w * hc),
                    wrap(2.0 * (w * hc / 2.0).atan() - 4.0 * (w * hc / 4.0).atan()),
                    z(w, previous.q[i]).length() / z(w, first[i]).length()
                );
            }
        }
        // (C) The gate's coarse run from the start, against (A) at its frames.
        let mut c = scene(FRAME as f32, 1e-2);
        let hframe = f64::from(FRAME as f32);
        for k in 1..=6 {
            let clock = Instant::now();
            c.step();
            let frame = c.total_lagrangian_frame().expect("on the path");
            let now = sample(&c, &modes, &sensors, k as f64 * hframe);
            println!(
                "(C) frame {k}: {} steps, {} rejected, contacts {}, {:.1} s",
                frame.steps,
                frame.rejected,
                frame.contacts,
                clock.elapsed().as_secs_f64()
            );
            print_ledger("(C)", &now.ledger);
            let Some(j) = record.iter().position(|s| s.t >= now.t) else {
                continue;
            };
            print_ledger("(A) same time", &record[j].ledger);
            for &i in &excited {
                let w = modes.omega[i];
                // (A) carried to (C)'s exact time along its own mode.
                let reference =
                    DVec2::from_angle(w * (now.t - record[j].t)).rotate(z(w, record[j].q[i]));
                let coarse = z(w, now.q[i]);
                println!(
                    "    mode {i:3}: (C) / (A) amplitude {:.4}, phase {:+.4} rad (of pi: {:+.3})",
                    coarse.length() / reference.length(),
                    turn(reference, coarse),
                    turn(reference, coarse) / PI
                );
            }
        }
    });
}

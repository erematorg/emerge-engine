//! Probes of the path inside `Simulation`.

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// The frictionless slide of `tests/total_lagrangian_path.rs` with
/// every Newton iteration of frame `EMERGE_PROBE_TL_FRAME` (default 3)
/// printed. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_wired_slide_newton() {
    super::cap::capped("probe_wired_slide_newton", 1800, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 2.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        for v in sim.particles_mut().v.iter_mut() {
            *v = Vec2::new(50.0, 0.0);
        }
        let watched: u64 = std::env::var("EMERGE_PROBE_TL_FRAME")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        for frame in 1..=watched {
            if frame == watched {
                crate::spacetime::total_lagrangian::hooks::start(0);
            }
            sim.step();
            if frame == watched {
                for (solve, lines) in crate::spacetime::total_lagrangian::hooks::finish()
                    .iter()
                    .enumerate()
                {
                    for line in lines {
                        println!("  solve {solve}: {line}");
                    }
                }
            }
            println!("frame {frame}: {:?}", sim.total_lagrangian_frame());
        }
    });
}

/// How far a free body's elastic modes are from orthogonal to its rigid
/// rotation on the particles, `leak_i = sum_p m_p (X_p - C) x phi_i(X_p)`,
/// against the angular momentum allowance modal flight checks
/// (`Flight::keeps_angular_momentum`): the ringing bar of `tests/zener.rs`
/// and the clay block of `tests/modal_flight.rs`, after one stepped frame.
/// A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_modes_rotation_leak() {
    super::cap::capped("probe_modes_rotation_leak", 1800, || {
        let bar = || {
            let mut config = SimConfig::earth(48, 0.01, 2.0e-3);
            config.gravity = Vec2::ZERO;
            config.min_dt = config.min_dt.min(1e-9);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_flight());
            let props = ElasticProps {
                e_pa: 1.0e5,
                nu: 0.0,
                rho_kg_m3: 1000.0,
            };
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
                .0;
            let _bar = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(32, 2))
                    .at(Vec2::new(24.0, 24.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            let x = sim.particles().x.clone();
            for (v, q) in sim.particles_mut().v.iter_mut().zip(x) {
                *v = Vec2::new(
                    5.0e-3 * (std::f32::consts::PI * (q.x - 8.0) / 32.0).cos(),
                    0.0,
                );
            }
            sim
        };
        let block = || {
            let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1).with_modal_flight());
            let props = ElasticProps {
                e_pa: 1.125e7,
                nu: 0.3,
                rho_kg_m3: 1700.0,
            };
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
                .0;
            let floor = config.boundary_thickness as f32;
            let _block = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(8, 4))
                    .at(Vec2::new(20.0, floor + 12.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            sim
        };
        for (name, mut sim) in [("bar", bar()), ("block", block())] {
            sim.step();
            let state = sim.total_lagrangian.as_ref().expect("on the path");
            let Some(Some(modes)) = &state.modes else {
                println!("{name}: no modes");
                continue;
            };
            let n = state.particles;
            let total: f64 = state.mass[..n].iter().sum();
            let origin =
                (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.reference[p]) / total;
            let inertia: f64 = (0..n)
                .map(|p| state.mass[p] * (state.reference[p] - origin).length_squared())
                .sum();
            let velocity = (0..n).fold(DVec2::ZERO, |a, p| a + state.mass[p] * state.v[p]) / total;
            let (mut worst, mut predicted, mut size) = (0.0f64, 0.0f64, 0.0f64);
            for i in crate::spacetime::total_lagrangian::RIGID_MODES..modes.omega.len() {
                let leak: f64 = (0..n)
                    .map(|p| state.mass[p] * (state.reference[p] - origin).perp_dot(modes.at[i][p]))
                    .sum();
                let rate: f64 = (0..n)
                    .map(|p| state.mass[p] * (state.v[p] - velocity).dot(modes.at[i][p]))
                    .sum();
                worst = worst.max(leak.abs() / inertia.sqrt());
                predicted += leak.abs() * rate.abs();
            }
            for p in 0..n {
                let arm = state.x[p] - origin;
                size += state.mass[p] * arm.length() * (state.v[p].length() + velocity.length());
            }
            // One flown frame by hand: the angular momentum before and after
            // the rebuild, and the leak's signed prediction `sum_i leak_i
            // q'_i` from the rebuilt velocities.
            let momentum = |s: &TotalLagrangianState| -> f64 {
                let total: f64 = s.mass[..n].iter().sum();
                let (c, v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(c, v), p| {
                    (c + s.mass[p] * s.x[p], v + s.mass[p] * s.v[p])
                });
                let (c, v) = (c / total, v / total);
                (0..n)
                    .map(|p| s.mass[p] * (s.x[p] - c).perp_dot(s.v[p] - v))
                    .sum()
            };
            let mut trial = state.clone();
            let length = f64::from(sim.config().dt);
            let before = momentum(&trial);
            let mut flight =
                Flight::start(&trial, modes, (trial.time, length)).expect("a flight starts");
            assert!(flight.rebuild(modes, DVec2::ZERO, trial.time + length, &mut trial));
            let after = momentum(&trial);
            let rebuilt_velocity =
                (0..n).fold(DVec2::ZERO, |a, p| a + trial.mass[p] * trial.v[p]) / total;
            let signed: f64 = (crate::spacetime::total_lagrangian::RIGID_MODES..modes.omega.len())
                .map(|i| {
                    let leak: f64 = (0..n)
                        .map(|p| {
                            state.mass[p] * (state.reference[p] - origin).perp_dot(modes.at[i][p])
                        })
                        .sum();
                    let rate: f64 = (0..n)
                        .map(|p| {
                            trial.mass[p] * (trial.v[p] - rebuilt_velocity).dot(modes.at[i][p])
                        })
                        .sum();
                    leak * rate
                })
                .sum();
            // The eigen-solve itself: each elastic mode's residual and its
            // mass inner product with the solve's own rigid modes.
            let grid = &state.cache.as_ref().expect("stepped").grid;
            let system = crate::spacetime::total_lagrangian::damped_system(
                grid,
                (&state.mass, &state.v0, &state.solids),
                n,
            )
            .expect("an elastic body")
            .0;
            let k = system.n;
            let (squared, phi, _) =
                crate::spacetime::total_lagrangian::eigen_pairs(system.m.clone(), &system.k, k)
                    .expect("definite");
            let column = |e: usize| -> Vec<f64> { (0..k).map(|r| phi[r * k + e]).collect() };
            let times = |a: &[f64], x: &[f64]| -> Vec<f64> {
                (0..k)
                    .map(|r| (0..k).map(|c| a[r * k + c] * x[c]).sum())
                    .collect()
            };
            let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
            let norm_k = dot(&system.k, &system.k).sqrt();
            let (mut residual, mut rigid_overlap) = (0.0f64, 0.0f64);
            let rigid: Vec<Vec<f64>> = (0..3).map(column).collect();
            for (e, &w2) in squared.iter().enumerate().skip(3) {
                let x = column(e);
                let (kx, mx) = (times(&system.k, &x), times(&system.m, &x));
                let r: f64 = kx
                    .iter()
                    .zip(&mx)
                    .map(|(a, b)| (a - w2 * b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                residual = residual.max(r / (norm_k * dot(&x, &x).sqrt()));
                for g in &rigid {
                    rigid_overlap = rigid_overlap.max(dot(g, &mx).abs());
                }
            }
            println!(
                "{name}: one flown frame: L {before:.3e} -> {after:.3e}, change {:.3e}; sum_i leak_i q'_i {signed:.3e}; eigen residual at most {residual:.3e} of |K|, rigid overlap phi_r^T M phi_i at most {rigid_overlap:.3e}",
                after - before
            );
            println!(
                "{name}: {} modes (largest {:.0} Hz, {} above 3 kHz); largest leak over sqrt(I) {worst:.3e}; sum |leak_i| |q'_i| {predicted:.3e} against the allowance 2 n eps sum m |r| (|v| + |v_c|) {:.3e}",
                modes.omega.len(),
                modes.omega.iter().fold(0.0f64, |a, &w| a.max(w)) / std::f64::consts::TAU,
                modes
                    .omega
                    .iter()
                    .filter(|&&w| w / std::f64::consts::TAU > 3000.0)
                    .count(),
                2.0 * n as f64 * f64::EPSILON * size
            );
        }
    });
}

/// Each flown frame of `tests/modal_flight.rs`'s drop against the step
/// replayed from the same start state: the particles' position difference
/// (largest and RMS, m) and the energy plus heat difference (J/m); and, at
/// each frame a flight hands back, which check refused it. A probe.
///
/// The reference is the step at the drop's tolerance, 1e-1, not an exact
/// solution: its own error is not separated from the flight's, and at that
/// tolerance a body is known to hop at the frame rate after impacts (the
/// Plasticine block, `tests/zener.rs`'s `probe_plasticine_block_drop`).
/// Measured: the first flight agrees to 1e-14 m; after the impacts the
/// flown frames differ by RMS 7e-5 to 1.4e-4 m per frame.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_flown_frames_against_the_step() {
    super::cap::capped("probe_flown_frames_against_the_step", 1800, || {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1).with_modal_flight());
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        let length = f64::from(config.dt);
        for k in 1..=45 {
            let before = sim.total_lagrangian.clone();
            sim.step();
            let frame = sim.total_lagrangian_frame().expect("on");
            let Some(start) = before else { continue };
            // A hand-back: which check the flight under way failed.
            if !frame.modal && frame.contacts == 0 {
                if let (Some(flight), Some(Some(modes))) = (&start.flight, &start.modes) {
                    let gravity = sim.total_lagrangian_gravity();
                    let planes = sim.total_lagrangian_walls().0;
                    let clear = flight.clear(gravity, &planes, (start.time, start.time + length));
                    let mut trial = start.clone();
                    let mut f = flight.clone();
                    let rebuilt = f.rebuild(modes, gravity, start.time + length, &mut trial);
                    println!(
                        "frame {k}: handed back with no contact: clear {clear}, rebuilt {rebuilt}, keeps L {}",
                        f.keeps_angular_momentum(&trial)
                    );
                } else {
                    println!("frame {k}: stepped with no contact, no flight under way");
                }
            }
            if !frame.modal {
                continue;
            }
            let flown = sim.total_lagrangian.take().expect("on");
            sim.total_lagrangian = Some(start);
            let mut stepped_config = config;
            stepped_config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
            sim.config = stepped_config;
            sim.step();
            let replay = sim.total_lagrangian_frame().expect("on");
            let stepped = sim.total_lagrangian.take().expect("on");
            sim.config = config;
            let n = flown.particles;
            let (mut largest, mut squared) = (0.0f64, 0.0f64);
            for p in 0..n {
                let d = (flown.x[p] - stepped.x[p]).length();
                largest = largest.max(d);
                squared += d * d;
            }
            println!(
                "frame {k}: flown against stepped from the same start: positions largest {largest:.3e} m, RMS {:.3e} m; energy+heat {:+.3e} J/m (flown {:.6e}, stepped {:.6e}); stepped contacts {}",
                (squared / n as f64).sqrt(),
                frame.energy + frame.heat - replay.energy - replay.heat,
                frame.energy + frame.heat,
                replay.energy + replay.heat,
                replay.contacts
            );
            sim.total_lagrangian = Some(flown);
        }
    });
}

/// The condition numbers of the step's Newton Jacobian and of its consistent
/// mass (`hooks::watch_condition`, infinity norm), per Newton iteration, on
/// E2's drop (the clay block of `tests/modal_flight.rs` dropped 10 cells,
/// tolerance 1e-1, 45 frames) and on the settled slab of
/// `tests/settled_slab.rs` (1 cm cells, 30 frames): their least, median and
/// largest, and the iterations sampled. Whether an inner solve in f32 is
/// possible: Carson and Higham's simple LU-IR needs `kappa_inf` up to about
/// 1e8 (agent report, `tmp/SCALE_DEMANDS.md`, not read here). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_tangent_condition_numbers() {
    super::cap::capped("probe_tangent_condition_numbers", 3000, || {
        let clay = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let e2 = || {
            let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
            let mut sim = Simulation::empty(config);
            let id = sim
                .register_material(Box::new(CorotatedMaterial::from_physical(&clay, &config)))
                .0;
            let floor = config.boundary_thickness as f32;
            let _block = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(8, 4))
                    .at(Vec2::new(20.0, floor + 12.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&clay, &config),
            );
            (sim, 45)
        };
        let slab = || {
            // `tests/settled_slab.rs`'s scene at 1 cm: E 1e5 Pa, nu 0,
            // 1000 kg/m^3, 0.30 m by 0.12 m on the floor, its dashpot 400 Pa s.
            let props = ElasticProps {
                e_pa: 1.0e5,
                nu: 0.0,
                rho_kg_m3: 1000.0,
            };
            let dx = 0.01f32;
            let mut config = SimConfig::earth(48, dx, 1.0 / 60.0);
            config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
            let mut material = CorotatedMaterial::from_physical(&props, &config);
            material.elastic_viscosity = config.visc_from_si(400.0);
            material.elastic_bulk_viscosity = config.visc_from_si(2.0 * 400.0 / 3.0);
            let mut sim = Simulation::empty(config);
            let id = sim.register_material(Box::new(material)).0;
            let floor = config.boundary_thickness as f32;
            let _slab = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(30, 12))
                    .at(Vec2::new(24.0, floor + 6.0))
                    .spacing(0.5)
                    .material(id)
                    .mass_from(&props, &config),
            );
            (sim, 30)
        };
        for (name, (mut sim, frames)) in [("E2 drop", e2()), ("settled slab", slab())] {
            crate::spacetime::total_lagrangian::hooks::watch_condition(400);
            for _ in 0..frames {
                sim.step();
            }
            let rows = crate::spacetime::total_lagrangian::hooks::take_condition();
            let stats = |pick: &dyn Fn(&(usize, f64, f64)) -> f64| -> (f64, f64, f64) {
                let mut v: Vec<f64> = rows.iter().map(pick).collect();
                v.sort_by(f64::total_cmp);
                (v[0], v[v.len() / 2], v[v.len() - 1])
            };
            let (j, m) = (stats(&|r| r.1), stats(&|r| r.2));
            println!(
                "{name}: {} Newton iterations sampled, {} to {} unknowns; Jacobian kappa_inf least {:.3e}, median {:.3e}, largest {:.3e}; mass kappa_inf least {:.3e}, median {:.3e}, largest {:.3e}",
                rows.len(),
                rows.iter().map(|r| r.0).min().unwrap_or(0),
                rows.iter().map(|r| r.0).max().unwrap_or(0),
                j.0,
                j.1,
                j.2,
                m.0,
                m.1,
                m.2
            );
        }
    });
}

/// The Norton block of `tests/norton.rs` (E 30 MPa, its stand-in density),
/// frame by frame: the largest equivalent plastic strain and the largest
/// per-frame equivalent plastic rate (`Delta alpha / frame`) over all
/// particles and over the bottom row (the contact particles); per frame the
/// particles whose rate is below 4e-4 1/s, inside, above 80 1/s, and those
/// under a tensile mean stress (`det F > 1`); and the distribution of the
/// return's iterations over every return the steps made. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_norton_block_flow_and_returns() {
    super::cap::capped("probe_norton_block_flow_and_returns", 3000, || {
        use crate::matter::materials::solid::viscoplastic_records::ViscoplasticRecord;
        let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
        let rho = 119.0 * 0.453_592_37 / (0.3048f64 * 0.3048 * 0.3048);
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
        let material = record.material((record.e_pa, rho), &config);
        let props = ElasticProps {
            e_pa: record.e_pa as f32,
            nu: record.nu as f32,
            rho_kg_m3: rho as f32,
        };
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        let frame = f64::from(config.dt);
        let alpha = |state: &TotalLagrangianState, p: usize| match state.solids[p].law {
            crate::spacetime::total_lagrangian::ElasticLaw::HenckyJ2(_, metric) => metric.alpha,
            _ => 0.0,
        };
        crate::matter::materials::solid::j2_hencky::watch_returns(true);
        crate::spacetime::total_lagrangian::hooks::watch_factors(true);
        let mut previous: Option<Vec<f64>> = None;
        let mut timing = crate::spacetime::total_lagrangian::Timing::default();
        let mut wall = 0.0;
        let mut mass_total = 0.0;
        for k in 1..=20 {
            sim.step();
            let state = sim.total_lagrangian.as_ref().expect("on the path");
            let n = state.particles;
            if mass_total == 0.0 {
                mass_total = state.mass[..n].iter().sum();
            }
            let now: Vec<f64> = (0..n).map(|p| alpha(state, p)).collect();
            let lowest = (0..n)
                .map(|p| state.reference[p].y)
                .fold(f64::INFINITY, f64::min);
            let bottom: Vec<usize> = (0..n)
                .filter(|&p| state.reference[p].y < lowest + 1e-9)
                .collect();
            let rates: Vec<f64> = match &previous {
                Some(before) => (0..n).map(|p| (now[p] - before[p]) / frame).collect(),
                None => now.iter().map(|a| a / frame).collect(),
            };
            let (below, above) = (
                rates.iter().filter(|&&r| r < 4e-4).count(),
                rates.iter().filter(|&&r| r > 80.0).count(),
            );
            let tension = (0..n).filter(|&p| state.f[p].determinant() > 1.0).count();
            let largest = |set: &mut dyn Iterator<Item = usize>, v: &[f64]| {
                set.map(|p| v[p]).fold(0.0f64, f64::max)
            };
            let f = sim.total_lagrangian_frame().expect("on");
            timing.add(&f.timing);
            let built = crate::spacetime::total_lagrangian::hooks::take_factors();
            let mut distinct = built.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            let ratios: Vec<f64> = built
                .windows(2)
                .map(|w| (w[1] / w[0]).max(w[0] / w[1]))
                .collect();
            let median_ratio = {
                let mut r = ratios.clone();
                r.sort_by(f64::total_cmp);
                r.get(r.len() / 2).copied().unwrap_or(1.0)
            };
            let tries = crate::spacetime::total_lagrangian::hooks::take_tries();
            println!(
                "  frame {k:2} tries (smooth/switching): kept {}/{}, rejected on the error {}/{}, unconverged {}/{}",
                tries[0], tries[1], tries[2], tries[3], tries[4], tries[5]
            );
            println!(
                "  frame {k:2} solver: {} kept steps, {} rejected tries, {} Newton, {} GMRES ({:.1} per Newton), {} factors built ({} distinct coefficients, median ratio between consecutive builds {median_ratio:.3}), {:.0} ms",
                f.steps,
                f.rejected,
                f.newton,
                f.gmres,
                f64::from(f.gmres) / f64::from(f.newton.max(1)),
                built.len(),
                distinct.len(),
                1e3 * f.wall
            );
            wall += f.wall;
            let vy = (0..n).map(|p| state.mass[p] * state.v[p].y).sum::<f64>() / mass_total;
            println!(
                "frame {k:2}: contacts {:2}, centre vy {vy:+.4} m/s; alpha max {:.3e} (bottom {:.3e}); frame rate max {:.3e} 1/s (bottom {:.3e}); particles: rate < 4e-4 {below}, in range {}, > 80 {above}; tensile {tension} of {n}; heat {:.4e} J/m",
                f.contacts,
                largest(&mut (0..n), &now),
                largest(&mut bottom.iter().copied(), &now),
                largest(&mut (0..n), &rates),
                largest(&mut bottom.iter().copied(), &rates),
                n - below - above,
                f.heat
            );
            previous = Some(now);
        }
        // Where the frames' time goes (the step's own clocks, summed over
        // the frames and the controller's tries).
        let share = |t: f64| 100.0 * t / wall;
        println!(
            "time over 20 frames: wall {wall:.1} s; residual {:.1} s ({:.1} %), linearisation (the law's tangents) {:.1} s ({:.1} %), Jacobian products {:.1} s ({:.1} %), preconditioner build {:.1} s + solve {:.1} s, GMRES in all {:.1} s ({:.1} %), contact {:.1} s, grid {:.1} s, controller {:.1} s; internal force evaluations {}",
            timing.residual,
            share(timing.residual),
            timing.linearisation,
            share(timing.linearisation),
            timing.jacobian,
            share(timing.jacobian),
            timing.preconditioner_build,
            timing.preconditioner_solve,
            timing.gmres,
            share(timing.gmres),
            timing.contact,
            timing.grid,
            timing.controller,
            timing.internal_evaluations
        );
        let split = crate::spacetime::total_lagrangian::hooks::take_return_split();
        // Taken before the timing below, whose calls would be recorded.
        let recorded = crate::matter::materials::solid::j2_hencky::take_returns();
        let kept_total = split[0] + split[1] + split[2];
        println!(
            "returns by try: kept pairs' whole steps {} ({:.1} %), kept halves {} ({:.1} %), rejected pairs {} ({:.1} %); per particle per frame {:.0}",
            split[0],
            100.0 * split[0] as f64 / kept_total.max(1) as f64,
            split[1],
            100.0 * split[1] as f64 / kept_total.max(1) as f64,
            split[2],
            100.0 * split[2] as f64 / kept_total.max(1) as f64,
            kept_total as f64 / (128.0 * 20.0)
        );
        // The cost of one return and of one law step, timed alone on one
        // thread: a flowing state of the impact (sigma_tr 3.4e5 Pa, dt 5e-4
        // s) and a strained C.
        {
            use crate::matter::materials::solid::j2_hencky::{
                HenckyJ2, PlasticMetric, norton_return,
            };
            let law = record.law;
            let shear = record.e_pa / (2.0 * (1.0 + record.nu));
            let calls = 1_000_000;
            let clock = std::time::Instant::now();
            let mut sink = 0.0;
            for k in 0..calls {
                sink += norton_return(3.4e5 * (1.0 + 1e-9 * k as f64), shear, &law, 5e-4).0;
            }
            let per_return = clock.elapsed().as_secs_f64() / calls as f64;
            let j2 = HenckyJ2 {
                shear,
                bulk: record.e_pa / (3.0 * (1.0 - 2.0 * record.nu)),
                yield_stress: 0.0,
                hardening: 0.0,
                viscosity: 0.0,
                norton: Some(law),
            };
            let (c0, c1) = (
                DMat2::from_cols(DVec2::new(1.002, 0.001), DVec2::new(0.001, 0.997)),
                DMat2::from_cols(DVec2::new(1.004, 0.002), DVec2::new(0.002, 0.995)),
            );
            let clock = std::time::Instant::now();
            for k in 0..calls {
                let c = c1 + DMat2::IDENTITY * (1e-12 * k as f64);
                sink += j2
                    .step(c0, c, &PlasticMetric::unstrained(), 5e-4)
                    .dissipated;
            }
            let per_step = clock.elapsed().as_secs_f64() / calls as f64;
            println!(
                "one Norton return {:.0} ns, one J2 law step with it {:.0} ns (one thread; checksum {sink:.3e})",
                per_return * 1e9,
                per_step * 1e9
            );
        }
        if let Some((counts, slowest)) = recorded {
            let total: u64 = counts.iter().sum();
            let quantile = |q: f64| {
                let target = (q * total as f64).ceil() as u64;
                let mut seen = 0;
                for (i, &c) in counts.iter().enumerate() {
                    seen += c;
                    if seen >= target {
                        return i;
                    }
                }
                counts.len() - 1
            };
            println!(
                "returns: {total}, iterations median {}, 99th percentile {}, 99.99th {}, max {}; the slowest: sigma_tr {:.4e} Pa, root d {:.4e}, dt {:.4e} s (rate d/dt {:.4e} 1/s)",
                quantile(0.5),
                quantile(0.99),
                quantile(0.9999),
                slowest.0,
                slowest.1,
                slowest.2,
                slowest.3,
                slowest.2 / slowest.3
            );
        }
    });
}

/// Whether the reused band factors (`solve`, `rho = 4`) change GMRES's
/// stopping point: the Norton block's impact frame and one flight frame,
/// every Newton linear solve whose factor was reused solved again on the
/// same state with its own exact factor; the iterations and the TRUE residual
/// `|b - J delta| / |b|` at exit for both, and for the solves that used their
/// own factor. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_factor_reuse_replay() {
    super::cap::capped("probe_factor_reuse_replay", 3000, || {
        use crate::matter::materials::solid::viscoplastic_records::ViscoplasticRecord;
        let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
        let rho = 119.0 * 0.453_592_37 / (0.3048f64 * 0.3048 * 0.3048);
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
        let props = ElasticProps {
            e_pa: record.e_pa as f32,
            nu: record.nu as f32,
            rho_kg_m3: rho as f32,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(record.material((record.e_pa, rho), &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        for _ in 0..9 {
            sim.step();
        }
        crate::spacetime::total_lagrangian::hooks::watch_gmres_replay(true);
        for _ in 0..2 {
            sim.step();
        }
        let (reused, own) =
            crate::spacetime::total_lagrangian::hooks::take_gmres_replay().expect("watched");
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let largest = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::max);
        let (ir, rr, ie, re): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = (
            reused.iter().map(|r| f64::from(r.0)).collect(),
            reused.iter().map(|r| r.1).collect(),
            reused.iter().map(|r| f64::from(r.2)).collect(),
            reused.iter().map(|r| r.3).collect(),
        );
        let (io, ro): (Vec<f64>, Vec<f64>) = (
            own.iter().map(|r| f64::from(r.0)).collect(),
            own.iter().map(|r| r.1).collect(),
        );
        println!(
            "reused-factor solves {}: iterations mean {:.2} (max {:.0}) against their own exact factor {:.2} (max {:.0}); true residual at exit, largest {:.3e} against {:.3e}, mean {:.3e} against {:.3e}",
            reused.len(),
            mean(&ir),
            largest(&ir),
            mean(&ie),
            largest(&ie),
            largest(&rr),
            largest(&re),
            mean(&rr),
            mean(&re)
        );
        println!(
            "own-factor solves {}: iterations mean {:.2} (max {:.0}); true residual at exit, largest {:.3e}, mean {:.3e}",
            own.len(),
            mean(&io),
            largest(&io),
            largest(&ro),
            mean(&ro)
        );
        let worse = reused.iter().filter(|r| r.0 > r.2).count();
        let better = reused.iter().filter(|r| r.0 < r.2).count();
        println!(
            "per solve: reused took more iterations in {worse}, fewer in {better}, as many in {}",
            reused.len() - worse - better
        );
    });
}

/// Why the whole step's band factor preconditions the halves better than
/// their own (`solve`'s reuse, measured by `probe_factor_reuse_replay`):
/// `kappa_inf(P(c)^-1 A)` against `kappa_inf(P(c/4)^-1 A)` on the dense
/// matrices, per Newton iteration, on the Norton block's impact and flight
/// frames and on E2's clay drop. Report only.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_preconditioned_condition() {
    super::cap::capped("probe_preconditioned_condition", 3000, || {
        use crate::matter::materials::solid::viscoplastic_records::ViscoplasticRecord;
        let block = |norton: bool| {
            let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
            config.total_lagrangian =
                Some(TotalLagrangianConfig::new(if norton { 1e-2 } else { 1e-1 }));
            let mut sim = Simulation::empty(config);
            let props = if norton {
                let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
                let rho = 119.0 * 0.453_592_37 / (0.3048f64 * 0.3048 * 0.3048);
                let id = sim
                    .register_material(Box::new(record.material((record.e_pa, rho), &config)))
                    .0;
                (
                    id,
                    ElasticProps {
                        e_pa: record.e_pa as f32,
                        nu: record.nu as f32,
                        rho_kg_m3: rho as f32,
                    },
                )
            } else {
                let clay = ElasticProps {
                    e_pa: 1.125e7,
                    nu: 0.3,
                    rho_kg_m3: 1700.0,
                };
                let id = sim
                    .register_material(Box::new(CorotatedMaterial::from_physical(&clay, &config)))
                    .0;
                (id, clay)
            };
            let floor = config.boundary_thickness as f32;
            let _block = sim.add_body(
                SpawnRegion::for_sim(&config)
                    .box_of(IVec2::new(8, 4))
                    .at(Vec2::new(20.0, floor + 12.0))
                    .spacing(0.5)
                    .material(props.0)
                    .mass_from(&props.1, &config),
            );
            sim
        };
        for (name, norton) in [("Norton block", true), ("E2 clay", false)] {
            let mut sim = block(norton);
            for _ in 0..9 {
                sim.step();
            }
            crate::spacetime::total_lagrangian::hooks::watch_preconditioned(60);
            for _ in 0..2 {
                sim.step();
            }
            let rows = crate::spacetime::total_lagrangian::hooks::take_preconditioned();
            let stats = |pick: &dyn Fn(&(f64, f64)) -> f64| {
                let mut v: Vec<f64> = rows.iter().map(pick).collect();
                v.sort_by(f64::total_cmp);
                (v[0], v[v.len() / 2], v[v.len() - 1])
            };
            let (own, quarter) = (stats(&|r| r.0), stats(&|r| r.1));
            let better = rows.iter().filter(|r| r.1 < r.0).count();
            println!(
                "{name}: {} Newton iterations; kappa_inf(P(c)^-1 A) least {:.3e}, median {:.3e}, largest {:.3e}; kappa_inf(P(c/4)^-1 A) least {:.3e}, median {:.3e}, largest {:.3e}; the quarter's smaller in {better}",
                rows.len(),
                own.0,
                own.1,
                own.2,
                quarter.0,
                quarter.1,
                quarter.2
            );
        }
    });
}

/// Which of the Norton block's tries build a band factor, one flight frame
/// (frame 11) traced (`hooks::set_control_trace`): each try's line and each
/// build's line with the nearest kept factor's coefficient ratio. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_factor_builds_traced() {
    super::cap::capped("probe_factor_builds_traced", 3000, || {
        use crate::matter::materials::solid::viscoplastic_records::ViscoplasticRecord;
        let record = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
        let rho = 119.0 * 0.453_592_37 / (0.3048f64 * 0.3048 * 0.3048);
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
        let props = ElasticProps {
            e_pa: record.e_pa as f32,
            nu: record.nu as f32,
            rho_kg_m3: rho as f32,
        };
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(record.material((record.e_pa, rho), &config)))
            .0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&props, &config),
        );
        for _ in 0..10 {
            sim.step();
        }
        crate::spacetime::total_lagrangian::hooks::set_control_trace(true);
        sim.step();
        crate::spacetime::total_lagrangian::hooks::set_control_trace(false);
        let f = sim.total_lagrangian_frame().expect("on");
        println!(
            "traced frame: {} kept steps, {} rejected tries",
            f.steps, f.rejected
        );
    });
}

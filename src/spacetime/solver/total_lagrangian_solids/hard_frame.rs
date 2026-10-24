//! E2's 4th impact frame (`tests/realtime_profile.rs`), which took 46.7 s
//! with the responses' warm start against 8.8 s without, run from one
//! identical state both ways. Equal costs mean the runs' states
//! had diverged by then (the trajectory's sensitivity); unequal costs mean
//! the warm start itself hurts there. A probe: it measures, and changes no
//! answer of the path.

use std::time::Instant;

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// `tests/realtime_profile.rs`'s E2: the 8 x 4 cm clay block dropped 10
/// cells, frame 1/60 s, step tolerance 1e-1.
pub(super) fn e2() -> Simulation {
    e2_lifted(0.0)
}

/// E2 with its lift raised by `offset` cells.
fn e2_lifted(offset: f32) -> Simulation {
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 10.0 + 2.0 + offset))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    sim
}

/// The responses' warm start on or off (`hooks::set_cold_responses`).
fn warm_start(on: bool) {
    crate::spacetime::total_lagrangian::hooks::set_cold_responses(!on);
}

/// One frame from `state`, its wall time and its counts.
fn frame_from(sim: &mut Simulation, state: &TotalLagrangianState, on: bool) -> String {
    sim.total_lagrangian = Some(state.clone());
    warm_start(on);
    let clock = Instant::now();
    sim.step();
    let wall = clock.elapsed().as_secs_f64();
    let f = sim.total_lagrangian_frame().expect("on the path");
    format!(
        "warm start {}: {wall:.2} s; steps {}, rejected {}, Newton {}, GMRES {}, passes {}, responses {:.2} s",
        if on { "on " } else { "off" },
        f.steps,
        f.rejected,
        f.newton,
        f.gmres,
        f.passes,
        f.timing.responses
    )
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_hard_frame_from_one_state() {
    super::cap::capped("probe_e2_hard_frame_from_one_state", 1800, || {
        // The frame of E2's 4th impact in the profile's frame list (1-based).
        const FRAME: usize = 44;
        for (label, on) in [
            ("the run without warm start", false),
            ("the run with it", true),
        ] {
            let mut sim = e2();
            warm_start(on);
            for _ in 1..FRAME {
                sim.step();
            }
            let state = sim.total_lagrangian.as_ref().expect("on the path").clone();
            println!("frame {FRAME} from the state {label} reached:");
            for second in [false, true] {
                println!("  {}", frame_from(&mut sim, &state, second));
            }
        }
        warm_start(true);
    });
}

/// What the controller does in that frame, from the state the warm-start
/// run reached: its counters by cause (`TotalLagrangianFrame`), and every
/// try printed (`hooks::set_control_trace`): which branch rejects and whether the step-ratio clamp or the proposal
/// after a rejection streak sets the count. Report only.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_hard_frame_controller() {
    super::cap::capped("probe_e2_hard_frame_controller", 1800, || {
        const FRAME: usize = 44;
        let mut sim = e2();
        warm_start(true);
        for _ in 1..FRAME {
            sim.step();
        }
        crate::spacetime::total_lagrangian::hooks::set_control_trace(true);
        sim.step();
        crate::spacetime::total_lagrangian::hooks::set_control_trace(false);
        let f = sim.total_lagrangian_frame().expect("on the path");
        println!(
            "frame {FRAME}: kept {}, rejected {} (Newton unconverged {}), at the estimator's floor {}, reaching {}, passes {}, unconverged kept {}, stagnated {}",
            f.steps,
            f.rejected,
            f.rejected_unconverged,
            f.at_floor,
            f.reaching,
            f.passes,
            f.unconverged,
            f.stagnated
        );
    });
}

/// A block at rest on the floor (`tests/realtime_profile.rs`'s E1): why its
/// frames still build Jacobian factors, each build printed with what was
/// new (the step's inertia coefficient, or the held springs).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rest_factorizations() {
    super::cap::capped("probe_rest_factorizations", 1800, || {
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
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
        for k in 1..=6 {
            crate::spacetime::total_lagrangian::hooks::set_control_trace(k > 4);
            sim.step();
            let f = sim.total_lagrangian_frame().expect("on the path");
            println!(
                "frame {k}: steps {}, rejected {}, passes {}, factorizations {}",
                f.steps, f.rejected, f.passes, f.timing.factorizations
            );
        }
        crate::spacetime::total_lagrangian::hooks::set_control_trace(false);
    });
}

/// The E1 block spawned undeformed on the floor, at a fine frame: whether its
/// undamped ringing keeps the floor contact held (a resting body) or lifts
/// it periodically. A probe for the resting-body design.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rest_contact_persistence() {
    super::cap::capped("probe_rest_contact_persistence", 1800, || {
        let props = ElasticProps {
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        };
        let frame = (1.0 / 60.0 / 1024.0) as f32;
        let mut config = SimConfig::earth(40, 0.01, frame);
        config.min_dt = config.min_dt.min(frame);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
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
        let (mut held, mut free, mut switches, mut last) = (0, 0, 0, None::<bool>);
        let mut line = String::new();
        for k in 1..=1024 {
            sim.step();
            let on = sim.total_lagrangian_frame().expect("on the path").contacts > 0;
            if on {
                held += 1
            } else {
                free += 1
            }
            if last.is_some_and(|l| l != on) {
                switches += 1;
            }
            last = Some(on);
            if k <= 400 {
                line.push(if on { '#' } else { '.' });
            }
        }
        println!("first 400 frames of 16.3 us (# contact held, . none): {line}");
        println!("over 1024 frames (16.7 ms): held {held}, none {free}, switches {switches}");
    });
}

/// E1's block placed at its resting compression (`rest::Rest::settle`),
/// optionally with its first constrained mode kicked, then run stepped and
/// with modal rest from the same state: whether the rest holds, and what
/// the modes carry against what the step resolves. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_settled_rest() {
    super::cap::capped("probe_settled_rest", 900, settled_rest);
}

fn settled_rest() {
    // `EMERGE_PROBE_NO_RESOLUTION_FLOOR` runs the walls without their
    // resolution floor (`contact::gap_resolution`), as before it: the
    // stepper then never settles a contact whose points start at rest.
    println!(
        "resolution floor {}",
        std::env::var_os("EMERGE_PROBE_NO_RESOLUTION_FLOOR").is_none()
    );
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let build_at = |modal: bool, kick: f64, tolerance: f64| -> Simulation {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        // Built at one tolerance, the placement is the same for every run;
        // the run's own tolerance is set after.
        let path = TotalLagrangianConfig::new(1e-1);
        config.total_lagrangian = Some(if modal { path.with_modal_rest() } else { path });
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
        for _ in 0..2 {
            sim.step();
        }
        let planes = sim.total_lagrangian_walls().0;
        let gravity = sim.total_lagrangian_gravity();
        let mut state = sim.total_lagrangian.take().expect("on the path");
        let cache = state.cache.as_ref().expect("stepped");
        let basis =
            super::rest::Basis::build(&state, (&cache.grid, &planes, 0.0)).expect("a basis");
        let Ok(rest) = super::rest::Rest::start(
            basis,
            &state,
            (&cache.grid, &planes),
            (gravity, state.time, f64::from(sim.config().dt)),
        ) else {
            panic!("a compressive start");
        };
        rest.settle(kick, &mut state);
        sim.total_lagrangian = Some(state);
        if let Some(path) = sim.config.total_lagrangian.as_mut() {
            path.tolerance = tolerance;
        }
        sim
    };
    let build = |modal: bool, kick: f64| build_at(modal, kick, 1e-1);
    for kick in [0.1, 0.5, 0.0] {
        let (mut stepped, mut modal) = (build(false, kick), build(true, kick));
        let (mut stepped_wall, mut modal_wall, mut apart) = (0.0f64, 0.0f64, 0.0f64);
        let mut start = (0.0, 0.0);
        for k in 1..=60 {
            stepped.step();
            modal.step();
            let (s, m) = (
                stepped.total_lagrangian_frame().expect("on"),
                modal.total_lagrangian_frame().expect("on"),
            );
            println!(
                "  kick {kick} frame {k}: stepped {} steps {} rejected {:.2} s; modal {} steps {} rejected {:.3} s{}",
                s.steps,
                s.rejected,
                s.wall,
                m.steps,
                m.rejected,
                m.wall,
                if m.modal_rest { " rest" } else { "" }
            );
            if k == 1 {
                start = (s.energy + s.heat, m.energy + m.heat);
            }
            stepped_wall += s.wall;
            modal_wall += m.wall;
            let (a, b) = (
                stepped.total_lagrangian.as_ref().expect("on"),
                modal.total_lagrangian.as_ref().expect("on"),
            );
            apart =
                a.x.iter()
                    .zip(&b.x)
                    .map(|(p, q)| (*p - *q).length())
                    .fold(apart, f64::max);
        }
        let (fs, fm) = (
            stepped.total_lagrangian_frame().expect("on"),
            modal.total_lagrangian_frame().expect("on"),
        );
        println!(
            "kick {kick}: modal rest {} frames, {} hand-backs; positions apart at most {:.3e} m; energy+heat change over frames 1-60 stepped {:+.3e}, modal {:+.3e} J/m (frame 1: stepped {:.6e}, modal {:.6e}); stored stepped {:.4e}, modal {:.4e}; wall stepped {:.2} s, modal {:.3} s",
            fm.modal_rest_frames,
            fm.rest_hand_backs,
            apart,
            fs.energy + fs.heat - start.0,
            fm.energy + fm.heat - start.1,
            start.0,
            start.1,
            fs.parts[1],
            fm.parts[1],
            stepped_wall,
            modal_wall
        );
    }
    // The kicked rest against finer stepped references: the coarse one
    // takes one step per frame, too long to carry the vibration's phase.
    let frames = 10;
    let mut modal = build(true, 0.1);
    let mut finer: Vec<(f64, Simulation)> = [1e-3, 1e-5]
        .into_iter()
        .map(|t| (t, build_at(false, 0.1, t)))
        .collect();
    for k in 1..=frames {
        modal.step();
        let m = modal.total_lagrangian.as_ref().expect("on");
        let rest = modal.total_lagrangian_frame().expect("on").modal_rest;
        for (tolerance, sim) in finer.iter_mut() {
            sim.step();
            let frame = sim.total_lagrangian_frame().expect("on");
            let s = sim.total_lagrangian.as_ref().expect("on");
            let apart =
                s.x.iter()
                    .zip(&m.x)
                    .map(|(p, q)| (*p - *q).length())
                    .fold(0.0, f64::max);
            println!(
                "  kick 0.1 frame {k}: tolerance {tolerance:.0e} {} steps, {:.2} s, stored {:.6e}; modal{} stored {:.6e}; positions apart {apart:.3e} m",
                frame.steps,
                frame.wall,
                frame.parts[1],
                if rest { " rest" } else { "" },
                modal.total_lagrangian_frame().expect("on").parts[1]
            );
        }
    }
}

/// E2's rejections by cause, with the walls' resolution floor and without
/// it (`hooks::set_no_resolution_floor`): per frame with any rejection, the
/// tries rejected unconverged (Newton's or the walls' passes) and those
/// rejected on the error estimate, then the run's totals. Report only.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_rejections_by_cause() {
    super::cap::capped("probe_e2_rejections_by_cause", 3600, || {
        const FRAMES: usize = 50;
        for floored in [false, true] {
            crate::spacetime::total_lagrangian::hooks::set_no_resolution_floor(!floored);
            let mut sim = e2();
            let (mut kept, mut unconverged, mut error, mut wall) = (0u32, 0u32, 0u32, 0.0f64);
            for k in 1..=FRAMES {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("on the path");
                kept += f.steps;
                unconverged += f.rejected_unconverged;
                error += f.rejected.saturating_sub(f.rejected_unconverged);
                wall += f.wall;
                if f.rejected > 0 {
                    println!(
                        "  floor {floored}, frame {k}: kept {}, rejected unconverged {}, on the error {}, passes {}, {:.2} s",
                        f.steps,
                        f.rejected_unconverged,
                        f.rejected.saturating_sub(f.rejected_unconverged),
                        f.passes,
                        f.wall
                    );
                }
            }
            println!(
                "floor {floored}, {FRAMES} frames: kept {kept}, rejected unconverged {unconverged}, on the error {error}, wall {wall:.1} s"
            );
        }
        crate::spacetime::total_lagrangian::hooks::set_no_resolution_floor(false);
    });
}

/// Whether E2's frames of hundreds of rejections (543 in one smoke run of
/// the live harness, none in this tree's E2) are a property of the ringing
/// block on the floor, met on some trajectories: E2 rerun from lifts a few
/// f32 ulps apart (`k` x 1e-6 cell; the ulp near 12 cells is 9.5e-7), per
/// run the most rejections in a frame and how the frames spread over
/// rejection counts. Counts only.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_e2_rejection_tail() {
    super::cap::capped("probe_e2_rejection_tail", 10800, || {
        const FRAMES: usize = 45;
        let bins = [0u32, 10, 50, 100, 300];
        for k in 0..6 {
            let mut sim = e2_lifted(k as f32 * 1e-6);
            let mut spread = [0u32; 6];
            let (mut most, mut at, mut kept, mut rejected, mut unconverged) =
                (0u32, 0usize, 0, 0, 0);
            for frame in 1..=FRAMES {
                sim.step();
                let f = sim.total_lagrangian_frame().expect("on the path");
                kept += f.steps;
                rejected += f.rejected;
                unconverged += f.rejected_unconverged;
                if f.rejected > most {
                    (most, at) = (f.rejected, frame);
                }
                spread[bins.iter().filter(|&&b| f.rejected > b).count()] += 1;
                if f.rejected > 100 {
                    println!(
                        "  lift +{k}e-6: frame {frame} rejected {} ({} unconverged), kept {}, {:.1} s",
                        f.rejected, f.rejected_unconverged, f.steps, f.wall
                    );
                }
            }
            println!(
                "lift +{k}e-6 cell, {FRAMES} frames: kept {kept}, rejected {rejected} ({unconverged} unconverged); most in a frame {most} (frame {at}); frames with 0 / 1-10 / 11-50 / 51-100 / 101-300 / over 300 rejections: {spread:?}"
            );
        }
    });
}

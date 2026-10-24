//! Damped modal rest (`rest`, with relaxation branches): the Plasticine
//! record on a block small enough to rest inside its linear limit, and the
//! E1 block, which does not.
//!
//! The 1 cm block is a test geometry for the algorithm, forced small by the
//! record's validity: a block of ordinary size at rest under its own weight
//! is beyond the record's limit (4 cm: a static strain of 3.7e-5 against
//! 1e-5), a declared gap whose removal needs a loss factor measured under
//! static prestrain.

use super::rest::DampedModes;
use super::*;
use crate::matter::materials::relaxation::ViscoelasticRecord;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// A Plasticine block `height` cells high (16 cells wide), cell `dx`, on
/// the floor, placed at its resting compression (`rest::Rest::settle`) with
/// `kick` of the loaded mode's offset as its rate; modal rest on or off.
fn settled_plasticine(dx: f32, height: i32, kick: f64, modal: bool) -> Simulation {
    settled_plasticine_at(dx, height, kick, (modal, 1e-2))
}

/// The same at step tolerance `tolerance`.
fn settled_plasticine_at(
    dx: f32,
    height: i32,
    kick: f64,
    (modal, tolerance): (bool, f64),
) -> Simulation {
    let record = ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut config = SimConfig::earth(64, dx, 1.0 / 60.0);
    let path = TotalLagrangianConfig::new(tolerance);
    config.total_lagrangian = Some(if modal { path.with_modal_rest() } else { path });
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(16, height))
            .at(Vec2::new(32.0, floor + 0.5 * height as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    // Stepped 30 frames: spawned on the floor, it first chatters, and its
    // own damping settles it; then until its last step holds a contact.
    for k in 0..60 {
        sim.step();
        let state = sim.total_lagrangian.as_ref().expect("on the path");
        if k >= 30 && state.cache.as_ref().is_some_and(|c| !c.contacts.is_empty()) {
            break;
        }
    }
    let planes = sim.total_lagrangian_walls().0;
    let gravity = sim.total_lagrangian_gravity();
    let length = f64::from(config.dt);
    let mut state = sim.total_lagrangian.take().expect("on the path");
    let cache = state.cache.as_ref().expect("stepped");
    let basis = rest::Basis::build(&state, (&cache.grid, &planes, 0.0)).expect("a basis");
    let rest = match rest::Rest::start_for_settling(
        basis,
        &state,
        (&cache.grid, &planes),
        (gravity, state.time, length),
    ) {
        Ok(rest) => rest,
        Err(refused) => panic!(
            "the block does not start a rest (beyond the limit: {})",
            refused.beyond_limit
        ),
    };
    rest.settle(kick, &mut state);
    sim.total_lagrangian = Some(state);
    sim
}

/// One run: the frame before it (the counters' baseline), then per frame
/// the frame record and the particles' positions (m), and the rest's
/// damped modes after the first rest frame (`Rest::damped_modes`).
type Frames = Vec<(TotalLagrangianFrame, Vec<DVec2>)>;
type Run = (TotalLagrangianFrame, Frames, Option<DampedModes>);

fn run(mut sim: Simulation, frames: usize) -> Run {
    let before = sim.total_lagrangian_frame().expect("on the path");
    let mut modes = None;
    let rows = (0..frames)
        .map(|_| {
            sim.step();
            let state = sim.total_lagrangian.as_ref().expect("on the path");
            let frame = sim.total_lagrangian_frame().expect("on the path");
            if frame.modal_rest && modes.is_none() {
                modes = state.rest.as_ref().and_then(|r| r.damped_modes());
            }
            (frame, state.x[..state.particles].to_vec())
        })
        .collect();
    (before, rows, modes)
}

/// `y' = A y` for a damped mode (`damped_propagator`'s `A`), integrated by
/// the classical fourth-order Runge-Kutta method in `steps` steps: the
/// exact constrained mode, independently of the matrix exponential.
fn integrated(omega: f64, memory: &[(f64, f64)], length: f64, y: &[f64], steps: usize) -> Vec<f64> {
    let n = y.len();
    let derivative = |y: &[f64]| -> Vec<f64> {
        let mut d = vec![0.0; n];
        d[0] = omega * y[1];
        d[1] = -omega * y[0];
        for (l, &(theta, kappa)) in memory.iter().enumerate() {
            d[1] -= omega * kappa.sqrt() * y[2 + l];
            d[2 + l] = omega * kappa.sqrt() * y[1] - theta * y[2 + l];
        }
        d
    };
    let h = length / steps as f64;
    let mut y = y.to_vec();
    for _ in 0..steps {
        let at = |y: &[f64], d: &[f64], s: f64| -> Vec<f64> {
            y.iter().zip(d).map(|(a, b)| a + s * b).collect()
        };
        let k1 = derivative(&y);
        let k2 = derivative(&at(&y, &k1, 0.5 * h));
        let k3 = derivative(&at(&y, &k2, 0.5 * h));
        let k4 = derivative(&at(&y, &k3, h));
        for e in 0..n {
            y[e] += h / 6.0 * (k1[e] + 2.0 * k2[e] + 2.0 * k3[e] + k4[e]);
        }
    }
    y
}

/// The balance within the solves' bound plus each modal frame's
/// linearisation residual and the sums' round-off, every frame
/// (`tests/modal_flight.rs`'s criterion); no frame's heat negative.
fn check_ledger(name: &str, frames: &[(TotalLagrangianFrame, Vec<DVec2>)]) {
    let mut allowed = 0.0;
    for (k, (f, x)) in frames.iter().enumerate() {
        allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
        let round_off = x.len() as f64
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        assert!(
            f.balance.abs() <= allowed + round_off,
            "{name}, frame {}: balance {:e} beyond {:e}",
            k + 1,
            f.balance,
            allowed + round_off
        );
        assert!(f.relaxation_heat >= 0.0, "{name}: negative heat");
    }
}

/// Gates 3 and 5 (`DAMPED_REST_GATES`): the 1 cm block settled, kick 0,
/// rested against stepped over 30 frames. Criteria fixed before measuring:
/// every frame after the two the rest needs is rested (from the settled
/// state), no refusal, no
/// propagator check failing, the ledger closes; the difference to the
/// stepped run (positions, energy plus heat) is reported.
#[test]
fn a_damped_block_rests_by_its_constrained_modes() {
    const FRAMES: usize = 30;
    let (base, rested, _) = run(settled_plasticine(0.0025, 2, 0.0, true), FRAMES);
    let (_, stepped, _) = run(settled_plasticine(0.0025, 2, 0.0, false), FRAMES);
    let last = rested.last().expect("frames").0;
    let (rest_frames, hand_backs, refused) = (
        last.modal_rest_frames - base.modal_rest_frames,
        last.rest_hand_backs - base.rest_hand_backs,
        last.modal_rest_refused - base.modal_rest_refused,
    );
    let (mut largest, mut energy) = (0.0f64, 0.0f64);
    for ((fr, xr), (fs, xs)) in rested.iter().zip(&stepped) {
        for (a, b) in xr.iter().zip(xs) {
            largest = largest.max((*a - *b).length());
        }
        energy = energy.max((fr.energy + fr.heat - fs.energy - fs.heat).abs());
    }
    println!(
        "damped rest, kick 0: {} rest frames of {FRAMES}, {} hand-backs, {} refused, {} checks failed; against stepped: positions at most {largest:.3e} m apart, energy plus heat at most {energy:.3e} J/m apart; heat {:.3e} (rested) {:.3e} (stepped) J/m",
        rest_frames,
        hand_backs,
        refused,
        last.modal_check_failed,
        last.heat,
        stepped.last().expect("frames").0.heat
    );
    check_ledger("damped rest", &rested);
    assert_eq!((refused, hand_backs), (0, 0));
    assert_eq!(last.modal_check_failed, 0);
    assert!(
        rest_frames as usize >= FRAMES - 2,
        "{rest_frames} rest frames"
    );
}

/// The kicked block (0.1 of the loaded offset): its first rest frame, and
/// the most damped mode's state over it (`Rest::damped_modes`); the heat it
/// made, J/m, and the particles' positions at its end.
fn kicked_rest() -> (f64, Vec<DVec2>, DampedModes, Frames) {
    let (base, rested, modes) = run(settled_plasticine(0.0025, 2, 0.1, true), 6);
    assert!(rested[0].0.modal_rest, "the first frame rests");
    (
        rested[0].0.heat - base.heat,
        rested[0].1.clone(),
        modes.expect("a damped rest"),
        rested,
    )
}

/// Gate 4: kicked (0.1 of the loaded offset), criteria fixed before
/// measuring:
/// - the mode that lost the most energy in the first rest frame against
///   the exact constrained mode from the same start (integrated apart by
///   Runge-Kutta, `integrated`), its energy lost within 2 %;
/// - the rest's heat over that frame against a stepped damped run that
///   resolves the loaded mode, within 5 %: the same 1/60 s covered by
///   frames of 1/6000 s from the same settled state. The positions at the
///   interval's end are reported.
///
/// Why the resolved reference: the frame-rate stepped run is not a
/// reference here (`probe_kicked_rest_heat_against_the_step_tolerance` and
/// the ignored `a_kicked_damped_block_rests_against_the_frame_rate_step`):
/// the kick's energy is about 4.5e-8 of the step tolerance's energy, so
/// the controller keeps one step per frame at any tolerance, and the loaded
/// mode (about 8.8e3 rad/s, `omega dt` near 147) is not resolved. The
/// exact-mode agreement above is the independent check that the rest is
/// right; the finer reference is the better approximation of the true
/// solution (the tolerance-scale gap, as in `tests/zener.rs`'s
/// `probe_plasticine_block_drop`).
#[test]
fn a_kicked_damped_block_rests_and_its_kick_becomes_heat() {
    let (h_rest, x_rest, (modes, memory), rested) = kicked_rest();
    let energy = |y: &[f64]| 0.5 * y.iter().map(|e| e * e).sum::<f64>();
    let (omega, y0, y1) = modes
        .iter()
        .max_by(|a, b| (energy(&a.1) - energy(&a.2)).total_cmp(&(energy(&b.1) - energy(&b.2))))
        .expect("modes");
    let exact = integrated(*omega, &memory, 1.0 / 60.0, y0, 400_000);
    let (lost, lost_exact) = (energy(y0) - energy(y1), energy(y0) - energy(&exact));
    // The resolved reference: settled as the rest's run, then 100 frames of
    // 1/6000 s.
    let mut sim = settled_plasticine(0.0025, 2, 0.1, false);
    sim.config.dt = 1.0 / 6000.0;
    let base = sim.total_lagrangian_frame().expect("on the path").heat;
    let (mut steps, mut fine) = (0u32, None);
    for _ in 0..100 {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("on the path");
        steps += f.steps;
        fine = Some(f);
    }
    let h_fine = fine.expect("frames").heat - base;
    let state = sim.total_lagrangian.as_ref().expect("on the path");
    let apart = x_rest
        .iter()
        .zip(&state.x)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0f64, f64::max);
    let period = std::f64::consts::TAU / omega;
    println!(
        "kicked damped rest: the most damped mode (omega {omega:.4e} rad/s) lost {lost:.6e} J/m against the exact {lost_exact:.6e} ({:.6}); the rest's heat {h_rest:.4e} J/m against the resolved step's {h_fine:.4e} ({:.4}), {steps} steps over 1/60 s ({:.1} per period of that mode); positions at most {apart:.3e} m apart",
        lost / lost_exact,
        h_rest / h_fine,
        f64::from(steps) * period * 60.0
    );
    check_ledger("kicked damped rest", &rested);
    assert!((lost / lost_exact - 1.0).abs() < 0.02);
    assert!(
        (h_rest / h_fine - 1.0).abs() < 0.05,
        "heat {h_rest:e} against the resolved {h_fine:e}"
    );
}

/// The gate as first written, kept as evidence: the rest's heat over its
/// first frame against the stepped damped run's over the same frame at the
/// frame rate, within 5 %. It fails under the tolerance-scale gap
/// (measured: the step's heat 7.0e-11, 1.2e-11 and 3.4e-14 J/m at
/// tolerances 1e-2, 1e-3 and 1e-4, one step per frame each, against the
/// rest's 8.452e-10, which the exact constrained mode confirms): the kick's
/// energy is about 4.5e-8 of the tolerance's energy and the loaded mode's
/// `omega dt` is near 147. Removal criterion: the tolerance's reference
/// scale resolved (option 3 of the gap), after which this gate must pass.
#[test]
#[ignore = "fails under the tolerance-scale gap: the frame-rate step does not resolve the kicked mode"]
fn a_kicked_damped_block_rests_against_the_frame_rate_step() {
    let (h_rest, _, _, _) = kicked_rest();
    let (base, stepped, _) = run(settled_plasticine(0.0025, 2, 0.1, false), 1);
    let h_step = stepped[0].0.heat - base.heat;
    println!("rest heat {h_rest:.4e} J/m against the frame-rate step's {h_step:.4e}");
    assert!(
        (h_rest / h_step - 1.0).abs() < 0.05,
        "heat {h_rest:e} against {h_step:e}"
    );
}

/// Gate 2: E1's Plasticine block (8 by 4 cells of 1 cm on the floor) at
/// rest is beyond the record's limit under its own weight: never rested,
/// each refusal counted.
#[test]
fn an_ordinary_plasticine_block_at_rest_is_refused_and_counted() {
    let record = ViscoelasticRecord::plasticine_crandall_1970_fresh();
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_rest());
    let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
    material.relaxation = Some(record.spectrum);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&record.relaxed, &config),
    );
    for _ in 0..8 {
        sim.step();
    }
    let f = sim.total_lagrangian_frame().expect("on the path");
    println!(
        "E1 Plasticine block at rest: {} rest frames, {} refused",
        f.modal_rest_frames, f.modal_rest_refused
    );
    assert_eq!(f.modal_rest_frames, 0);
    assert!(f.modal_rest_refused > 0, "the refusals are counted");
}

/// The kicked block's first frame stepped at tolerances 1e-2, 1e-3 and
/// 1e-4 from the settled state: its heat and steps against the rest's
/// (the exact constrained modes'). Whether the stepped heat's shortfall
/// is the step's (the loaded mode, near 8.8e3 rad/s, is not resolved at
/// one step per frame) or the rest's. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_kicked_rest_heat_against_the_step_tolerance() {
    let (base, rested, _) = run(settled_plasticine(0.0025, 2, 0.1, true), 1);
    println!("rest: heat {:.4e} J/m", rested[0].0.heat - base.heat);
    for tolerance in [1e-2, 1e-3, 1e-4] {
        let (base, stepped, _) = run(settled_plasticine_at(0.0025, 2, 0.1, (false, tolerance)), 1);
        println!(
            "stepped at {tolerance:.0e}: heat {:.4e} J/m in {} steps ({} rejected)",
            stepped[0].0.heat - base.heat,
            stepped[0].0.steps,
            stepped[0].0.rejected
        );
    }
}

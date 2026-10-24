//! The rod gates: a Zener rod dropped on its end against the exact
//! restitution, below and in the record's band, and its probe against the
//! step tolerance.

use super::*;

/// The impact scene: the `nu = 0` rod standing on its end, in zero
/// gravity, moving at the scene's speed onto the floor from a gap of 1e-4
/// cell. Gravity is left out: over a slow rod's contact (64 ms at 7.81 Hz)
/// its impulse would dwarf an impact slow enough for the record's linear
/// range.
struct Drop {
    /// The centre of mass's speed over the impact's, just after the first
    /// separation and at the run's end.
    first: f64,
    end: f64,
    /// At the first separation, over the start (all kinetic) energy: the
    /// rigid rebound (`first^2`), the rod's vibration (its kinetic and
    /// stored energy beyond the rebound's) and the heat made.
    split: (f64, f64, f64),
    /// The kept steps over the run.
    steps: u32,
    ledger: Ledger,
}

fn drop_rod_at(rod: &Rod, relaxation: Option<RelaxationSpectrum>, frames: usize) -> Drop {
    let props = rod.props();
    let config = rod.config();
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let l = rod.cells(0.32) as f32;
    let _rod = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(rod.cells(0.02), rod.cells(0.32)))
            .at(Vec2::new(rod.cells(0.24) as f32, floor + 1e-4 + 0.5 * l))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let impact = rod.speed_m_s / rod.dx;
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(0.0, -impact);
    }
    let speed = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        p.v.iter()
            .zip(&p.mass)
            .map(|(v, &m)| f64::from(v.y) * f64::from(m))
            .sum::<f64>()
            / mass
            / f64::from(impact)
    };
    let mut ledger = Ledger::default();
    let (mut touched, mut first, mut steps) = (false, None, 0u32);
    for _ in 0..frames {
        sim.step();
        ledger.frame(&sim);
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        steps += frame.steps;
        if frame.contacts > 0 {
            touched = true;
        } else if touched && first.is_none() {
            let e = speed(&sim);
            let start = frame.start_energy;
            first = Some((
                e,
                (
                    e * e,
                    (frame.parts[0] + frame.parts[1]) / start - e * e,
                    frame.heat / start,
                ),
            ));
        }
    }
    let (first, split) = first.expect("the rod leaves the floor within the run");
    Drop {
        first,
        end: speed(&sim),
        split,
        steps,
        ledger,
    }
}

/// The exact Zener rod on a rigid wall (`common/zener.rs`, 1024 modes) at a
/// rod's own elastic speed of sound `c`.
fn exact_rod(record: &RelaxationSpectrum, rod: &Rod, c: f64) -> zener::RodContact {
    zener::rod_on_wall(
        &zener::Zener {
            rho: f64::from(RHO),
            relaxed: f64::from(RHO) * c * c,
            kappa: record.mechanisms().iter().map(|m| m.kappa_shear).collect(),
            theta: record.mechanisms().iter().map(|m| m.theta).collect(),
        },
        f64::from(rod.cells(0.32) as f32 * rod.dx),
        f64::from(rod.speed_m_s),
        1024,
    )
}

/// One rod at one scale: its elastic and damped drops against the exact
/// restitution and energy split, printed; returns (damped restitution over
/// the exact, the damped drop).
fn rod_row(name: &str, rod: &Rod, frames: usize) -> (f64, Drop) {
    let c = rod.sound_speed();
    let elastic = drop_rod_at(rod, None, frames);
    let record = RelaxationSpectrum::plasticine_crandall_1970();
    let damped = drop_rod_at(rod, Some(record), frames);
    let exact = exact_rod(&record, rod, c);
    let length = f64::from(rod.cells(0.32) as f32 * rod.dx);
    println!(
        "{name} (cell {} m, quarter-wave {:.2} Hz, c {c:.4} m/s): elastic leaves with {:.5} (end {:.5}); Plasticine leaves with {:.5} (end {:.5}) against the exact {:.5} (ratio {:.4}); split rebound / vibration / heat {:.4} / {:.4} / {:.4} against the exact {:.4} / {:.4} / {:.4}; damped {}",
        rod.dx,
        c / (4.0 * length),
        elastic.first,
        elastic.end,
        damped.first,
        damped.end,
        exact.restitution,
        damped.first / exact.restitution,
        damped.split.0,
        damped.split.1,
        damped.split.2,
        exact.split.0,
        exact.split.1,
        exact.split.2,
        damped.ledger.summary()
    );
    (damped.first / exact.restitution, damped)
}

/// Below the record's band: the rod's quarter-wave at 7.81 Hz is twelve
/// times below the 100-3000 Hz its spectrum was fitted over. This gate
/// tests the discretisation against the same record, not the record's
/// physics in its band (the in-band gate below). Criterion fixed before
/// measuring: within 5 % at 1 cm cells; the half cell reported as the
/// convergence check.
///
/// Measured (step tolerance 1e-2):
/// - 1 cm cells: elastic 0.97858, damped 0.84177 against the exact
///   0.83858 (1.0038);
/// - 0.5 cm cells: elastic 0.98496, damped 0.84211 (1.0042).
///
/// The elastic loss shrinks with the cell; the damped excess does not move
/// with it.
///
/// With the step tolerance instead (`probe_rod_restitution_against_the_step_tolerance`,
/// 1 cm cells), the damped excess falls toward zero: 1.0038, 1.0025, 1.0014 at
/// 1e-2, 1e-3, 1e-4. The elastic rod is not monotone there: 0.97858, 0.98624,
/// 0.98563.
///
/// Declared gap: the energy split is read at the end of the frame after the
/// separation, the exact one at the separation itself, and the branches keep
/// turning vibration into heat in flight, so the two splits are not
/// comparable. Only the rebound (`e^2`) is. Removal criterion: the exact
/// solution followed into flight, sampled at the run's own times.
#[test]
fn a_zener_rod_dropped_on_its_end_leaves_with_the_exact_restitution() {
    const FRAMES: usize = 120;
    let (ratio, damped) = rod_row("rod below the band", &BAR, FRAMES);
    damped.ledger.check("rod below the band");
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "restitution {ratio} of the exact"
    );
    let half = Rod {
        dx: 0.5 * DX_M,
        ..BAR
    };
    let (half_ratio, half_damped) = rod_row("rod below the band, half cell", &half, FRAMES);
    half_damped.ledger.check("rod below the band, half cell");
    println!(
        "convergence: restitution over the exact {ratio:.4} at 1 cm, {half_ratio:.4} at 0.5 cm"
    );
}

/// In the record's band: the rod's quarter-wave near 530 Hz (`E` 4.6e8 Pa,
/// `c` near 678 m/s), the E2 block's own band, against the exact
/// restitution 0.64039 of the reference's table, at 1 cm and 0.5 cm cells.
/// Criterion fixed before measuring: the restitution within 5 % at the
/// finest cell run. The energy split is reported, under the gap declared
/// above (only the rebound `e^2` is comparable).
///
/// Measured: 0.64444 (1 cm, 1.0063) and 0.64379 (0.5 cm, 1.0053); the
/// rebound fraction 0.4145 against the exact 0.4102.
#[test]
fn a_zener_rod_in_its_records_band_leaves_with_the_exact_restitution() {
    const FRAMES: usize = 40;
    let coarse = Rod {
        dx: DX_M,
        e_pa: 4.6e8,
        frame_s: 5.0e-5,
        speed_m_s: 4.0e-3,
        tolerance: 1e-2,
        modal_flight: false,
    };
    let (ratio, damped) = rod_row("rod in the band", &coarse, FRAMES);
    damped.ledger.check("rod in the band");
    let half = Rod {
        dx: 0.5 * DX_M,
        ..coarse
    };
    let (half_ratio, half_damped) = rod_row("rod in the band, half cell", &half, FRAMES);
    half_damped.ledger.check("rod in the band, half cell");
    println!("in band: restitution over the exact {ratio:.4} at 1 cm, {half_ratio:.4} at 0.5 cm");
    assert!(
        (half_ratio - 1.0).abs() < 0.05,
        "restitution {half_ratio} of the exact at 0.5 cm"
    );
}

/// Whether the damped rod's small excess restitution over the exact one
/// (+0.4 % below the band at both 1 cm and 0.5 cm cells, while the elastic
/// rod's grid loss halves) comes from the time step: the same rod, 1 cm
/// cells, at step tolerances 1e-2, 1e-3 and 1e-4, elastic and damped.
/// Report only.
///
/// Measured:
/// - damped over exact 1.0038, 1.0025, 1.0014 (125, 166, 345 steps);
/// - elastic 0.97858, 0.98624, 0.98563 (243, 2628, 3859 steps), not
///   monotone.
///
/// The damped excess is the step's, not the cell's.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rod_restitution_against_the_step_tolerance() {
    const FRAMES: usize = 120;
    let c = BAR.sound_speed();
    let record = RelaxationSpectrum::plasticine_crandall_1970();
    let exact = exact_rod(&record, &BAR, c);
    for tolerance in [1e-2, 1e-3, 1e-4] {
        let rod = Rod { tolerance, ..BAR };
        let elastic = drop_rod_at(&rod, None, FRAMES);
        let damped = drop_rod_at(&rod, Some(record), FRAMES);
        println!(
            "tolerance {tolerance:.0e}: elastic {:.5} ({} steps), damped {:.5} ({} steps), exact {:.5}, damped over exact {:.4}; damped {}",
            elastic.first,
            elastic.steps,
            damped.first,
            damped.steps,
            exact.restitution,
            damped.first / exact.restitution,
            damped.ledger.summary()
        );
    }
}

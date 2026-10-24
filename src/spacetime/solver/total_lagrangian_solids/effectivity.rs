//! The one-solve error estimate's effectivity (`total_lagrangian::estimate`),
//! MEASURED beside the step controller's doubling on the gates' scenes: per
//! sampled pair of tries, the estimate of the whole step against its error
//! from a 64-substep reference, and the doubling's estimate of the halves
//! against theirs. Nothing here chooses a step: the user decides from these
//! numbers whether the estimate may (`ESTIMATOR_DESIGN.md` sec. 5).

use super::*;
use crate::SimConfig;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::spacetime::total_lagrangian::hooks::{self, EffectivityRow};
use glam::IVec2;

/// The Norton plasticine block of `tests/norton.rs` (E 30 MPa).
fn norton_block() -> Simulation {
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
    sim
}

/// The `q`-quantile of `v` (nearest rank), NaN when empty.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = v.iter().copied().filter(|e| e.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)]
}

/// One scene's rows, by class: the effectivity indices' 5th percentile,
/// median and 95th percentile, and the halving checks' slopes; then how far
/// the kept halves' error is from the controller's target on the pairs it
/// accepted, and the halving samples whose index at `dt/8` is below 0.5.
fn report(scene: &str, rows: &[EffectivityRow]) {
    let kept: Vec<f64> = rows
        .iter()
        .filter(|r| r.compared <= r.target)
        .map(|r| r.halves_error / r.target)
        .collect();
    let over = kept.iter().filter(|&&e| e > 1.0).count();
    println!(
        "{scene}: accepted pairs' kept error / target: 5th {:.3}, median {:.3}, 95th {:.3}, largest {:.3}; above the target in {over} of {}",
        quantile(&kept, 0.05),
        quantile(&kept, 0.5),
        quantile(&kept, 0.95),
        kept.iter().copied().fold(0.0f64, f64::max),
        kept.len()
    );
    let flowing: Vec<f64> = rows
        .iter()
        .filter(|r| !r.switching && !r.held && r.plastic > 0.0)
        .map(|r| r.estimate / r.whole_error)
        .collect();
    let elastic: Vec<f64> = rows
        .iter()
        .filter(|r| !r.switching && !r.held && r.plastic <= 0.0)
        .map(|r| r.estimate / r.whole_error)
        .collect();
    for (name, v) in [
        ("flowing (adaptation 4 active)", &flowing),
        ("no plastic work", &elastic),
    ] {
        println!(
            "{scene}, smooth, no contact, {name}: estimate / whole step's error 5th {:.3}, median {:.3}, 95th {:.3} ({} pairs)",
            quantile(v, 0.05),
            quantile(v, 0.5),
            quantile(v, 0.95),
            v.len()
        );
    }
    for r in rows
        .iter()
        .filter(|r| r.halving.len() == 4 && r.halving[3].1 / r.halving[3].2 < 0.5)
    {
        println!(
            "{scene}: halving sample below 0.5 at dt/8: frame {}, dt {:.3e}, switching {}, held {}, speeds {:?}, plastic {:.3e}",
            r.frame, r.dt, r.switching, r.held, r.speeds, r.plastic
        );
    }
    for (class, switching, held) in [
        ("smooth, no contact held", false, false),
        ("smooth, a contact held", false, true),
        ("switching", true, true),
        ("switching, none held", true, false),
    ] {
        let set: Vec<&EffectivityRow> = rows
            .iter()
            .filter(|r| r.switching == switching && r.held == held)
            .collect();
        if set.is_empty() {
            println!("{scene}, {class}: no sampled pair");
            continue;
        }
        let index =
            |f: &dyn Fn(&EffectivityRow) -> f64| -> Vec<f64> { set.iter().map(|r| f(r)).collect() };
        let line = |name: &str, v: Vec<f64>| {
            println!(
                "{scene}, {class}, {name}: 5th {:.3}, median {:.3}, 95th {:.3} ({} pairs)",
                quantile(&v, 0.05),
                quantile(&v, 0.5),
                quantile(&v, 0.95),
                v.len()
            );
        };
        line(
            "estimate / whole step's error",
            index(&|r| r.estimate / r.whole_error),
        );
        line(
            "estimate / whole step's error, velocity form",
            index(&|r| r.estimate_velocity / r.whole_error_velocity),
        );
        line(
            "doubling / halves' error",
            index(&|r| r.doubling / r.halves_error),
        );
        // The same quantity for both: the whole step's error, which the
        // doubling estimates as `4/3 |halves - whole|` (`2^p (2^p - 1)^-1`,
        // `p = 2`).
        line(
            "doubling's whole-step estimate / whole step's error",
            index(&|r| 4.0 * r.doubling / r.whole_error),
        );
        line(
            "start speed / mean speed (largest, the contact alternation's count)",
            index(&|r| r.speeds[0] / r.speeds[1].max(f64::MIN_POSITIVE)),
        );
        line(
            "end speed / start speed",
            index(&|r| r.speeds[2] / r.speeds[0].max(f64::MIN_POSITIVE)),
        );
        for r in set.iter().filter(|r| !r.halving.is_empty()) {
            let h = &r.halving;
            let slope =
                |i: usize, f: &dyn Fn(&(f64, f64, f64)) -> f64| (f(&h[i]) / f(&h[i + 1])).log2();
            println!(
                "{scene}, {class}, halving from dt {:.3e}: estimate {:?}, error {:?}; log2 ratios estimate {:.2} {:.2} {:.2}, error {:.2} {:.2} {:.2}; effectivity {:.3} {:.3} {:.3} {:.3}",
                r.dt,
                h.iter().map(|e| format!("{:.3e}", e.1)).collect::<Vec<_>>(),
                h.iter().map(|e| format!("{:.3e}", e.2)).collect::<Vec<_>>(),
                slope(0, &|e| e.1),
                slope(1, &|e| e.1),
                slope(2, &|e| e.1),
                slope(0, &|e| e.2),
                slope(1, &|e| e.2),
                slope(2, &|e| e.2),
                h[0].1 / h[0].2,
                h[1].1 / h[1].2,
                h[2].1 / h[2].2,
                h[3].1 / h[3].2
            );
        }
    }
}

/// A scene: its name, its builder, the frames run, and the sampling (every
/// how many pairs, every how many samples the halving check).
type Scene = (&'static str, fn() -> Simulation, usize, (u32, u32));

/// The scenes, each sampled every `every`-th pair, the halving check on
/// every `halving`-th sample: the E2 clay drop (flight, impacts), the
/// Norton block (impact and its flowing ringing), the settled clay block
/// kicked (an enduring contact). The DEP bar runs through its own probe
/// with `EMERGE_PROBE_EFFECTIVITY` set. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_estimator_effectivity() {
    super::cap::capped("probe_estimator_effectivity", 6000, || {
        let scenes: [Scene; 3] = [
            ("E2 clay drop", super::hard_frame::e2, 40, (4, 5)),
            ("Norton block", norton_block, 16, (30, 8)),
            (
                "settled block kicked",
                || super::at_rest_tests::settled_block(1.0),
                10,
                (1, 3),
            ),
        ];
        for (name, build, frames, (every, halving)) in scenes {
            let mut sim = build();
            hooks::set_effectivity(every, halving);
            for k in 1..=frames {
                hooks::set_effectivity_frame(k as u32);
                sim.step();
            }
            hooks::set_effectivity(0, 0);
            report(name, &hooks::take_effectivity());
        }
    });
}

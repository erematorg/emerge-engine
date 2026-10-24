//! What sets a body's fast modes at an impact, after the vibration
//! recovery experiment (`vibration`). A probe: it measures, and changes no
//! answer of the path.
//!
//! The block of `vibration::scene` meets the floor; through the whole
//! impact, from the first touch to the release, at three frames, it records
//! the contacts held and the vibration energy below and above `FAST`. The
//! first touch stops the flat bottom face against the rigid wall; what the
//! modes hold after it, and how it moves through the impact, is compared
//! across the three steps.

use std::f64::consts::TAU;

use super::vibration::{FRAME, Modes, RIGID, Sample, modes, sample, scene, sensors, z};

/// The edge between the modes the impact's excitation converged for and
/// the faster ones (measured by `vibration::probe_vibration_recovery`
/// against half and a quarter of its frame), Hz.
const FAST: f64 = 1200.0;

/// The vibration energy below and above `FAST`, J/m.
fn parts(modes: &Modes, s: &Sample) -> (f64, f64) {
    (RIGID..modes.omega.len()).fold((0.0, 0.0), |(slow, fast), i| {
        let e = 0.5 * z(modes.omega[i], s.q[i]).length_squared();
        if modes.omega[i] / TAU < FAST {
            (slow + e, fast)
        } else {
            (slow, fast + e)
        }
    })
}

/// One run through the impact at frame `frame`: per frame, its sample and
/// the contacts held, until the contact has stayed off for 1 ms.
fn through_the_impact(
    frame: f32,
    modes: &Modes,
    sensors: &[(usize, usize); 3],
) -> (Vec<Sample>, Vec<usize>) {
    let h = f64::from(frame);
    let mut sim = scene(frame, 1e-2);
    let (mut samples, mut held) = (Vec::new(), Vec::new());
    let (mut touched, mut quiet) = (false, 0.0);
    while !(touched && quiet >= 1e-3) {
        sim.step();
        let contacts = sim.total_lagrangian_frame().expect("on the path").contacts;
        touched |= contacts > 0;
        quiet = if contacts > 0 { 0.0 } else { quiet + h };
        samples.push(sample(&sim, modes, sensors, (samples.len() + 1) as f64 * h));
        held.push(contacts);
        assert!(samples.len() as f64 * h < 0.05, "no release");
    }
    (samples, held)
}

#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_fast_modes_at_the_impact() {
    super::cap::capped("probe_fast_modes_at_the_impact", 1800, || {
        let fine = (FRAME / 1024.0) as f32;
        let mut a = scene(fine, 1e-2);
        a.step();
        let state = a.total_lagrangian.as_ref().expect("on the path");
        let modes = modes(state);
        let sensors = sensors(state);
        let runs: Vec<(f64, Vec<Sample>, Vec<usize>)> = [1.0f32, 2.0, 4.0]
            .iter()
            .map(|&d| {
                let (samples, held) = through_the_impact(fine / d, &modes, &sensors);
                (f64::from(fine / d), samples, held)
            })
            .collect();
        for (h, samples, held) in &runs {
            let first = held.iter().position(|&n| n > 0).expect("an impact");
            let last = held.iter().rposition(|&n| n > 0).expect("an impact");
            let release = &samples[last + 1];
            let during = &held[first..=last];
            let (peak, at) = samples[first..=last]
                .iter()
                .map(|s| (parts(&modes, s).1, s.t))
                .fold((0.0, 0.0), |b, c| if c.0 > b.0 { c } else { b });
            let (slow_first, fast_first) = parts(&modes, &samples[first]);
            let (slow_end, fast_end) = parts(&modes, release);
            println!(
                "frame {:.3} us: touch {:.4} ms, release {:.4} ms ({:.4} ms; 2 H / c_p = 0.848 ms); contact held in {} of its {} frames",
                h * 1e6,
                samples[first].t * 1e3,
                release.t * 1e3,
                (release.t - samples[first].t) * 1e3,
                during.iter().filter(|&&n| n > 0).count(),
                during.len()
            );
            println!(
                "  vibration J/m after the first touch's frame: below {FAST} Hz {slow_first:.5e}, above {fast_first:.5e}; above at its peak {peak:.4e} ({:.1} us after the touch); at the release: below {slow_end:.5e}, above {fast_end:.5e}",
                (at - samples[first].t) * 1e6
            );
        }
        // The three runs at common times through the impact, every 32.55 us
        // from the finest run's touch.
        let (_, finest, finest_held) = &runs[2];
        let touch = finest[finest_held.iter().position(|&n| n > 0).expect("an impact")].t;
        println!(
            "  t from the touch us | above {FAST} Hz J/m at 16.3 / 8.1 / 4.1 us | contacts held since the last row (16.3 / 8.1 / 4.1 us)"
        );
        let (mut row_t, mut previous) = (touch, touch - 2.0 * f64::from(fine));
        while row_t
            <= runs
                .iter()
                .map(|r| r.1.last().expect("sampled").t)
                .fold(f64::INFINITY, f64::min)
                - 1e-3
        {
            let cells: Vec<String> = runs
                .iter()
                .map(|(h, samples, _)| {
                    let j = ((row_t / h).round() as usize).clamp(1, samples.len()) - 1;
                    format!("{:.4e}", parts(&modes, &samples[j]).1)
                })
                .collect();
            let held: Vec<String> = runs
                .iter()
                .map(|(h, _, held)| {
                    let (from, to) = (
                        (previous / h).round() as usize,
                        (row_t / h).round() as usize,
                    );
                    let count = (from..to.min(held.len())).filter(|&j| held[j] > 0).count();
                    format!("{count}/{}", to.saturating_sub(from))
                })
                .collect();
            println!(
                "    {:8.2} | {} | {}",
                (row_t - touch) * 1e6,
                cells.join(" "),
                held.join(" ")
            );
            previous = row_t;
            row_t += 2.0 * f64::from(fine);
        }
    });
}

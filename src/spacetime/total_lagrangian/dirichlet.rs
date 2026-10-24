//! A research variant of the step with prescribed boundary motion, for the
//! manufactured-solution probes.

use super::*;

/// The conserving step with `held` faces kept at zero both ways, a
/// Dirichlet condition with no release, and a per-particle body force, the
/// grid rebuilt at the step's start as in `step`: the probes' manufactured
/// solutions, whose roller boundaries are exact.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn step_dirichlet(
    layout: &GridLayout,
    particles: Particles<'_>,
    solids: &[Elastic],
    (body, dt): (&[DVec2], f64),
    (held, settings): (Held<'_>, &ConservingSettings),
) -> StepReport {
    let Particles { x, v, f, mass, v0 } = particles;
    let n = x.len();
    let volume: Vec<f64> = v0
        .iter()
        .zip(f.iter())
        .map(|(&v, m)| v * m.determinant())
        .collect();
    let clock = std::time::Instant::now();
    let grid = Grid::new(layout, x, (mass, &volume), held, settings.occupation);
    let mut phases = [clock.elapsed().as_secs_f64(), 0.0, 0.0, 0.0, 0.0];
    let inputs = Inputs {
        x,
        v,
        f,
        mass,
        v0,
        solids,
        gravity: DVec2::ZERO,
        dt,
        settings,
        planes: &[],
        penalty: &[],
        lagrangian: false,
        body: Some(body),
        friction: 0.0,
        responses: false,
    };
    let mut factors = Factors::default();
    let Solution {
        u, v_n, mut report, ..
    } = solve(&grid, &inputs, &[], &mut factors);
    for (total, part) in phases.iter_mut().zip(report.phases).skip(1) {
        *total += part;
    }
    report.passes = 1;
    let finishing = std::time::Instant::now();
    let dv: Vec<f64> = (0..grid.len)
        .map(|k| {
            if grid.active[k] {
                2.0 * (u[k] / dt - v_n[k])
            } else {
                0.0
            }
        })
        .collect();
    for p in 0..n {
        let (_, gu) = grid.read(&u, p, 0);
        let (_, gv) = grid.read(&u, p, 1);
        let grad = DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
        let mut step_x = DVec2::ZERO;
        let mut step_v = DVec2::ZERO;
        for a in 0..2 {
            step_x[a] = grid.read(&u, p, a).0;
            step_v[a] = grid.read(&dv, p, a).0;
        }
        x[p] += step_x;
        v[p] += step_v;
        f[p] = (DMat2::IDENTITY + grad) * f[p];
    }
    phases[4] = finishing.elapsed().as_secs_f64();
    report.phases = phases;
    report
}

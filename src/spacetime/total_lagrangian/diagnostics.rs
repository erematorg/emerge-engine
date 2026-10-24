//! Measurements of the step's grid for the probes: mass shares, the mass
//! matrix's conditioning, and a report on the basis.

use super::*;

/// Per particle and lattice, the share of the partition of unity the
/// unknowns with mass hold at its reference position (`Grid::massive`, the
/// massless faces of `hooks::set_massless`): a field whose massive
/// coefficients are all `c` reads `c` times this there, the probes'
/// uniform start on the massive unknowns.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn massive_share(
    layout: &GridLayout,
    reference: &[DVec2],
    (mass, v0): (&[f64], &[f64]),
    occupation: Option<f64>,
) -> Vec<[f64; 2]> {
    let grid = Grid::new(layout, reference, (mass, v0), &|_, _, _| None, occupation);
    let massive = grid.massive();
    (0..reference.len())
        .map(|p| {
            [0, 1].map(|a| {
                let mut share = 0.0;
                grid.visit(p, a, |k, w, _| {
                    if massive[k] {
                        share += w;
                    }
                });
                share
            })
        })
        .collect()
}

/// The smallest over the largest eigenvalue of the masked consistent mass
/// `M_eps` of the grid a step would build for these particles, by inverse
/// iteration on its exact band factor and power iteration (the probes'
/// measure of how near singular the step's system is).
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn mass_conditioning(
    layout: &GridLayout,
    x: &[DVec2],
    (mass, volume): (&[f64], &[f64]),
    held: Held<'_>,
    (occupation, eps): (Option<f64>, f64),
) -> f64 {
    let grid = Grid::new(layout, x, (mass, volume), held, occupation);
    let factor = grid.mass_band(mass, eps);
    let norm = |y: &[f64]| y.iter().map(|e| e * e).sum::<f64>().sqrt();
    let mask = |y: Vec<f64>| -> Vec<f64> {
        y.into_iter()
            .zip(&grid.active)
            .map(|(e, &on)| if on { e } else { 0.0 })
            .collect()
    };
    let start = mask(
        (0..grid.len)
            .map(|k| ((k * 7919) % 101) as f64 - 50.0)
            .collect(),
    );
    let (mut largest, mut z) = (0.0, start.clone());
    for _ in 0..30 {
        let n0 = norm(&z).max(f64::MIN_POSITIVE);
        z.iter_mut().for_each(|e| *e /= n0);
        z = mask(grid.mass_times(&z, mass, eps));
        largest = norm(&z);
    }
    let (mut smallest, mut y) = (0.0, start);
    for _ in 0..30 {
        let n0 = norm(&y).max(f64::MIN_POSITIVE);
        y.iter_mut().for_each(|e| *e /= n0);
        y = mask(factor.solve(&y));
        smallest = 1.0 / norm(&y).max(f64::MIN_POSITIVE);
    }
    smallest / f64::max(largest, f64::MIN_POSITIVE)
}

/// What the extended basis of a step's grid is made of for these
/// particles, for the probes: the faces classified stable, extrapolated and
/// left their own B-spline; how many particles reach each active unknown;
/// the largest extension weight; the lumped masses below zero; and where
/// the masked consistent mass's nearly null mode lies, its five largest
/// entries with their face, the particles reaching them and the volume
/// fraction of the face's own cell.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn basis_report(
    layout: &GridLayout,
    x: &[DVec2],
    (mass, volume): (&[f64], &[f64]),
    held: Held<'_>,
    (occupation, eps): (Option<f64>, f64),
) -> String {
    let grid = Grid::new(layout, x, (mass, volume), held, occupation);
    let mut reach = vec![0usize; grid.len];
    for p in 0..x.len() {
        let mut seen = std::collections::HashSet::new();
        for a in 0..2 {
            grid.visit(p, a, |k, w, _| {
                if w.abs() > 1e-12 && seen.insert(k) {
                    reach[k] += 1;
                }
            });
        }
    }
    let active: Vec<usize> = (0..grid.len).filter(|&k| grid.active[k]).collect();
    let stable = active.iter().filter(|&&k| grid.stable_unknown[k]).count();
    let own = active.len() - stable;
    let extended = grid.extension.iter().filter(|e| e.is_some()).count();
    let largest_weight = grid
        .extension
        .iter()
        .flatten()
        .flat_map(|list| list.iter().map(|&(_, e)| e.abs()))
        .fold(0.0f64, f64::max);
    let mut reaches: Vec<usize> = active.iter().map(|&k| reach[k]).collect();
    reaches.sort_unstable();
    let few = |n: usize| reaches.iter().filter(|&&r| r <= n).count();
    let negative = active.iter().filter(|&&k| grid.lumped[k] < 0.0).count();
    // The nearly null mode, by inverse iteration on the exact factor.
    let factor = grid.mass_band(mass, eps);
    let mut y: Vec<f64> = (0..grid.len)
        .map(|k| {
            if grid.active[k] {
                ((k * 7919) % 101) as f64 - 50.0
            } else {
                0.0
            }
        })
        .collect();
    for _ in 0..30 {
        let n0 = y
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        y.iter_mut().for_each(|e| *e /= n0);
        y = factor
            .solve(&y)
            .into_iter()
            .zip(&grid.active)
            .map(|(e, &on)| if on { e } else { 0.0 })
            .collect();
    }
    let n0 = y
        .iter()
        .map(|e| e * e)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let mut order: Vec<usize> = active.clone();
    order.sort_by(|&a, &b| y[b].abs().total_cmp(&y[a].abs()));
    // Each lattice's knot spans' volume fractions as `Grid::new` takes them
    // (Kakouris eq. 19), and each stable face's interior spans: where in
    // its 3 x 3 support they lie and the B-spline's value at their centre,
    // 0.75 per axis at the middle span and 0.125 at a side one.
    let h = f64::from(layout.dx);
    let cc = occupation.unwrap_or(f64::INFINITY);
    let fractions: Vec<Vec<f64>> = grid
        .lattices
        .iter()
        .map(|l| {
            let mut out = vec![0.0; l.ni * l.nj];
            for (q, &vol) in x.iter().zip(volume) {
                let s = *q / h - l.offset;
                let (m, n) = ((s.x + 0.5).floor() as i64, (s.y + 0.5).floor() as i64);
                if m >= 0 && n >= 0 && (m as usize) < l.ni && (n as usize) < l.nj {
                    out[m as usize + l.ni * n as usize] += vol / (h * h);
                }
            }
            out
        })
        .collect();
    let place = |k: usize| -> (usize, i64, i64) {
        let a = grid.axis_of[k];
        let l = grid.lattices[a];
        let local = grid.stacked_of[k] - l.start;
        (a, (local % l.ni) as i64, (local / l.ni) as i64)
    };
    let interior = |k: usize| -> Vec<(i64, i64, f64)> {
        let (a, i, j) = place(k);
        let l = grid.lattices[a];
        let mut out = Vec::new();
        for dj in -1..=1 {
            for di in -1..=1 {
                let (m, n) = (i + di, j + dj);
                if m >= 0 && n >= 0 && (m as usize) < l.ni && (n as usize) < l.nj {
                    let f = fractions[a][m as usize + l.ni * n as usize];
                    if f > cc {
                        out.push((di, dj, f));
                    }
                }
            }
        }
        out
    };
    let centre = |d: i64| if d == 0 { 0.75 } else { 0.125 };
    let best = |k: usize| {
        interior(k)
            .iter()
            .map(|&(di, dj, _)| centre(di) * centre(dj))
            .fold(0.0f64, f64::max)
    };
    let stable_faces: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&k| grid.stable_unknown[k])
        .collect();
    let corner_only = stable_faces.iter().filter(|&&k| best(k) < 0.02).count();
    let side_best = stable_faces
        .iter()
        .filter(|&&k| (0.02..0.5).contains(&best(k)))
        .count();
    // The particles' weights each unknown gathers, `sum_p N_k(x_p)`: about
    // four on a face inside a body at four particles per cell.
    let mut gathered = vec![0.0; grid.len];
    for p in 0..x.len() {
        for a in 0..2 {
            grid.visit(p, a, |k, w, _| gathered[k] += w);
        }
    }
    let top: Vec<String> = order
        .iter()
        .take(5)
        .map(|&k| {
            let (a, i, j) = place(k);
            let spans: Vec<String> = interior(k)
                .iter()
                .map(|&(di, dj, f)| format!("({di:+},{dj:+}) {f:.2}"))
                .collect();
            format!(
                "{} face ({i},{j}) amplitude {:.2} reached by {} particles gathering weight {:.3}, lumped {:.2}, {}, interior spans of its support [{}]",
                ["u", "v"][a],
                y[k] / n0,
                reach[k],
                gathered[k],
                grid.lumped[k],
                if grid.stable_unknown[k] { "stable" } else { "own spline" },
                spans.join(" ")
            )
        })
        .collect();
    format!(
        "{} active unknowns ({stable} stable, {own} own spline), {extended} faces extrapolated, largest weight {largest_weight:.1}, {negative} negative lumped masses; stable faces whose interior spans are only corners (B-spline 1/64 at their centre): {corner_only}, whose best is a side span (3/32): {side_best}; particles reaching an unknown: fewest {}, median {}, unknowns reached by at most 1 / 2 / 4 particles: {} / {} / {}; nearly null mode: {}",
        active.len(),
        reaches.first().copied().unwrap_or(0),
        reaches.get(reaches.len() / 2).copied().unwrap_or(0),
        few(1),
        few(2),
        few(4),
        top.join("; ")
    )
}

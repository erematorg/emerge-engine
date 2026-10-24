//! A body's contact conditions as weighted gaps: one per multiplier patch
//! on its outline, the weak form of the Signorini condition instead of one
//! condition per contact point.
//!
//! Source: Burman, *Projection stabilisation of Lagrange multipliers for
//! the imposition of constraints on interfaces and boundaries*, arXiv
//! 1203.4116 (2013): the classical condition for the multipliers' inf-sup
//! stability is a multiplier mesh coarser than the trace of the
//! displacement's, `h_Gamma,L > C h_Gamma,V`, `C > 1` (sec. 2, after
//! Babuska); for a boundary cutting the mesh, multipliers constant on
//! boundary macro patches `F_j`, `H <= meas(F_j cap Gamma) <= H + h`, `H >
//! c h` (sec. 4, the trace inequality eq. 4.14). The weighted gap of each
//! multiplier and its push shared by the same weights: Hueber, Stadler and
//! Wohlmuth, SIAM J. Sci. Comput. 30(2) (2008), eqs. 2.6-2.8; Puso, Porter
//! and Slavik, LLNL 2024, eqs. 7-8 and 12-15 (each multiplier's pushes on the
//! two sides balance, so the momentum is kept). At a corner the material has
//! two normals, so two conditions (Puso et al., footnote 1).
//! Adaptation, this engine's, declared:
//! - the patches are cut on the body's REFERENCE outline, between the exact
//!   shape's corners (`spawn_outline`), and move with the material: tied to
//!   the body, not to the grid, so a body's conditions do not depend on
//!   where the grid falls under it;
//! - `H = 2 h`, the smallest whole-cell `C > 1`: a chosen constant, its
//!   inf-sup measured at three cells and several offsets from the grid
//!   (`the_floor_conditions_pass_a_discrete_inf_sup_at_three_cells`).
//!   Removal criterion: a derived or sourced `H`, or a larger one if the
//!   measured inf-sup decays;
//! - each patch's gap is the exact integral over it of the outline's
//!   piecewise linear gap between its points, normalised to a mean;
//! - each condition integrates its whole patch (Puso et al.'s eq. 8);
//!   against a body only over the points that project onto it along their
//!   normal (their set `S_A`, eq. 10), renormalised. No test of which way a
//!   point faces: on a side perpendicular to the obstacle its sign is the
//!   solve's noise (measured: a flat block's corner conditions decided
//!   differently left and right, the block left a flat bounce turning at
//!   4.9 rad/s);
//! - between two bodies, single pass: the first outline is the non-mortar
//!   side, as in Puso et al.'s sec. 3 (their two-pass repeats it flipped).
//!
//! History, measured (each step's reason in the review): one condition per
//! outline point held two points per cell, their gap gradients dependent
//! (rank 28 of 41), the multipliers' Newton cycling at an impact; the grid
//! vertices' functions traced on the outline (Ferte, Massin and Moes 2014)
//! were full rank but tied to the grid, a flat block off its lines tipping
//! over on landing (87 degrees) where the same block on them did not.

use super::*;

/// One outline's multiplier functions: for each, the outline points (their
/// index in the outline) its trace reaches and the quadrature weight times
/// its value there, `omega_r N_v(X_r)`.
#[derive(Clone, Debug, Default)]
pub(crate) struct OutlineTrace {
    pub(crate) functions: Vec<Vec<(usize, f64)>>,
}

/// The multiplier functions of the closed outline `outline` (particles,
/// counter-clockwise) at its reference positions, for the grid of cell
/// `cell` (module doc): each face (the outline cut at `corners`, indices
/// into it where the exact shape's normal jumps) divided into `n =
/// max(1, floor(L / H))` equal patches, `H = 2 cell`; each patch's function
/// the constant one on it, its members the outline points and the exact
/// integral over the patch of their piecewise linear hat. A corner point
/// belongs to both faces, with its half segment on each.
pub(crate) fn outline_trace(
    reference: &[DVec2],
    outline: &[usize],
    corners: &[usize],
    cell: f64,
) -> OutlineTrace {
    let count = outline.len();
    if count < 2 {
        return OutlineTrace::default();
    }
    let at = |r: usize| reference[outline[r % count]];
    let mut cuts: Vec<usize> = corners.iter().copied().filter(|&c| c < count).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let starts = if cuts.is_empty() { vec![0] } else { cuts };
    let patch = 2.0 * cell;
    let mut functions = Vec::new();
    for (k, &start) in starts.iter().enumerate() {
        let end = if starts.len() == 1 {
            start + count
        } else {
            let next = starts[(k + 1) % starts.len()];
            if next > start { next } else { next + count }
        };
        // The face's arclength at each of its points.
        let mut arc = vec![0.0f64];
        for s in start..end {
            let last = *arc.last().expect("started at 0");
            arc.push(last + (at(s + 1) - at(s)).length());
        }
        let length = *arc.last().expect("at least one segment");
        if length <= 0.0 {
            continue;
        }
        // The arclength is a sum of `end - start` lengths, each off by
        // `eps` of the running sum: a face of exactly `n H` reads within
        // that of it, and must still give `n` patches.
        let summed = (end - start) as f64 * f64::EPSILON;
        let n = ((length / patch * (1.0 + summed)).floor() as usize).max(1);
        let width = length / n as f64;
        let mut patches: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let add = |members: &mut Vec<(usize, f64)>, r: usize, w: f64| {
            if w <= 0.0 {
                return;
            }
            match members.iter_mut().find(|(q, _)| *q == r) {
                Some(entry) => entry.1 += w,
                None => members.push((r, w)),
            }
        };
        for (i, s) in (start..end).enumerate() {
            let (a, b) = (arc[i], arc[i + 1]);
            if b <= a {
                continue;
            }
            let first = ((a / width).floor() as usize).min(n - 1);
            let last = (((b / width).ceil() as usize).max(1) - 1).min(n - 1);
            for (j, members) in patches.iter_mut().enumerate().take(last + 1).skip(first) {
                let (c, d) = (a.max(j as f64 * width), b.min((j + 1) as f64 * width));
                if d <= c {
                    continue;
                }
                // The integral over [c, d] of the segment's two hats.
                let toward_end = ((d - a).powi(2) - (c - a).powi(2)) / (2.0 * (b - a));
                add(members, s % count, (d - c) - toward_end);
                add(members, (s + 1) % count, toward_end);
            }
        }
        functions.extend(patches.into_iter().filter(|m| !m.is_empty()));
    }
    OutlineTrace { functions }
}

/// The outward normal of an outline's point `r` (counter-clockwise: on the
/// right of the direction from its previous neighbour to its next), and the
/// neighbours, at the positions `x`.
fn outward(x: &[DVec2], outline: &[usize], r: usize) -> (DVec2, (usize, usize)) {
    let count = outline.len();
    let (previous, next) = (outline[(r + count - 1) % count], outline[(r + 1) % count]);
    let along = x[next] - x[previous];
    (
        DVec2::new(along.y, -along.x) / along.length().max(f64::MIN_POSITIVE),
        (previous, next),
    )
}

/// The weighted-gap conditions that reach their obstacle within the step
/// (`reaches` on the weighted start gap and its rate), for the velocity
/// form of the step: each point against a plane gets one of its own at its
/// start position (pushed onto `own`), each against a segment the relative
/// normal displacement along its nodal normal, so every condition is the
/// weighted normal displacement over the step (Laursen and Chawla's
/// velocity condition on the weighted gap).
pub(crate) fn reaching_conditions(
    traces: &[OutlineTrace],
    outlines: &[Vec<usize>],
    (x, v): (&[DVec2], &[DVec2]),
    planes: &[Plane],
    own: &mut Vec<Plane>,
    reaches: &dyn Fn(f64, f64) -> bool,
) -> Vec<Contact> {
    let mut found = Vec::new();
    let mut function = 0usize;
    for (o, (trace, outline)) in traces.iter().zip(outlines).enumerate() {
        for members in &trace.functions {
            for (index, plane) in planes.iter().enumerate() {
                let mut inward = DVec2::ZERO;
                inward[plane.axis] = plane.inward;
                let facing: &[(usize, f64)] = members;
                let total: f64 = facing.iter().map(|&(_, w)| w).sum();
                if total <= 0.0 {
                    continue;
                }
                let (gap, rate) = facing.iter().fold((0.0, 0.0), |(g, d), &(r, w)| {
                    let q = outline[r];
                    (
                        g + w / total * plane.inward * (x[q][plane.axis] - plane.at),
                        d + w / total * plane.inward * v[q][plane.axis],
                    )
                });
                if !reaches(gap, rate) {
                    continue;
                }
                let points = facing
                    .iter()
                    .map(|&(r, w)| {
                        let q = outline[r];
                        own.push(Plane {
                            at: x[q][plane.axis],
                            ..*plane
                        });
                        (
                            ContactPoint {
                                particle: q,
                                plane: own.len() - 1,
                                partner: None,
                                segment: None,
                                nodal: None,
                            },
                            w / total,
                        )
                    })
                    .collect();
                found.push(Contact {
                    points,
                    key: (3, function, index),
                    multiplier: 0.0,
                    tangential: 0.0,
                });
            }
            for (other, master) in outlines.iter().enumerate().skip(o + 1) {
                if master.len() < 2 {
                    continue;
                }
                let mut facing: Vec<(ContactPoint, f64, f64, f64)> = Vec::new();
                for &(r, w) in members {
                    let p = outline[r];
                    let (normal, nodal) = outward(x, outline, r);
                    // The master's segment the point's normal line meets,
                    // nearest along it, facing the point.
                    let mut best: Option<(f64, (usize, usize), f64)> = None;
                    for k in 0..master.len() {
                        let (first, second) = (master[k], master[(k + 1) % master.len()]);
                        let side = x[second] - x[first];
                        let length = side.length();
                        if length <= f64::MIN_POSITIVE {
                            continue;
                        }
                        let denominator = normal.perp_dot(side);
                        if denominator.abs() <= f64::MIN_POSITIVE {
                            continue;
                        }
                        let xi = normal.perp_dot(x[p] - x[first]) / denominator;
                        if !(0.0..=1.0).contains(&xi) {
                            continue;
                        }
                        let hit = x[first] + xi * side;
                        let signed = (hit - x[p]).dot(normal);
                        if best.is_none_or(|(s, _, _)| signed.abs() < s.abs()) {
                            let closing =
                                ((1.0 - xi) * v[first] + xi * v[second] - v[p]).dot(normal);
                            best = Some((signed, (first, second), closing));
                        }
                    }
                    if let Some((signed, ends, closing)) = best {
                        facing.push((
                            ContactPoint {
                                particle: p,
                                plane: usize::MAX,
                                partner: None,
                                segment: Some(ends),
                                nodal: Some(nodal),
                            },
                            w,
                            signed,
                            closing,
                        ));
                    }
                }
                let total: f64 = facing.iter().map(|f| f.1).sum();
                if total <= 0.0 {
                    continue;
                }
                let gap: f64 = facing.iter().map(|f| f.1 / total * f.2).sum();
                let rate: f64 = facing.iter().map(|f| f.1 / total * f.3).sum();
                if !reaches(gap, rate) {
                    continue;
                }
                found.push(Contact {
                    points: facing.iter().map(|f| (f.0, f.1 / total)).collect(),
                    key: (4, function, other),
                    multiplier: 0.0,
                    tangential: 0.0,
                });
            }
            function += 1;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's outline, `width` by `height` cells from `corner`, points
    /// every half cell, counter-clockwise; and its four corners' indices.
    fn block(corner: DVec2, (width, height): (usize, usize)) -> (Vec<DVec2>, Vec<usize>) {
        let mut points = Vec::new();
        let (w, h) = (2 * width, 2 * height);
        for i in 0..w {
            points.push(corner + DVec2::new(0.5 * i as f64, 0.0));
        }
        for j in 0..h {
            points.push(corner + DVec2::new(width as f64, 0.5 * j as f64));
        }
        for i in 0..w {
            points.push(corner + DVec2::new(width as f64 - 0.5 * i as f64, height as f64));
        }
        for j in 0..h {
            points.push(corner + DVec2::new(0.0, height as f64 - 0.5 * j as f64));
        }
        (points, vec![0, w, w + h, 2 * w + h])
    }

    /// Each face is tiled by patches of `H = 2` cells: a 6 by 4 block has
    /// 3 + 2 + 3 + 2 of them, each patch's weights summing to its width,
    /// the whole to the perimeter; and the same block moved off the grid's
    /// lines has the same patches, weight for weight (tied to the body).
    #[test]
    fn faces_are_tiled_by_body_tied_patches() {
        let mut first: Option<Vec<Vec<(usize, f64)>>> = None;
        for corner in [DVec2::new(5.0, 2.0), DVec2::new(5.3, 2.7)] {
            let (reference, corners) = block(corner, (6, 4));
            let outline: Vec<usize> = (0..reference.len()).collect();
            let trace = outline_trace(&reference, &outline, &corners, 1.0);
            assert_eq!(trace.functions.len(), 10);
            let total: f64 = trace.functions.iter().flatten().map(|&(_, w)| w).sum();
            assert!((total - 20.0).abs() < 1e-12, "perimeter {total}");
            for f in &trace.functions {
                let width: f64 = f.iter().map(|&(_, w)| w).sum();
                assert!((width - 2.0).abs() < 1e-12, "patch width {width}");
            }
            match &first {
                None => first = Some(trace.functions.clone()),
                Some(before) => {
                    for (f, g) in before.iter().zip(&trace.functions) {
                        for ((r, w), (q, v)) in f.iter().zip(g) {
                            assert_eq!(r, q);
                            assert!((w - v).abs() < 1e-12);
                        }
                    }
                }
            }
        }
    }

    /// The eigenvalues of the symmetric matrix `a`, by cyclic Jacobi
    /// rotations until the off-diagonal is round-off.
    fn eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
        let n = a.len();
        for _ in 0..100 {
            let off: f64 = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
                .map(|(i, j)| a[i][j] * a[i][j])
                .sum();
            let diagonal: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
            if off <= 1e-30 * diagonal {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    if a[p][q] == 0.0 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let t = if theta == 0.0 { 1.0 } else { t };
                    let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                    for row in a.iter_mut() {
                        let (akp, akq) = (row[p], row[q]);
                        row[p] = c * akp - s * akq;
                        row[q] = s * akp + c * akq;
                    }
                    let (low, high) = a.split_at_mut(q);
                    for (apk, aqk) in low[p].iter_mut().zip(high[0].iter_mut()) {
                        let (x, y) = (*apk, *aqk);
                        *apk = c * x - s * y;
                        *aqk = s * x + c * y;
                    }
                }
            }
        }
        (0..n).map(|i| a[i][i]).collect()
    }

    /// The discrete inf-sup, measured: a 6 by 4 block resting on a floor at
    /// cells of 1, 1/2 and 1/4, its floor conditions' rows `G` (each the
    /// weighted gap's gradient in the unknowns). Expected, stated before
    /// measuring (Ferte et al. Thm 1, the P1-on-quadratic analog): the
    /// ratio `sigma_min(G) / sigma_max(G)` bounded away from zero
    /// independently of the cell, and the Schur operator `G M^-1 G^T`
    /// (lumped mass) scaled by its natural `rho h^2` alike. An unstable
    /// multiplier space loses at least `h^(1/2)`: over the 4x refinement a
    /// factor 2. The bar: less than that.
    #[test]
    fn the_floor_conditions_pass_a_discrete_inf_sup_at_three_cells() {
        let mut ratios = Vec::new();
        let mut schur = Vec::new();
        for (k, offset) in [
            (1usize, 0.0),
            (2, 0.0),
            (4, 0.0),
            (1, 0.3),
            (2, 0.3),
            (4, 0.3),
        ] {
            let h = 1.0 / k as f64;
            let layout = GridLayout::new(10 * k, 8 * k, h as f32);
            let corner = DVec2::new(2.0 + offset * h, 1.0);
            let mut x = Vec::new();
            let mut i = 0.0;
            while i < 6.0 {
                let mut j = 0.0;
                while j < 4.0 {
                    x.push(corner + DVec2::new(i + 0.25 * h, j + 0.25 * h));
                    j += 0.5 * h;
                }
                i += 0.5 * h;
            }
            let particles = x.len();
            let mut mass = vec![0.25 * h * h; particles];
            // The outline, a point every half cell.
            let start = x.len();
            let mut edge = |a: DVec2, b: DVec2| {
                let count = ((b - a).length() / (0.5 * h)).round() as usize;
                for s in 0..count {
                    x.push(a + (b - a) * (s as f64 / count as f64));
                }
            };
            let (c0, c1, c2, c3) = (
                corner,
                corner + DVec2::new(6.0, 0.0),
                corner + DVec2::new(6.0, 4.0),
                corner + DVec2::new(0.0, 4.0),
            );
            edge(c0, c1);
            edge(c1, c2);
            edge(c2, c3);
            edge(c3, c0);
            mass.resize(x.len(), 0.0);
            let volume = mass.clone();
            let outline: Vec<usize> = (start..x.len()).collect();
            let open = |_: usize, _: usize, _: usize| None;
            let grid = Grid::new(&layout, &x, (&mass, &volume), &open, Some(0.75));
            let sides = [6.0, 4.0, 6.0, 4.0].map(|l: f64| (l / (0.5 * h)).round() as usize);
            let corners = vec![
                0,
                sides[0],
                sides[0] + sides[1],
                sides[0] + sides[1] + sides[2],
            ];
            let trace = outline_trace(&x, &outline, &corners, h);
            let mut rows: Vec<Vec<f64>> = Vec::new();
            for members in &trace.functions {
                // The bottom face's functions: every point on the floor.
                if members
                    .iter()
                    .any(|&(r, _)| (x[outline[r]].y - 1.0).abs() > 1e-9)
                {
                    continue;
                }
                let facing: &[(usize, f64)] = members;
                let total: f64 = facing.iter().map(|&(_, w)| w).sum();
                let mut row = vec![0.0; grid.len];
                for &(r, w) in facing {
                    grid.visit(outline[r], 1, |kk, n, _| row[kk] += w / total * n);
                }
                rows.push(row);
            }
            let m = rows.len();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
            let gram: Vec<Vec<f64>> = (0..m)
                .map(|i| (0..m).map(|j| dot(&rows[i], &rows[j])).collect())
                .collect();
            let mut values = eigenvalues(gram);
            values.sort_by(f64::total_cmp);
            let ratio = (values[0] / values[m - 1]).sqrt();
            let inverse: Vec<f64> = grid
                .lumped
                .iter()
                .map(|&l| if l > 0.0 { 1.0 / l } else { 0.0 })
                .collect();
            let operator: Vec<Vec<f64>> = (0..m)
                .map(|i| {
                    (0..m)
                        .map(|j| {
                            rows[i]
                                .iter()
                                .zip(&rows[j])
                                .zip(&inverse)
                                .map(|((p, q), w)| p * q * w)
                                .sum()
                        })
                        .collect()
                })
                .collect();
            let mut operator_values = eigenvalues(operator);
            operator_values.sort_by(f64::total_cmp);
            println!(
                "cell {h}, offset {offset} cell: {m} floor conditions, sigma_min / sigma_max of G {ratio:.4}, Schur's smallest eigenvalue times rho h^2 {:.4}, its largest {:.4}",
                operator_values[0] * h * h,
                operator_values[m - 1] * h * h
            );
            ratios.push(ratio);
            schur.push(operator_values[0] * h * h);
        }
        assert!(ratios.iter().all(|&r| r > 0.0), "full rank: {ratios:?}");
        for start in [0, 3] {
            assert!(
                ratios[start + 2] > 0.5 * ratios[start],
                "G's ratio decays: {ratios:?}"
            );
            assert!(
                schur[start + 2] > 0.5 * schur[start],
                "the Schur operator's decays: {schur:?}"
            );
        }
    }
}

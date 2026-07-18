//! `fit_contact_normal_lr` + `solve3x3` -- split out of `mod.rs` (was ~240 of
//! its ~1320 lines including tests). Standalone free functions, only called
//! from `Grid::resolve_contact`; no dependency on `Grid`/`Cell` internals.

use glam::Vec2;

/// Fits the contact-interface separating plane through a labeled particle point cloud
/// via logistic regression — Nairn, "New Material Point Method Contact Algorithms for
/// Improved Accuracy" (2020), the LR method, eq. 19-21 + Appendix eq. 53-57. Verified
/// against the actual paper text, not a secondary description (see project memory
/// `locomotion_core_frictional_contact_2026-07-11`).
///
/// Replaces Bardenhagen's own original normal — the spatial gradient of the grip
/// field's grid mass — which this paper's own Figure 3C independently identifies as
/// unreliable near a material edge/corner: a node near a corner of one body sees a
/// tilted gradient from that body while the other body's gradient stays vertical;
/// even AVERAGING the two (an improvement over using either body's gradient alone,
/// tried and rejected here as `MIN_MASS_FRACTION`/Sobel/wider-stencil attempts before
/// this) still leaves a real residual tilt. Fitting a plane through actual particle
/// POSITIONS instead sidesteps grid-discretization artifacts entirely — confirmed
/// empirically here too: forcing an exact known normal in a real test made friction
/// behave correctly, and every grid-gradient smoothing attempt failed to reproduce
/// that, which is exactly the failure mode this paper diagnoses and fixes.
///
/// `points`: (position, label) pairs gathered by `gather_contact_point_cloud` — every
/// particle (both bodies) whose kernel touches this node, label `+1.0` grip / `-1.0`
/// rest. `node_pos`: this contact node's own grid position, used ONLY to CENTER the
/// point cloud before fitting (`x_p - node_pos`, not raw absolute grid coordinates) —
/// a real numerical-conditioning fix, not cosmetic: fitting directly against raw grid
/// coordinates (e.g. X≈32, Y≈10 rather than both near 0) left the Newton iteration
/// ill-conditioned enough to converge to a badly wrong plane at genuinely asymmetric
/// (edge/corner-like) point clouds, confirmed by direct instrumentation — recentering
/// fixed it. Returns `None` if both labels aren't present (no real interface at this
/// node, same meaning as the old gradient path's "no gradient" case).
///
/// Uses the paper's own recommended numerics, not guessed: uniform weights (`w_p=1` —
/// the paper tried several weighting schemes, none improved on this), penalty
/// `Γ=1e-7·Δx²·(1,1,0)` (only the plane's normal components are regularized, not its
/// offset), convergence on normal-direction change `1-n̂'·n̂<1e-5`, capped at 15
/// iterations (the paper's own cap, "to guard against needless iterations" on slow-
/// converging point clouds). Starting from `β⁽⁰⁾=0` makes the first NLLS update reduce
/// exactly to a closed-form linear-regression plane fit (the paper's own appendix
/// derives this) — so this is one iteration loop, not two separate code paths.
pub(super) fn fit_contact_normal_lr(
    points: &[(Vec2, f32)],
    node_pos: Vec2,
    grid_cell_size: f32,
) -> Option<Vec2> {
    let has_grip = points.iter().any(|&(_, c)| c > 0.0);
    let has_rest = points.iter().any(|&(_, c)| c < 0.0);
    if !has_grip || !has_rest {
        return None;
    }

    let dx2 = grid_cell_size * grid_cell_size;
    let penalty = [1.0e-7 * dx2, 1.0e-7 * dx2, 0.0];

    let mut beta = [0.0f32; 3];
    let mut prev_n: Option<Vec2> = None;

    for _ in 0..15 {
        let mut m = [[0.0f32; 3]; 3];
        let mut rhs = [0.0f32; 3];
        for &(pos, c) in points {
            let rel = pos - node_pos;
            let xp = [rel.x, rel.y, 1.0];
            // Clamped before exp() -- REAL BUG FOUND AND FIXED 2026-07-12: an
            // ill-constrained point cloud (e.g. very few points on one side) can send
            // the Newton iteration's beta, and therefore z, far enough that `ez` alone
            // overflows to f32::INFINITY, making `2.0*ez/(denom*denom)` compute
            // `inf/inf = NaN` -- confirmed via direct instrumentation, not theoretical
            // (a specific recurring 4-grip/30-rest point cloud produced `Vec2(NaN, NaN)`
            // on every substep). The logistic function saturates to exactly ±1 (and its
            // derivative to 0) long before |z|=40 in f32 anyway, so clamping changes
            // nothing about the converged answer -- it only removes the overflow path.
            let z: f32 = (xp[0] * beta[0] + xp[1] * beta[1] + xp[2] * beta[2]).clamp(-40.0, 40.0);
            let ez = (-z).exp();
            let denom = 1.0 + ez;
            let f = 2.0 / denom - 1.0;
            let sigma = 2.0 * ez / (denom * denom);
            let sigma_sq = sigma * sigma;
            for k in 0..3 {
                for l in 0..3 {
                    m[k][l] += sigma_sq * xp[k] * xp[l];
                }
                rhs[k] += sigma * (c - f) * xp[k];
            }
        }
        for k in 0..3 {
            m[k][k] += penalty[k];
            rhs[k] -= penalty[k] * beta[k];
        }

        let Some(delta) = solve3x3(m, rhs) else {
            break;
        };
        if delta.iter().any(|d| !d.is_finite()) {
            // Defensive: shouldn't happen now that `z` is clamped above, but a
            // degenerate point cloud (e.g. near-collinear) could still leave `m`
            // ill-conditioned enough for `solve3x3` to hand back a non-finite
            // update. Stop here and use whatever `prev_n` already converged to
            // (or `None`, handled the same as "no confident normal" elsewhere).
            break;
        }
        for k in 0..3 {
            beta[k] += delta[k];
        }

        let normal_raw = Vec2::new(beta[0], beta[1]);
        if normal_raw.length_squared() <= f32::EPSILON || !normal_raw.is_finite() {
            continue;
        }
        let n = normal_raw.normalize();
        if let Some(prev) = prev_n
            && 1.0 - n.dot(prev) < 1.0e-5
        {
            prev_n = Some(n);
            break;
        }
        prev_n = Some(n);
    }

    // Sign-consistency check against the ACTUAL labels the plane was fit from — real,
    // general safeguard, not a hardcoded direction. REAL BUG FOUND AND FIXED 2026-07-12:
    // Newton's method on the logistic-regression objective can converge (by this
    // function's own angle-based criterion) to a plateau whose normal direction is
    // backwards relative to the labels, especially for point clouds it takes many
    // iterations to resolve -- confirmed directly: forcing a hand-verified-correct
    // normal gave a clean, fully-decoupled frictionless result, while the UNCHECKED
    // fitted normal (same points, same iteration) reproduced the exact "fully stuck
    // regardless of friction" bug this whole feature was built to fix. The fitted
    // plane's normal is only meaningful up to which side is which -- verify it here by
    // projecting the ACTUAL point cloud onto it and confirming grip (label +1) points
    // project higher on average than rest (label -1); flip if not. Applies to every
    // point cloud/geometry uniformly, not tuned to this test.
    prev_n.map(|n| {
        let grip_mean: f32 = points
            .iter()
            .filter(|&&(_, c)| c > 0.0)
            .map(|&(p, _)| (p - node_pos).dot(n))
            .sum::<f32>()
            / points.iter().filter(|&&(_, c)| c > 0.0).count().max(1) as f32;
        let rest_mean: f32 = points
            .iter()
            .filter(|&&(_, c)| c < 0.0)
            .map(|&(p, _)| (p - node_pos).dot(n))
            .sum::<f32>()
            / points.iter().filter(|&&(_, c)| c < 0.0).count().max(1) as f32;
        if grip_mean < rest_mean { -n } else { n }
    })
}

/// Solves a general 3x3 linear system via Cramer's rule — closed-form is simpler and
/// faster than a general decomposition for this fixed, tiny size (one call per NLLS
/// iteration in `fit_contact_normal_lr`). Returns `None` if singular (determinant ~0);
/// the caller's Tikhonov-style penalty term keeps this from happening in practice.
fn solve3x3(m: [[f32; 3]; 3], rhs: [f32; 3]) -> Option<[f32; 3]> {
    let det3 = |a: [[f32; 3]; 3]| -> f32 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let det = det3(m);
    if det.abs() <= f32::EPSILON {
        return None;
    }
    let solve_col = |col: usize| -> f32 {
        let mut mm = m;
        for row in 0..3 {
            mm[row][col] = rhs[row];
        }
        det3(mm) / det
    };
    Some([solve_col(0), solve_col(1), solve_col(2)])
}

#[cfg(test)]
mod fit_contact_normal_lr_tests {
    use super::*;

    #[test]
    fn nairn_fig3c_corner_case_recovers_horizontal_normal() {
        // Replicates Nairn 2020's own Figure 3C worked example: a node sits near a
        // CORNER of material A (grip) but a FULL EDGE of material B (rest) below it.
        // The paper's own text (section 3.1, discussing Fig 3C) claims LR converges to
        // "the preferred, horizontal plane" here -- "despite the absence of material A
        // in the upper-right grid cell" -- specifically contrasting this with the older
        // grid-gradient/AG method, which tilts ~18 degrees. Real permanent regression:
        // confirms our LR implementation matches the paper's own claimed behavior on
        // its own worked example (verified 2026-07-14, not assumed) -- ruling this out
        // as the source of the real leading-edge skew found in `snake_on_terrain`-style
        // long-horizon runs (see project memory `snake_on_real_terrain_contact_instability`),
        // which turns out to be a low-point-count/imbalanced-sample confidence issue,
        // not a fundamental corner-topology failure of plain LR.
        let mut points = Vec::new();
        // Material B (rest): full horizontal edge, spans the WHOLE x range below the node.
        for i in 0..8 {
            for j in 0..4 {
                let x = 28.0 + i as f32 * 0.5;
                let y = 8.25 + j as f32 * 0.3;
                points.push((Vec2::new(x, y), -1.0));
            }
        }
        // Material A (grip): only a CORNER -- upper-left quadrant relative to the node,
        // absent entirely from the upper-right (matching the paper's own description).
        for i in 0..4 {
            for j in 0..4 {
                let x = 28.0 + i as f32 * 0.5;
                let y = 10.25 + j as f32 * 0.3;
                points.push((Vec2::new(x, y), 1.0));
            }
        }
        let node_pos = Vec2::new(30.0, 10.0);
        let n = fit_contact_normal_lr(&points, node_pos, 1.0).expect("should find a normal");
        assert!(
            n.x.abs() < 0.1,
            "LR should recover a near-horizontal normal on Nairn 2020's own Fig 3C \
             corner example (paper explicitly claims this over the older AG method) -- \
             got {n:?}"
        );
    }

    #[test]
    fn clean_horizontal_interface_36v36() {
        // Grip particles on a 6x6 grid at y in [10.25..11.75], rest on a 6x6 grid
        // at y in [8.25..9.75] -- a clean, perfectly flat, well-separated interface.
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                let x = 30.0 + i as f32 * 0.5;
                let y_grip = 10.25 + j as f32 * 0.3;
                let y_rest = 8.25 + j as f32 * 0.3;
                points.push((Vec2::new(x, y_grip), 1.0));
                points.push((Vec2::new(x, y_rest), -1.0));
            }
        }
        let node_pos = Vec2::new(32.0, 10.0);
        let n = fit_contact_normal_lr(&points, node_pos, 1.0).expect("should find a normal");
        assert!(
            n.x.abs() < 0.1,
            "expected near-vertical normal for a clean flat interface, got {n:?}"
        );
    }
}

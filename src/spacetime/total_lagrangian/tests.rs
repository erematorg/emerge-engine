use super::*;

/// The eG(2) scheme's matrices against Gross's Appendix C.1.2 (C.2):
/// `A^R_m = [0 1; -8 3]`, `A^R_p = [-1; 1]`, `A^p_m = [2 1/2; -8 4]`,
/// `A^p_p = [1/2; -1]`.
#[test]
fn galerkin2_matches_the_thesis_matrices() {
    let s = Galerkin2::new();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(s.a_m.col(0).x, 0.0) && close(s.a_m.col(1).x, 1.0));
    assert!(close(s.a_m.col(0).y, -8.0) && close(s.a_m.col(1).y, 3.0));
    assert!(close(s.a_p.x, -1.0) && close(s.a_p.y, 1.0));
    assert!(close(s.a_v.col(0).x, 2.0) && close(s.a_v.col(1).x, 0.5));
    assert!(close(s.a_v.col(0).y, -8.0) && close(s.a_v.col(1).y, 4.0));
    assert!(close(s.b_v.x, 0.5) && close(s.b_v.y, -1.0));
    assert!(close(s.w[0] + s.w[1], 1.0));
}

/// The discrete gradient makes `S : dC / 2` exactly the energy's
/// change, however large the step.
#[test]
fn the_algorithmic_stress_carries_the_exact_energy_change() {
    let solid = Elastic::isochoric(3.0, 11.0);
    let f0 = DMat2::from_cols(DVec2::new(1.1, 0.05), DVec2::new(-0.2, 0.95));
    let f1 = DMat2::from_cols(DVec2::new(0.8, 0.3), DVec2::new(0.1, 1.25));
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let s = solid.algorithmic_stress(c0, c1);
    let dc = c1 - c0;
    let work = 0.5 * (s.x_axis.dot(dc.x_axis) + s.y_axis.dot(dc.y_axis));
    let change = solid.energy(f1) - solid.energy(f0);
    assert!(
        (work - change).abs() < 1e-12 * change.abs().max(1.0),
        "{work} vs {change}"
    );
    assert!((s.x_axis.y - s.y_axis.x).abs() < 1e-12, "S symmetric");
}

/// `S(C)` is twice the derivative of `W(C)`, checked by central
/// differences.
#[test]
fn the_stress_is_twice_the_energy_derivative() {
    let solid = Elastic::isochoric(2.0, 7.0);
    let f = DMat2::from_cols(DVec2::new(1.05, 0.1), DVec2::new(0.07, 0.9));
    let c = f.transpose() * f;
    let s = solid.second_pk(c);
    let h = 1e-6;
    for (i, j) in [(0, 0), (1, 1), (0, 1)] {
        let mut e = DMat2::ZERO;
        e.col_mut(j)[i] = 1.0;
        e.col_mut(i)[j] = 1.0;
        let e = if i == j { 0.5 * e } else { e };
        let d = (solid.energy_of_c(c + h * e) - solid.energy_of_c(c - h * e)) / (2.0 * h);
        let s_e = s.x_axis.dot(e.x_axis) + s.y_axis.dot(e.y_axis);
        assert!(
            (2.0 * d - s_e).abs() < 1e-6,
            "({i}, {j}): {} vs {s_e}",
            2.0 * d
        );
    }
}
/// A block whose edges cut the cells: particles at four per cell, the
/// block spanning cells `[3.3, 9.15] x [2.6, 7.2]` of a 14 x 12 grid.
fn cut_block() -> (GridLayout, Vec<DVec2>, Vec<f64>) {
    let layout = GridLayout::new(14, 12, 1.0);
    let mut x = Vec::new();
    let (lo, hi) = (DVec2::new(3.3, 2.6), DVec2::new(9.15, 7.2));
    let mut y = lo.y + 0.25;
    while y < hi.y {
        let mut q = lo.x + 0.25;
        while q < hi.x {
            x.push(DVec2::new(q, y));
            q += 0.5;
        }
        y += 0.5;
    }
    let volume = vec![0.25; x.len()];
    (layout, x, volume)
}

/// The extended basis keeps the partition of unity and reproduces
/// linear fields at every particle, values and gradients: both momenta
/// rest on it.
#[test]
fn the_extended_basis_reproduces_linear_fields() {
    let (layout, x, volume) = cut_block();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&volume, &volume), &open, Some(0.75));
    let extended = grid.extension.iter().filter(|e| e.is_some()).count();
    assert!(extended > 0, "the cut block has degenerate faces");
    // Each unknown's position, from its stacked face.
    let mut position = vec![DVec2::ZERO; grid.len];
    for l in &grid.lattices {
        for j in 0..l.nj {
            for i in 0..l.ni {
                let unknown = grid.slot[l.start + i + l.ni * j];
                if unknown != usize::MAX {
                    position[unknown] =
                        (DVec2::new(i as f64, j as f64) + l.offset) * f64::from(layout.dx);
                }
            }
        }
    }
    for (p, &q) in x.iter().enumerate() {
        for a in 0..2 {
            let (mut one, mut at, mut grad_one) = (0.0, DVec2::ZERO, DVec2::ZERO);
            let mut grad_at = DMat2::ZERO;
            grid.visit(p, a, |k, w, g| {
                one += w;
                at += w * position[k];
                grad_one += g;
                grad_at += DMat2::from_cols(g * position[k].x, g * position[k].y);
            });
            assert!((one - 1.0).abs() < 1e-12, "particle {p}: sum {one}");
            assert!((at - q).length() < 1e-12, "particle {p}: {at} vs {}", q);
            assert!(grad_one.length() < 1e-12, "particle {p}: gradient sum");
            let identity = grad_at - DMat2::IDENTITY;
            assert!(
                identity.x_axis.length() + identity.y_axis.length() < 1e-11,
                "particle {p}: gradient of x is {grad_at}"
            );
        }
    }
}

/// The smallest eigenvalue of the consistent mass over its largest
/// diagonal entry, by inverse iteration: the original B-splines of a
/// cut block make it nearly singular, the extended ones lift it.
fn mass_conditioning(occupation: Option<f64>) -> f64 {
    let (layout, x, volume) = cut_block();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&volume, &volume), &open, occupation);
    let diag = grid.mass_diagonal(&volume, 0.0);
    let largest = diag.iter().fold(0.0f64, |m, &d| m.max(d));
    let apply = |y: &[f64]| grid.mass_times(y, &volume, 0.0);
    let mut y: Vec<f64> = (0..grid.len)
        .map(|k| ((k * 7919) % 101) as f64 - 50.0)
        .collect();
    let mut lambda = 0.0;
    for _ in 0..40 {
        let norm = y.iter().map(|e| e * e).sum::<f64>().sqrt();
        y.iter_mut().for_each(|e| *e /= norm);
        let z = conjugate_gradient(&apply, &y, &diag, &grid.active);
        lambda = 1.0 / z.iter().map(|e| e * e).sum::<f64>().sqrt();
        y = z;
    }
    lambda / largest
}

#[test]
fn the_extended_basis_lifts_the_mass_conditioning() {
    let original = mass_conditioning(None);
    let extended = mass_conditioning(Some(0.75));
    println!(
        "smallest eigenvalue over the largest diagonal: original {original:.3e}, extended {extended:.3e}"
    );
    assert!(
        extended > 10.0 * original,
        "{original:.3e} -> {extended:.3e}"
    );
}

/// Two blocks side by side on the same rows, apart: the band is as wide
/// as one block's, each numbered as a block of its own.
#[test]
fn bodies_apart_keep_the_band_of_one() {
    let block = |left: f64| {
        let mut x = Vec::new();
        let mut y = 2.25;
        while y < 7.0 {
            let mut q = left + 0.25;
            while q < left + 5.0 {
                x.push(DVec2::new(q, y));
                q += 0.5;
            }
            y += 0.5;
        }
        x
    };
    let layout = GridLayout::new(30, 10, 1.0);
    let open = |_: usize, _: usize, _: usize| None;
    let width = |x: &[DVec2]| {
        let volume = vec![0.25; x.len()];
        let grid = Grid::new(&layout, x, (&volume, &volume), &open, Some(0.75));
        grid.empty_band(x.len()).width
    };
    let one = block(3.0);
    let mut two = one.clone();
    two.extend(block(20.0));
    assert_eq!(width(&two), width(&one));
}

/// The band factor solves a symmetric positive definite band system,
/// its unknowns renumbered: `A x = b` back to rounding.
#[test]
fn the_band_factor_solves_its_system() {
    let n = 9;
    let width = 2;
    // A renumbering and a band matrix in the band's numbering.
    let order = vec![3, 0, 7, 1, 8, 2, 6, 4, 5];
    let dense = |i: usize, j: usize| -> f64 {
        match i.abs_diff(j) {
            0 => 6.0 + i as f64 * 0.1,
            1 => -1.5,
            2 => 0.25,
            _ => 0.0,
        }
    };
    let mut band = Band {
        n,
        width,
        data: vec![0.0; n * (width + 1)],
        order: order.clone(),
    };
    for i in 0..n {
        for j in i.saturating_sub(width)..=i {
            let slot = band.at(i, j);
            band.data[slot] = dense(i, j);
        }
    }
    band.factor();
    let b: Vec<f64> = (0..n).map(|k| (k as f64 * 0.7).sin()).collect();
    let x = band.solve(&b);
    for k in 0..n {
        let i = order[k];
        let ax: f64 = (0..n).map(|m| dense(i, order[m]) * x[m]).sum();
        assert!((ax - b[k]).abs() < 1e-12, "row {k}: {ax} vs {}", b[k]);
    }
}

/// The neo-Hookean law on the step (gate (d)): the discrete gradient makes
/// `S : dC / 2` exactly the energy's change at a large step, and the
/// analytic step tangent agrees with central differences of the step stress
/// to 1e-6 (criteria fixed before measuring).
#[test]
fn the_neo_hookean_step_stress_carries_the_energy_change_and_its_tangent() {
    let solid = Elastic::neo_hookean(2.9, 1.3);
    let f0 = DMat2::from_cols(DVec2::new(1.1, 0.05), DVec2::new(-0.2, 0.95));
    let f1 = DMat2::from_cols(DVec2::new(1.4, 0.3), DVec2::new(0.1, 0.7));
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let s = solid.algorithmic_stress(c0, c1);
    let dc = c1 - c0;
    let colon = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
    let work = 0.5 * colon(s, dc);
    let change = solid.energy(f1) - solid.energy(f0);
    assert!(
        (work - change).abs() < 1e-12 * change.abs().max(1.0),
        "{work} vs {change}"
    );
    let direction = DMat2::from_cols(DVec2::new(0.2, -0.1), DVec2::new(-0.1, 0.4));
    let exact = solid.step_stress_tangent(c0, c1, 1e-3, direction);
    let fd = solid.step_stress_tangent_by_differences(c0, c1, 1e-3, direction);
    let off = colon(exact - fd, exact - fd).sqrt();
    assert!(off <= 1e-6 * colon(exact, exact).sqrt(), "off {off:e}");
}

/// Gate (g): a neo-Hookean block released from a planar stretch of 1.3 in
/// free space rings with its energy kept: over 20 steps the kinetic plus
/// stored energy stays the start's within what the solves allow (their
/// residual work, `StepReport::residual_work`, accumulated) and the sums'
/// round-off (criterion fixed before measuring).
#[test]
fn a_released_neo_hookean_block_keeps_its_energy() {
    let (layout, reference, v0) = cut_block();
    let n = reference.len();
    let solids = vec![Elastic::neo_hookean(2.9, 1.3); n];
    let centre = reference.iter().copied().sum::<DVec2>() / n as f64;
    let stretch = DMat2::from_diagonal(DVec2::new(1.3, 1.0 / 1.3));
    let mut x: Vec<DVec2> = reference
        .iter()
        .map(|&q| centre + stretch * (q - centre))
        .collect();
    let mut v = vec![DVec2::ZERO; n];
    let mut f = vec![stretch; n];
    let mass = v0.clone();
    let energy = |v: &[DVec2], f: &[DMat2]| -> (f64, f64) {
        let kinetic: f64 = (0..n).map(|p| 0.5 * mass[p] * v[p].length_squared()).sum();
        let stored: f64 = (0..n).map(|p| v0[p] * solids[p].energy(f[p])).sum();
        (kinetic, stored)
    };
    let start = energy(&v, &f);
    let settings = ConservingSettings::default();
    let free = |_: usize, _: usize, _: usize| None;
    let mut cache = None;
    let mut allowed = 0.0;
    let mut largest_kinetic = 0.0f64;
    for k in 0..20 {
        let report = step_total_lagrangian(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &mass,
                v0: &v0,
            },
            &reference,
            &solids,
            (DVec2::ZERO, None, 0.2),
            (&free, &settings),
            &mut cache,
        );
        assert!(report.converged, "step {k} did not converge");
        allowed += report.residual_work;
        let now = energy(&v, &f);
        largest_kinetic = largest_kinetic.max(now.0);
        let round_off = n as f64 * f64::EPSILON * (now.0 + now.1 + start.1);
        let balance = now.0 + now.1 - start.0 - start.1;
        assert!(
            balance.abs() <= allowed + round_off,
            "step {k}: balance {balance:e} beyond {:e}",
            allowed + round_off
        );
    }
    println!(
        "neo-Hookean block: stored at the start {:.6e}, largest kinetic {largest_kinetic:.6e}",
        start.1
    );
    assert!(largest_kinetic > 0.1 * start.1, "the block rings");
}

/// Meunier et al. 2008's RTV 141 Ogden set (MPa), with a bulk modulus of
/// 1e3 times its shear modulus chosen for these gates (no measured `K`).
fn ogden_solid() -> Elastic {
    use crate::matter::materials::solid::ogden::Ogden;
    let law = Ogden::new(&[(0.46, 1.4), (0.00027, 10.0), (-0.0074, -3.3)], 0.0);
    Elastic::ogden(Ogden::new(&law.terms[..law.count], 1e3 * law.shear()))
}

/// Ogden's law on the step: the discrete gradient makes `S : dC / 2`
/// exactly the energy's change at a large step, and the analytic step
/// tangent agrees with central differences of the step stress to 1e-6 (the
/// neo-Hookean gate's criteria).
#[test]
fn the_ogden_step_stress_carries_the_energy_change_and_its_tangent() {
    let solid = ogden_solid();
    let f0 = DMat2::from_cols(DVec2::new(1.1, 0.05), DVec2::new(-0.2, 0.95));
    let f1 = DMat2::from_cols(DVec2::new(1.4, 0.3), DVec2::new(0.1, 0.7));
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let s = solid.algorithmic_stress(c0, c1);
    let dc = c1 - c0;
    let colon = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
    let work = 0.5 * colon(s, dc);
    let change = solid.energy(f1) - solid.energy(f0);
    assert!(
        (work - change).abs() < 1e-12 * change.abs().max(1.0),
        "{work} vs {change}"
    );
    let direction = DMat2::from_cols(DVec2::new(0.2, -0.1), DVec2::new(-0.1, 0.4));
    let exact = solid.step_stress_tangent(c0, c1, 1e-3, direction);
    let fd = solid.step_stress_tangent_by_differences(c0, c1, 1e-3, direction);
    let off = colon(exact - fd, exact - fd).sqrt();
    assert!(off <= 1e-6 * colon(exact, exact).sqrt(), "off {off:e}");
}

/// An Ogden block released from a planar stretch of 1.3 in free space rings
/// with its energy kept: over 20 steps the kinetic plus stored energy stays
/// the start's within what the solves allow and the sums' round-off (the
/// neo-Hookean gate's criterion).
#[test]
fn a_released_ogden_block_keeps_its_energy() {
    let (layout, reference, v0) = cut_block();
    let n = reference.len();
    let solids = vec![ogden_solid(); n];
    let centre = reference.iter().copied().sum::<DVec2>() / n as f64;
    let stretch = DMat2::from_diagonal(DVec2::new(1.3, 1.0 / 1.3));
    let mut x: Vec<DVec2> = reference
        .iter()
        .map(|&q| centre + stretch * (q - centre))
        .collect();
    let mut v = vec![DVec2::ZERO; n];
    let mut f = vec![stretch; n];
    let mass = v0.clone();
    let energy = |v: &[DVec2], f: &[DMat2]| -> (f64, f64) {
        let kinetic: f64 = (0..n).map(|p| 0.5 * mass[p] * v[p].length_squared()).sum();
        let stored: f64 = (0..n).map(|p| v0[p] * solids[p].energy(f[p])).sum();
        (kinetic, stored)
    };
    let start = energy(&v, &f);
    let settings = ConservingSettings::default();
    let free = |_: usize, _: usize, _: usize| None;
    let mut cache = None;
    let mut allowed = 0.0;
    let mut largest_kinetic = 0.0f64;
    for k in 0..20 {
        let report = step_total_lagrangian(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &mass,
                v0: &v0,
            },
            &reference,
            &solids,
            (DVec2::ZERO, None, 0.2),
            (&free, &settings),
            &mut cache,
        );
        assert!(report.converged, "step {k} did not converge");
        allowed += report.residual_work;
        let now = energy(&v, &f);
        largest_kinetic = largest_kinetic.max(now.0);
        let round_off = n as f64 * f64::EPSILON * (now.0 + now.1 + start.1);
        let balance = now.0 + now.1 - start.0 - start.1;
        assert!(
            balance.abs() <= allowed + round_off,
            "step {k}: balance {balance:e} beyond {:e}",
            allowed + round_off
        );
    }
    println!(
        "Ogden block: stored at the start {:.6e}, largest kinetic {largest_kinetic:.6e}",
        start.1
    );
    assert!(largest_kinetic > 0.1 * start.1, "the block rings");
}

/// Water's modulus at 20 C (IAPWS SR6-08, `LiquidWater`), Pa.
fn water_solid() -> Elastic {
    let water =
        crate::matter::materials::liquid::liquid_water::LiquidWater::at(293.15).expect("in range");
    Elastic::barotropic(water.isentropic_bulk())
}

/// The barotropic law on the step: `S : dC / 2` is exactly the energy's
/// change at a step of the size water sees (0.3 % volume change), and the
/// analytic tangent agrees with central differences to 1e-6 (the criteria
/// of the other laws' gates).
#[test]
fn the_barotropic_step_stress_carries_the_energy_change_and_its_tangent() {
    let solid = water_solid();
    let f0 = DMat2::from_cols(DVec2::new(1.001, 0.0005), DVec2::new(-0.0004, 0.9985));
    let f1 = DMat2::from_cols(DVec2::new(0.998, 0.001), DVec2::new(0.0006, 1.0015));
    let (c0, c1) = (f0.transpose() * f0, f1.transpose() * f1);
    let s = solid.algorithmic_stress(c0, c1);
    let dc = c1 - c0;
    let colon = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
    let work = 0.5 * colon(s, dc);
    let change = solid.energy(f1) - solid.energy(f0);
    assert!(
        (work - change).abs() < 1e-10 * change.abs().max(1.0),
        "{work} vs {change}"
    );
    let direction = DMat2::from_cols(DVec2::new(0.2, -0.1), DVec2::new(-0.1, 0.4));
    let exact = solid.step_stress_tangent(c0, c1, 1e-6, direction);
    let fd = solid.step_stress_tangent_by_differences(c0, c1, 1e-6, direction);
    let off = colon(exact - fd, exact - fd).sqrt();
    assert!(off <= 1e-6 * colon(exact, exact).sqrt(), "off {off:e}");
}

/// A water block released from a uniform 0.5 % compression in free space
/// rings acoustically with its energy kept: over 20 steps of 1e-4 s the
/// kinetic plus stored energy stays the start's within the solves' bound
/// and the sums' round-off (the criterion of the other laws' gates).
#[test]
fn a_released_water_block_keeps_its_energy() {
    let (layout, reference, v0) = cut_block();
    let n = reference.len();
    let solids = vec![water_solid(); n];
    let centre = reference.iter().copied().sum::<DVec2>() / n as f64;
    let squeeze = DMat2::from_diagonal(DVec2::splat(0.995f64.sqrt()));
    let mut x: Vec<DVec2> = reference
        .iter()
        .map(|&q| centre + squeeze * (q - centre))
        .collect();
    let mut v = vec![DVec2::ZERO; n];
    let mut f = vec![squeeze; n];
    let mass: Vec<f64> = v0.iter().map(|&volume| 998.2 * volume).collect();
    let energy = |v: &[DVec2], f: &[DMat2]| -> (f64, f64) {
        let kinetic: f64 = (0..n).map(|p| 0.5 * mass[p] * v[p].length_squared()).sum();
        let stored: f64 = (0..n).map(|p| v0[p] * solids[p].energy(f[p])).sum();
        (kinetic, stored)
    };
    let start = energy(&v, &f);
    let settings = ConservingSettings::default();
    let free = |_: usize, _: usize, _: usize| None;
    let mut cache = None;
    let mut allowed = 0.0;
    let mut largest_kinetic = 0.0f64;
    for k in 0..20 {
        let report = step_total_lagrangian(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &mass,
                v0: &v0,
            },
            &reference,
            &solids,
            (DVec2::ZERO, None, 1e-4),
            (&free, &settings),
            &mut cache,
        );
        assert!(report.converged, "step {k} did not converge");
        allowed += report.residual_work;
        let now = energy(&v, &f);
        largest_kinetic = largest_kinetic.max(now.0);
        let round_off = n as f64 * f64::EPSILON * (now.0 + now.1 + start.1);
        let balance = now.0 + now.1 - start.0 - start.1;
        assert!(
            balance.abs() <= allowed + round_off,
            "step {k}: balance {balance:e} beyond {:e}",
            allowed + round_off
        );
    }
    println!(
        "water block: stored at the start {:.6e}, largest kinetic {largest_kinetic:.6e}",
        start.1
    );
    assert!(largest_kinetic > 0.01 * start.1, "the block rings");
}

/// The released water block's 20 steps with every value multiplied by its
/// unit factor, lengths `s_l`, times `s_t`, masses `s_m` (positions and the
/// cell `s_l`, velocities `s_l / s_t`, mass per depth `s_m / s_l`, areas
/// `s_l^2`, the modulus `s_m / (s_l s_t^2)`, the step `s_t`), and the end
/// state divided back: positions, velocities, `F`, and the steps' residual
/// work over `s_m s_l / s_t^2` (energy per depth).
fn scaled_water_block(
    (s_l, s_t, s_m): (f64, f64, f64),
) -> (Vec<DVec2>, Vec<DVec2>, Vec<DMat2>, f64) {
    let (layout, reference, v0) = cut_block();
    // The cell is `f32`: a power of two scales it exactly.
    let layout = GridLayout::new(layout.nx, layout.ny, layout.dx * s_l as f32);
    let reference: Vec<DVec2> = reference.iter().map(|&q| q * s_l).collect();
    let v0: Vec<f64> = v0.iter().map(|&a| a * s_l * s_l).collect();
    let n = reference.len();
    let bulk = crate::matter::materials::liquid::liquid_water::LiquidWater::at(293.15)
        .expect("in range")
        .isentropic_bulk();
    let solids = vec![Elastic::barotropic(bulk * s_m / (s_l * s_t * s_t)); n];
    let centre = reference.iter().copied().sum::<DVec2>() / n as f64;
    let squeeze = DMat2::from_diagonal(DVec2::splat(0.995f64.sqrt()));
    let mut x: Vec<DVec2> = reference
        .iter()
        .map(|&q| centre + squeeze * (q - centre))
        .collect();
    let mut v = vec![DVec2::ZERO; n];
    let mut f = vec![squeeze; n];
    let mass: Vec<f64> = v0
        .iter()
        .map(|&volume| 998.2 * s_m / (s_l * s_l * s_l) * volume)
        .collect();
    let settings = ConservingSettings::default();
    let free = |_: usize, _: usize, _: usize| None;
    let mut cache = None;
    let mut residual = 0.0;
    for k in 0..20 {
        let report = step_total_lagrangian(
            &layout,
            Particles {
                x: &mut x,
                v: &mut v,
                f: &mut f,
                mass: &mass,
                v0: &v0,
            },
            &reference,
            &solids,
            (DVec2::ZERO, None, 1e-6 * s_t),
            (&free, &settings),
            &mut cache,
        );
        assert!(report.converged, "step {k} did not converge");
        residual += report.residual_work;
    }
    (
        x.iter().map(|&q| q / s_l).collect(),
        v.iter().map(|&q| q * s_t / s_l).collect(),
        f,
        residual * s_t * s_t / (s_m * s_l),
    )
}

/// The unit-scaling test: with length, time and mass units rescaled by
/// powers of two, every rounding of the step is the same unless a
/// unit-carrying constant enters it, so the dimensionless end state is
/// bit-identical (criterion fixed before running; a difference is a unit
/// error, not a tolerance question). Factors `2^-3, 2^5, 2^9`, so the
/// modulus scales by `2^2` and mass per depth by `2^12`. The same with
/// factors that are not powers of two is printed, not held.
#[test]
fn the_water_step_is_unit_scale_invariant() {
    let si = scaled_water_block((1.0, 1.0, 1.0));
    let scaled = scaled_water_block((0.125, 32.0, 512.0));
    assert_eq!(si.0, scaled.0, "positions");
    assert_eq!(si.1, scaled.1, "velocities");
    assert_eq!(si.2, scaled.2, "F");
    assert_eq!(si.3, scaled.3, "residual work");
    let other = scaled_water_block((0.37, 3.1, 7.3));
    let worst = |a: &[DVec2], b: &[DVec2]| {
        let size = a.iter().map(|q| q.length()).fold(0.0, f64::max);
        a.iter()
            .zip(b)
            .map(|(p, q)| (*p - *q).length())
            .fold(0.0, f64::max)
            / size
    };
    println!(
        "unit factors 0.37, 3.1, 7.3: largest relative difference of positions {:.2e}, of velocities {:.2e}",
        worst(&si.0, &other.0),
        worst(&si.1, &other.1)
    );
}

/// The consistent mass assembled once (`Grid::consistent_mass_times`) is the
/// particle pass's `sum_p m_p N_k sum_m N_m y_m` up to the sums' order: on
/// the cut block, against the pass written out here, to 64 eps of the
/// product's size (criterion fixed before running); and a second mass
/// vector reassembles it (the kept matrix is never stale).
#[test]
fn the_assembled_mass_is_the_particle_pass() {
    let (layout, x, volume) = cut_block();
    let open = |_: usize, _: usize, _: usize| None;
    let grid = Grid::new(&layout, &x, (&volume, &volume), &open, Some(0.75));
    let y: Vec<f64> = (0..grid.len)
        .map(|k| ((k * 7 % 13) as f64 - 6.0) / 6.0)
        .collect();
    let pass = |mass: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; grid.len];
        for (p, &m) in mass.iter().enumerate() {
            for a in 0..2 {
                let (value, _) = grid.read(&y, p, a);
                grid.visit(p, a, |k, w, _| out[k] += w * m * value);
            }
        }
        out
    };
    let heavier: Vec<f64> = volume
        .iter()
        .enumerate()
        .map(|(p, v)| v * (1.0 + 0.1 * (p % 3) as f64))
        .collect();
    for mass in [&volume, &heavier, &volume] {
        let (expected, got) = (pass(mass), grid.consistent_mass_times(&y, mass));
        let size = expected.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let off = expected
            .iter()
            .zip(&got)
            .fold(0.0f64, |m, (e, g)| m.max((e - g).abs()));
        assert!(off <= 64.0 * f64::EPSILON * size, "off {off:e} of {size:e}");
    }
}

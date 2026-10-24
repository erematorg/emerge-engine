//! A research variant of the step: the eG(2) time discretization, kept as a
//! probe.

use super::*;

/// The time scheme of the enhanced Galerkin method eG(2): the continuous
/// Galerkin method cG(2) in time (quadratic trial, linear test functions,
/// two-point Gauss rule) with the enhanced assumed gradient that makes it
/// conserve energy, from Gross, *Conserving Time Integrators for Nonlinear
/// Elastodynamics*, Dr.-Ing. thesis, TU Kaiserslautern (2004), sections
/// 3.2 and 6.3-6.8. The matrices are those of his eqs. 3.29-3.33 and 3.40,
/// computed here from their definitions (his Appendix C.1 lists them; for
/// k = 1 these definitions give exactly his C.1).
#[cfg(all(test, feature = "experimental"))]
pub(crate) struct Galerkin2 {
    /// The Gauss weights on the master step `[0, 1]`.
    pub(crate) w: [f64; 2],
    /// Trial functions (nodes 0, 1/2, 1) and their `alpha` derivatives at
    /// the Gauss points: `[l][J]`.
    pub(crate) m: [[f64; 3]; 2],
    pub(crate) dm: [[f64; 3]; 2],
    /// `A^R_m = A' A^-1 A'` and `A^R_p = b' - A' A^-1 b` (his 3.33).
    pub(crate) a_m: DMat2,
    pub(crate) a_p: DVec2,
    /// The nodal velocities from the nodal displacements (his 3.39 in
    /// velocities): `v = A^-1 A' d / h - A^-1 b v_1`.
    pub(crate) a_v: DMat2,
    pub(crate) b_v: DVec2,
    /// The test functions at the Gauss points, `W~[I][l]` (his 3.12).
    pub(crate) tilde: DMat2,
    /// The real part of the eigenvalues of `A^R_m^-1 W~ D_w Phi`, the
    /// preconditioner's stiffness shift.
    pub(crate) shift: f64,
}

#[cfg(all(test, feature = "experimental"))]
impl Galerkin2 {
    pub(crate) fn new() -> Self {
        let r = 3.0f64.sqrt() / 6.0;
        let xi = [0.5 - r, 0.5 + r];
        let w = [0.5, 0.5];
        let trial = |a: f64| {
            [
                2.0 * (a - 0.5) * (a - 1.0),
                -4.0 * a * (a - 1.0),
                2.0 * a * (a - 0.5),
            ]
        };
        let slope = |a: f64| [4.0 * a - 3.0, -8.0 * a + 4.0, 4.0 * a - 1.0];
        let test = |a: f64| [1.0 - a, a];
        // Exact integrals of the polynomial products (degree <= 3) by the
        // three-point Gauss rule on [0, 1].
        let s = (0.6f64).sqrt();
        let nodes = [0.5 * (1.0 - s), 0.5, 0.5 * (1.0 + s)];
        let weights = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];
        let integral = |f: &dyn Fn(f64) -> f64| -> f64 {
            nodes.iter().zip(&weights).map(|(&t, &c)| c * f(t)).sum()
        };
        // A[I][J] = int M~_I M_{J+1}, A'[I][J] = int M~_I M'_{J+1}, b and
        // b' the same with M_1.
        let mut a = [[0.0; 2]; 2];
        let mut a1 = [[0.0; 2]; 2];
        let mut b = [0.0; 2];
        let mut b1 = [0.0; 2];
        for i in 0..2 {
            for j in 0..2 {
                a[i][j] = integral(&|t| test(t)[i] * trial(t)[j + 1]);
                a1[i][j] = integral(&|t| test(t)[i] * slope(t)[j + 1]);
            }
            b[i] = integral(&|t| test(t)[i] * trial(t)[0]);
            b1[i] = integral(&|t| test(t)[i] * slope(t)[0]);
        }
        // DMat2 is column-major: from_cols of the columns.
        let mat = |m: [[f64; 2]; 2]| {
            DMat2::from_cols(DVec2::new(m[0][0], m[1][0]), DVec2::new(m[0][1], m[1][1]))
        };
        let (a, a1) = (mat(a), mat(a1));
        let (b, b1) = (DVec2::from_array(b), DVec2::from_array(b1));
        let a_inv = a.inverse();
        let a_m = a1 * a_inv * a1;
        let a_p = b1 - a1 * (a_inv * b);
        let a_v = a_inv * a1;
        let b_v = a_inv * b;
        let m = [trial(xi[0]), trial(xi[1])];
        let dm = [slope(xi[0]), slope(xi[1])];
        let tilde = mat([
            [test(xi[0])[0], test(xi[1])[0]],
            [test(xi[0])[1], test(xi[1])[1]],
        ]);
        let phi = mat([[m[0][1], m[0][2]], [m[1][1], m[1][2]]]);
        let weights = DMat2::from_diagonal(DVec2::from_array(w));
        let product = a_m.inverse() * tilde * weights * phi;
        let shift = 0.5 * (product.x_axis.x + product.y_axis.y);
        Self {
            w,
            m,
            dm,
            a_m,
            a_p,
            a_v,
            b_v,
            tilde,
            shift,
        }
    }
}

/// One step of eG(2) on the reference configuration's grid, the time
/// counterpart of `step_total_lagrangian` (same grid, cache, walls and
/// particle update), with the body force at the step's two Gauss times.
/// The unknowns are the grid's displacements at `alpha = 1/2` and `1`; the
/// residual is Gross's 3.32 in displacements (`A^R_m 1 = -A^R_q`), the
/// internal force at each Gauss point uses his enhanced assumed gradient
/// (6.31-6.34): per particle, with `C_J = F_J^T F_J` at the three time
/// nodes, the assumed `C(alpha) = sum M_J C_J`, the `cG` rate `C' = F'^T F +
/// F^T F'`, `G = W(C_3) - W(C_1) - sum_l w_l 1/2 S(C(xi_l)) : C'(xi_l)` and
/// `N = sum_l w_l dC/dalpha(xi_l) : C'(xi_l)`, the stress `S(C(xi_l)) + 2 G
/// / N dC/dalpha(xi_l)`. Adapted: the preconditioner is this engine's band
/// factor of `M / h + h s K` with `s` the real part of the scheme's
/// eigenvalues (his Newton uses the exact unsymmetric tangent and a direct
/// solver, Remark 3.2); Newton starts from rest or the free flight, the
/// better admissible one. Returns the report and the body force's work
/// over the step (`sum_l w_l f_ext(xi_l) . dq/dalpha(xi_l)`).
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn step_total_lagrangian_eg2(
    layout: &GridLayout,
    particles: Particles<'_>,
    reference: &[DVec2],
    solids: &[Elastic],
    (body, dt): ([&[DVec2]; 2], f64),
    (held, settings): (Held<'_>, &ConservingSettings),
    cache: &mut Option<LagrangianCache>,
) -> (StepReport, f64) {
    let Particles { x, v, f, mass, v0 } = particles;
    let n = x.len();
    let clock = std::time::Instant::now();
    let LagrangianCache { grid, factors, .. } = cache.get_or_insert_with(|| LagrangianCache {
        grid: Grid::new(layout, reference, (mass, v0), held, settings.occupation),
        factors: Factors::default(),
        contacts: Vec::new(),
        kept: Vec::new(),
        heat: Vec::new(),
        traces: Vec::new(),
    });
    let grid = &*grid;
    let eps = settings.epsilon;
    let scheme = Galerkin2::new();
    let len = grid.len;
    let h = dt;
    let mask = |y: &mut [f64]| {
        for (value, &on) in y
            .iter_mut()
            .zip(grid.active.iter().chain(grid.active.iter()))
        {
            if !on {
                *value = 0.0;
            }
        }
    };
    // The grid velocity: the consistent-mass projection, as `solve`.
    let mut rhs = vec![0.0; len];
    for (p, (&m, vp)) in mass.iter().zip(v.iter()).enumerate() {
        for a in 0..2 {
            let s = m * vp[a];
            grid.visit(p, a, |k, w, _| rhs[k] += w * s);
        }
    }
    for (value, &on) in rhs.iter_mut().zip(&grid.active) {
        if !on {
            *value = 0.0;
        }
    }
    let mass_factor = factors
        .mass
        .get_or_insert_with(|| grid.mass_band(mass, eps));
    let v_n = preconditioned_cg(
        &|y: &[f64]| grid.mass_times(y, mass, eps),
        &rhs,
        &|r: &[f64]| mass_factor.solve(r),
        &grid.active,
    );
    let momentum = grid.mass_times_unmasked(&v_n, mass, eps);
    // The body force on the grid at each Gauss time.
    let external: [Vec<f64>; 2] = [0, 1].map(|l| {
        parallel_scatter(len, n, |p, acc| {
            for (a, component) in body[l][p].to_array().into_iter().enumerate() {
                let s = mass[p] * component;
                grid.visit(p, a, |k, w, _| acc[k] += w * s);
            }
        })
    });
    let f_now: &[DMat2] = f;
    // Each particle's displacement gradients at nodes 2 and 3.
    let gradients = |d: &[f64], p: usize| -> [DMat2; 2] {
        [0, 1].map(|j| {
            let part = &d[j * len..(j + 1) * len];
            let (_, gu) = grid.read(part, p, 0);
            let (_, gv) = grid.read(part, p, 1);
            DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y))
        })
    };
    let contract = |a: DMat2, b: DMat2| a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis);
    // The internal force at both Gauss times, stacked.
    let internal = |d: &[f64]| -> Vec<f64> {
        parallel_scatter(2 * len, n, |p, acc| {
            let solid = solids[p];
            let [g2, g3] = gradients(d, p);
            let nodes = [f_now[p], f_now[p] + g2, f_now[p] + g3];
            let c_nodes = nodes.map(|m| m.transpose() * m);
            let mut s = [DMat2::ZERO; 2];
            let mut fl = [DMat2::ZERO; 2];
            let mut dc_assumed = [DMat2::ZERO; 2];
            let mut dc_rate = [DMat2::ZERO; 2];
            let mut c_assumed = [DMat2::ZERO; 2];
            for l in 0..2 {
                let (m, dm) = (scheme.m[l], scheme.dm[l]);
                fl[l] = nodes[0] * m[0] + nodes[1] * m[1] + nodes[2] * m[2];
                let rate = nodes[0] * dm[0] + nodes[1] * dm[1] + nodes[2] * dm[2];
                c_assumed[l] = c_nodes[0] * m[0] + c_nodes[1] * m[1] + c_nodes[2] * m[2];
                dc_assumed[l] = c_nodes[0] * dm[0] + c_nodes[1] * dm[1] + c_nodes[2] * dm[2];
                dc_rate[l] = rate.transpose() * fl[l] + fl[l].transpose() * rate;
                s[l] = solid.second_pk(c_assumed[l]);
            }
            let mut gap = solid.energy_of_c(c_nodes[2]) - solid.energy_of_c(c_nodes[0]);
            let mut norm = 0.0;
            for l in 0..2 {
                gap -= scheme.w[l] * 0.5 * contract(s[l], dc_rate[l]);
                norm += scheme.w[l] * contract(dc_assumed[l], dc_rate[l]);
            }
            for l in 0..2 {
                let stress = if norm.abs() > 1e-28 {
                    s[l] + (2.0 * gap / norm) * dc_assumed[l]
                } else {
                    s[l]
                };
                let first = v0[p] * fl[l] * stress;
                let block = &mut acc[l * len..(l + 1) * len];
                for a in 0..2 {
                    let row = first.row(a);
                    grid.visit(p, a, |k, _, g| block[k] += row.dot(g));
                }
            }
        })
    };
    // Gross 3.32 in displacements: (A^R_m (x) M) d / h + A^R_p (x) M v_n +
    // h (W~ (x) I) [w_l (f_int - f_ext)(xi_l)].
    let forces_part = |forces: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; 2 * len];
        for i in 0..2 {
            for l in 0..2 {
                let c = h * scheme.tilde.col(l)[i] * scheme.w[l];
                for k in 0..len {
                    out[i * len + k] += c * (forces[l * len + k] - external[l][k]);
                }
            }
        }
        out
    };
    let mass_part = |d: &[f64]| -> Vec<f64> {
        let md = [
            grid.mass_times_unmasked(&d[..len], mass, eps),
            grid.mass_times_unmasked(&d[len..], mass, eps),
        ];
        let mut out = vec![0.0; 2 * len];
        for i in 0..2 {
            for k in 0..len {
                out[i * len + k] =
                    (scheme.a_m.col(0)[i] * md[0][k] + scheme.a_m.col(1)[i] * md[1][k]) / h
                        + scheme.a_p[i] * momentum[k];
            }
        }
        out
    };
    let residual = |d: &[f64]| -> Vec<f64> {
        let mut out = mass_part(d);
        for (o, e) in out.iter_mut().zip(forces_part(&internal(d))) {
            *o += e;
        }
        mask(&mut out);
        out
    };
    let largest = |y: &[f64]| {
        y.iter().fold(0.0f64, |m, e| {
            if e.is_nan() || m.is_nan() {
                f64::NAN
            } else {
                m.max(e.abs())
            }
        })
    };
    // Admissible: no particle inverted at the nodes or the Gauss points.
    let admissible = |d: &[f64]| {
        (0..n).all(|p| {
            let [g2, g3] = gradients(d, p);
            let nodes = [f_now[p], f_now[p] + g2, f_now[p] + g3];
            nodes.iter().all(|m| m.determinant() > 0.0)
                && scheme.m.iter().all(|m| {
                    (nodes[0] * m[0] + nodes[1] * m[1] + nodes[2] * m[2]).determinant() > 0.0
                })
        })
    };
    let zero = vec![0.0; 2 * len];
    let scale = largest(&mass_part(&zero))
        .max(largest(&forces_part(&vec![0.0; 2 * len])))
        .max(largest(&forces_part(&internal(&zero))))
        .max(f64::MIN_POSITIVE);
    // The preconditioner: `(A^R_m (x) I)(I (x) (M / h + h s K))`.
    let shift = scheme.shift;
    let band = grid.approximate_jacobian((mass, v0, solids), eps, 1.0 / (h * h * shift), &[]);
    let a_m_inv = scheme.a_m.inverse();
    let precondition = |y: &[f64]| -> Vec<f64> {
        let mut z = vec![0.0; 2 * len];
        for k in 0..len {
            let r = a_m_inv * DVec2::new(y[k], y[len + k]);
            z[k] = r.x;
            z[len + k] = r.y;
        }
        let mut out = band.solve(&z[..len]);
        out.extend(band.solve(&z[len..]));
        for value in out.iter_mut() {
            *value /= h * shift;
        }
        mask(&mut out);
        out
    };
    let active2: Vec<bool> = grid
        .active
        .iter()
        .chain(grid.active.iter())
        .copied()
        .collect();
    // Starts: rest, or the free flight `d_J = alpha_J h v_n`.
    let flight: Vec<f64> = (0..2 * len)
        .map(|k| {
            let alpha = if k < len { 0.5 } else { 1.0 };
            if active2[k] {
                alpha * h * v_n[k % len]
            } else {
                0.0
            }
        })
        .collect();
    let mut d = zero.clone();
    let mut r = residual(&d);
    if admissible(&flight) {
        let r_flight = residual(&flight);
        if largest(&r_flight) < largest(&r) {
            (d, r) = (flight, r_flight);
        }
    }
    let mut report = StepReport {
        start_residual: largest(&r) / scale,
        grid_speed: largest(&v_n),
        ..Default::default()
    };
    let started = clock.elapsed().as_secs_f64();
    loop {
        let worst = largest(&r);
        report.residual = worst / scale;
        if report.residual <= settings.newton_tolerance {
            report.converged = true;
            break;
        }
        if report.newton >= settings.max_newton {
            break;
        }
        report.newton += 1;
        let jacobian = |delta: &[f64]| -> Vec<f64> {
            let size = delta.iter().fold(0.0f64, |m, e| m.max(e.abs()));
            if size == 0.0 {
                return vec![0.0; 2 * len];
            }
            let base = d.iter().fold(0.0f64, |m, e| m.max(e.abs()));
            let step = 1e-6 * (1.0 + base) / size;
            let plus: Vec<f64> = d.iter().zip(delta).map(|(a, b)| a + step * b).collect();
            let minus: Vec<f64> = d.iter().zip(delta).map(|(a, b)| a - step * b).collect();
            let mut out = mass_part(delta);
            // The mass part is linear; drop its constant term.
            for i in 0..2 {
                for k in 0..len {
                    out[i * len + k] -= scheme.a_p[i] * momentum[k];
                }
            }
            let (fp, fm) = (internal(&plus), internal(&minus));
            let difference: Vec<f64> = fp
                .iter()
                .zip(&fm)
                .map(|(a, b)| (a - b) / (2.0 * step))
                .collect();
            // The forces' part without the external force: its linear map.
            for i in 0..2 {
                for l in 0..2 {
                    let c = h * scheme.tilde.col(l)[i] * scheme.w[l];
                    for k in 0..len {
                        out[i * len + k] += c * difference[l * len + k];
                    }
                }
            }
            mask(&mut out);
            out
        };
        let minus_r: Vec<f64> = r.iter().map(|e| -e).collect();
        let (delta, iterations) = gmres(
            &jacobian,
            &minus_r,
            (&precondition, &active2),
            settings.gmres_tolerance,
            settings,
        );
        report.gmres += iterations;
        let mut t = 1.0;
        loop {
            let trial: Vec<f64> = d.iter().zip(&delta).map(|(a, b)| a + t * b).collect();
            if !admissible(&trial) {
                t *= 0.5;
                if t < 1e-12 {
                    break;
                }
                continue;
            }
            let rt = residual(&trial);
            if largest(&rt) < worst || t < 1e-3 {
                d = trial;
                r = rt;
                break;
            }
            t *= 0.5;
        }
    }
    // The nodal velocity at the step's end (his 3.39 in velocities).
    let v_end: Vec<f64> = (0..len)
        .map(|k| {
            if grid.active[k] {
                (scheme.a_v.col(0).y * d[k] + scheme.a_v.col(1).y * d[len + k]) / h
                    - scheme.b_v.y * v_n[k]
            } else {
                0.0
            }
        })
        .collect();
    // The body force's work: sum_l w_l f_ext(xi_l) . dq/dalpha(xi_l).
    let mut work = 0.0;
    for ((dm, w), force) in scheme.dm.iter().zip(scheme.w).zip(&external) {
        for (k, f) in force.iter().enumerate() {
            work += w * f * (dm[1] * d[k] + dm[2] * d[len + k]);
        }
    }
    let dv: Vec<f64> = (0..len).map(|k| v_end[k] - v_n[k]).collect();
    let end = &d[len..];
    for p in 0..n {
        let mut step_x = DVec2::ZERO;
        let mut step_v = DVec2::ZERO;
        let (_, gu) = grid.read(end, p, 0);
        let (_, gv) = grid.read(end, p, 1);
        for a in 0..2 {
            step_x[a] = grid.read(end, p, a).0;
            step_v[a] = grid.read(&dv, p, a).0;
        }
        x[p] += step_x;
        v[p] += step_v;
        f[p] += DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
    }
    report.passes = 1;
    report.phases = [
        0.0,
        started,
        0.0,
        clock.elapsed().as_secs_f64() - started,
        0.0,
    ];
    (report, work)
}

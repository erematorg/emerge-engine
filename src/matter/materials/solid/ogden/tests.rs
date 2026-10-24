use super::*;

/// Meunier et al. 2008's 3-term set for the RTV 141 silicone (Table 2, read
/// on the page image; the third `mu` printed `mu_1` is `mu_3`), MPa; with a
/// bulk modulus chosen for these law gates only (no measured `K`).
fn meunier(bulk: f64) -> Ogden {
    Ogden::new(&[(0.46, 1.4), (0.00027, 10.0), (-0.0074, -3.3)], bulk)
}

fn rotation(angle: f64) -> DMat2 {
    DMat2::from_angle(angle)
}

/// `C` with squared stretches `(l1, l2)` along axes turned by `angle`.
fn turned(l1: f64, l2: f64, angle: f64) -> DMat2 {
    let q = rotation(angle);
    q * DMat2::from_diagonal(DVec2::new(l1, l2)) * q.transpose()
}

fn states() -> Vec<DMat2> {
    vec![
        turned(1.69, 0.64, 0.37),
        turned(4.0, 0.3, -1.1),
        turned(1.2, 1.2 * (1.0 + 1e-5), 0.2),
        turned(1.2, 1.2 * (1.0 + 1e-7), 0.2),
        turned(1.2, 1.2 * (1.0 + 1e-9), 0.2),
        turned(1.44, 1.44, 0.0),
        DMat2::IDENTITY,
        turned(0.25, 9.0, 0.8),
    ]
}

fn directions() -> [DMat2; 3] {
    [
        DMat2::from_cols(DVec2::X, DVec2::ZERO),
        DMat2::from_cols(DVec2::ZERO, DVec2::Y),
        DMat2::from_cols(DVec2::Y, DVec2::X),
    ]
}

fn size(a: DMat2) -> f64 {
    colon(a, a).sqrt()
}

/// Gate (a): `S = 2 dW/dC` by central differences, to 1e-7 of `|S|` plus the
/// differences' own floor (criteria fixed before measuring, for every gate).
#[test]
fn the_stress_is_twice_the_energy_derivative() {
    let law = meunier(50.0);
    for c in states() {
        let s = ogden_second_pk(c, &law);
        let h = 1e-6;
        for e in directions() {
            let d = (ogden_energy(c + e * h, &law) - ogden_energy(c - e * h, &law)) / (2.0 * h);
            let expected = 0.5 * colon(s, e);
            assert!(
                (d - expected).abs() <= 1e-7 * size(s).max(law.shear()),
                "C {c:?}: {d} against {expected}"
            );
        }
    }
}

/// Gate (b): the tangent against central differences of `S`, at distinct,
/// nearly equal (relative spacing 1e-5, 1e-7, 1e-9) and equal stretches, to
/// 1e-6 of the tangent's size plus the differences' floor.
#[test]
fn the_tangent_is_the_stress_derivative_at_any_spacing() {
    let law = meunier(50.0);
    for c in states() {
        let h = 1e-6;
        for e in directions() {
            let fd =
                (ogden_second_pk(c + e * h, &law) - ogden_second_pk(c - e * h, &law)) * (0.5 / h);
            let exact = ogden_second_pk_tangent(c, e, &law);
            let off = size(fd - exact);
            assert!(
                off <= 1e-6 * size(exact).max(law.bulk),
                "C {c:?}, dC {e:?}: off {off:e} of {:e}",
                size(exact)
            );
        }
    }
}

/// Gate (c): at equal stretches the divided difference `s` equals
/// Connolly's eq. 25 limit, `l^-2 (1/2 gamma_bb - beta_b - 1/2 gamma_ab)`,
/// computed here from its own terms, to round-off.
#[test]
fn the_divided_difference_meets_the_repeated_stretch_limit() {
    let law = meunier(50.0);
    for l in [0.36, 1.0, 1.44, 6.25] {
        let c = DMat2::from_diagonal(DVec2::splat(l));
        let p = principal(&law, c);
        let (beta, s) = coefficients(&law, &p);
        let (_, g) = moments(&law, &p);
        let sum_g = (g[0] + g[1] + g[2]) / 9.0;
        let gamma_bb = g[1] - 2.0 * g[1] / 3.0 + sum_g;
        let gamma_ab = -(g[0] + g[1]) / 3.0 + sum_g;
        let limit = (0.5 * gamma_bb - beta[1] - 0.5 * gamma_ab) / (l * l);
        assert!(
            (s - limit).abs() <= 1e-14 * limit.abs().max(law.shear() / (l * l)),
            "l {l}: {s:e} against {limit:e}"
        );
    }
}

/// Gate (d): the exact increment against the energies' difference at a
/// large step (to their round-off), and at tiny steps against the midpoint
/// stress's work `1/2 S(C_mid) : dC`, equal to `O(|dC|^3)`. That reference
/// has its own floor: near the natural state `S` is a sum of terms of size
/// `mu + K` that cancel, so its round-off is about `eps (mu + K)` absolute,
/// and the work's `1/2 |dS| |dC|`; bounded here by `16 eps (mu + K) |dC|`
/// (derived, not fitted).
#[test]
fn the_increment_is_the_energy_change_at_any_size() {
    let law = meunier(50.0);
    for c0 in states() {
        let dir = DMat2::from_cols(DVec2::new(0.3, -0.2), DVec2::new(-0.2, 0.5));
        let large = c0 + dir * 0.2;
        let inc = ogden_energy_increment(c0, large, &law);
        let diff = ogden_energy(large, &law) - ogden_energy(c0, &law);
        assert!(
            (inc - diff).abs() <= 1e-12 * diff.abs().max(ogden_energy(c0, &law).abs()).max(1.0),
            "C {c0:?}: {inc} vs {diff}"
        );
        for step in [1e-6, 1e-9, 1e-12] {
            let c1 = c0 + dir * step;
            let inc = ogden_energy_increment(c0, c1, &law);
            let mid = ogden_second_pk(0.5 * (c0 + c1), &law);
            let work = 0.5 * colon(mid, c1 - c0);
            let floor = 16.0 * f64::EPSILON * (law.shear() + law.bulk) * size(c1 - c0);
            assert!(
                (inc - work).abs() <= 1e-10 * work.abs() + law.bulk * step.powi(3) + floor,
                "C {c0:?}, step {step}: {inc:e} against {work:e}"
            );
        }
    }
}

/// Gate (e): one term with `alpha = 2` is the neo-Hookean isochoric energy
/// `mu/2 (Ibar_1 - 3)`, `Ibar_1 = J^(-2/3) (tr C + 1)` in plane strain
/// (Connolly's eq. 39 with `lambda_3 = 1`), to round-off.
#[test]
fn one_term_of_alpha_two_is_the_neo_hookean_isochoric_energy() {
    let (mu, bulk) = (0.7, 30.0);
    let law = Ogden::new(&[(mu, 2.0)], bulk);
    for c in states() {
        let j = c.determinant().sqrt();
        let ibar = j.powf(-2.0 / 3.0) * (c.x_axis.x + c.y_axis.y + 1.0);
        let expected = 0.5 * mu * (ibar - 3.0) + 0.5 * bulk * j.ln().powi(2);
        let w = ogden_energy(c, &law);
        assert!(
            (w - expected).abs() <= 1e-14 * expected.abs().max(mu),
            "{w} against {expected}"
        );
    }
}

/// Gate (f): at `J = 1`, `F = diag(lambda, 1/lambda)` (pure shear for
/// `lambda > 1`, plane-strain compression for `lambda < 1`, Meunier's Figs.
/// 3b and 3d kinematics), `tau_1 - tau_2 = sum_i mu_i (lambda^alpha_i -
/// lambda^-alpha_i)` whatever `K` (derived on the page from eqs. 12, 16), so
/// the incompressible test's nominal stress `(tau_1 - tau_2) / lambda` is
/// Ogden's closed form, to round-off.
#[test]
fn pure_shear_and_plane_strain_compression_follow_the_closed_form() {
    for bulk in [10.0, 1e4] {
        let law = meunier(bulk);
        for stretch in [0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.15] {
            let f = DMat2::from_diagonal(DVec2::new(stretch, 1.0 / stretch));
            let tau = ogden_kirchhoff(f, &law);
            let nominal = (tau.x_axis.x - tau.y_axis.y) / stretch;
            let closed: f64 = law
                .each()
                .map(|(mu, alpha)| mu * (stretch.powf(alpha - 1.0) - stretch.powf(-alpha - 1.0)))
                .sum();
            assert!(
                (nominal - closed).abs() <= 1e-13 * closed.abs().max(law.shear()),
                "K {bulk}, stretch {stretch}: {nominal} against {closed}"
            );
        }
    }
}

/// Gate (g): frame indifference, `S(Q C Q^T) = Q S(C) Q^T`, and the energy
/// unchanged, to round-off.
#[test]
fn the_law_is_frame_indifferent() {
    let law = meunier(50.0);
    let c = turned(2.25, 0.49, 0.0);
    let s = ogden_second_pk(c, &law);
    for angle in [0.3, 1.2, -2.0] {
        let q = rotation(angle);
        let turned_c = q * c * q.transpose();
        let off = size(ogden_second_pk(turned_c, &law) - q * s * q.transpose());
        assert!(off <= 1e-13 * size(s), "off {off:e}");
        let w = (ogden_energy(turned_c, &law) - ogden_energy(c, &law)).abs();
        assert!(w <= 1e-14 * ogden_energy(c, &law));
    }
}

/// Meunier et al. 2008's pure-shear points (Fig. 10b), read off the page
/// image by eye by this session (stretch and nominal stress, MPa: an
/// uncertainty of about 0.01 in both, declared; the paper prints no table),
/// against the 3-term Ogden set fitted by the authors on all five tests,
/// through this law with the thickness free (`tau_2 = 0`, Newton on the
/// thickness stretch) and `K = 1e3 mu` and `1e5 mu`. A probe: it prints.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_meunier_pure_shear() {
    let data = [
        (1.02, 0.05),
        (1.05, 0.10),
        (1.10, 0.15),
        (1.15, 0.20),
        (1.22, 0.25),
        (1.30, 0.30),
        (1.38, 0.35),
        (1.48, 0.40),
        (1.57, 0.45),
        (1.66, 0.50),
        (1.75, 0.55),
        (1.83, 0.60),
        (1.91, 0.65),
        (1.97, 0.70),
        (2.03, 0.75),
        (2.08, 0.80),
        (2.12, 0.85),
        (2.16, 0.90),
    ];
    for factor in [1e3, 1e5] {
        let law = meunier(factor * meunier(0.0).shear());
        let mut rms = 0.0;
        for &(stretch, measured) in &data {
            // The free thickness `b`: `tau_22(diag(stretch, b)) = 0`.
            let mut b = 1.0 / stretch;
            for _ in 0..60 {
                let tau = |b: f64| {
                    ogden_kirchhoff(DMat2::from_diagonal(DVec2::new(stretch, b)), &law)
                        .y_axis
                        .y
                };
                let h = 1e-7 * b;
                let slope = (tau(b + h) - tau(b - h)) / (2.0 * h);
                b -= tau(b) / slope;
            }
            let f = DMat2::from_diagonal(DVec2::new(stretch, b));
            let nominal = ogden_kirchhoff(f, &law).x_axis.x / stretch;
            rms += (nominal - measured).powi(2);
            println!(
                "K {factor:e} mu, stretch {stretch:.2}: read {measured:.2} MPa, Ogden {nominal:.4} MPa ({:+.1} %), J {:.6}",
                100.0 * (nominal - measured) / measured,
                stretch * b
            );
        }
        println!(
            "K {factor:e} mu: rms {:.4} MPa over {} read points",
            (rms / data.len() as f64).sqrt(),
            data.len()
        );
    }
}

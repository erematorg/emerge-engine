//! The compressible neo-Hookean solid (Simo and Pister's form) in plane
//! strain: what the precise path's rubber-like solid is, its stored energy,
//! its stresses and their derivatives, in `f64`.
//!
//! Sources (read through the reading agent's report,
//! `NEOHOOKEAN_SOURCES_REPORT`, each with its page and equation there): the
//! energy `W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2` (Ghiba, Martin,
//! Apetrii and Neff 2025, arXiv 2509.00126, sec. 2.13, the "original
//! Simo-Pister energy"; Azevedo et al. 2020, arXiv 2009.06408, eq. B.1;
//! Jiang et al. 2016, MPM course, eq. 46); its second Piola-Kirchhoff stress
//! `S = mu (I - C^-1) + lambda ln J C^-1` (Azevedo eq. B.2); its material
//! tangent `2 dS/dC = lambda C^-1 (x) C^-1 + 2 (mu - lambda ln J) II`, `II =
//! -dC^-1/dC` (Azevedo eqs. B.3-B.4); the Kirchhoff stress
//! `tau = mu (B - I) + lambda ln J I` (Korobeynikov, Larichkin and Neff 2025,
//! arXiv 2506.22244, eqs. 47 and 49).
//!
//! Plane strain, derived in the report and checked there to round-off: with
//! `F_33 = 1`, `I1 = tr C_2 + 1` and `J = det F_2`, so
//! `W = mu/2 (tr C_2 - 2) - mu ln J + lambda/2 (ln J)^2` exactly, and the in-plane `S`, `tau` and
//! tangent are the formulas above on the 2 x 2 block; the out-of-plane
//! reaction `S_33 = tau_33 = lambda ln J` does no work.
//! Only planar tension (pure shear, `F = diag(a, 1/a)`) is native in plane
//! strain; uniaxial and equibiaxial tension need the out-of-plane stretch
//! free, which this law does not carry.
//!
//! Not the explicit path's `NeoHookeanMaterial`, whose stress is a 2D
//! isochoric split (`(mu/J) dev B + (lambda + mu) ln J I`) for which no source
//! was found: a different law at finite strain with the same linearisation
//! (pure shear `F = diag(2, 1/2)`: `tau = mu diag(3, -0.75)` here, `mu
//! diag(1.875, -1.875)` there). The two are kept apart.

use glam::{DMat2, DVec2};

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

/// `ln J = 1/2 ln det C`.
fn ln_j(c: DMat2) -> f64 {
    0.5 * c.determinant().ln()
}

/// The stored energy per reference volume at `C` (in-plane block), with
/// Lame's moduli `(lambda, mu)`; infinite where `det C <= 0`.
pub(crate) fn neo_hookean_energy(c: DMat2, lambda: f64, mu: f64) -> f64 {
    let det = c.determinant();
    if det <= 0.0 {
        return f64::INFINITY;
    }
    let l = 0.5 * det.ln();
    0.5 * mu * (c.x_axis.x + c.y_axis.y - 2.0) - mu * l + 0.5 * lambda * l * l
}

/// The second Piola-Kirchhoff stress `S = mu (I - C^-1) + lambda ln J C^-1`.
pub(crate) fn neo_hookean_second_pk(c: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let inverse = c.inverse();
    (DMat2::IDENTITY - inverse) * mu + inverse * (lambda * ln_j(c))
}

/// The Kirchhoff stress `tau = mu (B - I) + lambda ln J I`, `B = F F^T`, `J =
/// det F`.
pub(crate) fn neo_hookean_kirchhoff(f: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let b = f * f.transpose();
    (b - DMat2::IDENTITY) * mu + DMat2::from_diagonal(DVec2::splat(lambda * f.determinant().ln()))
}

/// `dS` along `dC`: half the material tangent applied to `dC`, `dS = lambda/2
/// (C^-1 : dC) C^-1 + (mu - lambda ln J) C^-1 dC C^-1`. Derived on the page
/// from `S`: `d ln J = 1/2 C^-1 : dC` and `d C^-1 = -C^-1 dC C^-1`, which is
/// Azevedo's eq. B.3 halved.
pub(crate) fn neo_hookean_second_pk_tangent(c: DMat2, dc: DMat2, lambda: f64, mu: f64) -> DMat2 {
    let inverse = c.inverse();
    inverse * (0.5 * lambda * colon(inverse, dc))
        + (inverse * dc * inverse) * (mu - lambda * ln_j(c))
}

/// `W(C_1) - W(C_0)` from the increment itself, so its round-off is relative
/// to the change (the discrete gradient divides it by `|dC|^2`).
///
/// Derived on the page, not sourced (the report's gap 4; Gonzalez 2000 eqs.
/// 11-12 cover any `W(C)`, and the gates cross-check against it): `W(C_1) -
/// W(C_0) = mu/2 tr dC - mu dl + lambda/2 dl (l_0 + l_1)`, `l = ln J`, with
/// `dl = l_1 - l_0 = 1/2 ln1p(d det / det C_0)` and the determinant's
/// increment exact for 2 x 2 matrices, `d det = tr(adj(C_0) dC) + det dC`.
pub(crate) fn neo_hookean_energy_increment(c0: DMat2, c1: DMat2, lambda: f64, mu: f64) -> f64 {
    let dc = c1 - c0;
    let det0 = c0.determinant();
    let adjugate_dot =
        c0.y_axis.y * dc.x_axis.x - c0.y_axis.x * dc.x_axis.y - c0.x_axis.y * dc.y_axis.x
            + c0.x_axis.x * dc.y_axis.y;
    let d_det = adjugate_dot + dc.determinant();
    let dl = 0.5 * (d_det / det0).ln_1p();
    let l0 = 0.5 * det0.ln();
    0.5 * mu * (dc.x_axis.x + dc.y_axis.y) - mu * dl + 0.5 * lambda * dl * (2.0 * l0 + dl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::materials::solid::corotated::stress::corotated_second_pk;

    const LAMBDA: f64 = 2.9;
    const MU: f64 = 1.3;

    fn strained() -> DMat2 {
        let f = DMat2::from_cols(DVec2::new(1.3, 0.2), DVec2::new(-0.15, 0.85));
        f.transpose() * f
    }

    fn directions() -> [DMat2; 3] {
        [
            DMat2::from_cols(DVec2::X, DVec2::ZERO),
            DMat2::from_cols(DVec2::ZERO, DVec2::Y),
            DMat2::from_cols(DVec2::Y, DVec2::X),
        ]
    }

    /// Gate (a): `S = 2 dW/dC`, by central differences, to 1e-8 relative
    /// (criteria fixed before measuring, for every gate below).
    #[test]
    fn the_stress_is_twice_the_energy_derivative() {
        let c = strained();
        let s = neo_hookean_second_pk(c, LAMBDA, MU);
        let h = 1e-6;
        for e in directions() {
            let d = (neo_hookean_energy(c + e * h, LAMBDA, MU)
                - neo_hookean_energy(c - e * h, LAMBDA, MU))
                / (2.0 * h);
            let expected = 0.5 * colon(s, e);
            assert!(
                (d - expected).abs() <= 1e-8 * colon(s, s).sqrt(),
                "{d} against {expected}"
            );
        }
    }

    /// Gate (b): the tangent against central differences of `S`, to 1e-7.
    #[test]
    fn the_tangent_is_the_stress_derivative() {
        let c = strained();
        let h = 1e-6;
        for e in directions() {
            let fd = (neo_hookean_second_pk(c + e * h, LAMBDA, MU)
                - neo_hookean_second_pk(c - e * h, LAMBDA, MU))
                * (0.5 / h);
            let exact = neo_hookean_second_pk_tangent(c, e, LAMBDA, MU);
            let off = colon(fd - exact, fd - exact).sqrt();
            assert!(off <= 1e-7 * colon(exact, exact).sqrt(), "off {off:e}");
        }
    }

    /// Gate (c): the exact increment against the energies' difference at a
    /// large step (to their round-off), and at tiny steps against the
    /// midpoint stress's work `1/2 S(C_mid) : dC`, which it equals to
    /// `O(|dC|^3)` (Gonzalez eq. 22), where the energies' difference has lost
    /// its digits.
    #[test]
    fn the_increment_is_the_energy_change_at_any_size() {
        let c0 = strained();
        let dir = DMat2::from_cols(DVec2::new(0.3, -0.2), DVec2::new(-0.2, 0.5));
        let large = c0 + dir * 0.4;
        let inc = neo_hookean_energy_increment(c0, large, LAMBDA, MU);
        let diff = neo_hookean_energy(large, LAMBDA, MU) - neo_hookean_energy(c0, LAMBDA, MU);
        assert!(
            (inc - diff).abs() <= 1e-14 * diff.abs().max(1.0),
            "{inc} vs {diff}"
        );
        for size in [1e-6, 1e-9, 1e-12] {
            let c1 = c0 + dir * size;
            let inc = neo_hookean_energy_increment(c0, c1, LAMBDA, MU);
            let mid = neo_hookean_second_pk(0.5 * (c0 + c1), LAMBDA, MU);
            let work = 0.5 * colon(mid, c1 - c0);
            assert!(
                (inc - work).abs() <= 1e-12 * work.abs() + 1e-3 * size.powi(3),
                "size {size}: {inc:e} against {work:e}"
            );
        }
    }

    /// Gate (e): planar tension (pure shear), `F = diag(a, 1/a)`, `J = 1`:
    /// `W = mu/2 (a^2 + a^-2 - 2)`, `P_11 = -P_22 = mu (a - 1/a)`, `tau =
    /// diag(mu (a^2 - 1), mu (a^-2 - 1))`, `lambda` dropping out (derived on
    /// the page in the report from the sourced `S`, `P`, `tau`; their values
    /// at `mu = 1.3`, `a = 1.5`: `P_11 = 1.083333`, `W = 0.451389`, where the
    /// explicit path's 2D split would give `tau = +-1.173611`), to 1e-14.
    #[test]
    fn planar_tension_follows_its_closed_form() {
        for a in [1.0, 1.5, 2.0, 5.0] {
            let f = DMat2::from_diagonal(DVec2::new(a, 1.0 / a));
            let c = f.transpose() * f;
            let p = f * neo_hookean_second_pk(c, LAMBDA, MU);
            let tau = neo_hookean_kirchhoff(f, LAMBDA, MU);
            let w = neo_hookean_energy(c, LAMBDA, MU);
            let exact = (
                MU * (a - 1.0 / a),
                0.5 * MU * (a * a + 1.0 / (a * a) - 2.0),
                (MU * (a * a - 1.0), MU * (1.0 / (a * a) - 1.0)),
            );
            let scale = MU * a * a;
            assert!((p.x_axis.x - exact.0).abs() <= 1e-14 * scale);
            assert!((p.y_axis.y + exact.0).abs() <= 1e-14 * scale);
            assert!((w - exact.1).abs() <= 1e-14 * scale);
            assert!((tau.x_axis.x - exact.2.0).abs() <= 1e-14 * scale);
            assert!((tau.y_axis.y - exact.2.1).abs() <= 1e-14 * scale);
            assert!(tau.x_axis.y.abs() <= 1e-14 * scale);
            // `S_33 = lambda ln J = 0` at `J = 1`.
            assert!((LAMBDA * f.determinant().ln()).abs() <= 1e-14 * scale);
        }
        let a = 1.5;
        let f = DMat2::from_diagonal(DVec2::new(a, 1.0 / a));
        let p = f * neo_hookean_second_pk(f.transpose() * f, LAMBDA, MU);
        assert!((p.x_axis.x - 1.083_333_333_333).abs() < 1e-12);
    }

    /// Gate (f): the linear limit is the corotated law's (both reduce to
    /// linear plane-strain elasticity with the same Lame moduli): at a strain
    /// `eps` their stresses differ by `O(eps^2)`, the difference over `eps^2`
    /// staying bounded as `eps` falls 1e-2, 1e-3, 1e-4.
    #[test]
    fn the_linear_limit_is_the_corotated_law() {
        let shape = DMat2::from_cols(DVec2::new(0.7, 0.3), DVec2::new(0.3, -0.4));
        let mut ratios = Vec::new();
        for eps in [1e-2, 1e-3, 1e-4] {
            let c = DMat2::IDENTITY + shape * (2.0 * eps);
            let d = neo_hookean_second_pk(c, LAMBDA, MU) - corotated_second_pk(c, LAMBDA, MU);
            ratios.push(colon(d, d).sqrt() / (eps * eps));
        }
        println!("|S_nh - S_corotated| / eps^2: {ratios:?}");
        assert!(ratios.iter().all(|r| *r < 2.0 * ratios[0] + 1.0));
        assert!((ratios[2] / ratios[1] - 1.0).abs() < 0.05, "{ratios:?}");
    }

    /// The first comparison with a real experiment: Treloar's planar tension
    /// (pure shear) of vulcanised rubber, nominal stress `P` against stretch,
    /// as tabulated by Ricker and Wriggers 2023 (supplementary file 11, CC BY;
    /// MPa, the original stretches), read through the reading agent's report
    /// (`TRELOAR_DATA_REPORT`). Planar tension holds the width (the
    /// out-of-plane axis, `F_33 = 1`), stretches by `a` and leaves the
    /// thickness free: `F = diag(a, b)` with `P_22 = 0`, i.e. `mu (b^2 - 1) +
    /// lambda ln(ab) = 0` (derived from `S`), solved by Newton for `b`; then
    /// `P_11 = mu (a - 1/a) + lambda ln(ab) / a`, which tends to the
    /// incompressible `mu (a - a^-3)` as `lambda / mu` grows.
    ///
    /// Declared: no Poisson's ratio or bulk modulus was measured, so `lambda =
    /// 1e4 mu` (nearly incompressible; the incompressible closed form is
    /// printed beside it); `mu` is the least-squares fit of `P = mu (a -
    /// a^-3)` to the points with `a <= 1.5` (the small-strain modulus, a
    /// measured coefficient of this rubber, not tuned to the large-stretch
    /// points). Expected before running: the neo-Hookean law misses
    /// Treloar's upturn above a stretch of about 2 to 3. Report only.
    ///
    /// Measured, as the relative error `|P_nh - P_exp| / P_exp` per point:
    /// `mu = 0.4045` MPa fitted on the five points with `a <= 1.5` (the Ogden
    /// fits' small-strain modulus is 0.39 to 0.41); in sample, within 5 %
    /// but at the first point (13.9 % at 1.04, where the sources disagree
    /// most); OUT of sample (1.85 to 4.94), 11.1 to 21.3 % high (worst at
    /// 2.97: the rubber softens there, which this law cannot), back to 6.0 %
    /// at 4.94 with the upturn; rms 0.146 MPa over the 13 points (the
    /// published Ogden fits: 0.03 to 0.08). `lambda = 1e4 mu` is a declared
    /// choice, no bulk modulus being measured; its check, `J` within 1e-4 of
    /// 1 at every point. A measured gap of the law, not of the solver.
    #[test]
    #[ignore = "real-experiment report: run with --ignored --nocapture"]
    fn probe_treloar_planar_tension() {
        let data = [
            (1.036544, 0.065100),
            (1.113805, 0.165353),
            (1.184235, 0.251341),
            (1.300373, 0.342713),
            (1.437191, 0.432369),
            (1.845503, 0.614107),
            (2.377238, 0.792785),
            (2.965335, 0.975862),
            (3.460044, 1.170418),
            (3.944203, 1.340856),
            (4.349349, 1.530862),
            (4.673152, 1.697720),
            (4.939253, 1.881396),
        ];
        let shape = |a: f64| a - a.powi(-3);
        let (num, den) = data
            .iter()
            .filter(|(a, _)| *a <= 1.5)
            .fold((0.0, 0.0), |(n, d), &(a, p)| {
                (n + p * shape(a), d + shape(a) * shape(a))
            });
        let mu = num / den;
        let lambda = 1e4 * mu;
        println!(
            "Treloar planar tension: mu {mu:.4} MPa (fit to a <= 1.5), lambda {lambda:.1} MPa"
        );
        let mut rms = 0.0;
        for &(a, measured) in &data {
            // Newton for the free thickness `b`: `mu (b^2 - 1) + lambda ln(ab)`.
            let mut b = 1.0 / a;
            for _ in 0..50 {
                let r = mu * (b * b - 1.0) + lambda * (a * b).ln();
                b -= r / (2.0 * mu * b + lambda / b);
            }
            let p11 = mu * (a - 1.0 / a) + lambda * (a * b).ln() / a;
            let incompressible = mu * shape(a);
            rms += (p11 - measured).powi(2);
            println!(
                "  a {a:.4}{}: measured {measured:.4} MPa, neo-Hookean {p11:.4} (incompressible {incompressible:.4}), ratio {:.3}, relative error {:.1} %, J {:.6}",
                if a <= 1.5 { " (fitted)" } else { "" },
                p11 / measured,
                100.0 * (p11 - measured).abs() / measured,
                a * b
            );
        }
        println!(
            "  rms error {:.4} MPa over {} points",
            (rms / data.len() as f64).sqrt(),
            data.len()
        );
    }
}

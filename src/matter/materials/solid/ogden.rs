//! Ogden's compressible solid in principal stretches, in plane strain: what
//! the precise path's rubber is when its stress-stretch curve stiffens (the
//! neo-Hookean's does not), its stored energy, its stresses and their
//! derivatives, in `f64`.
//!
//! Source: Connolly, Mackenzie and Gorash, Comput. Mech. 64:1273-1288
//! (2019), CC-BY, read in full: the split `psi = W(lbar_1, lbar_2, lbar_3) +
//! U(J)` with `lbar_a = J^(-1/3) lambda_a` (eqs. 7-8); `S_iso = sum_a beta_a
//! lambda_a^-2 N_a (x) N_a`, `beta_a = lbar_a dW/dlbar_a - 1/3 sum_b lbar_b
//! dW/dlbar_b` (eqs. 11-12); `S_vol = J U'(J) C^-1` (eq. 13); the material
//! tangent `C_iso` (eq. 22) with its coefficients `gamma_ab` (eq. 35) and
//! the repeated-stretch limit (eq. 25); `C_vol` (eq. 27). Ogden's energy in
//! the convention of Meunier et al. 2008 (Polymer Testing 27:765-777, Table
//! 1, read on the page image): `W = sum_i mu_i / alpha_i (lbar_1^alpha_i +
//! lbar_2^alpha_i + lbar_3^alpha_i - 3)`, so `lbar_a dW/dlbar_a = sum_i mu_i
//! lbar_a^alpha_i`, its small-strain shear modulus `1/2 sum_i mu_i alpha_i`.
//!
//! Adaptations and derivations, declared:
//! - Plane strain, `F_33 = 1`: `lambda_3 = 1`, `N_3 = e_3`, but `lbar_3 =
//!   J^(-1/3)` still enters `W` and `beta_a`; the in-plane block of the 3D
//!   formulas is kept (the pairs with index 3 fill only components with an
//!   index 3, and `dC_33 = 0`).
//! - For Ogden's energy, derived on the page from eq. 35: `gamma_ab =
//!   delta_ab G_a - 1/3 (G_a + G_b) + 1/9 sum_c G_c`, `G_a = sum_i mu_i alpha_i
//!   lbar_a^alpha_i`.
//! - The coefficient of eq. 22's second term is the divided difference `s =
//!   (g_1 - g_2) / (l_1 - l_2)` of `g = beta / lambda^2` over `l =
//!   lambda^2`, and `beta_1 - beta_2 = sum_i mu_i (lbar_1^alpha_i -
//!   lbar_2^alpha_i)` (the mean term cancels): written with `expm1` and
//!   `ln1p` it is evaluated without cancellation at any spacing, equal
//!   stretches included, where it equals eq. 25's limit (a gate checks it).
//!   So no tolerance switch is needed (the source's `10^-6`, its sec. 3.3).
//!   The same `s` gives `S_iso = (g_1 - s l_1) I + s C` without
//!   eigenvectors.
//! - The volumetric energy `U = K/2 (ln J)^2`, the engine's Hencky form
//!   (`j2_hencky`, Love and Sulsky TR 2005-001 eq. 45), so `J U' = K ln J`
//!   and `C_vol : dC / 2 = K/2 (C^-1 : dC) C^-1 - K ln J C^-1 dC C^-1`. Not
//!   measured for any rubber here: a declared choice until a measured
//!   `U(J)` for the target rubber (Connolly leaves `U` to the user, p.1278).

use glam::{DMat2, DVec2};

/// The most terms a law carries.
pub(crate) const MAX_TERMS: usize = 4;

/// Ogden's law: `count` terms `(mu_i, alpha_i)` (Pa, dimensionless,
/// `alpha_i != 0`) and the bulk modulus `K` of `U = K/2 (ln J)^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ogden {
    pub terms: [(f64, f64); MAX_TERMS],
    pub count: usize,
    pub bulk: f64,
}

impl Ogden {
    /// The law with `terms` and the bulk modulus `bulk`.
    pub fn new(terms: &[(f64, f64)], bulk: f64) -> Self {
        assert!(
            !terms.is_empty() && terms.len() <= MAX_TERMS,
            "1 to {MAX_TERMS} terms"
        );
        assert!(terms.iter().all(|t| t.1 != 0.0), "alpha_i != 0");
        let mut out = Self {
            terms: [(0.0, 1.0); MAX_TERMS],
            count: terms.len(),
            bulk,
        };
        out.terms[..terms.len()].copy_from_slice(terms);
        out
    }

    /// The small-strain shear modulus `1/2 sum_i mu_i alpha_i`.
    pub fn shear(&self) -> f64 {
        0.5 * self.each().map(|(mu, alpha)| mu * alpha).sum::<f64>()
    }

    fn each(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.terms[..self.count].iter().copied()
    }
}

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

/// `ln1p(x) / x`, 1 at `x = 0`.
fn log_ratio(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { x.ln_1p() / x }
}

/// `expm1(y) / y`, 1 at `y = 0`.
fn exp_ratio(y: f64) -> f64 {
    if y == 0.0 { 1.0 } else { y.exp_m1() / y }
}

/// The principal quantities of an in-plane `C`.
struct Principal {
    /// The squared stretches `l_1 >= l_2` and their half difference `r`.
    l: [f64; 2],
    r: f64,
    ln_j: f64,
    /// `lbar_a^alpha_i` for `a = 1, 2, 3`.
    powers: [[f64; MAX_TERMS]; 3],
}

fn principal(law: &Ogden, c: DMat2) -> Principal {
    let half = 0.5 * (c.x_axis.x + c.y_axis.y);
    let q = 0.5 * (c.x_axis.x - c.y_axis.y);
    let r = (q * q + c.y_axis.x * c.x_axis.y).max(0.0).sqrt();
    let det = c.determinant();
    let l1 = half + r;
    // `l_2 = det / l_1`: no cancellation when `l_2 << l_1`.
    let l2 = det / l1;
    let ln_j = 0.5 * det.ln();
    let log_bar = [
        0.5 * l1.ln() - ln_j / 3.0,
        0.5 * l2.ln() - ln_j / 3.0,
        -ln_j / 3.0,
    ];
    let mut powers = [[0.0; MAX_TERMS]; 3];
    for (a, row) in powers.iter_mut().enumerate() {
        for (i, (_, alpha)) in law.each().enumerate() {
            row[i] = (alpha * log_bar[a]).exp();
        }
    }
    Principal {
        l: [l1, l2],
        r,
        ln_j,
        powers,
    }
}

/// `B_a = lbar_a dW/dlbar_a` and `G_a = sum_i mu_i alpha_i lbar_a^alpha_i`.
fn moments(law: &Ogden, p: &Principal) -> ([f64; 3], [f64; 3]) {
    let mut b = [0.0; 3];
    let mut g = [0.0; 3];
    for a in 0..3 {
        for (i, (mu, alpha)) in law.each().enumerate() {
            b[a] += mu * p.powers[a][i];
            g[a] += mu * alpha * p.powers[a][i];
        }
    }
    (b, g)
}

/// `beta_1, beta_2`, `g_a = beta_a / l_a` and the divided difference `s`
/// (module doc).
fn coefficients(law: &Ogden, p: &Principal) -> ([f64; 2], f64) {
    let (b, _) = moments(law, p);
    let mean = (b[0] + b[1] + b[2]) / 3.0;
    let beta = [b[0] - mean, b[1] - mean];
    let [l1, l2] = p.l;
    // `(B_1 - B_2) / (l_1 - l_2) = sum_i mu_i p_2i alpha_i / (2 l_2) Lr(x)
    // Er(y_i)`, `x = 2 r / l_2`, `y_i = alpha_i / 2 ln1p(x)`.
    let x = 2.0 * p.r / l2;
    let lr = log_ratio(x);
    let spread: f64 = law
        .each()
        .enumerate()
        .map(|(i, (mu, alpha))| {
            let y = 0.5 * alpha * x.ln_1p();
            mu * p.powers[1][i] * alpha / (2.0 * l2) * lr * exp_ratio(y)
        })
        .sum();
    let s = spread / l1 - beta[1] / (l1 * l2);
    (beta, s)
}

/// The stored energy per reference volume at `C` (in-plane block);
/// infinite where `det C <= 0`.
pub(crate) fn ogden_energy(c: DMat2, law: &Ogden) -> f64 {
    if c.determinant() <= 0.0 {
        return f64::INFINITY;
    }
    let p = principal(law, c);
    let iso: f64 = law
        .each()
        .enumerate()
        .map(|(i, (mu, alpha))| {
            mu / alpha * (p.powers[0][i] + p.powers[1][i] + p.powers[2][i] - 3.0)
        })
        .sum();
    iso + 0.5 * law.bulk * p.ln_j * p.ln_j
}

/// The second Piola-Kirchhoff stress `S = S_iso + K ln J C^-1` (eqs. 11-13).
pub(crate) fn ogden_second_pk(c: DMat2, law: &Ogden) -> DMat2 {
    let p = principal(law, c);
    let (beta, s) = coefficients(law, &p);
    let g1 = beta[0] / p.l[0];
    DMat2::from_diagonal(DVec2::splat(g1 - s * p.l[0])) + c * s + c.inverse() * (law.bulk * p.ln_j)
}

/// The Kirchhoff stress `tau = F S F^T` (eq. 14).
pub(crate) fn ogden_kirchhoff(f: DMat2, law: &Ogden) -> DMat2 {
    f * ogden_second_pk(f.transpose() * f, law) * f.transpose()
}

/// `dS` along a symmetric `dC`: half the material tangent applied to `dC`
/// (eqs. 22, 27 with the module doc's `gamma_ab` and `s`): `sum_ab 1/2
/// (gamma_ab / (l_a l_b) - 2 delta_ab beta_a / l_a^2) (N_b dC N_b) N_a N_a + s
/// (N_1 dC N_2) (N_1 N_2 + N_2 N_1)` plus the volume's.
pub(crate) fn ogden_second_pk_tangent(c: DMat2, dc: DMat2, law: &Ogden) -> DMat2 {
    let p = principal(law, c);
    let (beta, s) = coefficients(law, &p);
    let (_, g) = moments(law, &p);
    let sum_g = (g[0] + g[1] + g[2]) / 9.0;
    let gamma = |a: usize, b: usize| {
        let delta = if a == b { g[a] } else { 0.0 };
        delta - (g[a] + g[b]) / 3.0 + sum_g
    };
    // `N_1` along the larger squared stretch.
    let angle = 0.5 * (2.0 * c.y_axis.x).atan2(c.x_axis.x - c.y_axis.y);
    let n = [
        DVec2::new(angle.cos(), angle.sin()),
        DVec2::new(-angle.sin(), angle.cos()),
    ];
    let along = |a: usize, b: usize| n[a].dot(dc * n[b]);
    let outer = |a: usize, b: usize| DMat2::from_cols(n[a] * n[b].x, n[a] * n[b].y);
    let mut ds = DMat2::ZERO;
    for (a, &beta_a) in beta.iter().enumerate() {
        for b in 0..2 {
            let mut coefficient = gamma(a, b) / (p.l[a] * p.l[b]);
            if a == b {
                coefficient -= 2.0 * beta_a / (p.l[a] * p.l[a]);
            }
            ds += outer(a, a) * (0.5 * coefficient * along(b, b));
        }
    }
    ds += (outer(0, 1) + outer(1, 0)) * (s * along(0, 1));
    let inverse = c.inverse();
    ds + inverse * (0.5 * law.bulk * colon(inverse, dc))
        - (inverse * dc * inverse) * (law.bulk * p.ln_j)
}

/// `W(C_1) - W(C_0)` from the increment itself, so its round-off is relative
/// to the change (the discrete gradient divides it by `|dC|^2`). Derived on
/// the page, not sourced: the squared stretches' increments from the
/// entries' (`dl_a = d half +- d r`, `d r^2 = d q (2 q_0 + d q) + d c_12 (2
/// c_12 + d c_12)`, as `j2_hencky`'s step), `d ln lbar_a = 1/2 ln1p(dl_a /
/// l_a) - 1/3 d ln J` with `d ln J = 1/2 ln1p(d det / det C_0)`, and `d
/// lbar^alpha = lbar_0^alpha expm1(alpha d ln lbar)`.
pub(crate) fn ogden_energy_increment(c0: DMat2, c1: DMat2, law: &Ogden) -> f64 {
    let dc = c1 - c0;
    let p = principal(law, c0);
    let d_half = 0.5 * (dc.x_axis.x + dc.y_axis.y);
    let (q0, d_q) = (
        0.5 * (c0.x_axis.x - c0.y_axis.y),
        0.5 * (dc.x_axis.x - dc.y_axis.y),
    );
    let (s0, d_s) = (c0.y_axis.x, dc.y_axis.x);
    let d_r2 = d_q * (2.0 * q0 + d_q) + d_s * (2.0 * s0 + d_s);
    let r1 = (p.r * p.r + d_r2).max(0.0).sqrt();
    let d_r = if p.r + r1 > 0.0 {
        d_r2 / (p.r + r1)
    } else {
        0.0
    };
    let det0 = c0.determinant();
    let adjugate_dot =
        c0.y_axis.y * dc.x_axis.x - c0.y_axis.x * dc.x_axis.y - c0.x_axis.y * dc.y_axis.x
            + c0.x_axis.x * dc.y_axis.y;
    let d_ln_j = 0.5 * ((adjugate_dot + dc.determinant()) / det0).ln_1p();
    let d_log_bar = [
        0.5 * ((d_half + d_r) / p.l[0]).ln_1p() - d_ln_j / 3.0,
        0.5 * ((d_half - d_r) / p.l[1]).ln_1p() - d_ln_j / 3.0,
        -d_ln_j / 3.0,
    ];
    let iso: f64 = law
        .each()
        .enumerate()
        .map(|(i, (mu, alpha))| {
            mu / alpha
                * (0..3)
                    .map(|a| p.powers[a][i] * (alpha * d_log_bar[a]).exp_m1())
                    .sum::<f64>()
        })
        .sum();
    iso + 0.5 * law.bulk * d_ln_j * (2.0 * p.ln_j + d_ln_j)
}

#[cfg(test)]
mod tests;

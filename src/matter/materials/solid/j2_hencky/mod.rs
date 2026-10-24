//! J2 (von Mises) plasticity on Hencky elasticity, multiplicative, in plane
//! strain: what a ductile solid is, its stored energy, its yield and its
//! plastic flow.
//!
//! Source: Love & Sulsky, *An energy consistent material-point method for
//! dynamic finite deformation plasticity*, HPC@UNM TR 2005-001 (read in
//! full): the Hencky energy `W = G |dev e|^2 + U(J^e) + H alpha^2 / 2` in the
//! principal elastic log strains `e_A = log lambda^e_A` (eq. 43, 45), the
//! yield function `phi = |dev tau| - sqrt(2/3) (Y + H alpha)` (eq. 37, with
//! `q = -H alpha`), the backward-Euler exponential return in principal log
//! strains (eq. 71), its closed form `Delta-gamma = phi_tr / (eta/dt + 2H/3 +
//! 2G)` (eq. 73), and the step's dissipation `Delta-gamma (eta
//! Delta-gamma/dt + sqrt(2/3) Y) >= 0` (eq. 74). The volumetric energy is
//! the Hencky one, `U(J^e) = K/2 (ln J^e)^2`, its pressure `K ln J^e`.
//! Adaptation, declared: plane strain. The elastic left Cauchy-Green
//! tensor `b^e` keeps its out-of-plane component `b^e_33` (the flow is
//! isochoric in 3D, so with the total `F_33 = 1` the plastic flow moves
//! `b^e_33`), a per-particle internal variable beside the in-plane 2 x 2.
//!
//! Written in the total Lagrangian variables: the state is the plastic
//! metric `Gp = C_p^-1` (with `Gp_33`) and `alpha`, `be = F Gp F^T`, whose
//! eigenvalues are those of `C Gp`; the stress `S = F^-1 tau F^-T = Gp g(C
//! Gp)` and the return's new metric `Gp k(C Gp)` are the eigenvalue maps
//! carried by that similarity (derived here: `F^T be F^-T = C Gp`), so no
//! `F` and no rotation enter; the step's energy change is exact from the
//! increment (`HenckyJ2::step`).
//!
//! (The explicit path's `VonMisesMaterial` is a different, 2D in-plane
//! form, its return applied to the deformation gradient of a corotated law:
//! a migration onto this law is pending, an engine-wide decision.)
//!
//! A second flow law, Norton's power law with no static yield (`Norton`),
//! in the same variables: `eps_bar_dot^p = rate_0 (sigma_eq / K)^(1/m)`,
//! `sigma_eq = sqrt(3/2) |dev tau|`, `eps_bar^p = sqrt(2/3) gamma` (LS05
//! eq. 37-41; the law as in Nguyen, Amores, Sanz and Montans, Comput. Mech.
//! 67 (2021), eq. 123-125, with no yield stress, read through the agent's
//! report, `VISCOPLAST_SOURCES_REPORT` row 9). Its backward-Euler return
//! (eq. 71) is the one scalar equation `r(d) = sigma_eq,tr - 3G d - K (d /
//! (rate_0 dt))^m = 0` in `d = Delta eps_bar^p`, solved by a bracketed
//! Newton (`norton_return`), and its step's dissipation is the step's own
//! irreversible work `sigma_eq,n+1 d` (eq. 74's third line with this law's
//! `phi_n+1`), fed to the algorithmic stress (eq. 81-83) as the linear
//! case's is. Adaptation, declared: LS05 writes the linear case only. The
//! variational alternative `dt psi*(d / dt)` is not used: it is not the
//! irreversible work for a power law (Noels, Stainier and Ponthot, CMAME
//! 197 (2008), p. 6 footnote 4 and eq. 99, read).

use glam::{DMat2, DVec2};

/// The law's moduli: shear `G`, bulk `K`, yield stress `Y` (uniaxial),
/// linear isotropic hardening `H`, viscoplastic viscosity `eta` (zero:
/// rate independent). Any consistent units (Pa and Pa s in SI).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HenckyJ2 {
    pub shear: f64,
    pub bulk: f64,
    pub yield_stress: f64,
    pub hardening: f64,
    pub viscosity: f64,
    /// Norton's flow law in place of the yield, hardening and linear
    /// viscosity above (which must then be zero); `None`: that law.
    pub norton: Option<Norton>,
}

/// Norton's power law (no static yield): `eps_bar_dot^p = rate_0 (sigma_eq
/// / K)^(1/m)`, `K` in the stress's units (Pa in SI), `m` its rate
/// sensitivity, `rate_0` the reference rate (1/s); and the equivalent
/// plastic rates its measurement covers, `valid_rates` (1/s), outside which
/// a step is counted (`Stepped::outside`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norton {
    pub k: f64,
    pub m: f64,
    pub rate_0: f64,
    pub valid_rates: (f64, f64),
}

/// The probes' record of Norton's returns (`watch_returns`): per iteration
/// count, how many returns took it, and the state of the slowest one
/// (`sigma_tr`, the root, `dt`).
#[cfg(all(test, feature = "experimental"))]
type Returns = Option<(Vec<u64>, (u32, f64, f64, f64))>;

#[cfg(all(test, feature = "experimental"))]
static RETURNS: std::sync::Mutex<Returns> = std::sync::Mutex::new(None);

#[cfg(all(test, feature = "experimental"))]
pub(crate) fn watch_returns(on: bool) {
    *RETURNS.lock().expect("returns lock") = on.then(|| (vec![0; 202], (0, 0.0, 0.0, 0.0)));
}

/// The returns recorded so far (`watch_returns`), zero when not watching.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn returns_seen() -> u64 {
    RETURNS
        .lock()
        .expect("returns lock")
        .as_ref()
        .map_or(0, |(counts, _)| counts.iter().sum())
}

#[cfg(all(test, feature = "experimental"))]
pub(crate) fn take_returns() -> Returns {
    RETURNS.lock().expect("returns lock").take()
}

/// Norton's return (`norton_root`), recorded for the probes when they
/// watch (`watch_returns`).
pub(crate) fn norton_return(sigma_tr: f64, shear: f64, law: &Norton, dt: f64) -> (f64, u32) {
    let (d, iterations) = norton_root(sigma_tr, shear, law, dt);
    #[cfg(all(test, feature = "experimental"))]
    if let Some((counts, slowest)) = RETURNS.lock().expect("returns lock").as_mut() {
        counts[(iterations as usize).min(201)] += 1;
        if iterations > slowest.0 {
            *slowest = (iterations, sigma_tr, d, dt);
        }
    }
    (d, iterations)
}

/// The root of Norton's return, `r(d) = sigma_tr - 3G d - K (d / (rate_0
/// dt))^m = 0`, and the iterations it took.
///
/// Derived on the page. `r` is strictly decreasing, and convex for `m < 1`
/// (`r'' = -K m (m - 1) (d / s)^(m - 2) / s^2 > 0`, `s = rate_0 dt`). Its
/// root lies in `[low, high]`:
/// - `high = min(sigma_tr / 3G, d_slow)`, `d_slow = s (sigma_tr / K)^(1/m)`
///   (the root when the elastic term is negligible): `r(sigma_tr / 3G) < 0`
///   and `r(d_slow) = -3G d_slow < 0`;
/// - `low = s ((sigma_tr - 3G high) / K)^(1/m)`: `r(low) = 3G (high - low)
///   >= 0`, and `low <= high`.
///
/// Newton's method from `low` (where `r >= 0`) on a convex decreasing `r`
/// moves right and never passes the root (the tangent lies under the
/// curve), so it converges monotonically inside the bracket; a step that
/// would leave it falls back to bisection, which keeps it a bracketed
/// method. It stops when the bracket is within `4 eps` of its top. In slow
/// flow (`3G d << sigma_tr`, a body at round-off stress included) the
/// bracket is already that narrow, and the return takes no iteration.
fn norton_root(sigma_tr: f64, shear: f64, law: &Norton, dt: f64) -> (f64, u32) {
    let scale = law.rate_0 * dt;
    let r = |d: f64| sigma_tr - 3.0 * shear * d - law.k * (d / scale).powf(law.m);
    let slope = |d: f64| -3.0 * shear - law.k * law.m * (d / scale).powf(law.m - 1.0) / scale;
    let slow = scale * (sigma_tr / law.k).powf(1.0 / law.m);
    let mut high = (sigma_tr / (3.0 * shear)).min(slow);
    let mut low = scale * ((sigma_tr - 3.0 * shear * high).max(0.0) / law.k).powf(1.0 / law.m);
    let mut d = low;
    let mut iterations = 0u32;
    while high - low > 4.0 * f64::EPSILON * high && iterations < 200 {
        iterations += 1;
        let value = r(d);
        if value == 0.0 {
            return (d, iterations);
        }
        if value > 0.0 {
            low = d;
        } else {
            high = d;
        }
        let newton = d - value / slope(d);
        d = if newton > low && newton < high {
            newton
        } else {
            0.5 * (low + high)
        };
    }
    (d, iterations)
}

/// The elastic state in the total Lagrangian variables: the plastic metric
/// `Gp = C_p^-1` (in plane) and its out-of-plane component, and `alpha`;
/// `be = F Gp F^T`, whose eigenvalues are those of `C Gp`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlasticMetric {
    pub gp: DMat2,
    pub gp33: f64,
    pub alpha: f64,
}

impl PlasticMetric {
    /// No plastic strain yet.
    pub(crate) fn unstrained() -> Self {
        Self {
            gp: DMat2::IDENTITY,
            gp33: 1.0,
            alpha: 0.0,
        }
    }
}

/// The two eigenvalues (larger first) of the 2x2 matrix `m` with real ones
/// (`C Gp`, similar to the symmetric `be`): its half trace and `r`, `l = m
/// +- r`.
fn mean_and_radius(m: DMat2) -> (f64, f64) {
    let half = 0.5 * (m.x_axis.x + m.y_axis.y);
    (half, radius_squared(m).max(0.0).sqrt())
}

/// `r^2 = (tr/2)^2 - det`, written `((m11 - m22)/2)^2 + m12 m21` (the same
/// identity for any 2 x 2 matrix): no cancellation. Through `(tr/2)^2 -
/// det`, both near 1 for a small strain, `r` lost its digits to the
/// subtraction (measured: the energy 1e-10 off at a 1e-3 strain).
fn radius_squared(m: DMat2) -> f64 {
    let q = 0.5 * (m.x_axis.x - m.y_axis.y);
    q * q + m.y_axis.x * m.x_axis.y
}

/// `g(M) = alpha I + beta M`, the matrix function taking the eigenvalue
/// `l_A` of `M` to `g_A`, by Lagrange interpolation on the two eigenvalues;
/// `slope` the divided difference `(g_1 - g_2) / (l_1 - l_2)` supplied by
/// the caller in a form stable when they meet.
fn interpolate(m: DMat2, (l1, g1): (f64, f64), slope: f64) -> DMat2 {
    let alpha = g1 - slope * l1;
    DMat2::from_diagonal(DVec2::splat(alpha)) + m * slope
}

impl HenckyJ2 {
    /// The principal elastic log strains `(e1, e2, e3)` of the state at `C`.
    fn logs(&self, c: DMat2, state: &PlasticMetric) -> ([f64; 3], f64, f64) {
        let (half, r) = mean_and_radius(c * state.gp);
        let (l1, l2) = (half + r, half - r);
        (
            [0.5 * l1.ln(), 0.5 * l2.ln(), 0.5 * state.gp33.ln()],
            l1,
            l2,
        )
    }

    /// The stored energy per reference volume at `C` in the state.
    pub(crate) fn energy_at(&self, c: DMat2, state: &PlasticMetric) -> f64 {
        let (e, _, _) = self.logs(c, state);
        let tr = e[0] + e[1] + e[2];
        let dev2 = e.iter().map(|x| (x - tr / 3.0).powi(2)).sum::<f64>();
        self.shear * dev2 + 0.5 * self.bulk * tr * tr + 0.5 * self.hardening * state.alpha.powi(2)
    }

    /// The second Piola-Kirchhoff stress `S = F^-1 tau F^-T` at `C` in the
    /// state, and `tau_33`: `S = Gp g(C Gp)`, `g(l_A) = tau_A / l_A`.
    pub(crate) fn second_pk_at(&self, c: DMat2, state: &PlasticMetric) -> (DMat2, f64) {
        let m = c * state.gp;
        let (e, l1, l2) = self.logs(c, state);
        let tr = e[0] + e[1] + e[2];
        let tau = |a: usize| 2.0 * self.shear * (e[a] - tr / 3.0) + self.bulk * tr;
        let t1 = tau(0);
        // (g1 - g2) / (l1 - l2), with tau1 - tau2 = G ln(l1 / l2):
        // [-t1 + l1 G ln(l1/l2) / (l1 - l2)] / (l1 l2), the last ratio
        // ln1p(d) / (d l2), d = (l1 - l2) / l2, -> 1 / l2 as l1 -> l2.
        let d = (l1 - l2) / l2;
        let log_ratio = if d == 0.0 { 1.0 } else { d.ln_1p() / d };
        let slope = (-t1 + l1 * self.shear * log_ratio / l2) / (l1 * l2);
        let g = interpolate(m, (l1, t1 / l1), slope);
        (state.gp * g, tau(2))
    }
}

/// What a step's return map gives in the total Lagrangian variables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stepped {
    pub(crate) state: PlasticMetric,
    /// `W(C1, state_n+1) - W(C0, state_n)`, from the increment (exact to
    /// round-off of the increment, never a difference of two energies).
    pub(crate) energy_change: f64,
    /// The step's dissipation per reference volume (Love and Sulsky eq. 74).
    pub(crate) dissipated: f64,
    /// Norton's return's iterations (`norton_return`); zero otherwise.
    pub(crate) iterations: u32,
    /// Where a Norton step flowed outside what its measurement covers, if
    /// it did (`Beyond`).
    pub(crate) outside: Option<Beyond>,
}

/// How a Norton step left its record's validity: an equivalent plastic rate
/// below or above `Norton::valid_rates`, or a tensile mean stress (the
/// record is compression-dominated). A tensile step is reported as such
/// whatever its rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Beyond {
    Slower,
    Faster,
    Tension,
}

impl HenckyJ2 {
    /// The step from `C0` to `C1 = C0 + dC` in `state`, step length `dt`:
    /// the trial `be_tr = F1 Gp F1^T` (eigenvalues of `C1 Gp`), the return
    /// in principal logs (eq. 71-73), the new metric `Gp_n+1 = Gp k(C1 Gp)`
    /// with `k(l_A) = exp(-2 dgamma nu_A)` (the eigen-scaling of the
    /// return, carried back by the same similarity), and the energy's change
    /// exactly: the trial's from the increments of `C Gp`'s trace and
    /// determinant, the return's in principal logs.
    pub(crate) fn step(&self, c0: DMat2, c1: DMat2, state: &PlasticMetric, dt: f64) -> Stepped {
        let gp = state.gp;
        let dc = c1 - c0;
        // Trial invariants of M = C Gp and their increments, from the
        // entries (`radius_squared`): `dM = dC Gp` exactly, `half` the half
        // trace, `q = (m11 - m22) / 2`, `r^2 = q^2 + m12 m21`.
        let m0 = c0 * gp;
        let dm = dc * gp;
        let half0 = 0.5 * (m0.x_axis.x + m0.y_axis.y);
        let d_half = 0.5 * (dm.x_axis.x + dm.y_axis.y);
        let q0 = 0.5 * (m0.x_axis.x - m0.y_axis.y);
        let d_q = 0.5 * (dm.x_axis.x - dm.y_axis.y);
        let (b0, c0m) = (m0.y_axis.x, m0.x_axis.y);
        let (db, dcm) = (dm.y_axis.x, dm.x_axis.y);
        let r0 = radius_squared(m0).max(0.0).sqrt();
        let d_r2 = d_q * (2.0 * q0 + d_q) + db * (c0m + dcm) + b0 * dcm;
        let r1 = (r0 * r0 + d_r2).max(0.0).sqrt();
        let d_r = if r0 + r1 > 0.0 { d_r2 / (r0 + r1) } else { 0.0 };
        let half1 = half0 + d_half;
        let m1 = m0 + dm;
        let (l0, k0) = (half0 + r0, half0 - r0);
        let (dl1, dl2) = (d_half + d_r, d_half - d_r);
        // Log differences, ln(l1 / l0) = ln1p(dl / l0), per branch.
        let dln = [(dl1 / l0).ln_1p(), (dl2 / k0).ln_1p()];
        let ln0 = [l0.ln(), k0.ln()];
        let e0 = [0.5 * ln0[0], 0.5 * ln0[1], 0.5 * state.gp33.ln()];
        let de = [0.5 * dln[0], 0.5 * dln[1], 0.0];
        let e_tr = [e0[0] + de[0], e0[1] + de[1], e0[2]];
        // The trial's energy change, from the increments: |dev e|^2 =
        // q + e3^2 - (tr e)^2 / 3 with q = e1^2 + e2^2.
        let d_q = de[0] * (2.0 * e0[0] + de[0]) + de[1] * (2.0 * e0[1] + de[1]);
        let (tr0, d_tr) = (e0[0] + e0[1] + e0[2], de[0] + de[1]);
        let d_tr2 = d_tr * (2.0 * tr0 + d_tr);
        let mut energy_change = self.shear * (d_q - d_tr2 / 3.0) + 0.5 * self.bulk * d_tr2;
        // The return (eq. 71-74), on the trial's principal logs.
        let tr = e_tr[0] + e_tr[1] + e_tr[2];
        let dev = [e_tr[0] - tr / 3.0, e_tr[1] - tr / 3.0, e_tr[2] - tr / 3.0];
        let norm = (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]).sqrt();
        let phi = match self.norton {
            // No static yield: any deviator flows.
            Some(_) => 2.0 * self.shear * norm,
            None => {
                2.0 * self.shear * norm
                    - (2.0f64 / 3.0).sqrt() * (self.yield_stress + self.hardening * state.alpha)
            }
        };
        let mut next = PlasticMetric {
            gp: state.gp,
            gp33: state.gp33,
            alpha: state.alpha,
        };
        let mut dissipated = 0.0;
        let (mut iterations, mut outside) = (0, None);
        if phi > 0.0 && norm > 0.0 && (self.norton.is_none() || dt > 0.0) {
            let dgamma = match &self.norton {
                Some(law) => {
                    // `sigma_eq,tr = sqrt(3/2) |dev tau_tr|`, `dgamma =
                    // sqrt(3/2) d`.
                    let sigma_tr = (1.5f64).sqrt() * phi;
                    let (d, count) = norton_return(sigma_tr, self.shear, law, dt);
                    iterations = count;
                    let rate = d / dt;
                    // The mean stress `K tr e` is tensile when `tr e > 0`
                    // (the flow does not move the volume).
                    outside = if tr > 0.0 {
                        Some(Beyond::Tension)
                    } else if rate < law.valid_rates.0 {
                        Some(Beyond::Slower)
                    } else if rate > law.valid_rates.1 {
                        Some(Beyond::Faster)
                    } else {
                        None
                    };
                    // The step's irreversible work, `sigma_eq,n+1 d`, with
                    // `sigma_eq,n+1 = sigma_tr - 3G d` the returned stress.
                    dissipated = (sigma_tr - 3.0 * self.shear * d) * d;
                    (1.5f64).sqrt() * d
                }
                None => {
                    let rate = if dt > 0.0 { self.viscosity / dt } else { 0.0 };
                    let dgamma = phi / (rate + 2.0 / 3.0 * self.hardening + 2.0 * self.shear);
                    dissipated =
                        dgamma * (rate * dgamma + (2.0f64 / 3.0).sqrt() * self.yield_stress);
                    dgamma
                }
            };
            let alpha1 = state.alpha + (2.0f64 / 3.0).sqrt() * dgamma;
            // dev e shrinks by dgamma along itself: |dev|^2 changes by
            // dgamma (dgamma - 2 |dev|); the volume part does not move.
            energy_change += self.shear * dgamma * (dgamma - 2.0 * norm)
                + 0.5 * self.hardening * (alpha1 * alpha1 - state.alpha * state.alpha);
            // be_A scales by exp(-2 dgamma nu_A), nu = dev / |dev|.
            let scale = |a: usize| (-2.0 * dgamma * dev[a] / norm).exp();
            let (k1, k2) = (scale(0), scale(1));
            // k(M1) by interpolation on M1's eigenvalues (larger first).
            let (h1, r1m) = (half1, r1);
            let (lam1, lam2) = (h1 + r1m, h1 - r1m);
            let slope = if lam1 > lam2 {
                (k1 - k2) / (lam1 - lam2)
            } else {
                0.0
            };
            let kmat = interpolate(m1, (lam1, k1), slope);
            next.gp = gp * kmat;
            next.gp33 = state.gp33 * scale(2);
            next.alpha = alpha1;
        }
        Stepped {
            state: next,
            energy_change,
            dissipated,
            iterations,
            outside,
        }
    }
}

mod elastic_range;

#[cfg(test)]
mod tests;

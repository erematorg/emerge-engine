//! The exact solution of a 1D generalised Zener bar, mode by mode: the
//! reference the engine's damped solids are measured against.
//!
//! The law (Blanc, Komatitsch, Chaljub, Lombard and Xie, GJI 205 (2016)
//! 427-439, eqs. 10 and 21-23, read): `N` relaxation mechanisms with
//! weights `kappa_l > 0` and relaxation frequencies `theta_l > 0`, the
//! modulus `M(omega) = E_r (1 + sum_l i omega kappa_l / (theta_l + i
//! omega))`. Writing each mechanism's stress `sigma_l = E_r (eps / N +
//! xi_l)` turns their eq. 10 into the memory variables (derived on the
//! page: divide eq. 10 by `tau_sigma_l`, use eq. 21):
//! - `sigma = E_r (eps + sum_l xi_l)`;
//! - `xi_l' + theta_l xi_l = kappa_l eps'`.
//!
//! Their Property 1 (eqs. 11-13), in these variables: the energy `1/2 rho
//! v^2 + 1/2 E_r eps^2 + sum_l 1/2 (E_r / kappa_l) xi_l^2` decays at the
//! rate `sum_l (E_r theta_l / kappa_l) xi_l^2 >= 0`, the heat made.
//!
//! A uniform bar's spatial modes are the elastic bar's, the law being the
//! same at every point. Mode `n` (wavenumber `k`) carries `q` (displacement)
//! and one `s_l` per mechanism (memory), and obeys exactly
//! `q'' = -w^2 (q + sum_l s_l)`, `s_l' + theta_l s_l = kappa_l q'`, with
//! `w^2 = E_r k^2 / rho`. That is a linear system `y' = A y`,
//! `y = (q, q', s_1..s_N)`. For an eigenvalue `p` the eigenvector is
//! `(1, p, kappa_l p / (p + theta_l))` (from the second equation), and `p`
//! is a root of (from the first)
//! `P(p) = (p^2 + w^2) prod_l (p + theta_l) + w^2 sum_l kappa_l p
//! prod_{m != l} (p + theta_m)`. With `p = i omega` this is Blanc et al.'s
//! dispersion relation (eq. 24) at a real wavenumber.
//!
//! Adaptation, declared: the roots come from the Weierstrass (Durand-Kerner)
//! iteration, and the coefficients of the start state from Gaussian
//! elimination with partial pivoting, not from a source read for either.
//! Both are therefore checked on every solution (`ModeSolution::checks`,
//! `ModeSolution::residual`):
//! - each root makes `P` vanish to its terms' round-off;
//! - the start state is reproduced;
//! - the assembled solution satisfies `y' = A y` at every time asked;
//! - the energy plus the heat (integrated in closed form) stays the start
//!   energy.

use rustfft::num_complex::Complex64 as C;

/// A generalised Zener material in 1D: density, relaxed modulus, and per
/// mechanism its weight and relaxation frequency (rad/s).
#[derive(Clone, Debug)]
pub struct Zener {
    pub rho: f64,
    pub relaxed: f64,
    pub kappa: Vec<f64>,
    pub theta: Vec<f64>,
}

impl Zener {
    /// The unrelaxed modulus, `E_r (1 + sum kappa)` (Blanc et al. eqs. 8,
    /// 21).
    pub fn unrelaxed(&self) -> f64 {
        self.relaxed * (1.0 + self.kappa.iter().sum::<f64>())
    }

    /// `M(omega) / E_r` (eq. 23).
    pub fn modulus_ratio(&self, omega: f64) -> C {
        let i_omega = C::new(0.0, omega);
        C::new(1.0, 0.0)
            + self
                .kappa
                .iter()
                .zip(&self.theta)
                .map(|(&k, &t)| i_omega * k / (C::new(t, 0.0) + i_omega))
                .sum::<C>()
    }

    /// The reciprocal quality factor (eq. 28).
    pub fn inverse_q(&self, omega: f64) -> f64 {
        let m = self.modulus_ratio(omega);
        m.im / m.re
    }
}

/// One mode's exact motion from a start state: `y(t) = sum_j c_j v_j
/// e^(p_j t)`.
#[derive(Clone, Debug)]
pub struct ModeSolution {
    /// `w^2 = E_r k^2 / rho`.
    pub w2: f64,
    pub kappa: Vec<f64>,
    pub theta: Vec<f64>,
    pub roots: Vec<C>,
    coefficients: Vec<C>,
    start: Vec<f64>,
}

/// The mode's state: displacement, velocity, and each mechanism's memory.
#[derive(Clone, Debug)]
pub struct ModeState {
    pub q: f64,
    pub rate: f64,
    pub memory: Vec<f64>,
}

impl ModeSolution {
    /// The mode of wavenumber `k` of a bar of `material`, from `start`.
    pub fn new(material: &Zener, k: f64, start: &ModeState) -> Self {
        let w2 = material.relaxed * k * k / material.rho;
        let roots = roots(&characteristic(w2, &material.kappa, &material.theta));
        let n = roots.len();
        let vector = |p: C| -> Vec<C> {
            let mut v = vec![C::new(1.0, 0.0), p];
            v.extend(
                material
                    .kappa
                    .iter()
                    .zip(&material.theta)
                    .map(|(&kap, &th)| p * kap / (p + th)),
            );
            v
        };
        // Columns v_j; solve V c = y0.
        let mut matrix: Vec<Vec<C>> = (0..n).map(|_| vec![C::new(0.0, 0.0); n]).collect();
        for (j, &p) in roots.iter().enumerate() {
            for (i, e) in vector(p).into_iter().enumerate() {
                matrix[i][j] = e;
            }
        }
        let mut start_vector = vec![start.q, start.rate];
        start_vector.extend(&start.memory);
        let coefficients = solve(
            matrix,
            start_vector.iter().map(|&e| C::new(e, 0.0)).collect(),
        );
        Self {
            w2,
            kappa: material.kappa.clone(),
            theta: material.theta.clone(),
            roots,
            coefficients,
            start: start_vector,
        }
    }

    fn vector(&self, p: C) -> Vec<C> {
        let mut v = vec![C::new(1.0, 0.0), p];
        v.extend(
            self.kappa
                .iter()
                .zip(&self.theta)
                .map(|(&k, &t)| p * k / (p + t)),
        );
        v
    }

    /// The state at `t`, with the largest imaginary part left by the sum
    /// (zero but for round-off: the state is real).
    pub fn at(&self, t: f64) -> (ModeState, f64) {
        let n = self.roots.len();
        let mut y = vec![C::new(0.0, 0.0); n];
        for (&p, &c) in self.roots.iter().zip(&self.coefficients) {
            let e = c * (p * t).exp();
            for (yi, vi) in y.iter_mut().zip(self.vector(p)) {
                *yi += e * vi;
            }
        }
        let imaginary = y.iter().fold(0.0f64, |m, e| m.max(e.im.abs()));
        (
            ModeState {
                q: y[0].re,
                rate: y[1].re,
                memory: y[2..].iter().map(|e| e.re).collect(),
            },
            imaginary,
        )
    }

    /// The mode's energy per unit modal mass (`1/2 q'^2 + 1/2 w^2 q^2 +
    /// sum 1/2 (w^2 / kappa) s^2`).
    pub fn energy(&self, s: &ModeState) -> f64 {
        0.5 * s.rate * s.rate
            + 0.5 * self.w2 * s.q * s.q
            + s.memory
                .iter()
                .zip(&self.kappa)
                .map(|(m, k)| 0.5 * self.w2 / k * m * m)
                .sum::<f64>()
    }

    /// The heat made over `[0, t]` per unit modal mass, `int sum_l (w^2
    /// theta_l / kappa_l) s_l^2`, in closed form: each `s_l` a sum of
    /// exponentials, its square integrates term by term.
    pub fn heat(&self, t: f64) -> f64 {
        let mut total = 0.0;
        for (l, (&k, &th)) in self.kappa.iter().zip(&self.theta).enumerate() {
            let amplitude: Vec<C> = self
                .roots
                .iter()
                .zip(&self.coefficients)
                .map(|(&p, &c)| c * self.vector(p)[2 + l])
                .collect();
            let mut integral = C::new(0.0, 0.0);
            for (i, &pi) in self.roots.iter().enumerate() {
                for (j, &pj) in self.roots.iter().enumerate() {
                    let sum = pi + pj;
                    let term = if sum.norm() > 0.0 {
                        ((sum * t).exp() - 1.0) / sum
                    } else {
                        C::new(t, 0.0)
                    };
                    integral += amplitude[i] * amplitude[j] * term;
                }
            }
            total += self.w2 * th / k * integral.re;
        }
        total
    }

    /// The equations' residual at `t`, `y' - A y` assembled from the sum
    /// of exponentials, each row against its terms' contributions'
    /// magnitudes (its round-off's scale): the solution's exactness, whatever the
    /// conditioning of a single eigenvector. A root next to `-theta_l` is
    /// held only to `eps theta_l` absolute, so its eigenvector's memory
    /// component `kappa_l p / (p + theta_l)` carries a relative error up to
    /// `eps theta_l / |p + theta_l|` (measured: 2e-12 at a 30 Hz mode);
    /// the assembled state is what this measures.
    pub fn residual(&self, t: f64) -> f64 {
        let n = self.roots.len();
        let (mut y, mut rate) = (vec![C::new(0.0, 0.0); n], vec![C::new(0.0, 0.0); n]);
        // Each sum's round-off scale: its contributions' magnitudes before
        // they cancel (a start at zero strain and memory makes some sums
        // vanish exactly, leaving only their round-off).
        let (mut y_size, mut rate_size) = (vec![0.0f64; n], vec![0.0f64; n]);
        for (&p, &c) in self.roots.iter().zip(&self.coefficients) {
            let e = c * (p * t).exp();
            for (i, vi) in self.vector(p).into_iter().enumerate() {
                y[i] += e * vi;
                rate[i] += e * p * vi;
                y_size[i] += (e * vi).norm();
                rate_size[i] += (e * p * vi).norm();
            }
        }
        let (y, rate): (Vec<f64>, Vec<f64>) = (
            y.iter().map(|e| e.re).collect(),
            rate.iter().map(|e| e.re).collect(),
        );
        let memory: f64 = y[2..].iter().sum();
        let mut rows = vec![
            ((rate[0] - y[1]).abs(), rate_size[0] + y_size[1]),
            (
                (rate[1] + self.w2 * (y[0] + memory)).abs(),
                rate_size[1]
                    + self.w2
                        * y_size
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != 1)
                            .map(|(_, e)| e)
                            .sum::<f64>(),
            ),
        ];
        for (l, (&k, &th)) in self.kappa.iter().zip(&self.theta).enumerate() {
            rows.push((
                (rate[2 + l] - k * y[1] + th * y[2 + l]).abs(),
                rate_size[2 + l] + k * y_size[1] + th * y_size[2 + l],
            ));
        }
        rows.iter()
            .map(|&(r, size)| r / size.max(f64::MIN_POSITIVE))
            .fold(0.0, f64::max)
    }

    /// The solution's own checks, each against its round-off: the roots'
    /// residuals (relative to their terms) and the start state's
    /// reproduction (relative to its size). Returns the worst of each.
    pub fn checks(&self) -> (f64, f64) {
        let polynomial = characteristic(self.w2, &self.kappa, &self.theta);
        let root = self
            .roots
            .iter()
            .map(|&p| {
                let (value, size) = evaluate(&polynomial, p);
                value.norm() / size
            })
            .fold(0.0, f64::max);
        let (s0, _) = self.at(0.0);
        let mut y0 = vec![s0.q, s0.rate];
        y0.extend(&s0.memory);
        let scale = self.start.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        let start = y0
            .iter()
            .zip(&self.start)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
            / scale;
        (root, start)
    }
}

/// The characteristic polynomial's coefficients, lowest degree first.
fn characteristic(w2: f64, kappa: &[f64], theta: &[f64]) -> Vec<f64> {
    let multiply = |a: &[f64], (c0, c1): (f64, f64)| -> Vec<f64> {
        // a(s) (c0 + c1 s)
        let mut out = vec![0.0; a.len() + 1];
        for (i, &e) in a.iter().enumerate() {
            out[i] += e * c0;
            out[i + 1] += e * c1;
        }
        out
    };
    let product = |skip: Option<usize>| -> Vec<f64> {
        theta
            .iter()
            .enumerate()
            .filter(|&(l, _)| Some(l) != skip)
            .fold(vec![1.0], |acc, (_, &t)| multiply(&acc, (t, 1.0)))
    };
    // (s^2 + w2) prod (s + theta)
    let all = product(None);
    let mut p = vec![0.0; all.len() + 2];
    for (i, &e) in all.iter().enumerate() {
        p[i] += w2 * e;
        p[i + 2] += e;
    }
    // + w2 sum kappa_l s prod_{m != l} (s + theta_m)
    for (l, &k) in kappa.iter().enumerate() {
        for (i, &e) in product(Some(l)).iter().enumerate() {
            p[i + 1] += w2 * k * e;
        }
    }
    p
}

/// `P(z)` and the sum of its terms' magnitudes (the scale of its round-off).
fn evaluate(p: &[f64], z: C) -> (C, f64) {
    let (mut value, mut size, mut power) = (C::new(0.0, 0.0), 0.0, C::new(1.0, 0.0));
    for &c in p {
        value += power * c;
        size += (power * c).norm();
        power *= z;
    }
    (value, size.max(f64::MIN_POSITIVE))
}

/// The roots of `p` (lowest degree first), by the Weierstrass iteration on
/// the monic polynomial from the start points `(0.4 + 0.9 i)^j` times the
/// root bound, polished by Newton on `P`.
fn roots(p: &[f64]) -> Vec<C> {
    let degree = p.len() - 1;
    let lead = p[degree];
    let monic: Vec<f64> = p.iter().map(|c| c / lead).collect();
    // Cauchy's bound on the roots' size: 1 + max |a_i|.
    let bound = 1.0 + monic[..degree].iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let seed = C::new(0.4, 0.9);
    let mut z: Vec<C> = (0..degree).map(|j| seed.powu(j as u32) * bound).collect();
    for _ in 0..2000 {
        let mut largest = 0.0f64;
        for i in 0..degree {
            let (value, _) = evaluate(&monic, z[i]);
            let mut denominator = C::new(1.0, 0.0);
            for (j, &zj) in z.iter().enumerate() {
                if j != i {
                    denominator *= z[i] - zj;
                }
            }
            let step = value / denominator;
            z[i] -= step;
            largest = largest.max(step.norm() / z[i].norm().max(1.0));
        }
        if largest < 1e-15 {
            break;
        }
    }
    let derivative: Vec<f64> = (1..=degree).map(|i| monic[i] * i as f64).collect();
    for zi in z.iter_mut() {
        for _ in 0..3 {
            let (value, _) = evaluate(&monic, *zi);
            let (slope, _) = evaluate(&derivative, *zi);
            if slope.norm() > 0.0 {
                *zi -= value / slope;
            }
        }
    }
    z
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<C>>, mut b: Vec<C>) -> Vec<C> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].norm().total_cmp(&a[j][col].norm()))
            .expect("a row");
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let (top, bottom) = a.split_at_mut(row);
            let (pivot_row, target) = (&top[col], &mut bottom[0]);
            let factor = target[col] / pivot_row[col];
            for (t, &e) in target[col..].iter_mut().zip(&pivot_row[col..]) {
                *t -= factor * e;
            }
            let e = b[col];
            b[row] -= factor * e;
        }
    }
    let mut x = vec![C::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    x
}

/// A Zener rod of `length` moving at `speed` onto a rigid wall at its end
/// `x = 0`, from first touch to separation.
///
/// In contact the rod is fixed at the wall and free at `x = L`: its modes
/// are `sin(k_n x)`, `k_n = (2n - 1) pi / (2L)`, each exact
/// (`ModeSolution`). The uniform start velocity `-speed` projects on them
/// as `q_n' = -4 speed / ((2n - 1) pi)`, at zero strain and memory. The
/// wall holds while it pushes, `sigma(0, t) = E_r sum_n k_n (q_n + sum_l
/// s_nl) <= 0` (Signorini: the rod leaves when the contact force would
/// pull), and the rod leaves with the momentum `P = rho sum_n q_n' / k_n`,
/// kept from then on (no force acts in flight).
///
/// Truncation, declared: `modes` terms. The stress series converges like a
/// Fourier series of a jump (`1/n`), so the separation time is found to
/// about `L / (modes c)`; the momentum series converges like `1/n^2`.
/// Results are reported against the number of modes. A second contact after
/// separation is not followed.
#[derive(Clone, Debug)]
pub struct RodContact {
    pub separation: f64,
    /// The rebound momentum over the impact's, `P / (rho L speed)` (the
    /// rebound moves along `+x`, away from the wall).
    pub restitution: f64,
    /// At separation, as fractions of the start energy: the rigid rebound's
    /// kinetic energy, the rod's vibration (elastic, memory, and kinetic
    /// beyond the rigid motion), and the heat made.
    pub split: (f64, f64, f64),
    /// The truncated series' energy plus heat at separation against their
    /// start energy.
    pub ledger: f64,
}

pub fn rod_on_wall(material: &Zener, length: f64, speed: f64, modes: usize) -> RodContact {
    let n_mech = material.kappa.len();
    let solutions: Vec<(f64, ModeSolution)> = (1..=modes)
        .map(|n| {
            let m = (2 * n - 1) as f64;
            let k = m * std::f64::consts::PI / (2.0 * length);
            let start = ModeState {
                q: 0.0,
                rate: -4.0 * speed / (m * std::f64::consts::PI),
                memory: vec![0.0; n_mech],
            };
            (k, ModeSolution::new(material, k, &start))
        })
        .collect();
    let wall_stress = |t: f64| -> f64 {
        solutions
            .iter()
            .map(|(k, mode)| {
                let (s, _) = mode.at(t);
                k * (s.q + s.memory.iter().sum::<f64>())
            })
            .sum::<f64>()
            * material.relaxed
    };
    // Scan at a fraction of the highest mode's period, from just after the
    // touch, over four relaxed transit times (the slowest speed).
    let relaxed_speed = (material.relaxed / material.rho).sqrt();
    let span = 4.0 * length / relaxed_speed;
    let samples = 8 * modes;
    let dt = span / samples as f64;
    let mut before = dt;
    let mut after = None;
    for i in 1..=samples {
        let t = dt * i as f64;
        if wall_stress(t) >= 0.0 {
            after = Some(t);
            break;
        }
        before = t;
    }
    let mut after = after.expect("the rod separates within four relaxed transit times");
    for _ in 0..60 {
        let mid = 0.5 * (before + after);
        if wall_stress(mid) >= 0.0 {
            after = mid;
        } else {
            before = mid;
        }
    }
    let separation = after;
    // Per unit area: each mode's modal mass is rho L / 2.
    let modal_mass = 0.5 * material.rho * length;
    let (mut momentum, mut energy, mut heat, mut start) = (0.0, 0.0, 0.0, 0.0);
    for (k, mode) in &solutions {
        let (s, _) = mode.at(separation);
        momentum += material.rho * s.rate / k;
        energy += modal_mass * mode.energy(&s);
        heat += modal_mass * mode.heat(separation);
        let (s0, _) = mode.at(0.0);
        start += modal_mass * mode.energy(&s0);
    }
    let rigid = 0.5 * momentum * momentum / (material.rho * length);
    RodContact {
        separation,
        restitution: momentum / (material.rho * length * speed),
        split: (rigid / start, (energy - rigid) / start, heat / start),
        ledger: (energy + heat - start).abs() / start,
    }
}

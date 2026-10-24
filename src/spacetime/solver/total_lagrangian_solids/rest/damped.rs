//! A damped rest (`Rest` with relaxation branches): its constrained modes'
//! memory, advanced frame by frame by the flight's propagator, the bounds
//! that keep the rest valid, and the branches rebuilt at each frame's end
//! (the model on `rest`'s head).

use super::*;

/// A damped rest's modes, advanced one frame at a time.
#[derive(Clone)]
pub(super) struct Damped {
    /// Per mechanism, `e^(-theta length)`, and `length`, s.
    pub(super) decay: Vec<f64>,
    pub(super) length: f64,
    /// Per mode, its propagator (empty for a zero mode) and its static
    /// offset `b / omega^2` (zero for a zero mode).
    pub(super) propagators: Vec<Vec<f64>>,
    pub(super) offset: Vec<f64>,
    /// Per mode, `y = (omega (q - q_p), q', omega s_l / sqrt(kappa_l))` at
    /// `t`, and at the frame before (empty for a zero mode).
    pub(super) y: Vec<Vec<f64>>,
    pub(super) previous: Vec<Vec<f64>>,
    pub(super) t: f64,
    /// Per mode and mechanism, `s_il` at the rest's start.
    pub(super) start_memory: Vec<Vec<f64>>,
    /// Per mechanism, the multipliers of `r_l`, and its `theta`.
    pub(super) outside: Vec<(f64, Vec<f64>)>,
}

/// Per damped mode, `(omega, y at the last frame's start, y at its end)`,
/// and the mechanisms `(theta, kappa)` (`Rest::damped_modes`).
#[cfg(all(test, feature = "experimental"))]
pub(crate) type DampedModes = (Vec<(f64, Vec<f64>, Vec<f64>)>, Vec<(f64, f64)>);

pub(super) fn squared(y: &[f64]) -> f64 {
    y.iter().map(|e| e * e).sum()
}

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

impl Rest {
    /// For the gates: per damped mode, its `omega`, its state at the last
    /// frame's start and end, and the mechanisms `(theta, kappa)`.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn damped_modes(&self) -> Option<DampedModes> {
        let damped = self.damped.as_ref()?;
        let modes = (0..self.basis.omega.len())
            .filter(|&i| !damped.y[i].is_empty())
            .map(|i| {
                (
                    self.basis.omega[i],
                    damped.previous[i].clone(),
                    damped.y[i].clone(),
                )
            })
            .collect();
        Some((modes, self.basis.memory.clone()))
    }

    /// `start` without the strain check, for the probes' `settle`: the
    /// resting state of a body whose current motion is beyond its record's
    /// limit (it is then placed at rest, inside it).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn start_for_settling(
        basis: Basis,
        state: &TotalLagrangianState,
        (grid, planes): (&Grid, &[Plane]),
        (gravity, t, length): (DVec2, f64, f64),
    ) -> Result<Self, Refused> {
        let limit = basis.strain_limit;
        let mut unchecked = basis;
        unchecked.strain_limit = f64::INFINITY;
        let mut rest = Self::start(unchecked, state, (grid, planes), (gravity, t, length))?;
        rest.basis.strain_limit = limit;
        Ok(rest)
    }

    /// A bound on the Green strain size over the rest to `tau1` after its
    /// start (from the current frame on, with memory): `b + b^2 / 2` with
    /// `b = max_p |grad u_static| + sum_i |grad phi_i| a_i`, `a_i` each
    /// mode's reach about its static offset (`|y_i| / omega_i` with memory,
    /// its energy never growing; a zero mode's excursion to `tau1`).
    pub(super) fn strain_bound(&self, tau1: f64) -> f64 {
        let Some(damped) = &self.damped else {
            return 0.0;
        };
        let reach: Vec<f64> = (0..self.basis.omega.len())
            .map(|i| {
                let w = self.basis.omega[i];
                if w == 0.0 {
                    self.reach(i, tau1).1
                } else {
                    squared(&damped.y[i]).sqrt() / w
                }
            })
            .collect();
        let b = (0..self.start_gradients.len())
            .map(|p| {
                let fixed = self
                    .basis
                    .gradients
                    .iter()
                    .zip(&damped.offset)
                    .fold(self.start_gradients[p], |g, (grad, &q)| g + grad[p] * q);
                colon(fixed, fixed).sqrt()
                    + self
                        .basis
                        .gradients
                        .iter()
                        .zip(&reach)
                        .map(|(grad, &a)| colon(grad[p], grad[p]).sqrt() * a)
                        .sum::<f64>()
            })
            .fold(0.0, f64::max);
        b + 0.5 * b * b
    }

    /// `holds` with memory, from the current frame's start, `tau0 = tau1 -
    /// length`. The multipliers are `lambda_0 + sum_i c_i (q_i + sum_l
    /// (s_il - s_il(0))) + sum_l d_l (e^(-theta_l tau) - 1)`. Each mode's
    /// `|(q_i - q_p) + sum_l s_il|` is at most `sqrt(1 + sum_l kappa_l)
    /// |y_i| / omega_i` (Cauchy-Schwarz on `y`, which never grows), and the
    /// `d_l` term is monotone in `tau` (its least at an end of the frame).
    /// The free points' gaps take `|q_i - q_p| <= |y_i| / omega_i`, and the
    /// strain stays within the record's limit.
    pub(super) fn damped_holds(&self, damped: &Damped, planes: &[Plane], tau1: f64) -> bool {
        let tau0 = tau1 - damped.length;
        let weight: f64 = (1.0
            + self
                .basis
                .memory
                .iter()
                .map(|&(_, kappa)| kappa)
                .sum::<f64>())
        .sqrt();
        // Per mode: its centre and its reach about it, for the multipliers
        // and for the gaps.
        let modes: Vec<(f64, f64, f64)> = (0..self.basis.omega.len())
            .map(|i| {
                let w = self.basis.omega[i];
                if w == 0.0 {
                    let (centre, amplitude) = self.reach(i, tau1);
                    return (centre, amplitude, amplitude);
                }
                let size = squared(&damped.y[i]).sqrt() / w;
                let memory: f64 = damped.start_memory[i].iter().sum();
                (damped.offset[i] - memory, weight * size, size)
            })
            .collect();
        let compressive = self.multiplier.iter().enumerate().all(|(h, &l0)| {
            let mut low = modes
                .iter()
                .zip(&self.basis.coupling)
                .fold(l0, |low, (&(centre, amplitude, _), c)| {
                    low + c[h] * centre - c[h].abs() * amplitude
                });
            for (theta, d) in &damped.outside {
                let at = |tau: f64| d[h] * ((-theta * tau).exp() - 1.0);
                low += at(tau0.max(0.0)).min(at(tau1));
            }
            low > 0.0
        });
        compressive
            && self.free.iter().all(|&(q, i, gap)| {
                let plane = planes[i];
                let moved: f64 = modes
                    .iter()
                    .zip(&self.basis.at)
                    .zip(&damped.offset)
                    .map(|((&(_, _, amplitude), at), &offset)| {
                        at[q][plane.axis].abs() * (offset.abs() + amplitude)
                    })
                    .sum();
                gap - moved > 0.0
            })
            && self.strain_bound(tau1) <= self.basis.strain_limit
    }

    /// After a kept damped rest frame (`rebuild`), each particle's branches
    /// rebuilt from the constrained modes' memory (`Relaxation::carried`),
    /// from the deformation gradients `start` at the frame's start; returns
    /// the heat made, J/m, its split kept in `heat` (as in flight) and the
    /// largest strain in `strain`. Without memory, nothing.
    pub(crate) fn carry_branches(
        &mut self,
        state: &mut TotalLagrangianState,
        start: &[DMat2],
    ) -> f64 {
        self.heat.clear();
        self.strain = 0.0;
        let Some(damped) = &self.damped else {
            return 0.0;
        };
        let (omega, memory) = (&self.basis.omega, &self.basis.memory);
        let modal: f64 = damped
            .y
            .iter()
            .zip(&damped.previous)
            .map(|(y1, y0)| 0.5 * (squared(y0) - squared(y1)))
            .sum();
        let s = |y: &[f64], i: usize, l: usize| y[2 + l] * memory[l].1.sqrt() / omega[i];
        let n = state.particles;
        let (mut heat, mut stored) = (vec![0.0; n], vec![0.0; n]);
        for p in 0..n {
            let mut carried: Vec<CarriedMemory> = memory
                .iter()
                .zip(&damped.decay)
                .map(|(&(_, kappa), &decay)| CarriedMemory {
                    decay,
                    kappa,
                    start: DMat2::ZERO,
                    end: DMat2::ZERO,
                })
                .collect();
            for (i, (y1, y0)) in damped.y.iter().zip(&damped.previous).enumerate() {
                if y1.is_empty() {
                    continue;
                }
                let g = self.basis.gradients[i][p];
                let sym = (g + g.transpose()) * 0.5;
                for (l, c) in carried.iter_mut().enumerate() {
                    c.start += sym * s(y0, i, l);
                    c.end += sym * s(y1, i, l);
                }
            }
            let (c0, c1) = (
                start[p].transpose() * start[p],
                state.f[p].transpose() * state.f[p],
            );
            let solid = &mut state.solids[p];
            let moduli = solid.lame();
            let before = solid.relaxation.energy(c0, moduli);
            heat[p] = state.v0[p] * solid.relaxation.carried((c0, c1), moduli, &carried);
            stored[p] = state.v0[p] * (before + solid.relaxation.energy(c1, moduli));
            let green = (c1 - DMat2::IDENTITY) * 0.5;
            self.strain = self.strain.max(colon(green, green).sqrt());
        }
        let total: f64 = stored.iter().sum();
        let volume: f64 = state.v0[..n].iter().sum();
        for p in 0..n {
            heat[p] += modal
                * if total > 0.0 {
                    stored[p] / total
                } else {
                    state.v0[p] / volume
                };
        }
        let made = heat.iter().sum();
        self.heat = heat;
        made
    }
}

impl Damped {
    /// The memory at a rest's start: each constrained mode's `s_il =
    /// c_i^T f_l / omega_i^2` from the branches' force `f_l`, its state
    /// about the static offset (the coordinates start at zero), the
    /// multipliers of the part `r_l = f_l - sum_i K c_i s_il` no mode
    /// carries, and the propagators over `length`.
    pub(super) fn start(
        basis: &Basis,
        (forcing, rate): (&[f64], &[f64]),
        memory_force: &[Vec<f64>],
        (t, length): (f64, f64),
    ) -> Self {
        let n = basis.n;
        let count = basis.omega.len();
        let start_memory: Vec<Vec<f64>> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                memory_force
                    .iter()
                    .map(|f| {
                        if w == 0.0 {
                            0.0
                        } else {
                            dot(&basis.columns[i], f) / (w * w)
                        }
                    })
                    .collect()
            })
            .collect();
        let outside = basis
            .memory
            .iter()
            .enumerate()
            .map(|(l, &(theta, _))| {
                let mut r = memory_force[l].clone();
                for (i, column) in basis.columns.iter().enumerate() {
                    if start_memory[i][l] == 0.0 {
                        continue;
                    }
                    let kc = times(&basis.k, (n, n), column, 1);
                    for (e, v) in r.iter_mut().zip(kc) {
                        *e -= v * start_memory[i][l];
                    }
                }
                (theta, basis.multipliers(&r))
            })
            .collect();
        let offset: Vec<f64> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                if w == 0.0 { 0.0 } else { forcing[i] / (w * w) }
            })
            .collect();
        let y: Vec<Vec<f64>> = (0..count)
            .map(|i| {
                let w = basis.omega[i];
                if w == 0.0 {
                    return Vec::new();
                }
                let mut y = vec![-w * offset[i], rate[i]];
                for (l, &(_, kappa)) in basis.memory.iter().enumerate() {
                    y.push(w * start_memory[i][l] / kappa.sqrt());
                }
                y
            })
            .collect();
        let propagators = y
            .iter()
            .zip(&basis.omega)
            .map(|(y, &w)| {
                if y.is_empty() {
                    Vec::new()
                } else {
                    damped_propagator(w, &basis.memory, length)
                }
            })
            .collect();
        Self {
            decay: basis
                .memory
                .iter()
                .map(|&(theta, _)| (-theta * length).exp())
                .collect(),
            length,
            propagators,
            offset,
            previous: y.clone(),
            y,
            t,
            start_memory,
            outside,
        }
    }
}

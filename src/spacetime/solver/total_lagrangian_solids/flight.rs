//! A body's flight between contacts carried by its modes: the exact
//! solution of the linearised elastic problem in the body's frame, its
//! rigid motion exact under gravity, with no step at all. The stepper
//! resolves every contact; a flight starts only when the last step held
//! none and ends at the first frame the body could touch a wall, the
//! stepper taking over from the flight's exact state.
//!
//! The motion splits into its rigid part and its vibration:
//! - rigid: the centre of mass falls with `g` exactly; the rotation is the
//!   polar factor of `sum m (x - c) (X - C)^T`, turning at the spin that
//!   carries the angular momentum, `L / I`, so both are kept;
//! - vibration: the body-frame displacement and velocity projected on the
//!   elastic modes (`total_lagrangian::body_modes`), each advanced exactly
//!   as `z(t) = z(0) e^(i omega t)`, `z = omega q - i q'`, the undamped
//!   linear mode's own solution, its energy kept.
//!
//! Rebuilt into the step's own state, `x = c + R (X - C + u_b)`, `v = v_c +
//! Omega perp(x - c) + R v_b`, `F = R (I + grad u_b)`: the grid stays on the
//! reference, so the modal state is the step's and the hand-over needs no
//! projection.
//!
//! A turning body's frame adds terms these oscillators leave out. In a moving
//! frame the discretised displacement obeys
//! `M u'' + (D + G(Omega)) u' + K u + P(Omega') + N(Omega) u = R(Omega) + F`,
//! with `G = int 2 rho Phi^T Omega Phi`, `N = int rho Phi^T Omega^2 Phi` and
//! `R = -int rho Phi^T (R^T s'' + Omega' x + Omega^2 x)` (Legrand,
//! *Introduction to rotordynamics*, McGill Structural Dynamics and Vibration
//! Laboratory, 2009, slides 10-13, read). In the plane at constant spin,
//! `Omega = Omega J` (`J` the quarter turn), this is
//! `M u'' + 2 Omega M J u' + (K - Omega^2 M) u = Omega^2 M (X - C)`. On the
//! modes:
//! - the Coriolis coupling `2 Omega G q'`, `G_ij = phi_i^T M J phi_j`, at most
//!   `2 Omega / omega_1` of the stiffness term;
//! - the centrifugal softening, `(Omega / omega_1)^2`;
//! - the static centrifugal stretch.
//!
//! The vibration also changes `I`, so a constant spin does not keep `L`
//! exactly (measured: 2e-6 of `L` at 10 rad/s, 5e-4 at 50 rad/s on the
//! clay block). So a frame is flown only if its rebuilt state keeps the
//! flight's angular momentum to the sums' round-off and the computed modes'
//! own leak (`keeps_angular_momentum`); otherwise it is
//! stepped. In practice only a body that does not turn flies; a turning
//! body's flight needs those terms (not built).
//!
//! The material energy of the rebuilt state differs from the modes'
//! quadratic energy by the linear basis's residual (the law's
//! nonlinearity), reported per frame
//! (`TotalLagrangianFrame::modal_energy_change`). The step resumes from the
//! rebuilt state, so each hand-back keeps that residual of its moment
//! (measured: up to 1.6e-2 J/m, 2e-3 of the clay block's total, per flight).
//!
//! A body with relaxation branches flies when its modes carry them
//! (`total_lagrangian::body_modes`: one proportional spectrum). Each mode is
//! then a generalised Zener oscillator, advanced frame by frame by its exact
//! propagator in energy variables (`damped_propagator`); its memory starts
//! as the projection of the branches' force, `s_il = phi_i^T f_l /
//! omega_i^2`, and each particle's branches are rebuilt from it at every
//! frame's end (`Relaxation::carried`), so the step resumes from a
//! consistent state. Exact in the linear limit only: a flight whose strain
//! could pass the record's limit (the bound in `start`) is refused. The
//! heat is exact in total (each mode's energy lost, plus the decay of the
//! branch stress no mode sees); its split among particles is a declared
//! adaptation, by each particle's branch energy at the frame's start and
//! end. Removal criterion: the per-particle closed-form integral of the
//! dissipation, measured for cost.
//!
//! A J2 body (`HenckyJ2` without Norton's flow) flies while it has never
//! yielded and provably cannot over the flight: below yield it is Hencky's
//! elasticity, with the corotated law's modes (`total_lagrangian::
//! body_modes`), and `start` refuses it unless every particle stays
//! elastic for every gradient its bound allows (`HenckyJ2::
//! elastic_within`). A body that has yielded stays refused (`Refusal::
//! Yielded`).

use crate::matter::materials::solid::j2_hencky::PlasticMetric;
use crate::spacetime::total_lagrangian::{
    CarriedMemory, ElasticLaw, Modes, Plane, RIGID_MODES, damped_propagator,
};
use glam::{DMat2, DVec2};

use super::TotalLagrangianState;

/// One flight: its start time, the rigid state and each mode's `z` there.
#[derive(Clone, Debug)]
pub(crate) struct Flight {
    t: f64,
    centre: DVec2,
    velocity: DVec2,
    angle: f64,
    spin: f64,
    /// The reference centre of mass, `C`.
    origin: DVec2,
    /// Per mode, `z = omega q - i q'` in the body's frame, as `(re, im)`
    /// (zero for the rigid ones).
    z: Vec<DVec2>,
    /// The angular momentum about the centre at the start, kg m^2/s per
    /// metre.
    angular: f64,
    /// Per contact point, its reference arm `X - C`, and the bound on its
    /// vibration's displacement, `sum_i |phi_i(X)| |z_i| / omega_i` (`|y_i|`
    /// in place of `|z_i|` with memory: the mode's energy never grows).
    arms: Vec<(DVec2, f64)>,
    /// Per mode, its angular momentum per unit rate about the reference
    /// centre, `leak_i = sum_p m_p (X_p - C) x phi_i(X_p)` (zero for an
    /// exact elastic mode; `keeps_angular_momentum`), and each mode's rate
    /// `q_i'` in the last rebuilt state.
    leak: Vec<f64>,
    rates: Vec<f64>,
    /// The modal memory of a body with relaxation branches.
    memory: Option<Memory>,
    /// What the last frame flown made: each particle's heat, J/m, and the
    /// largest Green strain size `|C - I| / 2` at its end.
    pub(crate) heat: Vec<f64>,
    pub(crate) strain: f64,
}

/// A damped flight's modes, advanced one frame at a time.
#[derive(Clone, Debug)]
struct Memory {
    /// Per mechanism, `e^(-theta length)`, and `length`, s.
    decay: Vec<f64>,
    length: f64,
    /// Per mode, its propagator over `length` (empty for the rigid ones).
    propagators: Vec<Vec<f64>>,
    /// Per mode, its state `y = (omega q, q', omega s_l / sqrt(kappa_l))`
    /// at `t`, and at the frame before.
    y: Vec<Vec<f64>>,
    previous: Vec<Vec<f64>>,
    t: f64,
}

fn squared(y: &[f64]) -> f64 {
    y.iter().map(|e| e * e).sum()
}

fn colon(a: DMat2, b: DMat2) -> f64 {
    a.x_axis.dot(b.x_axis) + a.y_axis.dot(b.y_axis)
}

/// Why a flight was not started (`Flight::start`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Refusal {
    /// The body carries no mass: nothing to fly.
    Massless,
    /// A J2 particle has yielded: its stress-free configuration is no
    /// longer the reference the modes are built about. Removal criterion:
    /// modes about the current plastic state.
    Yielded,
    /// The flight's strain bound could pass a damped record's strain limit,
    /// or a J2 particle's yield (`HenckyJ2::elastic_within`).
    Bound,
}

impl Flight {
    /// The flight from the state at time `t`, its frames `length` long.
    /// Refused (`Refusal`) when the body has no mass, when a J2 particle
    /// has yielded, or when the strain could pass a limit: over the whole
    /// flight (no mode's energy grows) `|grad u| <= b_p = sum_i |grad
    /// phi_i(X_p)| |y_i| / omega_i` at each particle, so a damped flight's
    /// Green strain stays below `b + b^2 / 2` (`b` the largest `b_p`),
    /// which must not exceed its record's limit, and a J2 particle must
    /// stay elastic for every `|grad u| <= b_p`.
    pub(crate) fn start(
        state: &TotalLagrangianState,
        modes: &Modes,
        (t, length): (f64, f64),
    ) -> Result<Self, Refusal> {
        let n = state.particles;
        let total: f64 = state.mass[..n].iter().sum();
        if total <= 0.0 {
            return Err(Refusal::Massless);
        }
        let plastic = |p: usize| match state.solids[p].law {
            ElasticLaw::HenckyJ2(law, metric) => Some((law, metric)),
            _ => None,
        };
        if (0..n).any(|p| plastic(p).is_some_and(|(_, m)| m != PlasticMetric::unstrained())) {
            return Err(Refusal::Yielded);
        }
        let (sum_x, sum_v, sum_reference) =
            (0..n).fold((DVec2::ZERO, DVec2::ZERO, DVec2::ZERO), |(a, b, c), p| {
                let m = state.mass[p];
                (
                    a + m * state.x[p],
                    b + m * state.v[p],
                    c + m * state.reference[p],
                )
            });
        let (centre, velocity, origin) = (sum_x / total, sum_v / total, sum_reference / total);
        let (momentum, inertia) = (0..n).fold((0.0, 0.0), |(l, i), p| {
            let arm = state.x[p] - centre;
            (
                l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
                i + state.mass[p] * arm.length_squared(),
            )
        });
        let spin = momentum / inertia.max(f64::MIN_POSITIVE);
        // The polar factor of `A = sum m (x - c) (X - C)^T`: in the plane,
        // the rotation by `atan2(A_10 - A_01, A_00 + A_11)`.
        let a = (0..n).fold(DMat2::ZERO, |a, p| {
            let (r, s) = (state.x[p] - centre, state.reference[p] - origin);
            a + state.mass[p] * DMat2::from_cols(r * s.x, r * s.y)
        });
        let angle = (a.x_axis.y - a.y_axis.x).atan2(a.x_axis.x + a.y_axis.y);
        let back = DMat2::from_angle(-angle);
        let z: Vec<DVec2> = modes
            .at
            .iter()
            .zip(&modes.omega)
            .enumerate()
            .map(|(i, (at, &w))| {
                if i < RIGID_MODES {
                    return DVec2::ZERO;
                }
                let (q, rate) = (0..n).fold((0.0, 0.0), |(q, rate), p| {
                    let arm = state.x[p] - centre;
                    let u = back * arm - (state.reference[p] - origin);
                    let v = back * (state.v[p] - velocity - spin * arm.perp());
                    (
                        q + state.mass[p] * u.dot(at[p]),
                        rate + state.mass[p] * v.dot(at[p]),
                    )
                });
                DVec2::new(w * q, -rate)
            })
            .collect();
        let leak = modes
            .at
            .iter()
            .enumerate()
            .map(|(i, at)| {
                if i < RIGID_MODES {
                    return 0.0;
                }
                (0..n)
                    .map(|p| state.mass[p] * (state.reference[p] - origin).perp_dot(at[p]))
                    .sum()
            })
            .collect();
        let memory =
            (!modes.memory.is_empty()).then(|| Memory::start(state, modes, &z, (t, length)));
        // Each mode's size: `|z|`, or `|y|` with memory.
        let size = |i: usize| match &memory {
            Some(m) => squared(&m.y[i]).sqrt(),
            None => z[i].length(),
        };
        let gradient_bound = |p: usize| -> f64 {
            (RIGID_MODES..modes.omega.len())
                .filter(|&i| modes.omega[i] > 0.0)
                .map(|i| {
                    let g = modes.gradients[i][p];
                    colon(g, g).sqrt() * size(i) / modes.omega[i]
                })
                .sum()
        };
        if memory.is_some() {
            let bound = (0..n).map(gradient_bound).fold(0.0, f64::max);
            if bound + 0.5 * bound * bound > modes.strain_limit {
                return Err(Refusal::Bound);
            }
        }
        let leaves_elastic = (0..n).any(|p| {
            plastic(p).is_some_and(|(law, metric)| !law.elastic_within(&metric, gradient_bound(p)))
        });
        if leaves_elastic {
            return Err(Refusal::Bound);
        }
        let arms = state
            .outlines
            .iter()
            .flatten()
            .map(|&o| {
                let bound: f64 = (RIGID_MODES..modes.omega.len())
                    .filter(|&i| modes.omega[i] > 0.0)
                    .map(|i| modes.at[i][o].length() * size(i) / modes.omega[i])
                    .sum();
                (state.reference[o] - origin, bound)
            })
            .collect();
        Ok(Self {
            t,
            centre,
            velocity,
            angle,
            spin,
            angular: momentum,
            origin,
            z,
            arms,
            leak,
            rates: Vec::new(),
            memory,
            heat: Vec::new(),
            strain: 0.0,
        })
    }

    /// The centre of mass and its velocity at time `t`.
    fn rigid_at(&self, gravity: DVec2, t: f64) -> (DVec2, DVec2) {
        let tau = t - self.t;
        (
            self.centre + self.velocity * tau + 0.5 * gravity * tau * tau,
            self.velocity + gravity * tau,
        )
    }

    /// Whether every contact point stays clear of every wall over `[t0,
    /// t1]`, by a lower bound on its gap: the centre's exact parabola plus
    /// the turned reference arm, less the vibration's bound (each mode's
    /// displacement never exceeds `|z| / omega`).
    pub(crate) fn clear(&self, gravity: DVec2, planes: &[Plane], (t0, t1): (f64, f64)) -> bool {
        planes.iter().all(|plane| {
            // The least of `inward c_axis(tau)` over the interval: at an end,
            // or at the parabola's vertex when it lies inside.
            let along = |t: f64| plane.inward * self.rigid_at(gravity, t).0[plane.axis];
            let mut lowest = along(t0).min(along(t1));
            let acceleration = plane.inward * gravity[plane.axis];
            if acceleration != 0.0 {
                let vertex = self.t - plane.inward * self.velocity[plane.axis] / acceleration;
                if vertex > t0 && vertex < t1 {
                    lowest = lowest.min(along(vertex));
                }
            }
            // The arm turned to `t0`, and at most `|Omega| |r| (t1 - t0)` from
            // there over the interval (the turning point's speed).
            let turn = DMat2::from_angle(self.angle + self.spin * (t0 - self.t));
            self.arms.iter().all(|&(reference, bound)| {
                let arm = turn * reference;
                let turning = self.spin.abs() * reference.length() * (t1 - t0);
                lowest + plane.inward * arm[plane.axis] - plane.inward * plane.at - bound - turning
                    > 0.0
            })
        })
    }

    /// Whether `state` (rebuilt) keeps the flight's angular momentum to the
    /// round-off of the two sums, `2 n eps sum m |x - c| (|v| + |v_c|)`,
    /// plus what the modes themselves carry, `sum_i |leak_i| |q_i'|`.
    ///
    /// Derived on the page: the rebuilt vibration's angular momentum is
    /// `sum_p m_p (X_p - C + u_p) x u_p'` (the turn cancels), whose first
    /// order is `sum_i leak_i q_i'`. An exact elastic mode has no angular
    /// momentum (it is mass-orthogonal to the rigid rotation), but the
    /// computed ones carry `leak_i`. They are orthogonal to the solve's own
    /// rigid modes to 7.8e-15, so it is that rigid span that differs from the
    /// geometric rotation; the size fits the shifted matrix's round-off over
    /// the gap to the lowest elastic mode, small on a slender body (not
    /// verified apart). Measured (`probe_modes_rotation_leak`): on the ringing bar the
    /// first order alone gives the change to four digits (1.638e-17 for
    /// 1.638e-17 kg m^2/s per metre), with `|leak_i|` up to 2.2e-10 of
    /// `sqrt(I)`; on the clay block 7.6e-14. So the bound includes it. The
    /// first order then holds by the triangle inequality, not with an
    /// independent margin (the bar's drift is 0.83 of the bound); what is
    /// left to detect is the second order and the turning frame's terms
    /// (`flight`'s head), and it does fire (frames 37 and 45 of
    /// `probe_flown_frames_against_the_step`).
    pub(crate) fn keeps_angular_momentum(&self, state: &TotalLagrangianState) -> bool {
        let n = state.particles;
        let total: f64 = state.mass[..n].iter().sum();
        let (sum_x, sum_v) = (0..n).fold((DVec2::ZERO, DVec2::ZERO), |(a, b), p| {
            (
                a + state.mass[p] * state.x[p],
                b + state.mass[p] * state.v[p],
            )
        });
        let (centre, velocity) = (sum_x / total, sum_v / total);
        let (momentum, size) = (0..n).fold((0.0, 0.0), |(l, s), p| {
            let arm = state.x[p] - centre;
            (
                l + state.mass[p] * arm.perp_dot(state.v[p] - velocity),
                s + state.mass[p] * arm.length() * (state.v[p].length() + velocity.length()),
            )
        });
        let carried: f64 = self
            .leak
            .iter()
            .zip(&self.rates)
            .map(|(l, r)| (l * r).abs())
            .sum();
        (momentum - self.angular).abs() <= 2.0 * n as f64 * f64::EPSILON * size + carried
    }

    /// The flight's state at time `t`, written into the step's state; with
    /// memory, `t` is one frame after the last. False, the state untouched,
    /// when a damped mode's computed propagator lengthened its state (the
    /// exact one never does, `damped_propagator`): the runtime check of the
    /// matrix exponential's rounding, which its source leaves open.
    pub(crate) fn rebuild(
        &mut self,
        modes: &Modes,
        gravity: DVec2,
        t: f64,
        state: &mut TotalLagrangianState,
    ) -> bool {
        let tau = t - self.t;
        let (centre, velocity) = self.rigid_at(gravity, t);
        let coordinates: Vec<DVec2> = if let Some(memory) = &mut self.memory {
            assert!(
                (t - memory.t - memory.length).abs()
                    <= 4.0 * f64::EPSILON * t.abs().max(memory.length),
                "a damped flight advances one frame at a time"
            );
            let next: Vec<Vec<f64>> = memory
                .y
                .iter()
                .zip(&memory.propagators)
                .map(|(y, p)| {
                    let k = y.len();
                    (0..k)
                        .map(|r| (0..k).map(|c| p[r * k + c] * y[c]).sum())
                        .collect()
                })
                .collect();
            if next
                .iter()
                .zip(&memory.y)
                .any(|(a, b)| squared(a) > squared(b))
            {
                return false;
            }
            memory.previous = std::mem::replace(&mut memory.y, next);
            memory.t = t;
            // `q = y_0 / omega`, `q' = y_1`.
            memory
                .y
                .iter()
                .zip(&modes.omega)
                .map(|(y, &w)| {
                    if y.is_empty() {
                        DVec2::ZERO
                    } else {
                        DVec2::new(y[0] / w, y[1])
                    }
                })
                .collect()
        } else {
            // `q = Re z / omega`, `q' = -Im z`, each `z` turned by `omega tau`.
            self.z
                .iter()
                .zip(&modes.omega)
                .map(|(z0, &w)| {
                    if w <= 0.0 {
                        return DVec2::ZERO;
                    }
                    let zt = DVec2::from_angle(w * tau).rotate(*z0);
                    DVec2::new(zt.x / w, -zt.y)
                })
                .collect()
        };
        self.rates = coordinates.iter().map(|c| c.y).collect();
        let turn = DMat2::from_angle(self.angle + self.spin * tau);
        for p in 0..state.x.len() {
            let (mut u, mut v) = (DVec2::ZERO, DVec2::ZERO);
            for (i, c) in coordinates.iter().enumerate().skip(RIGID_MODES) {
                u += modes.at[i][p] * c.x;
                v += modes.at[i][p] * c.y;
            }
            let arm = turn * (state.reference[p] - self.origin + u);
            state.x[p] = centre + arm;
            state.v[p] = velocity + self.spin * arm.perp() + turn * v;
            if p < state.particles {
                let g = coordinates
                    .iter()
                    .enumerate()
                    .skip(RIGID_MODES)
                    .fold(DMat2::ZERO, |g, (i, c)| g + modes.gradients[i][p] * c.x);
                state.f[p] = turn * (DMat2::IDENTITY + g);
            }
        }
        true
    }

    /// After a kept damped frame (`rebuild`), each particle's branches
    /// rebuilt from the modal memory (`Relaxation::carried`), from the
    /// deformation gradients `start` at the frame's start; returns the heat
    /// the frame made, J/m, and keeps its split and the largest strain in
    /// `heat` and `strain`. Without memory, nothing.
    pub(crate) fn carry_branches(
        &mut self,
        modes: &Modes,
        state: &mut TotalLagrangianState,
        start: &[DMat2],
    ) -> f64 {
        self.heat.clear();
        self.strain = 0.0;
        let Some(memory) = &self.memory else {
            return 0.0;
        };
        let n = state.particles;
        // Each mode's energy lost, `(|y_0|^2 - |y_1|^2) / 2`, never negative
        // (`rebuild`'s check).
        let modal: f64 = memory
            .y
            .iter()
            .zip(&memory.previous)
            .map(|(y1, y0)| 0.5 * (squared(y0) - squared(y1)))
            .sum();
        let s =
            |y: &[f64], i: usize, l: usize| y[2 + l] * modes.memory[l].1.sqrt() / modes.omega[i];
        let (mut heat, mut stored) = (vec![0.0; n], vec![0.0; n]);
        for p in 0..n {
            let mut carried: Vec<CarriedMemory> = modes
                .memory
                .iter()
                .zip(&memory.decay)
                .map(|(&(_, kappa), &decay)| CarriedMemory {
                    decay,
                    kappa,
                    start: DMat2::ZERO,
                    end: DMat2::ZERO,
                })
                .collect();
            for (i, (y1, y0)) in memory.y.iter().zip(&memory.previous).enumerate() {
                if y1.is_empty() {
                    continue;
                }
                let g = modes.gradients[i][p];
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

impl Memory {
    /// The memory at a flight's start: each mode's state with `s_il =
    /// phi_i^T f_l / omega_i^2`, `phi_i^T f_l = sum_p V0_p sigma_lp : grad
    /// phi_i(X_p)` with `sigma_lp` the branches' linearised stress
    /// (`Relaxation::linear_stress`), and the propagators over `length`.
    fn start(
        state: &TotalLagrangianState,
        modes: &Modes,
        z: &[DVec2],
        (t, length): (f64, f64),
    ) -> Self {
        let n = state.particles;
        let stresses: Vec<_> = (0..n)
            .map(|p| {
                let solid = &state.solids[p];
                solid
                    .relaxation
                    .linear_stress(state.f[p].transpose() * state.f[p], solid.lame())
            })
            .collect();
        let y: Vec<Vec<f64>> = (0..modes.omega.len())
            .map(|i| {
                let w = modes.omega[i];
                if i < RIGID_MODES || w <= 0.0 {
                    return Vec::new();
                }
                // `z = omega q - i q'`; `y_l = omega s_l / sqrt(kappa_l)`.
                let mut y = vec![z[i].x, -z[i].y];
                for (l, &(_, kappa)) in modes.memory.iter().enumerate() {
                    let force: f64 = (0..n)
                        .map(|p| state.v0[p] * colon(stresses[p][l], modes.gradients[i][p]))
                        .sum();
                    y.push(force / w / kappa.sqrt());
                }
                y
            })
            .collect();
        let propagators = y
            .iter()
            .zip(&modes.omega)
            .map(|(y, &w)| {
                if y.is_empty() {
                    Vec::new()
                } else {
                    damped_propagator(w, &modes.memory, length)
                }
            })
            .collect();
        Self {
            decay: modes
                .memory
                .iter()
                .map(|&(theta, _)| (-theta * length).exp())
                .collect(),
            length,
            propagators,
            previous: y.clone(),
            y,
            t,
        }
    }
}

//! Choosing the length of each time step from the error it makes: a step
//! is tried whole and in halves (and in thirds while a contact is active),
//! the difference between the tries estimates its error, and the step is
//! kept if that error is small enough, redone shorter if not. The next
//! step is as long as the error allows.
//!
//! Source: Klapproth, Schiela and Deuflhard, Numer. Math. 119 (2011)
//! 49-81 (ZIB-Report 10-09), sec. 4: the error in the energy norm `||(u,
//! u')||_E^2 = 1/2 ||u'||_M^2 + 1/2 a(u, u)` (eqs. 11-12); without contact
//! the Richardson estimate of the halved step against the extrapolation
//! (eqs. 31-33) and the next step `tau* = (rho TOL / err)^(1/3) tau` (eq.
//! 42); with contact the error model `e tau^2 + X tau^(1/2)` (eqs. 34-35),
//! the third try at `tau / 3`, the combined extrapolation (eqs. 36-38),
//! the estimate (39), `X`'s estimate (p. 26) and the next step as the root
//! of eq. 44.
//!
//! Adaptations:
//! - Each kept step is this engine's step (the total Lagrangian conserving
//!   step with route C contact), not their contact-stabilized Newmark: the
//!   estimate only compares tries, the extrapolations are never kept, so
//!   every kept step keeps the energy exactly. Their `tau^(1/2)` contact
//!   model is proven for their scheme; on this one it was measured (the
//!   velocity's error through a bar's impact falls as `tau^0.4` to
//!   `tau^0.5`, `probe_bar_impact`).
//! - The energy norm is taken on the particles: kinetic from their mean
//!   velocity over the step (`controlled_step`'s note), `a(u, u)` as `sum V0 (2 mu |sym dF|^2 + lambda (tr
//!   dF)^2)` from their deformation gradients.
//! - The tolerance is relative: `err <= tol sqrt(E0)`, so the error's
//!   energy per step is at most `tol^2 E0`, with `E0` the scene's
//!   `tolerance_energy` at its start (derived there).
//! - `X(t + tau*) = X(t + tau)` throughout (their assumption inside a
//!   contact interval); their ramp across a switch (fig. 5) and the reuse
//!   of a rejected step's `X` are not built.

use super::*;

/// The energy a scene's step tolerance is relative to (`E0` in `err <= tol
/// sqrt(E0)`): its kinetic and stored energy at the start, plus, for each
/// body, the work gravity does over the body's own extent along it, `m_b
/// |g| L_b`; `bodies` gives each body's mass and the points of its
/// reference configuration (its outline among them).
///
/// Derived: a relative tolerance needs a reference energy, and a scene's
/// accuracy must not depend on where it sits. Kinetic and stored energy
/// are invariant under a translation; gravity's potential `-m g.x` is not:
/// it is defined up to a constant, the origin's choice, so no absolute
/// value of it can enter. What gravity contributes physically is its work
/// between two positions; the one difference fixed by a body alone (a
/// fall's height depends on what lies below it) is its own extent along
/// `g`, the work of a settle or a topple of the body over its own size. A
/// body at rest and unstrained, where the first two terms vanish, still
/// gets a scale. No constant is chosen.
///
/// Measured before (`probe_translated_scene_step_sequence`): with the
/// start's total energy, the potential from the grid's corner, the same
/// 4-cell drop moved by 1 m took 30 steps instead of 800 and ended 8 mm
/// apart.
pub fn tolerance_energy<'a>(
    kinetic_and_stored: f64,
    gravity: DVec2,
    bodies: impl IntoIterator<Item = (f64, &'a [DVec2])>,
) -> f64 {
    let g = gravity.length();
    if g == 0.0 {
        return kinetic_and_stored;
    }
    let along = gravity / g;
    let work: f64 = bodies
        .into_iter()
        .map(|(mass, points)| {
            let (low, high) = points
                .iter()
                .map(|x| along.dot(*x))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), d| {
                    (l.min(d), h.max(d))
                });
            if high >= low {
                mass * g * (high - low)
            } else {
                0.0
            }
        })
        .sum();
    kinetic_and_stored + work
}

/// Whether `EMERGE_PROBE_ERROR_SPLIT` is set (read once): the controller's
/// error split probe (`controlled_step`). A probe knob, in test builds
/// only.
#[cfg(all(test, feature = "experimental"))]
fn error_split() -> bool {
    static SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SET.get_or_init(|| std::env::var_os("EMERGE_PROBE_ERROR_SPLIT").is_some())
}

/// The controller's settings: `tolerance` relative to the square root of
/// the scene's `tolerance_energy`, `safety` their `rho`, `growth` the largest ratio of one step to
/// the last, `largest` the longest step (s).
/// Source: Klapproth, Schiela and Deuflhard 2011, Table 2 (`rho` 0.9,
/// growth 10).
#[derive(Clone, Copy, Debug)]
pub struct StepControl {
    pub tolerance: f64,
    pub safety: f64,
    pub growth: f64,
    pub largest: f64,
    /// The three tries' substep counts: their `(1, 2, 3)`, or an even
    /// sequence `(2, 4, 6)`. The midpoint rule's error expansions differ
    /// for even and odd step counts, so extrapolation uses even sequences.
    /// Source: Exl, Mauser, Schrefl and Suess, arXiv 1703.02479, sec. 3
    /// (after Gragg 1965). Adaptation: KSD's eqs. 32-38 recomputed for
    /// these ratios.
    pub substeps: [usize; 3],
    /// Which estimate the tries feed (`Estimator`).
    pub estimator: Estimator,
}

/// The error estimate a controlled step uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimator {
    /// Klapproth, Schiela and Deuflhard's, with the `tau^(1/2)` contact
    /// term (this file's head).
    Klapproth,
    /// Acary's for the Moreau-Jean scheme: a step and its halves; on a step
    /// where a point reaches a wall (free at its start, beyond at its end
    /// in either try) the positions' difference alone at order 0, else the
    /// standard estimate at order 2.
    /// Source: Acary, INRIA RR-7151 (2010), sec. 5.2-5.3, eqs. 63-70 (his
    /// test values: step ratio within 0.5 and 5, alpha 1). He calls it
    /// heuristic, "only justified in several examples", and notes that
    /// "an adaptive time-step strategy is not very efficient on low-order
    /// time-stepping schemes". Adaptations: the energy norm against `rho
    /// tol sqrt(E0)` in place of his weighted norm (eq. 65), its elastic
    /// part as the positions' error; reaching detected against the planes
    /// only.
    Acary,
}

/// What one controlled step did: the time it covered, the substeps kept
/// (their reports, `dt / substeps` each), how many tries were rejected,
/// how many steps were solved in all, its error over the tolerance's
/// scale, whether a contact was active, and the next step's length.
pub struct ControlledStep {
    pub dt: f64,
    pub kept: Vec<StepReport>,
    pub rejected: u32,
    pub solves: u32,
    pub error: f64,
    pub contact: bool,
    pub next: f64,
    /// The step was taken at the estimator's validity floor without
    /// meeting the tolerance (`controlled_step`): a failure of the
    /// estimator, not an accuracy reached.
    pub at_floor: bool,
    /// The error's kinetic and elastic parts, over the start's norm.
    pub error_parts: (f64, f64),
    /// Of the rejected tries, those rejected because Newton did not
    /// converge (the others were over the tolerance).
    pub unconverged: u32,
    /// Kept tries whose Newton stagnated (`StepReport::stagnated`): a
    /// reported failure, not halved, as a shorter step does not lower the
    /// residual's noise.
    pub stagnated: u32,
    /// Whether the kept step was one where a point reached a wall
    /// (Acary's order-0 estimate).
    pub reaching: bool,
    /// Every try's time by part, the rejected ones' included, and the
    /// controller's own (`Timing::controller`).
    pub timing: Timing,
}

/// The particles' state a try changes.
#[derive(Clone)]
struct State {
    x: Vec<DVec2>,
    v: Vec<DVec2>,
    f: Vec<DMat2>,
    /// The solids, their plastic states advanced by each kept step.
    solids: Vec<Elastic>,
    cache: Option<LagrangianCache>,
}

/// `a s + b t`, the states' linear combination (positions, velocities and
/// deformation gradients alike; the cache is not combined).
fn combine(a: f64, s: &State, b: f64, t: &State) -> State {
    State {
        x: s.x.iter().zip(&t.x).map(|(p, q)| a * *p + b * *q).collect(),
        v: s.v.iter().zip(&t.v).map(|(p, q)| a * *p + b * *q).collect(),
        f: s.f.iter().zip(&t.f).map(|(p, q)| a * *p + b * *q).collect(),
        solids: s.solids.clone(),
        cache: None,
    }
}

/// A rejected pair's band factors kept in the step's cache (`Factors::
/// absorb`), so the next try, which starts from it, reuses them instead of
/// building its own again.
fn keep_factors(start: &mut State, tries: [&State; 2]) {
    if let Some(cache) = start.cache.as_mut() {
        for tried in tries {
            if let Some(other) = tried.cache.as_ref() {
                cache.factors.absorb(&other.factors);
            }
        }
    }
}

/// One controlled step of the total Lagrangian contact step from the
/// particles' state, starting with the step `tau` (s) and setting it to
/// the next step's length; `scale` is the start's energy norm, `sqrt(E0)`.
pub fn controlled_step(
    layout: &GridLayout,
    particles: Particles<'_>,
    (reference, solids): (&[DVec2], &mut [Elastic]),
    (gravity, tau): (DVec2, &mut f64),
    (set, penalty, settings): (ContactSet<'_>, &[f64], &ConservingSettings),
    cache: &mut Option<LagrangianCache>,
    (control, scale): (&StepControl, f64),
) -> ControlledStep {
    let Particles { x, v, f, mass, v0 } = particles;
    // The tries' time by part, and their wall time in all (each try's
    // whole run, timed around it: their phases leave out the multipliers'
    // responses and the contact passes, which then read as the
    // controller's), the rest the controller's own.
    let controlling = std::time::Instant::now();
    let (mut timing, trying) = (Timing::default(), std::cell::Cell::new(0.0f64));
    let account = |reports: &[StepReport], timing: &mut Timing| {
        for r in reports {
            timing.add(&r.timing);
        }
    };
    let mut start = State {
        x: x.to_vec(),
        v: v.to_vec(),
        f: f.to_vec(),
        solids: solids.to_vec(),
        cache: cache.take(),
    };
    // The kinetic part on the step's mean velocity, the displacement over
    // the step over its length: the tries end on different parities of
    // the contact points' velocity alternation (route C holds the mean,
    // `u' (n+1) = -u'(n)` at an active point), which the end velocity
    // would measure as error.
    // Source: Exl, Mauser, Schrefl and Suess, arXiv 1703.02479, sec. 3
    // (after Gragg 1965): the midpoint rule's error expansions differ for
    // even and odd step counts, and Gragg's smoothing removes the
    // alternating term. Adaptation: the analogy between the explicit
    // midpoint's parasitic root and the contact's alternation is this
    // engine's, measured, not proven.
    let norm = |d: &State, dt: f64| -> f64 {
        let mut energy = 0.0;
        for p in 0..d.x.len() {
            energy += 0.5 * mass[p] * (d.x[p] / dt).length_squared();
            let g = d.f[p];
            let shear =
                g.x_axis.x.powi(2) + g.y_axis.y.powi(2) + 0.5 * (g.y_axis.x + g.x_axis.y).powi(2);
            let trace = g.x_axis.x + g.y_axis.y;
            let lambda = solids[p].bulk - solids[p].mu;
            energy += 0.5 * v0[p] * (2.0 * solids[p].mu * shear + lambda * trace * trace);
        }
        energy.max(0.0).sqrt()
    };
    // The norm's two parts.
    let split = |d: &State, dt: f64| -> (f64, f64) {
        let (mut kinetic, mut elastic) = (0.0, 0.0);
        for p in 0..d.x.len() {
            kinetic += 0.5 * mass[p] * (d.x[p] / dt).length_squared();
            let g = d.f[p];
            let shear =
                g.x_axis.x.powi(2) + g.y_axis.y.powi(2) + 0.5 * (g.y_axis.x + g.x_axis.y).powi(2);
            let trace = g.x_axis.x + g.y_axis.y;
            let lambda = solids[p].bulk - solids[p].mu;
            elastic += 0.5 * v0[p] * (2.0 * solids[p].mu * shear + lambda * trace * trace);
        }
        (kinetic.max(0.0).sqrt(), elastic.max(0.0).sqrt())
    };
    let mut solves = 0u32;
    let mut run = |start: &State, pieces: usize, dt: f64| -> (State, Vec<StepReport>, bool) {
        let tried = std::time::Instant::now();
        let mut s = start.clone();
        let mut reports = Vec::with_capacity(pieces);
        let mut contact = false;
        for _ in 0..pieces {
            let stateful = s.solids.iter().any(Elastic::carries_state);
            let before = if stateful { s.f.clone() } else { Vec::new() };
            let mut report = step_total_lagrangian_contact(
                layout,
                Particles {
                    x: &mut s.x,
                    v: &mut s.v,
                    f: &mut s.f,
                    mass,
                    v0,
                },
                reference,
                &s.solids,
                (dt / pieces as f64, ContactMode::Velocity, gravity),
                (set, penalty, settings),
                &mut s.cache,
            );
            // The kept step advances the plastic states (the return map at
            // the step's end) and the relaxation branches (their update), their
            // dissipation into the particles' heat.
            if stateful {
                let heat = s.cache.as_mut().map(|c| &mut c.heat);
                let mut made = vec![0.0; s.solids.len()];
                let (mut plastic, mut relaxation) = (0.0, 0.0);
                for p in 0..s.solids.len() {
                    if !s.solids[p].carries_state() || v0[p] <= 0.0 {
                        continue;
                    }
                    let (c0, c1) = (
                        before[p].transpose() * before[p],
                        s.f[p].transpose() * s.f[p],
                    );
                    let committed = s.solids[p].committed(c0, c1, dt / pieces as f64);
                    let (next, dissipated, relaxed) =
                        (committed.solid, committed.plastic, committed.relaxed);
                    report.norton_iterations = report.norton_iterations.max(committed.iterations);
                    // Counted only where the flow made more heat than the
                    // step resolves (its residual work, the bound on its
                    // energy error): a body at round-off stress flows at
                    // 1e-50 1/s under a law with no yield, and that is no
                    // flow to report.
                    if v0[p] * committed.plastic > report.residual_work {
                        use crate::matter::materials::solid::j2_hencky::Beyond;
                        match committed.outside {
                            Some(Beyond::Slower) => report.norton_slower += 1,
                            Some(Beyond::Faster) => report.norton_faster += 1,
                            Some(Beyond::Tension) => report.norton_tension += 1,
                            None => {}
                        }
                    }
                    if s.solids[p].is_relaxing() {
                        report.relaxation_strain = report.relaxation_strain.max(relaxed.strain);
                        report.relaxation_beyond += u32::from(relaxed.beyond);
                    }
                    s.solids[p] = next;
                    plastic += v0[p] * dissipated;
                    relaxation += v0[p] * relaxed.heat;
                    made[p] = v0[p] * (dissipated + relaxed.heat);
                }
                (report.plastic, report.relaxation) = (plastic, relaxation);
                if let Some(heat) = heat {
                    heat.resize(made.len().max(heat.len()), 0.0);
                    for (h, m) in heat.iter_mut().zip(&made) {
                        *h += m;
                    }
                }
            }
            contact |= report.push > 0.0 || report.contacts > 0;
            reports.push(report);
        }
        solves += pieces as u32;
        trying.set(trying.get() + tried.elapsed().as_secs_f64());
        (s, reports, contact)
    };
    let target = control.tolerance * scale;
    // The estimator's validity floor: below this step the round-off of the
    // tries' displacements over the step, `4 eps |x| / tau` in the norm's
    // kinetic part, reaches the error the tolerance asks to resolve, so
    // the estimate no longer measures anything (a numerical-validity
    // floor, declared; never a physical one), nor below the time's own
    // resolution `4 eps t`.
    let positions = (0..start.x.len())
        .map(|p| mass[p] * start.x[p].length_squared())
        .sum::<f64>()
        .sqrt();
    let floor = (4.0 * f64::EPSILON * positions / (control.safety * target).max(f64::MIN_POSITIVE))
        .max(4.0 * f64::EPSILON * control.largest);
    let [k1, k2, k3] = control.substeps.map(|k| k as f64);
    // Eq. 32 for the counts `a < b`: the combination that cancels the
    // `tau^2` term, and the `tau^(1/2)` term it keeps.
    let pair = |a: f64, b: f64| -> (f64, f64, f64) {
        let (ca, cb) = (a * a / (a * a - b * b), -b * b / (a * a - b * b));
        (ca, cb, ca * a.powf(-0.5) + cb * b.powf(-0.5))
    };
    let mut rejected = 0;
    let mut unconverged = 0;
    // The measurement's solves (`hooks::EffectivityRow`), not the step's.
    #[cfg(all(test, feature = "experimental"))]
    let mut extra = 0u32;
    // A point beyond a plane, per particle (the walls' contact status,
    // Acary's `s_k`).
    let beyond = |x: &[DVec2]| -> Vec<bool> {
        x.iter()
            .map(|q| {
                set.planes
                    .iter()
                    .any(|plane| plane.inward * (q[plane.axis] - plane.at) < 0.0)
            })
            .collect()
    };
    let start_beyond = beyond(&start.x);
    loop {
        let dt = *tau;
        if control.estimator == Estimator::Acary {
            #[cfg(all(test, feature = "experimental"))]
            let returns_before = crate::matter::materials::solid::j2_hencky::returns_seen();
            // A sampled pair's measured error estimate (`estimate`): the
            // whole step's, recorded by its solve (`hooks::want_lobatto`).
            #[cfg(all(test, feature = "experimental"))]
            let sampled = hooks::effectivity_sample();
            #[cfg(all(test, feature = "experimental"))]
            hooks::want_lobatto(sampled.is_some());
            let (whole, whole_reports, _) = run(&start, 1, dt);
            #[cfg(all(test, feature = "experimental"))]
            let lobatto = hooks::take_lobatto();
            #[cfg(all(test, feature = "experimental"))]
            let returns_whole = crate::matter::materials::solid::j2_hencky::returns_seen();
            let (halves, halves_reports, _) = run(&start, 2, dt);
            // The pair's returns, whole then halves (`hooks::add_return_split`).
            #[cfg(all(test, feature = "experimental"))]
            let returns = (
                returns_whole - returns_before,
                crate::matter::materials::solid::j2_hencky::returns_seen() - returns_whole,
            );
            account(&whole_reports, &mut timing);
            account(&halves_reports, &mut timing);
            let difference = combine(1.0, &halves, -1.0, &whole);
            let parts = split(&difference, dt);
            // `EMERGE_PROBE_ERROR_SPLIT`, a measurement:
            // the estimate's energy split between the particles' rigid motion
            // as one body (the only motion below the frame rate for a block
            // whose first elastic mode is far above it) and the rest, its
            // deformation's error included; one line per pair of tries.
            #[cfg(all(test, feature = "experimental"))]
            if error_split() {
                let massive = (0..start.x.len()).filter(|&p| mass[p] > 0.0);
                let total: f64 = massive.clone().map(|p| mass[p]).sum();
                let centre = massive.clone().map(|p| mass[p] * start.x[p]).sum::<DVec2>() / total;
                let w = |p: usize| difference.x[p] / dt;
                let drift = massive.clone().map(|p| mass[p] * w(p)).sum::<DVec2>() / total;
                let arm = |p: usize| start.x[p] - centre;
                let inertia: f64 = massive
                    .clone()
                    .map(|p| mass[p] * arm(p).length_squared())
                    .sum();
                let spin = massive
                    .clone()
                    .map(|p| mass[p] * arm(p).perp_dot(w(p) - drift))
                    .sum::<f64>()
                    / inertia.max(f64::MIN_POSITIVE);
                let rigid: f64 = massive
                    .map(|p| 0.5 * mass[p] * (drift + spin * arm(p).perp()).length_squared())
                    .sum();
                let all = parts.0 * parts.0 + parts.1 * parts.1;
                println!(
                    "error split: dt {dt:.3e}, error {:.3e} of the target, energy {all:.3e} (kinetic {:.3e}, strain {:.3e}), rigid share {:.4e}",
                    norm(&difference, dt) / 3.0 / target.max(f64::MIN_POSITIVE),
                    parts.0 * parts.0,
                    parts.1 * parts.1,
                    rigid / all.max(f64::MIN_POSITIVE)
                );
            }
            // Eqs. 68-70: free at the start, beyond at the end, in either.
            let reaching = [beyond(&whole.x), beyond(&halves.x)]
                .iter()
                .any(|end| end.iter().zip(&start_beyond).any(|(&e, &s)| e && !s));
            // The measurement (`hooks::EffectivityRow`), from the same start
            // and discarded: it decides nothing. Its solves are taken out of
            // the step's count (`extra`).
            #[cfg(all(test, feature = "experimental"))]
            if let (Some(halving), Some((dx, df, dv, speeds))) = (sampled, lobatto) {
                let state_of = |dx: Vec<DVec2>, df: Vec<DMat2>, dv: Vec<DVec2>| State {
                    x: dx,
                    v: dv,
                    f: df,
                    solids: start.solids.clone(),
                    cache: None,
                };
                // The kinetic part from the velocity error instead of the
                // displacement's (Lacoma and Romero's eq. 32).
                let velocity_form = |d: &State, dt: f64| -> f64 {
                    let kinetic: f64 = (0..d.x.len())
                        .map(|p| 0.5 * mass[p] * d.v[p].length_squared())
                        .sum();
                    (kinetic + split(d, dt).1.powi(2)).sqrt()
                };
                let estimate = state_of(dx, df, dv);
                let (reference, _, _) = run(&start, 64, dt);
                extra += 64;
                let whole_off = combine(1.0, &whole, -1.0, &reference);
                let halving_rows = if halving {
                    (0..4)
                        .map(|k| {
                            let h = dt / f64::from(1u32 << k);
                            hooks::want_lobatto(true);
                            let (step, _, _) = run(&start, 1, h);
                            let guess = hooks::take_lobatto()
                                .map_or(f64::NAN, |(dx, df, dv, _)| norm(&state_of(dx, df, dv), h));
                            let (fine, _, _) = run(&start, 64, h);
                            extra += 65;
                            (h, guess, norm(&combine(1.0, &step, -1.0, &fine), h))
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let row = hooks::EffectivityRow {
                    dt,
                    switching: reaching,
                    held: whole_reports.iter().any(|r| r.contacts > 0),
                    speeds,
                    frame: hooks::effectivity_frame(),
                    target,
                    compared: if reaching {
                        parts.1
                    } else {
                        norm(&difference, dt) / 3.0
                    },
                    plastic: whole_reports.iter().map(|r| r.plastic).sum(),
                    estimate: norm(&estimate, dt),
                    estimate_velocity: velocity_form(&estimate, dt),
                    whole_error: norm(&whole_off, dt),
                    whole_error_velocity: velocity_form(&whole_off, dt),
                    doubling: norm(&difference, dt) / 3.0,
                    halves_error: norm(&combine(1.0, &halves, -1.0, &reference), dt),
                    halving: halving_rows,
                };
                println!("  effectivity: {row:?}");
                hooks::record_effectivity(row);
            }
            // Eq. 67 at order 0 (positions only), else eq. 63 at order 2.
            let (error, order) = if reaching {
                (parts.1, 0.0)
            } else {
                (norm(&difference, dt) / 3.0, 2.0)
            };
            let relative = error / (control.safety * target).max(f64::MIN_POSITIVE);
            // Eqs. 64 and 66.
            let ratio = (1.0 / relative.max(f64::MIN_POSITIVE))
                .powf(1.0 / (order + 1.0))
                .clamp(0.5, 5.0);
            let next = (ratio * dt).clamp(floor, control.largest);
            let at_floor = error > target && dt <= floor;
            #[cfg(all(test, feature = "experimental"))]
            if std::env::var_os("EMERGE_PROBE_CONTROL_TRACE").is_some() || hooks::control_trace() {
                println!(
                    "  try dt {dt:.3e}: error {:.3e} of the target (kinetic {:.2e}, elastic {:.2e}), reaching {reaching}, ratio {ratio:.3e}{}",
                    error / target,
                    parts.0 / target,
                    parts.1 / target,
                    if at_floor {
                        ", at the estimator's floor"
                    } else {
                        ""
                    }
                );
            }
            // A try Newton left unconverged is no solution of the step's
            // equations, and the step keeps the energy exactly only as a
            // solution: it is never kept, the step is halved.
            let failed = halves_reports.iter().any(|r| !r.converged && !r.stagnated);
            if failed && dt > floor {
                #[cfg(all(test, feature = "experimental"))]
                hooks::add_return_split(returns.0, returns.1, false);
                #[cfg(all(test, feature = "experimental"))]
                hooks::add_try(reaching, 2);
                keep_factors(&mut start, [&whole, &halves]);
                rejected += 1;
                unconverged += 1;
                *tau = (0.5 * dt).max(floor);
                continue;
            }
            if error <= target || at_floor {
                #[cfg(all(test, feature = "experimental"))]
                hooks::add_return_split(returns.0, returns.1, true);
                #[cfg(all(test, feature = "experimental"))]
                hooks::add_try(reaching, 0);
                x.copy_from_slice(&halves.x);
                solids.copy_from_slice(&halves.solids);
                v.copy_from_slice(&halves.v);
                f.copy_from_slice(&halves.f);
                *cache = halves.cache;
                if let (Some(kept), Some(other)) = (cache.as_mut(), whole.cache.as_ref()) {
                    kept.factors.absorb(&other.factors);
                }
                *tau = next;
                let contact = halves_reports
                    .iter()
                    .any(|r| r.push > 0.0 || r.contacts > 0);
                let stalled = halves_reports.iter().filter(|r| r.stagnated).count() as u32;
                timing.controller = controlling.elapsed().as_secs_f64() - trying.get();
                #[cfg(all(test, feature = "experimental"))]
                let solves = solves - extra;
                return ControlledStep {
                    timing,
                    dt,
                    kept: halves_reports,
                    rejected,
                    solves,
                    error: error / scale,
                    contact: contact || reaching,
                    next,
                    at_floor,
                    error_parts: (parts.0 / scale, parts.1 / scale),
                    unconverged,
                    stagnated: stalled,
                    reaching,
                };
            }
            #[cfg(all(test, feature = "experimental"))]
            hooks::add_return_split(returns.0, returns.1, false);
            #[cfg(all(test, feature = "experimental"))]
            hooks::add_try(reaching, 1);
            keep_factors(&mut start, [&whole, &halves]);
            rejected += 1;
            *tau = next.min(dt).max(floor);
            continue;
        }
        let (first, first_reports, contact_first) = run(&start, control.substeps[0], dt);
        let (second, second_reports, contact_second) = run(&start, control.substeps[1], dt);
        account(&first_reports, &mut timing);
        account(&second_reports, &mut timing);
        // The factors the tries built, for the kept one's cache.
        let spare: Vec<Factors> = [&first, &second]
            .iter()
            .filter_map(|s| s.cache.as_ref().map(|c| c.factors.clone()))
            .collect();
        let (c1, c2, beta) = pair(k1, k2);
        let second_two = combine(c1, &first, c2, &second);
        let contact = contact_first || contact_second;
        let (kept, reports, error, ratio, contact, parts) = if !contact {
            // Eqs. 33 and 42.
            let difference = combine(1.0, &second, -1.0, &second_two);
            let error = norm(&difference, dt);
            let ratio = (control.safety * target / error.max(f64::MIN_POSITIVE)).cbrt();
            let parts = split(&difference, dt);
            (second, second_reports, error, ratio, false, parts)
        } else {
            let (third, third_reports, _) = run(&start, control.substeps[2], dt);
            account(&third_reports, &mut timing);
            // Eqs. 36-38: the next pair's extrapolation and the
            // combination that cancels the `tau^(1/2)` term.
            let (d2, d3, alpha) = pair(k2, k3);
            let third_two = combine(d2, &second, d3, &third);
            let best = combine(
                alpha / (alpha - beta),
                &second_two,
                -beta / (alpha - beta),
                &third_two,
            );
            // Eq. 39.
            let difference = combine(1.0, &third, -1.0, &best);
            let error = norm(&difference, dt);
            // X from the same model: the two extrapolations differ by
            // `(beta - alpha) X tau^(1/2)`. Adaptation: this direct form of
            // their p. 26 estimate.
            let x_size = norm(&combine(1.0, &second_two, -1.0, &third_two), dt)
                / ((beta - alpha).abs() * dt.sqrt()).max(f64::MIN_POSITIVE);
            // Eq. 44 for `r = tau* / tau`, by bisection on `ln r`: its left
            // side grows with `r` from 0.
            let low = x_size * (dt / k3).sqrt();
            let side = |r: f64| error * r.powi(3) + low * (r.sqrt() - r.powi(3)).abs();
            let goal = control.safety * target;
            let ratio = if side(control.growth) <= goal {
                control.growth
            } else {
                let (mut a, mut b) = ((floor / dt).min(1.0).ln(), control.growth.ln());
                for _ in 0..80 {
                    let m = 0.5 * (a + b);
                    if side(m.exp()) > goal { b = m } else { a = m }
                }
                a.exp()
            };
            let parts = split(&difference, dt);
            (third, third_reports, error, ratio, true, parts)
        };
        let next = (ratio.min(control.growth) * dt).clamp(floor, control.largest);
        let at_floor = error > target && dt <= floor;
        #[cfg(all(test, feature = "experimental"))]
        if std::env::var_os("EMERGE_PROBE_CONTROL_TRACE").is_some() || hooks::control_trace() {
            let onset = reports
                .iter()
                .position(|r| r.contacts > 0)
                .map_or("-".to_string(), |k| (k + 1).to_string());
            println!(
                "  try dt {dt:.3e}: error {:.3e} of the target (kinetic {:.2e}, elastic {:.2e}), contact {contact} (first held at substep {onset} of {}), ratio {ratio:.3e}{}",
                error / target,
                parts.0 / target,
                parts.1 / target,
                reports.len(),
                if at_floor {
                    ", at the estimator's floor"
                } else {
                    ""
                }
            );
        }
        // A try Newton left unconverged is no solution (as above).
        if reports.iter().any(|r| !r.converged && !r.stagnated) && dt > floor {
            rejected += 1;
            unconverged += 1;
            *tau = (0.5 * dt).max(floor);
            continue;
        }
        if error <= target || at_floor {
            x.copy_from_slice(&kept.x);
            solids.copy_from_slice(&kept.solids);
            v.copy_from_slice(&kept.v);
            f.copy_from_slice(&kept.f);
            *cache = kept.cache;
            if let Some(kept) = cache.as_mut() {
                for other in &spare {
                    kept.factors.absorb(other);
                }
            }
            *tau = next;
            let stalled = reports.iter().filter(|r| r.stagnated).count() as u32;
            timing.controller = controlling.elapsed().as_secs_f64() - trying.get();
            return ControlledStep {
                timing,
                dt,
                kept: reports,
                rejected,
                solves,
                error: error / scale,
                contact,
                next,
                at_floor,
                error_parts: (parts.0 / scale, parts.1 / scale),
                unconverged,
                stagnated: stalled,
                reaching: false,
            };
        }
        rejected += 1;
        *tau = next.min(0.5 * dt).max(floor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tolerance_energy` does not depend on where the body sits, at any
    /// offset, whole cells or not (a whole scene, walls included, moves by
    /// whole cells only: its walls sit at a whole number of cells, so the
    /// scene-level check, `probe_translated_scene_step_sequence` and the
    /// path's translation test, cannot take a fractional one). Bound
    /// derived: each translated coordinate is off by at most `eps/2 |q|`, a
    /// projection on the unit `g` adds at most `2 eps (|q_x| + |q_y|)`, so
    /// each extreme is off by at most `3 eps (|q_x| + |q_y|)`, the extent by
    /// twice that, the energy by `m |g|` times it.
    #[test]
    fn the_tolerance_energy_does_not_depend_on_where_the_body_sits() {
        let outline: Vec<DVec2> = (0..=16)
            .map(|i| DVec2::new(16.0 + 0.5 * i as f64, 6.0))
            .chain((0..=8).map(|j| DVec2::new(24.0, 6.0 + 0.5 * j as f64)))
            .chain((0..=16).map(|i| DVec2::new(24.0 - 0.5 * i as f64, 10.0)))
            .chain((0..=8).map(|j| DVec2::new(16.0, 10.0 - 0.5 * j as f64)))
            .collect();
        let mass = 54.4;
        for gravity in [DVec2::new(0.0, -9.81), DVec2::new(1.7, -9.66)] {
            let at = |offset: DVec2| -> (f64, f64) {
                let moved: Vec<DVec2> = outline.iter().map(|p| *p + offset).collect();
                let reach = moved
                    .iter()
                    .map(|q| q.x.abs() + q.y.abs())
                    .fold(0.0, f64::max);
                (tolerance_energy(1.5, gravity, [(mass, &moved[..])]), reach)
            };
            let (here, _) = at(DVec2::ZERO);
            assert!(here > 1.5, "gravity's work missing");
            for offset in [
                DVec2::new(0.37, 0.61),
                DVec2::new(100.0, 100.0),
                DVec2::new(123.456, -78.9),
                DVec2::new(1000.1, 2000.3),
            ] {
                let (there, reach) = at(offset);
                let bound = mass * gravity.length() * 12.0 * f64::EPSILON * reach;
                assert!(
                    (there - here).abs() <= bound,
                    "offset {offset}: {there} vs {here} (bound {bound:e})"
                );
            }
        }
    }
}

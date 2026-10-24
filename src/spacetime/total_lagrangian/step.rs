//! The total-Lagrangian step: the grid stays on the reference configuration,
//! with contact against walls and between bodies (`ContactSet`).

use super::*;

/// A body's grid and band factors on its reference configuration, built
/// at its first total Lagrangian step and kept (`step_total_lagrangian`).
#[derive(Clone)]
pub struct LagrangianCache {
    pub(crate) grid: Grid,
    pub(crate) factors: Factors,
    /// The walls' contacts of the last step, the next one's start
    /// (`step_total_lagrangian_contact`).
    pub(crate) contacts: Vec<Contact>,
    /// Per particle, the plane it last touched and the energy the contact
    /// has taken from it since (`ContactMode::Recovered`).
    pub(crate) kept: Vec<(Option<usize>, f64)>,
    /// Per particle, the heat friction has made in it since the caller
    /// last emptied it (`take_heat`), in the step's energy units.
    pub(crate) heat: Vec<f64>,
    /// Per outline, its multiplier functions (`weighted_gap`), from the
    /// reference positions: built once.
    pub(crate) traces: Vec<OutlineTrace>,
}

impl LagrangianCache {
    /// The heat friction made in each particle since the last call, and
    /// starts counting again.
    pub fn take_heat(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.heat)
    }
}

#[cfg(all(test, feature = "experimental"))]
impl LagrangianCache {
    /// The cache of a basis weighted for held walls, the web-splines of
    /// Hollig and Reif, *Nonuniform web-splines*, CAGD 20(5) (2003),
    /// Definition 2: every basis function of lattice `a` is multiplied by
    /// `weight(a, X)` (value and gradient at the reference position), which
    /// vanishes on the walls that hold that component, so the homogeneous
    /// Dirichlet condition holds exactly there (their p.12). No face is
    /// held. Adapted: one weight per displacement component (a roller
    /// holds only the normal one), the extension is this engine's
    /// (`Grid::new`), and the source's normalization `w / w(x_i)` is left
    /// out (it rescales the unknowns, not the space).
    pub(crate) fn weighted(
        layout: &GridLayout,
        reference: &[DVec2],
        (mass, v0): (&[f64], &[f64]),
        occupation: Option<f64>,
        weight: &dyn Fn(usize, DVec2) -> (f64, DVec2),
    ) -> Self {
        let mut grid = Grid::new(layout, reference, (mass, v0), &|_, _, _| None, occupation);
        grid.weights = Some(
            reference
                .iter()
                .map(|&q| [weight(0, q), weight(1, q)])
                .collect(),
        );
        grid.relump(mass);
        // The weights change the basis: no mass matrix assembled before them.
        *grid.mass_matrix.lock().expect("the mass matrix's lock") = None;
        Self {
            grid,
            factors: Factors::default(),
            contacts: Vec::new(),
            kept: Vec::new(),
            heat: Vec::new(),
            traces: Vec::new(),
        }
    }
}

/// The conserving step with the grid kept on the body's reference
/// configuration instead of rebuilt at every step's start, an experiment
/// for the probes, without walls. The source's own algorithm allows it:
/// Love and Sulsky's step 4, returning the mesh to its start, is optional,
/// and "if step 4 is omitted ... the MPM algorithm essentially collapses to
/// a standard (updated) Lagrangian finite element analysis" (TR 2004-002,
/// section 3.2); the total Lagrangian MPM of de Vaucorbeil, Nguyen, Sinaie
/// and Wu (review, Adv. Appl. Mech. 53, 2020, section 2.6) computes the
/// weights and gradients once on the reference positions, and `F` from the
/// displacement's material gradient. Adapted, this engine's: their step is
/// explicit with a lumped mass; here the same midpoint with the discrete
/// gradient stress and the consistent mass, `F_{n+1} = F_n + grad_X u`,
/// whose energy keeping is a derivation to measure, not a source's result.
/// `reference` holds the particles' reference positions; the body's
/// volumes there are `v0`. `held` faces of the reference grid stay at
/// zero, both ways (a Dirichlet condition, not a wall's contact).
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn step_total_lagrangian(
    layout: &GridLayout,
    particles: Particles<'_>,
    reference: &[DVec2],
    solids: &[Elastic],
    (gravity, body, dt): (DVec2, Option<&[DVec2]>, f64),
    (held, settings): (Held<'_>, &ConservingSettings),
    cache: &mut Option<LagrangianCache>,
) -> StepReport {
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
    let mut phases = [clock.elapsed().as_secs_f64(), 0.0, 0.0, 0.0, 0.0];
    let inputs = Inputs {
        x,
        v,
        f,
        mass,
        v0,
        solids,
        gravity,
        dt,
        settings,
        planes: &[],
        penalty: &[],
        lagrangian: true,
        body,
        friction: 0.0,
        responses: false,
    };
    let Solution {
        u, v_n, mut report, ..
    } = solve(grid, &inputs, &[], factors);
    for (total, part) in phases.iter_mut().zip(report.phases).skip(1) {
        *total += part;
    }
    report.passes = 1;
    let finishing = std::time::Instant::now();
    let eps = settings.epsilon;
    let dv: Vec<f64> = (0..grid.len)
        .map(|k| {
            if grid.active[k] {
                2.0 * (u[k] / dt - v_n[k])
            } else {
                0.0
            }
        })
        .collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    report.stabilisation_loss = 0.5
        * (dot(&dv, &grid.mass_times(&dv, mass, eps)) - dot(&dv, &grid.mass_times(&dv, mass, 0.0)));
    for p in 0..n {
        let mut step_x = DVec2::ZERO;
        let mut step_v = DVec2::ZERO;
        let (_, gu) = grid.read(&u, p, 0);
        let (_, gv) = grid.read(&u, p, 1);
        for a in 0..2 {
            step_x[a] = grid.read(&u, p, a).0;
            step_v[a] = grid.read(&dv, p, a).0;
        }
        x[p] += step_x;
        v[p] += step_v;
        f[p] += DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
    }
    phases[4] = finishing.elapsed().as_secs_f64();
    report.timing.grid = phases[0];
    report.timing.finishing = phases[4];
    report.phases = phases;
    report
}

/// The grid `step_total_lagrangian_contact` builds on the reference
/// configuration and keeps in its cache.
pub(crate) fn contact_grid(
    layout: &GridLayout,
    reference: &[DVec2],
    (mass, v0): (&[f64], &[f64]),
    settings: &ConservingSettings,
) -> Grid {
    Grid::new(
        layout,
        reference,
        (mass, v0),
        &|_, _, _| None,
        settings.occupation,
    )
}

/// The velocity field `step_total_lagrangian_contact` starts from, at every
/// point: the points' velocities `v` projected on its grid (`contact_grid`,
/// `grid_velocity`). A point without mass (a body's contact point) has no
/// velocity of its own: its velocity is this field's, the one its step's
/// increments then move.
pub(crate) fn start_velocity_field(
    layout: &GridLayout,
    reference: &[DVec2],
    (mass, v0, v): (&[f64], &[f64], &[DVec2]),
    settings: &ConservingSettings,
) -> Vec<DVec2> {
    let grid = contact_grid(layout, reference, (mass, v0), settings);
    let factor = grid.mass_band(mass, settings.epsilon);
    let (v_n, _) = grid_velocity(&grid, (mass, v), settings.epsilon, &factor);
    (0..reference.len())
        .map(|p| DVec2::new(grid.read(&v_n, p, 0).0, grid.read(&v_n, p, 1).0))
        .collect()
}

/// A contact between two bodies for `step_total_lagrangian_contact`: a
/// particle of one, the facing particle of the other, and the plane whose
/// axis and inward sign measure their separation (`inward (x_p - x_q) >=
/// at`).
pub type Pair = (usize, usize, Plane);

/// How `step_total_lagrangian_contact` holds the contact points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactMode {
    /// Signorini on the gap at the step's end (Doyen, Ern and Piperno eq.
    /// 109; with massless faces, their Discretization 7.1).
    Position,
    /// The contact condition in velocity, on the points already beyond
    /// their plane at the step's start only: `G_n (u_{n+1} - u_n) >= 0`,
    /// `r >= 0`, `r . G_n (u_{n+1} - u_n) = 0` (Doyen, Ern and Piperno,
    /// eqs. 88-92, Discretization 6.1, after Laursen and Chawla). The walls
    /// then do no work: the step keeps the energy (their eq. 93), and a
    /// point may end up to one step's travel beyond its plane.
    Velocity,
    /// `Position`, then the two velocity updates of Puso, Porter and
    /// Slavik, *Structure preserving and energy dissipative contact
    /// approaches for implicit dynamics*, LLNL (2024), sections 4 to 6
    /// (their midstep-PC): the persistency update `M (v~ - v) + G^T l = 0`,
    /// `G v~ = 0` on the points in contact or just released (their eqs.
    /// 29-30, strictly dissipative, eq. 32), each point keeping the energy
    /// the walls and this update took from it (eqs. 26-27, 35, 40); and,
    /// for the points released, that energy returned as `M (v^ - v~) =
    /// alpha f`, `f = sum_C G_C^T kappa_C`, `alpha^2 = 2 sum kappa / f . M^-1
    /// f` (eqs. 41, 51-58). Needs the mass invertible: the standard mass,
    /// no massless faces. Adapted, this engine's: their contact is mortar
    /// with weights `G_C / int phi_C`; here collocation points, `f` without
    /// that division, and the plane's gap exact (`g^ = g` on a flat rigid
    /// wall, their eq. 22).
    Recovered,
}

/// The total Lagrangian step (`step_total_lagrangian`) against rigid walls
/// in contact, for the probes: the body's contact points may touch the
/// walls' `planes`, never cross them and never pull on them, held as
/// `mode` says (`ContactMode`; by default the Signorini conditions at the
/// step's end, `g >= 0, lambda >= 0, lambda g = 0`, Doyen, Ern and Piperno,
/// *Time-integration schemes for the finite element dynamic Signorini
/// problem*, SIAM J. Sci. Comput. 33 (2011), eqs. 98 and 109). They are
/// enforced as `step` enforces them, by its augmented Lagrangian with
/// Uzawa's multiplier updates until no point ends beyond its plane by more
/// than `ConservingSettings::contact_tolerance`, the multipliers of a step being the next
/// one's start. The contact points are particles of no mass and no volume
/// on the body's boundary: they add nothing to the mass or the internal
/// force, only to the walls' push.
///
/// With massless faces (`hooks::set_massless`), the contact boundary's
/// normal unknowns carry no inertia: the massless contact boundary of
/// Khenous, Laborde and Renard, *Mass redistribution method for finite
/// element contact problems in elastodynamics*, Eur. J. Mech. A/Solids 27
/// (2008), as Doyen, Ern and Piperno's section 7 builds it, their rows and
/// columns of the mass zero and the rest the standard mass (their eq.
/// 97). Adapted, this engine's: their unknowns are nodal finite element
/// values and the contact set the boundary's nodes; here the probe names
/// the B-spline coefficients of the normal component, and the conditions
/// hold at the contact points.
///
/// `penalty` is the augmented Lagrangian's per particle, an iteration
/// parameter: the converged multipliers do not depend on it. The report's
/// `work.0` is the walls' work over the step, `lambda` times the contact
/// points' normal displacement, the contact term of Doyen, Ern and
/// Piperno's energy balance (eq. 112): `-lambda_{n+1} g_n` once the end
/// gap is closed, never positive; `work.1` the kinetic energy the velocity
/// updates of `ContactMode::Recovered` changed.
pub fn step_total_lagrangian_contact(
    layout: &GridLayout,
    particles: Particles<'_>,
    reference: &[DVec2],
    solids: &[Elastic],
    (dt, mode, gravity): (f64, ContactMode, DVec2),
    (set, penalty, settings): (ContactSet<'_>, &[f64], &ConservingSettings),
    cache: &mut Option<LagrangianCache>,
) -> StepReport {
    let ContactSet {
        planes,
        pairs,
        outlines,
        corners,
        friction,
    } = set;
    assert!(
        friction == 0.0 || mode == ContactMode::Velocity,
        "friction is held in velocity only"
    );
    let Particles { x, v, f, mass, v0 } = particles;
    let n = x.len();
    let clock = std::time::Instant::now();
    let LagrangianCache {
        grid,
        factors,
        contacts,
        kept,
        heat,
        traces,
    } = cache.get_or_insert_with(|| LagrangianCache {
        grid: contact_grid(layout, reference, (mass, v0), settings),
        factors: Factors::default(),
        contacts: Vec::new(),
        kept: Vec::new(),
        heat: Vec::new(),
        traces: Vec::new(),
    });
    let grid = &*grid;
    // The multipliers' (Uzawa) penalty, an iteration parameter: the
    // caller's stiffness plus the step's inertia at the point, `2 m / dt^2`
    // with `m` the grid's lumped mass it reaches, so each update closes
    // most of the gap left whatever the step's length. With the stiffness
    // alone a step 1e5 times shorter than the frame (the step controller's
    // at a bar's impact) left Uzawa at its 400-pass cap and 0.9 percent of
    // the energy in the contact's work (measured). The converged contact
    // does not depend on it.
    let scaled: Vec<f64> = (0..n)
        .map(|p| {
            if penalty[p] > 0.0 {
                let mut reached = 0.0;
                grid.visit(p, 0, |k, w, _| reached += w * grid.lumped[k]);
                penalty[p] + 2.0 * reached / (dt * dt)
            } else {
                0.0
            }
        })
        .collect();
    let penalty = &scaled[..];
    if kept.len() != n {
        *kept = vec![(None, 0.0); n];
    }
    let mut phases = [clock.elapsed().as_secs_f64(), 0.0, 0.0, 0.0, 0.0];
    // In velocity, each point beyond a plane at the step's start gets a
    // plane of its own at its start position, and only it; each pair whose
    // points have crossed (`inward (x_p - x_q - at) < 0`) a plane at their
    // start separation, so the constraint holds their relative normal
    // displacement over the step. Pairs are held in velocity only.
    let velocity = mode == ContactMode::Velocity;
    assert!(
        velocity || (pairs.is_empty() && outlines.is_empty()),
        "pairs and outlines are held in velocity only"
    );
    // A point is held from the step in which it reaches the plane:
    // beyond it already or, with `forecast_activation`, closing on it
    // within the step in free flight, `g + dt g' <= 0`. Source: Acary,
    // INRIA RR-7151 (2010), eqs. 16e and 18 (the Moreau-Jean scheme's
    // forecast gap, gamma = 1) and Prop. 3 (an impact not forecast in
    // its own step leaves an error the step's length cannot reduce).
    // Held at its start position, a forecast point stands off by its
    // start gap instead of crossing by `v dt`.
    let reaches =
        |gap: f64, rate: f64| gap < 0.0 || (settings.forecast_activation && gap + dt * rate <= 0.0);
    let mut own: Vec<Plane> = Vec::new();
    // The conditions that may join the set in velocity, each holding its
    // points' normal displacement over the step.
    let mut candidates: Vec<Contact> = Vec::new();
    if velocity {
        contacts.clear();
        // An outline's points are held through its weighted gaps, one per
        // independent multiplier function (`weighted_gap`): a condition per
        // contact point held more points than the grid has independent
        // displacements along a body's edge (two per cell), their gap
        // gradients dependent (measured: rank 28 of 41 rows at two blocks'
        // impact, the dependent rows all floor points), so the multipliers
        // were not unique and their semismooth Newton cycled (period 4,
        // 400 passes) where Uzawa crept.
        if traces.len() != outlines.len() {
            *traces = outlines
                .iter()
                .enumerate()
                .map(|(o, outline)| {
                    let at = corners.get(o).map_or(&[][..], |c| &c[..]);
                    outline_trace(reference, outline, at, grid.cell)
                })
                .collect();
        }
        candidates = reaching_conditions(traces, outlines, (x, v), planes, &mut own, &reaches);
        let mut on_outline = vec![false; n];
        for &q in outlines.iter().flatten() {
            on_outline[q] = true;
        }
        for (index, plane) in planes.iter().enumerate() {
            for (p, q) in x.iter().enumerate() {
                // Only the contact points (a penalty of their own) are held:
                // a point without one can take no push, and held it froze
                // Uzawa at its cap with its gap open (measured: -0.09 of a
                // cell, the same 400 passes on a sliding block).
                if penalty[p] <= 0.0 || on_outline[p] {
                    continue;
                }
                let gap = plane.inward * (q[plane.axis] - plane.at);
                if reaches(gap, plane.inward * v[p][plane.axis]) {
                    own.push(Plane {
                        at: q[plane.axis],
                        ..*plane
                    });
                    candidates.push(Contact::at(
                        ContactPoint {
                            particle: p,
                            plane: own.len() - 1,
                            partner: None,
                            segment: None,
                            nodal: None,
                        },
                        (0, p, index),
                    ));
                }
            }
        }
        for &(p, q, plane) in pairs {
            let separation = x[p][plane.axis] - x[q][plane.axis];
            let rate = plane.inward * (v[p][plane.axis] - v[q][plane.axis]);
            if reaches(plane.inward * (separation - plane.at), rate) {
                own.push(Plane {
                    at: separation,
                    ..plane
                });
                candidates.push(Contact::at(
                    ContactPoint {
                        particle: p,
                        plane: own.len() - 1,
                        partner: Some(q),
                        segment: None,
                        nodal: None,
                    },
                    (1, p, q),
                ));
            }
        }
    }
    let used: &[Plane] = if velocity { &own } else { planes };
    let mut inputs = Inputs {
        x,
        v,
        f,
        mass,
        v0,
        solids,
        gravity,
        dt,
        settings,
        planes: used,
        penalty,
        lagrangian: true,
        body: None,
        friction,
        responses: false,
    };
    let mut most = contacts.len();
    let mut passes = 0;
    let mut newton_steps = 0u32;
    let mut newton_allowed = true;
    let mut fallback: Option<Vec<(f64, f64)>> = None;
    // The passes' time by part (`Timing`): the solves' own, the
    // multipliers' Newton steps, the rest the contact passes'.
    let mut timing = Timing::default();
    let mut solve_seconds = 0.0;
    let passing = std::time::Instant::now();
    let Solution {
        u,
        v_n,
        mut report,
        dissipated,
        ..
    } = loop {
        let solving = std::time::Instant::now();
        let solution = solve(grid, &inputs, contacts, factors);
        solve_seconds += solving.elapsed().as_secs_f64();
        timing.add(&solution.report.timing);
        for (total, part) in phases.iter_mut().zip(solution.report.phases).skip(1) {
            *total += part;
        }
        // Each contact's own tolerance on what it leaves
        // (`contact_allowance`).
        let allowed = |c: &Contact, shares: &[Share]| -> f64 {
            contact_allowance(
                grid,
                (x, v),
                used,
                &solution.u,
                shares,
                (
                    settings.contact_tolerance,
                    dt,
                    gap_resolution(c, penalty, solution.report.residual_allowed),
                ),
            )
        };
        // A point joins the set only when it ends beyond its plane by more
        // than the tolerance the set is judged settled by: within it, it
        // is as settled as a held point. Joining at any `g < 0` let a free
        // point at a round-off gap (measured: -4.7e-17 of a cell) join,
        // get no push, leave at `g > 0`, and join again, the set cycling
        // to Uzawa's cap on a sliding block.
        let beyond_tolerance = |c: &Contact| -> bool {
            let (g, shares) = contact_geometry(grid, x, used, &solution.u, c);
            -g > allowed(c, &shares)
        };
        let mut moved = false;
        let mut deepest = 0.0f64;
        let before: Vec<(f64, f64)> = contacts
            .iter()
            .map(|c| (c.multiplier, c.tangential))
            .collect();
        contacts.retain_mut(|c| {
            let (g, shares) = contact_geometry(grid, x, used, &solution.u, c);
            let scale = allowed(c, &shares);
            deepest = deepest.max(-g / scale);
            let g_t = if friction > 0.0 {
                contact_tangential_gap(grid, &solution.u, &shares)
            } else {
                0.0
            };
            let k = contact_penalty(c, penalty);
            let (size, along) =
                contact_force((c.multiplier, c.tangential), (g, g_t), (k, friction), None);
            // A sticking contact must not slide (Hueber, Stadler and
            // Wohlmuth eq. 3.5a), to the same tolerance as the gap.
            if friction > 0.0 && along.abs() < friction * size {
                deepest = deepest.max(g_t.abs() / scale);
            }
            // And the multipliers the solve used are their own projection,
            // to the same tolerance: the complementarity functions' residual
            // (Hueber, Stadler and Wohlmuth eqs. 4.1 and 7.1), `lambda -
            // proj(lambda - k g)` over `k`. Gaps within tolerance alone let a
            // step end on multipliers a Newton step had just set, a slipping
            // contact's along its slip (measured: the energy rising 1e-2 of
            // the weight scale on a sliding block).
            if k > 0.0 {
                deepest = deepest
                    .max((size - c.multiplier).abs() / (k * scale))
                    .max((along - c.tangential).abs() / (k * scale));
            }
            c.multiplier = size;
            c.tangential = along;
            // A released contact (no push, gap open) stays in the set
            // until the step ends: it takes no push while released
            // (`max(0, lambda - k g)`), and dropped then joined again, one
            // point flickering in and out every pass kept the set from ever
            // standing still (measured: 16 and 17 contacts alternating on a
            // sliding block), so Uzawa never settled and the multipliers'
            // Newton step never ran.
            true
        });
        // In velocity, the conditions found reaching at the step's start;
        // otherwise every particle against every plane, at a point.
        let everywhere: Vec<Contact> = if velocity {
            Vec::new()
        } else {
            (0..used.len())
                .flat_map(|i| {
                    (0..n).map(move |p| {
                        Contact::at(
                            ContactPoint {
                                particle: p,
                                plane: i,
                                partner: None,
                                segment: None,
                                nodal: None,
                            },
                            (0, p, i),
                        )
                    })
                })
                .collect()
        };
        for candidate in candidates.iter().chain(&everywhere) {
            if beyond_tolerance(candidate) && !contacts.iter().any(|c| c.key == candidate.key) {
                contacts.push(candidate.clone());
                moved = true;
            }
        }
        most = most.max(contacts.len());
        passes += 1;
        // A Newton step on the multipliers is kept only if the solve it
        // led to converged: otherwise its multipliers are replaced by those
        // Uzawa's update would have given, and the step goes on with Uzawa
        // alone. Requiring the deepest violation not to grow as well
        // disabled Newton on steps it then solved (a semismooth Newton
        // step is not monotone in the largest violation), leaving Uzawa at
        // its slow mode (measured: the collision's first frame, 4 tries
        // rejected at Uzawa's cap). Without
        // this, a Newton step taken while two bodies' segments touched sent
        // the next solve to Newton's cap (30 iterations, 61200 GMRES
        // iterations) with the deepest gap at 1e13 times its tolerance,
        // try after try (measured, a collision). A safeguard of this
        // engine's, declared: Hueber, Stadler and Wohlmuth report needing
        // none (sec. 4), on linear elasticity against flat contact.
        if let Some(uzawa) = fallback.take()
            && !solution.report.converged
            && !solution.report.stagnated
        {
            for (c, &(multiplier, tangential)) in contacts.iter_mut().zip(&uzawa) {
                c.multiplier = multiplier;
                c.tangential = tangential;
            }
            newton_allowed = false;
            inputs.responses = false;
            if passes < 400 {
                continue;
            }
        }
        let settled = !moved && deepest <= 1.0;
        // The set unchanged over this pass (none released, none joined),
        // the multipliers take a semismooth Newton step instead of
        // Uzawa's (`newton_on_multipliers`); the next solve then returns
        // the responses that step needs.
        let uzawa: Vec<(f64, f64)> = contacts
            .iter()
            .map(|c| (c.multiplier, c.tangential))
            .collect();
        if !settled && !moved && newton_allowed && passes < 400 && {
            let stepping = std::time::Instant::now();
            let stepped =
                newton_on_multipliers(grid, (x, used, friction), &solution, contacts, &before);
            timing.multipliers += stepping.elapsed().as_secs_f64();
            stepped
        } {
            newton_steps += 1;
            fallback = Some(uzawa);
        }
        inputs.responses = !moved;
        // A solve that stagnated on its residual's noise (`StepReport::
        // stagnated`) does not end the passes: the contacts still settle on
        // it. Ending there left them unsettled and the step's energy off by
        // 1e-7 of the start's per such step (measured, a slab at rest).
        let failed = !solution.report.converged && !solution.report.stagnated;
        if settled || passes == 400 || failed {
            // Stopped at the cap with a contact still beyond its
            // tolerance, the step solves its contact conditions no more
            // than an unconverged Newton solves its equations: reported
            // unconverged, so the step controller rejects it. Kept, one
            // such step at a block's impact did 0.23 of the weight times a
            // cell of contact work (measured).
            let mut solution = solution;
            solution.report.converged &= settled;
            solution.report.stagnated &= settled;
            // The released contacts leave the set with the step.
            contacts.retain(|c| {
                c.multiplier > 0.0 || contact_geometry(grid, x, used, &solution.u, c).0 <= 0.0
            });
            break solution;
        }
    };
    timing.contact = passing.elapsed().as_secs_f64() - solve_seconds - timing.multipliers;
    report.contacts = most;
    report.passes = passes;
    // Each condition ended within its tolerance, `|g_v| <= a_v` (the gap
    // past it, or the complementarity residual `k |g_v|` past `k a_v`,
    // kept the passes going), so its work `lambda_v g_v` is at most
    // `lambda_v a_v` (`StepReport::contact_work_bound`).
    report.contact_work_bound = contacts
        .iter()
        .map(|c| {
            let (_, shares) = contact_geometry(grid, x, used, &u, c);
            c.multiplier.abs()
                * contact_allowance(
                    grid,
                    (x, v),
                    used,
                    &u,
                    &shares,
                    (
                        settings.contact_tolerance,
                        dt,
                        gap_resolution(c, penalty, report.residual_allowed),
                    ),
                )
        })
        .sum();
    report.multiplier_steps = newton_steps;
    report.push = contacts.iter().map(|c| c.multiplier).sum();
    // The friction's work, `sum lambda_T g_T`, never positive (Acary and
    // Collins-Craft eq. 73): the mechanical energy friction turns into
    // heat, reported as the heat.
    report.friction = -contacts
        .iter()
        .map(|c| {
            let (_, shares) = contact_geometry(grid, x, used, &u, c);
            c.tangential * contact_friction_motion(grid, x, &u, &shares)
        })
        .sum::<f64>();
    // Where that heat goes: each contact point's share to the faces of
    // lattice 0 the point reaches, then to the particles in proportion to the mass each
    // gives a face, `m_q w_kq / m_k`, `m_k = sum_q m_q w_kq` (the explicit
    // path's carrier, `transfer::friction_heat`). On the plain B-spline
    // weights, never the basis's extrapolated ones: those can be negative
    // and would cool a particle (measured, -0.029 against +0.41 in a
    // frame's friction heat once the surface faces turned degenerate).
    // Adaptation: a contact's heat is shared over the faces that hold
    // mass, its weights renormalised on them (a face no particle reaches
    // has no material to heat). The heat goes to the body of the point in
    // contact; a point against another body's segment heats its own body,
    // every outline point being in turn the point against the other's
    // segments.
    if friction > 0.0 && report.friction != 0.0 {
        let faces = {
            let l = grid.lattices[1];
            l.start + l.ni * l.nj
        };
        let mut held = vec![0.0; faces];
        for (q, &m) in mass.iter().enumerate() {
            if m > 0.0 {
                grid.visit_stacked(q, 0, |k, w, _| held[k] += w * m);
            }
        }
        let mut nodes = vec![0.0; faces];
        for c in contacts.iter() {
            let (_, shares) = contact_geometry(grid, x, used, &u, c);
            for share in &shares {
                let point = share.point;
                let made = -c.tangential
                    * share.weight
                    * friction_motion(grid, x, &u, &point, share.normal, &share.spread);
                let mut reach = 0.0;
                grid.visit_stacked(point.particle, 0, |k, w, _| {
                    if held[k] > 0.0 {
                        reach += w;
                    }
                });
                if reach > 0.0 {
                    grid.visit_stacked(point.particle, 0, |k, w, _| {
                        if held[k] > 0.0 {
                            nodes[k] += w / reach * made;
                        }
                    });
                }
            }
        }
        heat.resize(n, 0.0);
        for (q, made) in heat.iter_mut().enumerate() {
            if mass[q] > 0.0 {
                grid.visit_stacked(q, 0, |k, w, _| {
                    if held[k] > 0.0 {
                        *made += mass[q] * w / held[k] * nodes[k];
                    }
                });
            }
        }
    }
    // The dashpots' dissipated energy, each particle's own, into the same
    // heat as friction's.
    if !dissipated.is_empty() {
        heat.resize(n, 0.0);
        for (made, &lost) in heat.iter_mut().zip(&dissipated) {
            *made += lost;
        }
        report.viscous = dissipated.iter().sum();
    }
    let finishing = std::time::Instant::now();
    // A massless unknown has no velocity of its own (its displacement
    // follows the others', Doyen, Ern and Piperno eq. 103): the particles'
    // velocities are the massive unknowns' alone, so their kinetic energy
    // is `1/2 v M* v`.
    let massive = grid.massive();
    let mut end_velocity: Vec<f64> = (0..grid.len)
        .map(|k| {
            if massive[k] {
                2.0 * u[k] / dt - v_n[k]
            } else {
                0.0
            }
        })
        .collect();
    let mut updates = 0.0;
    if mode == ContactMode::Recovered {
        let band = factors.mass.as_ref().expect("built by the solve");
        let inverse = |y: &[f64]| -> Vec<f64> {
            band.solve(y)
                .into_iter()
                .zip(&massive)
                .map(|(e, &on)| if on { e } else { 0.0 })
                .collect()
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let kinetic = |w: &[f64]| 0.5 * dot(w, &grid.mass_times(w, mass, settings.epsilon));
        let before = kinetic(&end_velocity);
        // The walls' work at each point, kept at the point (eq. 26).
        for c in contacts.iter() {
            let point = c.single();
            let plane = used[point.plane];
            kept[point.particle].0 = Some(point.plane);
            kept[point.particle].1 -=
                c.multiplier * plane.inward * grid.read(&u, point.particle, plane.axis).0;
        }
        // The points in contact now, and those just released.
        let now: Vec<bool> = (0..n)
            .map(|p| {
                contacts
                    .iter()
                    .any(|c| c.single().particle == p && c.multiplier > 0.0)
            })
            .collect();
        let set: Vec<usize> = (0..n).filter(|&p| now[p] || kept[p].0.is_some()).collect();
        let row = |p: usize| -> Vec<f64> {
            let plane = used[kept[p].0.unwrap_or(0)];
            let mut g = vec![0.0; grid.len];
            grid.visit(p, plane.axis, |k, w, _| g[k] += plane.inward * w);
            g
        };
        let rows: Vec<Vec<f64>> = set.iter().map(|&p| row(p)).collect();
        if !rows.is_empty() {
            // Persistency: `v~ = v - M^-1 G^T l`, `G v~ = 0` (eqs. 29-30).
            let columns: Vec<Vec<f64>> = rows.iter().map(|g| inverse(g)).collect();
            let size = rows.len();
            let mut system: Vec<Vec<f64>> = rows
                .iter()
                .map(|g| columns.iter().map(|z| dot(g, z)).collect())
                .collect();
            let rates: Vec<f64> = rows.iter().map(|g| dot(g, &end_velocity)).collect();
            let mut impulse = rates.clone();
            for i in 0..size {
                let pivot = system[i][i];
                let pivot_row = system[i].clone();
                for r in i + 1..size {
                    let factor = system[r][i] / pivot;
                    for (e, above) in system[r][i..].iter_mut().zip(&pivot_row[i..]) {
                        *e -= factor * above;
                    }
                    impulse[r] -= factor * impulse[i];
                }
            }
            for i in (0..size).rev() {
                let tail: f64 = (i + 1..size).map(|c| system[i][c] * impulse[c]).sum();
                impulse[i] = (impulse[i] - tail) / system[i][i];
            }
            for (z, &l) in columns.iter().zip(&impulse) {
                end_velocity
                    .iter_mut()
                    .zip(z)
                    .for_each(|(e, zk)| *e -= l * zk);
            }
            // Each point's share of the dissipation, `1/2 l (G v)` (eq. 35
            // at the average velocity, eq. 32).
            for ((&p, &l), &rate) in set.iter().zip(&impulse).zip(&rates) {
                kept[p].1 += 0.5 * l * rate;
            }
            // The released points get their energy back (eqs. 41, 51-58).
            let released: Vec<usize> = (0..size).filter(|&i| !now[set[i]]).collect();
            let energy: f64 = released.iter().map(|&i| kept[set[i]].1.max(0.0)).sum();
            if energy > 0.0 {
                let mut force = vec![0.0; grid.len];
                for &i in &released {
                    let share = kept[set[i]].1.max(0.0);
                    force
                        .iter_mut()
                        .zip(&rows[i])
                        .for_each(|(e, g)| *e += share * g);
                }
                let z = inverse(&force);
                let alpha = (2.0 * energy / dot(&force, &z)).sqrt();
                end_velocity
                    .iter_mut()
                    .zip(&z)
                    .for_each(|(e, zk)| *e += alpha * zk);
            }
            for &i in &released {
                kept[set[i]] = (None, 0.0);
            }
        }
        updates = kinetic(&end_velocity) - before;
    }
    report.work = (
        contacts
            .iter()
            .map(|c| {
                c.multiplier
                    * c.points
                        .iter()
                        .map(|(point, weight)| {
                            weight
                                * match point.segment {
                                    // The segment's gap is already the
                                    // relative normal displacement over the
                                    // step.
                                    Some(_) => point_geometry(grid, x, used, &u, point).0,
                                    None => {
                                        let plane = used[point.plane];
                                        let relative = grid.read(&u, point.particle, plane.axis).0
                                            - point
                                                .partner
                                                .map_or(0.0, |q| grid.read(&u, q, plane.axis).0);
                                        plane.inward * relative
                                    }
                                }
                        })
                        .sum::<f64>()
            })
            .sum::<f64>()
            - report.friction,
        updates,
    );
    let dv: Vec<f64> = (0..grid.len)
        .map(|k| {
            if massive[k] {
                end_velocity[k] - v_n[k]
            } else {
                0.0
            }
        })
        .collect();
    #[cfg(all(test, feature = "experimental"))]
    if hooks::torque_on() {
        let middle =
            |p: usize| x[p] + 0.5 * DVec2::new(grid.read(&u, p, 0).0, grid.read(&u, p, 1).0);
        let moment = |w: &[f64]| -> f64 {
            (0..n)
                .map(|p| {
                    let change = DVec2::new(grid.read(w, p, 0).0, grid.read(w, p, 1).0);
                    mass[p] * middle(p).perp_dot(change)
                })
                .sum()
        };
        let mut push = vec![0.0; grid.len];
        for c in contacts.iter() {
            let (_, shares) = contact_geometry(grid, x, used, &u, c);
            for share in &shares {
                let (normal, spread) = (share.normal, share.spread);
                let total = share.weight * (c.multiplier * normal + c.tangential * normal.perp());
                for (r, weight) in spread {
                    for a in 0..2 {
                        let force = weight * total[a];
                        grid.visit(r, a, |k, w, _| push[k] += w * force);
                    }
                }
                if let Some((first, second, couple)) =
                    friction_couple(grid, x, &u, &share.point, normal, &spread)
                {
                    for (r, sign) in [(second, 1.0), (first, -1.0)] {
                        for a in 0..2 {
                            let force = sign * share.weight * couple * c.tangential * normal[a];
                            grid.visit(r, a, |k, w, _| push[k] += w * force);
                        }
                    }
                }
            }
        }
        for (value, &on) in push.iter_mut().zip(&massive) {
            if !on {
                *value = 0.0;
            }
        }
        let band = factors.mass.as_ref().expect("built by the solve");
        let from_push: Vec<f64> = preconditioned_cg(
            &|y: &[f64]| grid.mass_times(y, mass, settings.epsilon),
            &push,
            &|r: &[f64]| band.solve(r),
            &massive,
        )
        .into_iter()
        .map(|e| dt * e)
        .collect();
        hooks::set_torque((moment(&dv), moment(&from_push)));
    }
    for p in 0..n {
        let mut step_x = DVec2::ZERO;
        let mut step_v = DVec2::ZERO;
        let (_, gu) = grid.read(&u, p, 0);
        let (_, gv) = grid.read(&u, p, 1);
        for a in 0..2 {
            step_x[a] = grid.read(&u, p, a).0;
            step_v[a] = grid.read(&dv, p, a).0;
        }
        x[p] += step_x;
        v[p] += step_v;
        f[p] += DMat2::from_cols(DVec2::new(gu.x, gv.x), DVec2::new(gu.y, gv.y));
    }
    phases[4] = finishing.elapsed().as_secs_f64();
    timing.grid = phases[0];
    timing.finishing = phases[4];
    report.timing = timing;
    report.phases = phases;
    report
}

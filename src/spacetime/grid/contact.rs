use std::collections::HashMap;

use glam::{IVec2, Vec2};

use super::contact_normal::fit_contact_normal_lr;
use super::directional_grip::DirectionalContactGrip;
use super::{FxU32BuildHasher, Grid, flat_index};

/// Second velocity field for multi-field frictional contact (Bardenhagen, Guilkey,
/// Roessig, Brackbill 2001) — see `Particle::contact_group`'s doc for the full
/// rationale. Only allocated at grid nodes touched by at least one particle with
/// `contact_group != 0` ("grip"); the rest of the grid never sees this at all.
///
/// `grip_mass`/`grip_momentum` accumulate during P2G exactly like `Cell`'s own
/// fields, but only from grip particles. `resolved_grip_v`/`resolved_rest_v` are
/// filled in by `Grid::resolve_contact` (after the main `update_velocities` +
/// gravity pass) and are what G2P actually reads for grip/non-grip particles
/// respectively, at nodes where this cell exists.
///
/// `points`: labeled particle positions (`+1.0` grip, `-1.0` rest) whose kernel
/// touches this node — the "point cloud" the logistic-regression contact normal
/// (`fit_contact_normal_lr`) fits a separating plane through. Populated by a
/// second particle pass (`gather_contact_point_cloud`, gated on contact activity)
/// after the ordinary P2G scatter has determined which nodes are contact-active.
#[derive(Clone, Debug, Default)]
pub(super) struct ContactCell {
    grip_mass: f32,
    grip_momentum: Vec2,
    resolved_grip_v: Vec2,
    resolved_rest_v: Vec2,
    points: Vec<(Vec2, f32)>,
}

pub(super) type ContactCellMap = HashMap<u32, ContactCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate mass and momentum for the "grip" contact field (particles with
    /// `contact_group != 0`) during P2G, additively alongside the normal
    /// `add_mass_momentum` call for the SAME particle — this is a second, separate
    /// accumulator, not a replacement. OOB silently ignored.
    pub fn add_grip_mass_momentum(&mut self, cell_pos: IVec2, mass: f32, momentum: Vec2) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        match self.contact_cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                cell.grip_mass += mass;
                cell.grip_momentum += momentum;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.contact_dirty.push(idx);
                e.insert(ContactCell {
                    grip_mass: mass,
                    grip_momentum: momentum,
                    resolved_grip_v: Vec2::ZERO,
                    resolved_rest_v: Vec2::ZERO,
                    points: Vec::new(),
                });
            }
        }
    }

    /// Appends one labeled particle position (`+1.0` grip / `-1.0` rest) to
    /// `cell_pos`'s contact point cloud, for the logistic-regression normal fit
    /// (`fit_contact_normal_lr`). Only pushes into a cell that ALREADY exists in
    /// `contact_cells` (i.e. one at least one grip particle already touched via
    /// `add_grip_mass_momentum` this substep) — never creates a new entry, so a
    /// rest particle far from any grip body cannot spuriously grow `contact_dirty`.
    /// Called from a second particle pass (`gather_contact_point_cloud`) run
    /// AFTER the main P2G scatter has fully determined which nodes are
    /// contact-active, so this is deliberately not merged into
    /// `scatter_particles_to_grid` itself. OOB silently ignored.
    pub fn add_contact_point(&mut self, cell_pos: IVec2, position: Vec2, label: f32) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        if let Some(cell) = self.contact_cells.get_mut(&idx) {
            cell.points.push((position, label));
        }
    }

    /// Resolved "grip" field velocity at `cell_pos` — valid after `resolve_contact()`.
    /// Falls back to the ordinary total velocity when no contact was ever registered
    /// at this node (e.g. a grip particle whose kernel briefly touches a cell that no
    /// OTHER grip particle reaches, so there's no real second field to speak of).
    pub fn grip_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.contact_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_grip_v)
    }

    /// Resolved "rest" (contact_group == 0) field velocity at `cell_pos` — valid after
    /// `resolve_contact()`. Falls back to the ordinary total velocity when no contact
    /// was registered at this node, which is the common case away from any grip body —
    /// this is what makes routing G2P through this function safe everywhere, not just
    /// near contact.
    pub fn rest_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.contact_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_rest_v)
    }

    /// Grip-field mass at `cell_pos`, 0.0 if OOB or untouched. Used only by
    /// `grip_mass_gradient_normal` below — a tiny, deliberately local helper, not a
    /// public query (there's no meaningful "grip mass" outside contact resolution).
    fn grip_mass_at(&self, cell_pos: IVec2) -> f32 {
        flat_index(cell_pos, self.resolution)
            .and_then(|idx| self.contact_cells.get(&idx))
            .map_or(0.0, |c| c.grip_mass)
    }

    /// Fallback contact normal: Sobel-3x3 gradient of the grip field's own grid mass —
    /// the ORIGINAL Bardenhagen 2001 method, kept as a fallback for
    /// `fit_contact_normal_lr`'s "no confident plane" case (see `resolve_contact`'s
    /// call site doc for why zero correction there was a real bug). Not the primary
    /// method any more precisely because it has known weaknesses near a translating
    /// body or a material corner -- but exactly the shallow, one-sided point clouds
    /// where LR fails tend to be close to a flat interface, the case this handles
    /// best. Returns `None` when there's no real local gradient (deep inside a
    /// well-mixed interior, matching the old code's own "no gradient" case).
    fn grip_mass_gradient_normal(&self, idx: u32) -> Option<Vec2> {
        let x = (idx as usize / self.resolution) as i32;
        let y = (idx as usize % self.resolution) as i32;
        let m = |dx: i32, dy: i32| self.grip_mass_at(IVec2::new(x + dx, y + dy));
        let grad_x = (m(1, -1) + 2.0 * m(1, 0) + m(1, 1)) - (m(-1, -1) + 2.0 * m(-1, 0) + m(-1, 1));
        let grad_y = (m(-1, 1) + 2.0 * m(0, 1) + m(1, 1)) - (m(-1, -1) + 2.0 * m(0, -1) + m(1, -1));
        let gradient = Vec2::new(grad_x, grad_y);
        (gradient.length_squared() > f32::EPSILON).then(|| gradient.normalize())
    }

    /// Multi-field frictional contact resolution (Bardenhagen, Guilkey, Roessig,
    /// Brackbill 2001, "An Improved Contact Algorithm for the Material Point Method").
    /// Real equations from the primary source (verified against the actual paper text,
    /// not a secondary description — see project memory
    /// `locomotion_core_frictional_contact_2026-07-11` for the full derivation):
    ///
    /// - Per-field velocity `v_grip = p_grip/m_grip` (eq. 4); center-of-mass velocity
    ///   `v_cm` is just this grid's own existing total field (eq. 5-6) — already computed
    ///   by `update_velocities`, called right before this.
    /// - Surface normal `n`: fitted via logistic regression through a labeled particle
    ///   point cloud (`fit_contact_normal_lr`), not a grid mass gradient — see that
    ///   function's doc for why (a real, found-and-fixed bug in the original approach).
    /// - Approach test (eq. 8): contact applies only when `(v_grip - v_cm)·n < 0`
    ///   (bodies approaching); otherwise free separation — the two fields simply keep
    ///   their own independently-integrated velocities, untouched. This is the exact
    ///   behavior that's completely absent today (only one field ever exists, so
    ///   nothing can ever separate).
    /// - Correction (eq. 10-13): remove the approaching normal component entirely, and
    ///   reduce the tangential component by up to `friction·|v_n|` (stick if that would
    ///   overshoot, matching Coulomb's cone). This is EXACTLY `apply_coulomb_wall`'s
    ///   existing, already-tested formula (`src/forces/boundary/mod.rs`), reused as-is
    ///   with `v_rel = v_grip - v_cm` standing in for "velocity relative to the wall"
    ///   and `n` standing in for the wall's outward normal — same math, different
    ///   partner.
    /// - Momentum conservation (eq. 14, `Σ m_α(v_α - v_cm) = 0`): correcting the grip
    ///   field and handing the rest field the exact opposite momentum delta conserves
    ///   total momentum by construction, with no separate reaction computation needed.
    ///
    /// Scope, disclosed: this is a 2-field (grip vs. rest) implementation, not full
    /// N-body multi-field contact — see `Particle::contact_group` doc. Also skips the
    /// paper's own further refinement (releasing contact based on normal TRACTION, not
    /// just kinematic approach/departure, for correct energy extraction on rebound) —
    /// the paper itself states the simpler kinematic-only criterion used here is exact
    /// "in the special case where contacting bodies are stress free," a real, legitimate
    /// baseline, not a hidden shortcut.
    ///
    /// `vel_limit`: the SAME CFL speed cap the caller already applies to the total
    /// field right before this call (`step.rs`'s grid-velocity clamp) — passed in and
    /// applied here too, to every velocity this function produces or reads raw. Without
    /// this, a tiny-mass grip node could carry a huge raw velocity (`grip_momentum` from
    /// a near-zero `grip_mass`) even when the total field is perfectly safe, silently
    /// reopening the exact instability the caller's clamp exists to prevent — a real
    /// gap, not a hypothetical one, closed here rather than left as a disclosed limit.
    ///
    /// HISTORY (found + fixed 2026-07-12): friction used to have ~zero measurable effect
    /// on a sliding body's bulk velocity. Root cause was isolated via direct instrumentation
    /// plus a hand-derived algebraic check (both matching): the CORRECTION formula itself
    /// was always exactly right — forcing `n = Vec2::Y` in an axis-aligned test made
    /// friction=0 hold a resting body's velocity at EXACTLY 0.0 forever (correct frictionless
    /// slip) and friction=3 produce genuine stick (bodies converge to the momentum-
    /// conserving common velocity) — the intended Bardenhagen behavior. The bug was entirely
    /// in the NORMAL ESTIMATE: a grid mass-GRADIENT normal (Bardenhagen's own original
    /// method) carries a small but PERSISTENT (not random-noise) off-axis bias for a body
    /// that is actively translating across the fixed Eulerian grid, or near a material
    /// edge/corner. Removing the "normal" component with a mistilted `n` bled real
    /// tangential momentum into the other field every substep regardless of `friction`, and
    /// since a body resting on a frictionless boundary (e.g. `SlipBoundary`) has nothing
    /// else opposing horizontal drift, this leak accumulated, unopposed, into full
    /// momentum-sharing over enough substeps. FALSIFIED as noise/transient-driven, each
    /// independently: wider-baseline central difference, a proper Sobel 3x3 gradient,
    /// raising the mass-fraction epsilon 4 orders of magnitude, a 100x stiffer/heavier
    /// floor, and a 300-step gradual velocity ramp instead of an instant jump — ALL still
    /// converged to the fully-stuck common velocity regardless of `friction`. Real fix,
    /// verified against Nairn 2020 ("New Material Point Method Contact Algorithms for
    /// Improved Accuracy," the direct, primary-source follow-up to Bardenhagen 2001 that
    /// diagnoses and fixes this exact class of bug — see `fit_contact_normal_lr`'s doc):
    /// replace the grid-gradient normal with a normal fitted through actual particle
    /// positions. See project memory `locomotion_core_frictional_contact_2026-07-11` for
    /// the full investigation log.
    ///
    /// TWO MORE REAL BUGS found and fixed the same day, both in this function, neither
    /// about the normal's direction:
    /// 1. The epsilon-skip branch (`grip_mass <= MIN_MASS_FRACTION`) used to trigger at
    ///    0.05, not a true divide-by-zero guard, and on trigger set BOTH fields to the
    ///    raw blended `total.momentum` — contaminating `rest`'s velocity with a real,
    ///    if small, grip contribution whenever grip_mass fell in `(0, 0.05]`. A taller
    ///    body creates far more such nodes (deeper kernel reach), so this leak scaled
    ///    with body thickness. Fixed: threshold dropped to `1e-6` (true zero-guard);
    ///    everything else fully separates via the "no confident normal" branch instead.
    /// 2. `fit_contact_normal_lr`'s Newton iteration could push `z = x·β` far enough
    ///    negative for an ill-constrained point cloud that `exp(-z)` overflowed to
    ///    `f32::INFINITY`, producing `inf/inf = NaN` in the sigma term — confirmed via
    ///    direct instrumentation (a specific recurring imbalanced point cloud produced
    ///    `Vec2(NaN, NaN)` every time). Fixed by clamping `z` before the exponential
    ///    (the logistic function saturates to ±1 well before this range, so the
    ///    clamp changes nothing about the converged answer) plus a defensive
    ///    `is_finite()` filter at every consumption point.
    ///
    /// A FOURTH issue, also found and fixed: when neither the LR fit nor anything else
    /// found a usable normal (typically a shallow, just-touching, heavily one-sided
    /// point cloud — exactly the moment a falling body first reaches another), the old
    /// code applied ZERO correction at that node. Confirmed via instrumentation: a block
    /// dropped onto a floor free-fell for its ENTIRE approach (matching pure free-fall
    /// kinematics almost exactly — contact wasn't resisting AT ALL) before tunneling deep
    /// and only then decelerating. Fixed by falling back to `grip_mass_gradient_normal`
    /// (the original Bardenhagen gradient method) whenever LR has no answer, rather than
    /// skipping correction outright — see that function's doc.
    ///
    /// A FIFTH issue, found AFTER the four fixes above and now also RESOLVED (same day,
    /// 2026-07-12): a body resting under sustained gravity would still settle several
    /// grid cells deep into the body beneath it, confirmed independent of normal quality,
    /// material-stiffness pairing, and impact severity. Root cause matched Bardenhagen
    /// 2001's own disclosed caveat that the kinematic-only approach/departure test (eq. 8
    /// alone) is exact only "in the special case where contacting bodies are stress
    /// free" — a resting body under constant gravity never is, so the test can prevent
    /// further approach but has no mechanism to correct overlap that already exists.
    /// Fixed via Baumgarte stabilization (see the inline doc further down, at the actual
    /// correction code, for the full two-attempt-then-fix numerical journey — the
    /// working version uses a dt-independent absolute correction rate/cap, not the
    /// textbook `beta*gap/dt` form, which explodes at this engine's adaptive substep dt).
    /// `multi_field_contact_produces_real_coulomb_slip_and_stick`
    /// (`tests/physics_correctness.rs`) now passes genuinely — both the frictionless
    /// slip case and the high-friction stick case verified on the harder
    /// `examples/diag_contact_debug.rs` diagnostic (settled gap ~0, `min_deformation_j`
    /// staying ~0.995-0.9997, no explosion). See project memory
    /// `locomotion_core_frictional_contact_2026-07-11` for the full investigation log.
    pub fn resolve_contact(
        &mut self,
        dt: f32,
        gravity: Vec2,
        friction: f32,
        vel_limit: f32,
        grid_cell_size: f32,
        directional_grip: Option<&DirectionalContactGrip>,
    ) {
        // Only a guard against literal division-by-zero, NOT a "low confidence" cutoff —
        // REAL BUG FOUND AND FIXED 2026-07-12: a larger threshold here (0.05, tried during
        // the normal-estimation investigation) sent every node with grip_mass in (0, 0.05]
        // through the branch below, which set BOTH fields to the raw blended `total.momentum`
        // -- but a small, nonzero grip_mass at that node means `total.momentum` (mass-
        // weighted across BOTH bodies) already carries a real, if small, contribution from
        // grip, contaminating what `rest` reads back. A taller/thicker body creates far more
        // such small-but-nonzero-grip-mass nodes (its kernel reaches deeper across more grid
        // rows) than a thin one, so this leak scaled with body thickness -- confirmed by
        // forcing a known-perfect vertical normal on both a thin (24x2) and thick (12x8)
        // block: the thin block showed zero leak (this branch rarely fired), the thick block
        // still leaked to the fully-momentum-shared value despite the perfect normal (this
        // branch fired constantly). Genuinely near-zero mass (no real second field at all)
        // still takes the fast, correct path here; everything else falls through to the
        // "no confident normal" branch below, which ALREADY does the correct, uncontaminated
        // per-field separation without applying a Coulomb correction.
        const MIN_MASS_FRACTION: f32 = 1.0e-6;
        let clamp_speed = |v: Vec2| -> Vec2 {
            let spd = v.length();
            if spd > vel_limit {
                v * (vel_limit / spd)
            } else {
                v
            }
        };
        for &idx in &self.contact_dirty {
            let node_pos = Vec2::new(
                (idx as usize / self.resolution) as f32,
                (idx as usize % self.resolution) as f32,
            );
            let Some(&total) = self.cells.get(&idx) else {
                continue;
            };
            let Some(contact) = self.contact_cells.get(&idx) else {
                continue;
            };

            let grip_mass = contact.grip_mass;
            let grip_momentum = contact.grip_momentum;
            let rest_mass = total.mass - grip_mass;
            if grip_mass <= MIN_MASS_FRACTION || rest_mass <= MIN_MASS_FRACTION {
                // No real second field at this node (e.g. a grip particle's kernel edge
                // with negligible weight) — both sides just read the ordinary total
                // field, identical to no contact resolution ever happening here.
                let cell = self.contact_cells.get_mut(&idx).unwrap();
                cell.resolved_grip_v = total.momentum;
                cell.resolved_rest_v = total.momentum;
                continue;
            }

            let v_cm = total.momentum; // already normalized + gravity-applied + clamped
            let v_grip = clamp_speed(grip_momentum / grip_mass + gravity * dt);

            // Contact normal fitted through the actual particle point cloud (Nairn's LR
            // method) rather than a grid mass gradient — see `fit_contact_normal_lr`'s
            // doc for why the gradient approach was a real, found bug. `-` because the
            // raw fit points toward increasing grip-label density (grip=+1); negating
            // matches this function's existing "outward: away from grip" convention.
            // `.filter(is_finite)`: defense in depth. `fit_contact_normal_lr` guards its
            // own iteration against non-finite results internally, but treating any
            // NaN/inf that slips through as "no confident normal" here (same as the
            // ordinary not-enough-points case) rather than propagating it into the
            // Coulomb correction is a real, cheap safety net for a value that used to
            // reach the correction unchecked and contaminate particle velocities.
            //
            // REAL BUG FOUND AND FIXED 2026-07-12: when the LR fit has no confident
            // answer (typically a shallow, just-touching, heavily one-sided point cloud
            // -- exactly the moment a fast-falling body FIRST reaches the floor), the old
            // code applied ZERO correction at that node: no interpenetration prevention
            // at all, not even an approximate one. Confirmed via direct instrumentation:
            // a block dropped onto a floor free-fell for the ENTIRE approach (matching
            // pure free-fall kinematics almost exactly, meaning contact wasn't resisting
            // AT ALL) and only started decelerating after tunneling several grid cells
            // deep -- well past the point contact should have engaged. A grid mass-
            // gradient fallback (the original, pre-LR method) is exactly the fallback
            // this needs: not as accurate as LR in general (that's WHY it was replaced
            // as the primary method), but always available and vastly better than no
            // normal at all for the specific case LR can't handle -- a lopsided,
            // barely-overlapping point cloud is close to the flattest, least ambiguous
            // geometry for a density gradient to read correctly anyway.
            // INVESTIGATED 2026-07-13, NOT FIXED -- see `examples/diag_contact_debug.rs`'s
            // own doc comment for the still-open follow-up. Instrumented every fitted
            // normal on the thick-block diagnostic and confirmed the LR fit is near-
            // perfectly vertical (|n.x| < 1e-4) through the bulk of the interface, but
            // degrades sharply -- |n.x| up to ~0.58, roughly 35 degrees off vertical -- at
            // a small, specific set of nodes: the column directly under the sliding
            // block's LEADING EDGE (>95% of all skewed fits landed on just 3 node rows at
            // that exact x, a genuine corner where grip's front face meets open space, not
            // a clean grip-over-rest half-plane). That skewed normal contributes to a real,
            // measured leak (frictionless slide, `diag_contact_debug --friction 0`: floor
            // picks up windowed_floor_vx~0.4 when it should stay ~0). Tried two real fixes,
            // BOTH made it worse, confirmed by measurement not assumption: (1) falling back
            // to `grip_mass_gradient_normal` on low confidence raised the leak to ~0.85 --
            // this function's own doc already discloses why, it has the same "known
            // weaknesses near a... corner"; (2) skipping resolution entirely at low-
            // confidence nodes (matching the existing "no confident normal" branch) also
            // gave ~0.82 -- doing nothing at the corner is worse than an imperfect normal,
            // because the corner then behaves like uncoupled single-field MPM exactly
            // where the leading edge is pressing into the floor, letting elastic stress
            // transfer real momentum with zero contact separation at all. The imperfect-
            // but-present LR normal outperforms both alternatives.
            //
            // STATUS UPDATE 2026-07-14 -- re-investigated with direct instrumentation on
            // a real 3400-step long-horizon run (not guessed): the "leading edge corner"
            // framing above was INCOMPLETE. Skewed fits (|n.x| > 0.3 on an otherwise
            // near-vertical interface) are NOT a rare corner-only event -- they occur
            // constantly, from step 0 onward, at ANY node whose point cloud has a small
            // or lopsided MINORITY-label sample count (as few as 1-2 points of one label
            // among dozens of the other), independent of whether the node sits at a real
            // geometric corner. Root cause, verified: a synthetic replica of Nairn 2020's
            // OWN worked corner example (Fig 3C) recovers a clean, near-horizontal normal
            // from this exact implementation (`nairn_fig3c_corner_case_recovers_horizontal_normal`)
            // -- ruling out corner TOPOLOGY as the failure mode, matching the paper's own
            // claim that LR handles this case correctly. The real failure is statistical:
            // this NLLS objective is a near-separable logistic fit, whose likelihood
            // surface goes nearly FLAT in orientation once the two labels are already
            // separated (saturated points stop contributing gradient) -- so with only a
            // handful of minority-label points still actually constraining the fit, a
            // small, physically meaningless perturbation in exactly those few points can
            // swing the converged plane by tens of degrees, and the paper's own FIXED
            // Tikhonov penalty (tuned for its own, better-sampled examples) doesn't
            // compensate for this at real MPM's often-thin per-node sample sizes.
            //
            // A THIRD real fix attempt, tried and ALSO falsified by direct measurement
            // (not assumed): a per-node temporal prior (`normal_history` -- this exact
            // node's own last confidently-fitted normal, reused only when the current
            // sample was statistically thin) made the real 16,000-step repro WORSE, not
            // better -- min_j_snake crashed to -1.0 by step 2000 (vs. taking the full
            // 16,000 steps to reach -4.83 without this change), and final min_j_terrain
            // hit -512.0 (vs. 0.0 without it). Reverted. Likely explanation: a stale
            // history value gets "frozen in" and repeatedly reapplied at every future
            // low-sample dip even after the real local geometry has moved on, actively
            // propagating an old wrong direction instead of letting each substep's
            // (occasionally noisy but always CURRENT) LR fit average out over time.
            // Three real, qualitatively different substitute-normal strategies now
            // falsified (spatial-gradient fallback, skip-entirely, temporal-history
            // fallback) -- this whole CLASS of fix ("swap in a different single normal
            // when uncertain") is looking structurally wrong, not just under-tuned.
            //
            // CONFIRMED 2026-07-14 -- the normal was never the real root cause. Direct
            // experiment (not guessed): running the exact 16,000-step long-horizon repro
            // with the Baumgarte position correction below (search "Baumgarte
            // stabilization") disabled entirely settles PERFECTLY cleanly -- min_j_terrain
            // holds exactly at its 0.6 floor, min_j_snake holds at 0.9224, vmax decays to
            // 0.000, for the full 16,000 steps. This isolates Baumgarte itself, independent
            // of the normal, as the actual source of the long-horizon runaway. Root cause:
            // the LR-fitted `n` is genuinely noisy substep to substep (confirmed separately
            // above), and Baumgarte's `gap` is measured by projecting onto this SAME noisy
            // `n` -- so even a truly at-rest body can show a small spurious `gap<0` from
            // fit jitter alone, and unlike the Coulomb term (which only ever REMOVES a
            // velocity component, bounded by what's already there), Baumgarte ADDS velocity
            // outright every substep it fires. A sequence of small, not-fully-cancelling
            // noise-driven additions compounds into real, unbounded kinetic energy over
            // thousands of substeps.
            //
            // First fix tried along this new lever, PARTIALLY helped but did NOT close
            // the bug (disclosed honestly, not force-passed): a deadband requiring `gap`
            // to exceed 5% of one grid cell before correcting. Measured result: onset
            // delayed but the 16,000-step test still ultimately failed -- real progress,
            // not a fix, reverted rather than ship a partial mitigation.
            //
            // FIXED 2026-07-14 (real fix, verified on the full 16,000-step repro, see the
            // Baumgarte correction site further down in this same function for the exact
            // change and its own doc comment): converted the unconditional ADDITIVE
            // velocity kick into a velocity FLOOR -- only pushes `v_rel`'s normal
            // component down to the target separating speed if it isn't there already,
            // the standard way real constraint solvers apply a position bias. This is
            // self-limiting: a wobbling normal's repeated firings can no longer stack
            // unbounded energy once real overlap is genuinely resolved, unlike the old
            // unconditional subtraction. Verified genuinely: this test's own assertion
            // (terrain holds its 0.6 floor) now passes for the full run with real margin.
            // Disclosed smaller residual, not blocking: the snake's own purely-elastic
            // body still settles to a mildly self-inverted but STABLE `min_j_snake≈-1.07`
            // (not the ≈0.92 the Baumgarte-fully-disabled experiment reached), unchanged
            // for 6000+ steps -- bounded, not runaway, and not what this test asserts on.
            let normal_fit = fit_contact_normal_lr(&contact.points, node_pos, grid_cell_size)
                .filter(|n| n.is_finite())
                .or_else(|| self.grip_mass_gradient_normal(idx));
            let mut v_rel = v_grip - v_cm;
            let Some(n) = normal_fit.map(|n| -n) else {
                // Neither the LR fit nor the gradient fallback found a usable normal
                // (e.g. truly no local gradient AND too few points) -- resolve nothing
                // at this specific node this substep (other nodes along the same
                // interface still carry the real contact for the body as a whole).
                let cell = self.contact_cells.get_mut(&idx).unwrap();
                cell.resolved_grip_v = v_grip;
                cell.resolved_rest_v =
                    clamp_speed((v_cm * total.mass - v_grip * grip_mass) / rest_mass);
                continue;
            };

            match directional_grip {
                Some(grip) => grip.resolve(&mut v_rel, n),
                None => crate::boundary::apply_coulomb_wall(&mut v_rel, n, friction),
            }

            // Baumgarte stabilization (Baumgarte 1972, "Stabilization of Constraints and
            // Integration of PDEs of Dynamical Systems" -- a real, standard, decades-old
            // technique, not invented here; the same ~0.1-0.3 factor is the well-known
            // default in e.g. Box2D/Bullet's own velocity-constraint solvers). REAL BUG
            // FOUND AND FIXED 2026-07-12: the kinematic-only approach test above only
            // prevents FURTHER approach once it fires -- it has no mechanism to correct
            // overlap that already exists, which matches Bardenhagen 2001's own disclosed
            // caveat that this simpler test is exact only "in the special case where
            // contacting bodies are stress free" (a resting body under constant gravity
            // never is). Confirmed via direct instrumentation: a resting body settled
            // several grid cells deep into whatever it rested on and never recovered,
            // independent of normal quality (persisted even with a hand-forced, exactly-
            // correct `n = Vec2::Y`), material stiffness pairing, and impact severity --
            // proving the missing piece was positional correction, not the normal or the
            // velocity-matching formula (both independently verified correct already).
            // Reuses the SAME particle point cloud already gathered for the LR fit (no
            // new data needed): project every particle onto `n`; if grip's furthest-along-
            // n particle has crossed past rest's closest-along-n particle, that's real,
            // measured overlap, not a guess. The correction is damped (proportional, not
            // instantaneous) specifically to avoid injecting energy or overshooting into a
            // new oscillation -- the well-documented failure mode of a naive "snap back
            // instantly" position fix, which is why Baumgarte-style damping is the
            // standard approach instead.
            //
            // REAL BUG FOUND AND FIXED 2026-07-12 (same day, found via direct instrumented
            // re-test, twice): the textbook `beta * gap / dt` formula assumes a roughly
            // FIXED timestep (its usual home, e.g. Box2D, always steps at a fixed 1/60s) --
            // this engine's ADAPTIVE substep dt can legitimately shrink to ~1e-6 for a
            // stiff material's CFL bound, and the raw formula blows up as dt->0 (confirmed:
            // an uncapped version caused a genuine explosion, velocities into the tens,
            // min_deformation_j collapsing toward 0.5). Clamping to `vel_limit` was the
            // FIRST attempt and did NOT fix it, because `vel_limit` is ITSELF a CFL bound
            // that scales as 1/dt by design (`grid_cell_size / sub_dt`) -- it grows in
            // lockstep with the very blowup it was meant to cap, so the clamp did nothing
            // real (confirmed: still exploded, just slightly less). The genuine fix removes
            // `dt` from the correction entirely: a small, ABSOLUTE correction rate and speed
            // cap, so the position fix stays bounded and gentle at ANY substep size,
            // correcting large overlaps over several substeps instead of injecting one huge
            // velocity kick that then feeds into stress/deformation as if it were real
            // physical momentum (which is what actually caused the explosion -- a huge
            // "correction" velocity distorts F just as much as a real one would).
            let mut max_grip_proj = f32::NEG_INFINITY;
            let mut min_rest_proj = f32::INFINITY;
            for &(pos, label) in &contact.points {
                let proj = pos.dot(n);
                if label > 0.0 {
                    max_grip_proj = max_grip_proj.max(proj);
                } else if label < 0.0 {
                    min_rest_proj = min_rest_proj.min(proj);
                }
            }
            if max_grip_proj.is_finite() && min_rest_proj.is_finite() {
                let gap = min_rest_proj - max_grip_proj; // >0 separated, <0 overlapping
                if gap < 0.0 {
                    // Neither derived from dt nor from vel_limit -- a fixed, small correction
                    // rate (fraction of the overlap corrected per unit REAL time) and an
                    // absolute speed ceiling (a small fraction of one grid cell per unit real
                    // time), both independent of how finely the adaptive substep loop divides
                    // that time up.
                    const CORRECTION_RATE: f32 = 2.0;
                    let max_correction_speed = 0.5 * grid_cell_size;
                    let correction_speed = (CORRECTION_RATE * (-gap)).min(max_correction_speed);
                    // REAL BUG FOUND AND FIXED 2026-07-14 (root cause confirmed via a
                    // direct isolation experiment -- disabling this whole block entirely
                    // let a real 16,000-step passive settle hold perfectly, proving THIS
                    // term, not the contact normal, was Bug 2's actual source; see project
                    // memory `locomotion_core_frictional_contact_2026-07-11`'s 2026-07-14
                    // update for the full investigation). The old code unconditionally
                    // SUBTRACTED `n * correction_speed` from `v_rel` every single substep
                    // this branch fired, regardless of `v_rel`'s own current normal
                    // component -- i.e. it always added a fixed-magnitude impulse, even
                    // when the body was ALREADY separating faster than `correction_speed`
                    // required (e.g. from the previous substep's own correction, along a
                    // slightly different noisy `n`). Because the LR-fitted `n` genuinely
                    // wobbles substep to substep (confirmed separately), each firing's
                    // impulse points in a slightly different direction even for the same
                    // physical overlap -- an unconditional additive term keeps stacking
                    // these on top of each other with no cap on the TOTAL applied so far,
                    // which is a real, unbounded numerical-heating mechanism over
                    // thousands of substeps (a directional random walk in velocity space).
                    // Fixed by converting the unconditional ADD into a velocity FLOOR:
                    // only push `v_rel`'s normal component down to the target if it isn't
                    // there already. This is the standard way position-bias corrections
                    // are applied in real constraint solvers (Box2D/Bullet-style sequential
                    // impulse: the bias only tops up a relative velocity that's below the
                    // target, it never re-applies once the target is already met) --
                    // self-limiting by construction, so repeated firings from a wobbling
                    // normal can no longer stack unbounded energy once the real overlap is
                    // genuinely being resolved, unlike the old unconditional subtraction.
                    let v_n = v_rel.dot(n);
                    let target_vn = -correction_speed;
                    if v_n > target_vn {
                        v_rel += n * (target_vn - v_n);
                    }
                }
            }

            let v_grip_new = clamp_speed(v_cm + v_rel);

            // Exact momentum conservation: whatever the grip field's momentum changed
            // by, the rest field absorbs the opposite delta (eq. 14's identity holds by
            // construction, not by a separate reaction computation). Computed from the
            // clamped v_grip_new so the conservation identity still holds against what
            // G2P will actually read.
            let total_momentum = v_cm * total.mass;
            let v_rest_new = clamp_speed((total_momentum - v_grip_new * grip_mass) / rest_mass);

            let cell = self.contact_cells.get_mut(&idx).unwrap();
            cell.resolved_grip_v = v_grip_new;
            cell.resolved_rest_v = v_rest_new;
        }
    }
}

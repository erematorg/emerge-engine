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
    /// `fit_contact_normal_lr`'s "no confident plane" case. Not the primary method
    /// (has known weaknesses near a translating body or a material corner), but
    /// exactly the shallow, one-sided point clouds where LR fails tend to be close
    /// to a flat interface, the case this handles best. Returns `None` when there's
    /// no local gradient (deep inside a well-mixed interior).
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
    ///
    /// - Per-field velocity `v_grip = p_grip/m_grip` (eq. 4); center-of-mass velocity
    ///   `v_cm` is just this grid's own existing total field (eq. 5-6) — already computed
    ///   by `update_velocities`, called right before this.
    /// - Surface normal `n`: fitted via logistic regression through a labeled particle
    ///   point cloud (`fit_contact_normal_lr`), not a grid mass gradient — see that
    ///   function's doc.
    /// - Approach test (eq. 8): contact applies only when `(v_grip - v_cm)·n < 0`
    ///   (bodies approaching); otherwise free separation — the two fields keep their
    ///   own independently-integrated velocities, untouched.
    /// - Correction (eq. 10-13): remove the approaching normal component entirely, and
    ///   reduce the tangential component by up to `friction·|v_n|` (stick if that would
    ///   overshoot, matching Coulomb's cone). This is exactly `apply_coulomb_wall`'s
    ///   existing formula (`src/forces/boundary/mod.rs`), reused with
    ///   `v_rel = v_grip - v_cm` standing in for "velocity relative to the wall" and
    ///   `n` for the wall's outward normal.
    /// - Momentum conservation (eq. 14, `Σ m_α(v_α - v_cm) = 0`): correcting the grip
    ///   field and handing the rest field the exact opposite momentum delta conserves
    ///   total momentum by construction.
    ///
    /// Scope, disclosed: this is a 2-field (grip vs. rest) implementation, not full
    /// N-body multi-field contact — see `Particle::contact_group` doc. Also skips the
    /// paper's own refinement of releasing contact based on normal TRACTION rather than
    /// kinematic approach/departure — the paper itself states the simpler kinematic-only
    /// criterion used here is exact "in the special case where contacting bodies are
    /// stress free."
    ///
    /// `multi_field_contact_produces_real_coulomb_slip_and_stick`
    /// (`tests/physics_correctness.rs`) verifies both the frictionless-slip and
    /// high-friction-stick cases.
    pub fn resolve_contact(
        &mut self,
        dt: f32,
        gravity: Vec2,
        friction: f32,
        grid_cell_size: f32,
        directional_grip: Option<&DirectionalContactGrip>,
    ) {
        // Only a guard against literal division-by-zero, NOT a "low confidence" cutoff —
        // a larger threshold here would route every node with small-but-nonzero grip_mass
        // through the branch below, which sets both fields to the raw blended
        // `total.momentum`. But any nonzero grip_mass means `total.momentum` (mass-weighted
        // across BOTH bodies) already carries a real contribution from grip, contaminating
        // what `rest` reads back — worse for thicker bodies (their kernel reaches more
        // small-but-nonzero-grip-mass nodes). Everything above this floor falls through to
        // the "no confident normal" branch below instead, which does correct,
        // uncontaminated per-field separation without a Coulomb correction.
        const MIN_MASS_FRACTION: f32 = 1.0e-6;
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

            let v_cm = total.momentum; // already normalized + gravity-applied
            let v_grip = grip_momentum / grip_mass + gravity * dt;

            // Contact normal fitted through the actual particle point cloud (Nairn's LR
            // method) rather than a grid mass gradient. `-` because the raw fit points
            // toward increasing grip-label density (grip=+1); negating matches this
            // function's "outward: away from grip" convention. `.filter(is_finite)`:
            // defense in depth — `fit_contact_normal_lr` guards its own iteration against
            // non-finite results internally, but any NaN/inf that slips through is treated
            // as "no confident normal" here rather than propagating into the Coulomb
            // correction and contaminating particle velocities.
            //
            // When the LR fit has no confident answer (typically a shallow, just-touching,
            // heavily one-sided point cloud -- exactly the moment a fast-falling body first
            // reaches the floor), fall back to the grid mass-gradient normal rather than
            // applying zero correction -- otherwise the body free-falls straight through
            // before tunneling deep and only then decelerating.
            //
            // Known disclosed limitation: the LR fit can be noisy at nodes with a small or
            // lopsided minority-label sample count (e.g. near a body's leading edge), which
            // is why the Baumgarte correction below must be a velocity floor rather than an
            // unconditional additive term (see that comment).
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
                cell.resolved_rest_v = (v_cm * total.mass - v_grip * grip_mass) / rest_mass;
                continue;
            };

            match directional_grip {
                Some(grip) => grip.resolve(&mut v_rel, n),
                None => crate::boundary::apply_coulomb_wall(&mut v_rel, n, friction),
            }

            // Baumgarte stabilization (Baumgarte 1972, "Stabilization of Constraints and
            // Integration of PDEs of Dynamical Systems"; the same ~0.1-0.3 factor is the
            // well-known default in e.g. Box2D/Bullet's own velocity-constraint solvers).
            // The kinematic-only approach test above only prevents FURTHER approach once
            // it fires — it has no mechanism to correct overlap that already exists, which
            // matches Bardenhagen 2001's own disclosed caveat that this simpler test is
            // exact only "in the special case where contacting bodies are stress free" (a
            // resting body under constant gravity never is). Reuses the SAME particle point
            // cloud already gathered for the LR fit: project every particle onto `n`; if
            // grip's furthest-along-n particle has crossed past rest's closest-along-n
            // particle, that's measured overlap. The correction is damped (proportional,
            // not instantaneous) to avoid injecting energy or overshooting into a new
            // oscillation.
            //
            // Correction rate/speed must be a dt-INDEPENDENT absolute value, not the
            // textbook `beta * gap / dt` (which assumes a roughly fixed timestep): the
            // engine's adaptive substep dt can shrink for stiff solids, and the raw formula
            // then blows up as dt->0. This is a contact-constraint stabilization for solid
            // scenes, not a fluid constitutive term; strict WC-MPM liquids reject multi-field
            // contact before reaching this solver.
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
                    // Neither derived from dt nor a generic velocity limiter -- a fixed, small correction
                    // rate (fraction of the overlap corrected per unit REAL time) and an
                    // absolute speed ceiling (a small fraction of one grid cell per unit real
                    // time), both independent of how finely the adaptive substep loop divides
                    // that time up.
                    const CORRECTION_RATE: f32 = 2.0;
                    let max_correction_speed = 0.5 * grid_cell_size;
                    let correction_speed = (CORRECTION_RATE * (-gap)).min(max_correction_speed);
                    // Must be a velocity FLOOR, not an unconditional additive term: only push
                    // `v_rel`'s normal component down to the target if it isn't there already.
                    // The LR-fitted `n` wobbles substep to substep, so an unconditional add
                    // would stack a slightly-different-direction impulse every firing with no
                    // cap on the total applied — an unbounded numerical-heating mechanism (a
                    // directional random walk in velocity space). A floor is self-limiting:
                    // it never re-applies once the target is already met (same principle as
                    // Box2D/Bullet-style sequential-impulse position bias).
                    let v_n = v_rel.dot(n);
                    let target_vn = -correction_speed;
                    if v_n > target_vn {
                        v_rel += n * (target_vn - v_n);
                    }
                }
            }

            let v_grip_new = v_cm + v_rel;

            // Exact momentum conservation: whatever the grip field's momentum changed
            // by, the rest field absorbs the opposite delta (eq. 14's identity holds by
            // construction, not by a separate reaction computation). Computed from the
            // clamped v_grip_new so the conservation identity still holds against what
            // G2P will actually read.
            let total_momentum = v_cm * total.mass;
            let v_rest_new = (total_momentum - v_grip_new * grip_mass) / rest_mass;

            let cell = self.contact_cells.get_mut(&idx).unwrap();
            cell.resolved_grip_v = v_grip_new;
            cell.resolved_rest_v = v_rest_new;
        }
    }
}

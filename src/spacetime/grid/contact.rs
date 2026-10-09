use std::collections::HashMap;

use glam::{IVec2, Mat2, Vec2};

use super::contact_normal::fit_contact_normal_lr;
use super::directional_grip::DirectionalContactGrip;
use super::{FxU32BuildHasher, Grid, flat_index};

/// Second velocity field for multi-field frictional contact (Bardenhagen, Guilkey,
/// Roessig, Brackbill 2001) -- see `Particle::contact_group`'s doc for the full
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
/// touches this node -- the "point cloud" the logistic-regression contact normal
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
    /// For each of `points`, its particle's inverse deformation gradient and
    /// undeformed half size, to find how far its deformed edge reaches along
    /// the contact normal (Nairn, Hammerquist and Smith 2020, eq. 25).
    extents: Vec<(Mat2, f32)>,
}

pub(super) type ContactCellMap = HashMap<u32, ContactCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate mass and momentum for the "grip" contact field (particles with
    /// `contact_group != 0`) during P2G, additively alongside the normal
    /// `add_mass_momentum` call for the SAME particle -- this is a second, separate
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
                    extents: Vec::new(),
                });
            }
        }
    }

    /// Whether `cell_pos` is a contact-active node this substep, the cells
    /// `add_contact_point` appends to. Read-only, so a parallel pass can ask
    /// before deciding what to append.
    pub(crate) fn is_contact_node(&self, cell_pos: IVec2) -> bool {
        flat_index(cell_pos, self.resolution)
            .is_some_and(|idx| self.contact_cells.contains_key(&idx))
    }

    /// Appends one labeled particle position (`+1.0` grip / `-1.0` rest) to
    /// `cell_pos`'s contact point cloud, for the logistic-regression normal fit
    /// (`fit_contact_normal_lr`). Only pushes into a cell that ALREADY exists in
    /// `contact_cells` (i.e. one at least one grip particle already touched via
    /// `add_grip_mass_momentum` this substep) -- never creates a new entry, so a
    /// rest particle far from any grip body cannot spuriously grow `contact_dirty`.
    /// Called from a second particle pass (`gather_contact_point_cloud`) run
    /// AFTER the main P2G scatter has fully determined which nodes are
    /// contact-active, so this is deliberately not merged into
    /// `scatter_particles_to_grid` itself. OOB silently ignored.
    /// `inverse_deformation` and `half_size` (the particle's undeformed half
    /// size, in cells) locate its deformed edge along any normal.
    pub fn add_contact_point(
        &mut self,
        cell_pos: IVec2,
        position: Vec2,
        label: f32,
        inverse_deformation: Mat2,
        half_size: f32,
    ) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        if let Some(cell) = self.contact_cells.get_mut(&idx) {
            cell.points.push((position, label));
            cell.extents.push((inverse_deformation, half_size));
        }
    }

    /// Resolved "grip" field velocity at `cell_pos` -- valid after `resolve_contact()`.
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

    /// Resolved "rest" (contact_group == 0) field velocity at `cell_pos` -- valid after
    /// `resolve_contact()`. Falls back to the ordinary total velocity when no contact
    /// was registered at this node, which is the common case away from any grip body --
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
    /// `grip_mass_gradient_normal` below -- a tiny, deliberately local helper, not a
    /// public query (there's no meaningful "grip mass" outside contact resolution).
    fn grip_mass_at(&self, cell_pos: IVec2) -> f32 {
        flat_index(cell_pos, self.resolution)
            .and_then(|idx| self.contact_cells.get(&idx))
            .map_or(0.0, |c| c.grip_mass)
    }

    /// Fallback contact normal: Sobel-3x3 gradient of the grip field's own grid mass --
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
    ///   `v_cm` is just this grid's own existing total field (eq. 5-6) -- already computed
    ///   by `update_velocities`, called right before this.
    /// - Surface normal `n`: fitted via logistic regression through a labeled particle
    ///   point cloud (`fit_contact_normal_lr`), not a grid mass gradient -- see that
    ///   function's doc.
    /// - Contact detection (Nairn, Hammerquist and Smith 2020, eq. 14-15, 22-25):
    ///   with `n` pointing from the grip body into the rest body, contact applies only
    ///   where the bodies approach, `(v_grip - v_cm)·n > 0`, AND touch, the separation
    ///   of their particles' deformed edges along `n` being negative. Otherwise the
    ///   two fields keep their own independently-integrated velocities, untouched.
    ///   There is no position correction: a body resting on another is held by the
    ///   approach its own weight produces each substep, corrected as it arises.
    /// - Correction (eq. 10-13): remove the approaching normal component entirely, and
    ///   reduce the tangential component by up to `friction·|v_n|` (stick if that would
    ///   overshoot, matching Coulomb's cone). This is exactly `apply_coulomb_wall`'s
    ///   existing formula (`src/forces/boundary/mod.rs`), reused with
    ///   `v_rel = v_grip - v_cm` standing in for "velocity relative to the wall" and
    ///   `-n` for the wall's outward normal.
    /// - Momentum conservation (eq. 14, `Σ m_α(v_α - v_cm) = 0`): correcting the grip
    ///   field and handing the rest field the exact opposite momentum delta conserves
    ///   total momentum by construction.
    /// - Small nodal mass (section 2.2, eq. 17-22): that opposite delta changes a body's
    ///   velocity by the other body's mass over its own, so at a node where one body
    ///   has almost no mass the correction becomes very large. The strain increment a
    ///   correction imposes, `max |Δv| dt / dx` over both bodies and both axes, is held
    ///   under `stability_fraction` (the paper's `γ`, used there at 0.5 and recommended
    ///   in 0.5 to 1; here `SimConfig::material_cfl_coefficient`) by scaling both
    ///   bodies' changes by the same factor, which keeps the momentum identity. Measured
    ///   on a sand bed with a resting body in contact, 1.1 percent of two-field nodes
    ///   exceeded one cell per substep, every one on the body holding under 10 percent
    ///   of the node's mass, the worst at 10 447 times.
    ///
    /// Scope, disclosed: this is a 2-field (grip vs. rest) implementation, not full
    /// N-body multi-field contact -- see `Particle::contact_group` doc. Also skips the
    /// paper's own refinement of releasing contact based on normal TRACTION rather than
    /// kinematic approach/departure -- the paper itself states the simpler kinematic-only
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
        stability_fraction: f32,
        directional_grip: Option<&DirectionalContactGrip>,
    ) {
        // Only a guard against literal division-by-zero, NOT a "low confidence" cutoff --
        // a larger threshold here would route every node with small-but-nonzero grip_mass
        // through the branch below, which sets both fields to the raw blended
        // `total.momentum`. But any nonzero grip_mass means `total.momentum` (mass-weighted
        // across BOTH bodies) already carries a contribution from grip, contaminating
        // what `rest` reads back -- worse for thicker bodies (their kernel reaches more
        // small-but-nonzero-grip-mass nodes). Everything above this floor falls through to
        // the "no confident normal" branch below instead, which does correct,
        // uncontaminated per-field separation without a Coulomb correction.
        const MIN_NODE_MASS: f32 = 1.0e-6;
        // Frictional dissipation found here is collected and written after the
        // loop: the loop already borrows `contact_dirty` and `contact_cells`.
        // Empty whenever nothing rubs, so a contact-free scene allocates nothing.
        let mut dissipated: Vec<(usize, f32)> = Vec::new();
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
            if grip_mass <= MIN_NODE_MASS || rest_mass <= MIN_NODE_MASS {
                // No real second field at this node (e.g. a grip particle's kernel edge
                // with negligible weight) -- both sides just read the ordinary total
                // field, identical to no contact resolution ever happening here.
                let cell = self.contact_cells.get_mut(&idx).unwrap();
                cell.resolved_grip_v = total.momentum;
                cell.resolved_rest_v = total.momentum;
                continue;
            }

            let v_cm = total.momentum; // already normalized + gravity-applied
            let v_grip = grip_momentum / grip_mass + gravity * dt;

            // Contact normal fitted through the actual particle point cloud (Nairn et
            // al.'s LR method) rather than a grid mass gradient. `-` because the raw fit points
            // toward increasing grip-label density (grip=+1); negating matches this
            // function's "outward: away from grip" convention. `.filter(is_finite)`:
            // defense in depth -- `fit_contact_normal_lr` guards its own iteration against
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
            // lopsided minority-label sample count (e.g. near a body's leading edge).
            let normal_fit = fit_contact_normal_lr(&contact.points, node_pos, grid_cell_size)
                .filter(|n| n.is_finite())
                .or_else(|| self.grip_mass_gradient_normal(idx));
            let mut v_rel = v_grip - v_cm;
            let v_rel_before = v_rel;
            let Some(n) = normal_fit.map(|n| -n) else {
                // Neither the LR fit nor the gradient fallback found a usable normal
                // (e.g. truly no local gradient AND too few points) -- resolve nothing
                // at this specific node this substep (other nodes along the same
                // interface still carry the contact for the body as a whole).
                let cell = self.contact_cells.get_mut(&idx).unwrap();
                cell.resolved_grip_v = v_grip;
                cell.resolved_rest_v = (v_cm * total.mass - v_grip * grip_mass) / rest_mass;
                continue;
            };

            // Contact exists only where the bodies touch (Nairn, Hammerquist and
            // Smith 2020, eq. 15, 22-25): the separation between their deformed
            // edges along `n`, `d = min_rest(X.n - R_p) - max_grip(X.n + R_p)`,
            // is negative. `R_p` is how far particle p's deformed edge reaches
            // along `n`: its undeformed half size over `|F_p^-1 n|` (eq. 25, an
            // inscribed circle deformed by F). Measured from the centres alone,
            // bodies touching edge to edge read a gap of a particle spacing, and
            // contact began only once the centres had passed each other.
            let mut max_grip_edge = f32::NEG_INFINITY;
            let mut min_rest_edge = f32::INFINITY;
            for (&(pos, label), &(inverse_deformation, half_size)) in
                contact.points.iter().zip(&contact.extents)
            {
                let reach = half_size / (inverse_deformation * n).length().max(f32::MIN_POSITIVE);
                let proj = pos.dot(n);
                if label > 0.0 {
                    max_grip_edge = max_grip_edge.max(proj + reach);
                } else if label < 0.0 {
                    min_rest_edge = min_rest_edge.min(proj - reach);
                }
            }
            let touching = min_rest_edge - max_grip_edge < 0.0;

            // Approaching (eq. 14) is `apply_coulomb_wall`'s own test. It takes the
            // wall's outward normal, pointing from the wall into the body it
            // corrects: from the rest body into the grip body, `-n`.
            let wall_normal = -n;
            let mut friction_per_reduced_mass = 0.0;
            if touching {
                match directional_grip {
                    Some(grip) => grip.resolve(&mut v_rel, wall_normal),
                    // Multi-field contact between two bodies dissipates too, and
                    // feeds the same frictional-heating ledger as a wall
                    // (`energy::thermodynamics::frictional_heating`), converted
                    // below: `apply_coulomb_wall` reports energy per unit mass of
                    // what it moved, here the RELATIVE velocity, so per unit
                    // reduced mass `m_grip m_rest / m_total`; the ledger is per
                    // unit node mass.
                    None => {
                        friction_per_reduced_mass =
                            crate::boundary::apply_coulomb_wall(&mut v_rel, wall_normal, friction);
                    }
                }
            }

            // Small nodal mass (Bardenhagen et al. 2001, eq. 17-22, see this
            // function's doc): the rest field's change is the grip field's times
            // `-grip_mass / rest_mass`, so both are scaled together when either
            // would strain a cell by more than `stability_fraction` in one substep.
            let grip_change = v_rel - v_rel_before;
            let rest_change = grip_change * (-grip_mass / rest_mass);
            let strain_increment = grip_change
                .abs()
                .max_element()
                .max(rest_change.abs().max_element())
                * dt
                / grid_cell_size;
            let scale = if strain_increment > stability_fraction {
                stability_fraction / strain_increment
            } else {
                1.0
            };
            let v_grip_new = if scale < 1.0 {
                v_grip + grip_change * scale
            } else {
                v_cm + v_rel
            };
            if friction_per_reduced_mass > 0.0 && total.mass > 0.0 {
                // `apply_coulomb_wall` reports the loss for the full correction. A
                // scaled one keeps the tangential direction and moves its speed
                // only part of the way, so the loss is recomputed from that speed.
                let per_reduced_mass = if scale < 1.0 {
                    let tangential = |v: Vec2| (v - n * v.dot(n)).length();
                    let before = tangential(v_rel_before);
                    let after = before + (tangential(v_rel) - before) * scale;
                    0.5 * (before * before - after * after)
                } else {
                    friction_per_reduced_mass
                };
                let reduced_mass = grip_mass * rest_mass / total.mass;
                dissipated.push((idx as usize, per_reduced_mass * reduced_mass / total.mass));
            }

            // Exact momentum conservation: whatever the grip field's momentum changed
            // by, the rest field absorbs the opposite delta (eq. 14's identity holds by
            // construction, not by a separate reaction computation). Computed from
            // v_grip_new so the conservation identity still holds against what G2P
            // will actually read.
            let total_momentum = v_cm * total.mass;
            let v_rest_new = (total_momentum - v_grip_new * grip_mass) / rest_mass;

            let cell = self.contact_cells.get_mut(&idx).unwrap();
            cell.resolved_grip_v = v_grip_new;
            cell.resolved_rest_v = v_rest_new;
        }
        for (idx, specific_energy) in dissipated {
            self.add_friction_heat(idx, specific_energy);
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{IVec2, Mat2, Vec2};

    use crate::grid::Grid;

    const NODE: IVec2 = IVec2::new(8, 8);
    const DT: f32 = 0.01;
    /// Particles of 0.5 cell, as spawned at spacing 0.5.
    const HALF_SIZE: f32 = 0.25;

    /// One node shared by two bodies: `grip_mass` at `grip_v` just above it,
    /// `rest_mass` at rest just below, their particle centres `centre_gap`
    /// apart across the node. Resolved with `stability_fraction` and
    /// `friction`. Returns (grip, rest) resolved velocities.
    fn resolved(
        (grip_mass, rest_mass): (f32, f32),
        grip_v: Vec2,
        centre_gap: f32,
        (friction, stability_fraction): (f32, f32),
    ) -> (Vec2, Vec2) {
        let mut grid = Grid::new(16);
        grid.add_mass_momentum(NODE, grip_mass + rest_mass, grip_v * grip_mass);
        grid.add_grip_mass_momentum(NODE, grip_mass, grip_v * grip_mass);
        let y = NODE.y as f32;
        for i in 0..6 {
            let x = NODE.x as f32 - 0.5 + 0.2 * i as f32;
            for (dy, label) in [(0.5 * centre_gap, 1.0), (-0.5 * centre_gap, -1.0)] {
                grid.add_contact_point(
                    NODE,
                    Vec2::new(x, y + dy),
                    label,
                    Mat2::IDENTITY,
                    HALF_SIZE,
                );
            }
        }
        grid.update_velocities(DT, Vec2::ZERO);
        grid.resolve_contact(DT, Vec2::ZERO, friction, 1.0, stability_fraction, None);
        (grid.grip_velocity_at(NODE), grid.rest_velocity_at(NODE))
    }

    /// Touching: centres half a cell apart, edges meeting.
    const TOUCHING: f32 = 0.45;

    #[test]
    fn approaching_bodies_stop_closing_and_rub() {
        // Grip coming down onto rest, and sliding along it.
        let (grip, rest) = resolved((1.0, 1.0), Vec2::new(0.4, -1.0), TOUCHING, (0.2, 0.5));
        let closing = rest.y - grip.y;
        assert!(
            closing.abs() < 1.0e-5,
            "still closing at {closing} (grip {grip:?}, rest {rest:?})"
        );
        // Coulomb: the sliding difference drops by friction times the normal
        // difference removed (1.0 here), from 0.4 to 0.2.
        let sliding = grip.x - rest.x;
        assert!((sliding - 0.2).abs() < 1.0e-4, "sliding {sliding}");
        assert!((grip + rest - Vec2::new(0.4, -1.0)).length() < 1.0e-5);
    }

    #[test]
    fn separating_bodies_are_left_free() {
        let grip_v = Vec2::new(0.4, 1.0);
        let (grip, rest) = resolved((1.0, 1.0), grip_v, TOUCHING, (0.2, 0.5));
        assert!((grip - grip_v).length() < 1.0e-5, "grip {grip:?}");
        assert!(rest.length() < 1.0e-5, "rest {rest:?}");
    }

    #[test]
    fn approaching_bodies_whose_edges_do_not_meet_are_left_free() {
        // Centres 1.5 cells apart: edges a whole cell apart.
        let grip_v = Vec2::new(0.4, -1.0);
        let (grip, rest) = resolved((1.0, 1.0), grip_v, 1.5, (0.2, 0.5));
        assert!((grip - grip_v).length() < 1.0e-5, "grip {grip:?}");
        assert!(rest.length() < 1.0e-5, "rest {rest:?}");
    }

    /// The strain increment a resolution imposed on either body, in cells.
    fn strain((grip, rest): (Vec2, Vec2), grip_v: Vec2) -> f32 {
        (grip - grip_v)
            .abs()
            .max_element()
            .max(rest.abs().max_element())
            * DT
    }

    #[test]
    fn a_nearly_massless_body_gets_a_bounded_correction_that_keeps_momentum() {
        // A body approaching faster than a cell a substep onto one holding
        // 1e-4 of the node's mass.
        let (masses, grip_v) = ((1.0, 1.0e-4), Vec2::new(30.0, -200.0));
        let unbounded = resolved(masses, grip_v, TOUCHING, (0.5, f32::INFINITY));
        assert!(strain(unbounded, grip_v) > 1.0, "premise: {unbounded:?}");
        let bounded = resolved(masses, grip_v, TOUCHING, (0.5, 0.5));
        // The rest body's velocity comes from the momentum identity divided by
        // its 1e-4 mass, which scales f32 rounding by 1e4: 0.5004 is read here.
        assert!(
            strain(bounded, grip_v) <= 0.5 * (1.0 + 1.0e-2),
            "{bounded:?}"
        );
        let momentum = bounded.0 * masses.0 + bounded.1 * masses.1;
        assert!(
            (momentum - grip_v * masses.0).length() < 1.0e-3,
            "momentum {momentum:?}"
        );
        // Scaled, not redirected: the same correction, shortened. Read on the
        // rest body, which started still: the grip body's own change is 1e4
        // times smaller and its direction lost in rounding.
        let (full, part) = (unbounded.1, bounded.1);
        assert!(
            full.perp_dot(part).abs() < 1.0e-2 * full.length() * part.length(),
            "{full:?} against {part:?}"
        );
        assert!(full.dot(part) > 0.0);
    }

    #[test]
    fn comparable_bodies_are_resolved_exactly_as_without_the_bound() {
        let grip_v = Vec2::new(0.3, -0.5);
        let unbounded = resolved((1.0, 1.0), grip_v, TOUCHING, (0.5, f32::INFINITY));
        assert_ne!(unbounded.0, grip_v, "premise: a correction happened");
        assert!(strain(unbounded, grip_v) < 0.5);
        assert_eq!(
            resolved((1.0, 1.0), grip_v, TOUCHING, (0.5, 0.5)),
            unbounded
        );
    }
}

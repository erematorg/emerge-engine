//! The law of a solid touching a wall or another solid: it may push but
//! never pull (Signorini), and while it pushes, friction resists sliding up
//! to a limit (Coulomb).
//!
//! Signorini: the gap `g` between the two surfaces stays at or above zero,
//! the normal force `N` stays at or above zero (contact pushes, never
//! pulls), and one of the two is zero at any time (`g N = 0`: no force at a
//! distance, no gap under load).
//! Source: Dabaghi, Krejci, Petrov, Pousin and Renard, arXiv 1601.00778
//! (2016), eq. 2.3.
//!
//! Coulomb: the tangential force `T` stays within `mu N`. Inside that
//! bound the surfaces stick (no sliding); on it they slide, with `T`
//! opposing the sliding. Written on the sliding velocity at the middle of
//! a time step, friction can then only turn motion into heat, never the
//! reverse (its work per step is `-mu N |slide|`).
//! Source: Acary and Collins-Craft, JTCAM 2025, doi 10.46298/jtcam.13480,
//! eqs. 56-57 and 73.
//!
//! The functions below state both laws as one projection that a solver can
//! iterate on: the forces are the trial values `lambda - k g` (the last
//! multipliers less a penalty times the gaps), clamped onto what the law
//! allows. At the solution the clamp holds exactly the conditions above.
//! Source: Hueber, Stadler and Wohlmuth, SIAM J. Sci. Comput. 30 (2008),
//! eq. 3.14 (the tangential projection) and sec. 7, eq. 7.1 (Coulomb's
//! bound taken from the normal force of the same iterate). Adaptation:
//! written in two dimensions, where the tangent is one direction and the
//! projection onto the disc is a clamp onto `[-mu N, mu N]`.

/// Which branch of the law a contact is on, held fixed while a semismooth
/// Newton method differentiates it: whether the surfaces press, and if so
/// whether they stick (`slip` 0) or slide (`slip` the sign of the
/// tangential force).
/// Source: Hueber, Stadler and Wohlmuth 2008, eq. 3.2 (the generalized
/// derivative of `max`, branch by branch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactBranch {
    pub pressed: bool,
    pub slip: f64,
}

/// The normal and tangential forces of one contact. `multipliers` are the
/// last normal and tangential forces `(lambda_N, lambda_T)`, `gaps` the
/// normal and tangential relative displacements over the step `(g_N,
/// g_T)`, `k` the penalty and `friction` Coulomb's coefficient `mu`.
/// Normal: `N = max(0, lambda_N - k g_N)`. Tangential: `lambda_T - k g_T`
/// clamped onto `[-mu N, mu N]`. With `frozen`, the same with the branch
/// held (each clamp replaced by the side it took), a smooth function of
/// the gaps.
pub fn contact_force(
    (normal_multiplier, tangential_multiplier): (f64, f64),
    (normal_gap, tangential_gap): (f64, f64),
    (k, friction): (f64, f64),
    frozen: Option<ContactBranch>,
) -> (f64, f64) {
    let normal = normal_multiplier - k * normal_gap;
    let trial = tangential_multiplier - k * tangential_gap;
    match frozen {
        None => {
            let normal = normal.max(0.0);
            (normal, trial.clamp(-friction * normal, friction * normal))
        }
        Some(ContactBranch { pressed: false, .. }) => (0.0, 0.0),
        Some(ContactBranch { slip: 0.0, .. }) => (normal, if friction > 0.0 { trial } else { 0.0 }),
        Some(ContactBranch { slip, .. }) => (normal, slip * friction * normal),
    }
}

/// The branch `contact_force` takes at these multipliers and gaps.
pub fn contact_branch(
    (normal_multiplier, tangential_multiplier): (f64, f64),
    (normal_gap, tangential_gap): (f64, f64),
    (k, friction): (f64, f64),
) -> ContactBranch {
    let normal = normal_multiplier - k * normal_gap;
    let trial = tangential_multiplier - k * tangential_gap;
    ContactBranch {
        pressed: normal > 0.0,
        slip: if trial.abs() < friction * normal {
            0.0
        } else {
            trial.signum()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_contact_pushes_but_never_pulls() {
        // Surfaces moving apart over the step (positive gap): no force.
        assert_eq!(
            contact_force((0.0, 0.0), (0.5, 0.0), (10.0, 0.3), None),
            (0.0, 0.0)
        );
        // Moving into each other: a push.
        let (normal, _) = contact_force((0.0, 0.0), (-0.5, 0.0), (10.0, 0.3), None);
        assert_eq!(normal, 5.0);
    }

    #[test]
    fn friction_sticks_inside_the_cone_and_slides_on_it() {
        // A small tangential pull is held in full: sticking.
        let (normal, along) = contact_force((10.0, 0.0), (0.0, 0.1), (10.0, 0.3), None);
        assert_eq!((normal, along), (10.0, -1.0));
        // A large one is capped at mu N, against the slide: sliding.
        let (normal, along) = contact_force((10.0, 0.0), (0.0, 1.0), (10.0, 0.3), None);
        assert_eq!(normal, 10.0);
        assert!((along + 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_frozen_branch_matches_the_law_where_it_was_taken() {
        let cases = [
            ((10.0, 0.0), (0.0, 0.1)),
            ((10.0, 0.0), (0.0, 1.0)),
            ((10.0, 0.0), (0.0, -1.0)),
            ((0.0, 0.0), (0.5, 0.0)),
        ];
        for (multipliers, gaps) in cases {
            let branch = contact_branch(multipliers, gaps, (10.0, 0.3));
            assert_eq!(
                contact_force(multipliers, gaps, (10.0, 0.3), Some(branch)),
                contact_force(multipliers, gaps, (10.0, 0.3), None),
            );
        }
    }

    #[test]
    fn friction_never_does_positive_work() {
        // The tangential force times the slide is never positive, sticking
        // or sliding, whichever way the slide goes.
        for slide in [-2.0, -0.5, -0.01, 0.0, 0.01, 0.5, 2.0] {
            let (_, along) = contact_force((10.0, 0.0), (0.0, slide), (10.0, 0.3), None);
            assert!(along * slide <= 0.0, "slide {slide}: force {along}");
        }
    }
}

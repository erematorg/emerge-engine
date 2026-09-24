//! C¹ smooth cutoff function for pair interactions.
//!
//! Cubic spline force-switch (GROMACS/LAMMPS standard).
//! Ensures forces → 0 smoothly at `r_cut` with no velocity discontinuity.
//! Used by gravity wells and Coulomb fields to limit spatial range.

/// Smooth multiplicative factor S(r) ∈ [0, 1].
///
/// - r ≤ r_on  → 1.0 (full force)
/// - r ≥ r_cut → 0.0 (no force)
/// - between   → cubic blend: S = 1 - 3x² + 2x³, x = (r - r_on) / (r_cut - r_on)
///
/// # Parameters
/// - `r_on`:  start of transition (typically 0.8 × r_cut)
/// - `r_cut`: beyond this distance force is zero
#[inline]
pub fn smooth_cutoff(r: f32, r_on: f32, r_cut: f32) -> f32 {
    if r >= r_cut {
        return 0.0;
    }
    if r <= r_on {
        return 1.0;
    }
    let x = (r - r_on) / (r_cut - r_on);
    1.0 - 3.0 * x * x + 2.0 * x * x * x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutoff_boundary_values() {
        assert_eq!(smooth_cutoff(7.0, 8.0, 10.0), 1.0);
        assert_eq!(smooth_cutoff(10.0, 8.0, 10.0), 0.0);
        assert_eq!(smooth_cutoff(11.0, 8.0, 10.0), 0.0);
    }

    #[test]
    fn cutoff_midpoint_in_range() {
        let v = smooth_cutoff(9.0, 8.0, 10.0);
        assert!(v > 0.0 && v < 1.0, "midpoint should be in (0,1), got {v}");
    }
}

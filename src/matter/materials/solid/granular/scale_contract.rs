//! Real, derived grid-resolution validity for materials with a genuine
//! discrete microstructure (grains) -- replaces guessing `dx_meters` "by
//! feel" with a citable Representative Elementary Volume (REV) argument.
//!
//! For a continuum discretization of scale `dx` to be physically meaningful
//! for a material with microscale `d` (e.g. a grain diameter), real
//! continuum mechanics requires `d << dx << L_macro`, where `L_macro` is the
//! smallest macroscopic feature the scene needs to resolve (Wikipedia,
//! "Representative elementary volume"; Kanatani & Chen, arXiv:cond-mat/0506385,
//! "Fluctuations, Correlation and Representative Elementary Volume in
//! Granular Materials"). The real multiple of `d` needed for a valid REV is
//! regime-dependent -- a few grain diameters at high inertial number (fast
//! flow), several dozen at low inertial number (quasi-static, the regime
//! repose-angle physics lives in) -- so the constant below is picked from
//! the conservative (quasi-static) end of that cited range, not the loosest.
//!
//! This is deliberately NOT a universal formula. It only applies to
//! materials with a real, discrete microstructure (grains). Fluids and
//! homogeneous elastic solids need a *different*-sourced lower bound (e.g. a
//! capillary length or the smallest resolvable surface wave for a fluid,
//! grounded in `surface_tension_coeff`), not this same formula with
//! different constants plugged in -- forcing it universally would produce a
//! physically meaningless number for those materials.

/// Minimum grain-diameter multiple for a valid Representative Elementary
/// Volume in the slow/quasi-static flow regime -- real, cited range is
/// ~10-50x; picked from the conservative half of that range since the point
/// is genuine validity, not the loosest number that lets everything through.
pub const MIN_REV_GRAIN_MULTIPLE: f32 = 30.0;

/// Minimum grid cells needed across the smallest macro feature of interest
/// for its shape to be resolved without obvious blockiness -- a standard
/// numerical-resolution convention, not derived from grain physics.
pub const MIN_CELLS_ACROSS_FEATURE: f32 = 20.0;

/// The real, derived valid `dx` (grid cell size, meters) window for a
/// granular material with the given real grain diameter, given the smallest
/// macro feature (meters) the scene needs to resolve faithfully.
///
/// Returns `None` when no valid window exists at all -- the macro feature is
/// too small relative to the grain for any `dx` to satisfy both constraints
/// simultaneously. That is a genuine, real answer ("this material cannot be
/// a valid continuum at this scene scale"), not a case to silently ignore.
pub fn granular_dx_window(grain_diameter_m: f32, macro_feature_m: f32) -> Option<(f32, f32)> {
    let lo = grain_diameter_m * MIN_REV_GRAIN_MULTIPLE;
    let hi = macro_feature_m / MIN_CELLS_ACROSS_FEATURE;
    if lo <= hi { Some((lo, hi)) } else { None }
}

/// Real, informational (non-panicking) check: is `dx_meters` inside the
/// valid REV window for this grain diameter and macro feature size?
///
/// Callers decide what to do with a `false` result -- this never blocks or
/// panics on an existing scene; it only reports the real physical validity
/// of the resolution actually chosen.
pub fn dx_in_valid_granular_range(
    dx_meters: f32,
    grain_diameter_m: f32,
    macro_feature_m: f32,
) -> bool {
    match granular_dx_window(grain_diameter_m, macro_feature_m) {
        Some((lo, hi)) => dx_meters >= lo && dx_meters <= hi,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Worked example: real dry-sand grain diameter (see
    // `DruckerPragerMaterial::GRAIN_DIAMETER_M`) against emerge's own
    // universal `cell_m = 0.01` convention (every CPU example uses this).
    const SAND_GRAIN_M: f32 = 0.3e-3;
    const LP_CELL_M: f32 = 0.01;

    #[test]
    fn lp_cell_size_is_barely_valid_for_a_20cm_or_larger_sand_feature() {
        // lo = 0.3mm * 30 = 9mm; LP's 10mm cell clears it, but only just.
        let (lo, hi) = granular_dx_window(SAND_GRAIN_M, 0.20).unwrap();
        assert!(lo <= LP_CELL_M, "lo={lo} should be <= {LP_CELL_M}");
        assert!(hi >= LP_CELL_M, "hi={hi} should be >= {LP_CELL_M}");
        assert!(dx_in_valid_granular_range(LP_CELL_M, SAND_GRAIN_M, 0.20));
    }

    #[test]
    fn a_marble_sized_pile_of_real_sand_has_no_valid_continuum_window() {
        // A ~1cm pile: even the loosest possible dx would need to be both
        // >= 9mm (REV) and <= 0.5mm (1cm / 20 cells) -- impossible.
        assert!(granular_dx_window(SAND_GRAIN_M, 0.01).is_none());
        assert!(!dx_in_valid_granular_range(LP_CELL_M, SAND_GRAIN_M, 0.01));
    }

    #[test]
    fn window_bounds_are_ordered_when_they_exist() {
        let (lo, hi) = granular_dx_window(SAND_GRAIN_M, 5.0).unwrap();
        assert!(lo < hi);
    }
}

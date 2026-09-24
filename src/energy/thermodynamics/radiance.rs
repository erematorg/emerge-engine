//! Real point-source radiative flux (irradiance) -- a star/luminous body's
//! own inverse-square law, distinct from `transfer::heat_radiation`'s
//! near-field surface-to-surface exchange (a defined view factor between
//! two close bodies) and from `SimConfig::light_dir`'s own uniform,
//! direction-only "sun at infinity" simplification (correct at Earth-
//! surface scale, wrong at solar-system scale where real distances differ
//! by orders of magnitude -- Mercury vs. Neptune is a real ~6000x
//! irradiance difference, not a uniform field).
//!
//! Real, standard technique: this is exactly how any point-source
//! photometric/radiometric field is modeled (inverse-square law from each
//! source, summed) -- the same real O(sources)-per-query shape
//! `GravityWellField` already uses for point-mass gravity.

use glam::Vec2;

use super::transfer::irradiance_at_distance;

/// One or more luminous point sources (stars) -- real irradiance (W/m²)
/// queryable at any position via the real inverse-square law.
///
/// Not a `Field` (force): a real, deliberate distinction -- irradiance is a
/// scalar geometric quantity derivable at any instant from current
/// positions, not something needing per-substep ODE integration the way
/// velocity/acceleration does. A caller queries it on demand (a demo's own
/// display, or a future `StageOp` writing it into `Particle::scalar_field`/
/// feeding a `ThermalDiffusion` source term) -- real, disclosed future
/// work, not built here since no real consumer needs the automatic
/// per-substep form yet.
pub struct RadianceField {
    /// Point sources: `(position in grid coords, luminosity in Watts)`.
    pub sources: Vec<(Vec2, f32)>,
}

impl RadianceField {
    pub const fn new(sources: Vec<(Vec2, f32)>) -> Self {
        Self { sources }
    }

    /// Single stationary luminous source.
    pub fn point(position: Vec2, luminosity_w: f32) -> Self {
        Self::new(vec![(position, luminosity_w)])
    }

    /// Real total irradiance (W/m²) at `position` (grid coords) from every
    /// registered source, summed -- real superposition (radiative flux
    /// adds linearly, no interaction between sources). `dx_meters` converts
    /// the real grid-unit distance to real meters for the inverse-square
    /// law.
    pub fn irradiance_at(&self, position: Vec2, dx_meters: f32) -> f32 {
        self.sources
            .iter()
            .map(|&(src_pos, luminosity_w)| {
                let distance_m = (src_pos - position).length() * dx_meters;
                irradiance_at_distance(luminosity_w, distance_m)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irradiance_field_sums_multiple_sources_linearly() {
        let field = RadianceField::new(vec![
            (Vec2::new(0.0, 0.0), 1.0e10),
            (Vec2::new(10.0, 0.0), 1.0e10),
        ]);
        let solo = RadianceField::point(Vec2::new(0.0, 0.0), 1.0e10);
        let combined = field.irradiance_at(Vec2::new(5.0, 5.0), 1.0);
        let solo_flux = solo.irradiance_at(Vec2::new(5.0, 5.0), 1.0);
        // By symmetry both sources are equidistant from (5,5), so combined
        // must be exactly 2x either one alone -- real superposition, not
        // approximate.
        assert!(
            (combined - 2.0 * solo_flux).abs() < 1.0,
            "combined={combined} solo={solo_flux}"
        );
    }

    #[test]
    fn irradiance_falls_off_as_inverse_square_through_the_field() {
        let field = RadianceField::point(Vec2::ZERO, 1.0e20);
        let f1 = field.irradiance_at(Vec2::new(10.0, 0.0), 1.0);
        let f2 = field.irradiance_at(Vec2::new(20.0, 0.0), 1.0);
        assert!((f1 / f2 - 4.0).abs() < 1e-3, "ratio={}", f1 / f2);
    }

    #[test]
    fn dx_meters_scales_the_real_grid_to_meters_distance_conversion() {
        // Same grid-unit separation, 10x coarser dx -> 10x real distance ->
        // 100x lower flux (inverse square) -- real unit-conversion check,
        // not just "runs without panicking".
        let field = RadianceField::point(Vec2::ZERO, 1.0e20);
        let f_fine = field.irradiance_at(Vec2::new(10.0, 0.0), 1.0);
        let f_coarse = field.irradiance_at(Vec2::new(10.0, 0.0), 10.0);
        assert!(
            (f_fine / f_coarse - 100.0).abs() < 1.0,
            "ratio={}",
            f_fine / f_coarse
        );
    }
}

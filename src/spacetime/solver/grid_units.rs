//! The grid's units against SI: the one declared, invertible scaling the
//! particles' `f32` state is kept in (`SimConfig::grid_mass_unit_kg_per_m`).
//!
//! Lengths are in cells of `dx_meters`, time in seconds, mass per metre of
//! depth in `grid_mass_unit` = `reference_density_kg_m3 * dx_meters^2`
//! (one cell's area at the reference density). Every derived quantity
//! follows from those three, so a value converts the same way for every
//! material:
//!
//! | quantity | grid unit, in SI |
//! |---|---|
//! | length | `dx` m |
//! | area (2D) | `dx^2` m^2 |
//! | velocity | `dx` m/s |
//! | acceleration | `dx` m/s^2 |
//! | mass (per metre of depth) | `M` kg/m |
//! | density | `M / dx^2` kg/m^3 = `reference_density` |
//! | stress, pressure, Lame parameter | `M` Pa (from `m~ a~ = -sum A~ sigma~ grad~N`) |
//! | viscosity | `M` Pa s |
//! | force (per metre of depth) | `M dx` N/m |
//! | energy (per metre of depth) | `M dx^2` J/m |
//! | specific energy | `dx^2` J/kg |
//!
//! Temperature and heat capacity are SI on both sides (kelvin, J/(kg K)).
//! The 2D model is per metre of depth throughout (J/m, kg/m, N/m).

use glam::{DMat2, DVec2, Vec2};

use super::config::SimConfig;

/// The grid's units of one scene, in `f64` (`grid_units` module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GridUnits {
    /// A cell, m.
    pub(crate) length: f64,
    /// The mass unit per metre of depth, kg/m.
    pub(crate) mass: f64,
}

impl GridUnits {
    pub(crate) fn of(config: &SimConfig) -> Self {
        Self {
            length: f64::from(config.dx_meters),
            mass: f64::from(config.grid_mass_unit_kg_per_m()),
        }
    }

    /// A position or a displacement, m.
    pub(crate) fn position(&self, grid: Vec2) -> DVec2 {
        grid.as_dvec2() * self.length
    }

    pub(crate) fn position_to_grid(&self, si: DVec2) -> Vec2 {
        (si / self.length).as_vec2()
    }

    /// A velocity, m/s (an acceleration, m/s^2, alike).
    pub(crate) fn velocity(&self, grid: Vec2) -> DVec2 {
        grid.as_dvec2() * self.length
    }

    pub(crate) fn velocity_to_grid(&self, si: DVec2) -> Vec2 {
        (si / self.length).as_vec2()
    }

    /// A mass per metre of depth, kg/m.
    pub(crate) fn mass(&self, grid: f32) -> f64 {
        f64::from(grid) * self.mass
    }

    /// An area, m^2.
    pub(crate) fn area(&self, grid: f32) -> f64 {
        f64::from(grid) * self.length * self.length
    }

    pub(crate) fn area_to_grid(&self, si: f64) -> f32 {
        (si / (self.length * self.length)) as f32
    }

    /// A density, kg/m^3.
    pub(crate) fn density_to_grid(&self, si: f64) -> f32 {
        (si * self.length * self.length / self.mass) as f32
    }

    /// A stress, a pressure or a Lame parameter, Pa.
    pub(crate) fn stress(&self, grid: f32) -> f64 {
        f64::from(grid) * self.mass
    }

    /// The deformation gradient is a ratio of lengths: the same in both.
    pub(crate) fn deformation(&self, grid: glam::Mat2) -> DMat2 {
        grid.as_dmat2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A particle's state through SI and back keeps its `f32` value, and
    /// the SI values are the physical ones: 1 cm cells, water as the
    /// reference.
    #[test]
    fn a_particle_round_trips_through_si() {
        let config = SimConfig::earth(32, 0.01, 1.0 / 60.0);
        let units = GridUnits::of(&config);
        let (x, v) = (Vec2::new(12.25, 3.5), Vec2::new(-40.0, 7.5));
        assert_eq!(units.position_to_grid(units.position(x)), x);
        assert_eq!(units.velocity_to_grid(units.velocity(v)), v);
        // `dx_meters` is `f32` (0.01 is 0.00999999978): the SI values hold
        // to its precision, 1e-7 relative.
        assert!((units.position(x).x / 0.1225 - 1.0).abs() < 1e-7);
        assert!((units.velocity(v).x / -0.4 - 1.0).abs() < 1e-7);
        let area = 0.25f32;
        assert_eq!(units.area_to_grid(units.area(area)), area);
        // A quarter cell of clay (1700 kg/m^3): 1700 * 0.25e-4 m^2 kg/m.
        let mass = config.density_from_si(1700.0) * area;
        assert!((units.mass(mass) / (1700.0 * 0.25e-4) - 1.0).abs() < 1e-6);
        assert_eq!(
            units.density_to_grid(units.mass(mass) / units.area(area)),
            config.density_from_si(1700.0)
        );
        // 11.25 MPa through the grid's stress unit and back.
        let e = config.stress_from_si(1.125e7);
        assert!((units.stress(e) - 1.125e7).abs() / 1.125e7 < 1e-6);
        let f = glam::Mat2::from_cols(Vec2::new(1.01, 0.002), Vec2::new(-0.003, 0.99));
        assert_eq!(units.deformation(f).as_mat2(), f);
    }
}

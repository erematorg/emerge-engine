//! Gyroscopic body forces: accelerations perpendicular to velocity, which turn
//! a particle's velocity without doing work.
//!
//! In two dimensions two real forces have this form, `dv/dt = omega J v` with
//! `J v = (-v_y, v_x)` (rotation by +90 degrees) and `omega` a rate in rad/s:
//!
//! - **Magnetic force on a charge.** McDonald, "Electrodynamics in 1 and 2
//!   Spatial Dimensions" (Princeton, 2014/2019), eq. 16: in two dimensions the
//!   magnetic field is a scalar `B` (out of plane) and the Lorentz force is
//!   `F = q (E + v_perp B / c)` with `v_perp = (v_y, -v_x)` (Gaussian units).
//!   In SI units, `F = q (E + v x B)` with `B = B_z z`, so the magnetic part is
//!   `a = (q/m) B_z (v_y, -v_x)`: `omega = -(q/m) B_z`.
//! - **Coriolis acceleration in a rotating frame.** Gerkema, Zimmerman, Maas &
//!   van Haren 2008, "Geophysical and astrophysical fluid dynamics beyond the
//!   traditional approximation", Rev. Geophys. 46, RG2004, eqs. 8-10 (f-plane)
//!   and Table 1, with `f = 2 Omega sin(latitude)` and `f~ = 2 Omega cos(latitude)`:
//!   - a vertical east-up slice (x east, y up, no northward motion) keeps
//!     `du/dt = -f~ w`, `dw/dt = +f~ u`: `omega = +f~`. Eastward motion is
//!     pushed up (the Eotvos effect), upward motion is pushed west;
//!   - a horizontal map-view slice (x east, y north, no vertical motion) keeps
//!     `du/dt = +f v`, `dv/dt = -f u`: `omega = -f` (deflection to the right
//!     in the northern hemisphere).
//!
//!   A two-dimensional slice cannot carry the third Coriolis term: in a
//!   vertical slice `f` drives northward motion out of the plane, in a
//!   horizontal slice `f~` drives vertical motion out of it. That term is
//!   dropped, not modelled. The centrifugal acceleration is not added either:
//!   it is part of the measured effective gravity (Gerkema et al. sec. 2,
//!   para. 8: the planet's surface is a level of the geopotential, gravitational
//!   plus centrifugal), so `SimConfig::gravity` should be that measured value.
//!
//! **Why not `v += dt a`.** Advancing `dv/dt = omega J v` with the explicit
//! update multiplies `|v|^2` by `1 + (omega dt)^2` every step: energy created
//! from nothing. The solver applies the Boris rotation instead (Qin, Zhang,
//! Xiao, Liu, Sun & Tang 2013, "Why is Boris algorithm so good?", PPPL-4872,
//! eqs. 5-7 and 10-12): half the ordinary acceleration, the Cayley transform of
//! `omega dt J` (an exact rotation, `|v|` unchanged), the other half. With no
//! gyroscopic field the solver keeps the plain update unchanged.

use std::collections::HashMap;

use glam::Vec2;

use crate::fields::Field;
use crate::particle::Particles;

/// Earth's rotation rate, rad/s: `7.292e-5`, the value tabulated by Gerkema et
/// al. 2008 (Rev. Geophys. 46, RG2004, Table 1 note).
pub const EARTH_ROTATION_RATE_RAD_S: f32 = 7.292e-5;

/// One substep of `dv/dt = a + omega J v` by the Boris scheme (Qin et al. 2013,
/// eqs. 5-7): `v- = v + a dt/2`, rotate `v-` by the Cayley transform of
/// `omega dt J`, `v' = v+ + a dt/2`.
///
/// With `h = omega dt / 2`, the Cayley rotation `(I - hJ)^-1 (I + hJ)` is
/// `((1 - h^2) v + 2h J v) / (1 + h^2)` because `J^2 = -I`: an exact rotation
/// by `2 atan(h)`, so a purely gyroscopic step keeps `|v|` to rounding.
pub(crate) fn boris_step(v: Vec2, acceleration: Vec2, omega: f32, dt: f32) -> Vec2 {
    let half_kick = 0.5 * dt * acceleration;
    let v_minus = v + half_kick;
    let h = 0.5 * omega * dt;
    let j_v = Vec2::new(-v_minus.y, v_minus.x);
    let v_plus = ((1.0 - h * h) * v_minus + 2.0 * h * j_v) / (1.0 + h * h);
    v_plus + half_kick
}

/// A uniform magnetic field `B_z` (tesla, out of the plane) acting on charged
/// matter through the magnetic part of the Lorentz force, `omega = -(q/m) B_z`.
///
/// Charge enters as the specific charge `q/m` (C/kg) of each material, a
/// measured property; materials not in the map are neutral. `(q/m) B_z` is a
/// rate in 1/s whatever the grid's length unit, so no unit conversion is needed.
#[derive(Debug)]
pub struct UniformMagneticField {
    /// Out-of-plane magnetic field, tesla. Positive points out of the page.
    pub b_z_tesla: f32,
    /// `material_id -> q/m` in C/kg.
    pub specific_charge_c_per_kg: HashMap<u32, f32>,
}

impl UniformMagneticField {
    pub const fn new(b_z_tesla: f32, specific_charge_c_per_kg: HashMap<u32, f32>) -> Self {
        Self {
            b_z_tesla,
            specific_charge_c_per_kg,
        }
    }
}

impl Field for UniformMagneticField {
    fn acceleration(&self, _particles: &Particles, _i: usize) -> Vec2 {
        Vec2::ZERO
    }

    fn gyration_rate(&self, particles: &Particles, i: usize) -> f32 {
        self.specific_charge_c_per_kg
            .get(&particles.material_id[i])
            .map_or(0.0, |q_over_m| -q_over_m * self.b_z_tesla)
    }
}

/// Which plane of the rotating planet the two-dimensional scene stands for
/// (see the module doc for which Coriolis term each keeps and which it drops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotatingSlice {
    /// x east, y up: keeps the `f~ = 2 Omega cos(latitude)` terms.
    VerticalEastUp,
    /// x east, y north: keeps the `f = 2 Omega sin(latitude)` terms.
    HorizontalEastNorth,
}

/// The Coriolis acceleration of a frame rotating with a planet, for every
/// particle alike (it acts on all matter, whatever its composition).
#[derive(Debug, Clone, Copy)]
pub struct CoriolisField {
    /// Signed rate `omega` of `dv/dt = omega J v`, rad/s.
    rate_rad_s: f32,
}

impl CoriolisField {
    /// `planet_rotation_rad_s`: the planet's rotation rate `Omega`
    /// (`EARTH_ROTATION_RATE_RAD_S` for the Earth). `latitude_rad`: positive in
    /// the northern hemisphere.
    pub fn new(planet_rotation_rad_s: f32, latitude_rad: f32, slice: RotatingSlice) -> Self {
        let rate_rad_s = match slice {
            RotatingSlice::VerticalEastUp => 2.0 * planet_rotation_rad_s * latitude_rad.cos(),
            RotatingSlice::HorizontalEastNorth => -2.0 * planet_rotation_rad_s * latitude_rad.sin(),
        };
        Self { rate_rad_s }
    }

    /// The signed rate `omega` this field applies, rad/s.
    pub const fn rate_rad_s(&self) -> f32 {
        self.rate_rad_s
    }
}

impl Field for CoriolisField {
    fn acceleration(&self, _particles: &Particles, _i: usize) -> Vec2 {
        Vec2::ZERO
    }

    fn gyration_rate(&self, _particles: &Particles, _i: usize) -> f32 {
        self.rate_rad_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;

    fn one_particle(material_id: u32) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.material_id = material_id;
        Particles::from(vec![p])
    }

    /// A purely gyroscopic step is a rotation: `|v|` is kept to rounding over
    /// 1000 steps at a coarse `omega dt = 0.5`, where the explicit update grows
    /// `|v|^2` by a factor 1.25 every step (3x the speed after 10 steps).
    #[test]
    fn boris_rotation_keeps_speed_where_explicit_update_creates_energy() {
        let (omega, dt) = (5.0, 0.1);
        let mut v = Vec2::new(3.0, -1.0);
        let speed0 = v.length();
        for _ in 0..1000 {
            v = boris_step(v, Vec2::ZERO, omega, dt);
        }
        assert!(
            (v.length() - speed0).abs() < 1e-3 * speed0,
            "|v| drifted from {speed0} to {}",
            v.length()
        );
        let mut v_explicit = Vec2::new(3.0, -1.0);
        for _ in 0..10 {
            v_explicit += dt * omega * Vec2::new(-v_explicit.y, v_explicit.x);
        }
        assert!(v_explicit.length() > 1.5 * speed0);
    }

    /// One Boris step turns `v` by `2 atan(omega dt / 2)`, counterclockwise for
    /// positive `omega`, the exact Cayley angle.
    #[test]
    fn boris_rotation_angle_is_the_cayley_angle() {
        let (omega, dt) = (2.0, 0.3);
        let v = boris_step(Vec2::X, Vec2::ZERO, omega, dt);
        let expected = 2.0 * (0.5 * omega * dt).atan();
        assert!((v.y.atan2(v.x) - expected).abs() < 1e-6);
    }

    /// Without rotation the scheme is the ordinary kick `v + a dt`.
    #[test]
    fn zero_rate_reduces_to_the_plain_kick() {
        let v = boris_step(Vec2::new(1.0, 2.0), Vec2::new(0.5, -4.0), 0.0, 0.2);
        assert!((v - Vec2::new(1.1, 1.2)).length() < 1e-6);
    }

    /// SI Lorentz force `q v x B` with `B = B_z z`: a positive charge moving
    /// along +x in a field out of the page is pushed toward -y.
    #[test]
    fn magnetic_force_direction_matches_q_v_cross_b() {
        let mut map = HashMap::new();
        map.insert(1, 2.0); // q/m = 2 C/kg
        let field = UniformMagneticField::new(0.5, map);
        let particles = one_particle(1);
        let omega = field.gyration_rate(&particles, 0);
        assert!(
            (omega + 1.0).abs() < 1e-6,
            "omega = -(q/m) B = -1, got {omega}"
        );
        let v = boris_step(Vec2::X, Vec2::ZERO, omega, 0.01);
        assert!(
            v.y < 0.0,
            "q v x B points to -y for v = +x, B = +z: got {v}"
        );
        assert_eq!(field.gyration_rate(&one_particle(0), 0), 0.0);
    }

    /// Gerkema et al. 2008, Table 1: in the northern hemisphere eastward motion
    /// gains an upward acceleration (vertical slice) and a southward one
    /// (horizontal slice).
    #[test]
    fn coriolis_signs_match_the_table_of_coriolis_components() {
        let lat = 45.0_f32.to_radians();
        let east = Vec2::X;
        let vertical = CoriolisField::new(
            EARTH_ROTATION_RATE_RAD_S,
            lat,
            RotatingSlice::VerticalEastUp,
        );
        let up = vertical.rate_rad_s() * Vec2::new(-east.y, east.x);
        assert!(
            up.y > 0.0,
            "eastward motion must be pushed up (Eotvos), got {up}"
        );
        let horizontal = CoriolisField::new(
            EARTH_ROTATION_RATE_RAD_S,
            lat,
            RotatingSlice::HorizontalEastNorth,
        );
        let south = horizontal.rate_rad_s() * Vec2::new(-east.y, east.x);
        assert!(
            south.y < 0.0,
            "eastward motion must be pushed south, got {south}"
        );
        let expected = 2.0 * EARTH_ROTATION_RATE_RAD_S * lat.cos();
        assert!((vertical.rate_rad_s() - expected).abs() < 1e-12);
    }
}

//! Sun position for a 2D side-view scene from Earth's rotation (24 h) and its
//! axial tilt around the orbit: the sun direction (a 2D unit vector,
//! horizontal sweep across the sky and elevation above the horizon) as a
//! function of simulated time, varying with the hour and the season.
//!
//! # Scope
//! - Earth's orbit (eccentricity ~0.0167) is taken as circular: eccentricity
//!   shifts solar intensity ~3.4%/1.7% at perihelion/aphelion, not the
//!   elevation angle computed here.
//! - Longitude and time zones are not modelled: local solar time only (hour
//!   angle from local solar noon), as any single-location solar calculator.
//! - Time is compressed by a caller-chosen `seconds_per_day`; the angles use
//!   the real ratios (axial tilt, 365.25-day year).
//!
//! # References
//! - Solar declination: Cooper, P.I. (1969), "The absolute solar radiation".
//!   Standard approximation used throughout solar-energy engineering
//!   (ASHRAE fundamentals; Duffie & Beckman, "Solar Engineering of Thermal
//!   Processes").
//! - Solar elevation from latitude/declination/hour angle: standard spherical
//!   astronomy formula, the same one underlying NOAA's solar position
//!   calculator.
//! - Earth's axial tilt (obliquity of the ecliptic): IAU reference value,
//!   23.439 deg (epoch J2000), rounded to 23.44 deg here.

use glam::Vec2;

/// Earth's axial tilt (obliquity of the ecliptic), IAU J2000 reference value.
pub const EARTH_AXIAL_TILT_DEG: f32 = 23.44;

/// Year length in days: the Julian year, exactly 365.25 days (not the
/// tropical year, which is slightly shorter).
pub const EARTH_YEAR_DAYS: f32 = 365.25;

/// Solar declination (the sun's angle above/below the equatorial plane) for a
/// given day of the year, via Cooper's (1969) approximation:
/// `delta = tilt * sin(360/365 * (284 + n))`, `n` = day of year (1-based,
/// n=1 is Jan 1). Real seasonal cause: this is what actually varies as Earth
/// orbits the sun with a fixed axial tilt, not an arbitrary sine wave -- the
/// 284-day phase offset and 365-day period are Cooper's real fit constants,
/// not tuned for this engine.
pub fn solar_declination_deg(day_of_year: f32) -> f32 {
    let n = day_of_year.rem_euclid(EARTH_YEAR_DAYS);
    EARTH_AXIAL_TILT_DEG * ((360.0 / EARTH_YEAR_DAYS) * (284.0 + n)).to_radians().sin()
}

/// Local solar hour angle in degrees: 0 at local solar noon, negative in the
/// morning, positive in the afternoon, +-180 at solar midnight. Earth rotates
/// 360 deg every 24h, i.e. 15 deg/hour -- the real rotational rate, not tuned.
pub fn hour_angle_deg(hour_of_day: f32) -> f32 {
    (hour_of_day.rem_euclid(24.0) - 12.0) * 15.0
}

/// Solar elevation angle above the local horizon, in degrees (negative =
/// below the horizon, i.e. night). Standard spherical-astronomy relation:
/// `sin(elevation) = sin(lat) sin(dec) + cos(lat) cos(dec) cos(hour_angle)`.
pub fn solar_elevation_deg(latitude_deg: f32, declination_deg: f32, hour_angle_deg: f32) -> f32 {
    let (lat, dec, ha) = (
        latitude_deg.to_radians(),
        declination_deg.to_radians(),
        hour_angle_deg.to_radians(),
    );
    (lat.sin() * dec.sin() + lat.cos() * dec.cos() * ha.cos())
        .clamp(-1.0, 1.0)
        .asin()
        .to_degrees()
}

/// Sun direction for a 2D side-view scene, as a unit vector: `x` = which way
/// across the sky (east/west, from the hour angle's sign), `y` = height above
/// the horizon (from the elevation angle). Combines daily rotation and the
/// tilt-driven seasonal declination.
pub fn sun_direction(latitude_deg: f32, day_of_year: f32, hour_of_day: f32) -> Vec2 {
    let dec = solar_declination_deg(day_of_year);
    let ha = hour_angle_deg(hour_of_day);
    let elevation = solar_elevation_deg(latitude_deg, dec, ha).to_radians();
    let sweep = ha.to_radians().sin();
    Vec2::new(sweep * elevation.cos(), elevation.sin()).normalize_or_zero()
}

/// Converts real simulated time into (day_of_year, hour_of_day) under a
/// caller-chosen time-compression factor -- the one deliberately non-physical
/// knob in this module (real angle math above stays real; only the pace at
/// which time advances is a disclosed convenience, same pattern this crate
/// already uses for compressed day/night demos).
#[derive(Debug, Clone, Copy)]
pub struct OrbitalClock {
    /// How many simulated seconds correspond to one real Earth day (24h of
    /// real rotation). Smaller = faster compressed cycling.
    pub seconds_per_day: f32,
    /// Day of year at `sim_time = 0` (1.0 = Jan 1st), so a scene can start at
    /// a chosen season instead of always at the year's start.
    pub start_day_of_year: f32,
}

impl OrbitalClock {
    pub const fn new(seconds_per_day: f32, start_day_of_year: f32) -> Self {
        Self {
            seconds_per_day,
            start_day_of_year,
        }
    }

    /// Earth day and 365.25-day year; the caller only picks `seconds_per_day`.
    pub const fn with_seconds_per_day(seconds_per_day: f32) -> Self {
        Self::new(seconds_per_day, 1.0)
    }

    pub fn day_of_year(&self, sim_time_seconds: f32) -> f32 {
        self.start_day_of_year + sim_time_seconds / self.seconds_per_day
    }

    pub fn hour_of_day(&self, sim_time_seconds: f32) -> f32 {
        (sim_time_seconds / self.seconds_per_day).rem_euclid(1.0) * 24.0
    }

    /// Sun direction at a given latitude and simulated time.
    pub fn sun_direction(&self, latitude_deg: f32, sim_time_seconds: f32) -> Vec2 {
        sun_direction(
            latitude_deg,
            self.day_of_year(sim_time_seconds),
            self.hour_of_day(sim_time_seconds),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declination_is_near_zero_at_equinox() {
        // Cooper's formula puts the equinoxes near day 80 and day 264, not the
        // astronomical ~79/266: a sine fit, not an ephemeris, within ~1 degree.
        assert!(solar_declination_deg(80.0).abs() < 1.0);
        assert!(solar_declination_deg(264.0).abs() < 1.0);
    }

    #[test]
    fn declination_peaks_near_axial_tilt_at_solstices() {
        // Summer solstice ~day 172 (Northern Hemisphere convention: max
        // positive declination), winter solstice ~day 355 (max negative).
        let summer = solar_declination_deg(172.0);
        let winter = solar_declination_deg(355.0);
        assert!((summer - EARTH_AXIAL_TILT_DEG).abs() < 0.5);
        assert!((winter + EARTH_AXIAL_TILT_DEG).abs() < 0.5);
    }

    #[test]
    fn hour_angle_is_zero_at_local_noon_and_wraps_correctly() {
        assert!((hour_angle_deg(12.0)).abs() < 1.0e-4);
        assert!((hour_angle_deg(0.0) - (-180.0)).abs() < 1.0e-4);
        assert!((hour_angle_deg(18.0) - 90.0).abs() < 1.0e-4);
        // Wraps past midnight the same as 0h.
        assert!((hour_angle_deg(24.0) - hour_angle_deg(0.0)).abs() < 1.0e-4);
    }

    #[test]
    fn equator_at_equinox_noon_sun_is_near_overhead() {
        // At the equator (lat=0) on an equinox (dec~=0) at local solar noon
        // (hour_angle=0), the sun sits almost at the zenith (elevation ~= 90 deg).
        let dec = solar_declination_deg(80.0);
        let elevation = solar_elevation_deg(0.0, dec, 0.0);
        assert!((elevation - 90.0).abs() < 1.0, "elevation={elevation}");
    }

    #[test]
    fn sun_is_below_horizon_at_solar_midnight() {
        let dec = solar_declination_deg(150.0);
        let elevation = solar_elevation_deg(45.0, dec, hour_angle_deg(0.0));
        assert!(
            elevation < 0.0,
            "expected night at solar midnight, got {elevation}"
        );
    }

    #[test]
    fn sun_direction_is_unit_length_and_points_up_near_noon() {
        let dir = sun_direction(45.0, 172.0, 12.0);
        assert!((dir.length() - 1.0).abs() < 1.0e-4);
        // Near local noon at mid-latitude summer, the sun should be well
        // above the horizon (positive y), not below it.
        assert!(dir.y > 0.5, "dir={dir:?}");
    }

    #[test]
    fn sun_direction_sweeps_from_east_to_west_over_a_day() {
        // The horizontal (x) component lies on opposite sides in the morning and
        // afternoon (east and west).
        let morning = sun_direction(45.0, 172.0, 8.0);
        let afternoon = sun_direction(45.0, 172.0, 16.0);
        assert!(morning.x.signum() != afternoon.x.signum() || morning.x == 0.0);
    }

    #[test]
    fn orbital_clock_advances_hour_and_day_from_sim_time() {
        let clock = OrbitalClock::with_seconds_per_day(24.0); // 1 sim second = 1 real hour
        assert!((clock.hour_of_day(0.0) - 0.0).abs() < 1.0e-4);
        assert!((clock.hour_of_day(12.0) - 12.0).abs() < 1.0e-4);
        assert!((clock.day_of_year(24.0) - 2.0).abs() < 1.0e-4);
    }
}

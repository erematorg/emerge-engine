//! Modal synthesis: vibration frequencies and damping from a rod's own
//! material properties (`ea`, `ei`, mass, damping), with no sample fitting.
//!
//! # Why rods
//! Procedural sound without samples needs per-material resonant frequencies,
//! either fitted to recordings or derived from physics. A rod already
//! carries what a beam's natural frequencies depend on: `RodMaterial`'s
//! `ea`/`ei` and the per-point masses in `RodPoints`, so no new parameter or
//! calibration is needed.
//!
//! # Euler-Bernoulli cantilever bending modes
//! Fixed-free (pinned root, free tip), how the rod scenes anchor a rod (grass
//! blade, branch, root: "pin points 0-1, not just point 0"). The angular
//! frequency of bending mode `n` of a uniform beam is:
//!
//!   ω_n = (β_n·L)² · sqrt(EI / (μ·L⁴))
//!
//! with `μ` the mass per length (kg/m), `L` the length (m), and `β_n·L` the
//! roots of the cantilever equation `cos(βL)·cosh(βL) = -1` (Blevins,
//! *Formulas for Natural Frequency and Mode Shape*, 1979, Table 8-1; also in
//! Rao, *Mechanical Vibrations*): 1.8751, 4.6941, 7.8548, 10.9955 for the
//! first four modes, then `(2n-1)·π/2` asymptotically (Blevins).
//!
//! # Damping: one ratio for every mode
//! `RodMaterial::bending_damping` is a single discrete-scale coefficient, not
//! a two-parameter Rayleigh model, and there is no per-mode damping data. It
//! is reduced to one ratio, the fraction of critical damping it represents
//! at the rod's reference discrete scale (mean segment length, mean point
//! mass, through `RodMaterial::critical_damping`'s formula), applied to every
//! mode: a constant modal damping ratio, the standard assumption without
//! per-mode data (Blevins 1979 §2).
//!
//! # Scope
//! Bending modes only (no axial modes; no torsion, 2D rods have no twist).
//! Uniform beam (constant `EI` and `μ`); a strongly non-uniform rod violates
//! it. Frequencies and damping only: exciting the modes and synthesising
//! audio are left to the caller (LP).

use crate::rod::Rod;

/// A single vibrational mode: real frequency (Hz) and dimensionless
/// damping ratio (0 = undamped, 1 = critically damped). Minimal, real
/// data for an oscillator-bank synthesizer -- not an audio sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticMode {
    pub frequency_hz: f32,
    pub damping_ratio: f32,
}

/// Tabulated roots (β_n·L) of the cantilever (fixed-free) Euler-Bernoulli
/// characteristic equation `cos(βL)·cosh(βL) = -1`.
/// Blevins 1979, Table 8-1 / Rao, *Mechanical Vibrations*, Table 8.2.
const CANTILEVER_BETA_L: [f32; 4] = [1.8751, 4.6941, 7.8548, 10.9955];

/// `β_n·L` for mode index `n` (0-based) -- tabulated for the first 4 modes,
/// asymptotic `(2n-1)·π/2` beyond that (Blevins 1979's own noted limit).
fn beta_l(mode_index: usize) -> f32 {
    CANTILEVER_BETA_L
        .get(mode_index)
        .copied()
        .unwrap_or((2.0 * (mode_index as f32 + 1.0) - 1.0) * std::f32::consts::FRAC_PI_2)
}

/// Cantilever bending-mode frequencies and damping for a `Rod`, as a uniform
/// Euler-Bernoulli beam with its `ei`, mass and length. Empty for a
/// degenerate rod (fewer than 2 points, zero length or zero mass).
pub fn cantilever_rod_modes(rod: &Rod, n_modes: usize) -> Vec<AcousticMode> {
    let n = rod.points.len();
    if n < 2 || n_modes == 0 {
        return Vec::new();
    }

    // Length (m) and mass per length (kg/m) from the rod's state:
    // rest_edge_length and mass are SI (see build_straight_rod).
    let length_m: f32 = rod.points.rest_edge_length.iter().sum();
    let total_mass_kg: f32 = rod.points.mass.iter().sum();
    if length_m <= 0.0 || total_mass_kg <= 0.0 {
        return Vec::new();
    }
    let mu = total_mass_kg / length_m; // kg/m

    // Damping-ratio reduction at the rod's own reference discrete scale --
    // reuses RodMaterial::critical_damping's exact bending-term formula.
    let l0_ref = (length_m / (n - 1) as f32).max(1.0e-9);
    let point_mass_ref = total_mass_kg / n as f32;
    let k_bend_gen = rod.material.ei / l0_ref;
    let m_bend_gen = point_mass_ref * l0_ref * l0_ref;
    let bending_critical = 2.0 * (k_bend_gen * m_bend_gen).sqrt();
    let zeta = if bending_critical > 0.0 {
        (rod.material.bending_damping / bending_critical).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (0..n_modes)
        .map(|i| {
            let bl = beta_l(i);
            let omega = (bl * bl) * (rod.material.ei / (mu * length_m.powi(4))).sqrt();
            AcousticMode {
                frequency_hz: omega / (2.0 * std::f32::consts::PI),
                damping_ratio: zeta,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rod::{RodMaterial, build_straight_rod};
    use glam::Vec2;

    /// A steel ruler, 30 cm x 3 cm x 0.5 mm, E = 200 GPa, rho = 7850 kg/m^3 (a
    /// textbook cantilever): A = 1.5e-5 m^2, I = w*t^3/12 = 3.125e-13 m^4,
    /// mu = rho*A = 0.1178 kg/m, f1 = (1.8751^2/(2*pi*L^2)) * sqrt(EI/mu) at
    /// L = 0.3 m.
    #[test]
    fn steel_ruler_fundamental_matches_hand_computed_value() {
        let e = 200.0e9_f32;
        let width = 0.03_f32;
        let thickness = 0.5e-3_f32;
        let rho = 7850.0_f32;
        let length = 0.3_f32;

        let area = width * thickness;
        let i = width.powi(3) * thickness / 12.0;
        let mu = rho * area;
        let ei = e * i;

        // Independent hand computation (not calling the function under test).
        let bl1 = 1.8751_f32;
        let omega1_expected = (bl1 * bl1) * (ei / (mu * length.powi(4))).sqrt();
        let f1_expected = omega1_expected / (2.0 * std::f32::consts::PI);

        let material = RodMaterial::new(e * area, ei, 0.0, 0.0);
        let points = build_straight_rod(Vec2::ZERO, Vec2::new(length, 0.0), 20, mu, 1.0);
        let rod = Rod::new(points, material);

        let modes = cantilever_rod_modes(&rod, 4);
        assert_eq!(modes.len(), 4);
        let rel_err = (modes[0].frequency_hz - f1_expected).abs() / f1_expected;
        assert!(
            rel_err < 1.0e-3,
            "fundamental frequency {:.4} Hz doesn't match independently hand-computed {:.4} Hz \
             (rel_err={rel_err:.6})",
            modes[0].frequency_hz,
            f1_expected
        );
        // Frequencies strictly increasing and positive.
        for w in modes.windows(2) {
            assert!(
                w[1].frequency_hz > w[0].frequency_hz,
                "modes must increase in frequency"
            );
        }
    }

    #[test]
    fn zero_damping_gives_zero_damping_ratio() {
        let material = RodMaterial::new(1.0e4, 1.0, 0.0, 0.0);
        let points = build_straight_rod(Vec2::ZERO, Vec2::new(1.0, 0.0), 10, 0.1, 1.0);
        let rod = Rod::new(points, material);
        let modes = cantilever_rod_modes(&rod, 1);
        assert_eq!(modes[0].damping_ratio, 0.0);
    }

    #[test]
    fn degenerate_rod_returns_empty() {
        let material = RodMaterial::new(1.0, 1.0, 0.0, 0.0);
        let points = build_straight_rod(Vec2::ZERO, Vec2::new(1.0, 0.0), 2, 0.1, 1.0);
        let rod = Rod::new(points, material);
        assert!(cantilever_rod_modes(&rod, 0).is_empty());
    }
}

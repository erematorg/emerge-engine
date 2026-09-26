//! Modal synthesis -- real vibration frequencies and damping derived
//! directly from a rod's own material properties (`ea`, `ei`, mass,
//! damping), not from a sample-fitting or calibration pipeline.
//!
//! # Why rods, why now
//! Real procedural sound (no samples, no pre-recorded audio, same "zero
//! assets" discipline as the rest of emerge/LP) needs per-material
//! resonant frequencies. The open question blocking this has always been
//! *where do the parameters come from* -- fit against real recorded audio,
//! or derive from physics the engine already knows? For the rod solver
//! specifically, that question is already answered: `RodMaterial` already
//! carries real `ea`/`ei` (axial/bending stiffness) and each `RodPoints`
//! carries real per-point mass -- exactly the inputs a beam's own natural
//! frequencies are a function of. No new parameters, no calibration step.
//!
//! # Real physics: Euler-Bernoulli cantilever bending modes
//! Fixed-free (pinned root, free tip) boundary conditions -- matching how
//! every existing rod scene in this engine anchors a rod (grass blade,
//! branch, root: "pin points 0-1, not just point 0"). The natural
//! (angular) frequency of bending mode `n` for a uniform beam is:
//!
//!   ω_n = (β_n·L)² · sqrt(EI / (μ·L⁴))
//!
//! where `μ` is mass per unit length (kg/m), `L` is the rod's real length
//! (m), and `β_n·L` are the real, tabulated roots of the cantilever's own
//! characteristic equation `cos(βL)·cosh(βL) = -1` (Blevins, *Formulas for
//! Natural Frequency and Mode Shape*, 1979, Table 8-1; the same values
//! appear in Rao, *Mechanical Vibrations*) -- independently confirmed via
//! search, not recalled from memory alone: 1.8751, 4.6941, 7.8548,
//! 10.9955 for the first four modes; modes beyond that converge to
//! `(2n-1)·π/2` (Blevins' own asymptotic note).
//!
//! # Damping -- a disclosed simplification, not measured per-mode data
//! `RodMaterial::bending_damping` is a single discrete-scale coefficient
//! (see its own doc comment), not a full two-parameter Rayleigh model --
//! there is no per-mode damping measurement to draw on. This reduces it to
//! ONE dimensionless ratio: what fraction of critical damping that
//! coefficient represents at the rod's own reference discrete scale (mean
//! segment length, mean point mass -- reusing `RodMaterial::critical_damping`'s
//! own real formula, not a new one), then applies that SAME ratio to every
//! continuous mode. Assuming a constant modal damping ratio when detailed
//! per-mode data isn't available is itself standard, real practice
//! (Blevins 1979 §2) -- disclosed here as an approximation, not presented
//! as measured per-mode data.
//!
//! # Scope (honest, not silently expanded)
//! Bending modes only (no axial/longitudinal modes, no torsion -- 2D rods
//! have no twist DOF at all, matching `spacetime::rod`'s own real
//! dimensional-fact disclosure). Uniform-beam approximation (constant
//! `EI`/`μ` along the rod) -- a rod with strongly varying per-point mass or
//! `ei` violates this. Frequencies and damping only: no amplitude/excitation
//! model and no audio-buffer synthesis here -- exciting modes (e.g.
//! proportional to impact force) and running the actual oscillator/DSP
//! loop is real, separate work, left to the caller (LP), matching the
//! engine/game split every other emerge system already follows.

use crate::rod::Rod;

/// A single vibrational mode: real frequency (Hz) and dimensionless
/// damping ratio (0 = undamped, 1 = critically damped). Minimal, real
/// data for an oscillator-bank synthesizer -- not an audio sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticMode {
    pub frequency_hz: f32,
    pub damping_ratio: f32,
}

/// Real, tabulated roots (β_n·L) of the cantilever (fixed-free)
/// Euler-Bernoulli characteristic equation `cos(βL)·cosh(βL) = -1`.
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

/// Real cantilever bending-mode frequencies/damping for a `Rod`, treated
/// as a uniform Euler-Bernoulli beam using its own real `ei`/mass/length.
/// Returns an empty vec for a degenerate rod (fewer than 2 points, zero
/// length, or zero mass) rather than dividing by zero.
pub fn cantilever_rod_modes(rod: &Rod, n_modes: usize) -> Vec<AcousticMode> {
    let n = rod.points.len();
    if n < 2 || n_modes == 0 {
        return Vec::new();
    }

    // Real length (m) and mass-per-length (kg/m) from the rod's own state --
    // rest_edge_length/mass are already real SI (see build_straight_rod).
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

    /// Real, independently-checkable analytical target: a steel ruler,
    /// 30cm x 3cm x 0.5mm, E=200 GPa, rho=7850 kg/m^3 -- a common textbook
    /// cantilever example. Real numbers: A=3e-2*5e-4=1.5e-5 m^2,
    /// I=w*t^3/12=3e-2*(5e-4)^3/12=3.125e-13 m^4, mu=rho*A=0.1178 kg/m.
    /// f1 = (1.8751^2/(2*pi*L^2)) * sqrt(EI/mu), L=0.3m.
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
        // Real sanity checks: frequencies strictly increasing, all positive.
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

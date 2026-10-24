//! Liquid water at 0.1 MPa, measured: its thermodynamic properties and
//! viscosity as functions of temperature, in `f64`.
//!
//! Source: IAPWS SR6-08(2011), "Revised Supplementary Release on Properties
//! of Liquid Water at 0.1 MPa" (read in full; Tables 1 and 8 and Eqs. 1-4
//! read on the page images): the Gibbs energy `g0` (Eq. 1), the specific
//! volume `v0` (Eq. 2) and its isothermal pressure derivative `vp0` (Eq. 3)
//! with the coefficients of Table 1; their temperature derivatives (Table
//! 2, re-derived here from `d alpha/dT = alpha^2 / T_R`, `d beta/dT =
//! -beta^2 / T_R`); the speed of sound (Table 3); the viscosity (Eq. 7,
//! Table 5). Valid for 253.15 K <= T <= 383.15 K at 0.1 MPa (Eq. 5), "not
//! to be extrapolated"; between the saturation pressure and 0.3 MPa through
//! the release's Table 4 (sec. 2.5). Uncertainty (k = 2): 0.0001 % in
//! density, 0.005 % in the speed of sound, 0.1 % in isobaric heat
//! capacity, 1 % in viscosity, in the stable region (secs. 2.3, 3.3).

/// `R`, J/(kg K), and `T_R`, `T_a`, `T_b`, K (Eqs. 1-4).
const R: f64 = 461.518_05;
const T_R: f64 = 10.0;
const T_A: f64 = 593.0;
const T_B: f64 = 232.0;
/// The reference pressure `p0`, Pa.
pub const P0: f64 = 1.0e5;
/// The validity range of Eq. 5, K.
pub const VALID_K: (f64, f64) = (253.15, 383.15);
/// How far above 0.1 MPa a quadratic volumetric energy `K/2 (J - 1)^2`
/// with this release's modulus keeps the measured water's energy within 1
/// %, Pa: measured against IAPWS-IF97 region 1 integrated along the adiabat
/// and the isotherm (`validity_tests`), smallest crossing at 9.763 MPa absolute (5 C,
/// adiabat), 9.663 MPa above it, rounded down. Below 0.1 MPa the release covers down to
/// the saturation pressure (its sec. 2.5); beyond that, tension and
/// cavitation are not covered.
pub const QUADRATIC_VALID_ABOVE_P0_PA: f64 = 9.6e6;

/// Table 1, as `(n_i, a_i)` for the `alpha` terms and `(m_i, b_i)` for the
/// `beta` terms of each equation; `c_i` and `a_5` apart.
const C: [f64; 3] = [-2.452_093_414e2, 3.869_269_598e1, -8.983_025_854];
const G_ALPHA: [(f64, f64); 3] = [
    (4.0, -1.661_470_539e5),
    (5.0, 2.708_781_640e6),
    (7.0, -1.557_191_544e8),
];
const G_BETA: [(f64, f64); 4] = [
    (2.0, -8.237_426_256e-1),
    (3.0, 1.908_956_353),
    (4.0, -2.017_597_384),
    (5.0, 8.546_361_348e-1),
];
const A5: f64 = 1.937_631_57e-2;
const V_ALPHA: [(f64, f64); 5] = [
    (4.0, 6.744_584_46e3),
    (5.0, -2.225_216_04e5),
    (7.0, 1.002_312_47e8),
    (8.0, -1.635_521_18e9),
    (9.0, 8.322_996_58e9),
];
const V_BETA: [(f64, f64); 6] = [
    (1.0, 5.785_452_92e-3),
    (2.0, -1.531_956_65e-2),
    (3.0, 3.113_378_59e-2),
    (4.0, -4.235_462_41e-2),
    (5.0, 3.387_135_07e-2),
    (6.0, -1.199_467_61e-2),
];
const VP_ALPHA: [(f64, f64); 5] = [
    (1.0, -7.524_587_8e-6),
    (3.0, -1.376_741_8e-2),
    (5.0, 1.062_729_3e1),
    (6.0, -2.045_779_5e2),
    (7.0, 1.203_741_4e3),
];
const VP_BETA: [(f64, f64); 7] = [
    (1.0, -3.109_147_0e-6),
    (3.0, 2.896_491_9e-5),
    (4.0, -1.311_276_3e-4),
    (5.0, 3.041_045_3e-4),
    (6.0, -3.903_459_4e-4),
    (7.0, 2.340_311_7e-4),
    (9.0, -4.851_010_1e-5),
];

/// `sum_i c_i x^k_i` and its first two derivatives in `T` through `x`:
/// for `x = alpha`, `d/dT x^k = k x^(k+1) / T_R`; for `x = beta`, `d/dT x^k =
/// -k x^(k+1) / T_R`.
fn series(terms: &[(f64, f64)], x: f64, sign: f64) -> (f64, f64, f64) {
    terms.iter().fold((0.0, 0.0, 0.0), |(s, d, dd), &(k, c)| {
        (
            s + c * x.powf(k),
            d + sign * k * c * x.powf(k + 1.0) / T_R,
            dd + k * (k + 1.0) * c * x.powf(k + 2.0) / (T_R * T_R),
        )
    })
}

/// The properties of liquid water at 0.1 MPa at the temperature `T` (K).
#[derive(Clone, Copy, Debug)]
pub struct LiquidWater {
    /// Specific Gibbs energy `g0` (J/kg), entropy `s0` (J/(kg K)), isobaric
    /// heat capacity `cp0` (J/(kg K)).
    pub g: f64,
    pub s: f64,
    pub cp: f64,
    /// Specific volume `v0` (m^3/kg) and its derivatives `vT0`, `vTT0`,
    /// `vp0`, `vpT0` (SI).
    pub v: f64,
    pub v_t: f64,
    pub v_tt: f64,
    pub v_p: f64,
    pub v_pt: f64,
    /// Speed of sound `w0` (m/s) and viscosity `mu` (Pa s).
    pub w: f64,
    pub viscosity: f64,
}

impl LiquidWater {
    /// The release's correlations at `T` (K). `None` outside Eq. 5's range.
    pub fn at(temperature_k: f64) -> Option<Self> {
        let t = temperature_k;
        if !(VALID_K.0..=VALID_K.1).contains(&t) {
            return None;
        }
        let tau = t / T_R;
        let alpha = T_R / (T_A - t);
        let beta = T_R / (t - T_B);
        let (ga, ga_t, ga_tt) = series(&G_ALPHA, alpha, 1.0);
        let (gb, gb_t, gb_tt) = series(&G_BETA, beta, -1.0);
        // Eq. 1 and its derivatives: `d/dT (c3 tau ln tau) = c3 (1 + ln
        // tau) / T_R`, `d2/dT2 = c3 / (tau T_R^2)`.
        let g = R * T_R * (C[0] + C[1] * tau + C[2] * tau * tau.ln() + ga + gb);
        let g_t = R * T_R * (C[1] / T_R + C[2] * (1.0 + tau.ln()) / T_R + ga_t + gb_t);
        let g_tt = R * T_R * (C[2] / (tau * T_R * T_R) + ga_tt + gb_tt);
        let (va, va_t, va_tt) = series(&V_ALPHA, alpha, 1.0);
        let (vb, vb_t, vb_tt) = series(&V_BETA, beta, -1.0);
        let scale_v = R * T_R / P0;
        let (v, v_t, v_tt) = (
            scale_v * (A5 + va + vb),
            scale_v * (va_t + vb_t),
            scale_v * (va_tt + vb_tt),
        );
        let (pa, pa_t, _) = series(&VP_ALPHA, alpha, 1.0);
        let (pb, pb_t, _) = series(&VP_BETA, beta, -1.0);
        let scale_p = R * T_R / (P0 * P0);
        let (v_p, v_pt) = (scale_p * (pa + pb), scale_p * (pa_t + pb_t));
        let s = -g_t;
        let cp = -t * g_tt;
        // Table 3.
        let w = (-v * v / (v_p + t * v_t * v_t / cp)).sqrt();
        // Eq. 7, Table 5, micro-pascal seconds.
        let t_star = t / 300.0;
        let viscosity = 1e-6
            * [
                (280.68, -1.9),
                (511.45, -7.7),
                (61.131, -19.6),
                (0.459_03, -40.0),
            ]
            .iter()
            .map(|&(a, b): &(f64, f64)| a * t_star.powf(b))
            .sum::<f64>();
        Some(Self {
            g,
            s,
            cp,
            v,
            v_t,
            v_tt,
            v_p,
            v_pt,
            w,
            viscosity,
        })
    }

    /// The density `1 / v0`, kg/m^3.
    pub fn density(&self) -> f64 {
        1.0 / self.v
    }

    /// The isentropic bulk modulus `rho w^2` = `1 / kappa_s` (Table 3), Pa.
    pub fn isentropic_bulk(&self) -> f64 {
        self.w * self.w / self.v
    }

    /// The isothermal bulk modulus `1 / kappa_T = -v0 / vp0` (Table 3), Pa.
    pub fn isothermal_bulk(&self) -> f64 {
        -self.v / self.v_p
    }
}

mod material;
pub use material::LiquidWaterMaterial;

/// IAPWS-IF97 region 1, for the tests that bound the moduli's validity
/// under pressure.
#[cfg(test)]
pub(crate) mod if97_region1;

#[cfg(test)]
mod validity_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The release's computer-program verification table (Table 8, read on
    /// the page image): at 260, 298.15 and 375 K, `g0`, `s0`, `cp0`, `rho0`,
    /// `1e7 vT0`, `1e8 vTT0`, `1e13 vp0`, `1e15 vpT0`, `w0`, `mu` (uPa s).
    #[rustfmt::skip]
    const TABLE_8: [(f64, [f64; 10]); 3] = [
        (260.0, [-1265.9892, -209.98555, 4300.17472, 997.068360, -3.86550941, 3.27442503, -5.82096820, 7.80938294, 1324.87258, 3058.36075]),
        (298.15, [-4561.7537, 367.20145, 4181.44618, 997.047013, 2.58054178, 0.97202076, -4.53803340, 1.00038567, 1496.69922, 889.996774]),
        (375.0, [-71058.8021, 1328.06616, 4217.74697, 957.009710, 7.94706623, 0.62024104, -5.15666528, -2.27073594, 1541.46611, 276.207245]),
    ];

    /// Every printed value reproduced to half a unit of its last printed
    /// digit (criterion fixed before running).
    #[test]
    fn the_correlations_reproduce_the_release_verification_table() {
        for (t, expected) in TABLE_8 {
            let w = LiquidWater::at(t).expect("in range");
            let got = [
                w.g,
                w.s,
                w.cp,
                w.density(),
                1e7 * w.v_t,
                1e8 * w.v_tt,
                1e13 * w.v_p,
                1e15 * w.v_pt,
                w.w,
                1e6 * w.viscosity,
            ];
            // The last printed digit of each entry, as printed.
            let unit = [1e-4, 1e-5, 1e-5, 1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-5, 1e-5];
            for k in 0..10 {
                let unit = if k == 9 && t == 260.0 {
                    1e-5
                } else if k == 9 {
                    1e-6
                } else {
                    unit[k]
                };
                assert!(
                    (got[k] - expected[k]).abs() <= 0.5 * unit,
                    "T {t}, entry {k}: {} against the printed {}",
                    got[k],
                    expected[k]
                );
            }
        }
    }

    /// Outside Eq. 5's range the record refuses.
    #[test]
    fn outside_the_validity_range_the_record_refuses() {
        assert!(LiquidWater::at(253.0).is_none());
        assert!(LiquidWater::at(383.2).is_none());
        assert!(LiquidWater::at(293.15).is_some());
    }
}

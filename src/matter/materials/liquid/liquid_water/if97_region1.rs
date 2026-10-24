//! IAPWS-IF97 region 1 (the liquid), its basic equation, for the tests that
//! bound where `LiquidWater`'s 0.1 MPa moduli hold under pressure.
//!
//! Source: IAPWS R7-97(2012), "Revised Release on the IAPWS Industrial
//! Formulation 1997 for the Thermodynamic Properties of Water and Steam",
//! sec. 5.1 (read; Tables 2-5 read on the page images): the dimensionless
//! Gibbs energy `gamma(pi, tau) = sum n_i (7.1 - pi)^I_i (tau - 1.222)^J_i`
//! (Eq. 7, Table 2), `pi = p / 16.53 MPa`, `tau = 1386 K / T`, `R = 0.461526
//! kJ/(kg K)` (Eq. 1); the properties of Table 3 from the derivatives of
//! Table 4. Valid for 273.15 K <= T <= 623.15 K, `p_s(T) <= p <= 100 MPa`.

/// Table 2: `(I_i, J_i, n_i)`.
#[rustfmt::skip]
const TERMS: [(i32, i32, f64); 34] = [
    (0, -2, 0.146_329_712_131_67), (0, -1, -0.845_481_871_691_14), (0, 0, -0.375_636_036_720_40e1),
    (0, 1, 0.338_551_691_683_85e1), (0, 2, -0.957_919_633_878_72), (0, 3, 0.157_720_385_132_28),
    (0, 4, -0.166_164_171_995_01e-1), (0, 5, 0.812_146_299_835_68e-3), (1, -9, 0.283_190_801_238_04e-3),
    (1, -7, -0.607_063_015_658_74e-3), (1, -1, -0.189_900_682_184_19e-1), (1, 0, -0.325_297_487_705_05e-1),
    (1, 1, -0.218_417_171_754_14e-1), (1, 3, -0.528_383_579_699_30e-4), (2, -3, -0.471_843_210_732_67e-3),
    (2, 0, -0.300_017_807_930_26e-3), (2, 1, 0.476_613_939_069_87e-4), (2, 3, -0.441_418_453_308_46e-5),
    (2, 17, -0.726_949_962_975_94e-15), (3, -4, -0.316_796_448_450_54e-4), (3, 0, -0.282_707_979_853_12e-5),
    (3, 6, -0.852_051_281_201_03e-9), (4, -5, -0.224_252_819_080_00e-5), (4, -2, -0.651_712_228_956_01e-6),
    (4, 10, -0.143_417_299_379_24e-12), (5, -8, -0.405_169_968_601_17e-6), (8, -11, -0.127_343_017_416_41e-8),
    (8, -6, -0.174_248_712_306_34e-9), (21, -29, -0.687_621_312_955_31e-18), (23, -31, 0.144_783_078_285_21e-19),
    (29, -38, 0.263_357_816_627_95e-22), (30, -39, -0.119_476_226_400_71e-22), (31, -40, 0.182_280_945_814_04e-23),
    (32, -41, -0.935_370_872_924_58e-25),
];

/// `R`, J/(kg K) (Eq. 1); the reducing pressure and temperature (Eq. 7).
pub(crate) const R: f64 = 461.526;
const P_STAR: f64 = 16.53e6;
const T_STAR: f64 = 1386.0;

/// The state at `p` (Pa) and `T` (K): specific volume, internal energy,
/// entropy, enthalpy, isobaric heat capacity, speed of sound (SI), the
/// Helmholtz energy `f = g - p v`, and the volume's derivatives `(dv/dp)_T
/// = R T gamma_pipi / p*^2` and `(dv/dT)_p = R (gamma_pi - tau
/// gamma_pitau) / p*` (derived from Table 3's `v = R T gamma_pi / p*`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Region1 {
    pub(crate) v: f64,
    pub(crate) u: f64,
    pub(crate) s: f64,
    pub(crate) h: f64,
    pub(crate) cp: f64,
    pub(crate) w: f64,
    pub(crate) f: f64,
    pub(crate) v_p: f64,
    pub(crate) v_t: f64,
}

/// Table 3 from Table 4's derivatives.
pub(crate) fn region1(p: f64, t: f64) -> Region1 {
    let (pi, tau) = (p / P_STAR, T_STAR / t);
    let (a, b) = (7.1 - pi, tau - 1.222);
    let (mut g, mut g_p, mut g_pp, mut g_t, mut g_tt, mut g_pt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(i, j, n) in &TERMS {
        let (fi, fj) = (f64::from(i), f64::from(j));
        g += n * a.powi(i) * b.powi(j);
        g_p += -n * fi * a.powi(i - 1) * b.powi(j);
        g_pp += n * fi * (fi - 1.0) * a.powi(i - 2) * b.powi(j);
        g_t += n * a.powi(i) * fj * b.powi(j - 1);
        g_tt += n * a.powi(i) * fj * (fj - 1.0) * b.powi(j - 2);
        g_pt += -n * fi * a.powi(i - 1) * fj * b.powi(j - 1);
    }
    let rt = R * t;
    let v = pi * g_p * rt / p;
    let gibbs = g * rt;
    let w2 = rt * g_p * g_p / ((g_p - tau * g_pt).powi(2) / (tau * tau * g_tt) - g_pp);
    Region1 {
        v,
        u: rt * (tau * g_t - pi * g_p),
        s: R * (tau * g_t - g),
        h: rt * tau * g_t,
        cp: -R * tau * tau * g_tt,
        w: w2.sqrt(),
        f: gibbs - p * v,
        v_p: rt * g_pp / (P_STAR * P_STAR),
        v_t: R * (g_p - tau * g_pt) / P_STAR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Table 5 (read on the page image), every printed value to half a unit
    /// of its last printed digit (nine significant digits).
    #[test]
    fn region_1_reproduces_the_release_verification_table() {
        #[rustfmt::skip]
        let table: [(f64, f64, [f64; 6]); 3] = [
            (3e6, 300.0, [0.100_215_168e-2, 0.115_331_273e3, 0.112_324_818e3, 0.392_294_792, 0.417_301_218e1, 0.150_773_921e4]),
            (80e6, 300.0, [0.971_180_894e-3, 0.184_142_828e3, 0.106_448_356e3, 0.368_563_852, 0.401_008_987e1, 0.163_469_054e4]),
            (3e6, 500.0, [0.120_241_800e-2, 0.975_542_239e3, 0.971_934_985e3, 0.258_041_912e1, 0.465_580_682e1, 0.124_071_337e4]),
        ];
        for (p, t, expected) in table {
            let r = region1(p, t);
            // v in m^3/kg; h, u in kJ/kg; s, cp in kJ/(kg K); w in m/s.
            let got = [r.v, r.h / 1e3, r.u / 1e3, r.s / 1e3, r.cp / 1e3, r.w];
            for k in 0..6 {
                let digit = 10f64.powf(expected[k].abs().log10().floor() - 8.0);
                assert!(
                    (got[k] - expected[k]).abs() <= 0.5 * digit,
                    "p {p}, T {t}, entry {k}: {} against the printed {}",
                    got[k],
                    expected[k]
                );
            }
        }
    }
}

//! Where J2 on Hencky stays elastic over deformations known only through a
//! bound on their displacement gradient: the law's own answer to "can this
//! particle yield anywhere in that set?".
//!
//! Derived here (no source states it): take `F = R (I + H)` with `R` a
//! rotation and `|H|_2 <= b < 1`. Hencky depends on `F` only through its
//! stretches, the singular values `lambda_k` of `I + H`, and Weyl's
//! inequality for singular values gives `|lambda_k - 1| <= |H|_2 <= b`, so
//! each in-plane log strain obeys `|ln lambda_k| <= L = -ln(1 - b)` (the
//! compressive side is the larger, `-ln(1 - b) >= ln(1 + b)`). A particle
//! that has never yielded (`Gp = I`, `gp33 = 1`, `alpha = 0`) has the out of
//! plane log strain `e_33 = ln(gp33) / 2 = 0` exactly (plane strain,
//! `F_33 = 1`), so its principal logs are `(e1, e2, 0)` with `|e_k| <= L`,
//! and `|dev e|^2 = e1^2 + e2^2 - (e1 + e2)^2 / 3 <= 2 L^2`, attained at
//! `e1 = -e2 = L`. With `dev tau = 2G dev e`, `sigma_eq = sqrt(3/2) |dev
//! tau| <= 2 sqrt(3) G L`. The law's yield test is `phi = 2G |dev e| -
//! sqrt(2/3) (Y + H alpha) > 0`, i.e. `sigma_eq > Y` at `alpha = 0`: below
//! `2 sqrt(3) G L < Y` no state of the set flows. (With `e_33` free the
//! same box gives `4 G L`, at `(L, -L, -L)`; it is not this law's set, as
//! `gp33 = 1` is required.)

use super::{HenckyJ2, PlasticMetric};

impl HenckyJ2 {
    /// Whether a particle in `state` stays strictly elastic for every
    /// deformation `F = R (I + H)` with `|H|_2 <= b` (module doc): the law is
    /// then exactly Hencky's elasticity there (no return, no heat, the state
    /// unchanged). False for a particle that has yielded (its stress-free
    /// configuration is no longer the reference), for Norton's flow (no
    /// elastic range), and for `b >= 1`.
    pub(crate) fn elastic_within(&self, state: &PlasticMetric, b: f64) -> bool {
        if self.norton.is_some() || *state != PlasticMetric::unstrained() || b.is_nan() || b >= 1.0
        {
            return false;
        }
        2.0 * 3.0f64.sqrt() * self.shear * -(-b).ln_1p() < self.yield_stress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat2, DVec2};

    /// A deterministic generator (xorshift64*), uniform on `[0, 1)`.
    struct Draws(u64);

    impl Draws {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn between(&mut self, low: f64, high: f64) -> f64 {
            low + (high - low) * self.next()
        }
    }

    fn law(yield_stress: f64) -> HenckyJ2 {
        HenckyJ2 {
            shear: 1.0e6,
            bulk: 2.0e6,
            yield_stress,
            hardening: 0.0,
            viscosity: 0.0,
            norton: None,
        }
    }

    /// The law's own `sigma_eq = sqrt(3/2) 2G |dev e|` at `C`, from its
    /// principal logs (`HenckyJ2::logs`, the ones its return reads).
    fn sigma_eq(law: &HenckyJ2, c: DMat2) -> f64 {
        let (e, _, _) = law.logs(c, &PlasticMetric::unstrained());
        let tr = e[0] + e[1] + e[2];
        let dev2: f64 = e.iter().map(|x| (x - tr / 3.0).powi(2)).sum();
        (1.5f64).sqrt() * 2.0 * law.shear * dev2.sqrt()
    }

    /// The constant `2 sqrt 3` dominates the law's `sigma_eq / (G L)` on a
    /// million states with principal logs in the box `|e_k| <= L` (its
    /// corners among them), under random rotations, and is reached: the
    /// bound is sharp.
    #[test]
    fn the_constant_dominates_the_law_over_the_log_box() {
        let j2 = law(1.0);
        let constant = 2.0 * 3.0f64.sqrt();
        let mut draws = Draws(0x9e37_79b9_7f4a_7c15);
        let mut largest = 0.0f64;
        for k in 0..1_000_000 {
            let l = draws.between(1e-6, 0.5);
            let (e1, e2) = match k % 10 {
                0 => (l, -l),
                1 => (-l, l),
                2 => (l, l),
                3 => (-l, -l),
                _ => (draws.between(-l, l), draws.between(-l, l)),
            };
            let q = DMat2::from_angle(draws.between(0.0, std::f64::consts::TAU));
            let c = q
                * DMat2::from_diagonal(DVec2::new((2.0 * e1).exp(), (2.0 * e2).exp()))
                * q.transpose();
            let ratio = sigma_eq(&j2, c) / (j2.shear * l);
            assert!(
                ratio <= constant * (1.0 + 1e-9),
                "ratio {ratio} beyond {constant} at e = ({e1}, {e2}), L = {l}"
            );
            largest = largest.max(ratio);
        }
        assert!(largest >= constant * (1.0 - 1e-9), "sup {largest}");
    }

    /// The whole chain on a million deformations `F = R (I + H)`, `|H|_2 <=
    /// b`: the law's `sigma_eq` never exceeds `2 sqrt(3) G (-ln(1 - b))`, and
    /// wherever `elastic_within` holds the law's step from the reference
    /// returns nothing: no plastic strain, no dissipation, the state
    /// unchanged.
    #[test]
    fn the_bound_dominates_the_law_over_bounded_gradients() {
        let mut draws = Draws(0x2545_f491_4f6c_dd1d);
        let (mut inside, mut outside) = (0u32, 0u32);
        for _ in 0..1_000_000 {
            let b = draws.between(1e-6, 0.5);
            let h = DMat2::from_cols(
                DVec2::new(draws.between(-1.0, 1.0), draws.between(-1.0, 1.0)),
                DVec2::new(draws.between(-1.0, 1.0), draws.between(-1.0, 1.0)),
            );
            // `|h|_2`, the larger singular value: the square root of the
            // larger eigenvalue of `h^T h`.
            let g = h.transpose() * h;
            let (half, det) = (0.5 * (g.x_axis.x + g.y_axis.y), g.determinant());
            let spectral = (half + (half * half - det).max(0.0).sqrt()).sqrt();
            if spectral == 0.0 {
                continue;
            }
            let h = h * (draws.between(0.0, 1.0) * b / spectral);
            let r = DMat2::from_angle(draws.between(0.0, std::f64::consts::TAU));
            let f = r * (DMat2::IDENTITY + h);
            let c = f.transpose() * f;
            let bound = 2.0 * 3.0f64.sqrt() * 1.0e6 * -(-b).ln_1p();
            let sigma = sigma_eq(&law(1.0), c);
            assert!(
                sigma <= bound * (1.0 + 1e-9),
                "sigma {sigma} beyond {bound} at b {b}"
            );
            // A yield stress around the bound, so both answers occur.
            let j2 = law(bound * draws.between(0.5, 1.5));
            if j2.elastic_within(&PlasticMetric::unstrained(), b) {
                inside += 1;
                let stepped = j2.step(DMat2::IDENTITY, c, &PlasticMetric::unstrained(), 1e-3);
                assert_eq!(stepped.state, PlasticMetric::unstrained());
                assert_eq!(stepped.dissipated, 0.0);
            } else {
                outside += 1;
            }
        }
        assert!(
            inside > 100_000 && outside > 100_000,
            "{inside} / {outside}"
        );
    }

    /// Refused: a yielded particle, Norton's flow, and `b >= 1`.
    #[test]
    fn yielded_norton_and_large_gradients_are_refused() {
        let j2 = law(1.0e9);
        let unstrained = PlasticMetric::unstrained();
        assert!(j2.elastic_within(&unstrained, 1e-3));
        let yielded = PlasticMetric {
            alpha: 1e-12,
            ..unstrained
        };
        assert!(!j2.elastic_within(&yielded, 1e-3));
        let squeezed = PlasticMetric {
            gp33: 1.0 + 1e-12,
            ..unstrained
        };
        assert!(!j2.elastic_within(&squeezed, 1e-3));
        assert!(!j2.elastic_within(&unstrained, 1.0));
        assert!(!j2.elastic_within(&unstrained, f64::NAN));
        let norton = HenckyJ2 {
            yield_stress: 0.0,
            norton: Some(crate::matter::materials::Norton {
                k: 6.37e5,
                m: 0.266,
                rate_0: 1.0,
                valid_rates: (4e-4, 80.0),
            }),
            ..j2
        };
        assert!(!norton.elastic_within(&unstrained, 1e-3));
    }
}

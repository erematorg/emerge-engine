use super::*;

fn law(hardening: f64, viscosity: f64) -> HenckyJ2 {
    HenckyJ2 {
        shear: 1.0e6,
        bulk: 2.0e6,
        yield_stress: 1.0e4,
        hardening,
        viscosity,
        norton: None,
    }
}

/// In-plane pure shear in principal axes, `F = diag(lambda, 1/lambda,
/// 1)`, `e = ln lambda`, driven in `steps` equal increments to `e_end`;
/// returns, per step, `e`, `tau_11 = F S F^T`, `alpha`.
fn pure_shear(law: &HenckyJ2, e_end: f64, steps: usize) -> Vec<(f64, f64, f64)> {
    let mut state = PlasticMetric::unstrained();
    let de = e_end / steps as f64;
    let c_at = |e: f64| DMat2::from_diagonal(DVec2::new((2.0 * e).exp(), (-2.0 * e).exp()));
    (1..=steps)
        .map(|k| {
            let (e0, e1) = ((k - 1) as f64 * de, k as f64 * de);
            state = law.step(c_at(e0), c_at(e1), &state, 1e-3).state;
            let (s, _) = law.second_pk_at(c_at(e1), &state);
            (e1, (2.0 * e1).exp() * s.x_axis.x, state.alpha)
        })
        .collect()
}

/// (b1) Past yield, `tau_11 = (Y + H alpha) / sqrt 3` and `alpha =
/// (2 G e - Y / sqrt 3) / (sqrt 3 G + H / sqrt 3)` exactly (the return
/// is exact along a fixed flow direction), whatever the increments.
#[test]
fn pure_shear_follows_the_exact_hardening_plateau() {
    let law = law(2.0e5, 0.0);
    let (g, y, h) = (law.shear, law.yield_stress, law.hardening);
    let s3 = 3f64.sqrt();
    let onset = y / (2.0 * s3 * g);
    for steps in [7, 50] {
        for (e, tau, alpha) in pure_shear(&law, 4.0 * onset, steps) {
            if e <= onset {
                assert!(
                    (tau - 2.0 * g * e).abs() <= 1e-9 * tau.abs(),
                    "elastic {tau}"
                );
                assert_eq!(alpha, 0.0);
                continue;
            }
            let exact_alpha = (2.0 * g * e - y / s3) / (s3 * g + h / s3);
            assert!(
                (alpha / exact_alpha - 1.0).abs() < 1e-9,
                "alpha {alpha} vs {exact_alpha}"
            );
            let plateau = (y + h * alpha) / s3;
            assert!((tau / plateau - 1.0).abs() < 1e-9, "tau {tau} vs {plateau}");
        }
    }
}

/// (b2) Without hardening the plateau is flat, `Y / sqrt 3`.
#[test]
fn perfect_plasticity_has_a_flat_plateau() {
    let law = law(0.0, 0.0);
    let s3 = 3f64.sqrt();
    let onset = law.yield_stress / (2.0 * s3 * law.shear);
    let last = *pure_shear(&law, 5.0 * onset, 20).last().unwrap();
    assert!((last.1 / (law.yield_stress / s3) - 1.0).abs() < 1e-12);
}

/// (b3) Viscoplastic (eq. 35, 73): after the step the overstress is
/// `eta Delta-gamma / dt`, `|dev tau| - sqrt(2/3)(Y + H alpha) = eta
/// Delta-gamma / dt`, and the dissipation is eq. 74's.
#[test]
fn viscoplastic_overstress_is_eta_times_the_rate() {
    let law = law(1.0e5, 50.0);
    let dt = 1e-3;
    let r = 0.03f64.exp();
    let c1 = DMat2::from_diagonal(DVec2::new(r * r, 1.0 / (r * r)));
    let stepped = law.step(DMat2::IDENTITY, c1, &PlasticMetric::unstrained(), dt);
    let delta_gamma = stepped.state.alpha / (2.0f64 / 3.0).sqrt();
    let (s, tau33) = law.second_pk_at(c1, &stepped.state);
    let f = DMat2::from_diagonal(DVec2::new(r, 1.0 / r));
    let tau = f * s * f.transpose();
    let mean = (tau.x_axis.x + tau.y_axis.y + tau33) / 3.0;
    let dev = ((tau.x_axis.x - mean).powi(2)
        + (tau.y_axis.y - mean).powi(2)
        + (tau33 - mean).powi(2)
        + 2.0 * tau.x_axis.y.powi(2))
    .sqrt();
    let over =
        dev - (2.0f64 / 3.0).sqrt() * (law.yield_stress + law.hardening * stepped.state.alpha);
    let expected = law.viscosity * delta_gamma / dt;
    assert!(delta_gamma > 0.0);
    assert!((over / expected - 1.0).abs() < 1e-9, "{over} vs {expected}");
    let dissipation =
        delta_gamma * (law.viscosity * delta_gamma / dt + (2.0f64 / 3.0).sqrt() * law.yield_stress);
    assert!((stepped.dissipated / dissipation - 1.0).abs() < 1e-12);
}

/// The total Lagrangian path reads a `VonMisesMaterial` into this law
/// with `Y = sqrt(3/2) yield_stress` and `H = 3/2 H_m`
/// (`total_lagrangian_solids`): in pure shear, where the in-plane and
/// 3D deviators coincide, the plastic strain it accumulates at each
/// applied strain `e` is the explicit path's return's, `gamma = (2
/// sqrt 2 mu e - yield_stress) / (2 mu + H_m)` (its strain deviator's
/// Frobenius norm is `sqrt 2 e`; small strain there, log strain here,
/// the same in pure shear's principal axes).
#[test]
fn the_von_mises_mapping_matches_in_pure_shear() {
    let (mu, frobenius_yield, slope) = (1.0e6, 8.0e3, 1.5e5);
    let law = HenckyJ2 {
        shear: mu,
        bulk: 3.0e6,
        yield_stress: (1.5f64).sqrt() * frobenius_yield,
        hardening: 1.5 * slope,
        viscosity: 0.0,
        norton: None,
    };
    let onset = frobenius_yield / (2.0 * 2f64.sqrt() * mu);
    for (e, _, alpha) in pure_shear(&law, 6.0 * onset, 40) {
        let gamma = alpha / (2.0f64 / 3.0).sqrt();
        let explicit =
            ((2.0 * 2f64.sqrt() * mu * e - frobenius_yield) / (2.0 * mu + slope)).max(0.0);
        assert!(
            (gamma - explicit).abs() <= 1e-9 * explicit.max(onset),
            "e {e}: {gamma} vs {explicit}"
        );
    }
}

fn clay_like() -> HenckyJ2 {
    HenckyJ2 {
        shear: 3.9e6,
        bulk: 9.4e6,
        yield_stress: 5.0e4,
        hardening: 1.0e5,
        viscosity: 0.0,
        norton: None,
    }
}

fn some_c(a: f64, b: f64, s: f64) -> DMat2 {
    let f = DMat2::from_cols(DVec2::new(1.0 + a, s), DVec2::new(0.3 * s, 1.0 + b));
    f.transpose() * f
}

/// The step's energy change, computed from the increment, is the
/// difference of the two states' energies (to round-off of the
/// energies, at increments where that difference is well resolved),
/// elastic and plastic alike.
#[test]
fn the_step_energy_change_is_the_energy_difference() {
    let law = clay_like();
    let start = PlasticMetric {
        gp: DMat2::IDENTITY,
        gp33: 1.0,
        alpha: 0.0,
    };
    for (c0, c1) in [
        (some_c(0.0, 0.0, 0.0), some_c(1e-3, -5e-4, 2e-4)),
        (some_c(1e-3, -5e-4, 2e-4), some_c(4e-2, -3e-2, 1e-2)),
    ] {
        let stepped = law.step(c0, c1, &start, 1e-3);
        let difference = law.energy_at(c1, &stepped.state) - law.energy_at(c0, &start);
        let scale = law
            .energy_at(c1, &stepped.state)
            .abs()
            .max(law.energy_at(c0, &start).abs());
        assert!(
            (stepped.energy_change - difference).abs() <= 1e-12 * scale,
            "{} against {difference}",
            stepped.energy_change
        );
    }
}

/// The second Piola-Kirchhoff stress is twice the energy's derivative in
/// `C` (central differences), in a plastically deformed state.
#[test]
fn the_second_piola_kirchhoff_stress_is_the_energy_gradient() {
    let law = clay_like();
    let start = PlasticMetric {
        gp: DMat2::IDENTITY,
        gp33: 1.0,
        alpha: 0.0,
    };
    let deformed = law
        .step(
            some_c(0.0, 0.0, 0.0),
            some_c(5e-2, -4e-2, 1e-2),
            &start,
            1e-3,
        )
        .state;
    let c = some_c(3e-2, -1e-2, 5e-3);
    let (s, _) = law.second_pk_at(c, &deformed);
    let h = 1e-7;
    for (i, j) in [(0usize, 0usize), (1, 1), (0, 1)] {
        let mut e = DMat2::ZERO;
        e.col_mut(j)[i] += 0.5;
        e.col_mut(i)[j] += 0.5;
        let d =
            (law.energy_at(c + h * e, &deformed) - law.energy_at(c - h * e, &deformed)) / (2.0 * h);
        let contracted = (s * e).x_axis.x + (s * e).y_axis.y;
        assert!(
            (2.0 * d - contracted).abs()
                <= 1e-6
                    * s.x_axis
                        .abs()
                        .max_element()
                        .max(s.y_axis.abs().max_element()),
            "{i}{j}: {} vs {contracted}",
            2.0 * d
        );
    }
}

/// The metric the step returns gives back the returned state's logs: its
/// energy at `C1` is the trial's less the return's change, and the yield
/// function there is on the hardened surface (pure shear past yield).
#[test]
fn the_returned_metric_lies_on_the_yield_surface() {
    let law = clay_like();
    let start = PlasticMetric {
        gp: DMat2::IDENTITY,
        gp33: 1.0,
        alpha: 0.0,
    };
    let c1 = some_c(5e-2, -5e-2, 0.0);
    let stepped = law.step(DMat2::IDENTITY, c1, &start, 1e-3);
    assert!(stepped.dissipated > 0.0);
    let (s, tau33) = law.second_pk_at(c1, &stepped.state);
    // Back to tau: F = diag in this pure stretch, tau = F S F^T.
    let f = DMat2::from_diagonal(DVec2::new(c1.x_axis.x.sqrt(), c1.y_axis.y.sqrt()));
    let tau = f * s * f.transpose();
    let mean = (tau.x_axis.x + tau.y_axis.y + tau33) / 3.0;
    let dev =
        ((tau.x_axis.x - mean).powi(2) + (tau.y_axis.y - mean).powi(2) + (tau33 - mean).powi(2))
            .sqrt();
    let radius = (2.0f64 / 3.0).sqrt() * (law.yield_stress + law.hardening * stepped.state.alpha);
    assert!((dev / radius - 1.0).abs() < 1e-9, "{dev} vs {radius}");
}

/// The law alone along the plastic slab's path (`tests/plastic_slab.rs`):
/// monotone uniaxial strain, `F = diag(1, 1 - eps)`, `e = ln(1 - eps)`.
/// Hencky in log strains gives exactly `tau_yy = (K + 4G/3) e` up to
/// the yield (`|e| = Y / (2G)`), then, the deviator saturated along the
/// same direction, `tau_yy = K e - 2Y/3` (no hardening): the slab's
/// closed form, checked on the law with no solver. The slab's moduli:
/// `E = 1 MPa`, `nu = 0.3`, `Y = 300 Pa`.
#[test]
fn uniaxial_strain_follows_the_slab_closed_form() {
    let (e_pa, nu) = (1.0e6, 0.3);
    let law = HenckyJ2 {
        shear: e_pa / (2.0 * (1.0 + nu)),
        bulk: e_pa / (3.0 * (1.0 - 2.0 * nu)),
        yield_stress: 300.0,
        hardening: 0.0,
        viscosity: 0.0,
        norton: None,
    };
    let onset = law.yield_stress / (2.0 * law.shear);
    let mut state = PlasticMetric::unstrained();
    let c_at = |e: f64| DMat2::from_diagonal(DVec2::new(1.0, (2.0 * e).exp()));
    let steps = 60;
    for k in 1..=steps {
        let (e0, e1) = (
            -3.0 * onset * (k - 1) as f64 / steps as f64,
            -3.0 * onset * k as f64 / steps as f64,
        );
        state = law.step(c_at(e0), c_at(e1), &state, 1e-3).state;
        let (s, _) = law.second_pk_at(c_at(e1), &state);
        let tau = (2.0 * e1).exp() * s.y_axis.y;
        let modulus = law.bulk + 4.0 * law.shear / 3.0;
        let exact = if -e1 <= onset {
            modulus * e1
        } else {
            law.bulk * e1 - 2.0 * law.yield_stress / 3.0
        };
        assert!(
            (tau - exact).abs() <= 1e-9 * exact.abs(),
            "e {e1}: {tau} vs {exact}"
        );
    }
}

/// Criterion (f) of the law's adoption: with `Y` infinite and no
/// viscosity the law is elastic Hencky. Along a sheared and rotated
/// path the state never moves, nothing is dissipated, the steps' energy
/// changes sum to the energy at the end, and that energy is `G |dev
/// e|^2 + K/2 (tr e)^2` on `e_A = 1/2 ln` of `C`'s own eigenvalues (the
/// symmetric 2 x 2 closed form, `e_3 = 0`), computed here apart.
#[test]
fn without_yield_the_law_is_elastic_hencky() {
    let law = HenckyJ2 {
        shear: 1.0e6,
        bulk: 2.0e6,
        yield_stress: f64::INFINITY,
        hardening: 0.0,
        viscosity: 0.0,
        norton: None,
    };
    let (sin, cos) = 0.3f64.sin_cos();
    let rotation = DMat2::from_cols(DVec2::new(cos, sin), DVec2::new(-sin, cos));
    let target = DMat2::from_cols(DVec2::new(1.02, 0.01), DVec2::new(0.03, 0.97));
    let c_at = |t: f64| {
        let f = rotation * (DMat2::IDENTITY + (target - DMat2::IDENTITY) * t);
        f.transpose() * f
    };
    let hencky = |c: DMat2| {
        let half = 0.5 * (c.x_axis.x + c.y_axis.y);
        let r = (0.25 * (c.x_axis.x - c.y_axis.y).powi(2) + c.x_axis.y * c.y_axis.x).sqrt();
        let e = [0.5 * (half + r).ln(), 0.5 * (half - r).ln(), 0.0];
        let tr = e[0] + e[1] + e[2];
        let dev2 = e.iter().map(|x| (x - tr / 3.0).powi(2)).sum::<f64>();
        law.shear * dev2 + 0.5 * law.bulk * tr * tr
    };
    let unstrained = PlasticMetric::unstrained();
    let mut state = unstrained;
    let mut total = 0.0;
    let steps = 10;
    for k in 1..=steps {
        let (c0, c1) = (
            c_at((k - 1) as f64 / steps as f64),
            c_at(k as f64 / steps as f64),
        );
        let stepped = law.step(c0, c1, &state, 1e-3);
        assert_eq!(stepped.state, unstrained);
        assert_eq!(stepped.dissipated, 0.0);
        total += stepped.energy_change;
        state = stepped.state;
        let exact = hencky(c1);
        assert!(
            (law.energy_at(c1, &state) - exact).abs() <= 1e-12 * exact,
            "energy {} vs {exact}",
            law.energy_at(c1, &state)
        );
    }
    let end = hencky(c_at(1.0));
    assert!((total - end).abs() <= 1e-12 * end, "{total} vs {end}");
}

/// Norton's law on SI14's technical plasticine (`K` 6.37e5 Pa, `m`
/// 0.266, `rate_0` 1/s; `E` 30 MPa, `nu` 0.3 for the shear modulus):
/// test data from the record's source, the record itself comes later.
fn norton(m: f64) -> HenckyJ2 {
    let (e, nu) = (30.0e6, 0.3);
    let shear = e / (2.0 * (1.0 + nu));
    HenckyJ2 {
        shear,
        bulk: e / (3.0 * (1.0 - 2.0 * nu)),
        yield_stress: 0.0,
        hardening: 0.0,
        viscosity: 0.0,
        norton: Some(Norton {
            k: 6.37e5,
            m,
            rate_0: 1.0,
            valid_rates: (4e-4, 80.0),
        }),
    }
}

/// `sigma_eq = sqrt(3/2) |dev tau|` of the state at `C`.
fn sigma_eq(law: &HenckyJ2, c: DMat2, state: &PlasticMetric) -> f64 {
    let (e, _, _) = law.logs(c, state);
    let tr = e[0] + e[1] + e[2];
    let dev2: f64 = e.iter().map(|x| (x - tr / 3.0).powi(2)).sum();
    (1.5f64).sqrt() * 2.0 * law.shear * dev2.sqrt()
}

/// In-plane pure shear at the log rate `rate`: `C(t) = diag(e^(2 rate
/// t), e^(-2 rate t))`, the equivalent total rate `(2 / sqrt 3) rate`.
fn pure_shear_c(rate: f64, t: f64) -> DMat2 {
    DMat2::from_diagonal(DVec2::new((2.0 * rate * t).exp(), (-2.0 * rate * t).exp()))
}

/// The stepped pure shear: `sigma_eq` after each of `steps` steps of
/// `dt`, at log rate `rate` while `t < hold`, then the strain held.
fn norton_run(law: &HenckyJ2, rate: f64, dt: f64, steps: usize, hold: f64) -> Vec<f64> {
    let mut state = PlasticMetric::unstrained();
    let strain_at = |t: f64| t.min(hold);
    (0..steps)
        .map(|k| {
            let (t0, t1) = (k as f64 * dt, (k + 1) as f64 * dt);
            let (c0, c1) = (
                pure_shear_c(rate, strain_at(t0)),
                pure_shear_c(rate, strain_at(t1)),
            );
            state = law.step(c0, c1, &state, dt).state;
            sigma_eq(law, c1, &state)
        })
        .collect()
}

/// Gate T4: at a constant log rate, steady flow is the backward-Euler
/// fixed point `d = rate_eq dt`, whose stress is `K (rate_eq /
/// rate_0)^m` exactly for ANY `dt`: to 1e-10 at dt 1e-6, 1e-3 and 1 s
/// (criterion fixed before measuring).
#[test]
fn norton_steady_flow_is_exact_at_any_step() {
    let law = norton(0.266);
    let rate = 1.0;
    let exact = 6.37e5 * (2.0 / 3.0f64.sqrt() * rate).powf(0.266);
    for (dt, steps) in [(1e-6, 1_000_000), (1e-3, 2000), (1.0, 12)] {
        let run = norton_run(&law, rate, dt, steps, f64::INFINITY);
        let last = *run.last().expect("steps");
        assert!(
            (last / exact - 1.0).abs() < 1e-10,
            "dt {dt}: {last} against {exact}"
        );
    }
}

/// `2F1(1, m; 1 + m; z) = sum_k m z^k / (m + k)`, `0 <= z < 1`.
fn hypergeometric(m: f64, z: f64) -> f64 {
    let (mut sum, mut power, mut k) = (0.0, 1.0, 0.0);
    while power > 1e-18 {
        sum += m * power / (m + k);
        power *= z;
        k += 1.0;
    }
    sum
}

/// Gates T2 and T5 (criteria fixed before measuring): the transient from
/// rest to `0.9` of the steady stress against its closed form, `t =
/// sigma / (3G r) 2F1(1, m; 1 + m; (sigma / sigma_ss)^(1/m))` (`r` the
/// equivalent rate; derived: `dsigma/dt = 3G r (1 - (sigma /
/// sigma_ss)^(1/m))`), and the relaxation at held strain from the
/// steady stress, `sigma(t) = [sigma_0^(1-n) + (n-1) 3G rate_0 K^-n
/// t]^(-1/(n-1))`, `n = 1/m`, at `t = 1 s`. The backward-Euler flow is
/// first order: each error (relative: the time to 0.9 of the steady
/// stress against its exact 1.86e-2 s, the relaxed stress against its
/// exact value) falls with `dt` at an order between 0.8 and 1.2, and is
/// below 1e-3 at the finest `dt`. Measured: 1.2e-3, 6.0e-4, 3.0e-4 and
/// 5.0e-5, 2.5e-5, 1.3e-5 at dt 4e-5, 2e-5, 1e-5, orders 1.000.
#[test]
fn norton_transient_and_relaxation_follow_their_closed_forms() {
    let m = 0.266;
    let law = norton(m);
    let (k, g, rate) = (6.37e5, law.shear, 1.0);
    let r = 2.0 / 3.0f64.sqrt() * rate;
    let steady = k * r.powf(m);
    let target = 0.9 * steady;
    let t_exact = target / (3.0 * g * r) * hypergeometric(m, (target / steady).powf(1.0 / m));
    let n = 1.0 / m;
    let relaxed_exact =
        (steady.powf(1.0 - n) + (n - 1.0) * 3.0 * g * k.powf(-n) * 1.0).powf(-1.0 / (n - 1.0));
    let mut errors = Vec::new();
    for dt in [4e-5, 2e-5, 1e-5] {
        let ramp = norton_run(&law, rate, dt, (0.05 / dt) as usize, f64::INFINITY);
        let k1 = ramp.iter().position(|&s| s >= target).expect("reached");
        let (s0, s1) = (if k1 > 0 { ramp[k1 - 1] } else { 0.0 }, ramp[k1]);
        let t = (k1 as f64 + (target - s0) / (s1 - s0)) * dt;
        // Held from steady (after 0.5 s of flow) for 1 s.
        let hold = 0.5;
        let held = norton_run(&law, rate, dt, ((hold + 1.0) / dt).round() as usize, hold);
        let relaxed = *held.last().expect("steps");
        errors.push((
            (t / t_exact - 1.0).abs(),
            (relaxed / relaxed_exact - 1.0).abs(),
        ));
        println!(
            "dt {dt:e}: transient t(0.9) {t:.6e} s against {t_exact:.6e}; relaxed {relaxed:.6e} Pa against {relaxed_exact:.6e}"
        );
    }
    for which in 0..2 {
        let e: Vec<f64> = errors
            .iter()
            .map(|p| if which == 0 { p.0 } else { p.1 })
            .collect();
        let orders = [(e[0] / e[1]).log2(), (e[1] / e[2]).log2()];
        println!("errors {e:?}, orders {orders:?}");
        assert!(e[2] < 1e-3, "finest error {}", e[2]);
        for order in orders {
            assert!((0.8..=1.2).contains(&order), "order {order}");
        }
    }
}

/// Gate T8: `m -> 0`. `K (d / (rate_0 dt))^m = K (1 + m ln(d / (rate_0
/// dt)) + O(m^2))`, so Norton's root differs from the rate-independent
/// return with `sigma_eq` yield `K` by `K m |ln(d / (rate_0 dt))| / 3G`
/// to first order (`r` moves by `K m ln(.)`, `|dr/dd| >= 3G`). The bound
/// is that analytic expansion; the gap is the engine's own difference
/// between its two returns (measured 6.757e-8 against the bound 6.757e-8
/// at 1e-6, 1e3 smaller at 1e-9). Asserted
/// within 1.1 of that bound at `m = 1e-6`, and the gap at `m = 1e-9`
/// about 1e3 smaller (between 1e2 and 1e4 smaller). Not to round-off.
#[test]
fn norton_tends_to_rate_independent_j2() {
    let dt = 1e-3;
    let c0 = DMat2::IDENTITY;
    let c1 = pure_shear_c(1.0, 0.05);
    let mut j2 = norton(0.5);
    j2.norton = None;
    j2.yield_stress = 6.37e5;
    let independent = j2
        .step(c0, c1, &PlasticMetric::unstrained(), dt)
        .state
        .alpha;
    let mut gaps = Vec::new();
    for m in [1e-6, 1e-9] {
        let law = norton(m);
        let flowed = law
            .step(c0, c1, &PlasticMetric::unstrained(), dt)
            .state
            .alpha;
        let bound = 6.37e5 * m * (flowed / dt).ln().abs() / (3.0 * law.shear);
        let gap = (flowed - independent).abs();
        println!(
            "m {m:e}: alpha {flowed:.12e} against {independent:.12e}, gap {gap:.3e}, first-order bound {bound:.3e}"
        );
        assert!(gap <= 1.1 * bound, "m {m}: gap {gap:e} beyond {bound:e}");
        gaps.push(gap);
    }
    let shrink = gaps[0] / gaps[1];
    assert!((1e2..=1e4).contains(&shrink), "shrink {shrink}");
}

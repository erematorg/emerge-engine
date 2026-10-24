//! The ringing gates: a free bar, a free square in pure shear and a confined
//! slab, each released in one mode, against the exact Zener mode, stepped and
//! flown by the modes.

use super::*;

#[test]
fn a_zener_bar_rings_down_at_the_exact_rate_and_its_energy_becomes_heat() {
    const FRAMES: usize = 100;
    let dt = f64::from(FRAME_S);
    let free = ring_bar(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    assert_eq!(free.ledger.beyond, 0, "no branches, none reported");
    println!(
        "elastic bar: period {period:.5} s (omega {omega:.3} rad/s), decrement {elastic_decrement:.2e}; {}",
        free.ledger.summary()
    );
    // Plasticine only: with `lambda = 0` the law has no volume part, so a
    // record whose P-wave relaxation differs from its shear's is refused on
    // it (`materials::relaxation`'s tests); the synthetic record is measured
    // where the volume part exists (the uniaxial-strain and shear gates).
    let (name, record) = ("Plasticine", RelaxationSpectrum::plasticine_crandall_1970());
    let damped = ring_bar(Some(record), FRAMES);
    let (period_d, decrement) = period_and_decrement(&damped.samples, dt);
    // With `lambda = 0` only the shear part stores energy: the bar
    // relaxes with the shear weights whatever the P ones.
    let (period_ref, decrement_ref) = reference(&record, |m| m.kappa_shear, omega, FRAMES);
    println!(
        "{name}: damped period {period_d:.5} s against the exact {period_ref:.5} (ratio {:.4}), decrement {decrement:.5} against {decrement_ref:.5} (ratio {:.4}); {}",
        period_d / period_ref,
        decrement / decrement_ref,
        damped.ledger.summary()
    );
    damped.ledger.check(name);
    assert!(
        (period_d / period_ref - 1.0).abs() < 0.05,
        "{name}: period {period_d} against {period_ref}"
    );
    assert!(
        (decrement / decrement_ref - 1.0).abs() < 0.05,
        "{name}: decrement {decrement} against {decrement_ref}"
    );
}

/// The ringing bar flown by its damped modes: Plasticine's proportional
/// record, zero gravity, no wall. Criteria fixed before measuring (the
/// design's gates 1 and 2):
/// - every frame after the first flown (the first builds the contact cache
///   that tells the body is free), no propagator check failing;
/// - its period and decrement within 5 % of the exact Zener mode at the
///   bar's own elastic frequency, and within 2 % of the stepped damped bar;
/// - the ledger: the balance within the solves' bound plus each flown
///   frame's linearisation residual, heat made, none beyond the limit, and
///   no frame's heat negative.
#[test]
fn a_damped_bar_flies_by_its_modes_at_the_exact_rate() {
    const FRAMES: usize = 100;
    let dt = f64::from(FRAME_S);
    let record = RelaxationSpectrum::plasticine_crandall_1970();
    let free = ring_bar(None, FRAMES);
    let (period, _) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    let stepped = ring_bar(Some(record), FRAMES);
    let flying = Rod {
        modal_flight: true,
        ..BAR
    };
    let flown = ring_bar_at(&flying, Some(record), FRAMES);
    let (period_s, decrement_s) = period_and_decrement(&stepped.samples, dt);
    let (period_f, decrement_f) = period_and_decrement(&flown.samples, dt);
    let (period_ref, decrement_ref) = reference(&record, |m| m.kappa_shear, omega, FRAMES);
    println!(
        "damped bar flown: {} of {FRAMES} frames flown, {} refused, {} hand-backs, {} checks failed; period {period_f:.5} s against the exact {period_ref:.5} ({:.4}) and the stepped {period_s:.5} ({:.4}); decrement {decrement_f:.5} against {decrement_ref:.5} ({:.4}) and {decrement_s:.5} ({:.4}); least heat in a frame {:.3e} J/m; {}",
        flown.ledger.flown,
        flown.ledger.refused,
        flown.ledger.hand_backs,
        flown.ledger.check_failed,
        period_f / period_ref,
        period_f / period_s,
        decrement_f / decrement_ref,
        decrement_f / decrement_s,
        flown.ledger.least_heat,
        flown.ledger.summary()
    );
    println!("stepped: {}", stepped.ledger.summary());
    assert_eq!(
        flown.ledger.flown as usize,
        FRAMES - 1,
        "every frame after the first flown"
    );
    assert_eq!(flown.ledger.check_failed, 0, "a propagator check failed");
    flown.ledger.check("damped bar flown");
    assert!(flown.ledger.least_heat >= 0.0, "a frame's heat negative");
    assert!((period_f / period_ref - 1.0).abs() < 0.05);
    assert!((decrement_f / decrement_ref - 1.0).abs() < 0.05);
    assert!((period_f / period_s - 1.0).abs() < 0.02);
    assert!((decrement_f / decrement_s - 1.0).abs() < 0.02);
}

/// The pure-shear scene: a free square of side `SQUARE_CELLS` in zero
/// gravity, `nu = 0.3`, released in its Lame mode, `v = V (cos kx sin ky,
/// -sin kx cos ky)`, `k = pi / L` from its corner. That field is exactly
/// isochoric (`div v = 0`) and leaves every edge free of traction
/// (`sigma_xx = 0` at `x = 0, L`, `sigma_yy = 0` at `y = 0, L`, `sigma_xy =
/// 0` everywhere), so it is a mode at `omega = sqrt(2) pi c_s / L` for any
/// `lambda`, loading the shear part alone.
const SQUARE_CELLS: i32 = 8;
/// The release speed, m/s: a peak shear strain `V / (sqrt(2) c_s)` of
/// about 5e-6 (`c_s` 6.2 m/s), below the record's limit.
const SQUARE_SPEED_M_S: f32 = 4.0e-5;

fn ring_square(relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.3,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(24, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let l = SQUARE_CELLS as f32;
    let corner = Vec2::splat(12.0 - 0.5 * l);
    let _square = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::splat(SQUARE_CELLS))
            .at(corner + Vec2::splat(0.5 * l))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let k = std::f32::consts::PI / l;
    let shape = move |x: Vec2| -> Vec2 {
        let q = x - corner;
        Vec2::new(
            (k * q.x).cos() * (k * q.y).sin(),
            -(k * q.x).sin() * (k * q.y).cos(),
        )
    };
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = shape(x) * (SQUARE_SPEED_M_S / DX_M);
    }
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let s = shape(p.x[i]);
            num += f64::from(p.mass[i]) * f64::from(p.v[i].dot(s));
            den += f64::from(p.mass[i]) * f64::from(s.dot(s));
        }
        num / den
    };
    run(sim, amplitude, frames)
}

/// The uniaxial-strain scene: a slab spanning the domain's frictionless
/// side walls, on its floor, `nu = 0.3`, set down unstrained under gravity.
/// The walls hold its sides (its weight keeps them pressed, `sigma_xx =
/// nu / (1 - nu) sigma_yy`), so it rings in its fixed-free P-wave mode,
/// `omega = pi c_p / (2 H)`, `c_p^2 = (lambda + 2 mu) / rho`, about its
/// settled compression.
const SLAB_GRID: usize = 16;
/// Stiff enough that its weight strains it by 2.4e-6 (peak about twice
/// that while it rings), below the record's limit.
const SLAB_E_PA: f32 = 3.0e8;
/// A twentieth of the mode's period (0.63 ms at 1.6 kHz).
const SLAB_FRAME_S: f32 = 3.0e-5;

fn ring_slab(relaxation: Option<RelaxationSpectrum>, frames: usize) -> (Ring, f64) {
    let props = Elastic {
        e_pa: SLAB_E_PA,
        nu: 0.3,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(SLAB_GRID, DX_M, SLAB_FRAME_S);
    config.min_dt = config.min_dt.min(1e-9);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let wall = config.boundary_thickness as f32;
    let width = SLAB_GRID as f32 - 2.0 * wall;
    let cells = width.round() as i32;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::splat(cells))
            .at(Vec2::splat(wall + 0.5 * cells as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let h = cells as f32;
    let shape = move |x: Vec2| -> f32 { (std::f32::consts::PI * (x.y - wall) / (2.0 * h)).sin() };
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let s = f64::from(shape(p.x[i]));
            num += f64::from(p.mass[i]) * f64::from(p.v[i].y) * s;
            den += f64::from(p.mass[i]) * s * s;
        }
        num / den
    };
    (run(sim, amplitude, frames), f64::from(h * DX_M))
}

/// The exact mode released from rest at its static offset's opposite, the
/// step response under a constant load: `q = q_p + q_h` with `q_p` the
/// relaxed static offset (no memory at rest), so the velocity is the
/// homogeneous part's from `q_h = -q_p`, at rest, no memory.
fn reference_step(
    record: &RelaxationSpectrum,
    kappa: impl Fn(&Mechanism) -> f64,
    omega: f64,
    frames: usize,
    dt: f64,
) -> (f64, f64) {
    let material = zener::Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: record.mechanisms().iter().map(&kappa).collect(),
        theta: record.mechanisms().iter().map(|m| m.theta).collect(),
    };
    let start = zener::ModeState {
        q: -1.0,
        rate: 0.0,
        memory: vec![0.0; record.mechanisms().len()],
    };
    let mode = zener::ModeSolution::new(&material, omega, &start);
    let (root, reproduced) = mode.checks();
    assert!(
        root < 1e-12 && reproduced < 1e-12,
        "reference: roots {root:e}, start {reproduced:e}"
    );
    let samples: Vec<f64> = (0..=frames)
        .map(|i| mode.at(dt * i as f64).0.rate)
        .collect();
    period_and_decrement(&samples, dt)
}

/// One gate's comparison: the damped run's period and decrement against
/// the exact mode's, with the ledger's criteria.
fn compare(name: &str, damped: &Ring, dt: f64, (period_ref, decrement_ref): (f64, f64)) {
    let (period, decrement) = period_and_decrement(&damped.samples, dt);
    println!(
        "{name}: damped period {period:.6e} s against the exact {period_ref:.6e} (ratio {:.4}), decrement {decrement:.5} against {decrement_ref:.5} (ratio {:.4}); {}",
        period / period_ref,
        decrement / decrement_ref,
        damped.ledger.summary()
    );
    damped.ledger.check(name);
    assert!(
        (period / period_ref - 1.0).abs() < 0.05,
        "{name}: period {period} against {period_ref}"
    );
    assert!(
        (decrement / decrement_ref - 1.0).abs() < 0.05,
        "{name}: decrement {decrement} against {decrement_ref}"
    );
}

#[test]
fn a_zener_square_rings_down_in_pure_shear_at_the_exact_shear_rate() {
    const FRAMES: usize = 100;
    let dt = f64::from(FRAME_S);
    let free = ring_square(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    let c_s = (f64::from(E_PA) / (2.0 * 1.3) / f64::from(RHO)).sqrt();
    println!(
        "elastic square: period {period:.5} s (omega {omega:.3} rad/s, the Lame mode's sqrt(2) pi c_s / L {:.3}), decrement {elastic_decrement:.2e}; {}",
        2f64.sqrt() * PI * c_s / (f64::from(SQUARE_CELLS) * f64::from(DX_M)),
        free.ledger.summary()
    );
    for (name, record) in [
        (
            "square, Plasticine",
            RelaxationSpectrum::plasticine_crandall_1970(),
        ),
        (
            "square, synthetic (P weights twice the shear's)",
            synthetic(),
        ),
    ] {
        let damped = ring_square(Some(record), FRAMES);
        compare(
            name,
            &damped,
            dt,
            reference(&record, |m| m.kappa_shear, omega, FRAMES),
        );
    }
}

#[test]
fn a_confined_zener_slab_rings_down_in_uniaxial_strain_at_the_exact_p_wave_rate() {
    const FRAMES: usize = 90;
    let dt = f64::from(SLAB_FRAME_S);
    let (free, h) = ring_slab(None, FRAMES);
    let (period, elastic_decrement) = period_and_decrement(&free.samples, dt);
    let omega = 2.0 * PI / period;
    let nu = 0.3f64;
    let m = f64::from(SLAB_E_PA) * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    println!(
        "elastic slab: period {period:.6e} s (omega {omega:.1} rad/s, the fixed-free P mode's pi c_p / 2H {:.1}), decrement {elastic_decrement:.2e}; {}",
        PI * (m / f64::from(RHO)).sqrt() / (2.0 * h),
        free.ledger.summary()
    );
    for (name, record) in [
        (
            "slab, Plasticine",
            RelaxationSpectrum::plasticine_crandall_1970(),
        ),
        ("slab, synthetic (P weights twice the shear's)", synthetic()),
    ] {
        let (damped, _) = ring_slab(Some(record), FRAMES);
        // The P-wave modulus relaxes with the record's P weights, its parts'
        // weights `(pi kappa^p - 2 mu kappa^s) / lambda` and `kappa^s` summed.
        compare(
            name,
            &damped,
            dt,
            reference_step(&record, |m| m.kappa_p, omega, FRAMES, dt),
        );
    }
}

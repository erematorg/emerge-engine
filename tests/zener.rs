//! The exact 1D generalised Zener solution (`common/zener.rs`) checked
//! against itself and against Blanc et al. 2016, then the engine's solids
//! with relaxation branches measured against it.
//!
//! Two parts, each with its criteria fixed before measuring: the
//! reference's own exactness, and the gates of the path's relaxation
//! branches. The gates are split by concept into `zener_gates/`.

extern crate emerge_engine as emerge;

#[path = "common/zener.rs"]
mod zener;

use std::f64::consts::PI;
use zener::{ModeSolution, ModeState, Zener};

use emerge::materials::physical_props::Elastic;
use emerge::materials::relaxation::{Mechanism, RelaxationSpectrum};
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

// The exact 1D generalised Zener bar (`common/zener.rs`) checked against
// itself and against Blanc et al. 2016, before any engine code uses it.
//
// A free-free bar's mode depends on the material only through `w^2 = E_r
// k^2 / rho` and the mechanisms, so the reference is checked at mode
// frequencies below, inside and above the band the mechanisms were fitted
// over. The mechanisms are the first record: Plasticine, a constant loss
// factor 0.225 over 100-3000 Hz (Crandall, Kurzweil, Nigam and Remington
// 1970) fitted with three mechanisms by Blanc et al.'s eq. 28 (minimax,
// positive weights, their eq. 33): the largest error in the loss factor
// over the band is 0.83 %.
//
// Criteria fixed before measuring:
// (a) every root makes the characteristic polynomial vanish to 1e-12 of
// its terms, the start state is reproduced to 1e-12, and the assembled
// solution satisfies the mode's equations to 1e-12 of their terms at
// every sampled time. (Fixed first as "every eigenvector satisfies `A v
// = p v` to 1e-12": that asks more than a root next to `-theta_l` can
// be held to in f64, 1.8e-12 measured at 30 Hz, so it was replaced by
// the assembled solution's residual, which is the exactness wanted.
// Removal criterion: a root and eigenvector method with a sourced
// conditioning bound, against which the eigenvectors are checked
// instead.)
// (b) Blanc et al.'s energy (Property 1) plus the heat made stays the
// start energy to 1e-12 of it over twenty periods, and the heat never
// decreases;
// (c) the state stays real (imaginary residue below 1e-12 of its size).
//
// Reported, not asserted: each mode's logarithmic decrement against
// `pi Q^-1` at its damped frequency (their eq. 28; equal to first order in
// `Q^-1` only).

/// Plasticine's three mechanisms: weights and relaxation frequencies
/// (rad/s), fitted to Crandall et al.'s loss factor 0.225 over 100-3000 Hz.
fn plasticine(rho: f64, relaxed: f64) -> Zener {
    Zener {
        rho,
        relaxed,
        kappa: vec![0.3807440105974887, 0.3612646181734217, 0.7539982742914667],
        theta: vec![654.8767306216079, 3865.458262006148, 25438.050835431713],
    }
}

#[test]
fn a_zener_mode_rings_down_exactly_and_its_energy_becomes_heat() {
    // A unit density and relaxed modulus: the mode's frequency is then its
    // wavenumber.
    let material = plasticine(1.0, 1.0);
    println!(
        "Plasticine record: unrelaxed over relaxed modulus {:.4}",
        material.unrelaxed() / material.relaxed
    );
    for f_hz in [30.0, 104.0, 530.0, 1500.0, 4049.0, 20000.0] {
        let k = 2.0 * PI * f_hz;
        // Released from rest at zero strain with the velocity of the mode,
        // its memory empty.
        let start = ModeState {
            q: 0.0,
            rate: 1.0,
            memory: vec![0.0; 3],
        };
        let mode = ModeSolution::new(&material, k, &start);
        let (root, reproduced) = mode.checks();
        assert!(
            root < 1e-12 && reproduced < 1e-12,
            "{f_hz} Hz: roots {root:.2e}, start {reproduced:.2e}"
        );
        // The oscillating pair: the root with the largest imaginary part.
        let pair = mode
            .roots
            .iter()
            .copied()
            .max_by(|a, b| a.im.total_cmp(&b.im))
            .expect("roots");
        let (decay, damped) = (-pair.re, pair.im);
        let period = 2.0 * PI / damped;
        let e0 = mode.energy(&start);
        let (mut ledger, mut imaginary, mut last_heat, mut monotone) = (0.0f64, 0.0f64, 0.0, true);
        let mut residual = 0.0f64;
        for i in 0..=2000 {
            let t = 20.0 * period * i as f64 / 2000.0;
            let (state, im) = mode.at(t);
            let heat = mode.heat(t);
            residual = residual.max(mode.residual(t));
            ledger = ledger.max((mode.energy(&state) + heat - e0).abs() / e0);
            // Against the start's velocity amplitude, `sqrt(2 E0)`.
            imaginary = imaginary.max(im / (2.0 * e0).sqrt());
            if heat < last_heat - 1e-15 * e0 {
                monotone = false;
            }
            last_heat = heat;
        }
        let decrement = 2.0 * PI * decay / damped;
        let q_inv = material.inverse_q(damped);
        let frame = mode.energy(&mode.at(1.0 / 60.0).0) / e0;
        println!(
            "{f_hz:>7} Hz: damped {:.4} Hz, decrement {decrement:.5} against pi Q^-1 {:.5} (ratio {:.4}); \
             roots {root:.1e}, start {reproduced:.1e}, equations {residual:.1e}, ledger {ledger:.1e}, imaginary {imaginary:.1e}; \
             energy left after a 1/60 s frame {frame:.3e}",
            damped / (2.0 * PI),
            PI * q_inv,
            decrement / (PI * q_inv)
        );
        assert!(
            ledger < 1e-12,
            "{f_hz} Hz: energy plus heat off the start by {ledger:.2e} of it"
        );
        assert!(monotone, "{f_hz} Hz: the heat decreased");
        assert!(
            residual < 1e-12,
            "{f_hz} Hz: the equations' residual {residual:.2e} of their terms"
        );
        assert!(
            imaginary < 1e-12,
            "{f_hz} Hz: imaginary residue {imaginary:.2e}"
        );
    }
}

/// A Zener rod dropped flat on a rigid wall, with the rod's quarter-wave
/// frequency `c_r / 4L` across the mechanisms' band. Criteria fixed before
/// measuring: without mechanisms the rod leaves at `2L / c` with all its
/// momentum, to the truncation's order (1e-3 at 1024 modes; measured 0.99921,
/// 0.99980, 0.99995 at 256, 1024, 4096); with them, the energy plus the heat
/// stays the truncated series' start energy to 1e-12 at separation. The
/// restitution is reported against the number of modes: the same to 5
/// digits at 256 and 1024 modes up to 1.5 kHz, 4e-4 apart at 4049 Hz,
/// where the rod leaves exactly at the unrelaxed front's return and the
/// separation time converges like the truncation's `1/M`.
///
/// Declared gaps, each with its removal criterion:
/// - a second contact after separation is not followed: the end's flight
///   after separation projected on the free-free modes, with any return to
///   the wall solved as a new contact;
/// - the record's relaxed modulus is a unit here (the mode depends on it
///   only through the quarter-wave frequency): before any scene uses the
///   record, `E_r` is set from Crandall et al.'s measured wave speed at a
///   reference frequency (Blanc et al. eq. 25).
#[test]
fn a_zener_rod_on_a_rigid_wall_leaves_with_the_momentum_its_damping_leaves() {
    let elastic = Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: vec![],
        theta: vec![],
    };
    let material = plasticine(1.0, 1.0);
    for f1 in [104.0, 530.0, 1500.0, 4049.0] {
        // c_r = 1: L = c_r / (4 f1).
        let length = 0.25 / f1;
        for modes in [256, 1024] {
            let free = zener::rod_on_wall(&elastic, length, 1.0, modes);
            let damped = zener::rod_on_wall(&material, length, 1.0, modes);
            println!(
                "quarter-wave {f1:>6} Hz, {modes:>5} modes: elastic leaves at {:.5} of 2L/c with {:.5} of its momentum; \
                 Zener leaves at {:.5} of 2L/c_r, restitution {:.5}, energy rebound {:.4} / vibration {:.4} / heat {:.4}, ledger {:.1e}",
                free.separation / (2.0 * length),
                free.restitution,
                damped.separation / (2.0 * length),
                damped.restitution,
                damped.split.0,
                damped.split.1,
                damped.split.2,
                damped.ledger
            );
            assert!(
                damped.ledger < 1e-12,
                "{f1} Hz, {modes} modes: ledger {:.2e}",
                damped.ledger
            );
            if modes == 1024 {
                assert!(
                    (free.separation / (2.0 * length) - 1.0).abs() < 1e-3
                        && (free.restitution - 1.0).abs() < 1e-3,
                    "{f1} Hz: the elastic rod leaves at {:.5} of 2L/c with {:.5} of its momentum",
                    free.separation / (2.0 * length),
                    free.restitution
                );
            }
        }
    }
}

// A solid with relaxation branches on the total Lagrangian path, measured
// against the exact 1D generalised Zener solution (`common/zener.rs`).
//
// The law: the corotated solid with a measured relaxation spectrum on its
// shear and volume parts (`materials::relaxation`), advanced by Liu and
// Guan's energy-consistent update (`total_lagrangian::relaxation`).
//
// Criteria fixed before measuring:
// (a) a free-free bar with `nu = 0` (`lambda = 0`: exactly uniaxial, only the
// shear part relaxing) released in its first axial mode, below the
// record's strain limit: its damped frequency and logarithmic decrement
// within 5 % of the exact Zener mode's at the bar's own elastic
// frequency (the same bar measured without branches), both read the
// same way from frame samples;
// (b) every frame's energy (kinetic, stored, the branches' included) plus
// the heat made equals the start's to 1e-8 of it, and the branches'
// heat is positive;
// (c) no particle-step beyond the record's strain limit (the run stays in
// its linear range), and none reported when the branches are absent.
//
// Records: Plasticine (Crandall et al. 1970, three mechanisms,
// proportional) and a synthetic one, test data and not a material, with
// its P-wave weights twice its shear weights.
const DX_M: f32 = 0.01;
const E_PA: f32 = 1.0e5;
const RHO: f32 = 1000.0;
const LENGTH_CELLS: i32 = 32;
const FRAME_S: f32 = 2.0e-3;
/// The release speed, m/s: a peak axial strain `v / c` of 5e-6, below
/// Plasticine's linear limit of 1e-5.
const SPEED_M_S: f32 = 5.0e-5;

/// The synthetic record: Plasticine's frequencies, its weights on shear and
/// twice them on P waves. Test data, not a material.
fn synthetic() -> RelaxationSpectrum {
    let plasticine = RelaxationSpectrum::plasticine_crandall_1970();
    let mechanisms: Vec<Mechanism> = plasticine
        .mechanisms()
        .iter()
        .map(|m| Mechanism {
            theta: m.theta,
            kappa_shear: m.kappa_shear,
            kappa_p: 2.0 * m.kappa_shear,
        })
        .collect();
    RelaxationSpectrum::new(&mechanisms, plasticine.strain_limit).expect("positive")
}

/// A run's energy ledger, frame by frame.
///
/// - `worst`: the largest `|balance|` over the start energy (the
///   Kelvin-Voigt gate's criterion, 1e-8, kept);
/// - `tightest`: the largest `|balance|` over what the solves allow, the
///   path suite's derived bound (`tests/total_lagrangian_path.rs`): the
///   frames' residual work and contact work bounds accumulated, plus the
///   energy sums' round-off `n eps (sum |parts| + |heat| + |start|)`. At or
///   below 1, the balance is the solves' own residual;
/// - the branches' heat, and the particle-steps beyond the record's limit.
///
/// A frame flown by the modes adds its linearisation residual
/// (`modal_energy_change`) to what the solves allow, as the modal flight
/// gate does (`tests/modal_flight.rs`).
#[derive(Default)]
struct Ledger {
    allowed: f64,
    worst: f64,
    worst_absolute: f64,
    tightest: f64,
    start: f64,
    heat: f64,
    beyond: u32,
    strain: f64,
    /// The frames flown by the modes, the least heat a frame made, and the
    /// damped frames whose propagator check failed.
    flown: u32,
    least_heat: f64,
    check_failed: u64,
    refused: u64,
    hand_backs: u64,
}

impl Ledger {
    fn frame(&mut self, sim: &Simulation) {
        let f = sim.total_lagrangian_frame().expect("the path is on");
        assert!(f.energy.is_finite());
        self.allowed += f.residual_work + f.contact_work_bound + f.modal_energy_change.abs();
        self.flown += u32::from(f.modal);
        self.least_heat = self.least_heat.min(f.relaxation_heat);
        self.check_failed = f.modal_check_failed;
        self.refused = f.modal_refused;
        self.hand_backs = f.hand_backs;
        let round_off = sim.particles().len() as f64
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        self.tightest = self
            .tightest
            .max(f.balance.abs() / (self.allowed + round_off));
        self.worst = self.worst.max(f.balance.abs() / f.start_energy.abs());
        self.worst_absolute = self.worst_absolute.max(f.balance.abs());
        self.start = f.start_energy;
        self.heat += f.relaxation_heat;
        self.beyond += f.relaxation_beyond;
        self.strain = self.strain.max(f.relaxation_strain);
    }

    fn summary(&self) -> String {
        format!(
            "balance at most {:.2e} J/m ({:.2e} of the start's {:.3e} J/m, {:.3e} of the solves' bound), branches' heat {:.3e} J/m, largest strain {:.2e}, beyond the limit {}",
            self.worst_absolute,
            self.worst,
            self.start,
            self.tightest,
            self.heat,
            self.strain,
            self.beyond
        )
    }

    /// The ledger's criteria: the balance within 1e-8 of the start energy
    /// and within the solves' bound, heat made, nothing beyond the record's
    /// limit.
    fn check(&self, name: &str) {
        assert!(self.worst < 1e-8, "{name}: energy balance {:e}", self.worst);
        assert!(
            self.tightest <= 1.0,
            "{name}: balance {:e} of the solves' bound",
            self.tightest
        );
        assert!(self.heat > 0.0, "{name}: no heat");
        assert_eq!(self.beyond, 0, "{name}: beyond the record's strain limit");
    }
}

/// What a ringing run measured: the modal amplitude per frame and its
/// ledger.
struct Ring {
    samples: Vec<f64>,
    ledger: Ledger,
}

/// A rod scene's scale: cell size, Young's modulus, frame and speed (the
/// release or impact speed). The rod itself is 0.32 m by 0.02 m, `nu = 0`,
/// 1000 kg/m^3, in a 0.48 m domain.
#[derive(Clone, Copy)]
struct Rod {
    dx: f32,
    e_pa: f32,
    frame_s: f32,
    speed_m_s: f32,
    /// The step controller's tolerance.
    tolerance: f64,
    /// Whether its flights are carried by its modes.
    modal_flight: bool,
}

/// The ringing gate's bar: quarter-wave 7.81 Hz, 1 cm cells.
const BAR: Rod = Rod {
    dx: DX_M,
    e_pa: E_PA,
    frame_s: FRAME_S,
    speed_m_s: SPEED_M_S,
    tolerance: 1e-2,
    modal_flight: false,
};

impl Rod {
    fn cells(&self, metres: f32) -> i32 {
        (metres / self.dx).round() as i32
    }

    fn config(&self) -> SimConfig {
        let mut config = SimConfig::earth(self.cells(0.48) as usize, self.dx, self.frame_s);
        config.gravity = Vec2::ZERO;
        config.min_dt = config.min_dt.min(1e-9);
        let path = TotalLagrangianConfig::new(self.tolerance);
        config.total_lagrangian = Some(if self.modal_flight {
            path.with_modal_flight()
        } else {
            path
        });
        config
    }

    fn props(&self) -> Elastic {
        Elastic {
            e_pa: self.e_pa,
            nu: 0.0,
            rho_kg_m3: RHO,
        }
    }

    /// The bar's own elastic speed of sound, m/s, from its free-free mode.
    fn sound_speed(&self) -> f64 {
        let free = ring_bar_at(self, None, 100);
        let (period, _) = period_and_decrement(&free.samples, f64::from(self.frame_s));
        2.0 * f64::from(self.cells(0.32) as f32 * self.dx) / period
    }
}

/// The free-free bar released in its first mode, `frames` frames.
fn ring_bar(relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    ring_bar_at(&BAR, relaxation, frames)
}

fn ring_bar_at(rod: &Rod, relaxation: Option<RelaxationSpectrum>, frames: usize) -> Ring {
    let props = rod.props();
    let config = rod.config();
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = relaxation;
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let x0 = rod.cells(0.08) as f32;
    let l = rod.cells(0.32) as f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(rod.cells(0.32), rod.cells(0.02)))
            .at(Vec2::new(x0 + 0.5 * l, rod.cells(0.24) as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    let speed = rod.speed_m_s / rod.dx;
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = Vec2::new(speed * (std::f32::consts::PI * (x.x - x0) / l).cos(), 0.0);
    }
    let amplitude = move |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for i in 0..p.len() {
            let shape = f64::from((std::f32::consts::PI * (p.x[i].x - x0) / l).cos());
            num += f64::from(p.mass[i]) * f64::from(p.v[i].x) * shape;
            den += f64::from(p.mass[i]) * shape * shape;
        }
        num / den
    };
    run(sim, amplitude, frames)
}

/// The damped period and the logarithmic decrement per period from samples
/// of an oscillating signal every `dt`: the peaks of `|signal|` (a parabola
/// through the three samples around each), every half period; over the
/// first five, two periods.
fn period_and_decrement(samples: &[f64], dt: f64) -> (f64, f64) {
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for k in 1..samples.len() - 1 {
        let (a, b, c) = (samples[k - 1].abs(), samples[k].abs(), samples[k + 1].abs());
        if b >= a && b > c {
            let curvature = a - 2.0 * b + c;
            let shift = if curvature != 0.0 {
                0.5 * (a - c) / curvature
            } else {
                0.0
            };
            peaks.push(((k as f64 + shift) * dt, b - 0.25 * (a - c) * shift));
        }
    }
    assert!(peaks.len() >= 5, "too few peaks: {peaks:?}");
    let ((t0, a0), (t4, a4)) = (peaks[0], peaks[4]);
    ((t4 - t0) / 2.0, (a0 / a4).ln() / 2.0)
}

/// The exact Zener mode of relaxed angular frequency `omega` for `record`'s
/// weights `kappa` (the shear's or the P wave's), sampled like the runs.
fn reference(
    record: &RelaxationSpectrum,
    kappa: impl Fn(&Mechanism) -> f64,
    omega: f64,
    frames: usize,
) -> (f64, f64) {
    let material = zener::Zener {
        rho: 1.0,
        relaxed: 1.0,
        kappa: record.mechanisms().iter().map(&kappa).collect(),
        theta: record.mechanisms().iter().map(|m| m.theta).collect(),
    };
    let start = zener::ModeState {
        q: 0.0,
        rate: 1.0,
        memory: vec![0.0; record.mechanisms().len()],
    };
    let mode = zener::ModeSolution::new(&material, omega, &start);
    let (root, reproduced) = mode.checks();
    assert!(
        root < 1e-12 && reproduced < 1e-12,
        "reference: roots {root:e}, start {reproduced:e}"
    );
    let dt = f64::from(FRAME_S);
    let samples: Vec<f64> = (0..=frames)
        .map(|i| mode.at(dt * i as f64).0.rate)
        .collect();
    period_and_decrement(&samples, dt)
}

/// Runs `frames` frames of `sim`, sampling `amplitude` after each.
fn run(mut sim: Simulation, amplitude: impl Fn(&Simulation) -> f64, frames: usize) -> Ring {
    let mut ring = Ring {
        samples: vec![amplitude(&sim)],
        ledger: Ledger::default(),
    };
    for _ in 0..frames {
        sim.step();
        ring.ledger.frame(&sim);
        ring.samples.push(amplitude(&sim));
    }
    ring
}

#[path = "zener_gates/ringing.rs"]
mod ringing;

#[path = "zener_gates/rods.rs"]
mod rods;

#[path = "zener_gates/refusals.rs"]
mod refusals;

#[path = "zener_gates/scenes.rs"]
mod scenes;

//! A ductile slab settling under its own weight between two frictionless
//! side walls, on the total Lagrangian path, against the exact static
//! answer: J2 plasticity on Hencky elasticity (Love and Sulsky 2005) in
//! uniaxial strain, the plastic test of the step's energy-consistent stress
//! and the check for volumetric locking (isochoric flow against a fixed
//! lateral constraint, the regime most prone to it).
//!
//! The exact answer, derived here for small strain (the strains are below
//! 1e-3): the side walls hold `eps_xx = 0`, plane strain `eps_zz = 0`, so the
//! strain is `eps_yy = eps(y)` alone and statics give `sigma_yy = -rho g (h -
//! y)` whatever the material. Elastic, `sigma_yy = (K + 4G/3) eps`; the
//! strain deviator `eps (-1, 2, -1) / 3`, of norm `sqrt(2/3) |eps|`, so the
//! yield `|dev tau| = sqrt(2/3) Y` is reached at `|eps| = Y / (2G)`, a stress
//! `(K + 4G/3) Y / (2G)`. Past it, with no hardening, the deviator stays
//! on the yield surface along the same direction, `s_yy = -2Y/3`, while the
//! pressure follows the volume (the flow is isochoric), `sigma_yy = K eps -
//! 2Y/3`, so `eps = (sigma_yy + 2Y/3) / K`. The settlement profile is the
//! integral of that piecewise linear strain.
//!
//! The closed form is the monotone loading's: released at once, an elastic
//! body overshoots its static strain and would flow further than the static
//! load asks. The Kelvin-Voigt dashpot is therefore three times the first
//! vertical mode's critical damping, so the slab creeps onto its static
//! state without overshoot; at rest it adds no stress. The dashpot is the
//! generic Newtonian one (`matter`'s `newtonian_viscous_stress`), shear
//! viscosity only, read from the material's `elastic_viscosity` as the
//! textbook Pa s (`total_lagrangian_solids`, declared).
//!
//! Measured as `tests/settled_slab.rs` measures the elastic slab: the drop
//! of the centre of mass relative to the bottom row, at three cells, the
//! gap's observed order and the Richardson extrapolation; criterion the same
//! 5 percent on the extrapolated compression, fixed before measuring. A
//! locked step would come out short of the exact compression, and would not
//! converge to it.
//!
//! Which cells: the side walls hold the slab through the weighted contact
//! conditions (`weighted_gap`), one multiplier per patch of `H = 2h` along
//! a face, `floor(L / H)` patches. One patch holds only the face's mean
//! normal motion, its tilt left free; the lateral pressure of uniaxial
//! strain grows with depth (`sigma_xx = sigma_yy + Y` where yielded), and
//! its first moment needs a second patch. A face therefore needs at least
//! two patches before the walls hold the slab as the closed form assumes.
//! The 4 cm cell (3 cells high, one patch) is outside: measured there 1.297
//! of exact, the face tilting (the bottom row 0.14 mm wider, the top row
//! 0.13 mm narrower), the yielded base flowing into the relief. The three
//! cells are 2 cm (6 cells high, 3 patches), 1 cm and 0.5 cm.
//!
//! The step tolerance, 1e-4: route C's contact lets a point reach a wall by
//! `O(v dt)` before holding it, and on the side walls that lateral give is
//! plastic flow kept for good. Measured at 2 cm (with the controller's
//! earlier scale, the start's total energy): 1.204 of exact at tolerance
//! 1e-1 (the first step one whole 16.7 ms frame, past the first vertical
//! period, 13 ms), 1.0488 at 1e-3, 1.0053 at 1e-4, 1.0032 at 1e-5; 4 cm
//! moved from 1.2961 to 1.2967 between 1e-3 and 1e-4. At 1e-4 the step's
//! part is within 0.3 percent, below the cell's.
extern crate emerge_engine as emerge;

#[path = "common/cap.rs"]
mod cap;

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{DuctileProps, FromSI, SimConfig, Simulation, SpawnRegion, VonMisesMaterial};
use glam::{IVec2, Vec2};

const E_PA: f64 = 1.0e6;
const NU: f64 = 0.3;
const RHO: f64 = 1000.0;
/// The uniaxial yield stress, Pa: the bottom's static stress is `rho g h`,
/// about 1180 Pa, the yield onset `1.75 Y` about 525 Pa, so the lower half
/// of the slab flows.
const YIELD_PA: f64 = 300.0;
/// The slab, m: 0.12 high, filling the domain's width between its walls.
const HEIGHT_M: f64 = 0.12;
const DOMAIN_M: f64 = 0.48;
const G: f64 = 9.81;

/// The moduli of `E_PA`, `NU`: shear and bulk.
fn moduli() -> (f64, f64) {
    let g = E_PA / (2.0 * (1.0 + NU));
    let k = E_PA / (3.0 * (1.0 - 2.0 * NU));
    (g, k)
}

/// The exact compressive strain at height `y` above the bottom (positive).
fn strain(y: f64) -> f64 {
    let (g, k) = moduli();
    let load = RHO * G * (HEIGHT_M - y);
    let onset = (k + 4.0 * g / 3.0) * YIELD_PA / (2.0 * g);
    if load <= onset {
        load / (k + 4.0 * g / 3.0)
    } else {
        (load - 2.0 * YIELD_PA / 3.0) / k
    }
}

/// The exact drop of the centre of mass relative to the bottom row at
/// height `y_b`: `(1/h) int_0^h int_{y_b}^{y} eps dy' dy`, by Simpson's
/// rule on each piece where the strain is linear (exact for it).
fn exact_drop(y_b: f64) -> f64 {
    let (g, k) = moduli();
    let onset = (k + 4.0 * g / 3.0) * YIELD_PA / (2.0 * g);
    let knee = (HEIGHT_M - onset / (RHO * G)).clamp(0.0, HEIGHT_M);
    // The compression between the bottom row and `y`, by Simpson on the
    // linear pieces of the strain.
    let piece = |a: f64, b: f64| -> f64 {
        if b <= a {
            return 0.0;
        }
        (b - a) / 6.0 * (strain(a) + 4.0 * strain(0.5 * (a + b)) + strain(b))
    };
    let between = |a: f64, b: f64| -> f64 {
        let (lo, hi, sign) = if a <= b { (a, b, 1.0) } else { (b, a, -1.0) };
        let split = knee.clamp(lo, hi);
        sign * (piece(lo, split) + piece(split, hi))
    };
    // The outer integral over `y`: a fine composite Simpson (the inner
    // integral is piecewise quadratic in `y`, Simpson exact on each panel
    // that does not straddle the knee; 20000 panels make that one's error
    // negligible).
    let n = 20000;
    let step = HEIGHT_M / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let (a, b) = (i as f64 * step, (i + 1) as f64 * step);
        sum += step / 6.0 * (between(y_b, a) + 4.0 * between(y_b, 0.5 * (a + b)) + between(y_b, b));
    }
    sum / HEIGHT_M
}

/// The settled compression over the exact one at cell `dx`, m.
fn settled_ratio(dx: f32) -> f64 {
    let props = DuctileProps {
        elastic: Elastic {
            e_pa: E_PA as f32,
            nu: NU as f32,
            rho_kg_m3: RHO as f32,
        },
        yield_stress_pa: YIELD_PA as f32,
    };
    let res = (DOMAIN_M / f64::from(dx)).round() as usize;
    let mut config = SimConfig::earth(res, dx, 1.0 / 60.0);
    // `EMERGE_SLAB_TOL`, `EMERGE_SLAB_FRAMES`: the step controller's
    // tolerance and the frames run, for exploring the settling.
    let knob = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    config.total_lagrangian = Some(TotalLagrangianConfig::new(knob("EMERGE_SLAB_TOL", 1e-4)));
    let wall = config.boundary_thickness as i32;
    let mut material = VonMisesMaterial::from_physical(&props, &config);
    // Three times the first vertical mode's critical damping: the mode
    // `omega = pi c / (2 h)`, `c = sqrt((K + 4G/3) / rho)`, damped by the
    // uniaxial viscosity `4 eta / 3` (shear only), ratio `(4 eta / 3)
    // omega / (2 (K + 4G/3))`.
    let (g, k) = moduli();
    let modulus = k + 4.0 * g / 3.0;
    let omega = std::f64::consts::PI * (modulus / RHO).sqrt() / (2.0 * HEIGHT_M);
    let eta = 3.0 * 2.0 * modulus / omega * 0.75;
    material.elastic_viscosity = config.visc_from_si(eta as f32);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = wall as f32;
    let cells = (HEIGHT_M / f64::from(dx)).round() as i32;
    let width = res as i32 - 2 * wall;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(width, cells))
            .at(Vec2::new(res as f32 * 0.5, floor + 0.5 * cells as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props.elastic, &config),
    );
    let lowest = sim
        .particles()
        .x
        .iter()
        .map(|x| x.y)
        .fold(f32::INFINITY, f32::min);
    let bottom: Vec<bool> = sim
        .particles()
        .x
        .iter()
        .map(|x| x.y < lowest + 0.25)
        .collect();
    let y_b = f64::from((lowest - floor) * dx);
    let centre = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        let mean: f64 =
            p.x.iter()
                .zip(&p.mass)
                .map(|(x, &m)| f64::from(x.y) * f64::from(m))
                .sum::<f64>()
                / mass;
        let (sum, count) =
            p.x.iter()
                .zip(&bottom)
                .filter(|(_, b)| **b)
                .fold((0.0f64, 0.0f64), |(s, c), (x, _)| {
                    (s + f64::from(x.y), c + 1.0)
                });
        (mean - sum / count) * f64::from(dx)
    };
    let start = centre(&sim);
    // The bottom and top rows' widths at the start, m (the trace's).
    let widths = {
        let p = sim.particles();
        let width_at = |y: f32| -> f64 {
            let (l, r) =
                p.x.iter()
                    .filter(|x| (x.y - y).abs() < 0.25)
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, r), x| {
                        (l.min(x.x), r.max(x.x))
                    });
            f64::from(r - l) * f64::from(dx)
        };
        (
            width_at(lowest),
            width_at(lowest + 0.5 * (2 * cells - 1) as f32),
        )
    };
    let mut worst_balance = 0.0f64;
    let mut plastic = 0.0f64;
    for _ in 0..knob("EMERGE_SLAB_FRAMES", 30.0) as usize {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        worst_balance = worst_balance.max(frame.balance.abs() / frame.tolerance_energy);
        plastic += frame.plastic_work;
        if std::env::var_os("EMERGE_SLAB_TRACE").is_some() {
            // The side faces' give: the bottom and top rows' width change.
            let p = sim.particles();
            let row_width = |y: f32| -> f64 {
                let (l, r) =
                    p.x.iter()
                        .filter(|x| (x.y - y).abs() < 0.25)
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, r), x| {
                            (l.min(x.x), r.max(x.x))
                        });
                f64::from(r - l) * f64::from(dx)
            };
            let top = lowest + 0.5 * (2 * cells - 1) as f32;
            println!(
                "frame: balance {:+.2e} kinetic {:.3e} J/m plastic {:.3e} J/m steps {} drop {:.4e} m; bottom row width {:+.2e} m, top row {:+.2e} m",
                frame.balance / frame.tolerance_energy,
                frame.parts[0],
                frame.plastic_work,
                frame.steps,
                start - centre(&sim),
                row_width(lowest) - widths.0,
                row_width(top) - widths.1
            );
        }
    }
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let drop = start - centre(&sim);
    let exact = exact_drop(y_b);
    println!(
        "plastic slab, cell {dx} m: compression {drop:.4e} m, exact {exact:.4e} m, ratio {:.4}; plastic work {plastic:.3e} J/m; kinetic left {:.2e} J/m; worst balance {worst_balance:.2e}",
        drop / exact,
        frame.parts[0]
    );
    assert!(plastic > 0.0, "the slab did not flow");
    drop / exact
}

#[test]
#[ignore = "long: the three resolutions' settling, hours at 0.5 cm"]
fn a_ductile_slab_settles_to_the_exact_uniaxial_strain_compression() {
    cap::capped(
        "a_ductile_slab_settles_to_the_exact_uniaxial_strain_compression",
        21600,
        || {
            // `EMERGE_SLAB_CELL`: one resolution alone (m), for exploring.
            if let Some(dx) = std::env::var("EMERGE_SLAB_CELL")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                settled_ratio(dx);
                return;
            }
            let (r2, r1, r05) = (
                settled_ratio(0.02),
                settled_ratio(0.01),
                settled_ratio(0.005),
            );
            let order = ((r1 - r2) / (r05 - r1)).abs().log2();
            let extrapolated = r05 + (r05 - r1) / (2f64.powf(order) - 1.0);
            println!(
                "compression over exact: {r2:.4} at 2 cm, {r1:.4} at 1 cm, {r05:.4} at 0.5 cm; observed order {order:.2}; extrapolated to dx -> 0 {extrapolated:.4}"
            );
            assert!(
                (extrapolated - 1.0).abs() < 0.05,
                "extrapolated compression {extrapolated} of exact"
            );
        },
    );
}

//! A slab settling under its own weight on the total Lagrangian path,
//! against the exact static answer (issue #41's hydrostatic question, on
//! this path).
//!
//! With `nu = 0` the stress state on a frictionless floor is uniaxial:
//! `sigma_yy = -rho g (h - y)`, `eps_yy = sigma_yy / E`, so the displacement
//! is `u(y) = -(rho g / E) (h y - y^2 / 2)` and the centre of mass drops by
//! `rho g h^2 / (3 E)` exactly, relative to the floor. Measured relative to
//! the slab's own lowest points (its outline on the bottom face, `y_b = 0`),
//! the compression alone, not the contact's declared penetration into the
//! floor (route C's `O(v dt)`, about 0.03 of a cell, which moves the whole
//! slab down). A Kelvin-Voigt dashpot (shear viscosity `eta`, bulk `2 eta /
//! 3`, the uniaxial pair of `kelvin_voigt_bar.rs`) damps the ringing from
//! the stress-free spawn so the slab comes to rest; at rest it adds no
//! stress.
//!
//! Criterion: the 5 percent fixed before measuring, applied to the
//! compression extrapolated to `dx -> 0` with the observed order (a single
//! resolution measures the mesh, not the law). Measured: 1.0124 of exact at
//! 4 cm cells (3 over the height), 1.0022 at 2 cm (6), 1.0000 at 1 cm (12);
//! observed order 2.17, extrapolated 0.9993. The energy balance (stored,
//! gravity's, kinetic, heat) to round-off at each.
//!
//! History, measured: before the spawn laid its lattice at the sub-cells'
//! centres (issue #42) the ratios were 0.621, 0.802, 0.899, order 0.90, and
//! a first-order error at the body's boundary was declared here; it was the
//! spawn's quarter cell, the matter's centre and its outline disagreeing.

extern crate emerge_engine as emerge;

#[path = "common/cap.rs"]
mod cap;

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const E_PA: f32 = 1.0e5;
const RHO: f32 = 1000.0;
/// The slab, m: 0.12 high, 0.30 wide.
const HEIGHT_M: f32 = 0.12;
const WIDTH_M: f32 = 0.30;
/// Shear viscosity, Pa s: about half the critical damping of the slab's
/// first vertical mode (`omega = pi c / (2 h)`).
const ETA_PA_S: f32 = 400.0;

/// The settled compression over the exact one at cell `dx`, m (the slab's
/// size in metres the same at every cell).
fn settled_ratio(dx: f32) -> f64 {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let cells = |m: f32| (m / dx).round() as i32;
    let mut config = SimConfig::earth((0.48 / dx).round() as usize, dx, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.elastic_viscosity = config.visc_from_si(ETA_PA_S);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * ETA_PA_S / 3.0);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let floor = config.boundary_thickness as f32;
    let _slab = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(cells(WIDTH_M), cells(HEIGHT_M)))
            .at(Vec2::new(0.24 / dx, floor + 0.5 * cells(HEIGHT_M) as f32))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    // The bottom row: the particles within half a spacing of the lowest.
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
    // The centre of mass above the bottom row's mean height, m.
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
    let mut worst_balance = 0.0f64;
    let mut kinetic = f64::INFINITY;
    // 0.5 s: the centre is at rest from about 0.15 s (measured).
    for _ in 0..30 {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        worst_balance = worst_balance.max(frame.balance.abs() / frame.start_energy.abs());
        kinetic = frame.parts[0];
        if std::env::var_os("EMERGE_SLAB_TRACE").is_some() {
            println!(
                "frame: balance {:+.2e} kinetic {:.3e} stored {:.3e} steps {} rejected {} stagnated {} centre drop {:.4e} m",
                frame.balance / frame.start_energy.abs(),
                frame.parts[0],
                frame.parts[1],
                frame.steps,
                frame.rejected,
                frame.stagnated,
                start - centre(&sim)
            );
        }
    }
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let drop = start - centre(&sim);
    let h = f64::from(cells(HEIGHT_M) as f32 * dx);
    let exact =
        f64::from(RHO) * 9.81 / f64::from(E_PA) * (h * h / 3.0 - (h * y_b - 0.5 * y_b * y_b));
    println!(
        "settled slab: compression {drop:.4e} m, exact {exact:.4e} m, ratio {:.4}; kinetic left {kinetic:.2e} J/m (stored {:.3e}); worst balance {worst_balance:.2e} of the start's energy; stagnated steps in the last frame {}",
        drop / exact,
        frame.parts[1],
        frame.stagnated
    );
    assert!(worst_balance < 1e-8, "energy balance {worst_balance:e}");
    drop / exact
}

#[test]
#[ignore = "long: about 10 minutes (the 1 cm slab's 60 floor contacts)"]
fn a_damped_slab_settles_to_the_exact_self_weight_compression() {
    cap::capped(
        "a_damped_slab_settles_to_the_exact_self_weight_compression",
        3600,
        || {
            let (r4, r2, r1) = (
                settled_ratio(0.04),
                settled_ratio(0.02),
                settled_ratio(0.01),
            );
            // The observed order of the gap, and the compression extrapolated with
            // it (Richardson): `r1 + (r1 - r2) / (2^p - 1)`.
            let order = ((r2 - r4) / (r1 - r2)).log2();
            let extrapolated = r1 + (r1 - r2) / (2f64.powf(order) - 1.0);
            println!(
                "compression over exact: {r4:.4} at 4 cm, {r2:.4} at 2 cm, {r1:.4} at 1 cm; observed order {order:.2}; extrapolated to dx -> 0 {extrapolated:.4}"
            );
            assert!(
                (extrapolated - 1.0).abs() < 0.05,
                "extrapolated compression {extrapolated} of exact"
            );
        },
    );
}

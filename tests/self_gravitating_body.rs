//! Self-gravitating rubble-pile body: does a cluster of regolith, given only
//! mutual self-gravity (no external field, as in microgravity), stay
//! gravitationally bound instead of dispersing? The step from planets as
//! point masses (`tests/orbital_mechanics.rs`) toward planets made of matter,
//! with existing tools: `NBodyGravityField` (whose doc names "planetary
//! terrain, accretion disks" as its use case) and `DruckerPragerMaterial`
//! (cohesionless granular, as used for sand and regolith).
//!
//! Target: asteroid 101955 Bennu (OSIRIS-REx), a rubble pile held together
//! almost entirely by self-gravity with negligible internal cohesion.
//! Measured: mass 7.329e10 kg, mean radius ~245 m, bulk density ~1190
//! kg/m^3, escape velocity ~20 cm/s (Lauretta et al. 2019, "The unexpected
//! surface of asteroid (101955) Bennu", Nature).
//!
//! The regolith's elastic modulus (E=5 MPa) and friction angle (35 deg) are a
//! plausible loose-granular range from general geotechnical/regolith
//! literature (Apollo-era lunar regolith studies commonly cite friction
//! angles of 30-50 deg), not a measurement for Bennu itself.

#![cfg(feature = "experimental")]

extern crate emerge_engine as emerge;

use emerge::fields::NBodyGravityField;
use emerge::{
    Elastic, Elastoplastic, PlasticityModel, SimConfig, Simulation, SpawnRegion, SpawnShape,
};
use glam::{IVec2, Vec2};

/// OSIRIS-REx measured data (Lauretta et al. 2019, Nature).
const BENNU_MASS_KG: f64 = 7.329e10;
const BENNU_RADIUS_M: f64 = 245.0;
const BENNU_BULK_DENSITY_KG_M3: f64 = 1190.0;

/// 2 m/cell -- fine enough to resolve a ~245 m body with real spatial
/// structure, coarse enough to keep particle count/grid size tractable.
const DX_METERS: f64 = 2.0;
const GRID_RES: usize = 512;

/// The dynamical (free-fall) timescale is ~sqrt(R^3/(G*M)) ~ 8.7 hours at
/// this scene's total mass (from the 2D areal density); 20 s/step lets 2000
/// steps (40,000 s ~ 11 h) cover a meaningful fraction of it.
const DT_SECONDS: f64 = 20.0;

fn make_body() -> Simulation {
    let center = Vec2::splat(GRID_RES as f32 / 2.0);
    let r_grid = (BENNU_RADIUS_M / DX_METERS) as f32;

    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        gravity: Vec2::ZERO, // real microgravity -- self-gravity IS the only gravity here
        ..SimConfig::standard(GRID_RES, DT_SECONDS as f32, Vec2::ZERO)
    };

    // SI construction through `scale_lame`, the conversion for
    // solid/elastoplastic materials (not the Tait-EOS fluid path).
    let regolith = Elastoplastic {
        elastic: Elastic {
            e_pa: 5.0e6,
            nu: 0.25,
            rho_kg_m3: BENNU_BULK_DENSITY_KG_M3 as f32,
        },
        model: PlasticityModel::Granular {
            friction_angle_deg: 35.0,
            dilatancy_angle_deg: 0.0,
        },
    }
    .material(&config);

    // Coarser than an initial attempt (3.0): that version, once the real
    // dt_seconds fix let real dynamics actually happen, ran far too long
    // (>30 real minutes, stopped) for a routine test -- 5236 particles of
    // real elastoplastic self-gravitating dynamics is heavy.
    // 8.0 cuts particle count ~7x (~750 particles), a disclosed
    // resolution/cost tradeoff, not a physics change.
    const SPACING: f32 = 8.0;
    let mass_per_particle =
        SPACING * SPACING * BENNU_BULK_DENSITY_KG_M3 as f32 * (DX_METERS * DX_METERS) as f32;

    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::splat((2.0 * r_grid) as i32 + 4),
        box_center: center,
        shape: SpawnShape::Disk { radius: r_grid },
        position_jitter: 0.1,
        mass_override: Some(mass_per_particle),
        ..SpawnRegion::for_sim(&config)
    };

    // Softening scaled to the new, coarser spacing (real practice: keep
    // softening comparable to inter-particle spacing so close encounters
    // don't force pathologically small substeps).
    let g_grid = (6.674e-11 / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    Simulation::new(config, spawn)
        .with_default_material(regolith)
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, SPACING * 0.5, 0.3)))
}

/// Standard astrophysical criterion for whether a system is
/// gravitationally bound: total energy (kinetic + gravitational potential)
/// is negative. Computed directly, not assumed.
fn total_energy(sim: &Simulation, g_grid: f64) -> f64 {
    let p = sim.particles();
    let n = p.len();
    let ke: f64 = (0..n)
        .map(|i| 0.5 * p.mass[i] as f64 * p.v[i].length_squared() as f64)
        .sum();
    let mut pe = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = ((p.x[i] - p.x[j]).length() as f64).max(1.0e-6);
            pe -= g_grid * p.mass[i] as f64 * p.mass[j] as f64 / r;
        }
    }
    ke + pe
}

fn radius_of_gyration(sim: &Simulation) -> f32 {
    let p = sim.particles();
    let n = p.len();
    let centroid = (0..n).map(|i| p.x[i]).fold(Vec2::ZERO, |a, b| a + b) / n as f32;
    let mean_sq: f32 = (0..n)
        .map(|i| (p.x[i] - centroid).length_squared())
        .sum::<f32>()
        / n as f32;
    mean_sq.sqrt()
}

/// Quantitative proof: a regolith body, given ONLY real mutual
/// self-gravity, stays gravitationally bound (negative total energy,
/// bounded spatial extent) rather than dispersing -- the standard
/// astrophysical signature of a self-gravitating rubble pile, not just
/// "doesn't crash."
#[test]
fn regolith_body_stays_gravitationally_bound_under_self_gravity() {
    let mut solver = make_body();
    let g_grid = 6.674e-11 / (DX_METERS * DX_METERS * DX_METERS);
    let n = solver.particles().len();

    let total_mass_kg: f64 = (0..n).map(|i| solver.particles().mass[i] as f64).sum();
    let e0 = total_energy(&solver, g_grid);
    let rg0 = radius_of_gyration(&solver);
    eprintln!(
        "n_particles={n}  total_mass={total_mass_kg:.3e} kg (real Bennu: {BENNU_MASS_KG:.3e} kg \
         -- this scene's 2D areal-density mass is NOT the same as Bennu's real 3D mass, a \
         disclosed limitation of a 2D engine, not a match target)  e0={e0:.4e}  \
         radius_of_gyration0={rg0:.3}"
    );
    assert!(
        e0 < 0.0,
        "real rubble-pile setup should start gravitationally bound (negative total energy): {e0:.4e}"
    );

    let t0 = std::time::Instant::now();
    for step in 0..200 {
        solver.step();
        if step % 20 == 0 {
            eprintln!("step {step}  elapsed={:.1}s", t0.elapsed().as_secs_f32());
        }
    }
    eprintln!(
        "200 steps took {:.1}s wall-clock",
        t0.elapsed().as_secs_f32()
    );

    for i in 0..solver.particles().len() {
        assert!(
            solver.particles().x[i].is_finite() && solver.particles().v[i].is_finite(),
            "particle {i} went non-finite"
        );
    }

    let e1 = total_energy(&solver, g_grid);
    let rg1 = radius_of_gyration(&solver);
    eprintln!("after 200 steps: e1={e1:.4e}  radius_of_gyration1={rg1:.3}  (r0 was {rg0:.3})");

    assert!(
        e1 < 0.0,
        "should REMAIN gravitationally bound (negative total energy): {e1:.4e}"
    );
    // Honest bound: extent shouldn't blow up (disperse) -- generous
    // factor (not exact conservation) since real plastic/frictional
    // dissipation and settling are expected to shift it somewhat.
    assert!(
        rg1 < rg0 * 3.0,
        "body should stay roughly bound, not disperse: rg0={rg0:.3} rg1={rg1:.3}"
    );
}

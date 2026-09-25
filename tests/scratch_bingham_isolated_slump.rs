//! How far does each slump column spread when nothing else is in the way,
//! and what does a floor that grips change?
//!
//! The slump demo stands three columns close enough that their deposits
//! meet (`tests/scratch_bingham_column_symmetry.rs`), so its deposits are
//! not three independent slumps. Spacing them properly needs to know how
//! wide each one gets alone, and that depends on the floor. The demo's floor
//! is frictionless (`SlipBoundary`), and a yield-stress fluid on a
//! frictionless floor is not the slump test: the measurements the demo's
//! inversion comes from are made on a substrate the material does not slip
//! on. Woodbridge, Fonte and Juel (arXiv 2609.12229, 2026), modelling
//! yield-stress drop spreading, impose "no-slip and no-penetration
//! conditions" at the substrate, and measure the same fluids' rheology
//! between "cross-hatched parallel plates to minimise wall slip", because
//! such fluids slip on a smooth wall.
//!
//! `FrictionBoundary` is Coulomb friction, which acts as no slip only where
//! it can hold the shear the base has to carry: `mu * rho * g * h` against
//! roughly `tau_0`. At `mu = 1` that holds with a wide margin for the two
//! columns that spread (216 Pa against 60 under a 22 mm deposit, far more
//! against 2), and not for the 1200 Pa one, which does not spread and so
//! loads its base with almost nothing. A declared approximation (2).
//!
//! Each column alone, spawned on a mirror line of the grid, in a tank twice
//! the demo's width so the softest one does not reach the walls; if it does,
//! the row says so, because a wall-stopped spread is not a free one.
//!
//!   cargo test --profile quick --all-features --test scratch_bingham_isolated_slump -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, BoundaryCondition, FrictionBoundary, FromSI, SimConfig,
    Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

// The slump demo's own constants, in a tank twice as wide.
const GRID: usize = 128;
const DX_M: f32 = 0.002;
const RHO_KG_M3: f32 = 1000.0;
const G: f32 = 9.81;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
const FLOOR_CELLS: f32 = 2.0;
const YIELDS_PA: [f32; 3] = [2.0, 60.0, 1200.0];

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The demo's bulk modulus, from its own derivation: sound speed ten times
/// the free-fall speed over the column's height, `K = rho c^2`.
fn bulk_modulus_pa() -> f32 {
    let v_max = (2.0 * G * COLUMN_CELLS.y as f32 * DX_M).sqrt();
    RHO_KG_M3 * (10.0 * v_max).powi(2)
}

struct Deposit {
    h_mm: f32,
    half_width_mm: f32,
    /// Largest particle speed at the end, m/s: near zero means settled.
    vmax_m_s: f32,
    touched_wall: bool,
}

fn slump(tau0: f32, floor_mu: Option<f32>, seconds: f32, dt: f32) -> Deposit {
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: bulk_modulus_pa(),
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    // A quarter cell right of the tank's axis, so the lattice, which starts
    // at the box's corner (issue #42), is centred ON the axis.
    let axis = GRID as f32 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(axis + 0.25, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(&props, &config);
    let floor: Box<dyn BoundaryCondition> = match floor_mu {
        None => Box::new(SlipBoundary::new(config.boundary_thickness)),
        Some(mu) => Box::new(FrictionBoundary::new(config.boundary_thickness, mu)),
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(floor);
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
    }
    let p = sim.particles();
    let (mut top, mut lo, mut hi, mut vmax) = (f32::MIN, f32::MAX, f32::MIN, 0.0f32);
    for i in 0..p.len() {
        top = top.max(p.x[i].y);
        lo = lo.min(p.x[i].x);
        hi = hi.max(p.x[i].x);
        vmax = vmax.max(p.v[i].length());
    }
    // Same definitions as `bingham_slump_probe`: height above the floor,
    // half the horizontal extent.
    let wall = config.boundary_thickness as f32 + 1.0;
    Deposit {
        h_mm: (top - FLOOR_CELLS).max(0.0) * DX_M * 1000.0,
        half_width_mm: (hi - lo) * 0.5 * DX_M * 1000.0,
        vmax_m_s: vmax * DX_M,
        touched_wall: lo < wall || hi > GRID as f32 - wall,
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn each_column_alone_on_a_slip_and_a_gripping_floor() {
    let seconds = env("SLUMP_SECONDS", 3.0);
    let dt = env("SLUMP_DT", 0.001);
    println!(
        "each column alone, {GRID}-cell tank ({} mm), {seconds} s at {dt} s a frame",
        GRID as f32 * DX_M * 1000.0
    );
    println!("  tau_0 Pa  floor          h mm   L mm   h/L    tau_0 read back   vmax m/s   note");
    // `SLUMP_TAU` runs one column only, to follow a slow one for longer.
    let only = std::env::var("SLUMP_TAU")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    for tau0 in YIELDS_PA
        .into_iter()
        .filter(|t| only.is_none_or(|o| o == *t))
    {
        for (label, mu) in [("slip", None), ("friction 1", Some(1.0f32))] {
            let d = slump(tau0, mu, seconds, dt);
            let (h, l) = (d.h_mm * 1.0e-3, d.half_width_mm * 1.0e-3);
            let read_back = RHO_KG_M3 * G * h * h / (2.0 * l.max(1.0e-9));
            println!(
                "  {tau0:>7}   {label:<12} {:>6.1} {:>6.1}  {:>5.2}   {:>9.1} Pa     {:>7.4}   {}",
                d.h_mm,
                d.half_width_mm,
                d.h_mm / d.half_width_mm.max(1.0e-6),
                read_back,
                d.vmax_m_s,
                if d.touched_wall { "reached a wall" } else { "" }
            );
        }
    }
}

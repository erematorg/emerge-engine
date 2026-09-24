//! Why does a spread-out body keep gaining volume?
//!
//! Found by watching a demo: particles grow and drift apart the longer a
//! scene runs, and since the renderer draws each particle deformed by its
//! own F, what is visible IS the volume ratio. Measured on the slump
//! scene, the two columns that hold a shape are flat to the fourth digit
//! over twenty seconds while the one that spreads into a thin wide layer
//! gains about 0.08 % of its volume a second and accelerates. Its own
//! weight says it should sit at J = 0.998, so the sign is wrong and the
//! size is ten times too large: numerical, not a state.
//!
//! This sweeps the one thing that differs. Same material, same width,
//! same mass per particle, same gravity: only the layer's THICKNESS
//! changes. The code names a free-surface mechanism in `transfer/g2p.rs`
//! where a layer thinner than the kernel's own support reads as expanded
//! in every depth band, so if that is what this is, the drift has to grow
//! as the layer thins. If it does not, it is something else.
//!
//! Knobs: DRIFT_SECONDS, DRIFT_GRAVITY (fraction of g, to separate a
//! load-driven effect from one that happens at rest), DRIFT_MATERIAL
//! (`neohookean` for an ordinary solid instead of the yield-stress fluid,
//! to see whether it is one law or the transfer).
//!
//!   cargo test --profile quick --all-features --test scratch_thin_layer_volume_drift -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, DruckerPragerMaterial, FromSI, MaterialModel,
    NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
/// 2 mm cells, the slump scene's own scale, so the numbers are comparable.
const DX_M: f32 = 0.002;
const WIDTH_CELLS: i32 = 40;
const FLOOR_CELLS: f32 = 2.0;
const RHO_KG_M3: f32 = 1000.0;
/// The soft column's own yield stress, the one that drifts.
const YIELD_PA: f32 = 2.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const YOUNG_PA: f32 = 2.0e5;
const POISSON: f32 = 0.3;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Mean and worst volume ratio over the whole slab.
fn volume_state(sim: &Simulation) -> (f64, f32) {
    let p = sim.particles();
    let mut sum = 0.0f64;
    let mut worst = 0.0f32;
    for i in 0..p.len() {
        let j = p.deformation_gradient[i].determinant();
        sum += f64::from(j);
        worst = worst.max((j - 1.0).abs());
    }
    (sum / p.len() as f64, worst)
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn volume_drift_against_layer_thickness() {
    let seconds = env("DRIFT_SECONDS", 10.0);
    let gravity_fraction = env("DRIFT_GRAVITY", 1.0);
    let which = std::env::var("DRIFT_MATERIAL").unwrap_or_default();
    let neohookean = which == "neohookean";
    let sand = which == "sand";
    // shear_modulus = 0 selects Bingham's purely VISCOUS branch, which
    // carries J by the continuity equation directly with no SVD round
    // trip; anything else selects the elastoviscoplastic one that goes
    // through a decomposition and a reconstruction every substep.
    let viscous = which == "viscous";
    // The control the elastic slab cannot be: a body that FLOWS and
    // collapses like the yield-stress one, under the same stiffness and
    // the same viscosity, but whose law never decomposes and rebuilds F.
    // Comparing a standing elastic slab against a collapsing plastic one
    // changes the law and the motion together; this changes only the law.
    let newtonian = which == "newtonian";
    let dt = 0.002f32;

    println!(
        "{} slab, gravity x{gravity_fraction}, {seconds} s, 2 mm cells",
        if neohookean {
            "elastic (NeoHookean, no SVD round trip)"
        } else if sand {
            "sand (Drucker-Prager, SVD round trip)"
        } else if newtonian {
            "newtonian fluid (flows, no SVD)"
        } else if viscous {
            "yield-stress fluid (2 Pa, viscous branch, no SVD)"
        } else {
            "yield-stress fluid (2 Pa, SVD round trip)"
        }
    );
    println!(
        "  thickness   particles    mean J at 0 s   at end    drift per second   per substep   worst |J-1|"
    );

    for thickness in [1i32, 2, 4, 8, 16] {
        let mut config = SimConfig {
            min_dt: 1.0e-6,
            max_substeps_per_step: 256,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        config.gravity *= gravity_fraction;

        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: 78_480.0,
            yield_stress_pa: YIELD_PA,
            shear_modulus_pa: if viscous {
                0.0
            } else {
                YIELD_PA / YIELD_STRAIN
            },
        };
        let material: Box<dyn MaterialModel> = if neohookean {
            Box::new(NeoHookeanMaterial::from_young_modulus(YOUNG_PA, POISSON))
        } else if sand {
            Box::new(DruckerPragerMaterial::cohesionless(2000.0, 4000.0))
        } else if newtonian {
            // Built from the yield-stress material's OWN grid-unit fields,
            // so "same stiffness, same viscosity" holds by construction
            // rather than by redoing its unit conversion by hand.
            let bingham = BinghamFluidMaterial::from_physical(&props, &config);
            Box::new(NewtonianFluidMaterial::new(
                bingham.rest_density,
                bingham.dynamic_viscosity,
                bingham.eos_stiffness,
                bingham.eos_power,
            ))
        } else {
            Box::new(BinghamFluidMaterial::from_physical(&props, &config))
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(WIDTH_CELLS, thickness),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + thickness as f32 * 0.5),
            material_id: 0,
            mass_override: Some(RHO_KG_M3 * (0.5 * DX_M).powi(2)),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(material)
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let particles = sim.particles().len();

        // One step first: the spawn transient is common to every thickness
        // and measuring across it would put the same constant in every row.
        sim.step();
        let (start, _) = volume_state(&sim);
        let frames = (seconds / dt).round() as usize;
        let mut substeps = 0usize;
        for _ in 0..frames {
            sim.step();
            substeps += sim.diagnostics_snapshot().substeps_last_step;
        }
        let (end, worst) = volume_state(&sim);
        println!(
            "  {thickness:>6} cells  {particles:>8}    {start:>11.5}   {end:>7.5}   {:>14.5} %   {:>9.2e}   {worst:>9.5}",
            100.0 * (end - start) / f64::from(seconds),
            (end - start) / substeps.max(1) as f64
        );
    }
}

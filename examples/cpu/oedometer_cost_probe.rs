//! What does the `basic_oedometer` scene cost, and does its clay behave?
//!
//! Three identical kaolin samples differing only in the load they were
//! consolidated under, pressed the way the scene's cursor presses them.
//! Prints the substeps the CFL condition asks for, the frame cost, and for
//! each sample how far it sank and how much of that it kept, so the scene's
//! frame time rests on a measurement.
//!
//!   cargo run --release --example oedometer_cost_probe
//!   OEDO_PROBE_DT=0.0005 cargo run --release --example oedometer_cost_probe
extern crate emerge_engine as emerge;

use emerge::{
    Elastic, FromSI, NaccMaterial, NaccProps, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
/// 5 mm cells: a 32 cm box holding three 6 x 8 cm samples, the scale an
/// oedometer sample is actually cut at.
const DX_M: f32 = 0.005;
const FLOOR: f32 = 2.0;
const SAMPLE: IVec2 = IVec2::new(12, 16);
const SAMPLE_X: [f32; 3] = [12.0, 32.0, 52.0];
/// Spestone kaolin, Cambridge true triaxial tests (Muir Wood, Mackenzie and
/// Chan 1992): compression index, swelling index, void ratio at the
/// reference state, and the friction slope their stress ratio implies in 2D.
const LAMBDA: f32 = 0.245;
const KAPPA: f32 = 0.027;
const VOID_RATIO: f32 = 1.479;
const FRICTION_2D: f32 = 0.577;
/// The paper's elastic stiffness is not a constant: `K = v p' / kappa`, which
/// reads 13.8 MPa at their own 150 kPa. This scene works between 1 and
/// 10 kPa, and the model carries ONE bulk modulus (see the constant-modulus
/// entry in KNOWN_LIMITATIONS.md), so it is evaluated at 10 kPa, the highest
/// load any sample here remembers: `K = 2.479 * 10 kPa / 0.027 = 918 kPa`,
/// which is this Young's modulus at Poisson's ratio 0.270.
const YOUNG_PA: f32 = 1.27e6;
const POISSON: f32 = 0.270;
/// Saturated density at that void ratio, with kaolinite's own specific
/// gravity: rho = rho_w (G_s + e) / (1 + e).
const DENSITY: f32 = 1645.0;
/// The load each sample was consolidated under before the scene starts.
const PRECONSOLIDATION_PA: [f32; 3] = [0.0, 2.0e3, 10.0e3];

fn props(preconsolidation_pa: f32) -> NaccProps {
    NaccProps {
        elastic: Elastic {
            e_pa: YOUNG_PA,
            nu: POISSON,
            rho_kg_m3: DENSITY,
        },
        friction: FRICTION_2D,
        cohesion: 0.0,
        compression_index: LAMBDA,
        swelling_index: KAPPA,
        void_ratio: VOID_RATIO,
        preconsolidation_pa,
    }
}

fn main() {
    let dt: f32 = std::env::var("OEDO_PROBE_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.001);
    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: SAMPLE,
            box_center: Vec2::new(SAMPLE_X[slot], FLOOR + SAMPLE.y as f32 * 0.5),
            material_id: slot as u32,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(PRECONSOLIDATION_PA[slot]), &config)
    };
    let materials: [NaccMaterial; 3] = std::array::from_fn(|slot| {
        NaccMaterial::from_physical(&props(PRECONSOLIDATION_PA[slot]), &config)
    });
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(materials[0]))
        .with_material(1, Box::new(materials[1]))
        .with_material(2, Box::new(materials[2]))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));

    let top_of = |sim: &Simulation, slot: u32| {
        sim.particles()
            .iter()
            .filter(|p| p.material_id == slot)
            .map(|p| p.x.y)
            .fold(f32::MIN, f32::max)
    };

    let mut substeps = 0usize;
    let mut worst = 0usize;
    let settle_frames = (0.5 / dt).round() as usize;
    let wall = std::time::Instant::now();
    for _ in 0..settle_frames {
        sim.step();
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        worst = worst.max(s.substeps_last_step);
    }
    let settle_ms = wall.elapsed().as_secs_f64() * 1000.0 / settle_frames as f64;
    let before: Vec<f32> = (0..3).map(|slot| top_of(&sim, slot)).collect();
    for slot in 0..3usize {
        let settled = before[slot] - FLOOR;
        println!(
            "  sample {slot} (consolidated under {:.0} Pa): {settled:.2} cells tall after settling, {:.1} % of its spawn height",
            PRECONSOLIDATION_PA[slot],
            100.0 * settled / SAMPLE.y as f32
        );
    }
    println!(
        "dt={dt} settling: {:.1} substeps/frame (worst {worst}), {settle_ms:.2} ms/frame, {:.0} fps, {} particles",
        substeps as f64 / settle_frames as f64,
        1000.0 / settle_ms,
        sim.particles().len()
    );

    // Press each sample in turn, the way the scene's cursor does, then let
    // it recover.
    let press_frames = (0.3 / dt).round() as usize;
    let gravity = config.gravity.length();
    for slot in 0..3u32 {
        let press_g: f32 = std::env::var("OEDO_PROBE_PRESS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8.0);
        for frame in 0..press_frames {
            let top = top_of(&sim, slot);
            sim.apply_impulse(
                Vec2::new(SAMPLE_X[slot as usize], top),
                4.0,
                Vec2::new(0.0, -press_g * gravity * dt),
            );
            sim.step();
            if std::env::var("OEDO_PROBE_TRACE").is_ok() && frame % (press_frames / 6).max(1) == 0 {
                let mut j = 0.0;
                let mut p0 = 0.0;
                let mut n = 0.0;
                for p in sim.particles().iter().filter(|p| p.material_id == slot) {
                    j += p.deformation_gradient.determinant();
                    p0 += materials[slot as usize].preconsolidation_pressure(p.log_volume_strain);
                    n += 1.0;
                }
                println!(
                    "    trace slot {slot} frame {frame}: top={top:.3} mean_Je={:.4} p0={:.0} Pa",
                    j / n,
                    p0 / n * DENSITY * DX_M * DX_M
                );
            }
        }
        let sunk = before[slot as usize] - top_of(&sim, slot);
        for _ in 0..press_frames {
            sim.step();
        }
        let kept = before[slot as usize] - top_of(&sim, slot);
        let p0_mean = sim
            .particles()
            .iter()
            .filter(|p| p.material_id == slot)
            .map(|p| materials[slot as usize].preconsolidation_pressure(p.log_volume_strain))
            .sum::<f32>()
            / sim
                .particles()
                .iter()
                .filter(|p| p.material_id == slot)
                .count() as f32;
        println!(
            "  sample {slot} (consolidated under {:.0} Pa): sank {:.3} cells, kept {:.3} after release, p0 now {:.0} Pa",
            PRECONSOLIDATION_PA[slot as usize],
            sunk,
            kept,
            p0_mean * DENSITY * DX_M * DX_M
        );
    }
}

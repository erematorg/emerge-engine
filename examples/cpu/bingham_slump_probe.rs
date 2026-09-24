//! Does the slump test still read back the yield stress that went in?
//!
//! `basic_bingham` publishes a table in its header: three columns of one
//! material differing only in yield stress, each settling into a deposit
//! whose shape inverts back to a yield stress through
//! `tau_0 = rho g h^2 / (2 L)` (Liu and Mei 1989). That table had no way to
//! be reproduced without opening the window and reading the scene's own
//! printout. This runs the same scene headless and prints the same numbers,
//! so the claim in the doc rests on something anyone can re-run.
//!
//!   cargo run --release --example bingham_slump_probe
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, MaterialModel, SimConfig, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DX_M: f32 = 0.002;
const DT_S: f32 = 0.002;
const YIELD_STRESS_PA: [f32; 3] = [2.0, 60.0, 1200.0];
const COLUMN_LABEL: [&str; 3] = ["left", "mid", "right"];
const COLUMN_X: [f32; 3] = [12.0, 32.0, 52.0];
const RHO_KG_M3: f32 = 1000.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
const FLOOR_CELLS: f32 = 2.0;
const G: f32 = 9.81;

/// Same derivation as the scene's own: ten times the fastest speed free
/// fall over the column's height can produce, then `K = rho c^2`.
fn bulk_modulus_pa() -> f32 {
    let column_height_m = COLUMN_CELLS.y as f32 * DX_M;
    let v_max = (2.0 * G * column_height_m).sqrt();
    let c_ref = 10.0 * v_max;
    RHO_KG_M3 * c_ref * c_ref
}

fn main() {
    let seconds: f32 = std::env::var("SLUMP_PROBE_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, DT_S)
    };
    let k_pa = bulk_modulus_pa();
    let props = |tau0_pa: f32| BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: k_pa,
        yield_stress_pa: tau0_pa,
        shear_modulus_pa: tau0_pa / YIELD_STRAIN,
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: slot as u32,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(YIELD_STRESS_PA[slot]), &config)
    };
    let materials: [BinghamFluidMaterial; 3] = std::array::from_fn(|slot| {
        BinghamFluidMaterial::from_physical(&props(YIELD_STRESS_PA[slot]), &config)
    });
    let mut sim = emerge::Simulation::new(config, spawn(0))
        .with_default_material(Box::new(materials[0]))
        .with_material(1, Box::new(materials[1]))
        .with_material(2, Box::new(materials[2]))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));

    println!(
        "{} particles, bulk modulus {:.0} Pa, {} s of slump at {} ms a frame",
        sim.particles().len(),
        k_pa,
        seconds,
        DT_S * 1000.0
    );
    let frames = (seconds / DT_S).round() as usize;
    let (mut substeps, mut worst) = (0usize, 0usize);
    let wall = std::time::Instant::now();
    for frame in 0..frames {
        sim.step();
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        worst = worst.max(s.substeps_last_step);
        if frame % (frames / 10).max(1) == 0 {
            let mut line = format!("    t={:.2}s", frame as f32 * DT_S);
            for slot in 0..3u32 {
                let (mut top, mut lo, mut hi, mut v) = (f32::MIN, f32::MAX, f32::MIN, 0.0f32);
                for p in sim.particles().iter().filter(|p| p.material_id == slot) {
                    top = top.max(p.x.y);
                    lo = lo.min(p.x.x);
                    hi = hi.max(p.x.x);
                    v = v.max(p.v.length());
                }
                let mut peak = 0.0f32;
                for (i, p) in sim.particles().iter().enumerate() {
                    if p.material_id != slot {
                        continue;
                    }
                    let t = materials[slot as usize].kirchhoff_stress(sim.particles(), i);
                    let mean = 0.5 * (t.x_axis.x + t.y_axis.y);
                    let d = glam::Mat2::from_cols(
                        glam::Vec2::new(t.x_axis.x - mean, t.x_axis.y),
                        glam::Vec2::new(t.y_axis.x, t.y_axis.y - mean),
                    );
                    peak = peak.max(
                        (0.5 * (d.x_axis.length_squared() + d.y_axis.length_squared())).sqrt(),
                    );
                }
                let mut jsum = 0.0f64;
                let mut jn = 0u32;
                for p in sim.particles().iter().filter(|p| p.material_id == slot) {
                    jsum += f64::from(p.deformation_gradient.determinant());
                    jn += 1;
                }
                line += &format!(
                    "  {}[h={:.1} vmax={:.2} shear/yield={:.2} meanJ={:.4}]",
                    COLUMN_LABEL[slot as usize],
                    (top - FLOOR_CELLS).max(0.0) * DX_M * 1000.0,
                    v * DX_M,
                    peak / materials[slot as usize].yield_stress.max(1.0e-12),
                    jsum / f64::from(jn.max(1))
                );
            }
            println!("{line}");
        }
    }
    let ms = wall.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!(
        "  {:.1} substeps/frame (worst {worst}), {ms:.2} ms/frame, {:.0} fps, real time {:.3} s per second",
        substeps as f64 / frames as f64,
        1000.0 / ms,
        f64::from(DT_S) / (ms / 1000.0)
    );
    // The law's own answer, without going through any geometry: at rest a
    // yield-stress fluid holds a shear stress up to tau_0 and no further,
    // so the largest one standing in the deposit IS the yield stress it is
    // really obeying. Grid stress is already pascals here (see
    // tests/spawn_contract.rs for the derivation).
    for slot in 0..3 {
        let m = &materials[slot];
        println!(
            "  {} entered {:.0} Pa -> grid yield_stress={:.4} shear_modulus={:.2} eos_stiffness={:.2}",
            COLUMN_LABEL[slot],
            YIELD_STRESS_PA[slot],
            m.yield_stress,
            m.shear_modulus,
            m.eos_stiffness
        );
    }
    println!(
        "     tau_0 in   shear standing at rest, as a fraction of its own yield   deposit h    half-width L    tau_0 read back"
    );
    for slot in 0..3u32 {
        let (mut top, mut lo, mut hi, mut n) = (f32::MIN, f32::MAX, f32::MIN, 0u32);
        for p in sim.particles().iter().filter(|p| p.material_id == slot) {
            top = top.max(p.x.y);
            lo = lo.min(p.x.x);
            hi = hi.max(p.x.x);
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let h_m = (top - FLOOR_CELLS).max(0.0) * DX_M;
        let half_width_m = ((hi - lo) * 0.5).max(1.0e-6) * DX_M;
        let measured = RHO_KG_M3 * G * h_m * h_m / (2.0 * half_width_m);
        let parts = sim.particles();
        let mut peak_shear = 0.0f32;
        for i in 0..parts.len() {
            if parts.material_id[i] != slot {
                continue;
            }
            let tau = materials[slot as usize].kirchhoff_stress(parts, i);
            let mean = 0.5 * (tau.x_axis.x + tau.y_axis.y);
            let dev = glam::Mat2::from_cols(
                glam::Vec2::new(tau.x_axis.x - mean, tau.x_axis.y),
                glam::Vec2::new(tau.y_axis.x, tau.y_axis.y - mean),
            );
            // Second invariant, the measure a yield criterion is written in.
            let shear = (0.5 * (dev.x_axis.length_squared() + dev.y_axis.length_squared())).sqrt();
            peak_shear = peak_shear.max(shear);
        }
        println!(
            "  {:>5}  {:6.0} Pa                 {:6.3} of its own yield          {:6.1} mm      {:6.1} mm        {:8.0} Pa",
            COLUMN_LABEL[slot as usize],
            YIELD_STRESS_PA[slot as usize],
            peak_shear / materials[slot as usize].yield_stress.max(1.0e-12),
            h_m * 1000.0,
            half_width_m * 1000.0,
            measured
        );
    }
}

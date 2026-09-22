//! What does the `basic_nacc` scene cost, and what does it do headless?
//!
//! Runs the three clay columns without a window and prints, per column,
//! the deposit shape and the preconsolidation pressure it carries, plus
//! the engine's per-phase timing, so the default frame budget rests on a
//! measurement rather than a guess.
//!
//!   cargo run --release --example nacc_cost_probe
extern crate emerge_engine as emerge;

use emerge::Simulation;
use emerge::{
    Elastic, FromSI, MaterialModel, NaccMaterial, NaccProps, SimConfig, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DX_M: f32 = 0.01;
const FLOOR: f32 = 2.0;
const COLUMN: IVec2 = IVec2::new(10, 20);
const COLUMN_X: [f32; 3] = [12.0, 32.0, 52.0];

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let step_seconds = env_f32("NACC_PROBE_DT", 0.001);
    let seconds = env_f32("NACC_PROBE_SECONDS", 1.0);
    let e_pa = env_f32("NACC_PROBE_E", 5.0e6);
    let rho = env_f32("NACC_PROBE_RHO", 1600.0);

    let phi_deg = env_f32("NACC_PROBE_PHI", 23.0);
    let beta = env_f32("NACC_PROBE_BETA", 0.0);
    let pc: [f32; 3] = [
        env_f32("NACC_PROBE_PC0", 0.0),
        env_f32("NACC_PROBE_PC1", 3.0e3),
        env_f32("NACC_PROBE_PC2", 30.0e3),
    ];
    let sin_phi = phi_deg.to_radians().sin();
    let m = (8.0 / 3.0f32.sqrt()) * sin_phi / (3.0 - sin_phi);

    let config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, step_seconds)
    };
    let props = |p: f32| NaccProps {
        elastic: Elastic {
            e_pa,
            nu: 0.3,
            rho_kg_m3: rho,
        },
        friction: m,
        cohesion: beta,
        compression_index: 0.12,
        swelling_index: 0.023,
        void_ratio: 1.7,
        preconsolidation_pa: p,
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR + COLUMN.y as f32 * 0.5),
            material_id: slot as u32,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(pc[slot]).elastic, &config)
    };
    let mats: [NaccMaterial; 3] =
        std::array::from_fn(|slot| NaccMaterial::from_physical(&props(pc[slot]), &config));

    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(mats[0]))
        .with_material(1, Box::new(mats[1]))
        .with_material(2, Box::new(mats[2]))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));

    let pa_per_grid = rho * DX_M * DX_M;
    println!(
        "particles={} M={m:.3} E={e_pa:.2e} rho={rho} pc={pc:?} Pa  base sigma_v={:.0} Pa",
        sim.particles().len(),
        rho * 9.81 * COLUMN.y as f32 * DX_M
    );

    for (slot, mat) in mats.iter().enumerate() {
        let (mut sum, mut n) = (0.0f32, 0u32);
        for p in sim
            .particles()
            .iter()
            .filter(|p| p.material_id == slot as u32)
        {
            sum += mat.kappa
                * (1.0e-5 + (mat.hardening_factor * (-p.log_volume_strain).max(0.0)).sinh());
            n += 1;
        }
        println!(
            "t=0 column {slot}: p0mean={:.0} Pa",
            sum / n.max(1) as f32 * pa_per_grid
        );
    }
    let frames = (seconds / step_seconds).round() as usize;
    let report_every = (frames / 10).max(1);
    let mut substeps = 0usize;
    let mut total_us = 0u64;
    let mut p_max_seen = vec![0.0f32; sim.particles().len()];
    let wall = std::time::Instant::now();
    for frame in 0..frames {
        sim.step();
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        total_us += s.timing.total_us;
        {
            let parts = sim.particles();
            for i in 0..parts.len() {
                let mat = &mats[parts.material_id[i] as usize];
                let tau = mat.kirchhoff_stress(parts, i);
                let j = parts.deformation_gradient[i].determinant().max(1.0e-6);
                let p = -(tau.x_axis.x + tau.y_axis.y) * 0.5 / j;
                p_max_seen[i] = p_max_seen[i].max(p);
            }
        }
        if (frame + 1) % report_every == 0 {
            let mut line = format!("t={:.2}s", (frame + 1) as f32 * step_seconds);
            for (slot, mat) in mats.iter().enumerate() {
                let (mut top, mut lo, mut hi, mut n) = (f32::MIN, f32::MAX, f32::MIN, 0u32);
                let (mut p0_sum, mut p0_max, mut vmax) = (0.0f32, 0.0f32, 0.0f32);
                let (mut seen_sum, mut seen_max) = (0.0f32, 0.0f32);
                for (i, p) in sim.particles().iter().enumerate() {
                    if p.material_id != slot as u32 {
                        continue;
                    }
                    seen_sum += p_max_seen[i];
                    seen_max = seen_max.max(p_max_seen[i]);
                    top = top.max(p.x.y);
                    lo = lo.min(p.x.x);
                    hi = hi.max(p.x.x);
                    vmax = vmax.max(p.v.length());
                    let p0 = mat.preconsolidation_pressure(p.log_volume_strain);
                    p0_sum += p0;
                    p0_max = p0_max.max(p0);
                    n += 1;
                }
                line += &format!(
                    "  [h={:.1}cm w={:.1}cm p0mean={:.0}Pa p0max={:.0}Pa seen_mean={:.0}Pa seen_max={:.0}Pa v={:.2}m/s]",
                    (top - FLOOR) * DX_M * 100.0,
                    (hi - lo) * DX_M * 100.0,
                    p0_sum / n.max(1) as f32 * pa_per_grid,
                    p0_max * pa_per_grid,
                    seen_sum / n.max(1) as f32 * pa_per_grid,
                    seen_max * pa_per_grid,
                    vmax * DX_M,
                );
            }
            println!("{line}");
        }
    }
    let elapsed_ms = wall.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{frames} frames in {elapsed_ms:.0} ms -> {:.2} ms/frame, {:.1} fps, {:.1} substeps/frame, engine {:.2} ms/frame",
        elapsed_ms / frames as f64,
        1000.0 * frames as f64 / elapsed_ms,
        substeps as f64 / frames as f64,
        total_us as f64 / 1000.0 / frames as f64,
    );
}

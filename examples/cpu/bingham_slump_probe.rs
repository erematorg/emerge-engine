//! Does the slump test still read back the yield stress that went in?
//!
//! `basic_bingham` publishes a table in its header: three columns of one
//! material differing only in yield stress, each settling into a deposit
//! whose shape says something about its yield stress. This runs the demo's
//! own scene headless, from `bingham_slump_scene.rs`, which the demo
//! includes too, and prints the reading the demo's panel shows, so the claim
//! in the doc rests on something anyone can re-run and cannot drift from
//! the scene it describes.
//!
//! The reading is only given where its relation holds, and the relation is
//! chosen from the deposit's shape, never from the yield stress that went
//! in (see `SlumpWatch::read`): the long-wave force balance for a thin
//! deposit, Staron et al.'s fitted planar law for one that slumped without
//! thinning, a lower bound for a column that held its shape, and nothing
//! otherwise.
//!
//!   cargo run --release --example bingham_slump_probe
extern crate emerge_engine as emerge;

#[path = "bingham_slump_scene.rs"]
mod bingham_slump_scene;
use bingham_slump_scene::*;

use emerge::MaterialModel;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Largest shear standing in a column, `deviatoric_shear`, as a fraction of
/// its own yield stress: the law's own answer, without going through any
/// geometry.
fn standing_shear(
    sim: &emerge::Simulation,
    material: &emerge::BinghamFluidMaterial,
    slot: u32,
) -> f32 {
    let parts = sim.particles();
    let mut peak = 0.0f32;
    for i in 0..parts.len() {
        if parts.material_id[i] != slot {
            continue;
        }
        peak = peak.max(deviatoric_shear(material.kirchhoff_stress(parts, i)));
    }
    peak / material.yield_stress.max(1.0e-12)
}

fn main() {
    // Eight seconds by default: the softest column is still creeping at
    // 0.5 mm/s after that, and a deposit is only a measurement at rest.
    let seconds = env("SLUMP_PROBE_SECONDS", 8.0);
    // The demo's own default frame step.
    let dt = env("SLUMP_PROBE_DT", 0.001);
    let (mut sim, materials) = make_sim(1.0, 1.0, dt);

    println!(
        "{} particles, bulk modulus {:.0} Pa, {GRID}-cell tank, {seconds} s of slump at {} ms a frame",
        sim.particles().len(),
        bulk_modulus_pa(),
        dt * 1000.0
    );
    let frames = (seconds / dt).round() as usize;
    let (mut substeps, mut worst) = (0usize, 0usize);
    let wall = std::time::Instant::now();
    let mut watches = [SlumpWatch::default(); 3];
    for frame in 0..frames {
        sim.step();
        // Every frame, as the demo does: a reading needs the deposit's
        // fastest moment, not just its last.
        for (slot, watch) in watches.iter_mut().enumerate() {
            if let Some((h_m, half_width_m, speed)) = deposit(&sim, slot as u32) {
                watch.read(h_m, half_width_m, speed, 9.81);
            }
        }
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        worst = worst.max(s.substeps_last_step);
        if frame % (frames / 10).max(1) == 0 {
            let mut line = format!("    t={:.2}s", frame as f32 * dt);
            for slot in 0..3u32 {
                let Some((h_m, _, speed)) = deposit(&sim, slot) else {
                    continue;
                };
                let (mut jsum, mut jn) = (0.0f64, 0u32);
                for p in sim.particles().iter().filter(|p| p.material_id == slot) {
                    jsum += f64::from(p.deformation_gradient.determinant());
                    jn += 1;
                }
                line += &format!(
                    "  {}[h={:.1} vmax={:.4} shear/yield={:.2} meanJ={:.4}]",
                    COLUMN_LABEL[slot as usize],
                    h_m * 1000.0,
                    speed,
                    standing_shear(&sim, &materials[slot as usize], slot),
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
        f64::from(dt) / (ms / 1000.0)
    );
    println!(
        "     tau_0 in   deposit h   half-width L    h/L    standing shear   what the deposit says"
    );
    for slot in 0..3u32 {
        let Some((h_m, half_width_m, speed)) = deposit(&sim, slot) else {
            continue;
        };
        println!(
            "  {:>5} {:>6.0} Pa   {:>6.1} mm     {:>6.1} mm    {:>5.2}      {:>5.3}      {}",
            COLUMN_LABEL[slot as usize],
            YIELD_STRESS_PA[slot as usize],
            h_m * 1000.0,
            half_width_m * 1000.0,
            h_m / half_width_m,
            standing_shear(&sim, &materials[slot as usize], slot),
            describe(&watches[slot as usize].read(h_m, half_width_m, speed, 9.81))
        );
    }
}

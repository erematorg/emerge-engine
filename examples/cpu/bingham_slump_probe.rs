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
#[path = "../gui_common/cursor_traction.rs"]
mod cursor_traction;

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
    let mut peak_natural = [0.0f32; 3];
    for frame in 0..frames {
        sim.step();
        for (slot, peak) in peak_natural.iter_mut().enumerate() {
            if let Some((_, _, speed)) = deposit(&sim, slot as u32) {
                *peak = peak.max(speed);
            }
        }
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

    peak_speeds(dt, &peak_natural);
}

/// The fastest the scene moves, against the speed its stiffness assumes.
///
/// `bulk_modulus_pa` sizes the sound speed at ten times the free-fall speed
/// over a column's height, `sqrt(2 g H)`, the weakly compressible rule that
/// keeps density variation near one percent. That only holds while nothing
/// moves faster than that assumed speed. Measured here: the natural slump
/// (from the run above), then each column pushed for half a second and
/// released for half a second, at the panel's default push, 300 Pa, and at
/// its largest, 5000 Pa.
///
/// Found, 1 ms frames: the natural slump stays under the assumed 0.886 m/s
/// (0.662, 0.239 and 0.041 m/s for the 2, 60 and 1200 Pa columns), and so
/// does the default push (0.371, 0.265 and 0.077 m/s, at most 0.42 of it);
/// the largest push takes each column to about twice it (2.01, 1.87 and
/// 1.73 m/s, Mach 0.23, 0.21 and 0.20 against the sized 0.1), with a
/// traction measured under the cursor of 3950 to 5080 Pa. The earlier
/// scene's 0.787 m/s natural peak was the slip floor mostly: the 2 Pa column
/// alone peaks at 0.762 on a slip floor and 0.662 on this one
/// (`tests/scratch_bingham_isolated_slump.rs`).
fn peak_speeds(dt: f32, peak_natural: &[f32; 3]) {
    let assumed = (2.0 * 9.81 * COLUMN_CELLS.y as f32 * DX_M).sqrt();
    let sound = (bulk_modulus_pa() / RHO_KG_M3).sqrt();
    println!(
        "  peak speed: the stiffness assumes {assumed:.3} m/s, sound speed {sound:.2} m/s (Mach 0.1 at the assumed speed)"
    );
    let row = |v: f32| format!("{v:.3} m/s ({:.2} x, Mach {:.3})", v / assumed, v / sound);
    println!("     column   natural slump");
    for slot in 0..3usize {
        println!("  {:>7}   {}", COLUMN_LABEL[slot], row(peak_natural[slot]));
    }
    for push_pa in [300.0f32, 5000.0] {
        let (mut sim, _) = make_sim(1.0, 1.0, dt);
        let cursor = cursor_traction::CursorTraction::new(5.0, push_pa, push_pa)
            .with_lattice(RHO_KG_M3, SPACING, DX_M);
        sim.add_force_field(Box::new(cursor.field()));
        let frames = |seconds: f32| (seconds / dt).round() as usize;
        for _ in 0..frames(2.0) {
            sim.step();
        }
        let fastest = |sim: &emerge::Simulation| {
            (0..3u32)
                .filter_map(|k| deposit(sim, k).map(|(_, _, v)| v))
                .fold(0.0f32, f32::max)
        };
        println!(
            "     column   pushed at {push_pa:.0} Pa (traction measured, particles)   after release"
        );
        for slot in 0..3u32 {
            // On the column's top, where a player would press it.
            let top = deposit(&sim, slot).map_or(COLUMN_CELLS.y as f32, |(h_m, _, _)| h_m / DX_M);
            {
                let mut shared = cursor.shared();
                shared.position = glam::Vec2::new(COLUMN_X[slot as usize], FLOOR_CELLS + top);
                shared.pushing = true;
                shared.pulling = false;
            }
            let (mut push_peak, mut traction, mut touched) = (0.0f32, 0.0f32, 0usize);
            for _ in 0..frames(0.5) {
                sim.step();
                push_peak = push_peak.max(fastest(&sim));
                if let Some(contact) = cursor.shared().contact {
                    traction = traction.max(contact.traction_pa());
                    touched = touched.max(contact.particles);
                }
            }
            cursor.shared().pushing = false;
            let mut release_peak = 0.0f32;
            for _ in 0..frames(0.5) {
                sim.step();
                release_peak = release_peak.max(fastest(&sim));
            }
            println!(
                "  {:>7}   {} ({traction:.0} Pa, {touched})   {}",
                COLUMN_LABEL[slot as usize],
                row(push_peak),
                row(release_peak)
            );
        }
    }
}

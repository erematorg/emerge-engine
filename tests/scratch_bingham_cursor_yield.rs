//! Does the cursor's push, stated in pascals, yield each material at the
//! same multiple of its own yield stress?
//!
//! `CursorTraction` states a push `P` as a property of the cursor: its net
//! force over its diameter. A push `P` shears the material at about `P / 2`,
//! so it should yield a material near `P = 2 tau_0`, up to a factor of order
//! one set by the geometry. That absolute factor is not what is tested here.
//! What is tested is that it is the SAME factor for every material: the
//! push that yields a 60 Pa block must be thirty times the one that yields
//! a 2 Pa block, and twenty times less than the one for 1200 Pa. A cursor
//! whose threshold is not proportional to `tau_0` reports a number that
//! cannot be read against a yield stress, whatever it is called.
//!
//! Two things in the slump demo would spoil the measurement, and this
//! avoids both. Gravity preloads a column: one that has slumped sits
//! exactly ON its yield surface (the middle one measures 1.00 of it), so any
//! push at all yields it and its threshold reads zero. So there is no
//! gravity here. And in zero gravity a free block only slides away when
//! pushed, so each block rests against the right wall, which supplies the
//! reaction a shear needs.
//!
//! The yield criterion is permanent deformation. Below yield the
//! elastoviscoplastic branch is elastic and springs back once released;
//! above it the material flows and keeps the shape it was pushed into. So
//! each run pushes, releases, lets the elastic part recover, and measures
//! what is left, with the block's rigid motion taken out.
//!
//! # What it found
//!
//! The net force the field applies on a half-covered slab comes out at
//! 1.007 and 0.979 of `P * 2r` for two placements of the lattice, so the
//! `r^2 / 3` resultant holds on the real lattice to two percent. (Summing
//! the pushes' magnitudes instead, `pi r^2 / 6`, would read 1.57.)
//!
//! Change of shape after release, mm, translation and rotation removed:
//!
//! ```text
//!   tau_0 Pa   x0.5   x1     x1.5   x2     x3     x4     x6     x8
//!       2      0.025  0.072  0.244  0.556  1.264  1.817  2.553  3.058
//!      60      0.006  0.095  0.588  1.257  2.298  3.138  4.681  6.407
//!    1200      0.153  0.815  0.731  3.114  3.521  5.222  30.33  23.14
//! ```
//!
//! Monotone for 2 and 60 Pa. NOT for 1200 Pa, which dips from 0.815 at x1
//! to 0.731 at x1.5, and whose row is the noisy one (below).
//!
//! Onset is read where the change of shape rises fastest, the steepest
//! log-log slope between consecutive multiples: between x1 and x1.5 for
//! 2 Pa and 60 Pa (slopes 3.0 and 4.5), between x1.5 and x2 for 1200 Pa
//! (5.0), one step later. A fixed threshold in millimetres, first entry
//! past 0.2 mm or a hundredth of the block, gave x1.5, x1.5 and x1 instead;
//! it is not used because it reads each row against its own noise floor,
//! and that floor is not the same for the three. So the threshold is
//! proportional to `tau_0` to about one step of this grid, a factor of 1.5,
//! and not better. The 1200 Pa block's later onset is consistent with it
//! receiving a smaller share of `P`, below.
//!
//! The absolute threshold is LOWER than the `2 tau_0` the mean shear
//! `P / 2` predicts: the shear under the disk's centre exceeds the mean,
//! and the material yields near `P = 1` to `2 tau_0`.
//!
//! The 1200 Pa row is the weak one. Its floor below yield is higher, 0.15 mm
//! at x0.5, because below yield this branch is elastic with NO dissipation:
//! a released block rebounds off the wall and rings indefinitely, and the
//! averaging over the release only smooths that. A real gel damps. Above x6
//! the block is thrown out along the wall, 414 frames with nothing under the
//! cursor. That is this test's geometry failing, not the cursor, whose
//! largest per-substep push stays at 0.58 m/s, a fifteenth of the sound
//! speed. The traction the push really exerts over its contact differs by
//! material, 0.83, 0.55 and 0.43 of `P`, because a soft block wraps around
//! the cursor and a stiff one stays flat.
//!
//! # History
//!
//! The first cursor applied a whole frame's push as one velocity change
//! before the frame's substeps. Against 1200 Pa that single kick reached
//! 32.7 m/s, 3.7 times the sound speed, and the 1200 Pa row came out as
//! noise: 4.9 mm of permanent set at half the yield push, not monotone. The
//! cursor is now a force field the solver integrates every substep, and its
//! acceleration is fixed by `P` instead of being re-solved for whatever it
//! touches.
//!
//!   cargo test --profile quick --all-features --test scratch_bingham_cursor_yield -- --ignored --nocapture
extern crate emerge_engine as emerge;

#[path = "../examples/gui_common/cursor_traction.rs"]
mod cursor_traction;

use emerge::particle::{Particle, Particles};
use emerge::{
    BinghamFluidMaterial, BinghamProps, Field, FromSI, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};

// The slump demo's own material constants and scale.
const GRID: usize = 64;
const DX_M: f32 = 0.002;
const RHO_KG_M3: f32 = 1000.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const SPACING: f32 = 0.5;
/// The demo's bulk modulus, sound speed ten times the free-fall speed over
/// a 40 mm column: kept identical so the test reads the demo's materials.
const BULK_PA: f32 = 78_480.0;
const CURSOR_RADIUS: f32 = 5.0;
const BLOCK: IVec2 = IVec2::new(10, 10);
const YIELDS_PA: [f32; 3] = [2.0, 60.0, 1200.0];
/// Push as a multiple of each material's own yield stress. The same grid
/// of multiples for every material is what makes proportionality readable
/// straight off the table: the transition must fall in the same column.
const FACTORS: [f32; 8] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn cursor(push_pa: f32) -> cursor_traction::CursorTraction {
    cursor_traction::CursorTraction::new(CURSOR_RADIUS, push_pa, push_pa)
        .with_lattice(RHO_KG_M3, SPACING, DX_M)
}

/// The cursor's own definition, restated so the test does not take it from
/// the code under test: net force `P * 2r` from a disk half inside
/// material. Checked here against what the field actually applies.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_cursor_delivers_p_times_its_diameter_on_a_half_covered_slab() {
    const P: f32 = 1000.0;
    let radius_m = CURSOR_RADIUS * DX_M;
    let expected = P * 2.0 * radius_m;
    let mass_per_depth = RHO_KG_M3 * (SPACING * DX_M).powi(2);
    println!("cursor centred on the face of a slab filling x > 0, P = {P} Pa");
    println!("net force the field applies against P * 2r = {expected:.4} N/m:");
    println!(
        "  lattice                                  net N/m     ratio    reported N/m   width mm"
    );
    // Where the lattice sits relative to the cursor changes which discrete
    // points sample the disk. Two placements bracket that.
    for (label, x0, y0) in [
        (
            "sub-cell centres, off the face",
            0.5 * SPACING,
            0.5 * SPACING,
        ),
        ("first column on the face, row on axis", 0.0, 0.0),
    ] {
        let mut list = Vec::new();
        let span = (CURSOR_RADIUS / SPACING) as i32 + 2;
        for i in 0..span {
            for j in -span..=span {
                let mut p = Particle::zeroed();
                p.x = Vec2::new(x0 + i as f32 * SPACING, y0 + j as f32 * SPACING);
                list.push(p);
            }
        }
        let particles = Particles::from(list);
        let c = cursor(P);
        {
            let mut s = c.shared();
            s.position = Vec2::ZERO;
            s.pushing = true;
        }
        let mut field = c.field();
        field.prepare(&particles);
        // Summed here, from `acceleration`, independently of the field's own
        // report, which is printed beside it.
        let mut net = Vec2::ZERO;
        for i in 0..particles.len() {
            net += field.acceleration(&particles, i) * DX_M * mass_per_depth;
        }
        let reported = c.shared().contact.expect("the slab is under the cursor");
        println!(
            "  {label:<40} {:>9.4}   {:>6.3}    {:>9.4}     {:>6.2}",
            net.length(),
            net.length() / expected,
            reported.force_n_per_m,
            reported.width_m * 1000.0
        );
    }
}

/// RMS distance between two snapshots of the same particles once the best
/// rigid motion between them, translation AND rotation, is taken out
/// (two-dimensional Procrustes). What is left is change of shape.
fn shape_change(start: &[Vec2], now: &[Vec2]) -> f64 {
    let n = start.len() as f32;
    let c0 = start.iter().copied().sum::<Vec2>() / n;
    let c1 = now.iter().copied().sum::<Vec2>() / n;
    let (mut dot, mut cross) = (0.0f64, 0.0f64);
    for (a, b) in start.iter().zip(now) {
        let (p, q) = (*a - c0, *b - c1);
        dot += f64::from(p.dot(q));
        cross += f64::from(p.perp_dot(q));
    }
    let angle = cross.atan2(dot) as f32;
    let (sin, cos) = angle.sin_cos();
    let sum: f64 = start
        .iter()
        .zip(now)
        .map(|(a, b)| {
            let p = *a - c0;
            let rotated = Vec2::new(cos * p.x - sin * p.y, sin * p.x + cos * p.y);
            f64::from((*b - c1 - rotated).length_squared())
        })
        .sum();
    (sum / start.len() as f64).sqrt()
}

struct Outcome {
    /// Change of shape left after release, mm: rigid translation and
    /// rotation removed, averaged over the second half of the release.
    residual_mm: f64,
    /// Mean traction actually exerted over the contact while pushing,
    /// against the cursor's own `P`.
    exerted_over_p: f64,
    /// Frames on which the cursor found nothing to push.
    missed: usize,
    /// Upper bound on the velocity the cursor adds to one particle in one
    /// substep, m/s: its acceleration times the substep.
    kick_m_s: f64,
}

fn push_and_release(tau0: f32, push_pa: f32, push_s: f32, relax_s: f32, dt: f32) -> Outcome {
    let mut config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 512,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    config.gravity = Vec2::ZERO;
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: BULK_PA,
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    // Flush against the right wall, vertically centred.
    let wall = GRID as f32 - config.boundary_thickness as f32;
    let centre = Vec2::new(wall - BLOCK.x as f32 * 0.5, GRID as f32 * 0.5);
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: BLOCK,
        box_center: centre,
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(&props, &config);
    let c = cursor(push_pa);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_force_field(Box::new(c.field()));
    let start: Vec<Vec2> = sim.particles().x.clone();

    // The cursor is centred on the block's left face at mid-height: half
    // its disk inside material, the case `P` is defined on and the check
    // above measures. Every particle it touches lies to its right, so the
    // push nets out horizontally, into the wall.
    {
        let mut s = c.shared();
        s.position = Vec2::new(centre.x - BLOCK.x as f32 * 0.5, centre.y);
        s.pushing = true;
    }
    let accel_m_s2 = 6.0 * push_pa / (RHO_KG_M3 * CURSOR_RADIUS * DX_M);
    let (mut exerted, mut contact_frames, mut missed) = (0.0f64, 0usize, 0usize);
    let mut kick = 0.0f64;
    for _ in 0..(push_s / dt).round() as usize {
        sim.step();
        let substeps = sim.diagnostics_snapshot().substeps_last_step.max(1);
        kick = kick.max(f64::from(accel_m_s2 * dt) / substeps as f64);
        match c.shared().contact {
            Some(contact) => {
                exerted += f64::from(contact.traction_pa() / push_pa);
                contact_frames += 1;
                let _ = contact.particles;
            }
            None => missed += 1,
        }
    }
    c.shared().pushing = false;
    // Below yield this branch is elastic with no damping, so a released
    // block rebounds off the wall, turns and rings indefinitely. Rotation is
    // taken out by the fit; the ringing is averaged over the second half of
    // the release, whose mean is the shape it rings about.
    let relax_frames = (relax_s / dt).round() as usize;
    let (mut shape_sum, mut shape_samples) = (0.0f64, 0usize);
    for frame in 0..relax_frames {
        sim.step();
        if frame >= relax_frames / 2 {
            shape_sum += shape_change(&start, &sim.particles().x);
            shape_samples += 1;
        }
    }
    Outcome {
        residual_mm: shape_sum / shape_samples.max(1) as f64 * f64::from(DX_M) * 1000.0,
        exerted_over_p: exerted / contact_frames.max(1) as f64,
        missed,
        kick_m_s: kick,
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_cursor_yields_each_material_at_the_same_multiple_of_its_yield() {
    let push_s = env("CURSOR_PUSH_S", 0.2);
    let relax_s = env("CURSOR_RELAX_S", 0.2);
    let dt = env("CURSOR_DT", 0.001);
    println!(
        "zero gravity, block against the right wall, pushed {push_s} s then released {relax_s} s"
    );
    println!("change of shape after release, mm, rigid translation and rotation removed:");
    print!("  tau_0 Pa  ");
    for f in FACTORS {
        print!("  x{f:<5}");
    }
    println!("  exerted/P");
    let sound_m_s = (BULK_PA / RHO_KG_M3).sqrt();
    let mut kicks = Vec::new();
    for tau0 in YIELDS_PA {
        print!("  {tau0:>7}   ");
        let (mut exerted, mut missed) = (0.0, 0);
        let mut row = Vec::new();
        for f in FACTORS {
            let o = push_and_release(tau0, f * tau0, push_s, relax_s, dt);
            print!("  {:<6.3}", o.residual_mm);
            exerted += o.exerted_over_p;
            missed += o.missed;
            row.push(o.kick_m_s);
        }
        kicks.push((tau0, row));
        println!(
            "   {:.2}{}",
            exerted / FACTORS.len() as f64,
            if missed > 0 {
                format!("  ({missed} frames with nothing to push)")
            } else {
                String::new()
            }
        );
    }
    println!(
        "largest velocity one substep's push can add to a particle, m/s, against a sound speed of {sound_m_s:.2}:"
    );
    for (tau0, row) in kicks {
        print!("  {tau0:>7}   ");
        for k in row {
            print!("  {k:<6.3}");
        }
        println!();
    }
}

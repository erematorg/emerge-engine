//! Does the stress view (`ColorMode::ByStress`) turn red where a material
//! yields, before and after taking the pressure out of it?
//!
//! `MaterialRegistry::von_mises_stress_field` took the plane-stress von
//! Mises of the FULL stress, so a pure pressure read as its own magnitude.
//! The fix takes `sqrt(3 J2)` of the in-plane deviator alone. The only demo
//! that draws this field is `basic_vonmises`, which scales it by the soft
//! blobs' yield stress and says the colour saturates at yield onset. This
//! rebuilds that demo's scene headless (same constants as its `make_sim`)
//! and, while the blobs fall and land, compares three numbers per particle:
//!
//! - the old colour value, plane-stress von Mises of the full stress over
//!   the soft yield, as the demo computed it;
//! - the new one, `sqrt(3 J2)` over the same scale;
//! - the material's own criterion, `VonMisesMaterial::yield_ratio`:
//!   `2 mu |dev(eps)|` over `yield_stress + hardening_modulus * kappa`, by
//!   the code its return mapping runs. The demo now draws this.
//!
//! "Red" is a colour value of 0.75 or more, where the renderer's `heat`
//! map reaches full red.
//!
//! Found, over the drop and landing (frames 40 to 240), against the exact
//! criterion: the soft blob sits on its yield surface almost everywhere
//! once it lands (98.7 percent at frame 40, 97.3 at 240), so the old view's
//! all-red was right for it; its red under half the yield was 0.4 to 1.3
//! percent before the fix and 0 after. Taking the pressure out changes
//! the wrongly red share by under a point on this scene. The hardening and
//! stiff blobs stay wrong either way (74.0 and 35.6 percent red under half
//! their yield at 240 before, 73.3 and 35.3 after): the one display scale
//! is the soft blob's initial yield, which the hardening blob outgrows
//! through kappa and the stiff blob exceeds 25 times.
//!
//! A stress-based view also under-reads yield at these strains: with the
//! soft yield as its scale, only 20.8 percent of the soft blob is red at
//! frame 40 while 98.7 percent is on its surface. The criterion is on the
//! Hencky strain, `2 mu |dev(eps)|`, which the deviator of the corotated
//! stress matches only at small strain, and these blobs carry accumulated
//! plastic strains of order one. An earlier version of this probe
//! approximated the criterion by that stress and reported that the fix
//! took the soft blob's wrongly red share from 74 percent to 0; that came
//! from the approximation, not from the view. The view divided by each
//! particle's own yield, `yield_ratio`, is what the demo now draws.
//!
//! Also found: the stiff blob, which the demo's header meant to stay close
//! to elastic, yields everywhere on landing. Every particle is past 0.01 of
//! accumulated plastic strain by frame 40, median 0.88 (soft 1.07, hardening
//! 1.04), and 1.72 by frame 240.
//!
//!   cargo test --profile quick --all-features --test scratch_stress_view_before_after -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{MaterialModel, SimConfig, Simulation, SlipBoundary, SpawnRegion, VonMisesMaterial};
use glam::{IVec2, Mat2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const LAMBDA: f32 = 30.0;
const MU: f32 = 60.0;

/// The formula `von_mises_stress_field` applied before the fix.
fn full_plane_stress(s: Mat2) -> f32 {
    let (sxx, syy) = (s.x_axis.x, s.y_axis.y);
    let sxy = 0.5 * (s.x_axis.y + s.y_axis.x);
    (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
        .max(0.0)
        .sqrt()
}

/// `sqrt(J2)` of the in-plane deviator, `J2 = s:s / 2`.
fn root_j2(s: Mat2) -> f32 {
    let half_diff = 0.5 * (s.x_axis.x - s.y_axis.y);
    let sxy = 0.5 * (s.x_axis.y + s.y_axis.x);
    (half_diff * half_diff + sxy * sxy).sqrt()
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_stress_view_on_the_von_mises_scene() {
    let mut config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 8,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    config.gravity *= 0.003;
    let spawn = |c: Vec2, mat| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: c,
        material_id: mat,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let materials = [
        VonMisesMaterial::new(LAMBDA, MU, MU * 0.01),
        VonMisesMaterial::with_hardening(LAMBDA, MU, MU * 0.01, MU * 0.03),
        VonMisesMaterial::new(LAMBDA * 5.0, MU * 5.0, MU * 5.0 * 0.05),
    ];
    let mut sim = Simulation::new(config, spawn(Vec2::new(14.0, 20.0), 0))
        .with_default_material(Box::new(materials[0]))
        .with_material(1, Box::new(materials[1]))
        .with_material(2, Box::new(materials[2]))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(Vec2::new(32.0, 20.0), 1));
    let _ = sim.add_body(spawn(Vec2::new(50.0, 20.0), 2));

    // The demo's own scale: one over the soft blobs' yield stress.
    let scale = 1.0 / (MU * 0.01);
    println!(
        "frame  blob    at yield   old: red  red and under half its yield   new: red  red and under half its yield"
    );
    let mut frame = 0;
    for look in [5, 10, 20, 40, 80, 160, 240] {
        while frame < look {
            sim.step();
            frame += 1;
        }
        let p = sim.particles();
        // The engine's own field, through its dispatch, must be the new
        // value exactly: that is the path the demo draws.
        let field = sim.materials().von_mises_stress_field(p);
        for (i, &f) in field.iter().enumerate() {
            let m = &materials[p.material_id[i] as usize];
            let mine = 3.0f32.sqrt() * root_j2(m.kirchhoff_stress(p, i));
            assert!(
                (f - mine).abs() <= 1.0e-4 * mine.max(1.0),
                "frame {frame}, particle {i}: engine {f}, sqrt(3 J2) {mine}"
            );
        }
        for (slot, (label, m)) in ["soft", "hard", "stiff"].iter().zip(&materials).enumerate() {
            let idx: Vec<usize> = (0..p.len())
                .filter(|&i| p.material_id[i] == slot as u32)
                .collect();
            let n = idx.len().max(1) as f32;
            let (mut at_yield, mut old_red, mut old_false, mut new_red, mut new_false) =
                (0, 0, 0, 0, 0);
            for &i in &idx {
                let tau = m.kirchhoff_stress(p, i);
                let own = m.yield_ratio(p, i);
                let old = full_plane_stress(tau) * scale;
                let new = 3.0f32.sqrt() * root_j2(tau) * scale;
                at_yield += usize::from(own >= 0.95);
                old_red += usize::from(old >= 0.75);
                old_false += usize::from(old >= 0.75 && own < 0.5);
                new_red += usize::from(new >= 0.75);
                new_false += usize::from(new >= 0.75 && own < 0.5);
            }
            let pc = |k: usize| 100.0 * k as f32 / n;
            // Accumulated equivalent plastic strain: nonzero only where the
            // return mapping actually fired, the direct record of yielding.
            let kappa = idx
                .iter()
                .map(|&i| p.friction_hardening[i])
                .fold(0.0f32, f32::max);
            // More than one percent of plastic strain: yielding that shows.
            let yielded = idx
                .iter()
                .filter(|&&i| p.friction_hardening[i] > 0.01)
                .count();
            let mut sorted: Vec<f32> = idx.iter().map(|&i| p.friction_hardening[i]).collect();
            sorted.sort_by(f32::total_cmp);
            let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
            println!(
                "{frame:>5}  {label:<6}  {:>6.1} %   {:>6.1} %   {:>6.1} %                        {:>6.1} %   {:>6.1} %   kappa max {kappa:.4}, median {median:.4}, {:.0} % past 0.01",
                pc(at_yield),
                pc(old_red),
                pc(old_false),
                pc(new_red),
                pc(new_false),
                pc(yielded)
            );
        }
    }
}

/// Can these blobs hold their own weight at all? A block of height `h` on a
/// floor, free at its sides, carries a vertical stress `rho g h` at its
/// base; its in-plane deviator is then `rho g h / 2` on each diagonal term,
/// so the Frobenius norm the criterion tests is `rho g h / sqrt(2)`. Below
/// the yield stress the block stands; above it the weight alone makes it
/// flow, impact or not. Density read off the spawned particles, in the same
/// grid units as the yield stress; at the demo's gravity (0.003 of earth's)
/// and at earth's.
///
/// Found: at the demo's gravity the weight alone puts 26.9 at the base,
/// 45 times the soft blobs' yield and 1.8 times the stiff blob's; at earth's
/// gravity 14,946 and 598 times. None of the three can stand even at rest,
/// so they flow whatever the impact, and the reduced gravity was a crutch
/// that does not even hold them (issue #46).
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_blobs_against_their_own_weight() {
    for fraction in [0.003f32, 1.0] {
        let mut config = SimConfig {
            min_dt: 0.01,
            max_substeps_per_step: 8,
            ..SimConfig::earth(GRID, 0.01, DT)
        };
        config.gravity *= fraction;
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(14, 14),
            box_center: Vec2::new(14.0, 20.0),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(VonMisesMaterial::new(LAMBDA, MU, MU * 0.01)));
        let p = sim.particles();
        let rho = (0..p.len()).map(|i| p.density[i]).sum::<f32>() / p.len() as f32;
        let weight = rho * config.gravity.length() * 14.0 / std::f32::consts::SQRT_2;
        println!(
            "gravity {fraction} of earth: density {rho:.3}, weight's deviator at the base {weight:.2}; soft yield {:.2} ({:.0}x), stiff yield {:.2} ({:.0}x)",
            MU * 0.01,
            weight / (MU * 0.01),
            MU * 5.0 * 0.05,
            weight / (MU * 5.0 * 0.05)
        );
    }
}

/// The same scene's blob shapes, to compare with the scene as first
/// committed (9aacdff): height and width in cells, mean density, particle
/// mass and median accumulated plastic strain, same lines as a copy of this
/// test run on that commit.
///
/// Found: the two runs agree within 0.001 in median kappa and 0.1 cell in
/// height at every look, with the same widths and densities (soft blob 13.5
/// cells tall at frame 0, 5.1 at frame 40, median kappa 1.072, 9.445 at
/// frame 240). The scene has not changed since it was written: its blobs
/// collapsed and flowed from the first day.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_blob_shapes_now() {
    let mut config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 8,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    config.gravity *= 0.003;
    let spawn = |c: Vec2, mat| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: c,
        material_id: mat,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn(Vec2::new(14.0, 20.0), 0))
        .with_default_material(Box::new(VonMisesMaterial::new(LAMBDA, MU, MU * 0.01)))
        .with_material(
            1,
            Box::new(VonMisesMaterial::with_hardening(
                LAMBDA,
                MU,
                MU * 0.01,
                MU * 0.03,
            )),
        )
        .with_material(
            2,
            Box::new(VonMisesMaterial::new(
                LAMBDA * 5.0,
                MU * 5.0,
                MU * 5.0 * 0.05,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(Vec2::new(32.0, 20.0), 1));
    let _ = sim.add_body(spawn(Vec2::new(50.0, 20.0), 2));
    let mut frame = 0;
    for look in [0, 40, 80, 240] {
        while frame < look {
            sim.step();
            frame += 1;
        }
        let p = sim.particles();
        for slot in 0..3u32 {
            let idx: Vec<usize> = (0..p.len()).filter(|&i| p.material_id[i] == slot).collect();
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            let mut rho = 0.0f32;
            let mut k: Vec<f32> = Vec::new();
            for &i in &idx {
                lo = lo.min(p.x[i]);
                hi = hi.max(p.x[i]);
                rho += p.density[i];
                k.push(p.friction_hardening[i]);
            }
            k.sort_by(f32::total_cmp);
            println!(
                "frame {frame:>3} blob {slot}: height {:.1} width {:.1} cells, mean density {:.3}, mass {:.4}, kappa median {:.3}",
                hi.y - lo.y,
                hi.x - lo.x,
                rho / idx.len() as f32,
                p.mass[idx[0]],
                k[k.len() / 2]
            );
        }
    }
}

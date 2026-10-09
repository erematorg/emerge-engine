//! `rod::gravitropism::GravitropismMode::WholeOrgan` (Bastien, Bohr, Moulia,
//! Douady 2013, PNAS 110(2):755-760) and the sleep freeze
//! (`Rod::is_correcting_gravitropically`) it depends on: a rod that falls
//! asleep mid-correction never finishes, however correct the formula.
//!
//! # Scope boundary
//! `WholeOrgan` recovers a pushed organ's shape only when that shape is
//! mechanically stable (below its Euler/Greenhill self-weight buckling
//! height, see `Rod::buckling_warning`). The same push/release/correct
//! sequence converges to vertical and stays there on blade A (under its
//! critical height), but on blade B (built over it) settles into a sustained,
//! non-decaying oscillation whatever the gain (tested across a 500x range) and
//! whether the correction is per-vertex or a single global-chord signal. Not a
//! bug: past its critical height, straight is an unstable equilibrium for the
//! organ's EA/EI/mass, and no curvature-target correction can hold a structure
//! at an unstable equilibrium, as no amount of balancing keeps a pencil on its
//! tip. Plants handle buckling with another mechanism, secondary growth
//! (thigmomorphogenesis: growing a stiffer, thicker stem), not gravitropism.
//! `whole_organ_gravitropism_cannot_rescue_genuine_structural_buckling`
//! below tests the boundary.

extern crate emerge_engine as emerge;
use emerge::rod::{Gravitropism, GravitropismMode, Rod, RodMaterial, build_straight_rod};
use emerge::{SimConfig, Simulation};
use glam::Vec2;

const DX_METERS: f32 = 0.01;
const HEIGHT_M: f32 = 0.10;
const START: Vec2 = Vec2::new(9.0, 4.0);

/// Same real construction as `rod_blade_of_grass_gui.rs`'s two blades: 20
/// points, 0.003x0.001m rectangular section, `modal_critical_damping` at a
/// 0.15 fraction, implicit integration -- `young_modulus=1e7` reproduces
/// blade A (safely under its own Greenhill critical height), `5e6`
/// reproduces blade B (deliberately over it). `tilt_rad` (signed, from
/// vertical) builds the rod ALREADY leaning at that angle -- `build_straight_rod`
/// sets `rest_curvature=0` regardless of the chosen direction, so a rod
/// built at a tilt is passively STABLE at that same tilt (its own straight
/// rest shape already matches its current shape) -- this is deliberately a
/// "grown crooked" organ, not a momentary push, since a momentary push on a
/// mechanically stable blade is recovered by passive elasticity ALONE
/// (confirmed empirically) and can't demonstrate gravitropism's own real,
/// distinct value.
fn make_blade(young_modulus: f32, tilt_rad: f32, gravitropism: Option<Gravitropism>) -> Rod {
    let height_cells = HEIGHT_M / DX_METERS;
    let end = START + Vec2::new(tilt_rad.sin(), tilt_rad.cos()) * height_cells;
    let mut rod_points = build_straight_rod(START, end, 20, 0.01, DX_METERS);
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let ea = young_modulus * 0.003 * 0.001;
    let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
    let (axial_critical, bending_critical) =
        RodMaterial::modal_critical_damping(&rod_points, ea, ei);
    let frac = 0.15;
    let material = RodMaterial::from_young_modulus_rectangular(
        young_modulus,
        0.003,
        0.001,
        axial_critical * frac,
        bending_critical * frac,
    );
    let mut rod = Rod::new(rod_points, material);
    rod.push_radius = 3.0;
    rod.use_implicit_integration = true;
    rod.gravitropism = gravitropism;
    rod
}

fn make_sim(young_modulus: f32, tilt_rad: f32, gravitropism: Option<Gravitropism>) -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 5000,
        min_dt: 1.0e-8,
        rod_sleep_threshold: 0.02,
        ..SimConfig::earth(32, DX_METERS, 0.02)
    };
    let mut solver = Simulation::empty(config);
    solver.add_rod(make_blade(young_modulus, tilt_rad, gravitropism));
    solver
}

/// Same sustained hover-push contract `rod_blade_of_grass_gui.rs`
/// itself uses (persistent forcing every substep, not a one-shot impulse) --
/// hard enough to knock either blade into a large, real deviation.
fn push_hard(solver: &mut Simulation) {
    for _ in 0..100 {
        {
            let rod = &mut solver.rods_mut()[0];
            rod.push_center = Some(Vec2::new(START.x + 1.5, START.y + 8.0));
            rod.push_strength = 400.0;
        }
        solver.step();
    }
    let rod = &mut solver.rods_mut()[0];
    rod.push_center = None;
    rod.push_strength = 0.0;
}

fn tip_x_offset(solver: &Simulation) -> f32 {
    let points = &solver.rods()[0].points;
    points.x[points.len() - 1].x - START.x
}

#[test]
fn gravitropism_keeps_correcting_after_the_rod_would_otherwise_have_slept() {
    // A rod that starts misaligned (45 degrees off its GSA=0
    // target) but with near-zero initial velocity -- the elastic system has
    // nothing to mechanically settle, so any wake-time here is coming ONLY
    // from gravitropism's own ongoing correction, isolating the fix from
    // ordinary elastic-settle wake time.
    let dx_meters = 1.0;
    let mut points =
        build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.7, 0.7), 6, 0.01, dx_meters);
    points.pinned[0] = 1;
    points.pinned[1] = 1;
    let ea = 1.0e5 * 0.003 * 0.001;
    let ei = 1.0e5 * 0.003_f32.powi(3) * 0.001 / 12.0;
    // Deliberately SLOW sensitivity: real convergence must take longer than
    // `ROD_SLEEP_SETTLE_MAX_SECONDS` (8s) so this test can distinguish "still
    // correcting" from "already converged, correctly free to sleep."
    let gravitropism = Gravitropism::new(0.02, 0.002);
    let material = RodMaterial::new(ea, ei, ea * 0.5, ei * 0.5);
    let mut rod = Rod::new(points, material);
    rod.use_implicit_integration = true;
    rod.gravitropism = Some(gravitropism);

    let config = SimConfig {
        max_substeps_per_step: 5000,
        min_dt: 1.0e-8,
        rod_sleep_threshold: 0.02,
        ..SimConfig::earth(32, dx_meters, 0.02)
    };
    let mut solver = Simulation::empty(config);
    solver.add_rod(rod);

    // Step past the 8s max settle window -- must still be awake.
    for _ in 0..500 {
        // 500 * 0.02s = 10s
        solver.step();
    }
    assert!(
        !solver.rods()[0].sleeping,
        "a rod with real, unconverged gravitropism must not sleep just because \
         ROD_SLEEP_SETTLE_MAX_SECONDS elapsed"
    );

    // Keep stepping until gravitropism converges -- must
    // eventually sleep once it does (bounds the fix the other direction:
    // this isn't a rod that can NEVER sleep).
    let mut slept = false;
    for _ in 0..9500 {
        // up to 190 more simulated seconds
        solver.step();
        if solver.rods()[0].sleeping {
            slept = true;
            break;
        }
    }
    assert!(
        slept,
        "a rod must eventually sleep once gravitropism genuinely converges, not stay \
         awake forever"
    );
}

#[test]
fn whole_organ_gravitropism_recovers_a_grown_crooked_blade_while_tip_only_plateaus() {
    // Blade A -- E=1e7, confirmed SAFELY under its own Greenhill critical
    // height, so "straight" really is a mechanically stable shape here
    // (unlike blade B -- see this file's module doc). Built ALREADY
    // leaning 30 degrees off vertical -- NOT a momentary push (a
    // mechanically stable blade's own passive elasticity recovers a
    // momentary push on its own, confirmed empirically, so that can't show
    // gravitropism's distinct value). This is the actually-real case
    // gravitropism exists for: an organ whose GROWN rest shape itself needs
    // active correction (the same real scenario as the root's own 45-degree
    // start), not a transient nudge elastic springback already handles.
    const YOUNG_MODULUS_STABLE: f32 = 1.0e7;
    const TILT_RAD: f32 = 30.0 * std::f32::consts::PI / 180.0;

    let mut control = make_sim(YOUNG_MODULUS_STABLE, TILT_RAD, None);
    let mut tip_only = make_sim(
        YOUNG_MODULUS_STABLE,
        TILT_RAD,
        Some(Gravitropism::new(0.05, 0.005).with_gsa(std::f32::consts::PI)),
    );
    let mut whole_organ = make_sim(
        YOUNG_MODULUS_STABLE,
        TILT_RAD,
        Some(
            Gravitropism::new(0.05, 0.005)
                .with_gsa(std::f32::consts::PI)
                .with_mode(GravitropismMode::WholeOrgan),
        ),
    );

    let baseline_offset = tip_x_offset(&control).abs();
    assert!(
        baseline_offset > 1.0,
        "a 30-degree lean must be a real, meaningful deviation to recover from, got {baseline_offset}"
    );

    // Generous horizon -- gravitropism is a slow, real biological
    // process, not an instant snap.
    const HORIZON_STEPS: usize = 15_000; // 300 simulated seconds
    for _ in 0..HORIZON_STEPS {
        control.step();
        tip_only.step();
        whole_organ.step();
    }
    // No gravitropism at all: a tilt introduces a bending
    // moment from self-weight (absent for a perfectly vertical column,
    // where self-weight is purely axial) -- real additional elastic SAG is
    // expected and fine, this only checks passive elasticity has no reason
    // to REORIENT back toward vertical on its own.
    let control_offset = tip_x_offset(&control).abs();
    assert!(
        control_offset > baseline_offset * 0.8,
        "with no gravitropism, a stable organ built leaning should stay leaning (real \
         additional self-weight sag is fine, reorienting toward vertical is not): \
         baseline={baseline_offset:.4} control={control_offset:.4}"
    );

    let tip_only_offset = tip_x_offset(&tip_only).abs();
    let whole_organ_offset = tip_x_offset(&whole_organ).abs();

    assert!(
        whole_organ_offset < tip_only_offset * 0.5,
        "WholeOrgan must recover substantially more than TipOnly for a shape deviation \
         spread across the whole organ, not just its tip: \
         control={control_offset:.4} tip_only={tip_only_offset:.4} whole_organ={whole_organ_offset:.4}"
    );
    assert!(
        whole_organ_offset < 0.5,
        "WholeOrgan must recover close to true vertical for a mechanically stable, \
         grown-crooked organ: baseline={baseline_offset:.4} whole_organ={whole_organ_offset:.4}"
    );
}

#[test]
fn whole_organ_gravitropism_cannot_rescue_genuine_structural_buckling() {
    // Blade B -- E=5e6, deliberately built OVER its own Greenhill critical
    // height (confirmed via `Rod::buckling_warning`). This is a REAL,
    // disclosed negative control, not a silently-passing edge case: no
    // curvature-target correction can hold an organ at a unstable
    // equilibrium, so this must NOT converge to true vertical, no matter how
    // long it runs. If this test ever starts passing as "recovered," that's
    // a sign something about blade B's own parameters drifted below its
    // real critical height elsewhere in the codebase, not a gravitropism
    // improvement -- re-check `buckling_warning` first.
    const YOUNG_MODULUS_BUCKLING: f32 = 5.0e6;

    let mut whole_organ = make_sim(
        YOUNG_MODULUS_BUCKLING,
        0.0,
        Some(
            Gravitropism::new(0.05, 0.005)
                .with_gsa(std::f32::consts::PI)
                .with_mode(GravitropismMode::WholeOrgan),
        ),
    );
    push_hard(&mut whole_organ);

    // Run well past one full real oscillation period (empirically ~90-100s)
    // and sample near the end -- a converged organ would show
    // every sample small; a sustained oscillation shows at least one large.
    const HORIZON_STEPS: usize = 15_000; // 300 simulated seconds
    let mut max_late_offset = 0.0_f32;
    for step in 0..HORIZON_STEPS {
        whole_organ.step();
        if step >= HORIZON_STEPS - 5_000 && step % 500 == 0 {
            max_late_offset = max_late_offset.max(tip_x_offset(&whole_organ).abs());
        }
    }

    assert!(
        max_late_offset > 2.0,
        "a genuinely over-critical organ must NOT converge to true vertical under \
         WholeOrgan gravitropism -- if it did, that's a sign blade B's own parameters \
         no longer exceed its real Greenhill critical height, not a fix; \
         max_late_offset={max_late_offset:.4}"
    );
}

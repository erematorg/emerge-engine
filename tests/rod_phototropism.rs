//! `rod::gravitropism`'s `Phototropism`/`apply_phototropism`: Cholodny & Went
//! auxin-asymmetry theory, sharing the SAME curvature-relaxation core as
//! gravitropism (see that module's doc). Proves the tip rotates toward a
//! sensed light direction through the full mechanical pipeline (grid
//! coupling, elastic bending, damping, ordinary gravity sag), isolated via
//! A/B (phototropism on vs off) exactly like `tests/rod_gravitropism.rs`'s
//! own gravitropism test -- gravity is present and identical in BOTH arms
//! (real self-weight sag happens either way), while `light_dir` points a
//! DIFFERENT direction than gravity so ordinary sag alone cannot explain any
//! alignment gained with it.

extern crate emerge_engine as emerge;
use emerge::rod::{Phototropism, RodMaterial, build_straight_rod};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation};
use glam::Vec2;

/// Same fixture shape as `tests/rod_gravitropism.rs`'s `build_rod`: a short,
/// stiff, heavily-damped horizontal rod (pinned at the base, pointing due
/// +x). `light_dir` points straight UP -- opposite gravity's own downward
/// sag direction -- so any measured alignment with `light_dir` can only
/// come from phototropism itself, not from ordinary self-weight sag (which
/// would instead rotate the tip DOWN, away from "up").
fn build_rod(phototropism: Option<Phototropism>) -> Simulation {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        light_dir: Vec2::new(0.0, 1.0),
        max_substeps_per_step: 400,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    let span = 1.5;
    let n_points = 8usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(10.0, 24.0),
        Vec2::new(10.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 5.0e6 * 0.06 * 0.02;
    let ei = 5.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        5.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    let mut rod = emerge::rod::Rod::new(rod_points, material);
    if let Some(p) = phototropism {
        rod = rod.with_phototropism(p);
    }
    solver.add_rod(rod);
    solver
}

/// 1.0 = tip direction exactly aligned with `light_dir` (straight up), 0.0 =
/// perpendicular (still horizontal, the initial condition), negative =
/// rotated AWAY from light (e.g. ordinary downward gravity sag).
fn tip_light_alignment(solver: &Simulation) -> f32 {
    let points = &solver.rods()[0].points;
    let n = points.len();
    let tip_edge = points.x[n - 1] - points.x[n - 2];
    let tip_dir = tip_edge.normalize_or_zero();
    let light_dir = solver.config().light_dir.normalize_or_zero();
    tip_dir.dot(light_dir)
}

#[test]
fn phototropism_rotates_tip_toward_light_more_than_gravity_sag_alone() {
    let mut baseline = build_rod(None);
    let mut with_phototropism = build_rod(Some(Phototropism::new(0.05, 0.005)));

    let alignment_before = tip_light_alignment(&baseline);
    assert!(
        alignment_before.abs() < 0.05,
        "test setup isn't actually horizontal/perpendicular to light_dir: \
         alignment={alignment_before:.4}"
    );

    baseline.step_n(6000);
    with_phototropism.step_n(6000);

    let baseline_alignment = tip_light_alignment(&baseline);
    let phototropism_alignment = tip_light_alignment(&with_phototropism);

    assert!(
        baseline_alignment < 0.05,
        "baseline (gravity sag alone, no phototropism) must NOT gain real alignment with \
         a light direction opposite gravity: baseline={baseline_alignment:.4}"
    );
    assert!(
        phototropism_alignment > baseline_alignment + 0.1,
        "phototropism should rotate the tip measurably more toward light than gravity sag \
         alone: baseline={baseline_alignment:.4} with_phototropism={phototropism_alignment:.4}"
    );
}

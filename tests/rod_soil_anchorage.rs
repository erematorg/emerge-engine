//! A root embedded in sand gets mechanical anchorage from ordinary granular
//! contact, with no special "root anchor" mechanism: the rod<->grid coupling
//! and Drucker-Prager sand contact used elsewhere. A/B: identical root and
//! lateral push, with sand around it vs bare (empty grid); anchorage shows as
//! measurably less displacement.

extern crate emerge_engine as emerge;
use emerge::rod::{RodMaterial, build_straight_rod};
use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// A vertical, entirely UNPINNED root (no artificial clamp anywhere -- real
/// anchorage, if any, must come purely from the surrounding sand's granular
/// contact), pushed sideways near its top. `with_sand` controls whether it's
/// actually embedded in a sand block or sitting in an empty grid.
fn pushed_root(with_sand: bool, steps: usize) -> f32 {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 300,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = if with_sand {
        let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
        let sand_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(24, 24),
            box_center: Vec2::new(24.0, 14.0),
            ..SpawnRegion::for_sim(&config)
        };
        Simulation::new(config, sand_spawn).with_default_material(Box::new(sand))
    } else {
        Simulation::empty(config)
    };

    // Vertical root, base at y=6 (well inside the sand block), tip at y=20,
    // fully embedded when `with_sand` -- no pin anywhere.
    let span = 14.0;
    let n_points = 16usize;
    let rod_points = build_straight_rod(
        Vec2::new(24.0, 6.0),
        Vec2::new(24.0, 6.0 + span),
        n_points,
        0.3,
        1.0,
    );
    let material = RodMaterial::from_young_modulus_rectangular(3.0e6, 0.06, 0.02, 50.0, 5.0);
    let mut rod = emerge::rod::Rod::new(rod_points, material);
    // Sustained lateral push near the top -- a continuous toppling
    // force (wind-on-a-plant analog), not a one-shot impulse.
    rod.push_center = Some(Vec2::new(24.0, 18.0));
    rod.push_strength = 100.0;
    rod.push_radius = 4.0;
    solver.add_rod(rod);

    solver.step_n(steps);

    let points = &solver.rods()[0].points;
    let top_x_before = 24.0;
    (points.x.last().unwrap().x - top_x_before).abs()
}

#[test]
fn root_embedded_in_sand_resists_lateral_push_more_than_bare_root() {
    let bare_displacement = pushed_root(false, 2000);
    let anchored_displacement = pushed_root(true, 2000);

    // Negative control: the bare root must move a clear amount under this
    // push, so the push itself does work.
    assert!(
        bare_displacement > 1.0,
        "bare-root negative control barely moved -- push isn't doing real work: {bare_displacement:.4}"
    );
    // The same push gives measurably less displacement when the root is
    // embedded in sand: anchorage from granular contact.
    assert!(
        anchored_displacement < bare_displacement * 0.5,
        "root embedded in sand should resist the push far more than a bare root: \
         bare={bare_displacement:.4} anchored={anchored_displacement:.4}"
    );
}

//! Real end-to-end check that `rod::plasticity` actually reaches a rod through
//! the FULL `Simulation::step()` pipeline (both call sites in `step.rs`), not
//! just the isolated unit-level `apply_bending_plasticity` calls already
//! covered in `plasticity.rs`'s own tests.

extern crate emerge_engine as emerge;
use emerge::rod::{Rod, RodMaterial, RodPlasticity, build_straight_rod};
use emerge::{SimConfig, Simulation, SpawnRegion};
use glam::Vec2;

fn config(grid_res: usize) -> SimConfig {
    SimConfig {
        adaptive_timestep: true,
        ..SimConfig::earth(grid_res, 0.01, 0.02)
    }
}

/// A cantilever rod, deliberately bent into a real, sharp, LOCALIZED kink at
/// construction (not via a push -- direct control over the initial shape,
/// so the test doesn't depend on getting `push_center`/`push_radius`
/// falloff geometry exactly right). A smooth arc spread across many points
/// was tried first and checked numerically to keep the per-vertex curvature
/// far too small to reach any realistic yield moment even for a
/// large-looking overall tip displacement -- concentrating the whole turn
/// at one vertex is the real, correct way to get a genuinely large local
/// bending moment. `plasticity` is optional so both scenarios below share
/// this exact setup, differing only in that one field.
fn bent_cantilever(dx_meters: f32, plasticity: Option<RodPlasticity>) -> (Rod, Vec2) {
    let start = Vec2::new(24.0, 4.0);
    let height_m = 0.10;
    let mut points = build_straight_rod(
        start,
        Vec2::new(start.x, start.y + height_m / dx_meters),
        12,
        0.01,
        dx_meters,
    );
    points.pinned[0] = 1;
    points.pinned[1] = 1;
    let straight_tip = *points.x.last().unwrap();

    let kink_index = 6;
    let pivot = points.x[kink_index];
    let angle = 50.0_f32.to_radians();
    let (sin_a, cos_a) = angle.sin_cos();
    for i in (kink_index + 1)..points.x.len() {
        let rel = points.x[i] - pivot;
        points.x[i] =
            pivot + Vec2::new(rel.x * cos_a - rel.y * sin_a, rel.x * sin_a + rel.y * cos_a);
    }

    let young_modulus = 1.0e7_f32;
    let width_m = 0.003;
    let thickness_m = 0.001;
    let (axial_damping, bending_damping) = RodMaterial::modal_critical_damping(
        &points,
        young_modulus * width_m * thickness_m,
        young_modulus * width_m.powi(3) * thickness_m / 12.0,
    );
    let material = RodMaterial::from_young_modulus_rectangular(
        young_modulus,
        width_m,
        thickness_m,
        axial_damping,
        bending_damping,
    );

    let mut rod = Rod::new(points, material);
    rod.use_implicit_integration = true;
    rod.implicit_substeps = 8;
    rod.plasticity = plasticity;
    (rod, straight_tip)
}

fn settle(dx_meters: f32, plasticity: Option<RodPlasticity>) -> (f32, Vec<f32>) {
    let cfg = config(48);
    let spawn = SpawnRegion::for_sim(&cfg);
    let mut solver = Simulation::new(cfg, spawn);
    let (rod, straight_tip) = bent_cantilever(dx_meters, plasticity);
    solver.add_rod(rod);

    for _ in 0..600 {
        solver.step();
    }

    let tip_end = *solver.rods()[0].points.x.last().unwrap();
    let offset = (tip_end - straight_tip).length();
    let rest_curvature = solver.rods()[0].points.rest_curvature.clone();
    (offset, rest_curvature)
}

/// Real, physically meaningful claim: a rod bent into a sharp kink, well
/// past its own real yield moment, must settle with a genuine PERMANENT bend
/// substantially larger than the SAME rod/kink with no plasticity at all
/// (which still keeps a small residual offset from real gravity sag alone --
/// comparing against that real baseline, not zero, avoids conflating
/// gravity sag with plasticity's own effect).
#[test]
fn overloaded_rod_stays_bent_substantially_more_than_the_same_rod_without_plasticity() {
    let dx_meters = 0.01;

    // Real, deliberately low yield stress relative to E -- a soft, easily
    // yielded material (well below any real engineering material's own
    // yield-to-modulus ratio), chosen so the real 50-degree kink clearly
    // exceeds it, not tuned to look right.
    let plasticity = RodPlasticity::from_young_modulus_rectangular(1.0e4, 0.003, 0.001);
    let (plastic_offset, plastic_rest_curvature) = settle(dx_meters, Some(plasticity));
    let (elastic_offset, _) = settle(dx_meters, None);

    assert!(
        plastic_offset > 2.0 * elastic_offset,
        "a plastically overloaded rod must settle with a substantially larger permanent \
         offset than the identical rod with no plasticity (plastic={plastic_offset:.3}, \
         elastic={elastic_offset:.3} grid cells) -- otherwise plasticity isn't doing \
         anything real through the full step pipeline"
    );

    // Real, independent confirmation: at least one interior vertex's own
    // rest_curvature must be genuinely nonzero -- the actual mechanism, not
    // just a position coincidence.
    assert!(
        plastic_rest_curvature.iter().any(|&k| k.abs() > 1.0e-4),
        "permanent bend must be backed by a real nonzero rest_curvature, not just a \
         position that happens to look offset: {plastic_rest_curvature:?}"
    );
}

//! `rod::network` variable stiffness: proves a `RodNetwork` chain with
//! alternating HIGH-`ei` "segment" vertices and LOW-`ei` "joint" vertices
//! concentrates real bending curvature at the soft joints while the stiff
//! segments stay nearly straight -- the actual mechanism a rigid-feeling
//! multi-segment leg needs (coxa/femur/tibia stiff, the joints between them
//! soft), using zero new engine code: `NetworkBendingVertex::ei` is already
//! per-vertex (`network.rs`), this only proves what it already does under a
//! real, checkable load.
//!
//! Real physics behind the prediction: standard Euler-Bernoulli beam theory,
//! kappa = M/EI -- along an unbranched chain under a single end load, the
//! bending moment M varies smoothly with arc length, so a sharp EI drop at a
//! joint vertex must produce a correspondingly sharp curvature spike there,
//! not a gradual blend. This is the same formula this engine's own
//! `Rod::buckling_warning`/`RodMaterial::greenhill_critical_height_m` already
//! rely on elsewhere, applied per-vertex instead of to a whole uniform rod.

extern crate emerge_engine as emerge;
use emerge::rod::{
    NetworkBendingVertex, NetworkEdge, RodNetwork, discrete_curvature, network_cfl_dt, step_network,
};
use glam::Vec2;

/// Builds a straight N-point chain (rest-straight, horizontal) with `ei`
/// supplied per bending vertex by the caller -- same point/edge layout
/// `build_straight_rod` uses, generalized to `RodNetwork`'s explicit
/// edge/bending lists (no branching involved, `network.rs` has no plain
/// straight-chain constructor of its own to reuse).
fn build_straight_network(n_points: usize, ea: f32, ei_per_vertex: &[f32]) -> RodNetwork {
    assert_eq!(
        ei_per_vertex.len(),
        n_points - 2,
        "one ei per interior point"
    );
    let seg_len = 1.0_f32;
    let mut x = Vec::with_capacity(n_points);
    let mut v = Vec::with_capacity(n_points);
    let mut mass = Vec::with_capacity(n_points);
    let mut pinned = Vec::with_capacity(n_points);
    for i in 0..n_points {
        x.push(Vec2::new(i as f32 * seg_len, 0.0));
        v.push(Vec2::ZERO);
        mass.push(0.05);
        pinned.push(0);
    }
    let edges = (0..n_points - 1)
        .map(|i| NetworkEdge {
            a: i,
            b: i + 1,
            rest_length_m: seg_len,
            ea,
            axial_damping: 2.0,
        })
        .collect();
    let bending = (0..n_points - 2)
        .map(|i| NetworkBendingVertex {
            p0: i,
            p1: i + 1,
            p2: i + 2,
            rest_curvature: 0.0,
            ei: ei_per_vertex[i],
            bending_damping: 0.5,
            voronoi_length_m: seg_len,
        })
        .collect();
    let n = x.len();
    RodNetwork {
        x,
        v,
        mass,
        pinned,
        position_compensation: vec![Vec2::ZERO; n],
        edges,
        bending,
        axial_damping: 2.0,
        bending_damping: 0.5,
    }
}

#[test]
fn joint_vertices_concentrate_curvature_far_more_than_stiff_segment_vertices() {
    // 13 points -> 11 bending vertices: coxa(3) - joint(1) - femur(3) - joint(1) - tibia(3),
    // indices 0..10 into ei_per_vertex (point index i+1 for bending vertex i).
    let n_points = 13;
    let stiff_ei = 50.0_f32;
    let joint_ei = stiff_ei / 100.0; // real arthropod joints (arthrodial membrane) are
    // dramatically more compliant than the rigid cuticle segments (sclerites) they connect --
    // qualitatively real, this specific ratio chosen for an unambiguous test, not calibrated to
    // a measured species.
    let joint_vertex_indices = [3usize, 7usize]; // bending-vertex-array indices, not point indices
    let mut ei_per_vertex = vec![stiff_ei; n_points - 2];
    for &j in &joint_vertex_indices {
        ei_per_vertex[j] = joint_ei;
    }

    let ea = 5.0e3_f32;
    let mut net = build_straight_network(n_points, ea, &ei_per_vertex);
    net.pinned[0] = 1;
    net.pinned[1] = 1;

    let gravity = Vec2::new(0.0, -0.5);
    let dx_meters = 1.0;
    let mut elapsed = 0.0_f32;
    while elapsed < 3.0 {
        let dt = network_cfl_dt(&net, 0.4).min(0.002);
        step_network(&mut net, gravity, dx_meters, dt);
        elapsed += dt;
    }

    for p in net.x.iter() {
        assert!(p.is_finite(), "chain must stay numerically finite");
    }

    let curvature_at = |bv_idx: usize| -> f32 {
        let bv = &net.bending[bv_idx];
        discrete_curvature(net.x[bv.p0], net.x[bv.p1], net.x[bv.p2]).abs()
    };

    let joint_curvatures: Vec<f32> = joint_vertex_indices
        .iter()
        .map(|&j| curvature_at(j))
        .collect();
    let segment_indices: Vec<usize> = (0..n_points - 2)
        .filter(|i| !joint_vertex_indices.contains(i))
        .collect();
    let segment_curvatures: Vec<f32> = segment_indices.iter().map(|&i| curvature_at(i)).collect();

    let mean_joint = joint_curvatures.iter().sum::<f32>() / joint_curvatures.len() as f32;
    let mean_segment = segment_curvatures.iter().sum::<f32>() / segment_curvatures.len() as f32;

    assert!(
        mean_joint > mean_segment * 5.0,
        "soft joint vertices should concentrate dramatically more curvature than stiff segment \
         vertices under the same end-to-end bending load (kappa = M/EI): mean_joint={mean_joint:.6} \
         mean_segment={mean_segment:.6} joint_curvatures={joint_curvatures:?} \
         segment_curvatures={segment_curvatures:?}"
    );
}

/// Appends one coxa-joint-femur-joint-tibia leg (same 3-1-3-1-3 stiffness layout as
/// `joint_vertices_concentrate_curvature_far_more_than_stiff_segment_vertices`) to an
/// existing network, starting FROM `hub_idx` (an already-existing point, typically the
/// shared body/thorax attachment point -- not a new point of its own, the same
/// "junction is a real shared index" pattern `build_y_branch` already established for
/// 2 children, generalized here to as many legs as the caller appends). Returns
/// (joint_bending_vertex_indices, segment_bending_vertex_indices) as GLOBAL indices
/// into `net.bending`, for the test to check independently per leg.
fn append_leg(
    net: &mut RodNetwork,
    hub_idx: usize,
    direction: Vec2,
    ea: f32,
    stiff_ei: f32,
    joint_ei: f32,
) -> (Vec<usize>, Vec<usize>) {
    let seg_len = 1.0_f32;
    let n_new_points = 12; // matches the single-leg test's 13-point (incl. hub) chain
    let dir = direction.normalize();
    let hub_pos = net.x[hub_idx];

    let mut chain = vec![hub_idx];
    for i in 1..=n_new_points {
        net.x.push(hub_pos + dir * (i as f32 * seg_len));
        net.v.push(Vec2::ZERO);
        net.mass.push(0.05);
        net.pinned.push(0);
        net.position_compensation.push(Vec2::ZERO);
        chain.push(net.x.len() - 1);
    }

    for w in chain.windows(2) {
        net.edges.push(NetworkEdge {
            a: w[0],
            b: w[1],
            rest_length_m: seg_len,
            ea,
            axial_damping: net.axial_damping,
        });
    }

    let local_ei = |local_bv: usize| -> f32 {
        // local_bv indexes chain[1..], same 3-1-3-1-3 layout as the single-leg test:
        // seg(0,1,2) - joint(3) - seg(4,5,6) - joint(7) - seg(8,9,10).
        if local_bv == 3 || local_bv == 7 {
            joint_ei
        } else {
            stiff_ei
        }
    };
    let mut joint_bvs = Vec::new();
    let mut segment_bvs = Vec::new();
    for w in chain.windows(3).enumerate() {
        let (local_bv, pts) = w;
        let global_idx = net.bending.len();
        net.bending.push(NetworkBendingVertex {
            p0: pts[0],
            p1: pts[1],
            p2: pts[2],
            rest_curvature: 0.0,
            ei: local_ei(local_bv),
            bending_damping: net.bending_damping,
            voronoi_length_m: seg_len,
        });
        if local_bv == 3 || local_bv == 7 {
            joint_bvs.push(global_idx);
        } else {
            segment_bvs.push(global_idx);
        }
    }
    (joint_bvs, segment_bvs)
}

#[test]
fn three_legs_share_one_hub_without_fighting_each_other() {
    // The real "body + legs assembly" question: does ONE shared body point support
    // several independent legs at once -- each still showing the same real
    // joint-concentrates-curvature behavior -- without the legs corrupting each
    // other's state or the hub itself drifting off its pinned position. Zero grid
    // coupling, zero rendering: purely whether the RodNetwork data model holds
    // together under a real multi-leg topology, the same "prove it before building
    // the demo" discipline this file's first test already used for one leg.
    let hub = Vec2::new(0.0, 20.0);
    let mut net = RodNetwork {
        x: vec![hub],
        v: vec![Vec2::ZERO],
        mass: vec![0.2], // heavier hub -- a real thorax outweighs one leg
        pinned: vec![1],
        position_compensation: vec![Vec2::ZERO],
        edges: Vec::new(),
        bending: Vec::new(),
        axial_damping: 2.0,
        bending_damping: 0.5,
    };

    let ea = 5.0e3_f32;
    let stiff_ei = 50.0_f32;
    let joint_ei = stiff_ei / 100.0;
    // None of these may point exactly straight down (0,-1): a chain hanging
    // perfectly antiparallel to gravity carries pure axial tension and zero
    // bending moment by symmetry, same real caveat `mod.rs`'s own
    // `sustained_bending_stress_raises_greenhill_height` test already documents --
    // caught live by this test's first run (leg 1 measured exactly zero curvature).
    let leg_directions = [
        Vec2::new(-0.7, -0.7),
        Vec2::new(0.15, -0.99),
        Vec2::new(0.7, -0.7),
    ];
    let legs: Vec<(Vec<usize>, Vec<usize>)> = leg_directions
        .iter()
        .map(|&dir| append_leg(&mut net, 0, dir, ea, stiff_ei, joint_ei))
        .collect();

    let gravity = Vec2::new(0.0, -0.5);
    let dx_meters = 1.0;
    let mut elapsed = 0.0_f32;
    while elapsed < 3.0 {
        let dt = network_cfl_dt(&net, 0.4).min(0.002);
        step_network(&mut net, gravity, dx_meters, dt);
        elapsed += dt;
    }

    for p in net.x.iter() {
        assert!(p.is_finite(), "assembly must stay numerically finite");
    }
    assert_eq!(
        net.x[0], hub,
        "the pinned hub must not drift even with 3 legs pulling on it"
    );

    let curvature_at = |bv_idx: usize| -> f32 {
        let bv = &net.bending[bv_idx];
        discrete_curvature(net.x[bv.p0], net.x[bv.p1], net.x[bv.p2]).abs()
    };

    for (leg_i, (joint_bvs, segment_bvs)) in legs.iter().enumerate() {
        let mean_joint: f32 =
            joint_bvs.iter().map(|&i| curvature_at(i)).sum::<f32>() / joint_bvs.len() as f32;
        let mean_segment: f32 =
            segment_bvs.iter().map(|&i| curvature_at(i)).sum::<f32>() / segment_bvs.len() as f32;
        assert!(
            mean_joint > mean_segment * 5.0,
            "leg {leg_i}: soft joints should concentrate far more curvature than stiff \
             segments even sharing a hub with 2 other legs: mean_joint={mean_joint:.6} \
             mean_segment={mean_segment:.6}"
        );
    }
}

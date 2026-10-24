//! `rod::network` branching: the junction point is a shared array index (both
//! the trunk's last edge and the branch's first edge reference it), not two synchronized
//! `Rod`s. Checks internal force consistency at a branch vertex and that a branch
//! measurably loads the trunk through that shared point.

extern crate emerge_engine as emerge;
use emerge::rod::{YBranchSpec, build_y_branch, network_cfl_dt, step_network};
use glam::Vec2;

#[test]
fn straight_y_branch_at_rest_has_zero_force() {
    let net = build_y_branch(YBranchSpec {
        trunk_start: Vec2::new(0.0, 0.0),
        junction: Vec2::new(0.0, 5.0),
        branch_end: Vec2::new(3.0, 8.0),
        n_trunk_points: 6,
        n_branch_points: 5,
        linear_density_kg_per_m: 0.3,
        dx_meters: 1.0,
        ea: 1000.0,
        ei: 10.0,
        axial_damping: 0.0,
        bending_damping: 0.0,
    });
    let force = emerge::rod::compute_network_internal_forces(&net, 1.0);
    for (i, f) in force.iter().enumerate() {
        assert!(
            f.length() < 1.0e-3,
            "straight Y-branch at rest length/curvature should have zero internal force, point {i}: {f:?}"
        );
    }
}

#[test]
fn branch_measurably_loads_the_trunk_through_the_shared_junction() {
    let n_trunk_points = 8;
    let junction_idx = n_trunk_points - 1; // real shared point -- part of the trunk too
    let build = |with_branch: bool| {
        let n_branch = if with_branch { 6 } else { 2 };
        let mut net = build_y_branch(YBranchSpec {
            trunk_start: Vec2::new(0.0, 20.0),
            junction: Vec2::new(0.0, 15.0),
            branch_end: Vec2::new(4.0, 12.0),
            n_trunk_points,
            n_branch_points: n_branch,
            linear_density_kg_per_m: 0.3,
            dx_meters: 1.0,
            ea: 3.0e4 * 0.02 * 0.01,
            ei: 3.0e4 * 0.02_f32.powi(3) * 0.01 / 12.0,
            axial_damping: 5.0,
            bending_damping: 0.5,
        });
        net.pinned[0] = 1;
        if !with_branch {
            // "No branch" baseline: zero stiffness strictly beyond the junction (branch-only
            // points/edges) so it neither stretch- nor bend-resists the trunk. Must be `>
            // junction_idx`, not `>=` -- junction_idx is shared with the trunk, not branch-only.
            for edge in net.edges.iter_mut() {
                if edge.a > junction_idx || edge.b > junction_idx {
                    edge.ea = 0.0;
                }
            }
            for bv in net.bending.iter_mut() {
                if bv.p0 > junction_idx || bv.p1 > junction_idx || bv.p2 > junction_idx {
                    bv.ei = 0.0;
                }
            }
        }
        net
    };

    let mut with_branch = build(true);
    let mut without_branch = build(false);

    let gravity = Vec2::new(0.0, -0.3);
    let dx_meters = 1.0;
    let mut elapsed = 0.0_f32;
    while elapsed < 2.0 {
        let dt = network_cfl_dt(&with_branch, 0.4)
            .min(network_cfl_dt(&without_branch, 0.4))
            .min(0.002);
        step_network(&mut with_branch, gravity, dx_meters, dt);
        step_network(&mut without_branch, gravity, dx_meters, dt);
        elapsed += dt;
    }

    let sag_with_branch = 15.0 - with_branch.x[junction_idx].y;
    let sag_without_branch = 15.0 - without_branch.x[junction_idx].y;

    assert!(
        sag_with_branch > sag_without_branch + 0.02,
        "trunk should sag measurably more with a real branch loading it through the shared \
         junction: with_branch={sag_with_branch:.4} without_branch={sag_without_branch:.4}"
    );
}

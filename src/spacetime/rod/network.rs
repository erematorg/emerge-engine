//! Real branching rod topology -- a genuine graph, not two `Rod`s synchronized
//! after the fact.
//! A branch's root point literally IS a point in the SAME array as its
//! parent -- one array, one source of truth, no post-hoc synchronization.
//!
//! Reuses `forces::discrete_curvature`/`discrete_curvature_gradient`
//! completely unchanged -- those functions are already point-based (three
//! raw `Vec2`s), not index-based, so the real curvature math from Bergou et
//! al. 2008 needs no modification at all to work at a branch vertex; only
//! the TOPOLOGY (which points form which edges/bending triples) needed to
//! generalize from "always i-1,i,i+1" (implicit, linear-chain `RodPoints`)
//! to an explicit list.
//!
//! Scope, disclosed: binary branching only -- a point may have at most 3
//! incident edges (one "parent" side, two "child" sides), giving exactly 2
//! meaningful bending vertices at a branch point. Real botanical branching
//! is overwhelmingly bifurcation; true simultaneous trifurcation would need
//! a further real extension, not attempted here.
//!
//! Per-edge/per-bending-vertex stiffness and damping (`ea`/`ei`,
//! `axial_damping`/`bending_damping`) are stored explicitly, not one shared
//! `RodMaterial` -- a real trunk and its fine branches genuinely differ in
//! their mechanical response, unlike a single unbranched rod where one
//! material was always a reasonable assumption.

use glam::Vec2;

use super::forces::{discrete_curvature, discrete_curvature_gradient};

#[derive(Debug, Clone, Copy)]
pub struct NetworkEdge {
    pub a: usize,
    pub b: usize,
    pub rest_length_m: f32,
    pub ea: f32,
    pub axial_damping: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct NetworkBendingVertex {
    pub p0: usize,
    pub p1: usize,
    pub p2: usize,
    pub rest_curvature: f32,
    pub ei: f32,
    pub bending_damping: f32,
    pub voronoi_length_m: f32,
}

#[derive(Debug, Clone)]
pub struct RodNetwork {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    pub mass: Vec<f32>,
    pub pinned: Vec<u32>,
    pub position_compensation: Vec<Vec2>,
    pub edges: Vec<NetworkEdge>,
    pub bending: Vec<NetworkBendingVertex>,
    /// Default copied into new edges by network constructors.
    pub axial_damping: f32,
    /// Default copied into new bending vertices by network constructors.
    pub bending_damping: f32,
}

impl RodNetwork {
    pub const fn len(&self) -> usize {
        self.x.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Grouped parameters for `build_y_branch` -- one struct instead of an
/// 11-argument function signature.
#[derive(Debug, Clone, Copy)]
pub struct YBranchSpec {
    pub trunk_start: Vec2,
    pub junction: Vec2,
    pub branch_end: Vec2,
    pub n_trunk_points: usize,
    pub n_branch_points: usize,
    pub linear_density_kg_per_m: f32,
    pub dx_meters: f32,
    pub ea: f32,
    pub ei: f32,
    pub axial_damping: f32,
    pub bending_damping: f32,
}

/// Builds a simple Y-branch: a straight trunk from `trunk_start` to
/// `junction`, with a straight branch continuing from `junction` to
/// `branch_end` -- the smallest real, testable branching topology. The
/// junction point is shared: it is NOT duplicated, it is literally point
/// index `n_trunk - 1`, referenced by both the trunk's last edge and the
/// branch's first edge.
pub fn build_y_branch(spec: YBranchSpec) -> RodNetwork {
    let YBranchSpec {
        trunk_start,
        junction,
        branch_end,
        n_trunk_points,
        n_branch_points,
        linear_density_kg_per_m,
        dx_meters,
        ea,
        ei,
        axial_damping,
        bending_damping,
    } = spec;
    assert!(n_trunk_points >= 2, "trunk needs at least 2 points");
    assert!(n_branch_points >= 2, "branch needs at least 2 points");

    let mut x = Vec::new();
    let mut v = Vec::new();
    let mut mass = Vec::new();
    let mut pinned = Vec::new();
    let mut edges = Vec::new();
    let mut bending = Vec::new();

    // Trunk: points 0..n_trunk_points-1, last one is the shared junction.
    let trunk_len_m = (junction - trunk_start).length() * dx_meters;
    let trunk_seg_m = trunk_len_m / (n_trunk_points as f32 - 1.0);
    let trunk_point_mass = linear_density_kg_per_m * trunk_seg_m;
    for i in 0..n_trunk_points {
        let t = i as f32 / (n_trunk_points as f32 - 1.0);
        x.push(trunk_start.lerp(junction, t));
        v.push(Vec2::ZERO);
        let m = if i == 0 {
            trunk_point_mass * 0.5
        } else {
            trunk_point_mass
        };
        mass.push(m);
        pinned.push(0);
    }
    for i in 0..n_trunk_points - 1 {
        edges.push(NetworkEdge {
            a: i,
            b: i + 1,
            rest_length_m: trunk_seg_m,
            ea,
            axial_damping,
        });
    }
    for i in 0..n_trunk_points.saturating_sub(2) {
        let voronoi = trunk_seg_m; // uniform spacing: both adjoining edges equal
        bending.push(NetworkBendingVertex {
            p0: i,
            p1: i + 1,
            p2: i + 2,
            rest_curvature: 0.0,
            ei,
            bending_damping,
            voronoi_length_m: voronoi,
        });
    }

    // Branch: points n_trunk_points..(n_trunk_points+n_branch_points-2),
    // its own first edge starts AT the junction (index n_trunk_points-1) --
    // no new point created for it, the junction is reused directly.
    let junction_idx = n_trunk_points - 1;
    let branch_len_m = (branch_end - junction).length() * dx_meters;
    let branch_seg_m = branch_len_m / (n_branch_points as f32 - 1.0);
    let branch_point_mass = linear_density_kg_per_m * branch_seg_m;
    let mut branch_indices = vec![junction_idx];
    for i in 1..n_branch_points {
        let t = i as f32 / (n_branch_points as f32 - 1.0);
        x.push(junction.lerp(branch_end, t));
        v.push(Vec2::ZERO);
        let m = if i == n_branch_points - 1 {
            branch_point_mass * 0.5
        } else {
            branch_point_mass
        };
        mass.push(m);
        pinned.push(0);
        branch_indices.push(x.len() - 1);
    }
    for w in branch_indices.windows(2) {
        edges.push(NetworkEdge {
            a: w[0],
            b: w[1],
            rest_length_m: branch_seg_m,
            ea,
            axial_damping,
        });
    }
    for w in branch_indices.windows(3) {
        bending.push(NetworkBendingVertex {
            p0: w[0],
            p1: w[1],
            p2: w[2],
            rest_curvature: 0.0,
            ei,
            bending_damping,
            voronoi_length_m: branch_seg_m,
        });
    }
    // The junction itself also bends against the trunk's own last segment
    // (p0=trunk's second-to-last point, p1=junction, p2=branch's first real
    // point) -- this is the real, physically meaningful bending vertex that
    // makes the branch mechanically coupled to the trunk's own orientation,
    // not just sharing a point in name only.
    //
    // rest_curvature must be the ACTUAL discrete curvature of the constructed
    // geometry, not 0.0 -- a Y-branch genuinely bends here by construction (the
    // branch angles away from the trunk's own direction), so claiming
    // "straight/rest" here is false and injects a large spurious bending force
    // from the very first substep. `build_straight_rod`'s 0.0 is correct there
    // because that rod really is straight everywhere; this junction is not.
    if n_trunk_points >= 2 && n_branch_points >= 2 {
        let (p0, p1, p2) = (
            x[junction_idx - 1] * dx_meters,
            x[junction_idx] * dx_meters,
            x[branch_indices[1]] * dx_meters,
        );
        bending.push(NetworkBendingVertex {
            p0: junction_idx - 1,
            p1: junction_idx,
            p2: branch_indices[1],
            rest_curvature: discrete_curvature(p0, p1, p2),
            ei,
            bending_damping,
            voronoi_length_m: 0.5 * (trunk_seg_m + branch_seg_m),
        });
    }

    let n_points = x.len();
    RodNetwork {
        x,
        v,
        mass,
        pinned,
        position_compensation: vec![Vec2::ZERO; n_points],
        edges,
        bending,
        axial_damping,
        bending_damping,
    }
}

/// Real per-point internal force (Newtons), generalizing
/// `forces::compute_internal_forces` from implicit linear-chain adjacency
/// to an explicit edge/bending topology. Same stretch + bending + damping
/// physics, same underlying curvature math -- only the iteration is
/// different (over explicit lists instead of an index range).
pub fn compute_network_internal_forces(net: &RodNetwork, dx_meters: f32) -> Vec<Vec2> {
    let n = net.len();
    let mut force = vec![Vec2::ZERO; n];
    if n < 2 {
        return force;
    }

    for edge in &net.edges {
        let e = (net.x[edge.b] - net.x[edge.a]) * dx_meters;
        let l = e.length().max(1.0e-9);
        let l0 = edge.rest_length_m.max(1.0e-9);
        let dir = e / l;

        let f_stretch = edge.ea * (l - l0) / l0;
        let rel_v = (net.v[edge.b] - net.v[edge.a]) * dx_meters;
        let strain_rate = rel_v.dot(dir);
        let f_damp = edge.axial_damping * strain_rate;

        let f = (f_stretch + f_damp) * dir;
        force[edge.a] += f;
        force[edge.b] -= f;
    }

    for bv in &net.bending {
        let (p0, p1, p2) = (
            net.x[bv.p0] * dx_meters,
            net.x[bv.p1] * dx_meters,
            net.x[bv.p2] * dx_meters,
        );
        let kappa = discrete_curvature(p0, p1, p2);
        let grad = discrete_curvature_gradient(p0, p1, p2);

        let voronoi_length = bv.voronoi_length_m.max(1.0e-9);
        let coeff = (bv.ei / voronoi_length) * (kappa - bv.rest_curvature);

        let kappa_dot = grad[0].dot(net.v[bv.p0] * dx_meters)
            + grad[1].dot(net.v[bv.p1] * dx_meters)
            + grad[2].dot(net.v[bv.p2] * dx_meters);
        let damp_coeff = bv.bending_damping * kappa_dot;

        let total_coeff = coeff + damp_coeff;
        force[bv.p0] -= total_coeff * grad[0];
        force[bv.p1] -= total_coeff * grad[1];
        force[bv.p2] -= total_coeff * grad[2];
    }

    force
}

/// Real Gershgorin CFL bound, generalizing `integrator::rod_cfl_dt` from
/// implicit `i-1/i/i+1` neighbor lookups to an explicit edges/bending pass
/// -- same principle (sum every stiffness/damping term touching each point),
/// single accumulation pass instead of a per-point neighbor scan since the
/// topology is no longer a fixed linear pattern.
pub fn network_cfl_dt(net: &RodNetwork, safety: f32) -> f32 {
    let n = net.len();
    let mut omega_sq = vec![0.0f32; n];
    let mut damping_rate = vec![0.0f32; n];

    for edge in &net.edges {
        let m_a = net.mass[edge.a].max(1.0e-9);
        let m_b = net.mass[edge.b].max(1.0e-9);
        let l0 = edge.rest_length_m.max(1.0e-9);
        omega_sq[edge.a] += edge.ea / (m_a * l0);
        omega_sq[edge.b] += edge.ea / (m_b * l0);
        damping_rate[edge.a] += edge.axial_damping;
        damping_rate[edge.b] += edge.axial_damping;
    }
    for bv in &net.bending {
        let voronoi_length = bv.voronoi_length_m.max(1.0e-9);
        for &p in &[bv.p0, bv.p1, bv.p2] {
            let m = net.mass[p].max(1.0e-9);
            if bv.ei > 0.0 {
                omega_sq[p] += bv.ei / (m * voronoi_length.powi(3));
            }
            if bv.bending_damping > 0.0 {
                damping_rate[p] += bv.bending_damping / voronoi_length.powi(2);
            }
        }
    }

    let mut min_dt = f32::INFINITY;
    for i in 0..n {
        let m = net.mass[i].max(1.0e-9);
        if omega_sq[i] > f32::EPSILON {
            min_dt = min_dt.min(safety * 2.0 / omega_sq[i].sqrt());
        }
        if damping_rate[i] > f32::EPSILON {
            min_dt = min_dt.min(safety * 2.0 * m / damping_rate[i]);
        }
    }
    min_dt
}

/// Standalone explicit (symplectic Euler) network step -- no grid, mirrors
/// `integrator::step_rod`'s own Phase 0/1 role for the linear rod. No
/// built-in CFL enforcement, matching `step_rod`'s own contract: the caller
/// must pick a stable `dt`, typically via `network_cfl_dt`.
pub fn step_network(net: &mut RodNetwork, gravity: Vec2, dx_meters: f32, dt: f32) {
    let n = net.len();
    if n == 0 {
        return;
    }
    let internal = compute_network_internal_forces(net, dx_meters);
    for (i, internal_force) in internal.iter().enumerate() {
        if net.pinned[i] != 0 {
            net.v[i] = Vec2::ZERO;
            continue;
        }
        let a_internal = *internal_force / (net.mass[i].max(1.0e-9) * dx_meters);
        net.v[i] += (gravity + a_internal) * dt;
    }
    for i in 0..n {
        if net.pinned[i] != 0 {
            continue;
        }
        net.x[i] += net.v[i] * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_network() -> RodNetwork {
        build_y_branch(YBranchSpec {
            trunk_start: Vec2::new(0.0, 0.0),
            junction: Vec2::new(0.0, 2.0),
            branch_end: Vec2::new(2.0, 3.0),
            n_trunk_points: 3,
            n_branch_points: 3,
            linear_density_kg_per_m: 1.0,
            dx_meters: 1.0,
            ea: 0.0,
            ei: 0.0,
            axial_damping: 2.0,
            bending_damping: 3.0,
        })
    }

    #[test]
    fn constructor_copies_network_damping_defaults_to_each_branch_element() {
        let net = test_network();
        assert!(net.edges.iter().all(|edge| edge.axial_damping == 2.0));
        assert!(
            net.bending
                .iter()
                .all(|vertex| vertex.bending_damping == 3.0)
        );
    }

    #[test]
    fn internal_forces_use_per_element_damping_not_network_defaults() {
        let mut net = test_network();
        net.axial_damping = 1000.0;
        net.bending_damping = 1000.0;
        for edge in &mut net.edges {
            edge.axial_damping = 0.0;
        }
        for vertex in &mut net.bending {
            vertex.bending_damping = 0.0;
        }
        assert!(
            network_cfl_dt(&net, 0.5).is_infinite(),
            "CFL damping bound must also ignore network-wide construction defaults"
        );

        let edge = net.edges[0];
        let edge_dir = (net.x[edge.b] - net.x[edge.a]).normalize();
        net.v[edge.b] = edge_dir;
        assert!(
            compute_network_internal_forces(&net, 1.0)
                .iter()
                .all(|force| force.length() < 1.0e-6),
            "network-wide defaults must not leak into existing edges or vertices"
        );

        net.edges[0].axial_damping = 4.0;
        assert!(network_cfl_dt(&net, 0.5).is_finite());
        let axial_forces = compute_network_internal_forces(&net, 1.0);
        assert!(
            axial_forces[edge.a].length() > 1.0,
            "the edited edge's own axial damping must contribute force"
        );

        net.v.fill(Vec2::ZERO);
        net.edges[0].axial_damping = 0.0;
        let vertex = net.bending[0];
        let grad =
            discrete_curvature_gradient(net.x[vertex.p0], net.x[vertex.p1], net.x[vertex.p2]);
        net.v[vertex.p0] = grad[0];
        net.bending[0].bending_damping = 5.0;
        let bending_forces = compute_network_internal_forces(&net, 1.0);
        assert!(
            bending_forces.iter().any(|force| force.length() > 1.0e-3),
            "the edited bending vertex's own damping must contribute force"
        );
    }
}

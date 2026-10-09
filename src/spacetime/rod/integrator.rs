//! Standalone rod time integration -- no grid, no `Simulation` (Phase 0/1).
//! Phase 2's grid-coupled path reuses `forces::compute_internal_forces`
//! directly (see `coupling.rs`) rather than this integrator.

use glam::Vec2;

use super::{RodMaterial, RodPoints, RodRestState, compute_internal_forces};
use crate::spacetime::integration::advance_position;

/// One explicit (symplectic Euler) rod substep. Pinned points held at
/// `v=0`/position fixed -- identical semantics to `Particle::pinned`'s own
/// G2P handling (forces v=0 instead of gathering, position left completely
/// untouched, mass/forces still computed normally so the anchor is real).
pub fn step_rod(
    rod: &mut RodPoints,
    material: &RodMaterial,
    gravity: Vec2,
    wind_velocity: Vec2,
    wind_drag_coeff: f32,
    dx_meters: f32,
    dt: f32,
) {
    let n = rod.len();
    if n == 0 {
        return;
    }

    let internal = compute_internal_forces(
        &rod.x,
        &rod.v,
        RodRestState {
            rest_edge_length: &rod.rest_edge_length,
            rest_curvature: &rod.rest_curvature,
            ea: &rod.ea,
            ei: &rod.ei,
        },
        material,
        dx_meters,
    );

    for (i, internal_force) in internal.iter().enumerate() {
        if rod.pinned[i] != 0 {
            rod.v[i] = Vec2::ZERO;
            continue;
        }
        // Internal force (Newtons) -> grid acceleration: a = F/(m*dx_meters),
        // mirrors gravity_to_grid's own g_grid = g_SI/dx_meters (mass handled
        // explicitly here since force, unlike gravity, isn't already
        // per-unit-mass).
        let a_internal = *internal_force / (rod.mass[i] * dx_meters);
        // Wind drag: a = k*(target - v), same law LinearDragField implements.
        let a_wind = wind_drag_coeff * (wind_velocity - rod.v[i]);
        let a = gravity + a_internal + a_wind;
        rod.v[i] += a * dt;
    }
    // Symplectic Euler: advance position with the JUST-updated velocity.
    for i in 0..n {
        if rod.pinned[i] != 0 {
            continue;
        }
        advance_position(
            &mut rod.x[i],
            &mut rod.position_compensation[i],
            rod.v[i] * dt,
        );
    }
}

/// Longest step the rod's own explicit integrator (`step_rod`: symplectic
/// Euler, damping from the step's starting velocity) stays stable at, per
/// point, then the smallest; `fraction` of it is returned. The fraction is
/// `SimConfig::material_cfl_coefficient`'s definition.
///
/// Derived from the scheme. A mode `x'' = -omega^2 x - b x'` stepped this
/// way is stable exactly while `dt <= 4 / (b + sqrt(b^2 + 4 omega^2))`
/// (the Jury conditions on its 2x2 update; `2 / omega` without damping).
/// `omega^2` and `b` are bounded by Gershgorin row sums of the linearised
/// stiffness and damping over the point's mass (`point_stability_sums`).
///
/// It replaces a per-point sum that counted each edge's axial stiffness once
/// and each bending vertex once with weight one, 2 and 16/3 times too little,
/// with an empirical 0.4 in front: a cantilever at 1 cm cells under real
/// gravity blew up at step 32 at that step (`tests/subsystem_time_steps.rs`,
/// `probe_cantilever_coupling_cut`).
pub fn rod_cfl_dt(rod: &RodPoints, material: &RodMaterial, fraction: f32) -> f32 {
    let mut min_dt = f32::INFINITY;
    for i in 0..rod.len() {
        if rod.pinned[i] != 0 {
            continue;
        }
        let m = rod.mass[i].max(1.0e-9);
        let (k, c) = point_stability_sums(rod, material, i);
        let (omega_sq, b) = (k / m, c / m);
        if omega_sq > 0.0 || b > 0.0 {
            min_dt = min_dt.min(4.0 / (b + (b * b + 4.0 * omega_sq).sqrt()));
        }
    }
    fraction * min_dt
}

/// Point `i`'s Gershgorin row sums of the linearised stiffness `K` (N/m) and
/// damping `C` (N s/m), about the straight rest state, shared by
/// `rod_cfl_dt` and `apply_mass_scaling_for_target_dt`.
///
/// Axial: each edge is a spring `EA / l0` and a dashpot `c_a` along it, a
/// 2x2 block `[[1, -1], [-1, 1]]`, so each adjacent edge adds twice its
/// value to the row. Bending (`forces::compute_internal_forces`): vertex
/// `k` stores `EI / (2 L_v) kappa^2`, and for a straight rod
/// `kappa = (w2 - w1) / l_n - (w1 - w0) / l_p` in the lateral displacements,
/// so its gradient is `g = (1/l_p, -(1/l_p + 1/l_n), 1/l_n)`, its stiffness
/// `(EI / L_v) g g^T` and its damping `c_b g g^T`; point `i` at position `j`
/// in the vertex adds `|g_j| * sum|g|` times each. An interior point on
/// equal edges gets `4 EA / l0` and `16 EI / l0^3`.
fn point_stability_sums(rod: &RodPoints, material: &RodMaterial, i: usize) -> (f32, f32) {
    let n = rod.len();
    let ea_at = |k: usize| {
        if rod.ea.is_empty() {
            material.ea
        } else {
            rod.ea[k]
        }
    };
    let ei_at = |k: usize| {
        if rod.ei.is_empty() {
            material.ei
        } else {
            rod.ei[k]
        }
    };
    let (mut k_sum, mut c_sum) = (0.0f32, 0.0f32);
    for edge in [i.checked_sub(1), (i + 1 < n).then_some(i)]
        .into_iter()
        .flatten()
    {
        let l0 = rod.rest_edge_length[edge].max(1.0e-9);
        k_sum += 2.0 * ea_at(edge) / l0;
        c_sum += 2.0 * material.axial_damping;
    }
    for vertex in [i.checked_sub(2), i.checked_sub(1), Some(i)]
        .into_iter()
        .flatten()
    {
        if vertex + 2 >= n {
            continue;
        }
        let l_p = rod.rest_edge_length[vertex].max(1.0e-9);
        let l_n = rod.rest_edge_length[vertex + 1].max(1.0e-9);
        let voronoi_length = 0.5 * (l_p + l_n);
        let g = [1.0 / l_p, 1.0 / l_p + 1.0 / l_n, 1.0 / l_n];
        let row = g[i - vertex] * (g[0] + g[1] + g[2]);
        k_sum += ei_at(vertex) / voronoi_length * row;
        c_sum += material.bending_damping * row;
    }
    (k_sum, c_sum)
}

/// Mass scaling (a standard explicit-dynamics technique, e.g. LS-DYNA's own
/// `*CONTROL_TIMESTEP` option): raise a point's own inertia just enough
/// that `rod_cfl_dt` reaches `target_dt` at `fraction`. Only ever raises
/// mass, and changes real dynamics (gravity, wind, push response), so it is
/// an opt-in call, never applied silently.
///
/// From `rod_cfl_dt`: with `tau = target_dt / fraction`, `omega^2 = K / m`
/// and `b = C / m`, `4 / (b + sqrt(b^2 + 4 omega^2)) >= tau` exactly when
/// `m >= (K tau^2 + 2 C tau) / 4`.
pub fn apply_mass_scaling_for_target_dt(
    rod: &mut RodPoints,
    material: &RodMaterial,
    fraction: f32,
    target_dt: f32,
) {
    let tau = target_dt / fraction;
    for i in 0..rod.len() {
        if rod.pinned[i] != 0 {
            continue;
        }
        let (k, c) = point_stability_sums(rod, material, i);
        let m_required = (k * tau * tau + 2.0 * c * tau) / 4.0;
        rod.mass[i] = rod.mass[i].max(m_required);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::{rod_cfl_dt, step_rod};
    use crate::rod::{
        RodForceParams, RodImplicitStepParams, RodMaterial, RodPoints, YBranchSpec, advance_rod,
        build_straight_rod, build_y_branch, step_network, step_rod_implicit,
    };

    /// Half the spacing of f32 values around 40, where these tests sit.
    fn half_ulp_at_40() -> f32 {
        0.5 * (40.0f32.next_up() - 40.0)
    }

    /// A straight three-point rod near x = y = 40 cells at rest length,
    /// translating as a whole at `speed` cells/s: no internal force acts,
    /// so every stepper must carry it `speed * seconds`.
    fn translating_rod(speed: f32) -> (RodPoints, RodMaterial) {
        let dx = 0.01;
        let mut rod = build_straight_rod(Vec2::new(40.0, 40.0), Vec2::new(42.0, 40.0), 3, 0.1, dx);
        rod.v.iter_mut().for_each(|v| *v = Vec2::new(0.0, speed));
        let (axial, bending) = RodMaterial::critical_damping(dx, 0.1 * dx, 1000.0, 0.5);
        (rod, RodMaterial::new(1000.0, 0.5, axial, bending))
    }

    /// Each point's increment is `speed * h`; the tests pick `speed` so that
    /// it is well under half an ulp, and check the rod still covers the
    /// distance.
    fn assert_covered(rod: &RodPoints, start_y: f32, distance: f32, stepper: &str) {
        for (i, x) in rod.x.iter().enumerate() {
            let moved = x.y - start_y;
            assert!(
                (moved - distance).abs() < 0.01 * distance,
                "{stepper}: point {i} moved {moved:e}, expected {distance:e}"
            );
        }
    }

    #[test]
    fn step_rod_keeps_motion_below_half_an_ulp() {
        let speed = 1.0e-3;
        let (mut rod, material) = translating_rod(speed);
        let h = rod_cfl_dt(&rod, &material, 0.5);
        assert!(speed * h < 0.1 * half_ulp_at_40(), "premise: h {h:e}");
        let steps = (1.0 / h).ceil() as usize;
        for _ in 0..steps {
            step_rod(&mut rod, &material, Vec2::ZERO, Vec2::ZERO, 0.0, 0.01, h);
        }
        assert_covered(&rod, 40.0, speed * h * steps as f32, "step_rod");
    }

    #[test]
    fn advance_rod_keeps_motion_below_half_an_ulp() {
        let speed = 1.0e-3;
        let (mut rod, material) = translating_rod(speed);
        let h = rod_cfl_dt(&rod, &material, 0.5);
        assert!(speed * h < 0.1 * half_ulp_at_40(), "premise: h {h:e}");
        let frame = 0.01;
        let external = vec![Vec2::ZERO; rod.len()];
        for _ in 0..100 {
            let params = RodForceParams {
                wind_velocity: Vec2::ZERO,
                wind_drag_coeff: 0.0,
                push_center: None,
                push_strength: 0.0,
                push_radius: 0.0,
                dx_meters: 0.01,
                dt: frame,
                stability_fraction: 0.5,
            };
            advance_rod(&mut rod, &material, params, &external);
        }
        assert_covered(&rod, 40.0, speed * frame * 100.0, "advance_rod");
    }

    #[test]
    fn step_rod_implicit_keeps_motion_below_half_an_ulp() {
        let speed = 1.0e-3;
        let (mut rod, material) = translating_rod(speed);
        let h = 1.0e-4;
        assert!(speed * h < 0.1 * half_ulp_at_40(), "premise: h {h:e}");
        for _ in 0..10_000 {
            let params = RodImplicitStepParams {
                gravity: Vec2::ZERO,
                wind_velocity: Vec2::ZERO,
                wind_drag_coeff: 0.0,
                push_center: None,
                push_strength: 0.0,
                push_radius: 0.0,
                dx_meters: 0.01,
                dt: h,
            };
            step_rod_implicit(&mut rod, &material, params);
        }
        assert_covered(&rod, 40.0, speed * h * 10_000.0, "step_rod_implicit");
    }

    #[test]
    fn step_network_keeps_motion_below_half_an_ulp() {
        let speed = 1.0e-3;
        let mut net = build_y_branch(YBranchSpec {
            trunk_start: Vec2::new(40.0, 40.0),
            junction: Vec2::new(40.0, 42.0),
            branch_end: Vec2::new(42.0, 43.0),
            n_trunk_points: 3,
            n_branch_points: 3,
            linear_density_kg_per_m: 0.1,
            dx_meters: 0.01,
            // No stiffness: nothing but the translation acts.
            ea: 0.0,
            ei: 0.0,
            axial_damping: 0.0,
            bending_damping: 0.0,
        });
        net.pinned.iter_mut().for_each(|p| *p = 0);
        net.v.iter_mut().for_each(|v| *v = Vec2::new(speed, 0.0));
        let start: Vec<Vec2> = net.x.clone();
        let h = 1.0e-4;
        assert!(speed * h < 0.1 * half_ulp_at_40(), "premise: h {h:e}");
        for _ in 0..10_000 {
            step_network(&mut net, Vec2::ZERO, 0.01, h);
        }
        let distance = speed * h * 10_000.0;
        for (i, (x, x0)) in net.x.iter().zip(&start).enumerate() {
            let moved = x.x - x0.x;
            assert!(
                (moved - distance).abs() < 0.01 * distance,
                "step_network: point {i} moved {moved:e}, expected {distance:e}"
            );
        }
    }
}

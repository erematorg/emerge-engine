//! Standalone rod time integration -- no grid, no `Simulation` (Phase 0/1).
//! Phase 2's grid-coupled path reuses `forces::compute_internal_forces`
//! directly (see `coupling.rs`) rather than this integrator.

use glam::Vec2;

use super::{RodMaterial, RodPoints, RodRestState, compute_internal_forces};

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
        rod.x[i] += rod.v[i] * dt;
    }
}

/// CFL-safe `dt` bound for the rod's own explicit integrator, covering BOTH
/// stiffness (axial + bending natural frequencies, `dt < 2/omega`) AND
/// damping (axial + bending dashpots, `dt < 2*m/c` -- its own, independent
/// explicit-Euler stability limit) -- the rod's own direct analog of
/// `materials::utils::elastic_wave_dt` PLUS `ViscoelasticMaterial::
/// timestep_bound`'s separate `viscous_dt` term. No `dx_meters` parameter
/// needed -- `mass`/`rest_edge_length` are already real SI (kg/meters), so
/// every bound here comes out in real seconds directly.
///
/// Computed per-point rather than per-edge/vertex: an interior point is
/// coupled to 2 axial edges AND up to 3 overlapping bending vertices
/// simultaneously, so this sums every stiffness/damping term touching each
/// point -- a real Gershgorin circle row-sum bound (for `x''=-M^-1 K x`, the
/// spectral radius of `M^-1 K` is bounded by `max_i(sum_j |K_ij|)/m_i`, the
/// standard way to localize eigenvalues without a full eigendecomposition)
/// -- then takes the min across points. Endpoints see fewer coupled terms
/// and correctly get a larger safe dt than an interior point.
/// Point `i`'s own `(omega_sq, damping_rate)` Gershgorin row-sum, shared by
/// `rod_cfl_dt` and `apply_mass_scaling_for_target_dt` so both work from the
/// exact same real stiffness/damping aggregation -- not two hand-kept-in-
/// sync copies of the same math.
fn point_stability_terms(rod: &RodPoints, material: &RodMaterial, i: usize) -> (f32, f32) {
    let n = rod.len();
    let m = rod.mass[i].max(1.0e-9);
    let mut omega_sq = 0.0f32;
    let mut damping_rate = 0.0f32;
    // Per-vertex stiffness with a uniform-material fallback -- same
    // convention `forces::compute_internal_forces` uses, needed so this
    // CFL bound stays correct (and doesn't panic on an empty slice) for
    // both a uniform rod (built via `Rod::new`, `rod.ea`/`ei` empty until
    // filled) and a genuinely non-uniform one.
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

    if i > 0 {
        let l0 = rod.rest_edge_length[i - 1].max(1.0e-9);
        omega_sq += ea_at(i - 1) / (m * l0);
        damping_rate += material.axial_damping;
    }
    if i + 1 < n {
        let l0 = rod.rest_edge_length[i].max(1.0e-9);
        omega_sq += ea_at(i) / (m * l0);
        damping_rate += material.axial_damping;
    }

    if material.ei > 0.0 || material.bending_damping > 0.0 {
        for k in [i.checked_sub(2), i.checked_sub(1), Some(i)]
            .into_iter()
            .flatten()
        {
            if k + 2 >= n {
                continue;
            }
            let l0_prev = rod.rest_edge_length[k].max(1.0e-9);
            let l0_next = rod.rest_edge_length[k + 1].max(1.0e-9);
            let voronoi_length = 0.5 * (l0_prev + l0_next);
            let ei_k = ei_at(k);
            if ei_k > 0.0 {
                omega_sq += ei_k / (m * voronoi_length.powi(3));
            }
            if material.bending_damping > 0.0 {
                damping_rate += material.bending_damping / voronoi_length.powi(2);
            }
        }
    }

    (omega_sq, damping_rate)
}

pub fn rod_cfl_dt(rod: &RodPoints, material: &RodMaterial, safety: f32) -> f32 {
    let n = rod.len();
    let mut min_dt = f32::INFINITY;

    for i in 0..n {
        let m = rod.mass[i].max(1.0e-9);
        let (omega_sq, damping_rate) = point_stability_terms(rod, material, i);

        if omega_sq > f32::EPSILON {
            min_dt = min_dt.min(safety * 2.0 / omega_sq.sqrt());
        }
        if damping_rate > f32::EPSILON {
            min_dt = min_dt.min(safety * 2.0 * m / damping_rate);
        }
    }

    min_dt
}

/// Real, general mass scaling (Gershgorin CFL row-sum, same math `rod_cfl_dt`
/// already uses -- see `point_stability_terms`) -- a standard, established
/// explicit-FEM stability technique (selective/target mass scaling, e.g.
/// LS-DYNA's own `*CONTROL_TIMESTEP` mass-scaling option): raise a point's
/// OWN inertia just enough that its stiffness-driven CFL bound alone
/// reaches `target_dt`, rather than tuning per-scene multipliers by hand.
/// Real, disclosed tradeoff: this genuinely makes the point heavier (it
/// changes real dynamics -- gravity/wind/push response, not just a CFL-
/// check fudge), so it should only raise mass, never lower it, and should
/// be applied deliberately (an opt-in call), not silently baked into
/// construction. Generalizes to ANY rod/material combination -- not tuned
/// to one scene's own stiffness, the formula derives the exact minimum
/// mass increase needed from each point's own real stiffness terms.
///
/// Does NOT touch the damping-rate CFL term (unaffected by mass scaling in
/// the same way -- `dt < 2*m/damping_rate` already grows linearly with the
/// same added mass, so raising mass to fix the STIFFNESS term also loosens
/// the damping term for free, not fought against).
pub fn apply_mass_scaling_for_target_dt(
    rod: &mut RodPoints,
    material: &RodMaterial,
    safety: f32,
    target_dt: f32,
) {
    let n = rod.len();
    for i in 0..n {
        let (omega_sq, _damping_rate) = point_stability_terms(rod, material, i);
        if omega_sq <= f32::EPSILON {
            continue;
        }
        let m_old = rod.mass[i].max(1.0e-9);
        // omega_sq_old = stiffness_sum / m_old, so stiffness_sum = omega_sq_old * m_old.
        // Solve m_new from: target_dt = safety * 2 / sqrt(stiffness_sum / m_new).
        let stiffness_sum = omega_sq * m_old;
        let m_required = stiffness_sum * (target_dt / (2.0 * safety)).powi(2);
        rod.mass[i] = m_old.max(m_required);
    }
}

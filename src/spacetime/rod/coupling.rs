//! Rod <-> shared MPM grid coupling (Phase 2). Mirrors
//! `spacetime::transfer::p2g`/`g2p`'s own scatter/gather exactly, so a rod
//! and ordinary MPM particles exchange momentum through the identical
//! mechanism -- see `mod.rs`'s doc for why `Grid` being fully
//! source-agnostic makes this a not aspirational, integration.
//!
//! Wired into `Simulation::do_substep` (`solver/step.rs`): scatter after
//! particle P2G, gather after particle G2P, forces after particle force
//! fields -- see that file's comments at each call site for the exact
//! ordering and why it matters (gravity/wake propagation for free).

use glam::Vec2;

use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;

use super::{RodMaterial, RodPoints, RodRestState, compute_internal_forces, rod_cfl_dt};
use crate::spacetime::integration::advance_position;

/// Kernel support radius for `quadratic_weights` is 1.5 grid cells -- two
/// scatter locations spaced up to this far apart still have overlapping
/// stencils, so nothing between them is left uncovered. This is also the
/// trigger threshold for `coverage_samples` below: an edge shorter than
/// this needs no sub-sampling at all, so an already-dense rod (e.g. the
/// blade-of-grass demos) gets zero extra scatter calls -- not just
/// nominal, zero cost when unneeded.
const COVERAGE_SPACING: f32 = 1.5;

/// How many extra coverage sub-samples point `i` owes toward `neighbor`,
/// covering only the near half of that edge (the neighbor covers the far
/// half symmetrically when its own turn comes, so the two together tile the
/// whole edge without overlap or double-counting).
fn extra_samples_toward(rod: &RodPoints, i: usize, neighbor: usize) -> usize {
    let edge_len = (rod.x[neighbor] - rod.x[i]).length();
    if edge_len > COVERAGE_SPACING {
        (edge_len / (2.0 * COVERAGE_SPACING)).floor() as usize
    } else {
        0
    }
}

/// One quadratic-B-spline scatter at `pos`, called once per coverage sample
/// by `scatter_rod_to_grid`. Additive
/// second scatter into the grip field when `contact_group != 0` -- real
/// multi-field frictional contact (Bardenhagen 2001 + Nairn, Hammerquist,
/// Smith 2020), identical mechanism and identical zero-cost-when-unused
/// property as `transfer::p2g`'s own particle scatter (see
/// `Particle::contact_group`/`RodPoints::contact_group` doc).
fn scatter_point(grid: &mut Grid, pos: Vec2, mass: f32, momentum: Vec2, contact_group: u32) {
    let weights = quadratic_weights(pos);
    for gx in 0..3usize {
        for gy in 0..3usize {
            let weight = weights.wx[gx] * weights.wy[gy];
            if weight <= 0.0 {
                continue;
            }
            let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
            grid.add_mass_momentum(cell_pos, weight * mass, weight * momentum);
            if contact_group != 0 {
                grid.add_grip_mass_momentum(cell_pos, weight * mass, weight * momentum);
            }
        }
    }
}

/// Rod -> grid scatter. From the grid's point of view a rod point is just
/// another mass+momentum source (`Cell { momentum, mass }`, keyed by flat
/// cell index only -- `Grid` has no idea this isn't an MPM particle). No
/// stress-as-impulse term here (a rod's internal force is a per-point
/// Newtons force, not a stress tensor) -- mirrors only the mass/velocity part
/// of `scatter_particles_to_grid`, not its stress term.
///
/// **Coverage gap fix** (citable technique, not invented): a rod
/// whose points are spaced farther apart than the kernel's support radius
/// leaves a hole in its grid presence between them -- a small MPM
/// particle (rain) can pass straight through without ever registering
/// contact, since nothing was deposited into the cells in between. Guo,
/// Han, Fu, Gast, Tamstorf, Teran, "A Material Point Method for Thin Shells
/// with Frictional Contact" (SIGGRAPH 2018) hit the identical problem for a
/// coarse shell control mesh and solve it with extra quadrature points that
/// couple to the grid WITHOUT becoming new dynamical DOFs. Same idea here:
/// each point's own (UNCHANGED total) mass is redistributed -- never
/// added -- across extra sample locations placed toward each neighbor, up
/// to that edge's midpoint. Sample weights always sum to exactly 1.0, so
/// total scattered mass and momentum are provably identical to the
/// single-point scatter; only *where* it lands is denser. Every sample uses
/// the point's own velocity (not an inter-point interpolation), trading a
/// minor physical simplification for EXACT -- not merely small -- mass and
/// momentum conservation. Verified against the reproduced bug this
/// fixes in `tests/rod_grid_coupling.rs::coverage_gap_fix_catches_particle_
/// falling_through_sparse_rod_midpoint`.
pub fn scatter_rod_to_grid(rod: &RodPoints, grid: &mut Grid) {
    for i in 0..rod.len() {
        let left = i.checked_sub(1);
        let right = Some(i + 1).filter(|&j| j < rod.len());
        let n_extra_left = left.map_or(0, |j| extra_samples_toward(rod, i, j));
        let n_extra_right = right.map_or(0, |j| extra_samples_toward(rod, i, j));
        let n_samples = 1 + n_extra_left + n_extra_right;

        let point_mass = rod.mass[i] / n_samples as f32;
        let point_momentum = point_mass * rod.v[i];
        let contact_group = rod.contact_group[i];

        scatter_point(grid, rod.x[i], point_mass, point_momentum, contact_group);
        if let Some(j) = left {
            let edge = rod.x[j] - rod.x[i];
            let edge_len = edge.length();
            for k in 1..=n_extra_left {
                let t = (k as f32 * COVERAGE_SPACING) / edge_len;
                scatter_point(
                    grid,
                    rod.x[i] + edge * t,
                    point_mass,
                    point_momentum,
                    contact_group,
                );
            }
        }
        if let Some(j) = right {
            let edge = rod.x[j] - rod.x[i];
            let edge_len = edge.length();
            for k in 1..=n_extra_right {
                let t = (k as f32 * COVERAGE_SPACING) / edge_len;
                scatter_point(
                    grid,
                    rod.x[i] + edge * t,
                    point_mass,
                    point_momentum,
                    contact_group,
                );
            }
        }
    }
}

/// Whether any grid node the rod would scatter to already carries mass
/// (particles, grains, another rod): checked before the rod scatters. Its
/// points and every coverage sample between them count, with the same
/// quadratic stencil as the scatter.
pub fn rod_touches_grid_mass(rod: &RodPoints, grid: &Grid) -> bool {
    let touches = |pos: Vec2| {
        let weights = quadratic_weights(pos);
        (0..3).any(|gx| {
            (0..3).any(|gy| {
                let cell = weights.base_cell + glam::IVec2::new(gx - 1, gy - 1);
                grid.mass_at(cell) > 0.0
            })
        })
    };
    (0..rod.len()).any(|i| {
        if touches(rod.x[i]) {
            return true;
        }
        let Some(j) = Some(i + 1).filter(|&j| j < rod.len()) else {
            return false;
        };
        let edge = rod.x[j] - rod.x[i];
        let len = edge.length();
        let samples = (len / COVERAGE_SPACING).floor() as usize;
        (1..=samples).any(|k| touches(rod.x[i] + edge * (k as f32 * COVERAGE_SPACING / len)))
    })
}

/// Grid -> rod gather: what the grid did to each point's velocity over
/// the substep `dt` (gravity, the exchange with particles, and the pure-PIC
/// averaging with its neighbours), returned as a constant acceleration
/// `(v_grid - v) / dt` for `advance_rod` to apply over the substep. Pinned
/// points get zero. Pure PIC (no APIC/`C`-matrix -- a rod point has no
/// deformation gradient; its own "F" is already fully tracked via edge
/// lengths + curvature), disclosed as slightly more dissipative than the
/// APIC particles sharing its grid; momentum is conserved through the grid
/// either way.
///
/// Why an acceleration and not the grid velocity itself: a stiff rod
/// advances in sub-steps far shorter than `dt`. Taking the grid velocity
/// at the start meant gravity arrived as one kick of `g dt` that the rod
/// then worked off within the substep, so it hung about `g dt^2 / 2` too low
/// on average, 0.14 cell at 1/60 s and 1 cm cells, as large as a cantilever's
/// whole deflection (`tests/subsystem_time_steps.rs`, gate 3). Spread over
/// the substep, a rod at rest balances `g` exactly.
pub fn gather_grid_to_rod(rod: &RodPoints, grid: &Grid, dt: f32) -> Vec<Vec2> {
    let mut external = vec![Vec2::ZERO; rod.len()];
    for (i, out) in external.iter_mut().enumerate() {
        if rod.pinned[i] != 0 {
            continue;
        }
        let contact_group = rod.contact_group[i];
        let weights = quadratic_weights(rod.x[i]);
        let mut v = Vec2::ZERO;
        for gx in 0..3usize {
            for gy in 0..3usize {
                let weight = weights.wx[gx] * weights.wy[gy];
                if weight <= 0.0 {
                    continue;
                }
                let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                // Multi-field contact routing (Bardenhagen 2001), as in
                // `transfer::g2p`: a grip point reads the resolved grip field;
                // both helpers fall back to the ordinary velocity where no
                // contact was registered.
                let node_v = if contact_group != 0 {
                    grid.grip_velocity_at(cell_pos)
                } else {
                    grid.velocity_at(cell_pos)
                };
                v += weight * node_v;
            }
        }
        *out = (v - rod.v[i]) / dt.max(f32::MIN_POSITIVE);
    }
    external
}

/// Lateral push-cursor acceleration, shared by the explicit force pass below
/// and the implicit RHS (`implicit.rs`). Purely lateral, not radial (a
/// radial push degenerates to near-pure axial compression near the rod's
/// centerline, invisible given EA >> EI); defaults to +x on the cursor's own
/// x to avoid a divide-by-zero direction.
///
/// Cutoff and falloff use the 2D distance to the cursor, so the push is
/// local: gating on vertical distance alone let a cursor at the same height
/// as a rod point push it at nearly full strength from any horizontal
/// distance.
pub(crate) fn push_acceleration(
    pos: Vec2,
    push_center: Option<Vec2>,
    push_strength: f32,
    push_radius: f32,
) -> Vec2 {
    match push_center {
        Some(center) if push_radius > 0.0 => {
            let offset = pos - center;
            let dist = offset.length();
            if dist < push_radius {
                let falloff = 1.0 - dist / push_radius;
                let dir_x = if offset.x.abs() > 1.0e-3 {
                    offset.x.signum()
                } else {
                    1.0
                };
                Vec2::new(dir_x * push_strength * falloff, 0.0)
            } else {
                Vec2::ZERO
            }
        }
        _ => Vec2::ZERO,
    }
}

/// Advances the rod over one mechanics substep. `external` is what the grid
/// did to it (`gather_grid_to_rod`): gravity and every exchange with the
/// particles around it, so gravity is not applied again here. That and the
/// rod's own internal (stretch, bend, damping) forces, wind drag and push
/// are integrated in as many equal sub-steps as keep each within
/// `stability_fraction` of the rod's own stable step (`rod_cfl_dt`), each
/// moving the points, so a stiff rod neither needs a tiny mechanics substep
/// nor leaves the grid. Returns the number of sub-steps.
///
/// Positions use compensated summation (`spacetime::integration::advance_position`).
///
/// Bundles this function's own scalar/optional parameters -- the fix
/// for clippy::too_many_arguments rather than suppressing the lint.
/// Deliberately does NOT include `gravity` (unlike
/// `implicit::RodImplicitStepParams`): see above.
pub struct RodForceParams {
    pub wind_velocity: Vec2,
    pub wind_drag_coeff: f32,
    pub push_center: Option<Vec2>,
    pub push_strength: f32,
    pub push_radius: f32,
    pub dx_meters: f32,
    pub dt: f32,
    /// `SimConfig::material_cfl_coefficient`: the fraction of the stable step
    /// each sub-step uses.
    pub stability_fraction: f32,
}

pub fn advance_rod(
    rod: &mut RodPoints,
    material: &RodMaterial,
    params: RodForceParams,
    external: &[Vec2],
) -> u32 {
    let stable = rod_cfl_dt(rod, material, params.stability_fraction);
    let sub_steps = if stable.is_finite() && stable > 0.0 {
        (params.dt / stable).ceil().max(1.0) as u32
    } else {
        1
    };
    let h = params.dt / sub_steps as f32;
    for _ in 0..sub_steps {
        apply_forces(rod, material, &params, h);
        for (i, &acceleration) in external.iter().enumerate() {
            if rod.pinned[i] != 0 {
                rod.v[i] = Vec2::ZERO;
                continue;
            }
            rod.v[i] += acceleration * h;
            advance_position(
                &mut rod.x[i],
                &mut rod.position_compensation[i],
                rod.v[i] * h,
            );
        }
    }
    sub_steps
}

/// One sub-step's worth of internal, wind and push acceleration on `rod.v`.
fn apply_forces(rod: &mut RodPoints, material: &RodMaterial, params: &RodForceParams, dt: f32) {
    let RodForceParams {
        wind_velocity,
        wind_drag_coeff,
        push_center,
        push_strength,
        push_radius,
        dx_meters,
        ..
    } = *params;
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
            continue;
        }
        let a_internal = *internal_force / (rod.mass[i] * dx_meters);
        let a_wind = wind_drag_coeff * (wind_velocity - rod.v[i]);
        let a_push = push_acceleration(rod.x[i], push_center, push_strength, push_radius);
        rod.v[i] += (a_internal + a_wind + a_push) * dt;
    }
}

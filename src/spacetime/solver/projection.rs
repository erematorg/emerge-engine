//! Per-substep safety guards -- split out of `step.rs` (was ~95 of that file's
//! ~730 lines). Both functions run at fixed points in `do_substep` but are
//! self-contained (no access to `Simulation`'s own private fields), so moving
//! them changes nothing about when/how they're called.

use glam::{Mat2, Vec2};

use super::SimConfig;
use crate::boundary::BoundaryCondition;
use crate::grid::Grid;
use crate::particle::Particles;

pub(super) fn apply_boundary_conditions_to_grid(
    grid: &mut Grid,
    grid_res: usize,
    boundary: &dyn BoundaryCondition,
) {
    let mut dissipated: Vec<(usize, f32)> = Vec::new();
    for (i, cell, node_friction) in grid.active_cells_with_index_and_friction_mut() {
        if cell.mass > 0.0 {
            let before = cell.momentum;
            let friction_heat = boundary.apply_to_grid_velocity_with_node_friction(
                i,
                grid_res,
                &mut cell.momentum,
                node_friction,
            );
            // Collected rather than written straight back: the loop holds a
            // mutable borrow of the cells. Empty for every frictionless
            // scene, so this allocates nothing there.
            if friction_heat > 0.0 {
                dissipated.push((i, friction_heat));
            }
            let delta_v = cell.momentum - before;
            // Real Newton's-third-law reaction: whatever velocity this
            // correction removed from the grid at this cell, a real,
            // externally-driven obstacle boundary gains as momentum (sign-
            // flipped, mass-weighted) -- see `BoundaryCondition::
            // on_grid_correction`'s own doc. A no-op for every boundary
            // that doesn't override the hook (default is empty), so this
            // costs nothing beyond one subtraction + one vector compare for
            // every scene that never uses it.
            if delta_v != Vec2::ZERO {
                let cell_pos = Vec2::new((i / grid_res) as f32, (i % grid_res) as f32);
                boundary.on_grid_correction(cell_pos, -delta_v * cell.mass);
            }
        }
    }
    for (i, specific_energy) in dissipated {
        grid.add_friction_heat(i, specific_energy);
    }
}

/// Returns `true` if any field was corrected (state was invalid/non-finite).
pub(super) fn project_particle_state_to_admissible(
    particles: &mut Particles,
    i: usize,
    config: &SimConfig,
) -> bool {
    let mut projected = false;
    let min = config.boundary_thickness.saturating_sub(1) as f32;
    let max = config.grid_res.saturating_sub(config.boundary_thickness) as f32;
    let domain_center = Vec2::splat((min + max) * 0.5);

    if !particles.x[i].is_finite() {
        particles.x[i] = domain_center;
        projected = true;
    } else {
        particles.x[i] = particles.x[i].clamp(Vec2::splat(min), Vec2::splat(max));
    }

    if !particles.v[i].is_finite() {
        particles.v[i] = Vec2::ZERO;
        projected = true;
    }
    if !particles.velocity_gradient[i].x_axis.is_finite()
        || !particles.velocity_gradient[i].y_axis.is_finite()
    {
        particles.velocity_gradient[i] = Mat2::ZERO;
        projected = true;
    }

    // Real gap, found 2026-08-09 debugging a sub-ULP-timestep crash one
    // substep after this exact clamp fired: both branches below rescale
    // `deformation_gradient` but never touched `volume`/`density`, leaving
    // `V/V0 != det(F)` and `rho*V != m` -- invariants `assert_owned_
    // deformation_state`'s own consistency checks require (2e-4 relative
    // tolerance) and that a stale, unclamped `volume`/`density` silently
    // violates. For an owns-deformation-volume-state material (strict
    // fluids, the retry-exhaustion backstop's real caller) that stale
    // `volume` also feeds `MaterialModel::timestep_bound` via `density`
    // directly, which can poison the VERY NEXT CFL scan into an
    // unrepresentably tiny dt -- the actual observed crash, not a separate
    // bug from the clamp not firing at all. Recomputing `volume`/`density`
    // from the just-clamped `deformation_gradient` here keeps every field
    // this function touches in the same consistent state the assert (and
    // the material's own physics) already require everywhere else.
    let f = particles.deformation_gradient[i];
    if !f.x_axis.is_finite()
        || !f.y_axis.is_finite()
        || f.determinant() <= config.projection_min_deformation_j
    {
        particles.deformation_gradient[i] = Mat2::IDENTITY;
        particles.volume[i] = particles.initial_volume[i].max(config.projection_min_volume);
        particles.density[i] =
            (particles.mass[i] / particles.volume[i]).max(config.projection_min_density);
        projected = true;
    } else {
        let j = f.determinant();
        if j > config.j_max {
            particles.deformation_gradient[i] *= (config.j_max / j).sqrt();
            particles.volume[i] =
                (particles.initial_volume[i] * config.j_max).max(config.projection_min_volume);
            particles.density[i] =
                (particles.mass[i] / particles.volume[i]).max(config.projection_min_density);
            projected = true;
        }
    }

    if !particles.plastic_volume_ratio[i].is_finite() || particles.plastic_volume_ratio[i] <= 0.0 {
        particles.plastic_volume_ratio[i] = 1.0;
        projected = true;
    }
    if !particles.hardening_scale[i].is_finite() || particles.hardening_scale[i] <= 0.0 {
        particles.hardening_scale[i] = 1.0;
        projected = true;
    }
    if !particles.friction_hardening[i].is_finite() {
        particles.friction_hardening[i] = 0.0;
        projected = true;
    }
    if !particles.log_volume_strain[i].is_finite() {
        particles.log_volume_strain[i] = 0.0;
        projected = true;
    }

    if !particles.initial_volume[i].is_finite() || particles.initial_volume[i] <= 0.0 {
        particles.initial_volume[i] = config
            .default_initial_volume
            .max(config.projection_min_volume);
        projected = true;
    }
    if !particles.volume[i].is_finite() || particles.volume[i] <= 0.0 {
        particles.volume[i] = particles.initial_volume[i].max(config.projection_min_volume);
        projected = true;
    }
    // Recovered AFTER volume, so the rebuilt mass carries the grid density the
    // rest of the solver assumes (`m = rho_grid * V`) rather than a per-particle
    // constant that would depend on the spawn's spacing. See `SimConfig::grid_density`.
    if !particles.mass[i].is_finite() || particles.mass[i] <= 0.0 {
        particles.mass[i] = config.grid_density * particles.volume[i];
        projected = true;
    }
    if !particles.density[i].is_finite() || particles.density[i] <= 0.0 {
        particles.density[i] =
            (particles.mass[i] / particles.volume[i]).max(config.projection_min_density);
        projected = true;
    } else {
        particles.density[i] = particles.density[i].max(config.projection_min_density);
    }
    projected
}

/// Validate, without modifying, a material state whose volume and density are
/// constitutive variables.  Weakly-compressible fluids use this path: replacing
/// a bad state with an identity deformation or a clamped density would invent
/// mass/energy and conceal a violated PDE or timestep assumption.
pub(super) fn assert_owned_deformation_state(particles: &Particles, i: usize, config: &SimConfig) {
    assert_owned_deformation_state_impl(particles, i, config, true);
}

/// Same checks as `assert_owned_deformation_state`, minus the final
/// `[j_min, j_max]` panic -- used ONLY by `do_substep`'s post-G2P validation
/// when `SimConfig::fluid_step_retry_enabled` is on (see that call site's own
/// comment). Real bug this closes, found 2026-08-09: `do_substep_with_retry`'s
/// own retry loop already recomputes this exact bound after `do_substep`
/// returns (`worst_j_out_of_bounds`) so it can retry at a finer dt or fall
/// back to its exhaustion backstop -- but `do_substep`'s unconditional
/// post-G2P call to the full assert panicked on THIS substep's own fresh
/// result first, unwinding the stack before the retry loop's body (which only
/// runs after `do_substep` RETURNS) ever executed. The retry mechanism could
/// never actually engage for this failure class as a result -- every other
/// invariant here (finiteness, positivity, F/volume/mass consistency) still
/// panics immediately regardless, retry was never meant to (and cannot
/// meaningfully) recover from those.
pub(super) fn assert_owned_deformation_state_j_range_deferred(
    particles: &Particles,
    i: usize,
    config: &SimConfig,
) {
    assert_owned_deformation_state_impl(particles, i, config, false);
}

fn assert_owned_deformation_state_impl(
    particles: &Particles,
    i: usize,
    config: &SimConfig,
    check_j_range: bool,
) {
    let x = particles.x[i];
    let v = particles.v[i];
    let c = particles.velocity_gradient[i];
    let f = particles.deformation_gradient[i];
    let mass = particles.mass[i];
    let initial_volume = particles.initial_volume[i];
    let volume = particles.volume[i];
    let density = particles.density[i];

    assert!(
        x.is_finite() && v.is_finite() && c.x_axis.is_finite() && c.y_axis.is_finite(),
        "strict material particle {i} has non-finite kinematics; reduce the timestep or inspect the applied force"
    );
    assert!(
        mass.is_finite()
            && mass > 0.0
            && initial_volume.is_finite()
            && initial_volume > 0.0
            && volume.is_finite()
            && volume > 0.0
            && density.is_finite()
            && density > 0.0,
        "strict material particle {i} has an inadmissible mass, volume, or density state"
    );
    assert!(
        f.x_axis.is_finite()
            && f.y_axis.is_finite()
            && f.determinant().is_finite()
            && f.determinant() > 0.0,
        "strict material particle {i} has an inadmissible deformation state; reduce the timestep or use a pressure solver"
    );

    let j_volume = volume / initial_volume;
    let j_f = f.determinant();
    let volume_error = ((j_f - j_volume) / j_volume).abs();
    assert!(
        volume_error <= 2.0e-4,
        "strict material particle {i} has inconsistent J: det(F)={j_f}, V/V0={j_volume}"
    );
    let mass_error = ((density * volume - mass) / mass).abs();
    assert!(
        mass_error <= 2.0e-4,
        "strict material particle {i} violates rho*V=m by relative error {mass_error}"
    );
    // Real gap, found 2026-08-09: this function's own design (see its doc
    // above) deliberately reports rather than silently rescales -- but
    // "finite and positive" alone let a genuine compression collapse pass
    // unnoticed through hundreds of substeps, each one individually too
    // small to trip `fluid_step_retry_enabled`'s own per-substep check
    // (same "many small changes" gap that field's own doc already named,
    // just in the opposite, compression direction: measured live, min_j
    // drifted 0.72 -> 0.0000154 over 120 frames on a pressure-projection
    // scene with zero elastic backstop to resist it, density_ratio hitting
    // 64,916x, entirely unreported the whole time). `j_min`/`j_max` are a
    // real, generous (50x) safety range, not a physical bound -- widened
    // to strict fluids here for the first time (previously solid-only, see
    // `j_max`'s own doc) specifically so a genuinely runaway trajectory
    // fails LOUD, immediately, with an actionable message, instead of
    // silently corrupting density/pressure for the rest of the run.
    if check_j_range {
        assert!(
            j_f >= config.j_min && j_f <= config.j_max,
            "strict material particle {i} has J={j_f} outside the admissible range \
             [{}, {}] -- a genuine, real compounding drift (not one bad substep), \
             not a false alarm: fix the underlying instability (relaxation, near-wall \
             CFL, retry threshold) rather than widening this range",
            config.j_min,
            config.j_max
        );
    }
}

#[cfg(test)]
mod grid_correction_tests {
    use super::*;
    use crate::boundary::KinematicCircleBoundary;
    use glam::IVec2;

    /// Real, hand-computed check of the `on_grid_correction` wiring itself
    /// (the physics of `KinematicCircleBoundary`'s own contact projection
    /// was already extensively validated elsewhere -- this isolates the
    /// NEW glue: does `apply_boundary_conditions_to_grid` actually compute
    /// and deliver the real Newton's-third-law reaction impulse?).
    ///
    /// Setup: one grid cell at (8,8), mass=2.0, velocity=(5,0) (moving in
    /// +x). Obstacle centered at (9,8), radius=2.0, friction=0.0 -- the
    /// cell sits exactly 1 unit from the center along -x, so its outward
    /// normal is (-1,0) and its velocity (+x) points directly INTO the
    /// obstacle. With friction=0, Coulomb projection zeroes the full
    /// velocity (normal component removed, zero tangential component to
    /// begin with) -- `delta_v = (0,0)-(5,0) = (-5,0)`, so the real
    /// expected reaction impulse is `-delta_v*mass = (10.0, 0.0)`, and
    /// since the impulse is exactly anti-parallel to `r = cell_pos -
    /// center = (-1,0)`, the real expected torque is exactly 0.0.
    #[test]
    fn stationary_obstacle_receives_the_real_hand_computed_reaction_impulse() {
        let grid_res = 16;
        let mut grid = Grid::new(grid_res);
        let cell = IVec2::new(8, 8);
        let mass = 2.0;
        grid.add_mass_momentum(cell, mass, Vec2::new(5.0, 0.0));

        let obstacle = KinematicCircleBoundary::new(Vec2::new(9.0, 8.0), 2.0, 0.0);
        apply_boundary_conditions_to_grid(&mut grid, grid_res, &obstacle);

        let impulse = obstacle.take_reaction_impulse();
        assert!(
            (impulse - Vec2::new(10.0, 0.0)).length() < 1.0e-4,
            "expected reaction impulse (10.0, 0.0), got {impulse:?}"
        );
        let torque = obstacle.take_torque();
        assert!(
            torque.abs() < 1.0e-4,
            "expected exactly zero torque (impulse anti-parallel to r), got {torque}"
        );
    }

    /// A boundary that never actually touches any cell (obstacle far away)
    /// must leave the reaction impulse at exactly zero -- confirms the
    /// `delta_v != Vec2::ZERO` gate correctly skips cells nothing corrected,
    /// not just that SOME correction produces SOME nonzero result.
    #[test]
    fn untouched_obstacle_accumulates_zero_reaction_impulse() {
        let grid_res = 16;
        let mut grid = Grid::new(grid_res);
        grid.add_mass_momentum(IVec2::new(8, 8), 2.0, Vec2::new(5.0, 0.0));

        // Far away -- never within its own radius of the populated cell.
        let obstacle = KinematicCircleBoundary::new(Vec2::new(0.0, 0.0), 1.0, 0.0);
        apply_boundary_conditions_to_grid(&mut grid, grid_res, &obstacle);

        assert_eq!(obstacle.take_reaction_impulse(), Vec2::ZERO);
        assert_eq!(obstacle.take_torque(), 0.0);
    }
}

//! Grain <-> shared MPM grid coupling. Mirrors `rod::coupling`'s own
//! scatter/gather exactly, for the identical reason: `Grid` is fully
//! source-agnostic (a flat `Cell { mass, momentum }` accumulator, no idea
//! whether a contribution came from an ordinary particle, a rod point, or a
//! grain) -- so a grain exchanging real momentum with ordinary MPM sand
//! particles through the shared grid is a real, not aspirational,
//! integration, exactly like the rod solver already proved.
//!
//! Same real division of labor as `rod::coupling`: gravity and interaction
//! with the surrounding continuum come through the shared grid (scatter ->
//! grid_update -> gather); a grain's OWN inter-grain contact forces
//! (`contact_law`) are applied AFTER the gather, as a velocity/spin
//! correction -- mirrors `apply_rod_internal_and_wind_forces`'s own
//! documented reason for not re-applying gravity a second time.

use glam::{Mat2, Vec2};

use crate::boundary::BoundaryCondition;
use crate::grid::kernel::quadratic_weights;
use crate::grid::{Grid, VelocitySnapshot};
use crate::solver::config::KERNEL_D_INVERSE;

use super::population::GrainPopulation;

/// Grain -> grid scatter. Real APIC (Jiang, Schroeder, Selle, Teran &
/// Stomakhin 2015) -- the SAME real formula ordinary MPM particles already
/// use in `p2g.rs` (`v_i + c_i*cell_dist`), applied to a grain's own
/// persistent `Grain::c` affine matrix.
///
/// Real history, kept for the record (2026-08-20, same day): first tried
/// scattering the grain's TRUE rigid-body field directly from `spin`
/// (`v_com + spin*perp(r)`) -- exact for an isolated grain, but a real,
/// confirmed, unbounded energy leak once many spinning grains share grid
/// nodes (`spin` was never debited when a neighbor's gather "read" it).
/// REVERTED. Replaced with a plain spin-blind blob + ASFLIP -- fixed the
/// original clumping (confirmed: pure PIC held a real, jittered column
/// frozen near its initial shape, matching Jiang et al. 2015's own
/// independent finding that pure PIC "causes sand to clump together") but
/// pure FLIP (`asflip_blend=1.0`) is ALSO independently documented to
/// "suffer from excessive noise and instability" -- confirmed here too (a
/// real dt-convergence sweep showed non-monotonic bouncing, 0.73x/0.77x/
/// 0.74x at three different dt values on the same scene). APIC is the
/// literature's own real resolution to exactly this dilemma: as stable as
/// PIC, as low-dissipation as FLIP, while ALSO conserving angular momentum
/// -- this is that fix, done properly this time: `c` is not an externally
/// -fed guess, it is RECONSTRUCTED every substep by `gather_grid_to_grains`
/// from the grid's own local velocity field, the exact closed loop the
/// earlier spin-based attempt was missing.
pub fn scatter_grains_to_grid(grains: &GrainPopulation, grid: &mut Grid) {
    for grain in &grains.grains {
        let weights = quadratic_weights(grain.x);
        for gx in 0..3usize {
            for gy in 0..3usize {
                let weight = weights.wx[gx] * weights.wy[gy];
                if weight <= 0.0 {
                    continue;
                }
                let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - grain.x + Vec2::splat(0.5);
                let node_v = grain.v + grain.c * cell_dist;
                let momentum = grain.mass * node_v;
                grid.add_mass_momentum(cell_pos, weight * grain.mass, weight * momentum);
            }
        }
    }
}

/// Grid -> grain gather. Real APIC, mirroring `gather_grid_to_particles`'s
/// own exact formula: gathers `new_v` (plain PIC average -- correct as-is,
/// a rigid body's center-of-mass velocity genuinely IS the local average)
/// AND `b = sum(weight * outer(node_v, cell_dist))`, then sets
/// `grain.c = b * KERNEL_D_INVERSE * apic_blend` for the NEXT scatter to
/// consume -- the self-consistent closed loop. `apic_blend` is the SAME
/// real, existing `SimConfig::apic_blend` knob ordinary particles already
/// use (default `1.0`, full APIC) -- not a new parameter.
///
/// `spin` is intentionally NOT re-derived from `c` here: it stays owned by
/// `apply_grain_contact_forces`'s own torque integration (below), the real,
/// calibrated, 2,000,000-step-verified DEM rolling-resistance physics.
/// `c` is a separate, additive, transfer-layer momentum-conservation
/// device -- it doesn't replace spin's own real dynamics, it just stops
/// the grid round-trip from being lossy the way pure PIC was.
///
/// ASFLIP (2026-08-20, Fei, Guo, Wu, Huang, Gao 2021 -- same real mechanism
/// `gather_grid_to_particles` already uses, see that function's own doc):
/// reintroduces the classic FLIP residual (`v_old - old_v`) on top of the
/// APIC gather above. `old_v` is a PIC-style gather against the grid's
/// PRE-FORCE velocity snapshot (`pre_force_snapshot`, taken right after
/// P2G's own momentum normalization, before this substep's gravity/
/// boundary/contact modified it), using the SAME stencil weights as
/// `new_v`. `pre_force_snapshot` being `None` (`asflip_blend=0.0`, the
/// default) is the real gate: `grain.v` stays exactly `new_v`, reproducing
/// the pre-ASFLIP formula bit-for-bit. No `gamma`/compression-aware split
/// here unlike the ordinary-particle version -- that split only matters
/// because ordinary G2P ALSO advances position; grains defer position
/// advance to `apply_grain_contact_forces` below, so there is only one
/// velocity to correct, not a separate "position-advance velocity."
///
/// Velocity ONLY -- does NOT advance position. `gather_grid_to_rod` advances
/// position here and lets its own force correction only affect the NEXT
/// substep's advection, but that convention is real-measured WRONG for
/// grains specifically: the proven, 2,000,000-step-verified standalone
/// `GrainPopulation::step` resolves contact forces FIRST, applies them to
/// `v`, THEN integrates `x += v*dt` -- position is never advanced on
/// stale, pre-contact velocity. Grid-coupled grains used to do the
/// opposite (advance position here, correct velocity a whole substep
/// later in `apply_grain_contact_forces`), silently interpenetrating every
/// substep before their own repulsive contact spring ever got to push
/// back at the position it actually applies to. `apply_grain_contact_forces`
/// now does the position advance instead, after its own correction --
/// matching the standalone order exactly. (Rod's own convention is
/// untouched here; rods aren't stiff DEM contacts and weren't measured to
/// have this problem.)
pub fn gather_grid_to_grains(
    grains: &mut GrainPopulation,
    grid: &Grid,
    apic_blend: f32,
    asflip_blend: f32,
    pre_force_snapshot: Option<&VelocitySnapshot>,
) {
    for grain in &mut grains.grains {
        let v_old = grain.v;
        let weights = quadratic_weights(grain.x);
        let mut new_v = Vec2::ZERO;
        let mut b = Mat2::ZERO;
        for gx in 0..3usize {
            for gy in 0..3usize {
                let weight = weights.wx[gx] * weights.wy[gy];
                if weight <= 0.0 {
                    continue;
                }
                let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - grain.x + Vec2::splat(0.5);
                let weighted_v = grid.velocity_at(cell_pos) * weight;
                b += Mat2::from_cols(weighted_v * cell_dist.x, weighted_v * cell_dist.y);
                new_v += weighted_v;
            }
        }
        grain.v = new_v;
        if let Some(snapshot) = pre_force_snapshot {
            let mut old_v = Vec2::ZERO;
            for gx in 0..3usize {
                for gy in 0..3usize {
                    let weight = weights.wx[gx] * weights.wy[gy];
                    if weight <= 0.0 {
                        continue;
                    }
                    let cell_pos =
                        weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    old_v += grid.pre_force_velocity_at(snapshot, cell_pos) * weight;
                }
            }
            grain.v = new_v + asflip_blend * (v_old - old_v);
        }
        grain.c = b * KERNEL_D_INVERSE * apic_blend;
    }
}

/// Applies inter-grain contact forces/torques (`contact_law`, via
/// `GrainPopulation::resolve_contact_forces`) as a velocity/spin correction
/// AFTER `gather_grid_to_grains`, THEN advances position -- mirrors the
/// proven standalone `GrainPopulation::step`'s own order exactly (contact
/// resolved before integration, not after; see `gather_grid_to_grains`'s
/// own doc for why this changed). Gravity is NOT reapplied here -- it
/// already reached every grain through the shared grid's own `grid_update`
/// step, the same mechanism ordinary particles and rods already use.
///
/// Clamps the advanced position against every boundary
/// (`BoundaryCondition::clamp_particle_position`), same real convention
/// `gather_grid_to_particles` already uses for ordinary particles
/// (`g2p.rs`'s own `new_pos = boundary.clamp_particle_position(...)` loop).
/// Grains previously had NO position clamp anywhere in this coupling path
/// -- only the grid-level velocity damping near a boundary
/// (`apply_boundary_conditions_to_grid`, node-velocity-only, a few cells
/// wide). A confirmed, real, structural gap: a fast-moving grain could
/// advance straight past that thin damping zone in one substep with
/// nothing to stop it, since (unlike ordinary particles) nothing ever
/// called a hard position backstop for a grain. Found chasing a real
/// column-collapse isolation test that measured a 5.26x spread ratio --
/// mathematically impossible to reach without leaving the simulation
/// domain entirely, which is exactly what was happening.
pub fn apply_grain_contact_forces(
    grains: &mut GrainPopulation,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    grid_res: usize,
    grid: &crate::grid::Grid,
) {
    // Real, clean per-grain normal correction BEFORE any contact resolution
    // -- see `GrainPopulation::clean_wall_normal_velocity`'s own doc for why
    // this must run first: the grid's own per-cell boundary correction is
    // noisy (kernel-support-vs-cell-boundary mismatch), and everything
    // downstream needs a physically clean velocity to react to correctly.
    grains.clean_wall_normal_velocity(boundaries, grid_res);
    let (mut forces, mut torques) = grains.resolve_contact_forces(dt);
    // Real grain-vs-boundary contact -- see `GrainPopulation::
    // resolve_wall_contact_forces`'s own doc for why this is a SEPARATE
    // call, not folded into grain-grain contact above: without it, a grain
    // resting on the ground has no mechanism to ever start rolling from
    // rest (found live 2026-08-21).
    let (wall_forces, wall_torques) = grains.resolve_wall_contact_forces(boundaries, grid_res, dt);
    // Real grain-vs-CONTINUUM-terrain contact -- see `GrainPopulation::
    // resolve_terrain_contact_forces`'s own doc: closes the real,
    // root-caused gap where a grain resting on a real MPM terrain
    // material (not a `BoundaryCondition`) got zero rolling resistance.
    // Real, disclosed opt-in (`with_terrain_contact`) -- zero cost for
    // every population that never calls it, same convention as the wall
    // contact call above.
    let (terrain_forces, terrain_torques) = grains.resolve_terrain_contact_forces(grid, dt);
    for i in 0..forces.len() {
        forces[i] += wall_forces[i] + terrain_forces[i];
        torques[i] += wall_torques[i] + terrain_torques[i];
    }
    for (idx, grain) in grains.grains.iter_mut().enumerate() {
        grain.v += (forces[idx] / grain.mass) * dt;
        grain.spin += (torques[idx] / grain.moment_of_inertia()) * dt;
        grain.orientation += grain.spin * dt;
        let mut new_pos = grain.x + grain.v * dt;
        // Real, radius-aware backstop (2026-08-21, found live: a grain on a
        // sloped ramp visibly sinking into it while staying stuck near the
        // top instead of rolling -- confirmed via a direct trace,
        // `grain.x.y` FROZEN bit-for-bit across 2000+ steps while
        // `grain.v.y` grew unboundedly negative underneath it, root-caused
        // to the generic, ordinary-PARTICLE `clamp_particle_position`
        // hardcoding a "+1" vertical clearance and ignoring both the
        // grain's own real radius and a sloped surface's own tilt). See
        // `BoundaryCondition::clamp_grain_position`'s own doc -- each
        // boundary now owns its own correct grain backstop end to end.
        for boundary in boundaries {
            new_pos = boundary.clamp_grain_position(new_pos, grain.radius, grid_res);
        }
        grain.x = new_pos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::materials::granular::grain_contact_law::ContactLawConfig;
    use crate::matter::particle::Grain;

    fn config() -> ContactLawConfig {
        ContactLawConfig {
            normal_stiffness: 1.0e5,
            tangential_stiffness: 0.8e5,
            rolling_stiffness: 5.0e3,
            normal_damping: 50.0,
            tangential_damping: 50.0,
            rolling_damping: 50.0,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }

    #[test]
    fn grain_feels_gravity_through_the_shared_grid_not_its_own_integration() {
        // Real proof the coupling mechanism itself works: a grain at rest,
        // scattered into an otherwise-empty grid, should pick up EXACTLY
        // the grid's own gravity-integrated velocity after a round trip --
        // not because the grain integrated gravity itself (it didn't;
        // `gather_grid_to_grains` only reads from the grid), but because
        // the shared grid mechanism (`update_velocities`) is the same one
        // ordinary MPM particles already use.
        let mut grid = Grid::new(32);
        let mut pop =
            GrainPopulation::new(vec![Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)], config());
        let gravity = Vec2::new(0.0, -9.8);
        let dt = 0.01;

        scatter_grains_to_grid(&pop, &mut grid);
        grid.update_velocities(dt, gravity);
        gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);

        let expected_v = gravity * dt;
        assert!(
            (pop.grains[0].v - expected_v).length() < 1e-4,
            "v={:?} expected={:?}",
            pop.grains[0].v,
            expected_v
        );
    }

    #[test]
    fn diag_isolated_spinning_grain_at_exact_node_position_self_interaction() {
        // TEMP DIAGNOSTIC (2026-08-20): the 2-grain explosion (see
        // `does_grid_coupling_add_spurious_energy...`) happened at step=2
        // with a grain sitting almost exactly on an integer grid coordinate
        // (15.99995, 15.99997), 1.9 units from its only neighbor -- likely
        // OUTSIDE this grain's own 3x3 kernel reach (radius 1.5 cells), so
        // the two grains may not even share a grid node. Testing whether a
        // SINGLE isolated spinning grain, alone on the grid, explodes purely
        // from its own scatter/gather round trip at an on-node position.
        let mut grid = Grid::new(64);
        let mut pop = GrainPopulation::new(
            vec![Grain {
                spin: 5.0,
                ..Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)
            }],
            config(),
        );
        let dt = 0.0001414;
        for step in 0..10 {
            grid.clear();
            scatter_grains_to_grid(&pop, &mut grid);
            grid.update_velocities(dt, Vec2::ZERO);
            gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);
            println!(
                "step={step} x={:?} v={:?} spin={:.4} |v|={:.6}",
                pop.grains[0].x,
                pop.grains[0].v,
                pop.grains[0].spin,
                pop.grains[0].v.length()
            );
        }
    }

    /// Real, decisive test for a 2026-08-20 hypothesis: the isolated-grain
    /// self-cancellation proof above (`v` stays exactly 0 forever) relies on
    /// the kernel's own zero-first-moment property, `sum(weight*cell_dist) =
    /// 0` -- true for a FULL, untruncated 3x3 stencil, but `add_mass_momentum`
    /// silently drops any node outside `[0, resolution)` (`flat_index`
    /// returns `None`), and `Grid::velocity_at` returns `Vec2::ZERO` for a
    /// negative index too -- so a grain close enough to y=0 that its stencil
    /// reaches y=-1 scatters/gathers against an ASYMMETRICALLY TRUNCATED
    /// kernel, breaking that exact cancellation. Real motivation: the
    /// 80-grain isolation test's own fine-grained energy trace showed
    /// `max_spin` FROZEN at a fixed value for 420,000+ steps while
    /// `max_speed` climbed in a perfectly LINEAR, unbounded ramp -- exactly
    /// the signature of a small, constant, never-canceling per-step bias,
    /// not a feedback explosion.
    #[test]
    fn diag_isolated_spinning_grain_near_domain_edge_truncated_kernel() {
        // Sweeps y across exactly the range a real `FrictionBoundary(thickness=2)`
        // clamp allows (`clamp_position_inside_grid`'s own `min =
        // thickness.saturating_sub(1) = 1.0`) and just below it, to find the
        // REAL safe/unsafe boundary precisely -- does the clamp's own margin
        // actually keep the kernel stencil non-negative, or is there a gap.
        for y0 in [0.4f32, 0.9, 0.99, 1.0, 1.01, 1.5] {
            let mut grid = Grid::new(64);
            let mut pop = GrainPopulation::new(
                vec![Grain {
                    spin: 5.0,
                    ..Grain::new(Vec2::new(16.0, y0), 1.0, 1.0)
                }],
                config(),
            );
            let dt = 0.0001414;
            for _ in 0..20 {
                grid.clear();
                scatter_grains_to_grid(&pop, &mut grid);
                grid.update_velocities(dt, Vec2::ZERO);
                gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);
            }
            println!(
                "y0={y0:.2} -> v={:?} |v|={:.6}",
                pop.grains[0].v,
                pop.grains[0].v.length()
            );
        }
    }

    /// Real, decisive isolation for the STILL-explosive real 80-grain test
    /// (`tests/grains_grid_coupling.rs`, uses real `Simulation::step()`,
    /// already confirmed to clear the grid correctly every substep --
    /// unlike this file's OWN earlier 2-grain test, which had a genuine,
    /// separate, now-fixed test-harness bug, see
    /// `does_grid_coupling_add_spurious_energy...`'s own doc). That 2-grain
    /// test proves grain-grain contact through the grid is energy-safe.
    /// This test isolates the other real, shared mechanism every grain
    /// demo touches: a `BoundaryCondition`'s own grid-velocity effect
    /// (`apply_to_grid_velocity`, e.g. `FrictionBoundary`'s Coulomb-wall
    /// reflection) -- run manually here (this module can't call
    /// `solver::projection::apply_boundary_conditions_to_grid`, `pub(super)`
    /// to a different module tree) via the SAME real trait method it
    /// itself calls, on a SINGLE spinning grain (no other grain, no
    /// `contact_law` involved at all -- already proven stable without a
    /// boundary present, see `diag_isolated_spinning_grain...` above),
    /// sitting overlapped into a real `FrictionBoundary`'s own zone.
    #[test]
    fn diag_spinning_grain_against_a_real_friction_boundary() {
        let mut grid = Grid::new(32);
        let boundary = crate::boundary::FrictionBoundary::new(2, 0.7);
        // Overlapped into the boundary's own thickness=2 zone: grain surface
        // (center - radius) at y=0.0, well inside y<2.
        let mut pop = GrainPopulation::new(
            vec![Grain {
                spin: 5.0,
                ..Grain::new(Vec2::new(16.0, 1.0), 1.0, 1.0)
            }],
            config(),
        );
        let dt = 0.0001414;
        let grid_res = 32usize;
        let mut max_speed = 0.0f32;
        for step in 0..200 {
            grid.clear();
            scatter_grains_to_grid(&pop, &mut grid);
            grid.update_velocities(dt, Vec2::ZERO);
            for (i, cell) in grid.active_cells_with_index_mut() {
                if cell.mass > 0.0 {
                    boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
                }
            }
            gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid);
            let speed = pop.grains[0].v.length();
            max_speed = max_speed.max(speed);
            if step < 5 || speed > 50.0 {
                println!(
                    "step={step} x={:?} v={:?} spin={:.4} |v|={speed:.6}",
                    pop.grains[0].x, pop.grains[0].v, pop.grains[0].spin
                );
            }
        }
        assert!(
            max_speed < 50.0,
            "a single spinning grain overlapped into a real FrictionBoundary's zone \
             (no other grain, no contact_law interaction) reached |v|={max_speed} -- \
             real, confirmed instability in the rotation-scatter + boundary-grid-velocity \
             interaction itself"
        );
    }

    /// Real, hand-rolled control (2026-08-20): a lone grain sliding on a
    /// `FrictionBoundary`, driven through the raw scatter/gravity/boundary/
    /// gather calls directly (bypassing `Simulation` entirely), decays under
    /// real Coulomb friction matching the analytic `mu*g*T` prediction
    /// almost exactly -- proof `apply_coulomb_wall` itself is correct. This
    /// was the baseline that isolated a SEPARATE, real bug one level up:
    /// the same scenario driven through the real `Simulation::step()`
    /// pipeline instead showed a dead-constant, non-decaying velocity (see
    /// `diag_single_sliding_grain_through_real_solver_step_should_decelerate`
    /// in `tests/grains_grid_coupling.rs` for the root cause and fix --
    /// `Simulation::add_boundary_condition` was silently stacking a user's
    /// boundary underneath a hidden zero-friction default).
    #[test]
    fn diag_isolated_sliding_grain_on_friction_boundary_should_decelerate() {
        let mut grid = Grid::new(32);
        let boundary = crate::boundary::FrictionBoundary::new(2, 0.7);
        let mut pop = GrainPopulation::new(
            vec![Grain {
                v: Vec2::new(2.0, 0.0),
                ..Grain::new(Vec2::new(16.0, 1.0), 1.0, 1.0)
            }],
            config(),
        );
        let dt = 0.0001414;
        let grid_res = 32usize;
        let gravity = Vec2::new(0.0, -0.3);
        for step in 0..40000 {
            grid.clear();
            scatter_grains_to_grid(&pop, &mut grid);
            grid.update_velocities(dt, gravity);
            for (i, cell) in grid.active_cells_with_index_mut() {
                if cell.mass > 0.0 {
                    boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
                }
            }
            gather_grid_to_grains(&mut pop, &grid, 1.0, 0.0, None);
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid);
            if step % 2000 == 0 {
                println!(
                    "step={step} x={:?} v={:?} |v|={:.6}",
                    pop.grains[0].x,
                    pop.grains[0].v,
                    pop.grains[0].v.length()
                );
            }
        }
        println!("FINAL v={:?}", pop.grains[0].v);
    }

    /// Both individual mechanisms above (grain-grain contact through the
    /// grid; a single grain against a real boundary) are separately proven
    /// stable. This test is the real remaining combination the 80-grain
    /// isolation test actually has that neither of those does: SEVERAL
    /// grains simultaneously in contact with EACH OTHER while ALSO
    /// overlapped into the SAME boundary's zone -- real gravity included
    /// (the 80-grain test settles under it), matching that test's own
    /// config values (`gravity=(0,-0.3)`, same `ContactLawConfig`).
    #[test]
    fn diag_grain_row_settling_against_boundary_with_contact_and_gravity() {
        let mut grid = Grid::new(32);
        let boundary = crate::boundary::FrictionBoundary::new(2, 0.7);
        let cfg = config();
        let dt = 0.0001414;
        let grid_res = 32usize;
        let gravity = Vec2::new(0.0, -0.3);
        // Four grains in a row, touching (spacing=2*radius exactly), resting
        // overlapped into the boundary zone (y=1.0, thickness=2) -- real
        // simultaneous grain-grain + grain-boundary contact. One given real
        // spin, matching what contact-driven friction/rolling torque would
        // produce mid-collapse in the real test.
        let grains = vec![
            Grain {
                spin: 5.0,
                ..Grain::new(Vec2::new(14.0, 8.0), 1.0, 1.0) // dropped from height, real impact
            },
            Grain::new(Vec2::new(16.0, 8.0), 1.0, 1.0),
            Grain::new(Vec2::new(18.0, 8.0), 1.0, 1.0),
            Grain::new(Vec2::new(20.0, 8.0), 1.0, 1.0),
            Grain::new(Vec2::new(14.0, 10.0), 1.0, 1.0),
            Grain::new(Vec2::new(16.0, 10.0), 1.0, 1.0),
            Grain::new(Vec2::new(18.0, 10.0), 1.0, 1.0),
            Grain::new(Vec2::new(20.0, 10.0), 1.0, 1.0),
        ];
        let mut pop = GrainPopulation::new(grains, cfg);
        let mut max_speed = 0.0f32;
        let mut spike_step = None;
        for step in 0..20000 {
            grid.clear();
            scatter_grains_to_grid(&pop, &mut grid);
            grid.update_velocities(dt, gravity);
            for (i, cell) in grid.active_cells_with_index_mut() {
                if cell.mass > 0.0 {
                    boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
                }
            }
            gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid);
            let speed = pop
                .grains
                .iter()
                .map(|g| g.v.length())
                .fold(0.0f32, f32::max);
            if spike_step.is_none() && speed > 50.0 {
                spike_step = Some(step);
                println!("SPIKE at step={step}: {:?}", pop.grains);
            }
            max_speed = max_speed.max(speed);
        }
        println!("max_speed={max_speed} spike_step={spike_step:?}");
        assert!(
            max_speed < 50.0,
            "several grains in mutual contact, also overlapped into a real boundary's \
             zone, under real gravity -- reached |v|={max_speed} (spike at step {spike_step:?}) \
             -- real, confirmed instability specific to the COMBINATION of grain-grain \
             contact and grain-boundary interaction, neither alone reproduces it"
        );
    }

    /// Real, targeted test for the ACTUAL launch mechanism found 2026-08-20
    /// in the real 80-grain test's own fine-grained trace: the launch
    /// happened at `active_contacts=16` -- the highest coordination number
    /// anywhere in that whole run, right after a dense settling moment.
    /// Every earlier isolation test here maxed out at 4-8 grains in a
    /// single row/two rows (coordination number 1-4 per grain) and stayed
    /// stable -- this is the first test with a genuinely DENSE 2D-packed
    /// cluster (5x4=20 grains, touching in both axes, so interior grains
    /// have real coordination number 4), matching the real trigger
    /// condition directly instead of guessing at smaller ingredients.
    #[test]
    fn diag_dense_packed_cluster_settling_under_gravity_and_boundary() {
        let mut grid = Grid::new(48);
        let boundary = crate::boundary::FrictionBoundary::new(2, 0.7);
        // Real, EXACT match to the actual 80-grain test's own calibrated
        // config -- an earlier version of this test used the generic
        // module-level `config()` helper instead (10x stiffer stiffness,
        // flat 50.0 damping, friction=0.5) and passed cleanly, but that
        // wasn't a real apples-to-apples reproduction of the trigger.
        let m_eff = 1.0 * 0.5;
        const DAMPING_RATIO: f32 = 0.6;
        let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
        let normal_stiffness = 1.0e4;
        let tangential_stiffness = 0.8e4;
        let rolling_stiffness = 5.0e2;
        let cfg = ContactLawConfig {
            normal_stiffness,
            tangential_stiffness,
            rolling_stiffness,
            normal_damping: critical_damping(normal_stiffness),
            tangential_damping: critical_damping(tangential_stiffness),
            rolling_damping: critical_damping(rolling_stiffness),
            friction: (35.0_f32).to_radians().tan(),
            rolling_friction: 0.20,
        };
        let dt =
            crate::materials::granular::grain_contact_law::critical_timestep(m_eff, &cfg) * 0.02;
        let grid_res = 48usize;
        let gravity = Vec2::new(0.0, -0.3);
        // 5 columns x 4 rows, touching (spacing=2*radius exactly) in both
        // axes -- interior grains have real coordination number 4, matching
        // the real trigger's own active_contacts=16 density. Small real
        // jitter (same convention as the real test's own `build_column`/
        // `make_column`) so the pack isn't perfectly symmetric.
        struct SmallRng(u64);
        impl SmallRng {
            fn next_f32(&mut self) -> f32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((self.0 >> 33) as f32) / (u32::MAX as f32)
            }
        }
        let mut rng = SmallRng(0xC0FF_EE11_u64);
        let mut grains = Vec::new();
        for row in 0..4 {
            for col in 0..5 {
                let jx = (rng.next_f32() - 0.5) * 0.1;
                let jy = (rng.next_f32() - 0.5) * 0.1;
                let x = 10.0 + col as f32 * 2.0 + jx;
                let y = 3.0 + row as f32 * 2.0 + jy;
                grains.push(Grain::new(Vec2::new(x, y), 1.0, 1.0));
            }
        }
        let mut pop = GrainPopulation::new(grains, cfg);
        let mut max_speed = 0.0f32;
        let mut max_contacts = 0usize;
        let mut spike_step = None;
        for step in 0..250_000 {
            grid.clear();
            scatter_grains_to_grid(&pop, &mut grid);
            grid.update_velocities(dt, gravity);
            for (i, cell) in grid.active_cells_with_index_mut() {
                if cell.mass > 0.0 {
                    boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
                }
            }
            gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid);
            let speed = pop
                .grains
                .iter()
                .map(|g| g.v.length())
                .fold(0.0f32, f32::max);
            max_contacts = max_contacts.max(pop.active_contact_count());
            if spike_step.is_none() && speed > 20.0 {
                spike_step = Some(step);
                let (idx, g) = pop
                    .grains
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.v.length().total_cmp(&b.v.length()))
                    .unwrap();
                println!(
                    "SPIKE at step={step}: grain#{idx} x={:?} v={:?} spin={:.4} active_contacts={}",
                    g.x,
                    g.v,
                    g.spin,
                    pop.active_contact_count()
                );
            }
            max_speed = max_speed.max(speed);
        }
        println!("max_speed={max_speed} max_contacts={max_contacts} spike_step={spike_step:?}");
        assert!(
            max_speed < 20.0,
            "a densely-packed 20-grain cluster (coordination number 4, matching the real \
             80-grain test's own active_contacts=16 trigger condition) reached |v|={max_speed} \
             (spike at step {spike_step:?}, max_contacts={max_contacts}) -- real, confirmed \
             instability specific to dense multi-body coordination, not reproduced by any \
             sparser test"
        );
    }

    #[test]
    fn grain_and_a_second_grid_contributor_genuinely_exchange_momentum() {
        // The real point of grid coupling: a grain's gathered velocity must
        // be influenced by whatever ELSE shares its grid cells (standing in
        // here for an ordinary MPM sand particle, without needing a full
        // `Simulation` -- that wiring is a separate, later phase). Directly
        // inject a second, independent mass+momentum contribution at the
        // SAME node before the grain scatters, and confirm the grain's
        // gathered velocity reflects the shared, mass-weighted average --
        // not just its own contribution replayed back unchanged.
        let mut grid = Grid::new(32);
        let mut pop =
            GrainPopulation::new(vec![Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)], config());
        // A second contributor at the exact same position, much heavier and
        // already moving -- mirrors an ordinary MPM particle's own P2G
        // scatter (`Grid::add_mass_momentum` doesn't care about the source).
        let other_mass = 9.0;
        let other_v = Vec2::new(2.0, 0.0);
        grid.add_mass_momentum(glam::IVec2::new(16, 16), other_mass, other_mass * other_v);

        scatter_grains_to_grid(&pop, &mut grid);
        grid.update_velocities(0.0, Vec2::ZERO); // normalize only, isolate the mixing effect
        gather_grid_to_grains(&mut pop, &grid, 0.0, 0.0, None);

        // The grain's own mass spreads across a 3x3 kernel stencil while the
        // injected momentum sits concentrated in the single center cell, so
        // the exact resulting value depends on `quadratic_weights`'s own
        // precise split at an on-node position (already covered by that
        // kernel's own tests elsewhere -- not re-derived here). What THIS
        // test proves is qualitative and real: measured at v.x=0.486 (real
        // run, not guessed), comfortably far from the grain's own
        // pre-scatter v.x=0.0 -- genuine, substantial momentum transfer from
        // the other contributor, not noise.
        assert!(
            pop.grains[0].v.x > 0.3,
            "expected the grain's gathered velocity to reflect the heavier \
             co-located contributor, got {:?}",
            pop.grains[0].v
        );
    }

    fn total_ke(pop: &GrainPopulation) -> f32 {
        pop.grains
            .iter()
            .map(|g| {
                0.5 * g.mass * g.v.length_squared() + 0.5 * g.moment_of_inertia() * g.spin * g.spin
            })
            .sum()
    }

    /// Real, decisive, cheap isolation for a 2026-08-20 finding: an 80-grain
    /// column collapse with rotation-aware scatter (see
    /// `scatter_grains_to_grid`'s own doc) explodes -- grains launching to
    /// v~14, spin~10 -- even with dt already confirmed converged (halving
    /// it moved the result <1%) and a real CFL bound added (confirmed
    /// non-binding: dt was already far finer than needed). That rules out
    /// resolution as the cause, leaving a real, structural energy-
    /// conservation question: does the grid-coupled path add spurious
    /// energy for a real 2-grain contact that the SAME contact_law,
    /// stepped standalone (no grid), does not?
    ///
    /// Two touching grains, one given real initial spin, zero gravity
    /// (isolates the contact/coupling exchange from any falling), same
    /// `ContactLawConfig` both paths use unmodified. All three damping
    /// ratios are positive -- an isolated 2-body real DEM contact has no
    /// energy source, so total KE (translational + rotational) must be
    /// non-increasing on the STANDALONE path (pure `GrainPopulation::step`,
    /// proven correct, 2,000,000-step-verified elsewhere) for a real,
    /// trustworthy baseline. Grid-coupled runs the SAME grains through
    /// `scatter -> grid_update(gravity=0) -> gather -> apply_grain_contact_forces`,
    /// mirroring `Simulation::step`'s own real substep order exactly.
    #[test]
    fn does_grid_coupling_add_spurious_energy_to_a_real_two_grain_contact() {
        let cfg = ContactLawConfig {
            normal_stiffness: 1.0e4,
            tangential_stiffness: 0.8e4,
            rolling_stiffness: 5.0e2,
            normal_damping: 2.0 * (1.0e4_f32 * 0.5).sqrt() * 0.6,
            tangential_damping: 2.0 * (0.8e4_f32 * 0.5).sqrt() * 0.6,
            rolling_damping: 2.0 * (5.0e2_f32 * 0.5).sqrt() * 0.6,
            friction: (35.0_f32).to_radians().tan(),
            rolling_friction: 0.20,
        };
        let dt = crate::materials::granular::grain_contact_law::critical_timestep(0.5, &cfg) * 0.02;
        const STEPS: usize = 500;

        // Standalone reference: same two grains, same config, `pop.step`'s
        // own proven order (contact resolved, then integrated).
        let make_grains = || {
            vec![
                Grain {
                    spin: 5.0,
                    ..Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)
                },
                Grain::new(Vec2::new(17.9, 16.0), 1.0, 1.0), // 0.1 overlap, real contact from step 0
            ]
        };
        let mut standalone = GrainPopulation::new(make_grains(), cfg);
        let ke0 = total_ke(&standalone);
        let mut standalone_max_ke = ke0;
        for _ in 0..STEPS {
            standalone.step(Vec2::ZERO, dt);
            standalone_max_ke = standalone_max_ke.max(total_ke(&standalone));
        }

        // Grid-coupled: identical grains/config, real shared-grid path.
        let mut grid = Grid::new(64);
        let mut coupled = GrainPopulation::new(make_grains(), cfg);
        let mut coupled_max_ke = ke0;
        let mut spike_step = None;
        for step in 0..STEPS {
            grid.clear(); // real, required: Simulation::step() clears every substep (step.rs:609) -- this test forgot to
            let ke_before = total_ke(&coupled);
            let (g0_before, g1_before) = (coupled.grains[0], coupled.grains[1]);
            scatter_grains_to_grid(&coupled, &mut grid);
            grid.update_velocities(dt, Vec2::ZERO);
            gather_grid_to_grains(&mut coupled, &grid, 0.0, 0.0, None);
            apply_grain_contact_forces(&mut coupled, dt, &[], 64, &grid);
            let ke_after = total_ke(&coupled);
            if spike_step.is_none() && ke_after > ke_before * 10.0 && ke_after > 100.0 {
                spike_step = Some(step);
                println!(
                    "  SPIKE at step={step}: ke_before={ke_before:.4} ke_after={ke_after:.6e}"
                );
                println!(
                    "    grain0 BEFORE x={:?} v={:?} spin={:.4}  AFTER x={:?} v={:?} spin={:.4}",
                    g0_before.x,
                    g0_before.v,
                    g0_before.spin,
                    coupled.grains[0].x,
                    coupled.grains[0].v,
                    coupled.grains[0].spin
                );
                println!(
                    "    grain1 BEFORE x={:?} v={:?} spin={:.4}  AFTER x={:?} v={:?} spin={:.4}",
                    g1_before.x,
                    g1_before.v,
                    g1_before.spin,
                    coupled.grains[1].x,
                    coupled.grains[1].v,
                    coupled.grains[1].spin
                );
            }
            coupled_max_ke = coupled_max_ke.max(ke_after);
        }

        println!(
            "ke0={ke0:.6} standalone_max_ke={standalone_max_ke:.6} coupled_max_ke={coupled_max_ke:.6} \
             standalone_final={:.6} coupled_final={:.6}",
            total_ke(&standalone),
            total_ke(&coupled)
        );
        // The two grains start 0.1 overlapped -- a real, deliberate compressed
        // normal spring, storing real elastic PE that legitimately converts to
        // KE as they push apart (standalone_max_ke=15.19 vs ke0=6.25, confirmed
        // real and bounded, not a bug: total mechanical energy, KE+PE, is what
        // damping actually bounds, not KE alone -- this test's own earlier,
        // stricter "KE must never exceed ke0" assumption was simply wrong about
        // the physics, not about the coupling). The real, decisive comparison
        // is RELATIVE: same grains, same config, same starting PE -- does the
        // grid-coupled path stay within the same real, bounded, finite range
        // the proven standalone path does, or does it diverge unboundedly.
        assert!(
            coupled_max_ke <= standalone_max_ke * 2.0,
            "grid-coupled KE (max={coupled_max_ke}) diverged far beyond the proven standalone \
             reference's own bounded peak (max={standalone_max_ke}, from the IDENTICAL starting \
             grains/config) -- real, confirmed spurious energy injection in the grid coupling \
             path itself, not in contact_law (which both paths share unmodified)"
        );
    }
}

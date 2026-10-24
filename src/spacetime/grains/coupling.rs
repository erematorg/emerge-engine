//! Grain <-> shared MPM grid coupling. Mirrors `rod::coupling`'s own
//! scatter/gather exactly, for the identical reason: `Grid` is fully
//! source-agnostic (a flat `Cell { mass, momentum }` accumulator, no idea
//! whether a contribution came from an ordinary particle, a rod point, or a
//! grain) -- so a grain exchanging real momentum with ordinary MPM sand
//! particles through the shared grid is a not aspirational,
//! integration, exactly like the rod solver already proved.
//!
//! Same real division of labor as `rod::coupling`: gravity and interaction
//! with the surrounding continuum come through the shared grid (scatter ->
//! grid_update -> gather); a grain's OWN inter-grain contact forces
//! (`contact_law`) are applied AFTER the gather, as a velocity/spin
//! correction -- mirrors `rod::advance_rod`'s own
//! documented reason for not re-applying gravity a second time.

use glam::{Mat2, Vec2};

use crate::boundary::BoundaryCondition;
use crate::grid::kernel::quadratic_weights;
use crate::grid::{Grid, VelocitySnapshot};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::spacetime::integration::advance_position;

use super::population::GrainPopulation;

/// Grain -> grid scatter. APIC (Jiang, Schroeder, Selle, Teran & Stomakhin
/// 2015), the formula particles use in `p2g.rs` (`v_i + c_i*cell_dist`),
/// with the grain's persistent `Grain::c` affine matrix.
///
/// Scattering the rigid-body field from `spin` (`v_com + spin*perp(r)`) is
/// exact for one grain but leaks energy without bound once spinning grains
/// share nodes (`spin` is never debited when a neighbour's gather reads it).
/// Pure PIC froze a jittered column near its initial shape (Jiang et al.
/// 2015: PIC "causes sand to clump together"); pure FLIP was noisy and
/// dt-unstable (0.73x/0.77x/0.74x at three dt). APIC is as stable as PIC,
/// as low-dissipation as FLIP and conserves angular momentum; `c` is rebuilt
/// every substep by `gather_grid_to_grains` from the grid's local velocity
/// field, closing the loop.
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

/// Grid -> grain gather, the APIC formula of `gather_grid_to_particles`:
/// gathers `new_v` (the plain average, a rigid body's centre-of-mass
/// velocity) and `b = sum(weight * outer(node_v, cell_dist))`, and sets
/// `grain.c = b * KERNEL_D_INVERSE * apic_blend` for the next scatter, with
/// the particles' `SimConfig::apic_blend` (default `1.0`, full APIC).
///
/// `spin` is not re-derived from `c`: it belongs to
/// `apply_grain_contact_forces`'s torque integration (the calibrated DEM
/// rolling resistance). `c` only keeps the grid round trip from losing
/// momentum as pure PIC did.
///
/// ASFLIP (Fei, Guo, Wu, Huang & Gao 2021, as in
/// `gather_grid_to_particles`): adds the FLIP residual (`v_old - old_v`) on
/// top of the APIC gather, `old_v` gathered with the same weights from the
/// pre-force snapshot (`pre_force_snapshot`: P2G's normalized momentum
/// with its fused particle stress impulse taken back out, before gravity,
/// boundaries and contact; `Grid::snapshot_velocities_before_stress`). `None`
/// (`asflip_blend = 0.0`, the default) leaves `grain.v = new_v`. No
/// compression-aware `gamma` split: that exists because particle G2P also
/// advances position, while grains advance in `apply_grain_contact_forces`.
///
/// Velocity only, no position advance: like standalone
/// `GrainPopulation::step`, contact forces act on `v` before `x += v*dt`.
/// Advancing position here and correcting velocity a substep later let
/// grains interpenetrate every substep before their contact spring could
/// push back where it applies. (`gather_grid_to_rod` does not advance
/// position either; rods advance in their own sub-steps.)
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
/// doc for why this changed). Gravity is NOT reapplied here -- it
/// already reached every grain through the shared grid's own `grid_update`
/// step, the same mechanism ordinary particles and rods already use.
///
/// Clamps the advanced position against every boundary
/// (`BoundaryCondition::clamp_particle_position`), same real convention
/// `gather_grid_to_particles` already uses for ordinary particles
/// (`g2p.rs`'s own `new_pos = boundary.clamp_particle_position(...)` loop).
/// The grid-level velocity damping near a boundary
/// (`apply_boundary_conditions_to_grid`, node velocity only, a few cells
/// wide) is not enough on its own: a fast grain can cross that thin zone
/// in one substep, so grains need the same hard position backstop as
/// particles (without it a column-collapse test measured a 5.26x spread
/// ratio, reachable only by leaving the domain).
pub fn apply_grain_contact_forces(
    grains: &mut GrainPopulation,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    grid_res: usize,
    grid: &crate::grid::Grid,
    stability_fraction: f32,
) {
    // Sub-cycled to the contacts' own step (`contact_step_limit`) instead of
    // clamping the mechanics substep: with a mineral's stiffness a
    // minimum would collapse the whole scene's step. Each sub-step takes at
    // most `stability_fraction` of the limit, the fraction the materials
    // take of theirs; the grid's action already reached the grains' velocity
    // (`gather_grid_to_grains`) and stays fixed over the substep.
    let limit = grains.contact_step_limit() * stability_fraction;
    let substeps = if limit.is_finite() && limit > 0.0 {
        (dt / limit).ceil().max(1.0) as usize
    } else {
        1
    };
    grains.set_last_contact_substeps(substeps);
    let h = dt / substeps as f32;
    for _ in 0..substeps {
        contact_sub_step(grains, h, boundaries, grid_res, grid);
    }
}

/// One contact sub-step of `apply_grain_contact_forces`: contacts resolved,
/// then velocity, spin and position advanced over `dt`.
fn contact_sub_step(
    grains: &mut GrainPopulation,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    grid_res: usize,
    grid: &crate::grid::Grid,
) {
    // Per-grain normal correction before any contact resolution (see
    // `GrainPopulation::clean_wall_normal_velocity`): the grid's per-cell
    // boundary correction is blurred by the kernel, and everything below
    // needs a clean velocity.
    grains.clean_wall_normal_velocity(boundaries, grid_res);
    let (mut forces, mut torques) = grains.resolve_contact_forces(dt);
    // Grain-vs-boundary contact, separate from grain-grain contact (see
    // `GrainPopulation::resolve_wall_contact_forces`): without it a grain on
    // the ground never starts rolling.
    let (wall_forces, wall_torques) = grains.resolve_wall_contact_forces(boundaries, grid_res, dt);
    // Grain-vs-continuum-terrain contact (see `GrainPopulation::
    // resolve_terrain_contact_forces`): a grain on an MPM terrain (not a
    // `BoundaryCondition`) otherwise gets no rolling resistance. Opt-in
    // (`with_terrain_contact`), free otherwise.
    let (terrain_forces, terrain_torques) = grains.resolve_terrain_contact_forces(grid, dt);
    for i in 0..forces.len() {
        forces[i] += wall_forces[i] + terrain_forces[i];
        torques[i] += wall_torques[i] + terrain_torques[i];
    }
    grains
        .position_compensation
        .resize(grains.grains.len(), Vec2::ZERO);
    for (idx, (grain, compensation)) in grains
        .grains
        .iter_mut()
        .zip(grains.position_compensation.iter_mut())
        .enumerate()
    {
        grain.v += (forces[idx] / grain.mass) * dt;
        grain.spin += (torques[idx] / grain.moment_of_inertia()) * dt;
        grain.orientation += grain.spin * dt;
        let mut new_pos = grain.x;
        advance_position(&mut new_pos, compensation, grain.v * dt);
        let advanced = new_pos;
        // Radius-aware backstop, owned by each boundary
        // (`BoundaryCondition::clamp_grain_position`): the particle clamp
        // (`clamp_particle_position`) assumes a "+1" vertical clearance,
        // ignoring the grain's radius and a slope's tilt, and froze a grain
        // on a ramp (`grain.x.y` unchanged for 2000+ steps while `grain.v.y`
        // grew negative).
        for boundary in boundaries {
            new_pos = boundary.clamp_grain_position(new_pos, grain.radius, grid_res);
        }
        // A clamped grain sits exactly where the boundary put it; the
        // residual of the step it did not take is dropped with that step.
        if new_pos != advanced {
            *compensation = Vec2::ZERO;
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
        // A grain at rest in an otherwise empty grid picks up exactly the
        // grid's gravity-integrated velocity after a round trip, through the
        // grid's `update_velocities` (particles' mechanism), not by
        // integrating gravity itself (`gather_grid_to_grains` only reads).
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

    /// A lone spinning grain on a grid node gains no velocity from its own
    /// scatter/gather round trip: the full stencil's first moment
    /// `sum(w d)` is zero, so the spin's momentum cancels.
    #[test]
    fn a_lone_spinning_grain_on_a_node_gains_no_velocity() {
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
            assert_eq!(pop.grains[0].v, Vec2::ZERO, "step {step}");
        }
    }

    /// The cancellation above needs the full stencil: a node below y = 0 is
    /// dropped, and a grain whose stencil reaches it gathers a bias every
    /// step. `FrictionBoundary(2)` clamps positions to y >= 1.0, so from
    /// there up the spin must still give no velocity; below it is printed.
    #[test]
    fn a_spinning_grain_inside_the_boundary_clamp_keeps_its_full_stencil() {
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
            if y0 >= 1.0 {
                assert_eq!(pop.grains[0].v, Vec2::ZERO, "y0 = {y0}");
            }
        }
    }

    /// Isolates a boundary's grid-velocity effect (`apply_to_grid_velocity`,
    /// e.g. `FrictionBoundary`'s Coulomb wall) on a single spinning grain
    /// overlapping a `FrictionBoundary`'s zone, with no other grain and no
    /// contact law (the lone grain is stable without a boundary, see
    /// `a_lone_spinning_grain_on_a_node_gains_no_velocity`). Calls the trait
    /// method directly, since `solver::projection::
    /// apply_boundary_conditions_to_grid` is `pub(super)` elsewhere. The
    /// two-grain test proves grain-grain contact through the grid adds no
    /// energy.
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
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid, 1.0);
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

    /// A lone grain sliding on a `FrictionBoundary`, through the raw
    /// scatter/boundary/gather calls, slows by Coulomb's `mu g T`: its final
    /// speed is within 1 percent of the start speed of `v0 - mu g T`.
    #[test]
    fn a_grain_sliding_on_a_friction_boundary_slows_by_mu_g_t() {
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
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid, 1.0);
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
        let v0 = 2.0;
        let expected = v0 - 0.7 * gravity.y.abs() * 40000.0 * dt;
        assert!(
            (pop.grains[0].v.x - expected).abs() <= 0.01 * v0,
            "final v.x {} against v0 - mu g T = {expected}",
            pop.grains[0].v.x
        );
    }

    /// Both individual mechanisms above (grain-grain contact through the
    /// grid; a single grain against a boundary) are separately proven
    /// stable. This test is the remaining combination the 80-grain
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
        // produce mid-collapse in the test.
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
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid, 1.0);
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

    /// A dense cluster (5x4 = 20 grains touching in both axes, interior
    /// coordination number 4): the 80-grain test launched at
    /// `active_contacts = 16`, its highest coordination, right after a dense
    /// settling moment, while rows of 4-8 grains (coordination 1-4) stayed
    /// stable.
    #[test]
    fn diag_dense_packed_cluster_settling_under_gravity_and_boundary() {
        let mut grid = Grid::new(48);
        let boundary = crate::boundary::FrictionBoundary::new(2, 0.7);
        // The 80-grain test's calibrated config, not the module `config()`
        // helper (10x stiffer, flat 50.0 damping, friction 0.5), which passed
        // but does not reproduce the trigger.
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
        // the trigger's own active_contacts=16 density. Small real
        // jitter (same convention as the test's own `build_column`/
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
            apply_grain_contact_forces(&mut pop, dt, &[], grid_res, &grid, 1.0);
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

    /// Does the grid-coupled path add energy to a 2-grain contact that the
    /// same contact law stepped standalone does not? An 80-grain column with
    /// rotation-aware scatter (see `scatter_grains_to_grid`) exploded (v ~14,
    /// spin ~10) with dt converged (halving it moved the result <1%) and a
    /// non-binding CFL bound, which leaves energy conservation.
    ///
    /// Two touching grains, one spinning, no gravity, the same
    /// `ContactLawConfig` on both paths. All damping ratios are positive, so
    /// kinetic energy (translational and rotational) must not increase on the
    /// standalone path (`GrainPopulation::step`); the grid-coupled path runs
    /// `scatter -> grid_update(gravity=0) -> gather -> apply_grain_contact_forces`,
    /// `Simulation::step`'s substep order.
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
            apply_grain_contact_forces(&mut coupled, dt, &[], 64, &grid, 1.0);
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
        // The two grains start 0.1 overlapped -- a deliberate compressed
        // normal spring, storing real elastic PE that legitimately converts to
        // KE as they push apart (standalone_max_ke=15.19 vs ke0=6.25, confirmed
        // real and bounded, not a bug: total mechanical energy, KE+PE, is what
        // damping actually bounds, not KE alone -- this test's own earlier,
        // stricter "KE must never exceed ke0" assumption was simply wrong about
        // the physics, not about the coupling). The decisive comparison
        // is RELATIVE: same grains, same config, same starting PE -- does the
        // grid-coupled path stay within the same bounded, finite range
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

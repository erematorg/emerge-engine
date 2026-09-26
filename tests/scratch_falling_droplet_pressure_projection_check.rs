//! Phase 0 gate (see plan: real-time fluid via pressure projection) for
//! whether `Grid::project_fluid_incompressibility`
//! (`src/spacetime/grid/pressure.rs`) can handle a fully AIRBORNE water
//! body -- free-surface Dirichlet boundaries on every side at once, no
//! wall touching anywhere.
//!
//! Real, disclosed context: this exact solver has a known, unresolved
//! "wall-free-pool instability" (`pressure.rs`'s own doc: a RESTING pool
//! with free-surface on its entire perimeter spikes to max_speed 100-250
//! on the very first step, reproducible with zero seed velocity). The
//! only diagnostic that ever reproduced it (`git show
//! 9d7fea3^:examples/cpu/diag_vortex_projection_headless.rs`) tested a
//! pool RESTING ON THE FLOOR -- one real Neumann wall edge to anchor
//! against. A falling droplet before it touches anything has NO wall edge
//! at all, a structurally different, never-tested case, and it is exactly
//! the regime `DropletImpact`'s first frames live in.
//!
//! **GATE FAILED (2026-09-17), confirmed, not a guess.** With
//! `fluid_pressure_iterations: 0` (the projection fully disabled, a pure
//! control), the blob free-falls exactly as physics predicts: `J` stays
//! at exactly 1.0000, `max_speed` matches `g*t` to 3 decimal places. The
//! instant the projection is turned on -- at ANY iteration count tried
//! (1, 2, 4, 8) -- `J` hits this engine's hard safety clamp `[0.5, 2.0]`
//! on the very first step, every time, regardless of iteration count.
//! This rules out "not enough solver passes" as the cause (1 pass fails
//! exactly as badly as 8) and confirms the wall-free-pool bug (or a
//! closely related variant of it) reproduces on a genuinely falling,
//! isolated droplet, not just a resting pool.
//!
//! **Real fix attempted the same day, also failed, reverted.** Root-caused
//! the divergence RHS computation (`Grid::project_fluid_incompressibility`,
//! `src/spacetime/grid/pressure.rs`) to a hard-zero `velocity_at` fallback
//! at untouched neighbor cells, and ported the exact fix already proven
//! for the same class of bug in the ordinary G2P gather stencil
//! (`velocity_at_or_extrapolated`, `grid/mod.rs`, 2026-08-13). Measured,
//! not assumed: it did NOT fix this gate (`J` still hit the clamp on
//! step 1) and, in an EARLIER, methodologically-flawed regression check using
//! the wrong (full, undreated) gravity, appeared to also break the
//! validated wall-contact scene -- that specific claim was later corrected
//! (`fluid_pressure_projection.rs`'s own GUI actually validates at
//! `gravity_fraction: 0.003`, ~333x gentler; re-tested at the CORRECT
//! gravity, the original unmodified code is genuinely stable there). The
//! fix itself was still reverted regardless, since it never solved this
//! gate's own real problem. See `falling_droplet_at_validated_derated_gravity`
//! below: even at that same correct, gentle gravity, `J` still hits the
//! clamp, just later (~step 6-7 instead of step 1) -- the instability is
//! real at any gravity magnitude tested, not an artifact of testing too
//! violent a fall.
//!
//! Per the plan's own pre-committed stop condition: Phase 1/2/3 do not
//! start. This test is kept (not deleted) as the permanent, reproducible
//! record of this finding, matching this project's standing practice for
//! a proven, disclosed negative result.
//!
//! Config mirrors `examples/cpu/fluid_pressure_projection.rs`'s own
//! already-proven pattern exactly (same solver settings that produced a
//! real, verified ~30fps on this engine's hardest WALL-CONTACT scene) --
//! the only variable changed is the scene geometry itself (isolated
//! falling blob instead of a wall-touching column).

extern crate emerge_engine as emerge;

use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;

#[test]
#[ignore = "real, disclosed negative result -- pressure projection's wall-free-pool \
            instability reproduces on a falling droplet with zero iteration-count \
            dependence (confirmed 1/2/4/8), not tuned to pass, see this file's own \
            top doc comment"]
fn falling_droplet_with_no_wall_contact_stays_physically_bounded() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Real, full earth gravity from `SimConfig::earth` (9.81 m/s² / cell_m)
    // -- matches the ORIGINAL wall-free-pool bug's own reproduction
    // (`diag_vortex_projection_headless.rs`, also full undreated gravity).
    // CORRECTION (2026-09-17): `fluid_pressure_projection.rs`'s own real
    // validated ~30fps run does NOT use this -- its GUI starts at
    // `gravity_fraction: 0.003`, i.e. ~333x gentler. See the sibling test
    // below for that comparison at the correctly-matched gravity.
    let g_grid = config.gravity.y.abs();

    // `DropletImpact`'s own blob box (`basic_fluids_gpu.rs`), placed at the
    // grid center with generous clearance from every boundary -- no pool
    // underneath, nothing else in the scene at all.
    const BLOB_BOX: IVec2 = IVec2::new(7, 7);
    let blob_center = Vec2::new(32.0, 32.0);

    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    const SPACING: f32 = 0.6;
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: BLOB_BOX,
        box_center: blob_center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    // Half the blob's own box height, plus the boundary thickness, plus one
    // real margin cell -- stop comfortably before the blob's lowest edge
    // could plausibly reach the floor or `boundary_thickness`, so a
    // failure can only come from free-surface geometry, never real impact.
    let lowest_safe_y = config.boundary_thickness as f32 + (BLOB_BOX.y as f32) * SPACING + 2.0;

    let mut sim_time = 0.0f32;
    let mut step = 0usize;
    loop {
        solver.step();
        sim_time += DT;
        step += 1;

        let snap = solver.diagnostics_snapshot();
        if step <= 5 {
            println!(
                "step {step} (t={sim_time:.2}s): max_speed={:.3} J=[{:.4},{:.4}] non_finite={} oob={}",
                snap.max_particle_speed,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.non_finite_particle_values,
                snap.out_of_bounds_particles,
            );
        }
        assert_eq!(
            snap.non_finite_particle_values, 0,
            "step {step} (t={sim_time:.2}s): non-finite particle value -- \
             the wall-free-pool instability (or a variant of it) reproduces \
             on a falling droplet with no wall contact at all"
        );
        assert_eq!(
            snap.out_of_bounds_particles, 0,
            "step {step} (t={sim_time:.2}s): particle left the grid"
        );

        // Physically-expected free-fall speed after this much real time,
        // ignoring drag/viscosity (this material's own viscosity is 0.0,
        // matching the proven config) -- v = g*t. A 3x margin separates
        // ordinary numerical noise from the known bug's own signature (a
        // spike "well beyond anything else seen," 100-250 vs an expected
        // free-fall speed on the order of tens of cells/s over this short
        // a fall).
        let expected_free_fall_speed = g_grid * sim_time;
        let speed_ceiling = (3.0 * expected_free_fall_speed).max(3.0);
        assert!(
            snap.max_particle_speed <= speed_ceiling,
            "step {step} (t={sim_time:.2}s): max_particle_speed={:.3} exceeds \
             3x the physically-expected free-fall speed ({expected_free_fall_speed:.3}) -- \
             this is the known wall-free-pool instability's own signature \
             (a spike far beyond anything gravity alone explains), now \
             reproduced on a falling droplet with zero wall contact",
            snap.max_particle_speed
        );

        assert!(
            (0.8..=1.25).contains(&snap.min_deformation_j)
                && (0.8..=1.25).contains(&snap.max_deformation_j),
            "step {step} (t={sim_time:.2}s): J=[{:.4},{:.4}] outside [0.8,1.25] -- \
             an untouched, uncompressed airborne droplet has no physical \
             reason for a large volume-ratio excursion",
            snap.min_deformation_j,
            snap.max_deformation_j
        );

        let particles = solver.particles();
        let lowest_y = (0..particles.x.len())
            .map(|i| particles.x[i].y)
            .fold(f32::MAX, f32::min);
        if lowest_y <= lowest_safe_y {
            println!(
                "stopped at step {step} (t={sim_time:.2}s), lowest_y={lowest_y:.2} \
                 approaching the floor -- gate passed with zero contact"
            );
            break;
        }
        assert!(
            step < 10_000,
            "droplet never approached the floor after {step} steps -- scene setup is wrong"
        );
    }
}

/// Same falling-droplet gate, but at `fluid_pressure_projection.rs`'s own
/// REAL validated gravity level (`gravity_fraction: 0.003`, ~333x gentler
/// than full earth gravity) instead of the original bug report's full
/// gravity -- does the wall-free-pool instability's severity scale down
/// with a much gentler, more game-realistic acceleration, or is it
/// independent of gravity magnitude entirely?
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn falling_droplet_at_validated_derated_gravity() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    const BLOB_BOX: IVec2 = IVec2::new(7, 7);
    let blob_center = Vec2::new(32.0, 32.0);

    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    const SPACING: f32 = 0.6;
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: BLOB_BOX,
        box_center: blob_center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    for step in 1..=60u32 {
        solver.step();
        let snap = solver.diagnostics_snapshot();
        println!(
            "step {step:2}: max_speed={:.4} J=[{:.4},{:.4}] non_finite={} oob={}",
            snap.max_particle_speed,
            snap.min_deformation_j,
            snap.max_deformation_j,
            snap.non_finite_particle_values,
            snap.out_of_bounds_particles,
        );
    }
}

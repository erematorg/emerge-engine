/// Headless simulation — no rendering, no Bevy, no feature flags.
///
/// Demonstrates the full emerge LP integration API:
///   SolverConfig · SpawnConfig · material registration · step_n()
///   phase rules · particles_near · apply_impulse · material_state · diagnostics
///
///   cargo run --example headless
use emerge::diagnostics::{FrameLogger, log_frame_full, per_material_stats};
use emerge::{
    MpmSolver, NeoHookeanMaterial, SandMaterial, SlipBoundary, SolverConfig, SpawnConfig,
};
use glam::{IVec2, Vec2};

const JELLY_ID: u32 = 0;
const SAND_ID: u32 = 1;
const LABELS: &[(u32, &str)] = &[(JELLY_ID, "jelly"), (SAND_ID, "sand")];

fn main() {
    let config = SolverConfig::standard(64, 0.05, Vec2::new(0.0, -0.3));

    let jelly_spawn = SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::new(16, 16),
        box_center: Vec2::new(24.0, 48.0),
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };
    let sand_spawn = SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::new(24, 12),
        box_center: Vec2::new(40.0, 40.0),
        precompute_initial_volumes: true,
        material_id: SAND_ID,
        ..SpawnConfig::for_solver(&config)
    };

    // Phase rule: jelly particles moving fast under compression become sand-like.
    // LP uses this pattern for matter-state transitions (water→ice, rock→gravel).
    let mut solver = MpmSolver::new(config, jelly_spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(400.0, 200.0)))
        .with_material(SAND_ID, Box::new(SandMaterial::new(1000.0, 500.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            // Example: high-speed jelly that has been compressed past 80% volume
            // transitions to sand — models fracture or granularization.
            if p.material_id == JELLY_ID
                && p.deformation_gradient.determinant() < 0.8
                && p.v.length() > 8.0
            {
                Some(SAND_ID)
            } else {
                None
            }
        });

    let _ = solver.spawn_group(sand_spawn);

    println!(
        "Spawned {} particles ({} jelly, {} sand)\n",
        solver.particles().len(),
        solver
            .particles()
            .iter()
            .filter(|p| p.material_id == JELLY_ID)
            .count(),
        solver
            .particles()
            .iter()
            .filter(|p| p.material_id == SAND_ID)
            .count(),
    );

    let mut logger = FrameLogger::open("headless_run.ndjson").expect("failed to open log file");

    // Step 100 frames, log every 60.
    for step in 1..=100u64 {
        solver.step_n(1);
        let snap = solver.diagnostics_snapshot();
        let stats = per_material_stats(solver.particles());
        log_frame_full(step, config.dt, solver.particles(), LABELS, &snap, 60);
        logger.log(step, config.dt, &stats, &snap, LABELS);
    }

    // LP sensor demo: count how many sand particles are near the jelly centroid.
    // Used by LP creatures to sense local material composition.
    let jelly_state = solver.material_state(JELLY_ID);
    let center = jelly_state.centroid;
    let sand_nearby = solver.count_near(center, 8.0, SAND_ID);
    println!(
        "\nSand particles within r=8 of jelly centroid ({:.1?}): {}",
        center, sand_nearby
    );

    // LP impulse demo: apply a radial push at the jelly centroid (models creature locomotion).
    solver.apply_radial_impulse(center, 6.0, 5.0);
    solver.step_n(1);

    // Final physics summary.
    let jelly = solver.material_state(JELLY_ID);
    let sand = solver.material_state(SAND_ID);
    let snap = solver.diagnostics_snapshot();

    println!("\n── Final physics summary ──");
    println!("  total_particle_mass  : {:.6}", snap.total_particle_mass);
    println!(
        "  mass_error           : {:.2e}  (P2G conservation)",
        snap.relative_mass_error
    );
    println!(
        "  momentum_error       : {:.2e}  (P2G conservation)",
        snap.relative_momentum_error
    );
    println!(
        "  global_J_range       : [{:.4}, {:.4}]",
        snap.min_deformation_j, snap.max_deformation_j
    );
    println!(
        "  cfl_number           : {:.4}  (< 1.0 = stable)",
        snap.cfl_number
    );
    println!("  substeps_last_step   : {}", snap.substeps_last_step);
    println!(
        "  non_finite_particles : {}",
        snap.non_finite_particle_values
    );
    println!(
        "  jelly  centroid={:.2?}  avg_speed={:.4}  avg_J={:.4}",
        jelly.centroid, jelly.avg_speed, jelly.avg_det_f
    );
    println!(
        "  sand   centroid={:.2?}  avg_speed={:.4}  avg_Jp={:.4}",
        sand.centroid, sand.avg_speed, sand.avg_volume_ratio
    );
}

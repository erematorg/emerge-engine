extern crate emerge_engine as emerge;

/// Headless simulation -- no rendering, no Bevy, no feature flags.
///
/// Demonstrates the full emerge LP integration API:
///   SimConfig / SpawnRegion / material registration / step_n()
///   phase rules / particles_near / apply_impulse / material_state / diagnostics
///
///   cargo run --example headless
use emerge::diagnostics::{FrameLogger, StepTiming, log_frame_full, per_material_stats};
use emerge::{
    Elastic, Elastoplastic, PlasticityModel, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const JELLY_ID: u32 = 0;
const SAND_ID: u32 = 1;
const LABELS: &[(u32, &str)] = &[(JELLY_ID, "jelly"), (SAND_ID, "sand")];

fn all_accounted(t: &StepTiming) -> u64 {
    t.p2g_us
        + t.grid_update_us
        + t.g2p_us
        + t.cfl_us
        + t.fields_us
        + t.thermal_us
        + t.spatial_hash_us
        + t.phase_sleep_us
        + t.project_us
        + t.density_us
}

fn print_timing(t: &StepTiming, step: u64, substeps: usize) {
    let other = t.total_us.saturating_sub(all_accounted(t));
    println!(
        "  timing step={step:3}  total={:.2}ms  \
         p2g={:.2}  grid={:.2}  g2p={:.2}  cfl={:.2}  \
         hash={:.2}  phase={:.2}  project={:.2}  density={:.2}  other={:.2}  sub={substeps}",
        t.total_us as f64 / 1000.0,
        t.p2g_us as f64 / 1000.0,
        t.grid_update_us as f64 / 1000.0,
        t.g2p_us as f64 / 1000.0,
        t.cfl_us as f64 / 1000.0,
        t.spatial_hash_us as f64 / 1000.0,
        t.phase_sleep_us as f64 / 1000.0,
        t.project_us as f64 / 1000.0,
        t.density_us as f64 / 1000.0,
        other as f64 / 1000.0,
    );
}

fn print_timing_full(t: &StepTiming, substeps: usize) {
    let total = t.total_us.max(1) as f64;
    let pct = |us: u64| us as f64 / total * 100.0;
    let ms = |us: u64| us as f64 / 1000.0;
    let other = t.total_us.saturating_sub(all_accounted(t));
    println!("  total        : {:.3} ms", total / 1000.0);
    println!(
        "  p2g          : {:.3} ms  ({:.1}%)",
        ms(t.p2g_us),
        pct(t.p2g_us)
    );
    println!(
        "  grid_update  : {:.3} ms  ({:.1}%)",
        ms(t.grid_update_us),
        pct(t.grid_update_us)
    );
    println!(
        "  g2p          : {:.3} ms  ({:.1}%)",
        ms(t.g2p_us),
        pct(t.g2p_us)
    );
    println!(
        "  cfl_select   : {:.3} ms  ({:.1}%)",
        ms(t.cfl_us),
        pct(t.cfl_us)
    );
    println!(
        "  spatial_hash : {:.3} ms  ({:.1}%)",
        ms(t.spatial_hash_us),
        pct(t.spatial_hash_us)
    );
    println!(
        "  phase_sleep  : {:.3} ms  ({:.1}%)",
        ms(t.phase_sleep_us),
        pct(t.phase_sleep_us)
    );
    println!(
        "  project      : {:.3} ms  ({:.1}%)",
        ms(t.project_us),
        pct(t.project_us)
    );
    println!(
        "  density      : {:.3} ms  ({:.1}%)",
        ms(t.density_us),
        pct(t.density_us)
    );
    println!(
        "  fields       : {:.3} ms  ({:.1}%)",
        ms(t.fields_us),
        pct(t.fields_us)
    );
    println!(
        "  thermal      : {:.3} ms  ({:.1}%)",
        ms(t.thermal_us),
        pct(t.thermal_us)
    );
    println!(
        "  other        : {:.3} ms  ({:.1}%)",
        other as f64 / 1000.0,
        other as f64 / total * 100.0
    );
    println!("  substeps     : {substeps}");
}

fn main() {
    // 64-cell grid, 1 cm/cell, 50 ms/step -- earth gravity auto-derived from dx_meters.
    let config = SimConfig::earth(64, 0.01, 0.05);

    let jelly_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(16, 16),
        box_center: Vec2::new(24.0, 48.0),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let sand_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 12),
        box_center: Vec2::new(40.0, 40.0),
        precompute_initial_volumes: true,
        material_id: SAND_ID,
        ..SpawnRegion::for_sim(&config)
    };

    // Property-driven material construction -- no names, just physics.
    // Elastic solid: E=500 Pa, nu=0.45, rho=1000 kg/m3
    let jelly = Elastic {
        e_pa: 500.0,
        nu: 0.45,
        rho_kg_m3: 1000.0,
    }
    .material(&config);
    // Cohesionless granular: E=50 MPa, phi=35 deg, rho=1600 kg/m3
    let sand = Elastoplastic {
        elastic: Elastic {
            e_pa: 50.0e6,
            nu: 0.3,
            rho_kg_m3: 1600.0,
        },
        model: PlasticityModel::Granular {
            friction_angle_deg: 35.0,
            dilatancy_angle_deg: 0.0,
        },
    }
    .material(&config);

    // Phase rule: jelly particles moving fast under compression become sand-like.
    // LP uses this pattern for matter-state transitions (water->ice, rock->gravel).
    let mut solver = Simulation::new(config, jelly_spawn)
        .with_default_material(jelly)
        .with_material(SAND_ID, sand)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            // Example: high-speed jelly that has been compressed past 80% volume
            // transitions to sand -- models fracture or granularization.
            if p.material_id == JELLY_ID
                && p.deformation_gradient.determinant() < 0.8
                && p.v.length() > 8.0
            {
                Some(SAND_ID)
            } else {
                None
            }
        });

    let _ = solver.add_body(sand_spawn);

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

        // Print per-phase timing every 20 steps so you can see where ms are going.
        if step % 20 == 0 {
            print_timing(&snap.timing, step, snap.substeps_last_step);
        }
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

    // Final timing summary — useful for spotting regressions across runs.
    {
        let snap = solver.diagnostics_snapshot();
        println!("\n-- Step timing (last step) --");
        print_timing_full(&snap.timing, snap.substeps_last_step);
    }

    // Final physics summary.
    let jelly = solver.material_state(JELLY_ID);
    let sand = solver.material_state(SAND_ID);
    let snap = solver.diagnostics_snapshot();

    println!("\n-- Final physics summary --");
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

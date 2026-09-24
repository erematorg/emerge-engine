extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real question: `tests/self_gravitating_body.rs` documents a real,
/// disclosed performance wall (SPACING=3.0, ~5236 particles, real
/// elastoplastic self-gravitating dynamics under NBodyGravityField, ran
/// >30 real minutes and was stopped -- never root-caused, just worked
/// > around by coarsening to SPACING=8.0 / ~750 particles). This probe
/// > reuses that exact scene at a few particle counts and prints the
/// > engine's own real per-substep `StepTiming` breakdown (already
/// > instrumented, no external profiler needed) for a small, bounded number
/// > of real steps -- to find WHERE the cost actually concentrates
/// > (force-field/Barnes-Hut cost vs. CFL-forced substep count vs. ordinary
/// > P2G/G2P) before touching any code.
use emerge::fields::NBodyGravityField;
use emerge::{Elastic, Elastoplastic, PlasticityModel, SimConfig, Simulation, SpawnRegion};
use glam::Vec2;

const BENNU_RADIUS_M: f64 = 245.0;
const BENNU_BULK_DENSITY_KG_M3: f64 = 1190.0;
const DX_METERS: f64 = 2.0;
const GRID_RES: usize = 512;
const DT_SECONDS: f64 = 20.0;

fn make_body(spacing: f32) -> Simulation {
    let center = Vec2::splat(GRID_RES as f32 / 2.0);
    let r_grid = (BENNU_RADIUS_M / DX_METERS) as f32;

    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: Vec2::ZERO,
        ..SimConfig::standard(GRID_RES, DT_SECONDS as f32, Vec2::ZERO)
    };

    let regolith = Elastoplastic {
        elastic: Elastic {
            e_pa: 5.0e6,
            nu: 0.25,
            rho_kg_m3: BENNU_BULK_DENSITY_KG_M3 as f32,
        },
        model: PlasticityModel::Granular {
            friction_angle_deg: 35.0,
            dilatancy_angle_deg: 0.0,
        },
    }
    .material(&config);

    let mass_per_particle =
        spacing * spacing * BENNU_BULK_DENSITY_KG_M3 as f32 * (DX_METERS * DX_METERS) as f32;

    let spawn = SpawnRegion {
        spacing,
        box_size: glam::IVec2::splat((2.0 * r_grid) as i32 + 4),
        box_center: center,
        shape: emerge::SpawnShape::Disk { radius: r_grid },
        position_jitter: 0.1,
        mass_override: Some(mass_per_particle),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    let g_grid = (6.674e-11 / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    Simulation::new(config, spawn)
        .with_default_material(regolith)
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, spacing * 0.5, 0.3)))
}

fn profile(spacing: f32, n_steps: usize) {
    let mut sim = make_body(spacing);
    let n = sim.particles().len();
    println!("\n=== spacing={spacing} -> {n} particles ===");

    let mut totals = emerge::diagnostics::StepTiming::default();
    let mut total_substeps = 0usize;
    let start = std::time::Instant::now();
    for step_i in 0..n_steps {
        sim.step();
        let t = sim.diagnostics_snapshot().timing;
        totals.p2g_us += t.p2g_us;
        totals.grid_update_us += t.grid_update_us;
        totals.pressure_us += t.pressure_us;
        totals.g2p_us += t.g2p_us;
        totals.fields_us += t.fields_us;
        totals.thermal_us += t.thermal_us;
        totals.cfl_us += t.cfl_us;
        totals.spatial_hash_us += t.spatial_hash_us;
        totals.total_us += t.total_us;
        total_substeps += sim.last_substeps();
        if step_i == 0 {
            println!(
                "  step0: substeps={} total_us={}",
                sim.last_substeps(),
                t.total_us
            );
        }
    }
    let wall = start.elapsed();
    println!(
        "  {n_steps} real step() calls, wall={:.2}s, avg_substeps/step={:.1}",
        wall.as_secs_f64(),
        total_substeps as f64 / n_steps as f64
    );
    println!(
        "  totals (us): p2g={} grid_update={} pressure={} g2p={} fields={} thermal={} cfl={} spatial_hash={} | total={}",
        totals.p2g_us,
        totals.grid_update_us,
        totals.pressure_us,
        totals.g2p_us,
        totals.fields_us,
        totals.thermal_us,
        totals.cfl_us,
        totals.spatial_hash_us,
        totals.total_us,
    );
    let pct = |x: u64| 100.0 * x as f64 / totals.total_us.max(1) as f64;
    println!(
        "  pct: p2g={:.1}% grid_update={:.1}% g2p={:.1}% fields={:.1}% cfl={:.1}% spatial_hash={:.1}%",
        pct(totals.p2g_us),
        pct(totals.grid_update_us),
        pct(totals.g2p_us),
        pct(totals.fields_us),
        pct(totals.cfl_us),
        pct(totals.spatial_hash_us),
    );
}

fn main() {
    // Real spacings spanning the documented "fine, ran >30min" (3.0) down
    // to the "coarsened workaround" (8.0) the existing test settled on --
    // plus one further, larger point to see the real scaling trend.
    profile(8.0, 10);
    profile(5.0, 10);
    profile(3.0, 5);
}

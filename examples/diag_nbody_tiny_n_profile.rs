extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real question, raised directly by the user noticing live lag in
/// `basic_solar_system_gui`: that demo has only 9 particles (Sun + 8
/// planets). The rayon-parallelized force-field loop added this session
/// (`step.rs`, `into_par_iter().with_min_len(...)`) was profiled and
/// verified to win at 732-5236 particles (the asteroid self-gravity
/// scene) -- but was NEVER checked at N=9. Parallel dispatch (task
/// spawn/steal, thread wake) has a real fixed per-call cost that can
/// exceed the actual work at tiny N. This probe reuses the exact solar
/// system scene (same constants) and profiles real `fields_us` with the
/// DEFAULT thread pool vs. `RAYON_NUM_THREADS=1` (an honest way to see
/// serial-equivalent cost without writing a second code path) to confirm
/// or rule out parallel-dispatch overhead as the real cause of the lag
/// before touching any code.
use emerge::fields::NBodyGravityField;
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};

const DX_METERS: f64 = 2.0e9;
const GRID: usize = 4096;
const DT_SECONDS: f64 = 3600.0;
const G_SI: f64 = 6.674e-11;
const SUN_MASS_KG: f64 = 1.9885e30;

const PLANETS: [(&str, f64, f64); 8] = [
    ("Mercury", 0.330e24, 57.9e9),
    ("Venus", 4.87e24, 108.2e9),
    ("Earth", 5.97e24, 149.6e9),
    ("Mars", 0.642e24, 228.0e9),
    ("Jupiter", 1898.0e24, 778.5e9),
    ("Saturn", 568.0e24, 1432.0e9),
    ("Uranus", 86.8e24, 2867.0e9),
    ("Neptune", 102.0e24, 4515.0e9),
];

fn make_sim() -> Simulation {
    let center = glam::Vec2::splat(GRID as f32 / 2.0);
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: glam::Vec2::ZERO,
        ..SimConfig::standard(GRID, DT_SECONDS as f32, glam::Vec2::ZERO)
    };
    let g_grid = (G_SI / DX_METERS.powi(3)) as f32;

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: center,
        position_jitter: 0.0,
        material_id: 0,
        mass_override: Some(SUN_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, 0.05, 0.1)));
    for mat_id in 1..=PLANETS.len() as u32 {
        solver = solver.with_material(mat_id, Box::new(NeoHookeanMaterial::new(1.0, 1.0)));
    }

    let mut planet_momentum = glam::Vec2::ZERO;
    for (idx, &(_, mass_kg, a_m)) in PLANETS.iter().enumerate() {
        let angle = idx as f32 * std::f32::consts::TAU / PLANETS.len() as f32;
        let (s, c) = angle.sin_cos();
        let r_grid = (a_m / DX_METERS) as f32;
        let v_mag = ((G_SI * SUN_MASS_KG / a_m).sqrt() / DX_METERS) as f32;
        let pos = center + glam::Vec2::new(c, s) * r_grid;
        let vel = glam::Vec2::new(-s, c) * v_mag;

        let spawn = SpawnRegion {
            spacing: 1.0,
            box_size: glam::IVec2::new(1, 1),
            box_center: pos,
            position_jitter: 0.0,
            material_id: idx as u32 + 1,
            mass_override: Some(mass_kg as f32),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(spawn);
        solver.particles_mut().v[idx + 1] = vel;
        planet_momentum += mass_kg as f32 * vel;
    }
    solver.particles_mut().v[0] = -planet_momentum / SUN_MASS_KG as f32;
    solver
}

fn main() {
    let threads = rayon::current_num_threads();
    let mut sim = make_sim();
    let n = sim.particles().len();
    println!("=== N={n} particles, rayon threads={threads} ===");

    const N_STEPS: usize = 2000; // matches ~500 real frames at steps_per_frame=4

    let mut totals = emerge::diagnostics::StepTiming::default();
    let start = std::time::Instant::now();
    for _ in 0..N_STEPS {
        sim.step();
        let t = sim.diagnostics_snapshot().timing;
        totals.p2g_us += t.p2g_us;
        totals.grid_update_us += t.grid_update_us;
        totals.g2p_us += t.g2p_us;
        totals.fields_us += t.fields_us;
        totals.cfl_us += t.cfl_us;
        totals.spatial_hash_us += t.spatial_hash_us;
        totals.total_us += t.total_us;
    }
    let wall = start.elapsed();
    println!(
        "  {N_STEPS} step() calls, wall={:.3}s ({:.1} us/step)",
        wall.as_secs_f64(),
        wall.as_secs_f64() * 1e6 / N_STEPS as f64
    );
    println!(
        "  totals (us): p2g={} grid_update={} g2p={} fields={} cfl={} spatial_hash={} | total={}",
        totals.p2g_us,
        totals.grid_update_us,
        totals.g2p_us,
        totals.fields_us,
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
    println!(
        "  avg fields_us/step()={:.2}",
        totals.fields_us as f64 / N_STEPS as f64
    );
}

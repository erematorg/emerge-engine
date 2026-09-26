//! Real, direct measurement of `sand_repose_angle.rs`'s own actual shipped
//! Grains-mode substep count -- NOT a hand-picked/reconstructed config.
//! Every constant/config value here is copied VERBATIM from that example's
//! own real source (`GRID`, `FLOOR`, `GRAIN_RADIUS`, `GRAIN_MASS`,
//! `GRAINS_R0`/`GRAINS_H0`, `grain_contact_config()`'s own real formula, the
//! real terrain material/boundary/spawn) -- this test exists specifically
//! to settle a real discrepancy found this session: an earlier session's
//! own memory claimed ~3504 substeps/rendered-frame for this scene, but a
//! from-scratch, careful re-derivation of `grain_contact_law::
//! critical_timestep` for the SAME nominal stiffness gave ~11-12. Rather
//! than trust either recalled/hand-derived number, this measures the REAL,
//! live `Simulation::last_substeps()` the actual shipped scene produces.

extern crate emerge_engine as emerge;
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use std::time::Instant;

// Verbatim from examples/cpu/sand_repose_angle.rs.
const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const GRAINS_R0: usize = 4;
const GRAINS_H0: usize = 10;
const GRAIN_RADIUS: f32 = 1.0;
const GRAIN_MASS: f32 = 1.0;
const GRAINS_TERRAIN_HALF_WIDTH_CELLS: i32 = 30;
const GRAINS_TERRAIN_HEIGHT_CELLS: i32 = 8;

struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

fn grain_contact_config() -> ContactLawConfig {
    let m_eff = GRAIN_MASS * 0.5;
    const DAMPING_RATIO: f32 = 0.6;
    let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
    let normal_stiffness = 1.0e4;
    let tangential_stiffness = 0.8e4;
    let rolling_stiffness = 5.0e2;
    ContactLawConfig {
        normal_stiffness,
        tangential_stiffness,
        rolling_stiffness,
        normal_damping: critical_damping(normal_stiffness),
        tangential_damping: critical_damping(tangential_stiffness),
        rolling_damping: critical_damping(rolling_stiffness),
        friction: (35.0_f32).to_radians().tan(),
        rolling_friction: 0.20,
    }
}

/// Real column-build recipe, generalized to an arbitrary
/// column size -- used by the real grain-count scaling sweep below. Not a
/// new/invented shape: identical jitter/polydispersity/spacing formula,
/// just parameterized instead of using the demo's own fixed R0/H0.
fn build_grain_column_sized(center_x: f32, base_y: f32, r0: usize, h0: usize) -> Vec<Grain> {
    let spacing = 2.6 * GRAIN_RADIUS;
    let mut rng = SmallRng(0xC0FF_EE11_u64);
    let mut grains = Vec::new();
    let column_width = 2 * r0 as i32;
    for row in 0..h0 {
        for col in 0..column_width {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let x = center_x - (column_width as f32 * spacing) * 0.5 + col as f32 * spacing + jx;
            let y = base_y + row as f32 * spacing + GRAIN_RADIUS + jy;
            let r = GRAIN_RADIUS * (0.9 + 0.2 * rng.next_f32());
            let mut g = Grain::new(Vec2::new(x, y), r, GRAIN_MASS * (r / GRAIN_RADIUS).powi(2));
            g.v = Vec2::ZERO;
            grains.push(g);
        }
    }
    grains
}

fn make_grains_mode_sim() -> Simulation {
    make_grains_mode_sim_sized(GRAINS_R0, GRAINS_H0, GRID, GRAINS_TERRAIN_HALF_WIDTH_CELLS)
}

/// Same real recipe as `make_grains_mode_sim`, generalized to an arbitrary
/// grain-column size and terrain/grid footprint -- used by the real
/// grain-count scaling sweep below to see whether/where a real substep- or
/// contact-cost bottleneck actually emerges beyond the shipped 80-grain
/// scene, using the SAME real stiffness/mass/radius/damping values, not
/// invented ones.
fn make_grains_mode_sim_sized(
    r0: usize,
    h0: usize,
    grid_res: usize,
    terrain_half_width_cells: i32,
) -> Simulation {
    let cfg = grain_contact_config();
    let m_eff = GRAIN_MASS * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let grain_safe_dt = (dt_crit * 0.2).min(0.02);

    let config = SimConfig {
        grid_res,
        dt: grain_safe_dt,
        gravity: Vec2::new(0.0, -0.3),
        adaptive_timestep: true,
        boundary_thickness: 2,
        ..SimConfig::default()
    };
    let terrain_center_y = FLOOR + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
    let terrain_center = Vec2::new(grid_res as f32 * 0.5, terrain_center_y);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(terrain_half_width_cells * 2, GRAINS_TERRAIN_HEIGHT_CELLS),
        box_center: terrain_center,
        material_id: 0,
        position_jitter: 0.3,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(2.0e3, 0.3)))
        .with_boundary(Box::new(FrictionBoundary::new(
            config.boundary_thickness,
            0.6,
        )));

    let terrain_top_y = terrain_center_y + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
    let max_jitter = 0.5 * 0.3 * (2.6 * GRAIN_RADIUS);
    let min_clearance = GRAIN_RADIUS * 1.1 - GRAIN_RADIUS + max_jitter;
    let grains =
        build_grain_column_sized(grid_res as f32 * 0.5, terrain_top_y + min_clearance, r0, h0);
    solver.add_grain_population(GrainPopulation::new(grains, grain_contact_config()));

    println!(
        "real dt_crit={dt_crit:.6}s, grain_safe_dt (config.dt)={grain_safe_dt:.6}s, \
         particles={}, grains={}",
        solver.particles().len(),
        solver.grain_populations()[0].grains.len()
    );
    solver
}

/// Real, direct measurement -- reports the ACTUAL `last_substeps()` the
/// real shipped scene's own `Simulation::step()` produces, and the real
/// wall-clock cost per call, settling the ~11 vs ~3504 discrepancy with
/// real data instead of a hand recomputation on either side.
#[test]
#[ignore = "perf diagnostic, run explicitly with --release --ignored --nocapture"]
fn real_shipped_grains_mode_substep_count_and_cost() {
    let mut solver = make_grains_mode_sim();
    // sim_speed=12 real solver.step() calls per rendered frame, the demo's
    // own real, currently-shipped pacing (see that file's own `sim_speed`
    // field doc) -- warm up a few frames first (initial settling transient),
    // then measure.
    for _ in 0..12 * 3 {
        solver.step();
    }
    let mut substep_counts = Vec::new();
    let mut frame_times_ms = Vec::new();
    for _frame in 0..10 {
        let start = Instant::now();
        for _ in 0..12 {
            solver.step();
            substep_counts.push(solver.last_substeps());
        }
        frame_times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_substeps: f64 =
        substep_counts.iter().sum::<usize>() as f64 / substep_counts.len() as f64;
    let min_substeps = *substep_counts.iter().min().unwrap();
    let max_substeps = *substep_counts.iter().max().unwrap();
    let avg_frame_ms: f64 = frame_times_ms.iter().sum::<f64>() / frame_times_ms.len() as f64;
    println!(
        "REAL measured: substeps/real-step min={min_substeps} max={max_substeps} avg={avg_substeps:.1}\n\
         REAL measured: avg wall-clock per rendered frame (12 steps) = {avg_frame_ms:.3}ms -> {:.1} fps",
        1000.0 / avg_frame_ms
    );
}

/// Real, direct measurement of whether substep count OR per-substep O(n^2)
/// contact-detection cost becomes the binding constraint as grain count
/// grows PAST the shipped demo's current 180-grain (80 base + 100 pour cap)
/// ceiling -- using the SAME real stiffness/mass/radius/damping values as
/// the shipped scene, just a bigger column and a proportionally bigger
/// domain (so a larger pile has room to sit without immediately spilling
/// past the terrain edges, not a change to per-grain physics).
#[test]
#[ignore = "perf diagnostic, run explicitly with --release --ignored --nocapture"]
fn real_grain_count_scaling_substep_and_fps_sweep() {
    // (r0, h0, grid_res, terrain_half_width_cells) -- grid_res/terrain held
    // COMPLETELY FIXED across every case (real, deliberate control): an
    // earlier version of this sweep scaled the domain up alongside grain
    // count, which also grows the terrain's own MPM particle count
    // (1920->4800) -- a real confound caught before trusting the result,
    // since that growth could dominate or mask the grain-specific cost this
    // sweep exists to isolate. A wider pile may spill past the fixed
    // terrain's edges at the largest case -- accepted, since this measures
    // raw per-call cost vs grain count, not physical settling realism.
    let cases: [(usize, usize, usize, i32); 4] = [
        (GRAINS_R0, GRAINS_H0, GRID, GRAINS_TERRAIN_HALF_WIDTH_CELLS), // 80 grains, shipped baseline
        (8, 20, GRID, GRAINS_TERRAIN_HALF_WIDTH_CELLS),                // 320 grains
        (11, 28, GRID, GRAINS_TERRAIN_HALF_WIDTH_CELLS),               // ~616 grains
        (14, 36, GRID, GRAINS_TERRAIN_HALF_WIDTH_CELLS),               // ~1008 grains
    ];
    for (r0, h0, grid_res, half_width) in cases {
        let mut solver = make_grains_mode_sim_sized(r0, h0, grid_res, half_width);
        let n_grains = solver.grain_populations()[0].grains.len();
        // Settling transient, same real 3-frame warmup as the baseline test.
        for _ in 0..12 * 3 {
            solver.step();
        }
        let mut substep_counts = Vec::new();
        let start = Instant::now();
        const CALLS: usize = 60;
        for _ in 0..CALLS {
            solver.step();
            substep_counts.push(solver.last_substeps());
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let avg_substeps: f64 =
            substep_counts.iter().sum::<usize>() as f64 / substep_counts.len() as f64;
        let max_substeps = *substep_counts.iter().max().unwrap();
        let ms_per_call = elapsed_ms / CALLS as f64;
        // Real fps at this scene's own shipped sim_speed=12 calls/rendered-frame.
        let fps = 1000.0 / (ms_per_call * 12.0);
        println!(
            "n_grains={n_grains:<5} avg_substeps={avg_substeps:<6.2} max_substeps={max_substeps:<4} \
             ms/call={ms_per_call:<8.4} implied_fps_at_sim_speed_12={fps:.1}"
        );
    }
}

extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-16) -- delete after use.
///
/// Answers the one open question in `HANDOFF_fluid_gpu_thin_layer_bug.md`:
/// does the CPU solver, run on the EXACT same dam-break scene as
/// `basic_fluids.rs`'s `make_sim()` (same grid/spacing/EOS/gravity
/// derivation, including the runtime `gravity_fraction=0.003` derating that
/// demo's egui panel applies every frame from frame 0), show the same
/// near-1D thin-layer collapse the GPU demo shows after a few thousand
/// frames? Headless (no window/render), so it can run thousands of steps
/// unattended and print `MaterialStats` (real bounding-box extent + J range)
/// at a fixed interval for direct comparison against the GPU log lines
/// already captured in the handoff doc.
///
///   cargo run --example fluid_thin_layer_diag
use emerge::diagnostics::per_material_stats;
use emerge::particle::Particles;
use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

/// Real, direct CPU/GPU comparison point: same depth-stratification helper
/// as `fluid_thin_layer_diag_gpu.rs`'s own (bins particles into N_BANDS
/// horizontal bands by current y, prints mean J/mean vy per band) -- ported
/// to CPU's SoA `Particles` layout instead of GPU's AoS `&[Particle]`.
fn print_depth_stratification(frame: u64, particles: &Particles, mat: u32) {
    const N_BANDS: usize = 8;
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..particles.x.len() {
        if particles.material_id[i] != mat {
            continue;
        }
        y_min = y_min.min(particles.x[i].y);
        y_max = y_max.max(particles.x[i].y);
    }
    let span = (y_max - y_min).max(1e-6);
    let mut count = [0u32; N_BANDS];
    let mut j_sum = [0f64; N_BANDS];
    let mut vy_sum = [0f64; N_BANDS];
    for i in 0..particles.x.len() {
        if particles.material_id[i] != mat {
            continue;
        }
        let mut band = (((particles.x[i].y - y_min) / span) * N_BANDS as f32) as usize;
        band = band.min(N_BANDS - 1);
        count[band] += 1;
        j_sum[band] += particles.deformation_gradient[i].determinant() as f64;
        vy_sum[band] += particles.v[i].y as f64;
    }
    println!(
        "  -- depth stratification @ frame {frame} (y_min={y_min:.2} y_max={y_max:.2} span={span:.3}) --"
    );
    for b in (0..N_BANDS).rev() {
        let n = count[b].max(1) as f64;
        println!(
            "     band {b} (floor-most={})  n={:4}  mean_J={:.4}  mean_vy={:.5}",
            b == 0,
            count[b],
            j_sum[b] / n,
            vy_sum[b] / n,
        );
    }
}

/// Ported from `examples/gpu/fluid_thin_layer_diag_gpu.rs`'s own function --
/// same exact measurement, CPU's SoA layout instead of GPU's AoS. Direct
/// apples-to-apples check of whether GPU's measured J<=1.1 positive-
/// divergence bias (0.3-6.0 throughout the violent splash phase, see
/// HANDOFF's "Eighth pass") is real splash physics (same magnitude on
/// CPU) or a GPU-specific computational discrepancy (much smaller on CPU).
fn print_fast_phase_divergence(frame: u64, particles: &Particles, mat: u32) {
    let mut fast_sum = 0f64;
    let mut fast_n = 0u32;
    let mut slow_sum = 0f64;
    let mut slow_n = 0u32;
    let mut high_j_sum = 0f64;
    let mut high_j_n = 0u32;
    let mut low_j_sum = 0f64;
    let mut low_j_n = 0u32;
    for i in 0..particles.x.len() {
        if particles.material_id[i] != mat {
            continue;
        }
        let c = particles.velocity_gradient[i];
        let div = (c.x_axis.x + c.y_axis.y) as f64;
        let j = particles.deformation_gradient[i].determinant();
        if particles.v[i].length() >= 1.0 {
            fast_sum += div;
            fast_n += 1;
        } else {
            slow_sum += div;
            slow_n += 1;
        }
        if j >= 1.9 {
            high_j_sum += div;
            high_j_n += 1;
        } else if j <= 1.1 {
            low_j_sum += div;
            low_j_n += 1;
        }
    }
    println!(
        "  -- fast-phase-divergence @ frame {frame}: fast(|v|>=1) n={} mean_div={:.5}  slow n={} mean_div={:.5}  |  J>=1.9 n={} mean_div={:.5}  J<=1.1 n={} mean_div={:.5}",
        fast_n,
        if fast_n > 0 {
            fast_sum / fast_n as f64
        } else {
            0.0
        },
        slow_n,
        if slow_n > 0 {
            slow_sum / slow_n as f64
        } else {
            0.0
        },
        high_j_n,
        if high_j_n > 0 {
            high_j_sum / high_j_n as f64
        } else {
            0.0
        },
        low_j_n,
        if low_j_n > 0 {
            low_j_sum / low_j_n as f64
        } else {
            0.0
        },
    );
}

const GRID: usize = 64;
const DT: f32 = 0.1;
const SPACING: f32 = 0.5;
const MAT_WATER: u32 = 0;
const GRAVITY_FRACTION: f32 = 0.003;
const N_STEPS: u64 = 1000;
const LOG_INTERVAL: u64 = 20;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 150,
        phase_rules_once_per_step: true,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    const WATER_EOS_POWER: f32 = 3.0;
    const COLUMN_HEIGHT_CELLS: f32 = 52.0 * SPACING;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water = NewtonianFluidMaterial::new(0.1, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);

    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let real_gravity = config.gravity;
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    // Same runtime derating `basic_fluids.rs`'s egui panel applies from
    // frame 0 (default `gravity_fraction = 0.003` slider value, never
    // touched in an unattended run) -- NOT part of `SimConfig::earth`
    // itself, so it must be replicated explicitly to match that demo.
    solver.set_gravity(real_gravity * GRAVITY_FRACTION);
    solver
}

fn main() {
    let mut solver = make_sim();
    // Real fix: `box_size` is already the region's real physical extent in
    // grid-units (SpawnRegion's own convention -- "box_size is the region's
    // extent in CELLS, so it stays untouched" regardless of spawn spacing,
    // confirmed against `basic_fluids_gpu.rs`'s own identical comment on its
    // own `box_size`). The previous `14.0*SPACING*52.0*SPACING` double-
    // counted spacing (particle SAMPLING density, not the box's own size),
    // giving 182 instead of the real 728 -- a 4x understatement of
    // `implied_h_if_area_conserved` below.
    let initial_area: f32 = 14.0 * 52.0;
    println!(
        "fluid_thin_layer_diag: n={}  initial_column_area={:.2} grid-units^2  (box 14x52 cells @ spacing={})",
        solver.particles().len(),
        initial_area,
        SPACING,
    );

    const TRACK_IDX: usize = 1456;
    const TRACK_STEPS: u64 = 60;
    for step in 1..=N_STEPS {
        solver.step_n(1);
        if step <= TRACK_STEPS {
            let p = solver.particles();
            let c = p.velocity_gradient[TRACK_IDX];
            let j = p.deformation_gradient[TRACK_IDX].determinant();
            println!(
                "TRACK[{TRACK_IDX}] step={step:3}  x=({:.4},{:.4})  v=({:.4},{:.4})  C=[{:.4},{:.4};{:.4},{:.4}]  divC={:.5}  J={:.6}",
                p.x[TRACK_IDX].x,
                p.x[TRACK_IDX].y,
                p.v[TRACK_IDX].x,
                p.v[TRACK_IDX].y,
                c.x_axis.x,
                c.y_axis.x,
                c.x_axis.y,
                c.y_axis.y,
                c.x_axis.x + c.y_axis.y,
                j,
            );
        }
        if step % LOG_INTERVAL == 0 || step == N_STEPS {
            let snap = solver.diagnostics_snapshot();
            let stats = per_material_stats(solver.particles());
            for s in &stats {
                let ext = s.extent_max - s.extent_min;
                let implied_h = if ext.x > 1e-6 {
                    initial_area / ext.x
                } else {
                    f32::NAN
                };
                println!(
                    "frame={:5}  {}  implied_h_if_area_conserved={:.3}  mass_err={:.2e}  y_bottom={:.4}  y_top={:.4}  subs={}",
                    step,
                    s.format(Some("water")),
                    implied_h,
                    snap.relative_mass_error,
                    s.extent_min.y,
                    s.extent_max.y,
                    solver.last_substeps(),
                );
            }
        }
        if matches!(step, 200 | 600 | 1000) {
            print_depth_stratification(step, solver.particles(), MAT_WATER);
        }
        if matches!(step, 20 | 40 | 60 | 80 | 100 | 150 | 200 | 300 | 400) {
            print_fast_phase_divergence(step, solver.particles(), MAT_WATER);
        }
    }
}

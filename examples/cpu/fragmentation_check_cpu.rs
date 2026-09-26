extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- the CPU twin of
/// `examples/gpu/fragmentation_check_gpu.rs`: identical DamBreak geometry,
/// material, config and drag, run through CPU's own `Simulation::step()`.
///
/// Answers one question the GPU fluid investigation never verified
/// directly: does CPU -- which has NO shear-relaxation damping at all, and
/// re-runs `choose_substep_dt` before EVERY substep -- explode at floor
/// impact on this exact scene? If CPU stays coherent, the explosion is
/// GPU-specific and the two paths can be diffed. If CPU explodes too, the
/// cause is in the shared physics, not in anything GPU-only.
///
/// Reports the PEAK of every metric across the run, not just the final
/// state -- a final-state-only readout already hid a real mid-run impact
/// transient once in this investigation.
///
///   cargo run --release --example fragmentation_check_cpu
///
/// Same knobs as the GPU twin where they apply: PATTERN, N_STEPS, MAT_CFL,
/// GRAV_SIZING, NEAR_WALL_SCALE, plus CFL_DIVISOR (forces more substeps).
use emerge::{
    GravityWellField, LinearDragField, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::collections::HashMap;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;
const WATER_RHO_GRID: f32 = 0.1;

fn isolated_count(xs: &[Vec2]) -> usize {
    const CELL: f32 = SPACING * 3.0;
    const ISOLATION_THRESHOLD: f32 = SPACING * 4.0;
    let mut buckets: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, x) in xs.iter().enumerate() {
        let key = ((x.x / CELL).floor() as i32, (x.y / CELL).floor() as i32);
        buckets.entry(key).or_default().push(i);
    }
    let mut count = 0;
    for (i, x) in xs.iter().enumerate() {
        let (bx, by) = ((x.x / CELL).floor() as i32, (x.y / CELL).floor() as i32);
        let mut best = f32::MAX;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(bucket) = buckets.get(&(bx + dx, by + dy)) {
                    for &j in bucket {
                        if j != i {
                            best = best.min((xs[j] - *x).length());
                        }
                    }
                }
            }
        }
        if best > ISOLATION_THRESHOLD {
            count += 1;
        }
    }
    count
}

fn main() {
    let dt = 0.1;
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 1000,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: std::env::var("MAT_CFL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3),
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        fluid_near_wall_cfl_scale: std::env::var("NEAR_WALL_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0),
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    // CFL_DIVISOR (env, default 1): divides both CFL coefficients to force
    // proportionally MORE substeps per frame with otherwise-identical physics.
    // A smaller CFL number can only make an explicit scheme MORE accurate; if
    // the scene gets WORSE as substeps rise, the instability has a per-SUBSTEP
    // gain (compounds with substep count), not a per-unit-time one.
    let cfl_divisor: f32 = std::env::var("CFL_DIVISOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let config = SimConfig {
        cfl_coefficient: config.cfl_coefficient / cfl_divisor,
        material_cfl_coefficient: config.material_cfl_coefficient / cfl_divisor,
        ..config
    };
    println!("CFL_DIVISOR={cfl_divisor}");
    let grav_sizing: f32 = std::env::var("GRAV_SIZING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    let v_max_grid = (2.0 * grav_sizing * 52.0f32).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_viscosity = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_viscosity,
        1000.0 * c_ref_m_s * c_ref_m_s / 3.0,
        3.0,
    );
    water.bulk_viscosity = 3.0 * water_viscosity;
    water.pressure_floor = config.stress_from_si_physical(-100_000.0, 1000.0);

    // PATTERN (env: dam | drop | vortex, default dam) -- the same three
    // geometries as basic_fluids_gpu.rs and fragmentation_check_gpu.rs.
    let pattern = std::env::var("PATTERN").unwrap_or_else(|_| "dam".into());
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_RHO_GRID * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    };
    let (spawn, extra) = match pattern.as_str() {
        "dam" => (
            water_region(IVec2::new(14, 52), Vec2::new(20.0, 30.0)),
            None,
        ),
        "drop" => (
            water_region(IVec2::new(50, 12), Vec2::new(32.0, 9.0)),
            Some(water_region(IVec2::new(7, 7), Vec2::new(32.0, 42.0))),
        ),
        "vortex" => (
            water_region(IVec2::new(54, 48), Vec2::new(32.0, 26.0)),
            None,
        ),
        other => panic!("unknown PATTERN={other}"),
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(water))
        .with_force_field(Box::new(LinearDragField::new(
            Vec2::ZERO,
            0.1,
            1 << MAT_WATER,
        )));
    if let Some(extra) = extra {
        let _blob_tag = sim.add_body(extra);
    }
    if pattern == "vortex" {
        let center = Vec2::new(32.0, 26.0);
        let edge_r = 17.0f32;
        let gm = 0.02 * config.gravity.length() * edge_r * edge_r;
        sim = sim.with_force_field(Box::new(GravityWellField::point(center, gm, 1.0, 2.0)));
        let seed_l = 1.0 * edge_r;
        let p = sim.particles_mut();
        for i in 0..p.x.len() {
            let r = p.x[i] - center;
            if r.length() < 22.0 {
                let d = r.length().max(2.0);
                p.v[i] = (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
            }
        }
    }

    println!("n={}  PATTERN={pattern}, CPU twin", sim.particles().x.len());
    let (mut peak_speed, mut peak_iso, mut peak_jmax, mut min_jmin) =
        (0.0f32, 0usize, 0.0f32, f32::MAX);
    let start = std::time::Instant::now();
    let n_steps: u64 = std::env::var("N_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    for step in 1..=n_steps {
        sim.step();
        let p = sim.particles();
        let max_speed = p.v.iter().map(|v| v.length()).fold(0.0f32, f32::max);
        let (mut jmin, mut jmax) = (f32::MAX, f32::MIN);
        for f in &p.deformation_gradient {
            let j = f.determinant();
            jmin = jmin.min(j);
            jmax = jmax.max(j);
        }
        let iso = isolated_count(&p.x);
        peak_speed = peak_speed.max(max_speed);
        peak_iso = peak_iso.max(iso);
        peak_jmax = peak_jmax.max(jmax);
        min_jmin = min_jmin.min(jmin);
        println!(
            "step={step:3}  max_speed={max_speed:8.3}  J=[{jmin:.3},{jmax:.3}]  isolated={iso:3}  sub={}",
            sim.last_substeps()
        );
    }
    println!(
        "PEAK over run: max_speed={peak_speed:.3}  J=[{min_jmin:.3},{peak_jmax:.3}]  isolated={peak_iso}  wall={:.1}s",
        start.elapsed().as_secs_f64()
    );
}

extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Ports basic_fluids_gpu.rs's own PROVEN, live-verified `Pattern::Vortex`
/// recipe (see that file's own extensive doc on the `Pattern::Vortex` match
/// arm, and project_vortex_siphon_saga_2026-08-15 memory) to the CPU path,
/// targeting basic_fluids_gui.rs's real config (GRID=64, SPACING=0.5, the
/// same as the GPU demo -- so POOL_BOX/DRAIN_EDGE_R/DRAIN_SOFTENING/
/// SWIRL_SEED_RADIUS/SEED_EDGE_SPEED, all in grid-cell units, port
/// unchanged). Only drain_gm needs re-deriving: the GPU demo used a FIXED
/// derated gravity (-0.3); this demo's REAL default live gravity is
/// `real_gravity(-981, from SimConfig::earth) * gravity_fraction(0.003
/// default) ~= -2.94` -- used as the reference here, same formula shape
/// (`DRAIN_EDGE_ACCEL_FRACTION * |gravity| * drain_edge_r^2`) as the proven
/// recipe, just evaluated against THIS demo's own real operating gravity.
///
/// Mechanism unchanged from the proven recipe: `GravityWellField` drain +
/// free-vortex (constant angular momentum) seed velocity, NO RadialConfinement,
/// NO mass sink -- Kelvin's circulation theorem (1869) is why a real vortex
/// persists without one.
use emerge::{
    GravityWellField, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;

// Ported unchanged from basic_fluids_gpu.rs's proven Pattern::Vortex (grid-
// cell units, independent of gravity/material specifics).
const POOL_BOX: IVec2 = IVec2::new(54, 48);
const DRAIN_EDGE_R: f32 = 17.0;
const DRAIN_EDGE_ACCEL_FRACTION: f32 = 0.02;
const DRAIN_SOFTENING: f32 = 2.0;
const SWIRL_SEED_RADIUS: f32 = 22.0;
const SEED_EDGE_SPEED: f32 = 1.0;

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 150,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let real_gravity = config.gravity; // (0, -981) at dx=0.01m, SimConfig::earth
    let gravity_fraction = 0.003_f32; // this demo's own real default
    let live_gravity = real_gravity * gravity_fraction;
    println!(
        "live_gravity={:?} (mag={:.3})",
        live_gravity,
        live_gravity.length()
    );

    let mut config = config;
    config.gravity = live_gravity;

    const WATER_EOS_POWER: f32 = 3.0;
    let column_height_cells = 52.0 * SPACING;
    let derated_gravity = live_gravity.length();
    let v_max_grid = (2.0 * derated_gravity * column_height_cells).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water =
        NewtonianFluidMaterial::new(0.1, 1.0e-3, water_tait_b_pa.max(1.0e-6), WATER_EOS_POWER);

    let pool_center = Vec2::new(32.0, 26.0);
    // Real change under test: drain near the FLOOR (y=10, pool spans
    // ~y=[2,50]), not the pool's vertical midpoint -- user's real complaint
    // was the swirl reading as "floating in the middle" instead of a
    // downward funnel. Same mechanism, same formulas, just a different
    // attraction point -- no new physics invented.
    let drain_center = Vec2::new(pool_center.x, 10.0);
    let drain_gm = DRAIN_EDGE_ACCEL_FRACTION * live_gravity.length() * DRAIN_EDGE_R * DRAIN_EDGE_R;
    println!("drain_gm={:.4}", drain_gm);

    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: POOL_BOX,
        box_center: pool_center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    let mut sim = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_named_force_field(
            "vortex_drain",
            Box::new(GravityWellField::new(
                vec![(drain_center, drain_gm)],
                1.0,
                DRAIN_SOFTENING,
            )),
        );

    let seed_l = SEED_EDGE_SPEED * DRAIN_EDGE_R;
    for i in 0..sim.particles().x.len() {
        let x = sim.particles().x[i];
        let r = x - drain_center;
        let dist = r.length();
        if dist < SWIRL_SEED_RADIUS {
            let d = dist.max(DRAIN_SOFTENING);
            sim.particles_mut().v[i] = (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
        }
    }

    let total_before = sim.particles().x.len();
    println!("{total_before} particles");

    let mut max_speed_ever = 0.0f32;
    let mut any_nonfinite = false;
    const STEPS: usize = 400;
    for step in 0..STEPS {
        sim.step();
        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
            max_speed_ever = max_speed_ever.max(v.length());
        }
        if any_nonfinite {
            println!("ABORTED step={step}: non-finite state");
            return;
        }
        if step % 50 == 0 {
            // Real structure check: mean tangential speed of particles still
            // within the seed radius, and mean radius (are they still
            // orbiting, not scattered/collapsed?).
            let mut sum_speed = 0.0f32;
            let mut sum_r = 0.0f32;
            let mut n = 0;
            for i in 0..sim.particles().x.len() {
                let r = sim.particles().x[i] - drain_center;
                let dist = r.length();
                if dist < SWIRL_SEED_RADIUS {
                    sum_speed += sim.particles().v[i].length();
                    sum_r += dist;
                    n += 1;
                }
            }
            println!(
                "step={step:>3} max_speed_ever={max_speed_ever:.2} core_n={n} core_mean_speed={:.3} core_mean_r={:.2}",
                sum_speed / n.max(1) as f32,
                sum_r / n.max(1) as f32
            );
        }
    }
    let total_after = sim.particles().x.len();
    println!(
        "DONE {STEPS} steps: count {total_before}->{total_after} max_speed_ever={max_speed_ever:.2} finite={}",
        !any_nonfinite
    );
}

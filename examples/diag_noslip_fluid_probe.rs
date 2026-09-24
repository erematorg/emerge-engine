extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real check before trusting `NoSlipBoundary::is_strict_wc_mpm_fluid_
/// compatible`: does forcing velocity to exactly zero at the wall (a much
/// more aggressive correction than SlipBoundary's normal-only clamp) stay
/// stable against real strict-fluid water, AND does it produce the real,
/// EXPECTED physical difference (fluid near the wall measurably slower than
/// with free-slip -- a real boundary layer, not just "doesn't crash")?
use emerge::{
    BoundaryCondition, NewtonianFluidMaterial, NoSlipBoundary, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.9;

fn run(name: &str, boundary: Box<dyn BoundaryCondition>) -> (bool, f32) {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        // Close to the left wall on purpose -- this is exactly the contact
        // regime being tested, not incidental.
        box_size: IVec2::new(14, 30),
        box_center: Vec2::new(9.0, 20.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(boundary);

    // Fixed particle index set, captured BEFORE any stepping -- tracks the
    // SAME particles across both runs (an apples-to-apples comparison),
    // instead of resampling "whoever happens to be near the wall now" at
    // the end, which the first pass showed can differ in count (34 vs 22)
    // between the two boundaries -- a real confound, not a clean measurement.
    let near_wall_indices: Vec<usize> = (0..sim.particles().x.len())
        .filter(|&i| sim.particles().x[i].x < 5.0)
        .collect();

    let total_before = sim.particles().x.len();
    let mut max_speed_ever = 0.0f32;
    let mut any_nonfinite = false;
    const STEPS: usize = 300;
    for _ in 0..STEPS {
        sim.step();
        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
            max_speed_ever = max_speed_ever.max(v.length());
        }
        if any_nonfinite {
            break;
        }
    }
    let total_after = sim.particles().x.len();

    // Same fixed particles, wherever they ended up.
    let n = near_wall_indices.len();
    let sum: f32 = near_wall_indices
        .iter()
        .map(|&i| sim.particles().v[i].length())
        .sum();
    let near_wall_mean_speed = sum / n.max(1) as f32;

    let ok = !any_nonfinite && total_before == total_after;
    println!(
        "{name}: finite={} mass_conserved={} max_speed_ever={:.2} near_wall_mean_speed={:.4} (n={n})",
        !any_nonfinite,
        total_before == total_after,
        max_speed_ever,
        near_wall_mean_speed
    );
    (ok, near_wall_mean_speed)
}

fn main() {
    let (slip_ok, slip_near_wall) = run("SlipBoundary (baseline)", Box::new(SlipBoundary::new(2)));
    let (noslip_ok, noslip_near_wall) = run(
        "NoSlipBoundary (new, flag TEMPORARILY forced compatible for this measurement)",
        Box::new(NoSlipBoundary::new(2)),
    );

    println!(
        "\nnear-wall mean speed: slip={:.4} no-slip={:.4} (no-slip should be LOWER -- real boundary layer)",
        slip_near_wall, noslip_near_wall
    );
    if slip_ok && noslip_ok && noslip_near_wall < slip_near_wall {
        println!(
            "PASS: both stable, no-slip measurably slower near the wall, as real physics predicts."
        );
    } else {
        println!(
            "FAIL or UNEXPECTED -- do not trust the fluid-compatible flag without investigating."
        );
    }
}

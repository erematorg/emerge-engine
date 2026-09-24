extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real, sourced validation methodology for `NoSlipBoundary`, replacing the
/// earlier ad-hoc "collapsing column" test (which gave an ambiguous,
/// unexplained result -- see that probe's own doc). WebSearch confirmed
/// Couette/Poiseuille flow is the ESTABLISHED real MPM benchmark
/// specifically for validating no-slip walls (arXiv 2402.11719, "Mixed
/// material point method formulation, stabilization, and validation" --
/// already cited elsewhere in this project's own research; Taylor-Couette
/// MPM validation reported <1% mean error vs. real experimental data).
///
/// Setup: a horizontal channel, no-slip on ALL four domain walls (this
/// engine's `NoSlipBoundary` isn't direction-selective -- a real, disclosed
/// simplification; sampled far from the left/right walls to minimize their
/// end-effect), fluid driven by a horizontal body force (gravity repurposed
/// as the Poiseuille driving force, same role a pressure gradient plays in
/// the classical derivation). Real analytical target (steady, laminar,
/// Newtonian, no-slip both walls -- standard result, e.g. White's "Viscous
/// Fluid Flow"): u(y) = (rho*f / (2*mu)) * y*(H-y) for y in [0,H] -- a
/// parabola, EXACTLY zero at both walls, maximum at the centerline.
use emerge::{NewtonianFluidMaterial, NoSlipBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.9;
const CHANNEL_H: f32 = 50.0; // fluid fills nearly the full vertical extent
const DRIVING_ACCEL: f32 = 0.3; // same order of magnitude as this engine's other demos' derated gravity
const VISCOSITY: f32 = 1.0e-3;
const REST_DENSITY: f32 = 0.1;

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        // Driving force acts HORIZONTALLY here -- it plays the Poiseuille
        // pressure-gradient role, not vertical gravity in this specific
        // validation scene.
        gravity: Vec2::new(DRIVING_ACCEL, 0.0),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::new(REST_DENSITY, VISCOSITY, 2.5, 3.0);

    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(REST_DENSITY * SPACING * SPACING),
        box_size: IVec2::new(50, CHANNEL_H as i32),
        box_center: Vec2::new(32.0, 32.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let mut sim = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(NoSlipBoundary::new(config.boundary_thickness)));

    // Run to a real steady state (viscous channel flow settles), not a
    // snapshot mid-transient.
    const STEPS: usize = 800;
    let mut any_nonfinite = false;
    for step in 0..STEPS {
        sim.step();
        for v in sim.particles().v.iter() {
            if !v.is_finite() {
                any_nonfinite = true;
            }
        }
        if any_nonfinite {
            println!("ABORTED step={step}: non-finite state");
            return;
        }
    }

    // Sample the velocity profile u(y) far from the left/right walls
    // (middle third of the domain horizontally, x in [22,42]) to minimize
    // their real, disclosed end-effect (NoSlipBoundary applies to all 4
    // walls, not just top/bottom -- see module doc).
    const N_BINS: usize = 12;
    let mut bin_sum = [0.0f32; N_BINS];
    let mut bin_n = [0usize; N_BINS];
    for i in 0..sim.particles().x.len() {
        let p = sim.particles().x[i];
        if p.x < 22.0 || p.x > 42.0 {
            continue;
        }
        let y0 = 4.0; // approx channel bottom (near the true wall + thickness)
        let h = CHANNEL_H;
        let rel_y = ((p.y - y0) / h).clamp(0.0, 0.999_999);
        let bin = (rel_y * N_BINS as f32) as usize;
        bin_sum[bin] += sim.particles().v[i].x;
        bin_n[bin] += 1;
    }

    println!("Poiseuille validation -- measured u(y) vs. analytical parabola:");
    println!(
        "(rho*f/(2*mu) = {:.4}, mu={VISCOSITY}, f={DRIVING_ACCEL}, rho={REST_DENSITY})",
        REST_DENSITY * DRIVING_ACCEL / (2.0 * VISCOSITY)
    );
    let coeff = REST_DENSITY * DRIVING_ACCEL / (2.0 * VISCOSITY);
    let mut max_measured = 0.0f32;
    for b in 0..N_BINS {
        if bin_n[b] == 0 {
            continue;
        }
        let mean_u = bin_sum[b] / bin_n[b] as f32;
        max_measured = max_measured.max(mean_u.abs());
        let y_frac = (b as f32 + 0.5) / N_BINS as f32;
        let y = y_frac * CHANNEL_H;
        let analytical_u = coeff * y * (CHANNEL_H - y);
        println!(
            "  y/H={:.2}  measured_u={:>8.4}  analytical_u={:>8.4}  n={}",
            y_frac, mean_u, analytical_u, bin_n[b]
        );
    }
    let wall_bin_speed_bottom = if bin_n[0] > 0 {
        (bin_sum[0] / bin_n[0] as f32).abs()
    } else {
        -1.0
    };
    let wall_bin_speed_top = if bin_n[N_BINS - 1] > 0 {
        (bin_sum[N_BINS - 1] / bin_n[N_BINS - 1] as f32).abs()
    } else {
        -1.0
    };
    println!(
        "\nnear-wall bins: bottom={:.4} top={:.4} (should be << center peak {:.4} if no-slip is real)",
        wall_bin_speed_bottom, wall_bin_speed_top, max_measured
    );
    if wall_bin_speed_bottom < max_measured * 0.3 && wall_bin_speed_top < max_measured * 0.3 {
        println!(
            "PASS: near-wall speed is meaningfully lower than the centerline peak -- real no-slip signature confirmed."
        );
    } else {
        println!(
            "FAIL: near-wall speed is NOT meaningfully lower than centerline -- no-slip signature not confirmed by this test either."
        );
    }
}

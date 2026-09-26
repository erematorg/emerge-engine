//! What does the acoustic CFL safety factor actually cost, and buy?
//!
//! `SimConfig::material_cfl_coefficient` is 0.5, and its own doc says so
//! plainly: "like any CFL number, it's a stability margin, not a measured
//! material property". The registry already holds the one measurement
//! that bounds it -- an isolated particle stays bounded up to 0.80 dx/c
//! and diverges at 0.85, against the analytic single-particle value
//! `sqrt((lambda + 2 mu) / (2 (lambda + mu)))`, which is 0.84 at Poisson
//! 0.3 but moves with Poisson: 0.89 at 0.2, 0.74 at 0.45.
//!
//! So the margin is real but unquantified, and the choice of how much of
//! it to use is not a decision this file should make. What it can do is
//! measure both sides of it on one scene:
//!
//! - the COST, in substeps and frame time, which falls as the factor rises;
//! - what the scene KEEPS, because this solver dissipates energy per
//!   substep (audit pass A: a free elastic block keeps 0.69, 0.51 and 0.37
//!   of its energy after the same physical time at 256, 1024 and 4096
//!   steps), so fewer substeps is not only cheaper, it is also less
//!   damped. Those two usually trade against each other. Here they do not.
//!
//! Energy is the real thing, not a proxy: kinetic plus the strain energy
//! THIS engine's own law stores (a Simo-Pister split, see `total_energy`),
//! integrated over each particle's own reference volume. The block floats
//! free with no gravity and no boundary contact, so nothing outside it can
//! add or remove any.
//!
//! # What this scene does NOT settle
//!
//! Read at Poisson 0.3, half a second, this block reaches `max |J - 1|` of
//! 2e-4 and nothing diverges anywhere up to 1.20 times the analytic
//! single-particle limit. That is not a stability verdict, it is a gentle
//! scene: the acoustic bound never binds hard enough to break.
//!
//! The cost side that can be read is the substep count, 7 a frame down to
//! 3 between factors 0.4 and 1.0, so a factor of 2.3. The frame rate in
//! the same table is NOT that measurement: factors 0.9 and 1.0 run the
//! identical 3.0 substeps and still read 575 against 832 fps, 45 % apart
//! at equal work, so wall-clock timing on this machine carries at least
//! that much noise and any speed-up quoted from it is partly that noise.
//!
//! The energy column spans 83 to 96 % in no order. The honest reading is
//! that this measurement cannot detect an effect of the factor at that
//! spread, which is not the same statement as the factor costing nothing:
//! an effect smaller than 13 points would be invisible here. Saying
//! "no trend" would claim the second while only having grounds for the
//! first.
//!
//! Choosing a factor needs the other half: a scene violent enough for the
//! bound to bind, which is the impact probe the core plan names next to
//! this one. Until that exists, this table is half the evidence and
//! should not be read as permission to raise anything.
//!
//!   cargo run --release --example cfl_margin_probe
//!   CFL_PROBE_NU=0.45 cargo run --release --example cfl_margin_probe
extern crate emerge_engine as emerge;

use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 48;
const DX_M: f32 = 0.01;
const BLOCK: IVec2 = IVec2::new(8, 8);
const RHO_KG_M3: f32 = 1000.0;
const YOUNG_PA: f32 = 2.0e5;
/// The block is set spinning and stretching at once, so both the kinetic
/// and the stored part of its energy are non-trivial from the first step.
const SPIN_PER_SECOND: f32 = 6.0;

/// Analytic single-particle stability limit for an explicit MLS-MPM step,
/// as a fraction of `dx / c_p`: `sqrt((lambda + 2 mu) / (2 (lambda + mu)))`.
/// In terms of Poisson's ratio alone, with `lambda/mu = 2 nu / (1 - 2 nu)`.
fn single_particle_limit(nu: f32) -> f32 {
    let ratio = 2.0 * nu / (1.0 - 2.0 * nu);
    ((ratio + 2.0) / (2.0 * (ratio + 1.0))).sqrt()
}

/// Kinetic plus stored elastic energy, in joules per metre of depth.
fn total_energy(sim: &Simulation, lambda: f32, mu: f32) -> f64 {
    let p = sim.particles();
    let mut energy = 0.0f64;
    for i in 0..p.len() {
        let v = p.v[i] * DX_M;
        energy += 0.5 * f64::from(p.mass[i]) * f64::from(v.length_squared());
        let f = p.deformation_gradient[i];
        let j = f.determinant();
        if j <= 0.0 {
            continue;
        }
        // The potential THIS engine's stress derives from, not the
        // textbook compressible Neo-Hookean one: `elastic.rs` uses a
        // Simo-Pister split, deviatoric `mu/J dev(B)` and volumetric
        // `k ln(J) I` with `k = lambda + mu` in plane strain, so
        // `W = mu/2 (tr(B)/J - 2) + k/2 (ln J)^2`. Measuring the classic
        // form instead reads a free block GAINING 20 to 70 % of its
        // energy, because that sum is simply not the one this solver
        // integrates.
        let i1 = f.x_axis.length_squared() + f.y_axis.length_squared();
        let ln_j = j.ln();
        let w = 0.5 * mu * (i1 / j - 2.0) + 0.5 * (lambda + mu) * ln_j * ln_j;
        energy += f64::from(w) * f64::from(p.initial_volume[i]);
    }
    energy
}

fn main() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let nu = env("CFL_PROBE_NU", 0.3);
    let seconds = env("CFL_PROBE_SECONDS", 0.5);
    let limit = single_particle_limit(nu);
    println!(
        "Poisson {nu}: the analytic single-particle limit is {limit:.3} dx/c, and this engine ships {:.2}",
        SimConfig::earth(GRID, DX_M, 0.001).material_cfl_coefficient
    );
    println!(
        "  factor   of limit   substeps/frame    ms/frame     fps    energy kept    max |J-1|"
    );

    for factor in [0.4f32, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        let config = SimConfig {
            min_dt: 1.0e-8,
            max_substeps_per_step: 512,
            material_cfl_coefficient: factor,
            gravity: Vec2::ZERO,
            ..SimConfig::earth(GRID, DX_M, 0.001)
        };
        let material = NeoHookeanMaterial::from_young_modulus(YOUNG_PA, nu);
        let mass = RHO_KG_M3 * (0.5 * DX_M).powi(2);
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: BLOCK,
            box_center: Vec2::splat(GRID as f32 * 0.5),
            material_id: 0,
            mass_override: Some(mass),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        // A rigid spin, with the affine field that goes WITH it. Setting
        // only the velocities leaves `velocity_gradient` at zero, so the
        // first transfer has to rebuild the rotation from the grid and the
        // body's energy jumps 20 to 70 % before any CFL question is even
        // asked. For `v = omega x r` the affine field is the antisymmetric
        // `[[0, -omega], [omega, 0]]`, which APIC carries exactly.
        {
            let centre = Vec2::splat(GRID as f32 * 0.5);
            let spin = glam::Mat2::from_cols(
                Vec2::new(0.0, SPIN_PER_SECOND),
                Vec2::new(-SPIN_PER_SECOND, 0.0),
            );
            let particles = sim.particles_mut();
            for i in 0..particles.len() {
                let r = particles.x[i] - centre;
                particles.v[i] = Vec2::new(-r.y, r.x) * SPIN_PER_SECOND;
                particles.velocity_gradient[i] = spin;
            }
        }
        let (lambda, mu) = (material.lambda, material.mu);
        // Baseline after one step, not before it: the very first transfer
        // is a spawn transient common to every factor, and measuring
        // across it would put the same constant in every row.
        sim.step();

        // Both ends are averaged over a window, not sampled at an
        // instant: a spinning elastic block trades energy between its
        // kinetic and stored forms continuously, and one reading lands
        // anywhere in that cycle. Sampling instead of averaging made this
        // column non-monotone (69 % at one factor, 89 % at the next).
        let frames = (seconds / config.dt).round() as usize;
        let window = (frames / 5).max(1);
        let mut substeps = 0usize;
        let (mut start, mut end) = (0.0f64, 0.0f64);
        let wall = std::time::Instant::now();
        for frame in 0..frames {
            sim.step();
            substeps += sim.diagnostics_snapshot().substeps_last_step;
            if frame < window {
                start += total_energy(&sim, lambda, mu);
            } else if frame >= frames - window {
                end += total_energy(&sim, lambda, mu);
            }
        }
        let ms = wall.elapsed().as_secs_f64() * 1000.0 / frames as f64;
        let start = start / f64::from(window as u32);
        let end = end / f64::from(window as u32);
        let worst_j = sim
            .particles()
            .iter()
            .map(|p| (p.deformation_gradient.determinant() - 1.0).abs())
            .fold(0.0f32, f32::max);
        let diverged = !end.is_finite() || worst_j > 1.0;
        println!(
            "  {factor:>5.2}    {:>6.2}      {:>8.1}       {ms:>7.2}   {:>6.0}    {:>9}    {:>9.4}{}",
            factor / limit,
            substeps as f64 / frames as f64,
            1000.0 / ms,
            if end.is_finite() {
                format!("{:.1} %", 100.0 * end / start)
            } else {
                "n/a".to_string()
            },
            worst_j,
            if diverged { "   DIVERGED" } else { "" }
        );
    }
}

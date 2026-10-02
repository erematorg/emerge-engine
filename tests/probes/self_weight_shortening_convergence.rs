//! Does a column standing under its own weight shorten by the amount
//! elasticity says, once the cells are small enough?
//!
//! The scene of `self_weight_strain_is_spacing_independent`: a NeoHookean
//! column 0.1 m tall and 0.06 m wide (E 1e5 Pa, nu 0.2, 1000 kg/m^3) on a
//! slip floor under earth gravity. For a laterally free column on a
//! frictionless floor the stress field is exact and simple,
//! `sigma_yy = -rho g (L - y)`, every other component zero; a 2D body
//! built from 3D Lame constants is in plane strain, so
//! `eps_yy = -(1 - nu^2) rho g (L - y) / E`, and the span between the
//! particle rows half a spacing `s` inside each end shortens by
//! `(1 - nu^2) rho g L (L - s) / (2 E)`.
//!
//! The same physical column at four cell sizes, each settled to rest with a
//! little Kelvin-Voigt viscosity (`NeoHookeanMaterial::viscosity`, `SW_DAMP`
//! seconds times the shear modulus): viscous stress vanishes at rest, so
//! it changes how fast the column settles, not where (0.8320 against
//! 0.8308 at 10 cells for damping times 0.001 and 0.01 s). Mean span
//! shortening over the last second of three, against the exact value:
//!
//! ```text
//!   cells over the height   10      20      40      80
//!   shortening / exact      0.831   0.901   0.933   0.955
//!   gap                     16.9 %  9.9 %   6.7 %   4.5 %
//! ```
//!
//! The gap shrinks with the cells, by about 1.5 times per halving (order
//! about 0.6); a Richardson extrapolation of the last three puts the limit
//! at the exact law (-0.3 %). Discretisation, converging slowly.
//!
//! Without damping the answer depends on the path. At 10 cells the
//! oscillating column's running mean reads 0.839 of exact at 2 s and 0.917
//! by 50 s, as its oscillation dies: the motion leaves a permanent extra
//! shortening that the deformation gradient does not record. Not explained
//! yet.
//!
//!   cargo test --test probes self_weight_shortening_convergence:: -- --ignored --nocapture
//!
//! Knobs: SW_CELLS (10), SW_DAMP (0.001), SW_SECONDS (3).

use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn self_weight_shortening_against_plane_strain_elasticity() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let cells = env("SW_CELLS", 10.0) as usize;
    let damp = env("SW_DAMP", 0.001);
    let seconds = env("SW_SECONDS", 3.0).round() as usize;
    const E_PA: f32 = 1.0e5;
    const NU: f32 = 0.2;
    const RHO: f32 = 1000.0;
    const HEIGHT_M: f32 = 0.1;
    const SPACING: f32 = 0.5;
    let dx = HEIGHT_M / cells as f32;
    let grid = (cells + 3 + 16).next_power_of_two().max(64);
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 200_000,
        ..SimConfig::earth(grid, dx, 0.005)
    };
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new((cells as f32 * 0.6).round() as i32, cells as i32),
        box_center: Vec2::new(grid as f32 * 0.5, 3.0 + cells as f32 * 0.5),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let (lambda, mu) = config.lame_from_si(E_PA, NU, RHO);
    let mut material = NeoHookeanMaterial::new(lambda, mu);
    material.viscosity = damp * mu;
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let span = |s: &Simulation| {
        let p = s.particles();
        p.x.iter().map(|x| x.y).fold(f32::MIN, f32::max)
            - p.x.iter().map(|x| x.y).fold(f32::MAX, f32::min)
    };
    let span0 = span(&sim);
    let g = config.gravity.length() * dx;
    let exact_m = (1.0 - NU * NU) * RHO * g * HEIGHT_M * (HEIGHT_M - SPACING * dx) / (2.0 * E_PA);
    println!(
        "{cells} cells, dx {dx} m, {} particles, damping {damp} s x mu, exact span shortening {exact_m:.4e} m",
        sim.particles().len()
    );
    let frames_per_second = (1.0 / config.dt).round() as usize;
    for second in 1..=seconds {
        let mut sum = 0.0f64;
        for _ in 0..frames_per_second {
            sim.step();
            sum += f64::from(span0 - span(&sim)) * f64::from(dx);
        }
        let shortening = sum / frames_per_second as f64;
        println!(
            "t {second:3} s | mean span shortening {shortening:.4e} m | against exact {:.4}",
            shortening / f64::from(exact_m)
        );
    }
}

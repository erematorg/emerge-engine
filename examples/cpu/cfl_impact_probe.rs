//! Where does the acoustic CFL factor break when the bound actually binds?
//!
//! `cfl_margin_probe` measured what the factor costs on a gentle scene and
//! says itself that it cannot settle a factor: nothing there came near
//! breaking. This is the other half. One violent scene, an elastic block
//! dropped at earth's gravity onto a floor, from several heights and at two
//! tilts so it can land on a corner, swept over the factor.
//!
//! # What counts as broken, fixed before any run
//!
//! A run breaks at the first frame where any of these holds:
//!
//! - a non-finite value on a particle or on the grid;
//! - a particle outside the grid;
//! - a volume ratio J outside [0.25, 4], a quarter or four times the
//!   volume: the impact pressure `rho v c` is about a fifth of the plane
//!   strain bulk modulus here, so an elastic response stays far inside;
//! - the total energy (kinetic, the engine's own stored strain energy and
//!   gravity's) rising more than 10 percent of the drop energy above its
//!   start: nothing in this scene can add energy;
//! - the solver projecting any particle's state back to admissible
//!   (`j_projection_count`), which its own doc calls divergence.
//!
//! # What is printed
//!
//! Per factor: substeps per frame, the cost (wall-clock time is not
//! reported; the cost probe found 45 percent noise in it at equal work),
//! which conditions broke and on what, and the fastest particle measured.
//! That speed comes from the same per-material statistics the
//! `FrameLogger` writes, one line per frame of every run with the factor
//! and condition as extra fields, to `CFL_IMPACT_LOG` (default
//! `target/cfl_impact_probe.ndjson`). No factor is recommended: this is a
//! table to choose from.
//!
//! # What the factor is
//!
//! The engine's acoustic bound (`elastic_wave_dt`) is `factor dx / c`,
//! evaluated per particle with its own gathered density, and the smallest
//! wins. Here the thinnest particle, a corner of the block, sits at 0.386
//! of the rest density, so its wave speed is about 1.6 times the bulk's
//! and the step actually taken is 0.64 times the factor, in units of the
//! bulk `dx / c`. The table therefore prints the largest substep actually
//! taken next to the factor; that, not the factor, is the number to hold
//! against a stability limit.
//!
//! # Found (E 2e5 Pa, Poisson 0.3, 8 x 8 cells, slip floor, 1 ms frames)
//!
//! ```text
//!   factor  largest substep (dx/c)  substeps/frame  broke
//!    0.30        0.19                   9           0/5
//!    0.50        0.32                   6           0/5    shipped
//!    0.70        0.45                   4           0/5
//!    0.90        0.58                   3           0/5
//!    1.10        0.71                   3           0/5
//!    1.12        0.73                   3           0/5
//!    1.14        0.74                   3           1/5    energy gain, flat 10 cm, after landing
//!    1.16        0.76                   3           2/5
//!    1.18        0.76                   3           3/5
//!    1.20        0.77                   3           5/5    three of five before they reach the floor
//!    1.40        0.91                   2           5/5
//! ```
//!
//! Every condition agrees on where it breaks, within 1.14 to 1.20, flat or
//! tilted, 10 to 30 cm: the impact does not lower the threshold, and at
//! 1.20 three runs diverge in free fall, before any contact. The block
//! breaks below the analytic single-particle limit, 0.84. The fastest
//! particle in every stable run is 4.7 m/s, Mach 0.29 of this material.
//! In free fall the measured energy rises by at most 1 percent of the
//! drop energy in stable runs, which checks that its three parts share
//! units.
//!
//! Over Poisson's ratio (`CFL_IMPACT_NU`) and with the lattice's exact
//! initial volume (`CFL_IMPACT_LATTICE_V0=1`), the last factor where all
//! five conditions hold and the first where one breaks, each with the
//! largest substep actually taken:
//!
//! ```text
//!   case                        holds to        first break     all break      analytic limit
//!   Poisson 0.3,  estimated V0  1.12 (0.73)     1.14 (0.74)     1.20 (0.77)    0.84
//!   Poisson 0.45, estimated V0  1.05 (0.67)     1.10 (0.70)     1.10 (0.70)    0.74
//!   Poisson 0.49, estimated V0  1.00 (0.63)     1.05 (0.66)     1.10 (0.69)    0.71
//!   Poisson 0.3,  lattice V0    0.90 (0.905)    1.00 (1.00)     1.00 (1.00)    0.84
//! ```
//!
//! With the estimated V0 the break stays at 0.89 to 0.94 of the analytic
//! limit at every Poisson ratio tried. With the lattice's V0 no particle is
//! thin (lowest density 1.000 of rest), the step taken equals the factor,
//! and the block holds a step of 0.905 dx/c, past the analytic limit: the
//! break moves down in nominal factor (1.14 to 1.00) and up in the step
//! actually taken (0.74 to 1.00). At the shipped 0.5 the lattice V0 takes 4
//! substeps a frame instead of 6.
//!
//! What this does NOT settle: one material, one resolution, a slip floor;
//! fluids, granular and stiffer solids are not covered, and the largest
//! substep is an estimate where the bound moves within a frame.
//!
//!   cargo run --release --example cfl_impact_probe
//!   CFL_IMPACT_FACTORS=1.12,1.14,1.16,1.18 cargo run --release --example cfl_impact_probe
//!   CFL_IMPACT_NU=0.49 cargo run --release --example cfl_impact_probe
//!   CFL_IMPACT_LATTICE_V0=1 cargo run --release --example cfl_impact_probe
//!
//! `CFL_IMPACT_NU` sets Poisson's ratio (0.3 by default). The analytic
//! single-particle limit moves with it, so the break should too.
//! `CFL_IMPACT_LATTICE_V0=1` replaces, after the spawn and in this probe
//! only, each particle's estimated initial volume by the lattice's exact
//! one, spacing squared, with volume and density to match. A solid's
//! density is `mass / (V0 J)` (`elastic.rs`), so the corner particles'
//! overestimated V0 (issue #41) is what makes them thin; this shows how far
//! the break moves in nominal factor once that is gone. The engine is not
//! changed.
extern crate emerge_engine as emerge;

use emerge::materials::Elastic;
use emerge::{
    FrameLogger, FromSI, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    per_material_stats,
};
use glam::{IVec2, Mat2, Vec2};

const GRID: usize = 64;
const DX_M: f32 = 0.01;
const FRAME_S: f32 = 0.001;
const SECONDS: f32 = 0.6;
const BLOCK: IVec2 = IVec2::new(8, 8);
const SPACING: f32 = 0.5;

/// The cost probe's material, through the SI route so speeds and the
/// wave speed are in metres per second; Poisson's ratio from
/// `CFL_IMPACT_NU`.
fn material_props() -> Elastic {
    let nu = std::env::var("CFL_IMPACT_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.3);
    Elastic {
        e_pa: 2.0e5,
        nu,
        rho_kg_m3: 1000.0,
    }
}

const J_MIN: f32 = 0.25;
const J_MAX: f32 = 4.0;
const ENERGY_GAIN_OF_DROP: f64 = 0.10;

/// Drop height of the lowest point above the floor, metres, and tilt,
/// degrees.
const CONDITIONS: [(&str, f32, f32); 5] = [
    ("flat 10 cm", 0.10, 0.0),
    ("flat 20 cm", 0.20, 0.0),
    ("flat 30 cm", 0.30, 0.0),
    ("tilt 10 30 cm", 0.30, 10.0),
    ("tilt 30 30 cm", 0.30, 30.0),
];
const FACTORS: [f32; 11] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4];

/// Analytic single-particle stability limit for an explicit MLS-MPM step,
/// as a fraction of `dx / c_p`, the same expression `cfl_margin_probe` uses.
fn single_particle_limit(nu: f32) -> f32 {
    let ratio = 2.0 * nu / (1.0 - 2.0 * nu);
    ((ratio + 2.0) / (2.0 * (ratio + 1.0))).sqrt()
}

/// Kinetic, stored and gravitational energy, all in the engine's grid
/// units. The stored part is the potential THIS engine's Neo-Hookean
/// stress derives from (`elastic.rs`, Simo-Pister split in plane strain):
/// `W = mu/2 (tr(B)/J - 2) + k/2 (ln J)^2`, `k = lambda + mu`, per unit of
/// reference volume.
fn total_energy(sim: &Simulation, lambda: f32, mu: f32, floor: f32) -> f64 {
    let p = sim.particles();
    let g = f64::from(sim.config().gravity.length());
    let mut energy = 0.0f64;
    for i in 0..p.len() {
        let m = f64::from(p.mass[i]);
        energy += 0.5 * m * f64::from(p.v[i].length_squared());
        energy += m * g * f64::from(p.x[i].y - floor);
        let f = p.deformation_gradient[i];
        let j = f.determinant();
        if j > 0.0 {
            let tr_b = f.x_axis.length_squared() + f.y_axis.length_squared();
            let ln_j = j.ln();
            let w = 0.5 * mu * (tr_b / j - 2.0) + 0.5 * (lambda + mu) * ln_j * ln_j;
            energy += f64::from(w) * f64::from(p.initial_volume[i]);
        }
    }
    energy
}

struct Outcome {
    broke: Option<(usize, &'static str)>,
    substeps_per_frame: f64,
    /// Largest substep taken, seconds. The solver takes substeps at its
    /// bound and then one for what remains of the frame, and the snapshot
    /// reports only that last one, so a full substep is read as
    /// `(frame - last) / (n - 1)`: exact when the bound holds still across
    /// the frame, an estimate when it moves.
    largest_substep_s: f32,
    fastest_m_s: f32,
    free_fall_drift: f64,
    /// Lowest particle density over the rest density at the first frame:
    /// the acoustic bound is evaluated per particle with its own gathered
    /// density, so the thinnest particle sets the step.
    lowest_density: f32,
}

fn run(
    props: &Elastic,
    lattice_v0: bool,
    factor: f32,
    condition: usize,
    log: &mut FrameLogger,
) -> Outcome {
    let (_, drop_m, tilt_deg) = CONDITIONS[condition];
    let config = SimConfig {
        min_dt: 1.0e-8,
        max_substeps_per_step: 4096,
        material_cfl_coefficient: factor,
        ..SimConfig::earth(GRID, DX_M, FRAME_S)
    };
    let material = NeoHookeanMaterial::from_physical(props, &config);
    let floor = config.boundary_thickness as f32;
    // Place the block so its lowest point, after the tilt, sits `drop_m`
    // above the floor.
    let half = BLOCK.as_vec2() * 0.5;
    let tilt = tilt_deg.to_radians();
    let lowest_below_centre = half.x * tilt.sin() + half.y * tilt.cos();
    let centre = Vec2::new(
        GRID as f32 * 0.5,
        floor + drop_m / DX_M + lowest_below_centre,
    );
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: BLOCK,
        box_center: centre,
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    if lattice_v0 {
        let particles = sim.particles_mut();
        let v0 = SPACING * SPACING;
        for i in 0..particles.len() {
            particles.initial_volume[i] = v0;
            particles.volume[i] = v0;
            particles.density[i] = particles.mass[i] / v0;
        }
    }
    if tilt != 0.0 {
        let rotation = Mat2::from_angle(tilt);
        let particles = sim.particles_mut();
        for i in 0..particles.len() {
            particles.x[i] = centre + rotation * (particles.x[i] - centre);
        }
    }

    let (lambda, mu) = (material.lambda, material.mu);
    let total_mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    let g = f64::from(sim.config().gravity.length());
    let drop_energy = total_mass * g * f64::from(drop_m / DX_M);
    let start = total_energy(&sim, lambda, mu, floor);
    // Free fall lasts sqrt(2 h / g); energy measured just before contact
    // checks the energy's own units and the integrator's drift, apart from
    // any impact.
    let contact_frame = ((2.0 * drop_m / 9.81).sqrt() / FRAME_S) as usize;
    let mut free_fall_drift = 0.0f64;

    let frames = (SECONDS / FRAME_S).round() as usize;
    let (mut substeps, mut fastest, mut broke) = (0usize, 0.0f32, None);
    let (mut largest_substep_s, mut lowest_density) = (0.0f32, f32::MAX);
    for frame in 1..=frames {
        sim.step();
        let snap = sim.diagnostics_snapshot();
        substeps += snap.substeps_last_step;
        let n = snap.substeps_last_step.max(1);
        let full = if n > 1 {
            (snap.configured_dt - snap.effective_dt) / (n - 1) as f32
        } else {
            snap.configured_dt
        };
        largest_substep_s = largest_substep_s.max(full);
        if frame == 1 {
            let rest = props.rho_kg_m3 / config.reference_density_kg_m3;
            lowest_density = sim
                .particles()
                .density
                .iter()
                .fold(f32::MAX, |a, &d| a.min(d))
                / rest;
        }
        let stats = per_material_stats(sim.particles());
        log.log(
            frame as u64,
            FRAME_S,
            &stats,
            &snap,
            &[(0, "block")],
            &[("factor", factor), ("condition", condition as f32)],
        );
        if let Some(s) = stats.first() {
            fastest = fastest.max(s.max_speed * DX_M);
        }
        let energy = total_energy(&sim, lambda, mu, floor);
        if frame + 5 == contact_frame {
            free_fall_drift = (energy - start) / drop_energy;
        }
        let outside = sim
            .particles()
            .x
            .iter()
            .any(|x| !(0.0..GRID as f32).contains(&x.x) || !(0.0..GRID as f32).contains(&x.y));
        let reason = if snap.non_finite_particle_values > 0 || snap.non_finite_grid_values > 0 {
            Some("non-finite")
        } else if outside || snap.out_of_bounds_particles > 0 {
            Some("left grid")
        } else if snap.min_deformation_j < J_MIN || snap.max_deformation_j > J_MAX {
            Some("J bounds")
        } else if !energy.is_finite() || energy - start > ENERGY_GAIN_OF_DROP * drop_energy {
            Some("energy gain")
        } else if snap.j_projection_count > 0 {
            Some("projection")
        } else {
            None
        };
        if let Some(reason) = reason {
            broke = Some((frame, reason));
            break;
        }
    }
    let ran = broke.map_or(frames, |(f, _)| f);
    Outcome {
        broke,
        substeps_per_frame: substeps as f64 / ran as f64,
        largest_substep_s,
        fastest_m_s: fastest,
        free_fall_drift,
        lowest_density,
    }
}

fn main() {
    let path = std::env::var("CFL_IMPACT_LOG")
        .unwrap_or_else(|_| "target/cfl_impact_probe.ndjson".to_string());
    let mut log = FrameLogger::open(&path).expect("failed to open the CFL impact log");
    let props = material_props();
    let lattice_v0 = std::env::var("CFL_IMPACT_LATTICE_V0").is_ok_and(|v| v == "1");
    let limit = single_particle_limit(props.nu);
    let (e, nu, rho) = (props.e_pa, props.nu, props.rho_kg_m3);
    let c = ((e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))) / rho).sqrt();
    println!(
        "Elastic block {}x{} cells of {} cm, E {e} Pa, Poisson {nu}, rho {rho}: wave speed {c:.1} m/s, single-particle limit {limit:.3} dx/c, shipped factor {:.2}",
        BLOCK.x,
        BLOCK.y,
        DX_M * 100.0,
        SimConfig::earth(GRID, DX_M, FRAME_S).material_cfl_coefficient
    );
    println!(
        "conditions: {}; initial volume: {}",
        CONDITIONS.map(|c| c.0).join(", "),
        if lattice_v0 {
            "lattice, spacing squared"
        } else {
            "estimated at spawn"
        }
    );
    let dx_over_c = DX_M / c;
    println!(
        "  factor  largest substep (dx/c)  substeps/frame (min-max)  broke   first failure per condition               fastest m/s (Mach)  free-fall drift  lowest density"
    );
    let factors: Vec<f32> = std::env::var("CFL_IMPACT_FACTORS")
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|f| f.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| FACTORS.to_vec());
    for factor in factors {
        let outcomes: Vec<Outcome> = (0..CONDITIONS.len())
            .map(|k| run(&props, lattice_v0, factor, k, &mut log))
            .collect();
        let costs: Vec<f64> = outcomes.iter().map(|o| o.substeps_per_frame).collect();
        let mean = costs.iter().sum::<f64>() / costs.len() as f64;
        let (lo, hi) = costs
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &c| (l.min(c), h.max(c)));
        let broke = outcomes.iter().filter(|o| o.broke.is_some()).count();
        let failures: Vec<String> = outcomes
            .iter()
            .map(|o| match o.broke {
                Some((frame, reason)) => format!("{reason}@{frame}"),
                None => "ok".to_string(),
            })
            .collect();
        let fastest = outcomes
            .iter()
            .map(|o| o.fastest_m_s)
            .fold(0.0f32, f32::max);
        let drift = outcomes
            .iter()
            .map(|o| o.free_fall_drift.abs())
            .fold(0.0f64, f64::max);
        let largest = outcomes
            .iter()
            .map(|o| o.largest_substep_s)
            .fold(0.0f32, f32::max);
        let lowest = outcomes
            .iter()
            .map(|o| o.lowest_density)
            .fold(f32::MAX, f32::min);
        println!(
            "  {factor:>5.2}      {:>6.3}             {mean:>6.2} ({lo:.2}-{hi:.2})          {broke}/{}    {:<42} {fastest:>6.2} ({:.2})        {drift:.4}          {lowest:.3}",
            largest / dx_over_c,
            CONDITIONS.len(),
            failures.join(", "),
            fastest / c,
        );
    }
    println!("log: {path}");
}
